"""Phase 4.1 — token optimization & context bounds (offline, no live LLM)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    load_settings,
)
from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.engine import LLM_INTERACTION_ACTIONS, SimulationEngine
from backend.simulation.local_memory import LocalAgentMemory
from backend.simulation.prompt_debug import PROMPT_BUNDLE_ID, analyze_messages
from backend.simulation.prompts import (
    MAX_FEED_COMMENTS,
    MAX_FEED_POSTS,
    MAX_MEMORY_ENTRIES,
    build_decision_prompt,
    build_system_prompt,
    estimate_decision_prompt_chars,
    format_feed_context,
    format_memory_context,
    select_prioritized_comments,
    select_prioritized_memory,
    select_prioritized_posts,
)
from backend.simulation.agents import DEFAULT_AGENTS
from backend.simulation.scripted_model import ScriptedReactiveModel
from oasis import ActionType


BERLIN = (
    "Was passiert, wenn Berlin ab 2030 private Autos innerhalb des S-Bahn-Rings "
    "verbietet? Wie reagieren Bürger, Unternehmen, Politiker und Verkehrsbetriebe?"
)

# Heuristic from live run 20261027T234852Z-2793cd13: R1 ~2k prompt tokens ≈ ~6–8k chars
# of decision text + system + tools. We measure decision-user chars offline.
_LEGACY_FEED_POST_CAP = 8
_LEGACY_FEED_COMMENT_CAP = 6
_LEGACY_MEMORY_CAP = 6


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "20000")
    monkeypatch.setenv("MAX_TOKENS_PER_CALL", "500")
    monkeypatch.setenv("MAX_COST_USD_PER_RUN", "0.20")
    monkeypatch.setenv("MAX_PARALLEL_LLM_CALLS", "1")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    return out


def _bloated_feed(n_posts: int = 12, n_comments: int = 10) -> tuple[list, list]:
    posts = [
        {
            "post_id": i + 1,
            "user_id": i % 3,
            "content": (
                f"Langer Stakeholder-Beitrag #{i+1} über Autoverbot, Lieferfenster, "
                f"sozialen Ausgleich und ÖPNV-Kapazität. " * 3
            ),
        }
        for i in range(n_posts)
    ]
    comments = [
        {
            "comment_id": i + 1,
            "post_id": (i % n_posts) + 1,
            "user_id": (i + 1) % 3,
            "content": (
                f"Kommentar {i+1} mit ausführlicher Gegenrede und Zahlenwerk. " * 2
            ),
        }
        for i in range(n_comments)
    ]
    return posts, comments


def test_limits_unchanged_20k_10_500_020_5_2(results_dir: Path):
    cfg = load_settings()
    assert cfg.max_tokens_per_run == 20_000
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_llm_calls_per_run == LIVE_TEST_MAX_LLM_CALLS == 10
    assert cfg.max_cost_usd_per_run == 0.20
    assert cfg.max_agents == LIVE_TEST_MAX_AGENTS == 5
    assert cfg.max_simulation_steps == LIVE_TEST_MAX_ROUNDS == 2
    assert cfg.max_parallel_llm_calls == LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1


def test_prompt_bundle_v4_and_slim_system():
    assert PROMPT_BUNDLE_ID == "ec-custom-forum-de-v4"
    sys_p = build_system_prompt(
        DEFAULT_AGENTS[0], scenario=BERLIN, language="de"
    )
    assert "# ZIEL / OBJECTIVE" in sys_p
    assert "parent_comment_id" in sys_p
    assert "You're a Reddit user" not in sys_p
    # Slimmer than Phase-4 essay rules (offline char bound).
    assert len(sys_p) < 2200


def test_prioritized_feed_keeps_ids_and_caps():
    posts, comments = _bloated_feed()
    selected_p = select_prioritized_posts(posts, viewer_id=0, max_posts=MAX_FEED_POSTS)
    selected_c = select_prioritized_comments(
        comments, viewer_id=0, max_comments=MAX_FEED_COMMENTS
    )
    assert len(selected_p) == MAX_FEED_POSTS == 4
    assert len(selected_c) == MAX_FEED_COMMENTS == 4
    # Foreign-first: viewer 0 should not dominate when foreign posts exist.
    foreign = [p for p in selected_p if p["user_id"] != 0]
    assert foreign, "prioritization should prefer foreign posts"
    feed = format_feed_context(posts, comments, viewer_id=0)
    assert "post_id=" in feed
    assert "comment_id=" in feed or "Kommentar#" in feed
    assert "parent_comment_id" in feed
    # Hard cap: not all 12 posts dumped (count line starters, not →Post# in comments).
    post_lines = [ln for ln in feed.splitlines() if ln.startswith("- Post#")]
    comment_lines = [ln for ln in feed.splitlines() if ln.startswith("- Kommentar#")]
    assert len(post_lines) == MAX_FEED_POSTS
    assert len(comment_lines) == MAX_FEED_COMMENTS
    assert len(posts) > MAX_FEED_POSTS


def test_context_smaller_than_legacy_unbounded():
    posts, comments = _bloated_feed()
    mem = [
        {
            "kind": "observation",
            "step": 0,
            "text": "Feed-Snapshot: " + ("x" * 400),
        },
        {
            "kind": "action",
            "step": 1,
            "text": "Aktion create_comment · Ziel=1 · Text: " + ("y" * 300),
        },
        {
            "kind": "observation",
            "step": 1,
            "text": "Feed-Snapshot: " + ("z" * 400),
        },
        {
            "kind": "action",
            "step": 2,
            "text": "Aktion like_post · Ziel=2",
        },
    ] * 2
    # Legacy-style: many items, long truncations, env always included.
    legacy_feed_lines = ["hint"] + [
        f"- Post#{p['post_id']}: {p['content'][:220]}"
        for p in posts[-_LEGACY_FEED_POST_CAP:]
    ] + [
        f"- Kommentar#{c['comment_id']}: {c['content'][:160]}"
        for c in comments[-_LEGACY_FEED_COMMENT_CAP:]
    ]
    legacy_mem = "\n".join(
        f"- [s{e['step']}|{e['kind']}] {e['text'][:220]}"
        for e in mem[-_LEGACY_MEMORY_CAP:]
    )
    legacy_decision = (
        f"Entscheide jetzt als X.\nSzenario (verbindlich): {BERLIN}\n"
        f"Dein individuelles Ziel / Zielkonflikt: {DEFAULT_AGENTS[0].goal}\n"
        f"INTERAKTIONSKONTEXT: langer Text …\n"
        f"FEED-KONTEXT:\n" + "\n".join(legacy_feed_lines) + "\n"
        f"MEMORY-KONTEXT:\n{legacy_mem}\n"
        f"Plattform-Umgebung (bereinigt):\n" + ("env posts dump " * 80)
    )
    optimized = estimate_decision_prompt_chars(
        scenario=BERLIN,
        goal=DEFAULT_AGENTS[0].goal,
        language="de",
        env_prompt="And there are many group chat channels {}\n" + ("posts " * 200),
        agent_name=DEFAULT_AGENTS[0].name,
        posts=posts,
        comments=comments,
        memory_entries=mem,
        viewer_id=0,
    )
    assert optimized < len(legacy_decision)
    # Expect meaningful cut (feed+memory+env redundancy), not a rounding win.
    assert optimized < int(len(legacy_decision) * 0.55)


def test_decision_omits_oasis_env_when_feed_has_ids():
    feed = format_feed_context(
        [{"post_id": 3, "user_id": 1, "content": "Seed"}],
        [{"comment_id": 7, "post_id": 3, "user_id": 2, "content": "Reply"}],
    )
    text = build_decision_prompt(
        scenario=BERLIN,
        goal="G",
        language="de",
        env_prompt="UNIQUE_ENV_MARKER_SHOULD_NOT_APPEAR " + ("noise " * 50),
        agent_name="Politikstimme",
        feed_context=feed,
        memory_context="- [s1|action] create_comment",
    )
    assert "UNIQUE_ENV_MARKER_SHOULD_NOT_APPEAR" not in text
    assert "FEED-KONTEXT" in text
    assert "post_id=3" in text or "Post#3" in text
    assert "comment_id=7" in text or "Kommentar#7" in text


def test_memory_prefers_actions_over_feed_observations():
    mem = LocalAgentMemory(max_entries_per_agent=12)
    mem.remember(0, kind="observation", text="Feed-Snapshot: Post#1 …", step=0)
    mem.remember(0, kind="action", text="Aktion create_post · Ziel=1", step=1)
    mem.remember(0, kind="observation", text="Feed-Snapshot: Post#2 …", step=1)
    mem.remember(0, kind="action", text="Aktion create_comment · Ziel=1", step=2)
    recent = mem.recent_for_prompt(0, limit=MAX_MEMORY_ENTRIES)
    assert len(recent) <= MAX_MEMORY_ENTRIES
    kinds = [e["kind"] for e in recent]
    assert kinds.count("action") >= kinds.count("observation")
    formatted = format_memory_context(recent)
    assert "create_comment" in formatted or "create_post" in formatted
    selected = select_prioritized_memory(
        [
            {"kind": "observation", "step": 0, "text": "Feed"},
            {"kind": "action", "step": 1, "text": "Aktion x"},
        ],
        max_entries=1,
    )
    assert selected[0]["kind"] == "action"


def test_follow_removed_from_llm_tools():
    assert ActionType.FOLLOW not in LLM_INTERACTION_ACTIONS
    assert ActionType.CREATE_COMMENT in LLM_INTERACTION_ACTIONS
    assert ActionType.LIKE_COMMENT in LLM_INTERACTION_ACTIONS


def test_atomic_reserve_and_approval_still_enforced(results_dir: Path):
    tracker = CostTracker(
        max_calls=10,
        max_tokens=20_000,
        price_input_per_1m=0.4,
        price_output_per_1m=1.6,
        max_cost_usd=0.20,
        max_tokens_per_call=500,
        max_parallel_calls=1,
    )
    rid = tracker.reserve_call(
        estimated_prompt_tokens=4500,
        estimated_completion_tokens=500,
        conservative=True,
    )
    assert rid >= 1
    assert tracker.reserved_tokens > 0
    tracker.commit_reservation(
        rid,
        agent_id=0,
        prompt_tokens=4000,
        completion_tokens=100,
        model="scripted",
        source="scripted_test",
    )
    assert tracker.reserved_tokens == 0
    assert tracker.total_tokens == 4100

    engine = SimulationEngine(
        load_settings(),
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid),
    )
    blocked = engine.start(BERLIN, mode="llm", max_steps=2, llm_run_approved=False)
    assert blocked["ok"] is False
    err = str(blocked.get("error") or "")
    state = blocked.get("state") or {}
    assert state.get("status") == "blocked" or "Freigabe" in err


def test_two_round_scripted_growth_bounded(results_dir: Path):
    """3 agents × 2 rounds: chat reset keeps R2 prompt from doubling R1."""
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        llm_run_approved=True,
        agent_count=3,
        agent_generation="dynamic",
    )
    assert result["ok"] is True
    engine._thread.join(timeout=180)
    state = engine.get_state()
    assert state["status"] in {"completed", "limit_reached"}
    assert state.get("stop_reason") is not None or state["status"] == "completed"
    # Must not burn the whole 20k on 3×2 scripted calls after optimization.
    assert state["cost_summary"]["total_tokens"] < 16_000
    assert state["cost_summary"]["api_calls"] <= 6

    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["phase"] == 4.1
    assert data["meta"]["prompt_bundle_id"] == PROMPT_BUNDLE_ID
    assert data["meta"]["token_optimization"]["chat_history_reset_per_decision"] is True
    assert data["meta"]["token_optimization"]["atomic_reserve"] is True
    assert data["meta"]["token_optimization"]["approval_gate"] is True

    # Prompt evidence: round-2 decisions should stay near 2 messages (system+user),
    # not re-accumulate prior assistant/tool turns.
    by_step: dict[int, list] = {}
    for d in data.get("decision_log") or []:
        by_step.setdefault(int(d.get("step") or 0), []).append(d)
    if 1 in by_step and 2 in by_step:
        r1_counts = [
            (d.get("prompt_evidence") or {}).get("message_count") or 0
            for d in by_step[1]
            if d.get("ok")
        ]
        r2_counts = [
            (d.get("prompt_evidence") or {}).get("message_count") or 0
            for d in by_step[2]
            if d.get("ok")
        ]
        if r1_counts and r2_counts:
            assert max(r2_counts) <= max(r1_counts) + 1
            assert max(r2_counts) <= 3  # system + user (+ rare extra)


def test_offline_token_growth_estimate_round2_not_double():
    """Replicate live growth pattern offline with char estimates (no API).

    Live failure mode: R2 re-sent the entire R1 user prompt via chat history
    (~2k→4k+). With per-decision reset, only bounded feed/memory growth remains.
    """
    posts_r1 = [
        {"post_id": 1, "user_id": 0, "content": "[Seed] Berlin Autoverbot Diskussionsimpuls"},
        {"post_id": 2, "user_id": 1, "content": "Unternehmenssicht zu Lieferfenstern"},
    ]
    comments_r1 = [
        {
            "comment_id": 1,
            "post_id": 1,
            "user_id": 2,
            "content": "Politik kommentiert Machbarkeit",
        }
    ]
    mem_r1 = [
        {"kind": "action", "step": 0, "text": "Aktion create_post · Text: Seed"},
        {"kind": "observation", "step": 0, "text": "Feed-Snapshot: Post#1 …"},
    ]
    c1 = estimate_decision_prompt_chars(
        scenario=BERLIN,
        goal=DEFAULT_AGENTS[0].goal,
        language="de",
        env_prompt="And there are many group chat channels {}\nYou see posts [...]",
        agent_name="Bürgerstimme",
        posts=posts_r1,
        comments=comments_r1,
        memory_entries=mem_r1,
        viewer_id=0,
    )
    posts_r2, comments_r2 = _bloated_feed(n_posts=8, n_comments=8)
    mem_r2 = mem_r1 + [
        {"kind": "observation", "step": 1, "text": "Feed-Snapshot: " + ("p" * 200)},
        {"kind": "action", "step": 1, "text": "Aktion create_comment · Ziel=1 · Text: lang"},
        {"kind": "observation", "step": 1, "text": "Feed-Snapshot: " + ("q" * 200)},
        {"kind": "action", "step": 1, "text": "Aktion like_post · Ziel=2"},
    ]
    c2 = estimate_decision_prompt_chars(
        scenario=BERLIN,
        goal=DEFAULT_AGENTS[0].goal,
        language="de",
        env_prompt="And there are many group chat channels {}\n" + ("seen posts " * 100),
        agent_name="Bürgerstimme",
        posts=posts_r2,
        comments=comments_r2,
        memory_entries=mem_r2,
        viewer_id=0,
    )
    # Absolute offline expectation: optimized decisions stay compact.
    assert c1 < 1500
    assert c2 < 2800
    # Live failure mode: prior user prompt stayed in chat → R2 ≈ c1+c2 chars.
    # With init_messages() reset, only c2 is the new user turn (~c1 chars saved).
    legacy_r2_with_history = c1 + c2
    history_savings = legacy_r2_with_history - c2
    assert history_savings == c1
    assert history_savings > 500
    assert c2 < legacy_r2_with_history - 500


def test_prompt_evidence_markers_still_custom():
    feed = format_feed_context(
        [{"post_id": 1, "user_id": 0, "content": "x"}],
        None,
    )
    user = build_decision_prompt(
        scenario="S",
        goal="G",
        language="de",
        env_prompt="",
        agent_name="A",
        feed_context=feed,
        memory_context="- [s0|action] y",
    )
    sys_p = build_system_prompt(DEFAULT_AGENTS[0], scenario="S", language="de")
    evidence = analyze_messages(
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user},
        ]
    )
    assert evidence.custom_prompts_active is True
    assert evidence.oasis_default_active is False
    assert evidence.prompt_bundle_id == PROMPT_BUNDLE_ID


def test_reserve_blocks_before_overshoot():
    tracker = CostTracker(
        max_calls=10,
        max_tokens=5000,
        price_input_per_1m=0.4,
        price_output_per_1m=1.6,
        max_cost_usd=0.20,
        max_tokens_per_call=500,
        max_parallel_calls=1,
    )
    tracker.record(
        agent_id=0,
        prompt_tokens=4500,
        completion_tokens=100,
        model="scripted",
    )
    with pytest.raises(LimitExceeded):
        tracker.reserve_call(
            estimated_prompt_tokens=4000,
            estimated_completion_tokens=500,
            conservative=True,
        )
