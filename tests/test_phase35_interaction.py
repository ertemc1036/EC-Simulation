"""Phase 3.5 — context-aware interaction, conflicts, memory, offline ScriptedModel."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import load_settings
from backend.simulation.agent_generator import (
    extract_content_keywords,
    extract_scenario_keywords,
    generate_agents,
)
from backend.simulation.engine import LLM_INTERACTION_ACTIONS, SimulationEngine
from backend.simulation.local_memory import LocalAgentMemory
from backend.simulation.prompt_debug import PROMPT_BUNDLE_ID, analyze_messages
from backend.simulation.prompts import (
    build_decision_prompt,
    format_feed_context,
    format_memory_context,
    sanitize_oasis_env_prompt,
)
from backend.simulation.scripted_model import ScriptedReactiveModel, _extract_post_ids
from oasis import ActionType


BERLIN = (
    "Was passiert, wenn Berlin ab 2030 private Autos innerhalb des S-Bahn-Rings "
    "verbietet? Wie reagieren Bürger, Unternehmen, Politiker und Verkehrsbetriebe?"
)


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "5000")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ENTRIES", "8")
    return out


def test_llm_actions_exclude_search_and_refresh_no_quotas_on_reactions():
    names = {a.value if hasattr(a, "value") else str(a) for a in LLM_INTERACTION_ACTIONS}
    # Prefer ActionType enum membership for clarity.
    assert ActionType.CREATE_COMMENT in LLM_INTERACTION_ACTIONS
    assert ActionType.LIKE_POST in LLM_INTERACTION_ACTIONS
    assert ActionType.DISLIKE_POST in LLM_INTERACTION_ACTIONS
    assert ActionType.SEARCH_POSTS not in LLM_INTERACTION_ACTIONS
    assert ActionType.REFRESH not in LLM_INTERACTION_ACTIONS
    # No per-action quotas encoded in the allow-list (set membership only).
    assert ActionType.CREATE_POST in LLM_INTERACTION_ACTIONS
    assert "search_posts" not in names or ActionType.SEARCH_POSTS not in LLM_INTERACTION_ACTIONS


def test_stakeholder_goals_conflict_not_shared_focus():
    agents = generate_agents(BERLIN, 5, mode="dynamic")
    goals = [a.goal for a in agents]
    assert len(set(goals)) == len(goals)
    joined = " ".join(goals).lower()
    assert "zielkonflikt" in joined
    # Must not paste the whole stakeholder list as shared Fokus.
    assert not all("unternehmen" in g.lower() and "politiker" in g.lower() for g in goals[:4])
    # Role stances should diverge.
    assert any("widersprich" in g.lower() or "fordere" in g.lower() for g in goals)
    content_kws = extract_content_keywords(BERLIN)
    assert "Berlin" in content_kws or any("bahn" in k.lower() or "auto" in k.lower() for k in content_kws)
    # Content keywords preferred over role-name dump.
    scen_kws = extract_scenario_keywords(BERLIN)
    assert scen_kws[0].lower() not in {"bürger", "buerger", "unternehmen"}


def test_feed_and_memory_formatters():
    feed = format_feed_context(
        [{"post_id": 3, "user_id": 1, "content": "Hallo Feed"}],
        [{"comment_id": 1, "post_id": 3, "user_id": 2, "content": "Antwort"}],
    )
    assert "Post#3" in feed
    assert "Kommentar#1→Post#3" in feed
    mem = LocalAgentMemory(max_entries_per_agent=4)
    mem.remember(0, kind="action", text="Aktion create_post", step=0)
    mem.remember(0, kind="observation", text="Feed-Snapshot: Post#3", step=1)
    recent = mem.recent_for_prompt(0, limit=2)
    block = format_memory_context(recent)
    assert "create_post" in block
    assert "Feed-Snapshot" in block


def test_sanitize_oasis_env_strips_group_noise_and_like_bait():
    raw = (
        "And there are many group chat channels {}\n"
        "And You are already in some groups []\n"
        "You receive some messages from them {}\n"
        "You can join the groups you are interested, leave the groups you already in, "
        "send messages to the group you already in.\n"
        "You must make sure you can only send messages to the group you are already in\n"
        "After refreshing, you see some posts [{\"post_id\": 1}]\n"
        "pick one you want to perform action that best reflects your current inclination "
        "based on your profile and posts content. Do not limit your action in just "
        "`like` to like posts"
    )
    clean = sanitize_oasis_env_prompt(raw)
    assert "group chat channels" not in clean
    assert "Do not limit your action" not in clean
    assert "post_id" in clean


def test_scripted_model_reads_feed_hash_ids():
    ids = _extract_post_ids('FEED-KONTEXT\n- Post#7 von Nutzer 0: x\n{"post_id": 2}')
    assert ids == [2, 7] or ids == [7, 2] or set(ids) == {2, 7}
    model = ScriptedReactiveModel(agent_id=1)
    resp = model._run(
        [{"role": "user", "content": "FEED-KONTEXT\n- Post#4 von Nutzer 0: Seed"}]
    )
    tool = resp.choices[0].message.tool_calls[0]
    assert tool.function.name in {
        "create_comment",
        "like_post",
        "dislike_post",
        "create_post",
    }
    args = json.loads(tool.function.arguments)
    if tool.function.name != "create_post":
        assert args.get("post_id") == 4


def test_decision_prompt_markers_in_prompt_debug():
    text = build_decision_prompt(
        scenario="S",
        goal="G",
        language="de",
        env_prompt="After refreshing, you see some posts []",
        agent_name="A",
        feed_context="- Post#1 von Nutzer 0: x",
        memory_context="- [s0|action] y",
    )
    evidence = analyze_messages(
        [
            {"role": "system", "content": "# ZIEL / OBJECTIVE\nLike-Ketten"},
            {"role": "user", "content": text},
        ]
    )
    assert evidence.markers["custom_decision"] is True
    assert evidence.markers["custom_feed_context"] is True
    assert evidence.markers["custom_memory_context"] is True
    assert evidence.prompt_bundle_id == PROMPT_BUNDLE_ID


def test_scripted_llm_run_produces_comments_or_reactions(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        llm_run_approved=True,
        agent_count=5,
        agent_generation="dynamic",
    )
    assert result["ok"] is True
    engine._thread.join(timeout=180)
    state = engine.get_state()
    assert state["status"] == "completed", state
    action_types = {e.get("action_type") for e in state["action_log"]}
    assert "search_posts" not in action_types
    assert "refresh" not in action_types or all(
        e.get("action_type") != "refresh" or e.get("source") != "LLMAction"
        for e in state["action_log"]
    )
    # Real platform interactions beyond seed create_post
    interactive = {"create_comment", "like_post", "dislike_post"}
    assert action_types & interactive, action_types
    assert state["comments"] or state["reactions"], state
    # Decision log carries feed + memory previews
    decisions = state["decision_log"]
    assert decisions
    assert any(d.get("feed_context_preview") for d in decisions)
    assert any(d.get("memory_context_preview") for d in decisions)
    assert any("feed+memory" in (d.get("note") or "") for d in decisions)
    # Memory populated and prompt bundle v2
    assert state["agent_memory"]["agent_count_with_memory"] >= 1
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["prompt_bundle_id"] == PROMPT_BUNDLE_ID
    # Goals conflict across Berlin roles
    goals = [a["goal"] for a in data["agents"]]
    assert len(set(goals)) == len(goals)
    assert any("Zielkonflikt" in g for g in goals)


def test_limit_call_regression_unchanged(results_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Phase 3.5 must not raise call/token ceilings — limit stop still works."""
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "3")
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "5")
    cfg = load_settings()
    assert cfg.max_llm_calls_per_run == 3
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Limit-Regression Phase 3.5",
        mode="llm",
        max_steps=5,
        llm_run_approved=True,
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "limit_reached", state
    assert state["cost_summary"]["api_calls"] <= 3
    assert "Call-Limit" in (state["cost_summary"]["limit_hit"] or "")


def test_budget_math_regression_five_by_two():
    """Live-test budget unchanged: 5 agents × 2 rounds ≤ 10 calls."""
    assert 5 * 2 == 10
    from backend.config import LIVE_TEST_MAX_AGENTS, LIVE_TEST_MAX_LLM_CALLS, LIVE_TEST_MAX_ROUNDS

    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_AGENTS * LIVE_TEST_MAX_ROUNDS <= LIVE_TEST_MAX_LLM_CALLS
