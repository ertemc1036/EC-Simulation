"""Phase 4 — Multi-Agent Interaction Engine (offline, no live LLM)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    load_settings,
)
from backend.simulation.agent_generator import generate_agents
from backend.simulation.engine import LLM_INTERACTION_ACTIONS, SimulationEngine
from backend.simulation.knowledge_graph import INTERACTION_RELATIONS, LocalKnowledgeGraph
from backend.simulation.oasis_reply import OASIS_REPLY_API, oasis_reply_status
from backend.simulation.prompt_debug import PROMPT_BUNDLE_ID
from backend.simulation.prompts import build_decision_prompt, format_feed_context
from backend.simulation.scripted_model import (
    ScriptedReactiveModel,
    _extract_comment_ids,
    _extract_post_ids,
)
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
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "20000")
    monkeypatch.setenv("MAX_TOKENS_PER_CALL", "500")
    monkeypatch.setenv("MAX_COST_USD_PER_RUN", "0.20")
    monkeypatch.setenv("MAX_PARALLEL_LLM_CALLS", "1")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ENTRIES", "8")
    return out


def test_oasis_reply_api_no_parent_comment_id():
    status = oasis_reply_status()
    assert status["parent_comment_id_supported"] is False
    assert status["separate_reply_action_type"] is False
    assert "create_comment" in status["reply_via"]
    assert "parent_comment_id" in status["documented_alternative"]
    assert OASIS_REPLY_API["parent_comment_id_supported"] is False
    assert set(status["graph_relations_evidenced_only"]) == INTERACTION_RELATIONS


def test_llm_actions_include_comment_reactions_without_quotas():
    assert ActionType.CREATE_COMMENT in LLM_INTERACTION_ACTIONS
    assert ActionType.LIKE_COMMENT in LLM_INTERACTION_ACTIONS
    assert ActionType.DISLIKE_COMMENT in LLM_INTERACTION_ACTIONS
    assert ActionType.LIKE_POST in LLM_INTERACTION_ACTIONS
    assert ActionType.DISLIKE_POST in LLM_INTERACTION_ACTIONS
    assert ActionType.SEARCH_POSTS not in LLM_INTERACTION_ACTIONS
    assert ActionType.REFRESH not in LLM_INTERACTION_ACTIONS
    assert ActionType.FOLLOW not in LLM_INTERACTION_ACTIONS
    # No per-action quotas — allow-list membership only.
    assert len(LLM_INTERACTION_ACTIONS) == len(set(LLM_INTERACTION_ACTIONS))


def test_conflicts_without_quotas_berlin():
    agents = generate_agents(BERLIN, 5, mode="dynamic")
    goals = [a.goal for a in agents]
    assert len(set(goals)) == 5
    assert any("Zielkonflikt" in g for g in goals)
    joined = " ".join(goals).lower()
    assert "widersprich" in joined or "fordere" in joined


def test_feed_context_documents_oasis_reply_alternative():
    feed = format_feed_context(
        [{"post_id": 2, "user_id": 0, "content": "Seed"}],
        [{"comment_id": 5, "post_id": 2, "user_id": 1, "content": "Antwort"}],
    )
    assert "parent_comment_id" in feed
    assert "post_id=2" in feed or "Post#2" in feed
    assert "comment_id=5" in feed or "Kommentar#5" in feed
    prompt = build_decision_prompt(
        scenario="S",
        goal="G",
        language="de",
        env_prompt="After refreshing, you see some posts []",
        agent_name="A",
        feed_context=feed,
        memory_context="- [s0|action] y",
    )
    assert "INTERAKTIONSKONTEXT" in prompt
    assert "parent_comment_id" in prompt
    assert "FEED-KONTEXT" in prompt
    assert "MEMORY-KONTEXT" in prompt


def test_scripted_parses_comment_ids_and_can_like_comment():
    blob = (
        "FEED-KONTEXT\n- Post#1 von Nutzer 0: x\n"
        "- Kommentar#9→Post#1 von Nutzer 2 (like/dislike_comment comment_id=9): y"
    )
    assert 1 in _extract_post_ids(blob)
    assert 9 in _extract_comment_ids(blob)
    # Force kind that prefers comment reaction when comments visible.
    model = ScriptedReactiveModel(agent_id=0)
    # call_count starts 0; after first decide with kind=(1+0)%5=1 → like_post
    # Drive until we hit kind==3 with comment ids: need (n+aid)%5==3 → n%5==3
    seen = set()
    for _ in range(8):
        resp = model._run([{"role": "user", "content": blob}])
        name = resp.choices[0].message.tool_calls[0].function.name
        seen.add(name)
        args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        if name in {"like_comment", "dislike_comment"}:
            assert args.get("comment_id") == 9
            break
    assert seen & {
        "create_comment",
        "like_post",
        "dislike_post",
        "like_comment",
        "dislike_comment",
        "create_post",
    }


def test_graph_interaction_edges_evidenced_only():
    g = LocalKnowledgeGraph()
    g.rebuild(
        scenario=BERLIN,
        agents=generate_agents(BERLIN, 3, mode="dynamic"),
        posts=[{"post_id": 1, "user_id": 0, "content": "Seed"}],
        comments=[
            {
                "comment_id": 1,
                "post_id": 1,
                "user_id": 1,
                "content": "Reply ohne parent_comment_id",
            }
        ],
        reactions=[
            {"type": "like", "user_id": 2, "post_id": 1},
            {"type": "dislike", "user_id": 0, "comment_id": 1, "target": "comment"},
        ],
        action_log=[],
    )
    data = g.to_dict()
    ir = data["interaction_relations"]
    assert ir["replied_to"] >= 1
    assert ir["agrees_with"] >= 1
    assert ir["disagrees_with"] >= 1
    # No invented interaction edges beyond evidence.
    for e in data["edges"]:
        if e["relation"] in INTERACTION_RELATIONS:
            assert e["meta"].get("evidence") or e["relation"] == "replied_to"


def test_graph_no_interaction_without_evidence():
    g = LocalKnowledgeGraph()
    g.rebuild(
        scenario="Leer",
        agents=generate_agents("Leer", 2, mode="dynamic"),
        posts=[],
        comments=[],
        reactions=[],
        action_log=[],
    )
    ir = g.to_dict()["interaction_relations"]
    assert ir == {"agrees_with": 0, "disagrees_with": 0, "replied_to": 0}


def test_scripted_llm_run_phase4_interactions(results_dir: Path):
    cfg = load_settings()
    assert cfg.max_tokens_per_run == 20000
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_cost_usd_per_run == 0.20
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
    assert state["oasis_reply"]["parent_comment_id_supported"] is False
    interactive = {
        "create_comment",
        "like_post",
        "dislike_post",
        "like_comment",
        "dislike_comment",
    }
    action_types = {e.get("action_type") for e in state["action_log"]}
    assert action_types & interactive, action_types
    assert "search_posts" not in action_types
    assert state["comments"] or state["reactions"], state
    kg = state["knowledge_graph"]
    ir = kg.get("interaction_relations") or {}
    assert (ir.get("replied_to") or 0) + (ir.get("agrees_with") or 0) + (
        ir.get("disagrees_with") or 0
    ) >= 1
    summary = state["interaction_summary"]
    assert summary["comments"] == len(state["comments"])
    # Local memory still populated; Zep not integrated.
    assert state["agent_memory"]["zep_integrated"] is False
    assert state["agent_memory"]["agent_count_with_memory"] >= 1
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["phase"] == 4.1
    assert data["meta"]["token_optimization"]["chat_history_reset_per_decision"] is True
    assert data["meta"]["token_optimization"]["ids_retained"] is True
    assert data["meta"]["token_optimization"]["atomic_reserve"] is True
    assert data["meta"]["token_optimization"]["approval_gate"] is True
    assert data["meta"]["prompt_bundle_id"] == PROMPT_BUNDLE_ID
    assert data["meta"]["oasis_reply"]["parent_comment_id_supported"] is False
    assert data["meta"]["zep_integrated"] is False


def test_budget_and_approval_regression(results_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Limits unchanged: 5 agents, 2 rounds, 10 calls, 20k tokens, 500 out, $0.20."""
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_AGENTS * LIVE_TEST_MAX_ROUNDS <= LIVE_TEST_MAX_LLM_CALLS

    cfg = load_settings()
    assert cfg.max_agents == 5
    assert cfg.max_simulation_steps == 2
    assert cfg.max_llm_calls_per_run == 10
    assert cfg.max_tokens_per_run == 20000
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_cost_usd_per_run == 0.20
    assert cfg.max_parallel_llm_calls == 1
    assert cfg.status_payload()["phase"] == "4.1"
    assert cfg.status_payload()["llm_run_requires_approval"] is True

    # Approval gate still blocks without llm_run_approved.
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    blocked = engine.start(BERLIN, mode="llm", max_steps=2, llm_run_approved=False)
    assert blocked["ok"] is False
    assert blocked["state"]["status"] == "blocked"

    # Call-limit still stops early.
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "3")
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    cfg2 = load_settings()
    assert cfg2.max_llm_calls_per_run == 3
    engine2 = SimulationEngine(
        cfg2, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    started = engine2.start(
        "Budget-Regression Phase 4",
        mode="llm",
        max_steps=2,
        llm_run_approved=True,
        agent_count=5,
    )
    assert started["ok"] is True
    engine2._thread.join(timeout=120)
    state = engine2.get_state()
    assert state["status"] == "limit_reached", state
    assert state["cost_summary"]["api_calls"] <= 3
