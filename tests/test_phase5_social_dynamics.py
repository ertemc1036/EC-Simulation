"""Phase 5 — Social Dynamics Engine (offline, no live LLM)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    load_settings,
)
from backend.simulation.engine import SimulationEngine
from backend.simulation.knowledge_graph import INTERACTION_RELATIONS, LocalKnowledgeGraph
from backend.simulation.oasis_reply import oasis_reply_status
from backend.simulation.prompt_debug import PROMPT_BUNDLE_ID
from backend.simulation.prompts import build_decision_prompt, format_feed_context
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import (
    SOCIAL_RELATIONS,
    PositionTracker,
    extract_local_comment_refs,
)


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
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ENTRIES", "8")
    return out


def test_limits_unchanged():
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    cfg = load_settings()
    assert cfg.max_tokens_per_run == 20000
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_cost_usd_per_run == 0.20
    assert cfg.max_parallel_llm_calls == 1
    assert cfg.status_payload()["phase"] == "5"
    assert cfg.status_payload()["llm_run_requires_approval"] is True


def test_local_comment_ref_parser():
    assert extract_local_comment_refs("siehe →Kommentar#3 bitte") == [3]
    assert extract_local_comment_refs("Kommentar#7 und comment_id=9") == [7, 9]
    assert extract_local_comment_refs("kein ref hier") == []


def test_oasis_reply_documents_phase5_relations():
    status = oasis_reply_status()
    assert status["parent_comment_id_supported"] is False
    assert SOCIAL_RELATIONS <= set(status["graph_relations_evidenced_only"])
    assert SOCIAL_RELATIONS <= INTERACTION_RELATIONS


def test_position_tracker_change_and_influence():
    tracker = PositionTracker()
    agents = [
        {"agent_id": 0, "goal": "Dafür: Autoverbot unterstützen."},
        {"agent_id": 1, "goal": "Dagegen: Autoverbot ablehnen."},
    ]
    tracker.seed_from_goals(agents, step=0)
    # Round 1 statements
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich unterstütze das Verbot klar."}
        ],
        comments=[],
        reactions=[],
        step=1,
    )
    # Round 2: agent 1 reacts then changes mind after interaction
    changes = tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich unterstütze das Verbot klar."}
        ],
        comments=[
            {
                "comment_id": 2,
                "post_id": 1,
                "user_id": 1,
                "content": (
                    "→Kommentar#2 war falsch — ich ändere meine Position und "
                    "werde überzeugt vom Verbot."
                ),
            }
        ],
        reactions=[
            {"type": "like", "user_id": 1, "post_id": 1},
        ],
        step=2,
    )
    assert any(c["agent_id"] == 1 for c in changes) or any(
        p.get("changed") for p in tracker.positions_payload()["positions"]
    )
    specs = tracker.social_edge_specs()
    relations = {s["relation"] for s in specs}
    assert "position_changed" in relations
    # influenced_by only when change + interaction with another actor
    assert any(s["relation"] == "influenced_by" for s in specs)


def test_knowledge_graph_social_edges_only_evidenced():
    kg = LocalKnowledgeGraph()
    kg.rebuild(
        scenario=BERLIN,
        agents=[
            {"agent_id": 0, "name": "A", "goal": "pro", "style": "x", "user_name": "a"},
            {"agent_id": 1, "name": "B", "goal": "contra", "style": "y", "user_name": "b"},
        ],
        posts=[{"post_id": 1, "user_id": 0, "content": "Seed dafür"}],
        comments=[
            {
                "comment_id": 5,
                "post_id": 1,
                "user_id": 1,
                "content": "→Kommentar#5 Antwort auf dich",
            },
            {
                "comment_id": 6,
                "post_id": 1,
                "user_id": 0,
                "content": "→Kommentar#5 ich reagiere lokal",
            },
        ],
        reactions=[
            {"type": "like", "user_id": 1, "comment_id": 6},
            {"type": "dislike", "user_id": 0, "comment_id": 5},
        ],
        social_edge_specs=[
            {
                "relation": "position_changed",
                "source": "actor:1",
                "target": "topic:scenario",
                "meta": {"evidence": "test"},
            },
            {
                "relation": "influenced_by",
                "source": "actor:1",
                "target": "actor:0",
                "meta": {"evidence": "test"},
            },
            # Invented relation must be ignored by SOCIAL_RELATIONS gate
            {
                "relation": "invented_edge",
                "source": "actor:0",
                "target": "actor:1",
                "meta": {},
            },
        ],
        agent_positions=[
            {
                "agent_id": 1,
                "stance": "umdenken",
                "polarity": "shifting",
                "stance_key": "shifting:x",
                "changed": True,
            }
        ],
    )
    data = kg.to_dict()
    sr = data["social_relations"]
    assert sr["agrees_with"] >= 1
    assert sr["disagrees_with"] >= 1
    assert sr["responds_to"] >= 1
    assert sr["position_changed"] >= 1
    assert sr["influenced_by"] >= 1
    assert not any(e["relation"] == "invented_edge" for e in data["edges"])
    # Actor meta carries position
    actor1 = next(n for n in data["nodes"] if n["id"] == "actor:1")
    assert actor1["meta"].get("position_changed") is True


def test_decision_prompt_round2_comment_reactions():
    feed = format_feed_context(
        [{"post_id": 1, "user_id": 0, "content": "Seed"}],
        [{"comment_id": 3, "post_id": 1, "user_id": 1, "content": "Meinung"}],
    )
    assert "→Kommentar" in feed or "like_comment" in feed
    r1 = build_decision_prompt(
        scenario=BERLIN,
        goal="G",
        language="de",
        env_prompt="",
        agent_name="A",
        feed_context=feed,
        memory_context="(Kein Memory.)",
        round_number=1,
        position_context="pro: unterstütze Verbot",
    )
    assert "Runde: 1" in r1
    assert "POSITION:" in r1
    r2 = build_decision_prompt(
        scenario=BERLIN,
        goal="G",
        language="de",
        env_prompt="",
        agent_name="A",
        feed_context=feed,
        memory_context="(Kein Memory.)",
        round_number=2,
        position_context="contra: dagegen",
    )
    assert "Runde: 2" in r2
    assert "like_comment" in r2
    assert "→Kommentar" in r2


def test_scripted_phase5_run_positions_and_relations(results_dir: Path):
    """3 agents × 2 rounds — positions persist; R2 yields agrees/disagrees/responds."""
    engine = SimulationEngine(
        load_settings(),
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid, label="p5"),
    )
    started = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        agent_count=3,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    assert started["ok"] is True
    deadline = time.time() + 90
    while time.time() < deadline:
        state = engine.get_state()
        if state["status"] in {"completed", "limit_reached", "failed"}:
            break
        time.sleep(0.2)
    state = engine.get_state()
    assert state["status"] in {"completed", "limit_reached"}
    assert state["status"] != "failed"

    positions = state.get("agent_positions") or {}
    assert positions.get("agent_count", 0) >= 3
    assert len(positions.get("positions") or []) >= 3

    summary = state["interaction_summary"]
    # Live run 0cfc5d1f had agrees/disagrees=0; Phase-5 scripted R2 must populate.
    social_hits = (
        int(summary.get("agrees_with") or 0)
        + int(summary.get("disagrees_with") or 0)
        + int(summary.get("responds_to") or 0)
    )
    assert social_hits >= 1
    assert int(summary.get("responds_to") or 0) >= 1

    kg = state["knowledge_graph"]
    sr = kg.get("social_relations") or {}
    assert set(SOCIAL_RELATIONS) <= set(sr.keys())

    # Meinungsänderung-Nachweis: at least one position_changed after R1→R2 content.
    change_count = int(positions.get("change_count") or 0)
    pc = int(sr.get("position_changed") or 0)
    assert change_count >= 1 or pc >= 1

    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["phase"] == 5
    assert data["meta"]["prompt_bundle_id"] == PROMPT_BUNDLE_ID
    assert data["meta"]["token_optimization"]["approval_gate"] is True
    assert data["meta"]["token_optimization"]["atomic_reserve"] is True
    assert data["meta"]["token_optimization"]["chat_history_reset_per_decision"] is True
    assert data["meta"]["social_dynamics"]["persistent_positions"] is True
    assert "agent_positions" in data
    assert data["oasis_reply"]["parent_comment_id_supported"] is False


def test_approval_and_atomic_reserve_kept(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    blocked = engine.start(BERLIN, mode="llm", max_steps=2, llm_run_approved=False)
    assert blocked["ok"] is False
    assert blocked["state"]["status"] == "blocked"
