"""Phase 3 — dynamic agents, local memory, knowledge graph (offline only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import load_settings
from backend.simulation.agent_generator import (
    assert_no_clones,
    extract_scenario_keywords,
    generate_agents,
)
from backend.simulation.agents import AgentProfile
from backend.simulation.engine import SimulationEngine
from backend.simulation.knowledge_graph import LocalKnowledgeGraph
from backend.simulation.local_memory import LocalAgentMemory
from backend.simulation.scripted_model import ScriptedReactiveModel


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ENTRIES", "8")
    return out


def test_keywords_and_no_clones():
    scenario = "Sollten Online-Foren KI-gestützte Moderation einsetzen?"
    kws = extract_scenario_keywords(scenario)
    assert "Online" in kws or "Foren" in kws or "Moderation" in kws
    agents = generate_agents(scenario, 5, mode="dynamic")
    assert len(agents) == 5
    assert_no_clones(agents)
    # Must not be identical copies of each other
    assert len({a.user_name for a in agents}) == 5
    assert len({a.description for a in agents}) == 5
    # Scenario binding
    joined = " ".join(a.description + a.goal for a in agents)
    assert any(k in joined for k in kws)
    # Must not silently fall back to Alice…Eve fixed roster
    assert {a.user_name for a in agents} != {
        "alice_analyst",
        "bob_builder",
        "cara_community",
        "dan_devil",
        "eve_empath",
    }


BERLIN_SCENARIO = (
    "Was passiert, wenn Berlin ab 2030 private Autos innerhalb des S-Bahn-Rings "
    "verbietet? Wie reagieren Bürger, Unternehmen, Politiker und Verkehrsbetriebe?"
)


def test_berlin_roles_not_alice_eve():
    from backend.simulation.agent_generator import extract_stakeholder_roles

    roles = extract_stakeholder_roles(BERLIN_SCENARIO)
    assert "Bürger" in roles or "Buerger" in roles
    assert "Unternehmen" in roles
    assert "Politiker" in roles
    assert "Verkehrsbetriebe" in roles
    kws = extract_scenario_keywords(BERLIN_SCENARIO)
    assert "passiert" not in {k.lower() for k in kws}
    assert "wenn" not in {k.lower() for k in kws}
    agents = generate_agents(BERLIN_SCENARIO, 5, mode="dynamic")
    names = {a.name for a in agents}
    assert "Alice Analyst" not in names
    assert "Bob Builder" not in names
    joined = " ".join(a.name + a.user_name + a.description for a in agents).lower()
    assert "bürger" in joined or "buerger" in joined
    assert "unternehmen" in joined
    assert "politiker" in joined or "politik" in joined
    assert "verkehr" in joined


def test_fixed_test_roster_is_classic_five():
    agents = generate_agents("ignored", 5, mode="fixed_test")
    assert [a.user_name for a in agents] == [
        "alice_analyst",
        "bob_builder",
        "cara_community",
        "dan_devil",
        "eve_empath",
    ]


def test_dynamic_count_three(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    result = engine.start(
        "Klimaschutz vs Wirtschaftswachstum in Städten",
        mode="manual",
        max_steps=2,
        agent_count=3,
        agent_generation="dynamic",
    )
    assert result["ok"] is True
    assert result["state"]["agent_count"] == 3
    assert result["state"]["agent_generation"] == "dynamic"
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert len(state["agents"]) == 3
    profiles = [
        AgentProfile(
            **{
                k: a[k]
                for k in (
                    "agent_id",
                    "user_name",
                    "name",
                    "description",
                    "goal",
                    "style",
                )
            }
        )
        for a in state["agents"]
    ]
    assert_no_clones(profiles)
    assert state["agent_memory"]["zep_integrated"] is False
    assert state["agent_memory"]["backend"] == "local"
    assert state["knowledge_graph"]["node_count"] >= 3
    assert state["knowledge_graph"]["edge_count"] >= 1
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["phase"] in (4, 4.1, 5, 6)
    assert data["meta"]["zep_integrated"] is False
    assert "agent_memory" in data
    assert "knowledge_graph" in data


def test_berlin_manual_memory_and_kg(results_dir: Path):
    """Offline regression: Berlin Autoverbot → dynamic roles, phase=4, Memory+KG filled."""
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    result = engine.start(
        BERLIN_SCENARIO,
        mode="manual",
        max_steps=2,
        agent_count=5,
        agent_generation="dynamic",
    )
    assert result["ok"] is True
    assert result["state"]["agent_generation"] == "dynamic"
    assert result["state"]["agent_count"] == 5
    names = {a["name"] for a in result["state"]["agents"]}
    assert "Alice Analyst" not in names
    assert "Eve Empath" not in names
    engine._thread.join(timeout=180)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["agent_generation"] == "dynamic"
    mem = state["agent_memory"]
    assert mem and mem.get("agent_count_with_memory", 0) >= 1
    assert mem.get("zep_integrated") is False
    kg = state["knowledge_graph"]
    assert kg and kg.get("node_count", 0) >= 5
    assert kg.get("edge_count", 0) >= 1
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["phase"] in (4, 4.1, 5, 6)
    assert data["meta"]["agent_generation"] == "dynamic"
    assert data["agent_memory"]["agent_count_with_memory"] >= 1
    assert data["knowledge_graph"]["node_count"] >= 5
    assert {a["user_name"] for a in data["agents"]} != {
        "alice_analyst",
        "bob_builder",
        "cara_community",
        "dan_devil",
        "eve_empath",
    }


def test_llm_action_interactions_include_reactions(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Sollten Schulen Tablets für alle verpflichtend machen?",
        mode="llm",
        max_steps=2,
        llm_run_approved=True,
        agent_count=5,
        agent_generation="dynamic",
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["agent_count"] == 5
    assert len({a["name"] for a in state["agents"]}) == 5
    action_types = {e.get("action_type") for e in state["action_log"]}
    # Real OASIS platform actions from ScriptedReactiveModel via LLMAction
    assert "LLMAction" in {e.get("source") for e in state["action_log"]}
    assert "ManualAction" in {e.get("source") for e in state["action_log"]}
    assert any(
        t in action_types
        for t in ("create_post", "create_comment", "like_post", "dislike_post")
    )
    assert state["model_calls"]
    assert all(c["source"] == "scripted_test" for c in state["model_calls"])
    # Memory + graph populated
    assert state["agent_memory"]["agent_count_with_memory"] >= 1
    kg = state["knowledge_graph"]
    assert kg["counts_by_kind"].get("actor") == 5
    assert "topic" in kg["counts_by_kind"]
    assert kg["backend"] == "local"


def test_local_memory_ring_buffer():
    mem = LocalAgentMemory(max_entries_per_agent=3)
    for i in range(5):
        mem.remember(0, kind="observation", text=f"item-{i}", step=i)
    snap = mem.snapshot()
    assert snap["zep_integrated"] is False
    entries = snap["agents"]["0"]
    assert len(entries) == 3
    assert entries[0]["text"] == "item-2"


def test_knowledge_graph_relations():
    from backend.simulation.agents import DEFAULT_AGENTS

    g = LocalKnowledgeGraph()
    g.rebuild(
        scenario="KI-Moderation in Foren",
        agents=DEFAULT_AGENTS[:3],
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Belege bitte."},
            {"post_id": 2, "user_id": 1, "content": "Lass uns pilotieren."},
        ],
        comments=[
            {
                "comment_id": 1,
                "post_id": 1,
                "user_id": 2,
                "content": "Bitte sachlich bleiben.",
            }
        ],
        reactions=[{"type": "like", "user_id": 2, "post_id": 1}],
        action_log=[],
    )
    data = g.to_dict()
    assert data["counts_by_kind"]["actor"] == 3
    assert data["counts_by_kind"]["statement"] >= 2
    rels = data["counts_by_relation"]
    assert rels.get("posted", 0) >= 2
    assert rels.get("commented", 0) >= 1
    # Phase 4: evidenced interaction edges (like → agrees_with; comment → replied_to)
    assert rels.get("agrees_with", 0) >= 1
    assert rels.get("replied_to", 0) >= 1
    assert rels.get("replies_to", 0) >= 1
    assert data["interaction_relations"]["agrees_with"] >= 1
    assert data["interaction_relations"]["replied_to"] >= 1
