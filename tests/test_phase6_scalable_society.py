"""Phase 6 — Scalable Multi-Agent Society (offline / scripted only).

No OpenAI / live LLM. Live ceilings stay 5/2/10/20k/500/$0.20.
Scale presets 10/25/50 are Manual/Scripted only.
"""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import pytest

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    OFFLINE_MAX_AGENTS,
    OFFLINE_SCALE_PRESETS,
    load_settings,
)
from backend.simulation.agent_generator import (
    LIVE_LLM_MAX_AGENTS,
    MAX_GENERATABLE_AGENTS,
    assert_no_clones,
    generate_agents,
)
from backend.simulation.engine import SimulationEngine
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import PositionTracker
from backend.simulation.social_scheduler import SocialInteractionScheduler


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


def _wait_terminal(engine: SimulationEngine, *, deadline_s: float = 240.0) -> dict:
    end = time.time() + deadline_s
    while time.time() < end:
        state = engine.get_state()
        if state["status"] in {"completed", "limit_reached", "failed"}:
            # Ensure persist finished (result_path set under same terminal flip).
            if state.get("result_path") or state["status"] == "failed":
                return state
        time.sleep(0.2)
    return engine.get_state()


def test_live_llm_ceilings_unchanged():
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    assert LIVE_TEST_MAX_AGENTS * LIVE_TEST_MAX_ROUNDS <= LIVE_TEST_MAX_LLM_CALLS
    cfg = load_settings()
    assert cfg.max_agents == 5
    assert cfg.max_tokens_per_run == 20000
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_cost_usd_per_run == 0.20
    assert cfg.max_parallel_llm_calls == 1
    payload = cfg.status_payload()
    assert payload["phase"] in {"7", "8"}
    assert payload["llm_run_requires_approval"] is True
    assert payload["live_test_limits"]["max_agents"] == 5
    assert payload["live_test_limits"]["max_model_calls"] == 10
    assert payload["live_test_limits"]["max_cost_usd_per_run"] == 0.20
    assert payload["offline_scale"]["presets"] == list(OFFLINE_SCALE_PRESETS)
    assert payload["offline_scale"]["paid_llm_unlocked"] is False
    assert payload["offline_max_agents"] == OFFLINE_MAX_AGENTS == 50


def test_offline_presets_and_generator_scale():
    assert OFFLINE_SCALE_PRESETS == (10, 25, 50)
    assert MAX_GENERATABLE_AGENTS == 50
    assert LIVE_LLM_MAX_AGENTS == 5
    for n in OFFLINE_SCALE_PRESETS:
        agents = generate_agents(BERLIN, n, mode="dynamic", max_agents=OFFLINE_MAX_AGENTS)
        assert len(agents) == n
        assert_no_clones(agents)
        # Deterministic re-run
        again = generate_agents(BERLIN, n, mode="dynamic", max_agents=OFFLINE_MAX_AGENTS)
        assert [a.user_name for a in agents] == [a.user_name for a in again]


def test_fixed_test_never_scales_past_five():
    agents = generate_agents(BERLIN, 50, mode="fixed_test", max_agents=50)
    assert len(agents) == 5


def test_scheduler_never_self_as_partner():
    sched = SocialInteractionScheduler(scenario=BERLIN)
    agent_ids = list(range(10))
    posts = [{"post_id": i + 1, "user_id": i % 10} for i in range(12)]
    comments = [
        {"comment_id": i + 1, "post_id": 1, "user_id": (i + 3) % 10} for i in range(15)
    ]
    for aid in agent_ids:
        for rnd in (1, 2):
            plan = sched.plan(
                agent_id=aid,
                round_n=rnd,
                agent_ids=agent_ids,
                posts=posts,
                comments=comments,
                call_index=rnd,
            )
            assert plan.partner_agent_id != aid
            assert plan.self_reference_blocked is True
    summary = sched.summary()
    assert summary["self_as_partner_count"] == 0
    assert summary["plans"] == 20


def test_position_tracker_polarity_distribution_scales():
    tracker = PositionTracker()
    agents = [{"agent_id": i, "goal": "Dafür: Autoverbot unterstützen."} for i in range(25)]
    agents[1]["goal"] = "Dagegen: Autoverbot ablehnen."
    tracker.seed_from_goals(agents, step=0)
    dist = tracker.polarity_distribution()
    assert sum(dist.values()) == 25
    payload = tracker.positions_payload()
    assert "polarity_distribution" in payload
    assert payload["agent_count"] == 25


def test_live_llm_path_clamps_to_five(results_dir: Path):
    """Paid LLM path must not unlock 10/25/50 even if requested."""
    engine = SimulationEngine(load_settings())  # no model_factory → live path
    started = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        agent_count=50,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    # Without API key → blocked, but agent_count already clamped to 5.
    state = started["state"]
    assert state["agent_count"] == 5
    assert state["status"] == "blocked"


def test_manual_offline_scale_10(results_dir: Path):
    engine = SimulationEngine(load_settings())
    started = engine.start(
        BERLIN,
        mode="manual",
        max_steps=2,
        agent_count=10,
        agent_generation="dynamic",
    )
    assert started["ok"] is True
    state = _wait_terminal(engine, deadline_s=180)
    assert state["status"] in {"completed", "limit_reached"}
    assert state["agent_count"] == 10
    assert state["execution_kind"] == "manual"
    assert state["society_overview"]["agent_count"] == 10
    assert state["society_overview"]["paid_llm_unlocked_for_scale"] is False
    assert len(state["agents"]) == 10
    assert state["agent_positions"]["agent_count"] == 10
    kg = state["knowledge_graph"]
    assert kg["counts_by_kind"].get("actor") == 10
    assert "network" in kg


@pytest.mark.parametrize("n", [10, 25, 50])
def test_scripted_offline_scale_presets(results_dir: Path, n: int):
    ScriptedReactiveModel.shared_scheduler = SocialInteractionScheduler(scenario=BERLIN)
    t0 = time.perf_counter()
    tracemalloc.start()
    engine = SimulationEngine(
        load_settings(),
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid, label=f"p6-{n}"),
    )
    started = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        agent_count=n,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    assert started["ok"] is True, started.get("error")
    assert started["state"]["agent_count"] == n
    assert started["state"]["execution_kind"] == "scripted"
    deadline = 120 if n == 10 else (300 if n == 25 else 600)
    state = _wait_terminal(engine, deadline_s=deadline)
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert state["status"] in {"completed", "limit_reached"}, (
        state.get("status"),
        state.get("stop_reason"),
        state.get("error"),
    )
    assert state["agent_count"] == n
    assert state["execution_kind"] == "scripted"
    assert len(state["agents"]) == n
    assert state["agent_positions"]["agent_count"] == n
    assert "polarity_distribution" in state["agent_positions"]
    kg = state["knowledge_graph"]
    assert kg["counts_by_kind"].get("actor") == n
    assert kg.get("network", {}).get("actors") == n
    # Graph should remain stable (finite edges, no crash).
    assert kg["edge_count"] >= 0
    assert state["scheduler"]["self_as_partner_count"] == 0
    overview = state["society_overview"]
    assert overview["agent_count"] == n
    assert overview["paid_llm_unlocked_for_scale"] is False

    # Persist meta checks
    result_path = state.get("result_path")
    assert result_path
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    assert data["meta"]["phase"] in (7, 8)
    assert data["meta"]["agent_count"] == n
    assert data["meta"]["execution_kind"] == "scripted"
    assert data["meta"]["live_llm_limits_unchanged"]["max_agents"] == 5
    assert data["meta"]["live_llm_limits_unchanged"]["max_cost_usd_per_run"] == 0.20

    # Soft runtime/memory evidence (recorded for the Phase-6 report).
    state["_phase6_bench"] = {
        "agents": n,
        "elapsed_s": round(elapsed, 3),
        "peak_kib": round(peak / 1024, 1),
        "status": state["status"],
        "edges": kg["edge_count"],
        "social": kg.get("social_relations"),
    }
    print(f"\n[phase6 bench] n={n} elapsed={elapsed:.2f}s peak={peak/1024:.0f}KiB "
          f"status={state['status']} edges={kg['edge_count']}")


def test_approval_and_reserve_still_required_for_live(results_dir: Path):
    engine = SimulationEngine(load_settings())
    blocked = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        agent_count=3,
        llm_run_approved=False,
    )
    assert blocked["ok"] is False
    assert blocked["state"]["status"] == "blocked"
    assert "Freigabe" in (blocked.get("error") or "")
