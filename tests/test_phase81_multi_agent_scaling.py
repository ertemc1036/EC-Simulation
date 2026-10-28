"""Phase 8.1 — Controlled Multi-Agent Scaling (offline only).

Live fixture ``20261028T130906Z-7080b1e4`` (KI-Lernassistenten): personas OK,
but scheduler waves=0 / plans=0 — V3 wires waves into execution.

No provider calls. Ceilings unchanged. $0.
"""
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
    OFFLINE_MAX_AGENTS,
    OFFLINE_SCALE_PRESETS,
    load_settings,
)
from backend.simulation.agent_generator import generate_dynamic_agents
from backend.simulation.cost_token_planner import plan_run, planner_status_payload
from backend.simulation.engine import SimulationEngine
from backend.simulation.interaction_engine import (
    MultiAgentInteractionEngine,
    expected_waves,
    interaction_status_payload,
)
from backend.simulation.simulation_intelligence import build_simulation_intelligence
from backend.simulation.social_scheduler import (
    SCHEDULER_WAVE_PATHS,
    SocialInteractionScheduler,
    wave_count_for,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_7080 = ROOT / "data" / "results" / "20261028T130906Z-7080b1e4.json"

DDF_SCENARIO = (
    "Deutschland führt ab 2028 KI-Lernassistenten an allen öffentlichen Schulen ein. "
    "Simuliere eine Diskussion zwischen Lehrkräften, Eltern und Bildungsforschern über "
    "Lernleistungen, Datenschutz, Chancengleichheit und die Rolle der Lehrkräfte."
)

VIER = (
    "Deutschland führt ab 2030 eine Vier-Tage-Woche bei vollem Lohnausgleich ein. "
    "Simuliere die Diskussion zwischen Arbeitnehmern, Unternehmern und "
    "Wirtschaftsexperten. Untersuche die Auswirkungen auf Produktivität, "
    "Beschäftigung, Lebensqualität und Wettbewerbsfähigkeit."
)


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "4")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "20000")
    monkeypatch.setenv("MAX_TOKENS_PER_CALL", "500")
    monkeypatch.setenv("MAX_COST_USD_PER_RUN", "0.20")
    monkeypatch.setenv("MAX_PARALLEL_LLM_CALLS", "1")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    return out


def _wait_terminal(engine: SimulationEngine, *, deadline_s: float = 420.0) -> dict:
    end = time.time() + deadline_s
    while time.time() < end:
        st = engine.get_state()
        if st.get("status") in {"completed", "limit_reached", "failed", "blocked"}:
            return st
        time.sleep(0.05)
    raise AssertionError("simulation did not reach terminal status")


def test_live_ceilings_unchanged_phase81():
    cfg = load_settings()
    payload = cfg.status_payload()
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    assert payload["live_test_limits"]["max_agents"] == 5
    assert payload["live_test_limits"]["max_cost_usd_per_run"] == 0.20
    assert payload["offline_scale"]["paid_llm_unlocked"] is False
    assert payload["phase"] == "8"
    assert payload["society_intelligence"]["quality_fix"] in {"8.1", "8.2", "8.3"}
    assert payload["scalable_foundation"]["persona_engine"] in {"v3", "v4"}
    assert payload["scalable_foundation"]["scheduler_version"] == 3
    assert payload["scalable_foundation"]["paid_llm_unlocked"] is False
    assert payload["society_intelligence"]["multi_agent_interaction_engine"] is True
    assert payload["society_intelligence"]["scheduler_waves_in_execution"] is True


def test_scheduler_v3_pending_take_plan_and_paths():
    sched = SocialInteractionScheduler(scenario="v3-waves", wave_size=10)
    ids = list(range(25))
    posts = [{"post_id": i + 1, "user_id": i % 25} for i in range(25)]
    comments = [{"comment_id": i + 1, "user_id": (i + 3) % 25} for i in range(40)]
    plans = sched.plan_round(round_n=1, agent_ids=ids, posts=posts, comments=comments)
    assert len(plans) == 25
    assert wave_count_for(25) == 3
    assert sched.summary()["backend"] == "local_scheduler_v3"
    assert sched.summary()["waves"] == 3
    assert sched.summary()["pending_plans"] == 25

    taken = sched.take_plan(7, 1)
    assert taken is not None
    assert taken.agent_id == 7
    assert taken.wave is not None
    assert sched.take_plan(7, 1) is None  # consumed

    sched.mark_wave_executed(round_n=1, wave=0, agent_ids=ids[:10], path="scripted")
    summary = sched.summary()
    assert summary["waves_executed"] == 1
    assert "scripted" in summary["execution_paths_used"]
    assert summary["self_as_partner_count"] == 0
    assert "scripted" in SCHEDULER_WAVE_PATHS
    assert SCHEDULER_WAVE_PATHS["scripted"]["consume_plan"] is True
    assert SCHEDULER_WAVE_PATHS["live_llm"]["consume_plan"] is False
    assert SCHEDULER_WAVE_PATHS["manual"]["wave_env_step"] is True


def test_interaction_engine_prepare_round():
    eng = MultiAgentInteractionEngine(scenario="ix", wave_size=10)
    ids = list(range(50))
    posts = [{"post_id": 1, "user_id": 0}]
    plans = eng.prepare_round(
        round_n=1, agent_ids=ids, posts=posts, comments=[], path="scripted"
    )
    assert len(plans) == 50
    eng.mark_wave(round_n=1, wave=0, agent_ids=ids[:10], path="scripted")
    snap = eng.summary()
    assert snap["backend"] == "multi_agent_interaction_v3"
    assert snap["quality_fix"] in {"8.1", "8.2", "8.3"}
    assert snap["waves_planned"] == 5
    assert snap["waves_executed"] == 1
    assert snap["self_as_partner_count"] == 0
    status = interaction_status_payload()
    assert status["paid_llm_unlocked"] is False
    assert status["sample_waves"]["n50"] == 5


def test_cost_planner_v3_waves():
    live = plan_run(agent_count=50, rounds=4, execution_kind="live_llm")
    assert live.agent_count == 5
    assert live.rounds == 2
    assert live.planner_version == 3
    assert live.planned_waves == 2  # 1 wave × 2 rounds
    assert live.paid_llm_unlocked is False

    offline = plan_run(agent_count=50, rounds=2, execution_kind="scripted")
    assert offline.agent_count == 50
    assert offline.planned_waves == 10  # 5 waves × 2
    assert offline.calls_per_wave == 10
    assert offline.planned_cost_usd == 0.0
    assert offline.paid_llm_unlocked is False

    status = planner_status_payload()
    assert status["version"] == 3
    assert status["paid_llm_unlocked"] is False


def test_persona_diversity_v3_scale():
    for n in (10, 25, 50):
        agents = generate_dynamic_agents(DDF_SCENARIO, n)
        assert len(agents) == n
        names = [a.name for a in agents]
        assert len(names) == len(set(names))
        roles = {a.role_label for a in agents}
        # Education stakeholders cycled — at least Lehr/Eltern/Bildung(+Schüler).
        blob = " ".join(f"{a.role_label} {a.occupation}" for a in agents).lower()
        assert "lehr" in blob
        assert "eltern" in blob or "mutter" in blob or "vater" in blob
        assert "bildung" in blob
        assert len(roles) >= 3
        for a in agents:
            occ = (a.occupation or "").lower()
            assert "datenschutz" not in occ
            assert "lehrkr f" not in occ


def test_fixture_7080b1e4_documents_missing_waves():
    """Historical live run: personas fixed, scheduler waves were 0 (pre-8.1)."""
    if not LIVE_FIXTURE_7080.is_file():
        pytest.skip("7080b1e4 fixture missing")
    data = json.loads(LIVE_FIXTURE_7080.read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] in {"8.0", "8.1", "8.2", "8.3"}
    agents = data.get("agents") or []
    assert len(agents) == 3
    occs = " ".join(str(a.get("occupation") or "") for a in agents).lower()
    assert "datenschutz" not in occs
    sched = data.get("scheduler") or {}
    # Pre-8.1 live path never planned waves — documented gap this phase closes offline.
    assert int(sched.get("waves") or 0) == 0
    assert int(sched.get("plans") or 0) == 0


@pytest.mark.parametrize("n", [10, 25, 50])
def test_offline_scripted_waves_in_execution(results_dir: Path, n: int):
    from backend.simulation.scripted_model import ScriptedReactiveModel

    cfg = load_settings()
    rounds = 2
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid, label=f"p81-{n}"),
    )
    started = engine.start(
        scenario=VIER,
        mode="llm",
        language="de",
        max_steps=rounds,
        agent_count=n,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    assert started.get("ok") is True, started.get("error")
    st = _wait_terminal(engine, deadline_s=600.0)
    assert st["status"] in {"completed", "limit_reached"}
    assert st.get("execution_kind") == "scripted"
    # Scripted counters are synthetic — provider unlock / plan stay $0.
    assert (st.get("cost_plan") or {}).get("paid_llm_unlocked") is False
    assert (st.get("cost_plan") or {}).get("planner_version") == 3
    assert (st.get("cost_plan") or {}).get("planned_cost_usd") == 0.0

    sched = st.get("scheduler") or {}
    assert sched.get("backend") == "local_scheduler_v3"
    assert sched.get("self_as_partner_count", 0) == 0
    assert int(sched.get("waves_executed") or 0) >= wave_count_for(n)
    assert int(sched.get("waves_planned") or sched.get("waves") or 0) >= wave_count_for(n)
    assert "scripted" in (sched.get("execution_paths_used") or [])

    iex = st.get("interaction_engine") or {}
    assert iex.get("backend") == "multi_agent_interaction_v3"
    assert iex.get("path") == "scripted"
    assert int(iex.get("waves_executed") or 0) >= wave_count_for(n)

    intel = st.get("simulation_intelligence") or {}
    assert intel.get("quality_fix") in {"8.1", "8.2", "8.3"}
    assert intel.get("persona_engine") in {"v3", "v4"}
    assert intel.get("paid_llm_unlocked") is False
    assert int((intel.get("scheduler") or {}).get("waves_executed") or 0) >= 1
    assert n <= OFFLINE_MAX_AGENTS
    assert expected_waves(n, rounds) == wave_count_for(n) * rounds


def test_offline_manual_waves_in_execution(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=DDF_SCENARIO,
        mode="manual",
        language="de",
        max_steps=2,
        agent_count=12,
        agent_generation="dynamic",
    )
    assert started.get("ok") is True
    st = _wait_terminal(engine)
    assert st["status"] in {"completed", "limit_reached"}
    assert (st.get("cost_summary") or {}).get("api_calls", 0) == 0
    assert float((st.get("cost_summary") or {}).get("estimated_cost_usd") or 0) == 0.0
    sched = st.get("scheduler") or {}
    assert sched.get("backend") == "local_scheduler_v3"
    assert int(sched.get("waves_executed") or 0) >= 2  # 12 agents → 2 waves × ≥1 round
    assert "manual" in (sched.get("execution_paths_used") or [])
    assert (st.get("simulation_intelligence") or {}).get("quality_fix") in {"8.1", "8.2", "8.3"}
    assert (st.get("cost_plan") or {}).get("paid_llm_unlocked") is False


def test_simulation_intelligence_v3_flags():
    agents = [a.to_dict() for a in generate_dynamic_agents(DDF_SCENARIO, 10)]
    intel = build_simulation_intelligence(
        agents=agents,
        execution_kind="scripted",
        agent_count=10,
        max_steps=2,
        step=1,
        cost_plan=plan_run(agent_count=10, rounds=2, execution_kind="scripted"),
        scheduler={
            "backend": "local_scheduler_v3",
            "plans": 10,
            "waves": 1,
            "waves_planned": 1,
            "waves_executed": 1,
            "self_as_partner_count": 0,
            "scalable": True,
            "execution_paths_used": ["scripted"],
        },
        society_analytics={"kpis": {"memory_coverage": 1.0}},
        discussion={"repetition_blocks": 0, "argument_count": 5},
        agent_positions={
            "polarity_distribution": {"conditional": 4, "contra": 3, "pro": 3},
            "change_count": 1,
        },
        interaction_engine={
            "backend": "multi_agent_interaction_v3",
            "quality_fix": "8.2",
            "waves_executed": 1,
            "waves_planned": 1,
            "path": "scripted",
        },
    )
    assert intel["quality_fix"] in {"8.1", "8.2", "8.3"}
    assert intel["persona_engine"] in {"v3", "v4"}
    assert intel["persona"]["ok"] is True
    assert intel["persona"]["unique_names"] == 10
    assert intel["kpis"]["waves_executed"] == 1
    assert intel["paid_llm_unlocked"] is False
