"""Phase 8.0 — Scalable Simulation Foundation (offline only).

Live fixture ``20261028T122857Z-ddfedb31`` (KI-Lernassistenten Schulen):
- Occupations ``lehrkr ften`` / ``eltern`` / ``datenschutz``
- Bildungsforscher missing; topic nouns as stakeholders
- English fallback names (Logan Harper, Noa Noa)

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
from backend.simulation.agent_generator import (
    extract_stakeholder_roles,
    generate_dynamic_agents,
)
from backend.simulation.cost_token_planner import plan_run, planner_status_payload
from backend.simulation.engine import SimulationEngine
from backend.simulation.simulation_intelligence import build_simulation_intelligence
from backend.simulation.social_scheduler import SocialInteractionScheduler
from backend.simulation.topic_context import (
    detect_topic_domain,
    measure_label,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_DDF = ROOT / "data" / "results" / "20261028T122857Z-ddfedb31.json"
LIVE_FIXTURE_732 = ROOT / "data" / "results" / "20261028T115954Z-c16c749b.json"
LIVE_FIXTURE_731 = ROOT / "data" / "results" / "20261028T112739Z-bdfb9e9d.json"

DDF_SCENARIO = (
    "Deutschland führt ab 2028 KI-Lernassistenten an allen öffentlichen Schulen ein. "
    "Simuliere eine Diskussion zwischen Lehrkräften, Eltern und Bildungsforschern über "
    "Lernleistungen, Datenschutz, Chancengleichheit und die Rolle der Lehrkräfte."
)

LIVE_SCENARIO_732 = (
    "Deutschland führt ab 2030 eine gesetzliche Vier-Tage-Woche mit 32 "
    "Arbeitsstunden bei vollem Lohnausgleich ein. Welche wirtschaftlichen und "
    "gesellschaftlichen Folgen hätte diese Entscheidung für Arbeitnehmer, "
    "Unternehmen und den Arbeitsmarkt?"
)

VIER = (
    "Deutschland führt ab 2030 eine Vier-Tage-Woche bei vollem Lohnausgleich ein. "
    "Simuliere die Diskussion zwischen Arbeitnehmern, Unternehmern und "
    "Wirtschaftsexperten. Untersuche die Auswirkungen auf Produktivität, "
    "Beschäftigung, Lebensqualität und Wettbewerbsfähigkeit."
)
BGE = (
    "Deutschland führt ab 2030 ein bedingungsloses Grundeinkommen von 1.500 € "
    "monatlich für alle Erwachsenen ein."
)
MIETE = "Deutschland führt einen bundesweiten Mietendeckel für Bestandswohnungen ein."
KI_STEUER = (
    "Einführung einer KI-Steuer auf automatisierte Entscheidungsysteme in Unternehmen."
)
SCHULREFORM = (
    "Schulreform: verpflichtender Digitalunterricht ab Klasse 5 in allen Bundesländern."
)

REGRESSION_TOPICS = (
    ("vier-tage", VIER, "labor", ("vier-tage", "lohnausgleich")),
    ("bge", BGE, "income", ("grundeinkommen",)),
    ("mietendeckel", MIETE, "generic", ("mietendeckel",)),
    ("ki-steuer", KI_STEUER, "generic", ("ki-steuer",)),
    ("schulreform", SCHULREFORM, "education", ("schulreform", "digitalunterricht")),
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


def _wait_terminal(engine: SimulationEngine, *, deadline_s: float = 300.0) -> dict:
    end = time.time() + deadline_s
    while time.time() < end:
        st = engine.get_state()
        if st.get("status") in {"completed", "limit_reached", "failed", "blocked"}:
            return st
        time.sleep(0.05)
    raise AssertionError("simulation did not reach terminal status")


def test_live_ceilings_unchanged_phase80():
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
    assert payload["society_intelligence"]["quality_fix"] == "8.0"
    assert payload["scalable_foundation"]["persona_engine"] == "v2"
    assert payload["scalable_foundation"]["paid_llm_unlocked"] is False


def test_ddfedb31_persona_engine_v2_fixes():
    """Offline reproduction of live fixture persona failures."""
    assert detect_topic_domain(DDF_SCENARIO) == "education"
    assert "KI-Lernassistenten" in measure_label(DDF_SCENARIO)
    roles = extract_stakeholder_roles(DDF_SCENARIO)
    assert roles[:3] == ["Lehrkräfte", "Eltern", "Bildungsforscher"]
    assert "Datenschutz" not in roles
    assert "Chancengleichheit" not in roles

    agents = generate_dynamic_agents(DDF_SCENARIO, 3)
    assert len(agents) == 3
    labels = {a.role_label for a in agents}
    occs = [a.occupation or "" for a in agents]
    names = [a.name for a in agents]

    assert any("Lehr" in (l or "") for l in labels)
    assert any("Eltern" in (l or "") for l in labels)
    assert any("Bildung" in (l or "") for l in labels)

    for occ in occs:
        low = occ.lower()
        assert "datenschutz" not in low
        assert "lehrkr f" not in low
        assert "lehrkraeften" not in low
        assert occ != "eltern"

    # No duplicate first/last token (Noa Noa).
    for name in names:
        parts = name.split()
        assert len(parts) >= 2
        assert parts[0].lower() != parts[1].lower()

    # Fixture document still shows the historical bug (analysis only).
    if LIVE_FIXTURE_DDF.is_file():
        data = json.loads(LIVE_FIXTURE_DDF.read_text(encoding="utf-8"))
        hist = [a.get("occupation") for a in data.get("agents") or []]
        assert any(
            isinstance(o, str) and ("datenschutz" in o.lower() or "lehrkr" in o.lower())
            for o in hist
        )


def test_phase732_labor_regression_personas():
    roles = extract_stakeholder_roles(LIVE_SCENARIO_732)
    assert "Arbeitnehmer" in roles
    assert "Unternehmen" in roles or "Unternehmer" in roles
    assert "Ökonomen" in roles
    agents = generate_dynamic_agents(LIVE_SCENARIO_732, 3)
    labels = " ".join(a.role_label or "" for a in agents).lower()
    occs = " ".join(a.occupation or "" for a in agents).lower()
    assert "moderator" not in labels
    assert "betriebsrat" not in occs
    assert "wirtschaft" in labels or "ökonomen" in labels or "finanz" in occs


def test_cost_token_planner_live_vs_offline():
    live = plan_run(agent_count=50, rounds=4, execution_kind="live_llm")
    assert live.agent_count == 5
    assert live.rounds == 2
    assert live.planned_calls <= 10
    assert live.planned_cost_usd <= 0.20
    assert live.paid_llm_unlocked is False
    assert live.live_clamp_applied is True

    offline = plan_run(agent_count=50, rounds=4, execution_kind="scripted")
    assert offline.agent_count == 50
    assert offline.rounds == 4
    assert offline.planned_cost_usd == 0.0
    assert offline.paid_llm_unlocked is False

    manual = plan_run(agent_count=25, rounds=2, execution_kind="manual")
    assert manual.planned_calls == 0
    assert manual.planned_cost_usd == 0.0

    status = planner_status_payload()
    assert status["paid_llm_unlocked"] is False
    assert status["live_ceilings"]["max_agents"] == 5


def test_scalable_scheduler_waves_no_self():
    sched = SocialInteractionScheduler(scenario="scale-test", wave_size=10)
    ids = list(range(25))
    posts = [{"post_id": i + 1, "user_id": i % 25} for i in range(25)]
    comments = [{"comment_id": i + 1, "user_id": (i + 3) % 25} for i in range(40)]
    plans = sched.plan_round(
        round_n=1, agent_ids=ids, posts=posts, comments=comments
    )
    assert len(plans) == 25
    summary = sched.summary()
    assert summary["scalable"] is True
    assert summary["waves"] == 3
    assert summary["self_as_partner_count"] == 0
    assert summary["agent_coverage"] == 25
    for p in plans:
        assert p.partner_agent_id != p.agent_id


def test_simulation_intelligence_flags_ddf_issues():
    bad = [
        {
            "agent_id": 0,
            "name": "Logan Harper",
            "occupation": "lehrkr ften",
            "role_label": "Lehrkräften",
        },
        {
            "agent_id": 1,
            "name": "Noa Noa",
            "occupation": "datenschutz",
            "role_label": "Datenschutz",
        },
    ]
    intel = build_simulation_intelligence(
        agents=bad,
        execution_kind="manual",
        agent_count=2,
        max_steps=2,
        step=1,
        cost_plan=plan_run(agent_count=2, rounds=2, execution_kind="manual"),
        scheduler={"plans": 0, "self_as_partner_count": 0},
        society_analytics={"kpis": {}},
        discussion={"repetition_blocks": 0},
        agent_positions={"polarity_distribution": {"unknown": 2}, "change_count": 0},
    )
    assert intel["phase"] == 8
    assert intel["quality_fix"] == "8.0"
    assert intel["persona"]["issue_count"] >= 2
    assert intel["paid_llm_unlocked"] is False

    good = [a.to_dict() for a in generate_dynamic_agents(DDF_SCENARIO, 3)]
    intel2 = build_simulation_intelligence(
        agents=good,
        execution_kind="manual",
        agent_count=3,
        max_steps=2,
        step=1,
        cost_plan=plan_run(agent_count=3, rounds=2, execution_kind="manual"),
        scheduler={"plans": 3, "waves": 1, "self_as_partner_count": 0, "scalable": True},
        society_analytics={"kpis": {"memory_coverage": 1.0}},
        discussion={"repetition_blocks": 0, "argument_count": 3},
        agent_positions={
            "polarity_distribution": {"conditional": 2, "contra": 1},
            "change_count": 0,
        },
    )
    assert intel2["persona"]["ok"] is True
    assert intel2["persona"]["issue_count"] == 0


@pytest.mark.parametrize("topic_id,scenario,domain,needles", REGRESSION_TOPICS)
def test_five_topic_persona_regression(topic_id, scenario, domain, needles):
    assert detect_topic_domain(scenario) == domain
    agents = generate_dynamic_agents(scenario, 3)
    measure = measure_label(scenario).lower()
    blob = " ".join(
        f"{a.name} {a.occupation} {a.role_label} {a.description}" for a in agents
    ).lower()
    assert any(n.lower() in measure or n.lower() in blob for n in needles)
    if domain == "labor":
        assert "grundeinkommen" not in blob
    if domain == "education":
        roles = extract_stakeholder_roles(scenario)
        assert "Datenschutz" not in roles
        assert detect_topic_domain(scenario) == "education"


def test_offline_manual_ddf_scenario_zero_calls(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=DDF_SCENARIO,
        mode="manual",
        language="de",
        max_steps=2,
        agent_count=3,
        agent_generation="dynamic",
    )
    assert started.get("ok") is True
    st = _wait_terminal(engine)
    assert st["status"] in {"completed", "limit_reached"}
    assert (st.get("cost_summary") or {}).get("api_calls", 0) == 0
    assert float((st.get("cost_summary") or {}).get("estimated_cost_usd") or 0) == 0.0
    agents = st.get("agents") or []
    assert len(agents) == 3
    occs = " ".join(str(a.get("occupation") or "") for a in agents).lower()
    assert "datenschutz" not in occs
    intel = st.get("simulation_intelligence") or {}
    assert intel.get("quality_fix") == "8.0"
    assert intel.get("paid_llm_unlocked") is False
    plan = st.get("cost_plan") or {}
    assert plan.get("planned_cost_usd") == 0.0
    assert plan.get("paid_llm_unlocked") is False


@pytest.mark.parametrize("n", [10, 25])
def test_offline_scale_scripted_phase80(results_dir: Path, n: int):
    from backend.simulation.scripted_model import ScriptedReactiveModel

    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid, label=f"p80-{n}"),
    )
    started = engine.start(
        scenario=VIER,
        mode="llm",
        language="de",
        max_steps=2,
        agent_count=n,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    assert started.get("ok") is True, started.get("error")
    st = _wait_terminal(engine, deadline_s=420.0)
    assert st["status"] in {"completed", "limit_reached"}
    # Scripted records synthetic call counters — never paid OpenAI.
    assert st.get("execution_kind") == "scripted"
    assert (st.get("cost_plan") or {}).get("paid_llm_unlocked") is False
    assert (st.get("simulation_intelligence") or {}).get("paid_llm_unlocked") is False
    assert st.get("agent_count") == n or len(st.get("agents") or []) == n
    sched = st.get("scheduler") or {}
    assert sched.get("self_as_partner_count", 0) == 0
    assert sched.get("scalable") is True
    assert (st.get("simulation_intelligence") or {}).get("phase") == 8
    assert n <= OFFLINE_MAX_AGENTS


def test_phase731_fixture_reingest_still_ok():
    if not LIVE_FIXTURE_731.is_file():
        pytest.skip("7.3.1 fixture missing")
    from backend.simulation.discussion_engine import DiscussionEngine

    data = json.loads(LIVE_FIXTURE_731.read_text(encoding="utf-8"))
    eng = DiscussionEngine()
    for step in range(3):
        eng.ingest_platform(
            posts=data["posts"], comments=data["comments"], step=step, round_n=step
        )
    snap = eng.snapshot()
    assert snap["repetition_blocks"] == 0
    assert snap["argument_count"] == 6
    assert snap["quality_fix"] in {"7.3.2", "8.0"}


def test_phase732_fixture_positions_documented():
    if not LIVE_FIXTURE_732.is_file():
        pytest.skip("7.3.2 fixture missing")
    data = json.loads(LIVE_FIXTURE_732.read_text(encoding="utf-8"))
    # Historical fixture keeps its pre-8.0 quality_fix tag.
    assert data["meta"]["quality_fix"] in {"7.3.1", "7.3.2", "8.0"}
    agents = generate_dynamic_agents(data["meta"]["scenario"], 3)
    assert all(a.role_label != "Moderator" for a in agents)
