"""Phase 8.4 — Simulation Reliability (offline, no live LLM).

Stance V4, Grounding V4, Opinion Dynamics V4, Quality Dashboard V4.
Fixtures: e251172c + d1fb1b1c + ab541048. Scenarios A–E, adversarial, 10/25/50.
"""
from __future__ import annotations

import json
import math
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
from backend.simulation.engine import SimulationEngine
from backend.simulation.interaction_quality import (
    GROUNDING_OVERLAP_FLOOR,
    build_interaction_quality,
    evaluate_argument_grounded_interactions,
    score_reply_grounding,
)
from backend.simulation.opinion_dynamics import OpinionDynamics
from backend.simulation.persona_motivation import motivation_coverage
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import (
    PositionTracker,
    classify_stance_v4,
    _polarity,
)
from backend.simulation.social_scheduler import wave_count_for
from backend.simulation.topic_context import detect_topic_domain

ROOT = Path(__file__).resolve().parents[1]
LIVE_E251 = ROOT / "data" / "results" / "20261028T145656Z-e251172c.json"
LIVE_D1FB = ROOT / "data" / "results" / "20261028T142011Z-d1fb1b1c.json"
LIVE_AB54 = ROOT / "data" / "results" / "20261028T134318Z-ab541048.json"

SCENARIO_A = (
    "Deutschland führt ab 2028 KI-Lernassistenten an allen öffentlichen Schulen ein. "
    "Simuliere eine kontroverse Diskussion zwischen Lehrkräften, Eltern und "
    "Bildungsforschern über Lernleistungen, Datenschutz, Chancengleichheit und "
    "die Rolle der Lehrkräfte."
)
SCENARIO_B = SCENARIO_A
SCENARIO_C = (
    "Deutschland führt ab 2030 eine Vier-Tage-Woche bei vollem Lohnausgleich ein. "
    "Simuliere die Diskussion zwischen Arbeitnehmern, Unternehmern und "
    "Wirtschaftsexperten."
)
SCENARIO_D = (
    "Deutschland führt ab 2030 ein bedingungsloses Grundeinkommen von 1.500 € "
    "monatlich ein. Simuliere die Diskussion zwischen Empfängergruppen, "
    "Arbeitgebern und Fiskalpolitik."
)
SCENARIO_E = (
    "Berlin plant, ab 2030 private Autos innerhalb des S-Bahn-Rings vollständig "
    "zu verbieten. Drei Interessengruppen diskutieren: Bürgerinnen und Bürger, "
    "Unternehmen sowie die Politik."
)

OFFLINE_SCENARIOS = {
    "B": (SCENARIO_B, "education"),
    "C": (SCENARIO_C, "labor"),
    "D": (SCENARIO_D, "income"),
    "E": (SCENARIO_E, "mobility"),
}


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


def _wait_terminal(engine: SimulationEngine, *, deadline_s: float = 600.0) -> dict:
    end = time.time() + deadline_s
    while time.time() < end:
        st = engine.get_state()
        if st.get("status") in {"completed", "limit_reached", "failed", "blocked"}:
            return st
        time.sleep(0.05)
    raise AssertionError("simulation did not reach terminal status")


@pytest.fixture()
def fixture_e251() -> dict:
    assert LIVE_E251.is_file(), f"missing fixture {LIVE_E251}"
    return json.loads(LIVE_E251.read_text(encoding="utf-8"))


def test_ceilings_unchanged_phase84():
    cfg = load_settings()
    payload = cfg.status_payload()
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    assert payload["live_test_limits"]["max_agents"] == 5
    assert payload["live_test_limits"]["max_cost_usd_per_run"] == 0.20
    assert payload["offline_scale"]["paid_llm_unlocked"] is False
    assert cfg.max_tokens_per_run == 20000
    assert cfg.max_tokens_per_call == 500
    assert payload["phase"] == "8"
    assert payload["society_intelligence"]["quality_fix"] == "8.4"
    assert payload["society_intelligence"]["stance_classification_v4"] is True
    assert payload["society_intelligence"]["opinion_dynamics_v4"] is True
    assert payload["society_intelligence"]["argument_grounded_interaction_v4"] is True


def test_scenario_a_e251_stance_v4_fixes_false_pro(fixture_e251: dict):
    """Audit false-pro Thomas C2 + Fatima C4 → conditional under Stance V4."""
    c2 = fixture_e251["comments"][1]["content"]
    c4 = fixture_e251["comments"][3]["content"]
    assert classify_stance_v4(c2) == "conditional"
    assert classify_stance_v4(c4) == "conditional"
    assert classify_stance_v4(fixture_e251["comments"][0]["content"]) == "conditional"
    assert classify_stance_v4(fixture_e251["comments"][4]["content"]) == "conditional"
    assert classify_stance_v4(fixture_e251["comments"][5]["content"]) == "conditional"
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=fixture_e251["agents"],
        posts=fixture_e251["posts"],
        comments=fixture_e251["comments"],
        reactions=fixture_e251.get("reactions") or [],
        step=2,
    )
    payload = tracker.positions_payload()
    assert payload["stance_classifier"] == "v4"
    assert payload["quality_fix"] == "8.4"
    dist = payload["polarity_distribution"]
    assert dist.get("unknown", 0) == 0
    assert dist.get("pro", 0) == 0
    assert dist.get("conditional", 0) == 3


def test_stance_v4_dafuer_not_false_pro():
    text = (
        "Deshalb plädiere ich dafür, dass der Einsatz von KI-Lernassistenten "
        "nicht einfach technikgetrieben erfolgen darf. Es braucht klare, "
        "pädagogisch fundierte Konzepte."
    )
    assert "dafür" in text.lower()
    assert classify_stance_v4(text) == "conditional"
    assert _polarity(text) == "conditional"


def test_stance_v4_hard_labels_and_regressions():
    assert classify_stance_v4("Ich unterstütze das Verbot klar — sinnvolles Verbot.") == "pro"
    assert classify_stance_v4("Ich lehne das Verbot ab — dagegen.") == "contra"
    if LIVE_D1FB.is_file():
        data = json.loads(LIVE_D1FB.read_text(encoding="utf-8"))
        assert classify_stance_v4(data["comments"][1]["content"]) == "conditional"
        assert classify_stance_v4(data["comments"][0]["content"]) == "conditional"
    if LIVE_AB54.is_file():
        data = json.loads(LIVE_AB54.read_text(encoding="utf-8"))
        assert classify_stance_v4(data["comments"][0]["content"]) == "conditional"
        assert classify_stance_v4(data["comments"][1]["content"]) == "conditional"


def test_grounding_v4_e251_and_floor(fixture_e251: dict):
    snap = evaluate_argument_grounded_interactions(comments=fixture_e251["comments"])
    assert snap["engine"] == "v4"
    assert snap["quality_fix"] == "8.4"
    assert snap["reply_count"] == 3
    assert snap["grounded_reply_rate"] == 1.0
    assert snap["invented_attribution_count"] == 0
    assert snap["overlap_floor"] == GROUNDING_OVERLAP_FLOOR
    # Bildung↔IT bag lifts C6 above fragile floor when possible.
    c6 = next(e for e in snap["evaluations"] if e["comment_id"] == 6)
    assert c6["grounded"] is True
    assert c6["overlap"] >= 0.18


def test_grounding_adversarial_digitale_kluft():
    parent = (
        "Als Bildungsforscherin sehe ich fehlende empirische Evidenz und plädiere "
        "für begleitete Pilotprojekte statt anekdotischer Erfolgsversprechen."
    )
    bad = (
        "→Kommentar#1 Ich stimme Ihnen zu, dass die Herausforderung der digitalen "
        "Kluft nicht unterschätzt werden darf und Förderprogramme nötig sind."
    )
    g = score_reply_grounding(bad, parent, parent_comment_id=1)
    assert g["engine"] == "v4"
    assert g["invented_attribution"] is True
    assert g["grounded"] is False
    iq = build_interaction_quality(
        comments=[
            {"comment_id": 1, "user_id": 0, "content": parent},
            {"comment_id": 2, "user_id": 1, "content": bad},
        ],
        agents=[],
        stance_clarity=1.0,
        debate_quality_score=0.5,
        opinion_stability=0.8,
    )
    assert iq["engine"] == "v4"
    assert iq["ok"] is False or iq["components"]["invented_attribution_rate"] > 0
    if LIVE_D1FB.is_file():
        data = json.loads(LIVE_D1FB.read_text(encoding="utf-8"))
        g2 = score_reply_grounding(
            data["comments"][4]["content"],
            data["comments"][2]["content"],
            parent_comment_id=3,
        )
        assert g2["invented_attribution"] is True


def test_opinion_dynamics_v4_shifting_share_and_signals():
    od = OpinionDynamics()
    od.set_interaction_signals(
        grounded_reply_share=1.0,
        motivation_coverage=1.0,
        weak_overlap_rate=0.0,
    )
    od.record_round(
        step=1,
        polarity_distribution={"pro": 1, "contra": 1, "conditional": 1},
        change_events=[],
        agent_count=3,
    )
    od.record_round(
        step=2,
        polarity_distribution={"conditional": 3},
        change_events=[
            {
                "agent_id": 0,
                "from_polarity": "pro",
                "to_polarity": "conditional",
                "step": 2,
                "evidence": "shift",
            }
        ],
        agent_count=3,
    )
    snap = od.snapshot()
    assert snap["engine"] == "v4"
    assert snap["version"] == 4
    assert snap["shifting_share"] == 0.0
    assert snap["stance_clarity"] == 1.0
    assert snap["weak_overlap_rate"] == 0.0
    # Bare shifting dilutes clarity under V4.
    assert _polarity(
        "Ich ändere meine Position nach dem Kommentar — werde überzeugt: "
        "Piloten und Ausnahmeregelungen statt pauschaler Vollauszahlung "
        "(nur mit Pilot und klaren Abbruchkriterien)."
    ) == "conditional"


@pytest.mark.parametrize("key", sorted(OFFLINE_SCENARIOS.keys()))
def test_offline_scenarios_b_e_reliability(results_dir: Path, key: str):
    scenario, domain = OFFLINE_SCENARIOS[key]
    assert detect_topic_domain(scenario) == domain
    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid, k=key: ScriptedReactiveModel(
            agent_id=aid, label=f"p84-{k}"
        ),
    )
    started = engine.start(
        scenario=scenario,
        mode="llm",
        max_steps=2,
        agent_count=10,
        agent_generation="dynamic",
        language="de",
        llm_run_approved=True,
    )
    assert started.get("ok") is True, started.get("error")
    st = _wait_terminal(engine)
    assert st["status"] in {"completed", "limit_reached"}
    assert st.get("execution_kind") == "scripted"
    plan = st.get("cost_plan") or {}
    assert plan.get("paid_llm_unlocked") is False
    assert float(plan.get("planned_cost_usd") or 0.0) == 0.0
    path = st.get("result_path")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] == "8.4"
    assert data["meta"]["persona_engine"] == "v4"
    assert data["agent_positions"]["stance_classifier"] == "v4"
    assert data["opinion_dynamics"]["engine"] == "v4"
    assert data["interaction_quality"]["engine"] == "v4"
    assert data["discussion"]["argument_grounded"]["engine"] == "v4"
    assert motivation_coverage(data["agents"])["coverage"] == 1.0
    dist = data["agent_positions"]["polarity_distribution"]
    total = sum(int(v) for v in dist.values()) or 1
    assert int(dist.get("unknown") or 0) / total == 0.0
    # Phase 8.4: bare shifting should not dominate income/scripted polarity.
    assert int(dist.get("shifting") or 0) / total <= 0.15
    grounded = evaluate_argument_grounded_interactions(comments=data["comments"])
    assert grounded["grounded_reply_rate"] == 1.0
    assert grounded["invented_attribution_count"] == 0
    assert float(data["interaction_quality"]["score"] or 0) >= 0.60
    blob = " ".join(
        (c.get("content") or "") for c in data.get("comments") or []
    ) + " ".join((p.get("content") or "") for p in data.get("posts") or [])
    if domain != "income":
        assert "Grundeinkommen" not in blob
    if domain != "education":
        assert "KI-Lernassistent" not in blob
    sources = {(c.get("source") or "") for c in (data.get("model_calls") or [])}
    assert "llm_api" not in sources
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False
    assert data["simulation_intelligence"]["quality_fix"] == "8.4"


@pytest.mark.parametrize("n", list(OFFLINE_SCALE_PRESETS))
def test_scripted_scale_10_25_50_reliability(results_dir: Path, n: int):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid: ScriptedReactiveModel(
            agent_id=aid, label=f"p84-{n}"
        ),
    )
    started = engine.start(
        scenario=SCENARIO_A if n <= 25 else SCENARIO_C,
        mode="llm",
        max_steps=2,
        agent_count=n,
        agent_generation="dynamic",
        language="de",
        llm_run_approved=True,
    )
    assert started.get("ok") is True, started.get("error")
    st = _wait_terminal(engine, deadline_s=600.0)
    assert st["status"] in {"completed", "limit_reached"}
    assert st.get("execution_kind") == "scripted"
    assert (st.get("cost_plan") or {}).get("paid_llm_unlocked") is False
    assert float((st.get("cost_plan") or {}).get("planned_cost_usd") or 0.0) == 0.0
    data = json.loads(Path(st["result_path"]).read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] == "8.4"
    assert data["agent_positions"]["stance_classifier"] == "v4"
    assert data["opinion_dynamics"]["engine"] == "v4"
    assert data["interaction_quality"]["engine"] == "v4"
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False
    assert data["simulation_intelligence"]["kpis"]["planned_cost_usd"] == 0.0
    assert all((a.get("motivation") or {}).get("values") for a in data["agents"])
    assert int((data.get("scheduler") or {}).get("waves_executed") or 0) >= wave_count_for(
        n
    )
    dist = data["agent_positions"]["polarity_distribution"]
    total = sum(int(v) for v in dist.values()) or 1
    assert int(dist.get("unknown") or 0) / total <= 0.35


def test_manual_n12_zero_cost_phase84(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=SCENARIO_A,
        mode="manual",
        max_steps=2,
        agent_count=12,
        agent_generation="dynamic",
        language="de",
    )
    assert started.get("ok") is True
    st = _wait_terminal(engine)
    assert st["status"] in {"completed", "limit_reached"}
    cost = st.get("cost_summary") or {}
    assert int(cost.get("api_calls") or 0) == 0
    assert float(cost.get("estimated_cost_usd") or 0.0) == 0.0
    data = json.loads(Path(st["result_path"]).read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] == "8.4"
    assert data["opinion_dynamics"]["engine"] == "v4"
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False


def test_offline_presets_and_ceilings_unchanged():
    assert tuple(OFFLINE_SCALE_PRESETS) == (10, 25, 50)
    assert OFFLINE_MAX_AGENTS == 50
    assert math.isclose(LIVE_TEST_MAX_AGENTS, 5)


def test_fixtures_present_for_regression():
    assert LIVE_E251.is_file()
    assert LIVE_D1FB.is_file()
    assert LIVE_AB54.is_file()
