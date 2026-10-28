"""Phase 8.5 — Realistic Multi-Agent Behavior (offline, no live LLM).

Stakeholder Motivation V5, Claim-Based Interaction V4, Argument Evolution V4,
Interaction Quality V5. Fixtures e251172c + d1fb1b1c + ab541048.
Scenarios A–E, adversarial, 10/25/50.
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
from backend.simulation.argument_intelligence import (
    ARGUMENT_ENGINE,
    build_argument_evolution,
    build_argument_intelligence,
    classify_claim_kind,
    classify_evolution_stage,
    enrich_argument_record,
)
from backend.simulation.claim_interaction import (
    CLAIM_INTERACTION_ENGINE,
    evaluate_claim_based_interactions,
    extract_claims,
    score_claim_addressing,
)
from backend.simulation.engine import SimulationEngine
from backend.simulation.interaction_quality import (
    build_interaction_quality,
    evaluate_argument_grounded_interactions,
    score_reply_grounding,
)
from backend.simulation.persona_motivation import (
    PERSONA_ENGINE,
    build_persona_motivation,
    motivation_coverage,
    stakeholder_affinity,
)
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import PositionTracker, classify_stance_v4
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


def test_ceilings_unchanged_phase85():
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
    assert payload["society_intelligence"]["quality_fix"] == "8.5"
    assert payload["society_intelligence"]["persona_engine"] == "v5"
    assert payload["society_intelligence"]["stakeholder_motivation_v5"] is True
    assert payload["society_intelligence"]["claim_based_interaction_v4"] is True
    assert payload["society_intelligence"]["argument_evolution_v4"] is True
    assert payload["society_intelligence"]["interaction_quality_v5"] is True
    assert payload["society_intelligence"]["stance_classification_v4"] is True
    assert PERSONA_ENGINE == "v5"
    assert ARGUMENT_ENGINE == "v4"
    assert CLAIM_INTERACTION_ENGINE == "v4"


def test_stakeholder_motivation_v5_fields_and_affinity():
    teach = build_persona_motivation(
        role_label="Lehrkraft",
        occupation="Gymnasiallehrerin",
        scenario=SCENARIO_A,
        style="evidenzbasiert",
    )
    parent = build_persona_motivation(
        role_label="Eltern",
        occupation="IT-Projektleiter",
        scenario=SCENARIO_A,
    )
    researcher = build_persona_motivation(
        role_label="Bildungsforscherin",
        occupation="Digitalisierungsforscherin",
        scenario=SCENARIO_A,
        style="analytisch",
    )
    assert teach.stakes and teach.claim_agenda
    assert teach.stakeholder_group == "practitioners"
    assert parent.stakeholder_group == "affected_public"
    assert researcher.engagement_bias == "prefer_evidence"
    assert researcher.domain == "education"
    # Same domain experts/practitioners share affinity signal.
    aff_tr = stakeholder_affinity(teach, researcher)
    aff_tp = stakeholder_affinity(teach, parent)
    assert aff_tr >= 0.2
    assert aff_tp >= 0.1
    # Mobility domain V5.
    citizen = build_persona_motivation(
        role_label="Bürgerin",
        occupation="Pendlerin",
        scenario=SCENARIO_E,
    )
    assert citizen.domain == "mobility"
    assert citizen.stakes


def test_scenario_a_e251_claim_and_evolution(fixture_e251: dict):
    """Fixture A: Stance V4 holds; claim addressing + evolution present."""
    comments = fixture_e251["comments"]
    for c in comments:
        assert classify_stance_v4(c["content"]) == "conditional"
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=fixture_e251["agents"],
        posts=fixture_e251["posts"],
        comments=comments,
        reactions=fixture_e251.get("reactions") or [],
        step=2,
    )
    pos = tracker.positions_payload()
    assert pos["stance_classifier"] == "v4"
    assert pos["polarity_distribution"].get("unknown", 0) == 0
    assert pos["polarity_distribution"].get("pro", 0) == 0

    grounded = evaluate_argument_grounded_interactions(comments=comments)
    assert grounded["engine"] == "v4"
    assert grounded["grounded_reply_rate"] == 1.0
    assert grounded["invented_attribution_count"] == 0

    claims = evaluate_claim_based_interactions(comments=comments)
    assert claims["engine"] == "v4"
    assert claims["reply_count"] == 3
    assert claims["claim_address_rate"] >= 0.34
    assert claims["ok"] is True

    args = [
        {
            "agent_id": c.get("user_id"),
            "snippet": c.get("content"),
            "comment_id": c.get("comment_id"),
        }
        for c in comments
    ]
    intel = build_argument_intelligence(arguments=args, agent_count=3)
    assert intel["engine"] == "v4"
    evo = intel["argument_evolution"]
    assert evo["engine"] == "v4"
    assert "stage_counts" in evo
    assert evo["score"] >= 0.0

    # Attach V5 motivations to fixture agents for IQ.
    agents = []
    for a in fixture_e251["agents"]:
        mot = build_persona_motivation(
            role_label=a.get("role_label"),
            occupation=a.get("occupation"),
            scenario=SCENARIO_A,
            style=str(a.get("style") or ""),
        )
        d = dict(a)
        d["motivation"] = mot.to_dict()
        agents.append(d)
    cov = motivation_coverage(agents)
    assert cov["engine"] == "v5"
    assert cov["coverage"] == 1.0
    assert cov["stake_coverage"] == 1.0

    iq = build_interaction_quality(
        comments=comments,
        agents=agents,
        stance_clarity=1.0,
        debate_quality_score=float(
            (intel.get("mean_quality_score") or 0.5)
        ),
        opinion_stability=0.85,
        argument_evolution_score=float(evo.get("score") or 0.0),
    )
    assert iq["engine"] == "v5"
    assert iq["quality_fix"] == "8.5"
    assert iq["score"] >= 0.55
    assert "claim_address_rate" in iq["components"]
    assert "argument_evolution_score" in iq["components"]
    assert iq["claim_based_interaction"]["engine"] == "v4"


def test_claim_addressing_positive_and_ignore():
    parent = (
        "Als Bildungsforscherin sehe ich fehlende empirische Evidenz und plädiere "
        "für begleitete Pilotprojekte statt anekdotischer Erfolgsversprechen. "
        "Ohne Langzeitdaten bleibt die Vollausrollung riskant."
    )
    claims = extract_claims(parent)
    assert len(claims) >= 1
    good = (
        "→Kommentar#1 Die fehlende empirische Evidenz und der Ruf nach "
        "Pilotprojekten sind nachvollziehbar; Langzeitdaten müssen zuerst greifen."
    )
    g = score_claim_addressing(good, parent, parent_comment_id=1)
    assert g["claim_address_rate"] >= 0.5
    assert g["claims_addressed"] >= 1
    weak = (
        "→Kommentar#1 Totally unrelated weather commentary about rain tomorrow."
    )
    w = score_claim_addressing(weak, parent, parent_comment_id=1)
    assert w["claim_address_rate"] <= g["claim_address_rate"]


def test_argument_evolution_stages():
    introduce = "KI-Lernassistenten haben großes Potenzial für individualisiertes Lernen."
    challenge = (
        "→Kommentar#1 Allerdings fehlen belastbare Studien und Evaluation; "
        "Pilotprojekte zuerst."
    )
    refine = (
        "→Kommentar#2 Ergänzend: nur schrittweise unter klaren Datenschutzregeln "
        "und pädagogischen Konzepten."
    )
    assert classify_evolution_stage(introduce) in {"introduce", "support"}
    assert classify_claim_kind(challenge) in {"evidence_demand", "caveat", "rebuttal", "develop"}
    assert classify_evolution_stage(challenge) in {"challenge", "refine", "develop"}
    assert classify_evolution_stage(refine) in {"refine", "develop", "challenge"}
    enriched = [
        enrich_argument_record({"agent_id": 0, "snippet": introduce}),
        enrich_argument_record({"agent_id": 1, "snippet": challenge}),
        enrich_argument_record({"agent_id": 2, "snippet": refine}),
    ]
    evo = build_argument_evolution(enriched=enriched)
    assert evo["engine"] == "v4"
    assert sum(evo["stage_counts"].values()) == 3


def test_grounding_adversarial_still_invented():
    parent = (
        "Als Bildungsforscherin sehe ich fehlende empirische Evidenz und plädiere "
        "für begleitete Pilotprojekte statt anekdotischer Erfolgsversprechen."
    )
    bad = (
        "→Kommentar#1 Ich stimme Ihnen zu, dass die Herausforderung der digitalen "
        "Kluft nicht unterschätzt werden darf und Förderprogramme nötig sind."
    )
    g = score_reply_grounding(bad, parent, parent_comment_id=1)
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
        argument_evolution_score=0.2,
    )
    assert iq["engine"] == "v5"
    assert iq["components"]["invented_attribution_rate"] > 0
    if LIVE_D1FB.is_file():
        data = json.loads(LIVE_D1FB.read_text(encoding="utf-8"))
        g2 = score_reply_grounding(
            data["comments"][4]["content"],
            data["comments"][2]["content"],
            parent_comment_id=3,
        )
        assert g2["invented_attribution"] is True


def test_fixture_ab541048_regression():
    if not LIVE_AB54.is_file():
        pytest.skip("ab541048 missing")
    data = json.loads(LIVE_AB54.read_text(encoding="utf-8"))
    assert classify_stance_v4(data["comments"][0]["content"]) == "conditional"
    assert classify_stance_v4(data["comments"][1]["content"]) == "conditional"


@pytest.mark.parametrize("key", sorted(OFFLINE_SCENARIOS.keys()))
def test_offline_scenarios_b_e_phase85(results_dir: Path, key: str):
    scenario, domain = OFFLINE_SCENARIOS[key]
    assert detect_topic_domain(scenario) == domain
    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid, k=key: ScriptedReactiveModel(
            agent_id=aid, label=f"p85-{k}"
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
    assert data["meta"]["quality_fix"] == "8.5"
    assert data["meta"]["persona_engine"] == "v5"
    assert data["agent_positions"]["stance_classifier"] == "v4"
    assert data["opinion_dynamics"]["engine"] == "v4"
    assert data["interaction_quality"]["engine"] == "v5"
    assert data["discussion"]["argument_grounded"]["engine"] == "v4"
    assert data["discussion"]["claim_based_interaction"]["engine"] == "v4"
    assert data["discussion"]["argument_intelligence"]["engine"] == "v4"
    assert "argument_evolution" in data["discussion"]["argument_intelligence"]
    cov = motivation_coverage(data["agents"])
    assert cov["coverage"] == 1.0
    assert cov["stake_coverage"] == 1.0
    assert all((a.get("motivation") or {}).get("stakes") for a in data["agents"])
    dist = data["agent_positions"]["polarity_distribution"]
    total = sum(int(v) for v in dist.values()) or 1
    assert int(dist.get("unknown") or 0) / total == 0.0
    grounded = evaluate_argument_grounded_interactions(comments=data["comments"])
    assert grounded["grounded_reply_rate"] == 1.0
    assert grounded["invented_attribution_count"] == 0
    claims = evaluate_claim_based_interactions(comments=data["comments"])
    assert claims["claim_address_rate"] >= 0.34
    assert float(data["interaction_quality"]["score"] or 0) >= 0.55
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
    assert data["simulation_intelligence"]["quality_fix"] == "8.5"
    assert data["simulation_intelligence"]["persona_engine"] == "v5"
    assert data["simulation_intelligence"]["kpis"]["claim_address_rate"] >= 0.0


@pytest.mark.parametrize("n", list(OFFLINE_SCALE_PRESETS))
def test_scripted_scale_10_25_50_phase85(results_dir: Path, n: int):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid: ScriptedReactiveModel(
            agent_id=aid, label=f"p85-{n}"
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
    assert data["meta"]["quality_fix"] == "8.5"
    assert data["meta"]["persona_engine"] == "v5"
    assert data["agent_positions"]["stance_classifier"] == "v4"
    assert data["opinion_dynamics"]["engine"] == "v4"
    assert data["interaction_quality"]["engine"] == "v5"
    assert data["discussion"]["claim_based_interaction"]["engine"] == "v4"
    assert data["discussion"]["argument_intelligence"]["engine"] == "v4"
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False
    assert data["simulation_intelligence"]["kpis"]["planned_cost_usd"] == 0.0
    assert all((a.get("motivation") or {}).get("stakes") for a in data["agents"])
    assert int((data.get("scheduler") or {}).get("waves_executed") or 0) >= wave_count_for(
        n
    )
    dist = data["agent_positions"]["polarity_distribution"]
    total = sum(int(v) for v in dist.values()) or 1
    assert int(dist.get("unknown") or 0) / total <= 0.35


def test_manual_n12_zero_cost_phase85(results_dir: Path):
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
    assert data["meta"]["quality_fix"] == "8.5"
    assert data["meta"]["persona_engine"] == "v5"
    assert data["opinion_dynamics"]["engine"] == "v4"
    assert data["interaction_quality"]["engine"] == "v5"
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False


def test_offline_presets_and_ceilings_unchanged():
    assert tuple(OFFLINE_SCALE_PRESETS) == (10, 25, 50)
    assert OFFLINE_MAX_AGENTS == 50
    assert math.isclose(LIVE_TEST_MAX_AGENTS, 5)


def test_fixtures_present_for_regression():
    assert LIVE_E251.is_file()
    assert LIVE_D1FB.is_file()
    assert LIVE_AB54.is_file()
