"""Phase 8.3 — Realistic Agent Behavior (offline only).

Live fixture ``20261028T142011Z-d1fb1b1c`` (KI-Lernassistenten):
- Thomas Bauer reply →Kommentar#3 invents «digitale Kluft» (parent = Evidenz/Piloten)
- Stance V2 left Thomas as unknown; soft Potenzial+Voraussetzungen missed
- No Persona Motivation V4 / Argument-Grounded V3 / Interaction Quality

Also regresses 8.1/8.2 fixtures. No provider calls. Ceilings unchanged. $0.
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
from backend.simulation.agent_generator import generate_dynamic_agents
from backend.simulation.discussion_engine import DiscussionEngine
from backend.simulation.engine import SimulationEngine
from backend.simulation.interaction_quality import (
    build_interaction_quality,
    evaluate_argument_grounded_interactions,
    score_reply_grounding,
)
from backend.simulation.opinion_dynamics import OpinionDynamics
from backend.simulation.persona_motivation import (
    PERSONA_ENGINE,
    build_persona_motivation,
    motivation_coverage,
)
from backend.simulation.simulation_intelligence import build_simulation_intelligence
from backend.simulation.social_dynamics import (
    PositionTracker,
    classify_stance_v3,
    _polarity,
)
from backend.simulation.social_scheduler import wave_count_for

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_D1FB = ROOT / "data" / "results" / "20261028T142011Z-d1fb1b1c.json"
LIVE_FIXTURE_AB54 = ROOT / "data" / "results" / "20261028T134318Z-ab541048.json"
LIVE_FIXTURE_3FE = ROOT / "data" / "results" / "20261028T140003Z-3fe364ce.json"

EDU_SCENARIO = (
    "Deutschland führt ab 2028 KI-Lernassistenten an allen öffentlichen Schulen ein. "
    "Simuliere eine Diskussion zwischen Lehrkräften, Eltern und Bildungsforschern über "
    "Lernleistungen, Datenschutz, Chancengleichheit und die Rolle der Lehrkräfte."
)

VIER = (
    "Deutschland führt ab 2030 eine Vier-Tage-Woche bei vollem Lohnausgleich ein. "
    "Simuliere die Diskussion zwischen Arbeitnehmern, Unternehmern und "
    "Wirtschaftsexperten."
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


def test_live_ceilings_unchanged_phase83():
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
    assert payload["society_intelligence"]["quality_fix"] in {"8.3", "8.4", "8.5"}
    assert payload["society_intelligence"]["persona_engine"] in {"v4", "v5"}
    assert payload["society_intelligence"]["persona_motivation_v4"] is True
    assert payload["society_intelligence"]["stance_classification_v3"] is True
    assert payload["society_intelligence"]["opinion_dynamics_v3"] is True
    assert payload["society_intelligence"]["argument_grounded_interaction_v3"] is True
    assert payload["society_intelligence"]["interaction_quality_evaluation"] is True
    assert payload["scalable_foundation"]["paid_llm_unlocked"] is False
    assert payload["scalable_foundation"]["quality_fix"] in {"8.3", "8.4", "8.5"}


def test_fixture_documents_thomas_bauer_gaps():
    """Stored live run proves pre-fix Thomas unknown + invented Antwortbezug."""
    assert LIVE_FIXTURE_D1FB.is_file()
    data = json.loads(LIVE_FIXTURE_D1FB.read_text(encoding="utf-8"))
    assert data["meta"]["run_id"] == "20261028T142011Z-d1fb1b1c"
    assert data["meta"]["quality_fix"] == "8.2"
    assert data["meta"]["persona_engine"] == "v3"
    dist = data["agent_positions"]["polarity_distribution"]
    assert dist.get("unknown", 0) >= 1
    thomas = next(a for a in data["agents"] if a["name"] == "Thomas Bauer")
    assert thomas["agent_id"] == 1
    assert thomas["role_label"] == "Elternstimme"
    # Latest stored polarity for Thomas is unknown.
    by_id = {p["agent_id"]: p for p in data["agent_positions"]["positions"]}
    assert by_id[1]["polarity"] == "unknown"
    # Reply invents digitale Kluft vs parent Evidenz.
    c3 = data["comments"][2]["content"]
    c5 = data["comments"][4]["content"]
    assert "digitalen Kluft" in c5 or "digitale Kluft" in c5
    assert "digitale Kluft" not in c3 and "digitalen Kluft" not in c3
    assert "empirische Evidenz" in c3 or "Pilot" in c3
    assert "interaction_quality" not in data
    assert (data.get("opinion_dynamics") or {}).get("engine") == "v2"


def test_stance_v3_thomas_bauer_conditional():
    data = json.loads(LIVE_FIXTURE_D1FB.read_text(encoding="utf-8"))
    c2 = data["comments"][1]["content"]
    c5 = data["comments"][4]["content"]
    assert classify_stance_v3(c2) == "conditional"
    assert classify_stance_v3(c5) == "conditional"
    assert _polarity(c2) == "conditional"
    # Soft Potenzial + Evidenzlücke → conditional (not hard pro).
    c1 = data["comments"][0]["content"]
    assert classify_stance_v3(c1) == "conditional"


def test_stance_v3_position_tracker_zero_unknown_on_d1fb():
    data = json.loads(LIVE_FIXTURE_D1FB.read_text(encoding="utf-8"))
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=data["agents"],
        posts=data["posts"],
        comments=data["comments"],
        reactions=data.get("reactions") or [],
        step=2,
    )
    payload = tracker.positions_payload()
    assert payload["stance_classifier"] in {"v3", "v4", "v5"}
    assert payload["quality_fix"] in {"8.3", "8.4", "8.5"}
    dist = payload["polarity_distribution"]
    assert dist.get("unknown", 0) == 0
    assert dist.get("conditional", 0) == 3
    by_id = {p["agent_id"]: p for p in payload["positions"]}
    assert by_id[1]["polarity"] == "conditional"  # Thomas Bauer


def test_stance_v3_keeps_ab541048_and_hard_labels():
    data = json.loads(LIVE_FIXTURE_AB54.read_text(encoding="utf-8"))
    assert classify_stance_v3(data["comments"][0]["content"]) == "conditional"
    assert classify_stance_v3(data["comments"][1]["content"]) == "conditional"
    assert _polarity("Ich lehne das Verbot ab.") == "contra"
    assert _polarity("Ich unterstütze das Verbot klar — sinnvolles Verbot.") == "pro"


def test_argument_grounded_v3_detects_thomas_invented_attribution():
    data = json.loads(LIVE_FIXTURE_D1FB.read_text(encoding="utf-8"))
    parent = data["comments"][2]["content"]
    reply = data["comments"][4]["content"]
    g = score_reply_grounding(reply, parent, parent_comment_id=3)
    assert g["engine"] in {"v3", "v4", "v5"}
    assert g["invented_attribution"] is True
    assert g["grounded"] is False
    # Fatima correctly addresses Evidenz/Lernfortschritte.
    g2 = score_reply_grounding(
        data["comments"][3]["content"], parent, parent_comment_id=3
    )
    assert g2["grounded"] is True
    assert g2["invented_attribution"] is False


def test_argument_grounded_snapshot_on_fixture():
    data = json.loads(LIVE_FIXTURE_D1FB.read_text(encoding="utf-8"))
    snap = evaluate_argument_grounded_interactions(comments=data["comments"])
    assert snap["engine"] in {"v3", "v4", "v5"}
    assert snap["quality_fix"] in {"8.3", "8.4", "8.5"}
    assert snap["reply_count"] == 3
    assert snap["invented_attribution_count"] >= 1
    assert 0.0 < snap["grounded_reply_rate"] < 1.0


def test_persona_motivation_v4_on_education_roles():
    profiles = generate_dynamic_agents(EDU_SCENARIO, 3)
    assert PERSONA_ENGINE in {"v4", "v5"}
    assert all(p.motivation for p in profiles)
    assert all("Motivation:" in p.goal for p in profiles)
    thomas_like = next(p for p in profiles if "Eltern" in (p.role_label or ""))
    mot = thomas_like.motivation or {}
    assert "Chancengleichheit" in (mot.get("values") or [])
    assert mot.get("decision_style")
    cov = motivation_coverage([p.to_dict() for p in profiles])
    assert cov["coverage"] == 1.0
    assert cov["engine"] in {"v4", "v5"}


def test_opinion_dynamics_v3_signals():
    od = OpinionDynamics()
    od.set_interaction_signals(grounded_reply_share=0.67, motivation_coverage=1.0)
    od.record_round(
        step=0,
        polarity_distribution={"unknown": 1, "conditional": 2},
        change_events=[],
        agent_count=3,
    )
    od.record_round(
        step=1,
        polarity_distribution={"conditional": 3},
        change_events=[],
        agent_count=3,
    )
    snap = od.snapshot()
    assert snap["engine"] in {"v3", "v4", "v5"}
    assert snap["version"] in {3, 4}
    assert snap["quality_fix"] in {"8.3", "8.4", "8.5"}
    assert snap["stance_clarity"] == 1.0
    assert snap["grounded_reply_share"] == 0.67
    assert snap["motivation_coverage"] == 1.0
    assert 0.0 <= snap["stability_index"] <= 1.0


def test_interaction_quality_evaluation_on_fixture():
    data = json.loads(LIVE_FIXTURE_D1FB.read_text(encoding="utf-8"))
    agents = generate_dynamic_agents(EDU_SCENARIO, 3)
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=data["agents"],
        posts=data["posts"],
        comments=data["comments"],
        reactions=[],
        step=2,
    )
    eng = DiscussionEngine()
    eng.ingest_platform(
        posts=data["posts"], comments=data["comments"], step=2, round_n=2
    )
    disc = eng.snapshot(
        polarity_distribution=tracker.polarity_distribution(), agent_count=3
    )
    iq = build_interaction_quality(
        comments=data["comments"],
        agents=[p.to_dict() for p in agents],
        stance_clarity=1.0,
        debate_quality_score=float((disc.get("debate_quality") or {}).get("score") or 0),
        opinion_stability=1.0,
    )
    assert iq["engine"] in {"v3", "v4", "v5"}
    assert iq["quality_fix"] in {"8.3", "8.4", "8.5"}
    assert iq["score"] >= 0.45
    assert iq["components"]["persona_motivation_coverage"] == 1.0
    assert iq["argument_grounded"]["invented_attribution_count"] >= 1


def test_adversarial_ungrounded_reply_lowers_quality():
    parent = (
        "Als Bildungsforscherin sehe ich fehlende empirische Evidenz und plädiere "
        "für begleitete Pilotprojekte statt anekdotischer Erfolgsversprechen."
    )
    bad = (
        "→Kommentar#1 Ich stimme Ihnen zu, dass die Herausforderung der digitalen "
        "Kluft nicht unterschätzt werden darf und Förderprogramme nötig sind."
    )
    good = (
        "→Kommentar#1 Ihre Sorge um fehlende empirische Evidenz teile ich. "
        "Deshalb fordere ich begleitete Pilotprojekte vor der flächendeckenden Einführung."
    )
    assert score_reply_grounding(bad, parent, parent_comment_id=1)["grounded"] is False
    assert score_reply_grounding(good, parent, parent_comment_id=1)["grounded"] is True
    iq_bad = build_interaction_quality(
        comments=[
            {"comment_id": 1, "user_id": 0, "content": parent},
            {"comment_id": 2, "user_id": 1, "content": bad},
        ],
        agents=[{"agent_id": 0, "goal": "x", "motivation": {"values": ["a"]}}],
        stance_clarity=0.3,
        debate_quality_score=0.2,
        opinion_stability=0.2,
    )
    assert iq_bad["score"] < 0.45 or iq_bad["ok"] is False


def test_simulation_intelligence_phase83_flags():
    agents = [p.to_dict() for p in generate_dynamic_agents(EDU_SCENARIO, 3)]
    intel = build_simulation_intelligence(
        agents=agents,
        execution_kind="scripted",
        agent_count=3,
        max_steps=2,
        step=2,
        cost_plan={
            "planned_cost_usd": 0.0,
            "paid_llm_unlocked": False,
            "within_live_ceilings": True,
            "planned_waves": 1,
        },
        scheduler={"waves_executed": 1, "waves_planned": 1, "self_as_partner_count": 0},
        society_analytics={"kpis": {}},
        discussion={
            "argument_count": 6,
            "repetition_blocks": 0,
            "cross_agent_replies": 3,
            "self_replies": 0,
            "quality_fix": "8.3",
            "debate_engine": "v2",
            "argument_intelligence": {
                "engine": "v2",
                "mean_quality_score": 0.7,
                "argument_diversity": 0.6,
                "claim_kinds": {"conditional_claim": 3},
            },
            "debate_quality": {"score": 0.72, "ok": True, "engine": "debate_v2"},
            "argument_grounded": {
                "engine": "v3",
                "grounded_reply_rate": 0.67,
                "invented_attribution_rate": 0.33,
            },
        },
        agent_positions={
            "polarity_distribution": {"conditional": 3},
            "change_count": 0,
            "stance_classifier": "v3",
            "quality_fix": "8.3",
        },
        opinion_dynamics={
            "engine": "v3",
            "version": 3,
            "stance_clarity": 1.0,
            "stability_index": 0.9,
            "polarization": 0.2,
            "shift_rate_per_agent_round": 0.0,
            "rounds_tracked": 3,
            "motivation_coverage": 1.0,
            "grounded_reply_share": 0.67,
        },
        interaction_quality={
            "engine": "v3",
            "score": 0.62,
            "quality_fix": "8.3",
            "components": {"persona_motivation_coverage": 1.0},
            "argument_grounded": {
                "engine": "v3",
                "grounded_reply_rate": 0.67,
                "invented_attribution_rate": 0.33,
            },
        },
    )
    assert intel["quality_fix"] in {"8.3", "8.4", "8.5"}
    assert intel["persona_engine"] in {"v4", "v5"}
    assert intel["stance_classifier"] in {"v3", "v4", "v5"}
    assert intel["opinion_engine"] in {"v3", "v4", "v5"}
    assert intel["kpis"]["interaction_quality_score"] >= 0.6
    assert intel["kpis"]["grounded_reply_rate"] == 0.67
    assert intel["paid_llm_unlocked"] is False


@pytest.mark.parametrize("n", list(OFFLINE_SCALE_PRESETS))
def test_scripted_scale_10_25_50_realistic_agents(results_dir: Path, n: int):
    from backend.simulation.scripted_model import ScriptedReactiveModel

    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid: ScriptedReactiveModel(
            agent_id=aid, label=f"p83-{n}"
        ),
    )
    started = engine.start(
        scenario=EDU_SCENARIO if n <= 25 else VIER,
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
    path = st.get("result_path")
    assert path and Path(path).is_file()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] in {"8.3", "8.4", "8.5"}
    assert data["meta"]["persona_engine"] in {"v4", "v5"}
    assert data["meta"]["society_intelligence"]["paid_llm_unlocked"] is False
    assert data["agent_positions"]["stance_classifier"] in {"v3", "v4", "v5"}
    assert data["opinion_dynamics"]["engine"] in {"v3", "v4", "v5"}
    assert data["discussion"]["debate_engine"] == "v2"
    assert "interaction_quality" in data
    assert data["interaction_quality"]["engine"] in {"v3", "v4", "v5"}
    assert data["simulation_intelligence"]["persona_engine"] in {"v4", "v5"}
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False
    assert data["simulation_intelligence"]["kpis"]["planned_cost_usd"] == 0.0
    # Motivations attached.
    assert all((a.get("motivation") or {}).get("values") for a in data["agents"])
    expected = wave_count_for(n)
    assert int((data.get("scheduler") or {}).get("waves_executed") or 0) >= expected
    dist = data["agent_positions"]["polarity_distribution"]
    total = sum(int(v) for v in dist.values()) or 1
    unk = int(dist.get("unknown") or 0)
    assert unk / total <= 0.35


def test_manual_n12_zero_cost_phase83_fields(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=EDU_SCENARIO,
        mode="manual",
        max_steps=2,
        agent_count=12,
        agent_generation="dynamic",
        language="de",
    )
    assert started.get("ok") is True
    st = _wait_terminal(engine)
    assert st["status"] in {"completed", "limit_reached"}
    assert st.get("execution_kind") == "manual"
    cost = st.get("cost_summary") or {}
    assert int(cost.get("api_calls") or 0) == 0
    assert float(cost.get("estimated_cost_usd") or 0.0) == 0.0
    path = st.get("result_path")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] in {"8.3", "8.4", "8.5"}
    assert data["meta"]["persona_engine"] in {"v4", "v5"}
    assert data["opinion_dynamics"]["engine"] in {"v3", "v4", "v5"}
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False


def test_offline_presets_and_agent_cap_unchanged():
    assert tuple(OFFLINE_SCALE_PRESETS) == (10, 25, 50)
    assert OFFLINE_MAX_AGENTS == 50
    assert math.isclose(LIVE_TEST_MAX_AGENTS, 5)


def test_phase81_82_fixtures_still_readable():
    """Prior fixtures remain on disk for regression analysis (no rewrite)."""
    assert LIVE_FIXTURE_AB54.is_file()
    assert LIVE_FIXTURE_D1FB.is_file()
    if LIVE_FIXTURE_3FE.is_file():
        data = json.loads(LIVE_FIXTURE_3FE.read_text(encoding="utf-8"))
        assert data["meta"]["run_id"].endswith("3fe364ce")


def test_persona_motivation_prompt_block():
    mot = build_persona_motivation(
        role_label="Lehrkraft",
        occupation="Grundschullehrerin",
        goal="Zielkonflikt: nur mit Evidenz.",
        scenario=EDU_SCENARIO,
        style="pädagogisch, evidenzbezogen",
    )
    block = mot.prompt_block()
    assert "Motivation:" in block
    assert "evidence_first" in block or "Evidenz" in ",".join(mot.drivers)
