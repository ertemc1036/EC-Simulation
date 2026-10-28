"""Phase 8.2 — Simulation Intelligence & Debate Quality (offline only).

Live fixture ``20261028T134318Z-ab541048`` (KI-Lernassistenten):
- Stance mislabels (Lehrerin Chancen+flächendeckend ablehnen → contra;
  Eltern «nur unter Voraussetzungen» → unknown)
- No Argument Intelligence / Debate Quality scores
- Opinion dynamics without clarity/stability

No provider calls. Ceilings unchanged. $0.
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
    build_argument_intelligence,
    classify_claim_kind,
    debate_quality_score,
    score_argument,
)
from backend.simulation.discussion_engine import DiscussionEngine
from backend.simulation.engine import SimulationEngine
from backend.simulation.opinion_dynamics import OpinionDynamics
from backend.simulation.simulation_intelligence import build_simulation_intelligence
from backend.simulation.social_dynamics import (
    PositionTracker,
    classify_stance_v2,
    _polarity,
)
from backend.simulation.social_scheduler import wave_count_for

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_AB54 = ROOT / "data" / "results" / "20261028T134318Z-ab541048.json"

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


def test_live_ceilings_unchanged_phase82():
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
    assert payload["society_intelligence"]["quality_fix"] in {"8.2", "8.3", "8.4"}
    assert payload["society_intelligence"]["argument_intelligence_v2"] is True
    assert payload["society_intelligence"]["debate_engine_v2"] is True
    assert payload["society_intelligence"]["stance_classification_v2"] is True
    assert payload["society_intelligence"]["opinion_dynamics_v2"] is True
    assert payload["society_intelligence"]["quality_dashboard"] is True
    assert payload["scalable_foundation"]["paid_llm_unlocked"] is False
    assert payload["scalable_foundation"]["quality_fix"] in {"8.2", "8.3", "8.4"}


def test_fixture_documents_stance_and_debate_gaps():
    """Stored live run proves pre-fix stance mislabels + missing V2 fields."""
    assert LIVE_FIXTURE_AB54.is_file()
    data = json.loads(LIVE_FIXTURE_AB54.read_text(encoding="utf-8"))
    assert data["meta"]["run_id"] == "20261028T134318Z-ab541048"
    dist = data["agent_positions"]["polarity_distribution"]
    # Pre-fix: one unknown + mislabeled contra on soft-reject comment.
    assert dist.get("unknown", 0) >= 1
    assert dist.get("contra", 0) >= 1
    assert "argument_intelligence" not in (data.get("discussion") or {})
    assert "debate_quality" not in (data.get("discussion") or {})
    assert (data.get("opinion_dynamics") or {}).get("engine") is None


def test_stance_v2_fixture_comments_clear_conditional():
    data = json.loads(LIVE_FIXTURE_AB54.read_text(encoding="utf-8"))
    c1 = data["comments"][0]["content"]
    c2 = data["comments"][1]["content"]
    # Soft-reject flächendeckend + Chancen → conditional (not hard contra).
    assert classify_stance_v2(c1) == "conditional"
    assert _polarity(c1) == "conditional"
    # «nur unter klaren Voraussetzungen» → conditional (not unknown).
    assert classify_stance_v2(c2) == "conditional"


def test_stance_v2_position_tracker_zero_unknown_on_fixture():
    data = json.loads(LIVE_FIXTURE_AB54.read_text(encoding="utf-8"))
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=data["agents"],
        posts=data["posts"],
        comments=data["comments"],
        reactions=data.get("reactions") or [],
        step=2,
    )
    payload = tracker.positions_payload()
    assert payload["stance_classifier"] in {"v2", "v3", "v4"}
    assert payload["quality_fix"] in {"8.2", "8.3", "8.4"}
    dist = payload["polarity_distribution"]
    assert dist.get("unknown", 0) == 0
    assert all(
        p["polarity"] in {"pro", "contra", "conditional", "exception"}
        for p in payload["positions"]
    )
    by_id = {p["agent_id"]: p for p in payload["positions"]}
    assert by_id[0]["polarity"] == "conditional"
    assert by_id[1]["polarity"] == "conditional"
    assert by_id[2]["polarity"] == "conditional"


def test_stance_v2_keeps_hard_contra_and_clear_pro():
    assert _polarity("Ich lehne das Verbot ab.") == "contra"
    assert _polarity("Ich unterstütze das Verbot klar — sinnvolles Verbot.") == "pro"
    assert (
        _polarity(
            "Ohne klare Evidenz lehne ich eine flächendeckende Einführung ab, "
            "obwohl ich Chancen sehe."
        )
        == "conditional"
    )


def test_argument_intelligence_v2_on_fixture():
    data = json.loads(LIVE_FIXTURE_AB54.read_text(encoding="utf-8"))
    eng = DiscussionEngine()
    eng.ingest_platform(
        posts=data["posts"],
        comments=data["comments"],
        step=2,
        round_n=2,
    )
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=data["agents"],
        posts=data["posts"],
        comments=data["comments"],
        reactions=[],
        step=2,
    )
    pol = tracker.polarity_distribution()
    snap = eng.snapshot(polarity_distribution=pol, agent_count=3)
    assert snap["debate_engine"] == "v2"
    assert snap["quality_fix"] in {"8.2", "8.3", "8.4"}
    intel = snap["argument_intelligence"]
    assert intel["engine"] == "v2"
    assert intel["argument_count"] >= 6
    assert intel["mean_quality_score"] >= 0.5
    assert intel["unique_claim_kinds"] >= 2
    dq = snap["debate_quality"]
    assert dq["score"] >= 0.45
    assert dq["ok"] is True
    assert "conditional_claim" in intel["claim_kinds"] or "evidence_demand" in intel[
        "claim_kinds"
    ]


def test_claim_kind_and_scores_adversarial():
    assert classify_claim_kind("Ich stimme zu.") in {"acknowledgment", "statement"}
    assert (
        classify_claim_kind(
            "→Kommentar#2 Allerdings dürfen wir Datenschutz nicht vergessen."
        )
        == "rebuttal"
    )
    assert (
        classify_claim_kind(
            "Potenzial ja, allerdings nur unter klaren Voraussetzungen und Evidenz."
        )
        in {"conditional_claim", "evidence_demand", "caveat"}
    )
    good = score_argument(
        "Als Lehrkraft sehe ich Chancen, aber nur mit Pilot und Evidenz.",
        addresses_foreign=False,
    )
    bad = score_argument("ok", is_repetition=True)
    assert good > bad
    assert good >= 0.5


def test_opinion_dynamics_v2_clarity_and_stability():
    od = OpinionDynamics()
    od.record_round(
        step=0,
        polarity_distribution={"unknown": 2, "pro": 1},
        change_events=[],
        agent_count=3,
    )
    od.record_round(
        step=1,
        polarity_distribution={"conditional": 2, "pro": 1},
        change_events=[],
        agent_count=3,
    )
    od.record_round(
        step=2,
        polarity_distribution={"conditional": 3},
        change_events=[],
        agent_count=3,
    )
    snap = od.snapshot()
    assert snap["engine"] in {"v2", "v3", "v4"}
    assert snap["version"] in {2, 3, 4}
    assert snap["quality_fix"] in {"8.2", "8.3", "8.4"}
    assert snap["stance_clarity"] == 1.0
    assert snap["unknown_share"] == 0.0
    assert 0.0 <= snap["stability_index"] <= 1.0
    assert len(snap["unknown_share_by_round"]) == 3


def test_simulation_intelligence_includes_debate_kpis():
    intel = build_simulation_intelligence(
        agents=[
            {
                "agent_id": 0,
                "name": "A One",
                "occupation": "Grundschullehrerin",
                "role_label": "Lehrkraft",
            },
            {
                "agent_id": 1,
                "name": "B Two",
                "occupation": "Elternteil",
                "role_label": "Elternstimme",
            },
            {
                "agent_id": 2,
                "name": "C Three",
                "occupation": "Bildungsforscher",
                "role_label": "Bildungsforscher",
            },
        ],
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
            "quality_fix": "8.2",
            "debate_engine": "v2",
            "argument_intelligence": {
                "engine": "v2",
                "mean_quality_score": 0.7,
                "argument_diversity": 0.6,
                "claim_kinds": {"conditional_claim": 3},
            },
            "debate_quality": {"score": 0.72, "ok": True, "engine": "debate_v2"},
        },
        agent_positions={
            "polarity_distribution": {"conditional": 3},
            "change_count": 0,
            "stance_classifier": "v2",
            "quality_fix": "8.2",
        },
        opinion_dynamics={
            "engine": "v2",
            "version": 2,
            "stance_clarity": 1.0,
            "stability_index": 0.9,
            "polarization": 0.2,
            "shift_rate_per_agent_round": 0.0,
            "rounds_tracked": 3,
        },
    )
    assert intel["quality_fix"] in {"8.2", "8.3", "8.4"}
    assert intel["debate_engine"] == "v2"
    assert intel["stance_classifier"] in {"v2", "v3", "v4"}
    assert intel["kpis"]["debate_quality_score"] >= 0.7
    assert intel["kpis"]["stance_clarity"] == 1.0
    assert intel["paid_llm_unlocked"] is False


@pytest.mark.parametrize("n", list(OFFLINE_SCALE_PRESETS))
def test_scripted_scale_10_25_50_debate_quality(results_dir: Path, n: int):
    from backend.simulation.scripted_model import ScriptedReactiveModel

    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid: ScriptedReactiveModel(
            agent_id=aid, label=f"p82-{n}"
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
    assert data["meta"]["quality_fix"] in {"8.2", "8.3", "8.4"}
    assert data["meta"]["society_intelligence"]["paid_llm_unlocked"] is False
    disc = data["discussion"]
    assert disc["debate_engine"] == "v2"
    assert disc["argument_intelligence"]["engine"] == "v2"
    assert disc["debate_quality"]["score"] >= 0.0
    assert data["opinion_dynamics"]["engine"] in {"v2", "v3", "v4"}
    assert data["agent_positions"]["stance_classifier"] in {"v2", "v3", "v4"}
    intel = data["simulation_intelligence"]
    assert intel["quality_fix"] in {"8.2", "8.3", "8.4"}
    assert intel["paid_llm_unlocked"] is False
    assert intel["kpis"]["planned_cost_usd"] == 0.0
    # Scheduler waves still in execution (8.1 invariant).
    expected = wave_count_for(n)
    assert int((data.get("scheduler") or {}).get("waves_executed") or 0) >= expected
    # Stance clarity should beat the historical fixture unknown-share.
    dist = data["agent_positions"]["polarity_distribution"]
    total = sum(int(v) for v in dist.values()) or 1
    unk = int(dist.get("unknown") or 0)
    assert unk / total <= 0.35


def test_manual_n12_zero_cost_debate_fields(results_dir: Path):
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
    assert data["meta"]["quality_fix"] in {"8.2", "8.3", "8.4"}
    assert data["discussion"]["debate_engine"] == "v2"
    assert data["simulation_intelligence"]["paid_llm_unlocked"] is False


def test_adversarial_repetition_lowers_debate_score():
    args = [
        {
            "agent_id": 0,
            "fingerprint": "a",
            "snippet": "Gleiche Aussage nochmal identisch wiederholt ohne neuen Punkt.",
            "evidence": "comment:1",
        },
        {
            "agent_id": 0,
            "fingerprint": "a",
            "snippet": "Gleiche Aussage nochmal identisch wiederholt ohne neuen Punkt.",
            "evidence": "comment:2",
        },
    ]
    intel = build_argument_intelligence(
        arguments=args,
        threads=[],
        repetition_blocks=4,
        agent_count=1,
    )
    dq = debate_quality_score(
        argument_intel=intel,
        unknown_stance_share=0.8,
        polarity_distribution={"unknown": 3},
    )
    assert dq["score"] < 0.45
    assert dq["ok"] is False


def test_offline_presets_and_agent_cap_unchanged():
    assert tuple(OFFLINE_SCALE_PRESETS) == (10, 25, 50)
    assert OFFLINE_MAX_AGENTS == 50
    assert math.isclose(LIVE_TEST_MAX_AGENTS, 5)
