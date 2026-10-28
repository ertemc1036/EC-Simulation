"""Phase 8.3.1 — Multi-scenario quality audit (offline, no live LLM).

Measures Stance V3 / Grounding V3 / IQ / Motivation across scenarios A–E.
Documents findings for Phase 8.4; does not ship classifier fixes.

Fixture: ``data/results/20261028T145656Z-e251172c.json`` (Phase 8.3 live edu).
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
    load_settings,
)
from backend.simulation.engine import SimulationEngine
from backend.simulation.interaction_quality import (
    evaluate_argument_grounded_interactions,
    score_reply_grounding,
)
from backend.simulation.persona_motivation import motivation_coverage
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import (
    PositionTracker,
    classify_stance_v3,
    _polarity,
)
from backend.simulation.topic_context import detect_topic_domain

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE = ROOT / "data" / "results" / "20261028T145656Z-e251172c.json"
D1FB_FIXTURE = ROOT / "data" / "results" / "20261028T142011Z-d1fb1b1c.json"

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


def _wait_terminal(engine: SimulationEngine, *, deadline_s: float = 420.0) -> dict:
    end = time.time() + deadline_s
    while time.time() < end:
        st = engine.get_state()
        if st.get("status") in {"completed", "limit_reached", "failed", "blocked"}:
            return st
        time.sleep(0.05)
    raise AssertionError("simulation did not reach terminal status")


@pytest.fixture()
def fixture_a() -> dict:
    assert LIVE_FIXTURE.is_file(), f"missing fixture {LIVE_FIXTURE}"
    return json.loads(LIVE_FIXTURE.read_text(encoding="utf-8"))


def test_ceilings_and_paid_flag_unchanged_phase831():
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
    assert payload["society_intelligence"]["quality_fix"] == "8.3"


def test_scenario_a_fixture_meta_and_engines(fixture_a: dict):
    meta = fixture_a["meta"]
    assert meta["run_id"] == "20261028T145656Z-e251172c"
    assert meta["quality_fix"] == "8.3"
    assert meta["persona_engine"] == "v4"
    assert detect_topic_domain(meta["scenario"]) == "education"
    assert fixture_a["agent_positions"]["stance_classifier"] == "v3"
    assert fixture_a["opinion_dynamics"]["engine"] == "v3"
    assert fixture_a["interaction_quality"]["engine"] == "v3"
    assert fixture_a["discussion"]["argument_grounded"]["engine"] == "v3"
    # Historical live cost only — this audit must not add provider spend.
    assert int(fixture_a["cost_summary"]["api_calls"]) == 6


def test_scenario_a_documents_false_pro_stance_gaps(fixture_a: dict):
    """Audit finding: Thomas C2 + Fatima C4 are semantically conditional, V3 → pro."""
    c2 = fixture_a["comments"][1]["content"]
    c4 = fixture_a["comments"][3]["content"]
    assert "großes Potenzial" in c2 or "grosses Potenzial" in c2
    assert "Datenschutz" in c2 or "datenschutz" in c2.lower()
    assert "schrittweise" in c2.lower() or "Eltern" in c2
    assert classify_stance_v3(c2) == "pro"  # documented false pro (8.4)
    assert "plädiere ich dafür" in c4
    assert "nicht einfach technikgetrieben" in c4
    assert classify_stance_v3(c4) == "pro"  # dafür false positive (8.4)
    # C1 / C5 remain correctly conditional.
    assert classify_stance_v3(fixture_a["comments"][0]["content"]) == "conditional"
    assert classify_stance_v3(fixture_a["comments"][4]["content"]) == "conditional"
    by_id = {p["agent_id"]: p for p in fixture_a["agent_positions"]["positions"]}
    assert by_id[0]["polarity"] == "pro"  # Fatima final = C4
    assert by_id[1]["polarity"] == "conditional"
    assert by_id[2]["polarity"] == "conditional"
    assert fixture_a["agent_positions"]["polarity_distribution"].get("unknown", 0) == 0


def test_scenario_a_grounding_perfect_invented_zero(fixture_a: dict):
    snap = evaluate_argument_grounded_interactions(comments=fixture_a["comments"])
    assert snap["engine"] == "v3"
    assert snap["reply_count"] == 3
    assert snap["grounded_reply_rate"] == 1.0
    assert snap["invented_attribution_count"] == 0
    assert fixture_a["interaction_quality"]["score"] >= 0.8
    # Weak overlap on Hannah C6 is documented (still grounded).
    c6_eval = next(e for e in snap["evaluations"] if e["comment_id"] == 6)
    assert c6_eval["grounded"] is True
    assert c6_eval["overlap"] < 0.20


def test_scenario_a_motivation_and_replay_unknown_zero(fixture_a: dict):
    cov = motivation_coverage(fixture_a["agents"])
    assert cov["coverage"] == 1.0
    assert cov["engine"] == "v4"
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=fixture_a["agents"],
        posts=fixture_a["posts"],
        comments=fixture_a["comments"],
        reactions=fixture_a.get("reactions") or [],
        step=2,
    )
    dist = tracker.positions_payload()["polarity_distribution"]
    assert dist.get("unknown", 0) == 0
    assert dist.get("pro", 0) == 1
    assert dist.get("conditional", 0) == 2


def test_stance_dafuer_false_positive_case():
    """Bare «dafür» in «plädiere ich dafür, dass … nicht» must be visible to 8.4."""
    text = (
        "Deshalb plädiere ich dafür, dass der Einsatz von KI-Lernassistenten "
        "nicht einfach technikgetrieben erfolgen darf. Es braucht klare, "
        "pädagogisch fundierte Konzepte."
    )
    assert "dafür" in text.lower()
    assert _polarity(text) == "pro"


def test_stance_hard_labels_still_work():
    assert classify_stance_v3("Ich unterstütze das Verbot klar — sinnvolles Verbot.") == "pro"
    assert classify_stance_v3("Ich lehne das Verbot ab — dagegen.") == "contra"


def test_grounding_adversarial_digitale_kluft_still_detected():
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
    if D1FB_FIXTURE.is_file():
        data = json.loads(D1FB_FIXTURE.read_text(encoding="utf-8"))
        parent_live = data["comments"][2]["content"]
        reply_live = data["comments"][4]["content"]
        g2 = score_reply_grounding(reply_live, parent_live, parent_comment_id=3)
        assert g2["invented_attribution"] is True


@pytest.mark.parametrize("key", sorted(OFFLINE_SCENARIOS.keys()))
def test_offline_scenarios_b_e_scripted(results_dir: Path, key: str):
    scenario, domain = OFFLINE_SCENARIOS[key]
    assert detect_topic_domain(scenario) == domain
    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid, k=key: ScriptedReactiveModel(
            agent_id=aid, label=f"p831-{k}"
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
    st = _wait_terminal(engine, deadline_s=600.0)
    assert st["status"] in {"completed", "limit_reached"}
    assert st.get("execution_kind") == "scripted"
    plan = st.get("cost_plan") or {}
    assert plan.get("paid_llm_unlocked") is False
    assert float(plan.get("planned_cost_usd") or 0.0) == 0.0
    path = st.get("result_path")
    assert path and Path(path).is_file()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] == "8.3"
    assert data["meta"]["persona_engine"] == "v4"
    assert data["agent_positions"]["stance_classifier"] == "v3"
    assert data["opinion_dynamics"]["engine"] == "v3"
    assert data["interaction_quality"]["engine"] == "v3"
    assert motivation_coverage(data["agents"])["coverage"] == 1.0
    dist = data["agent_positions"]["polarity_distribution"]
    total = sum(int(v) for v in dist.values()) or 1
    assert int(dist.get("unknown") or 0) / total == 0.0
    grounded = evaluate_argument_grounded_interactions(comments=data["comments"])
    assert grounded["grounded_reply_rate"] == 1.0
    assert grounded["invented_attribution_count"] == 0
    assert float(data["interaction_quality"]["score"] or 0) >= 0.65
    # Topic integrity: no cross-domain measure leaks in posts/comments.
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


def test_audit_doc_present():
    doc = ROOT / "docs" / "phase8.3.1-quality-audit.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "PHASE 8.3.1 QUALITY AUDIT COMPLETE" in text
    assert "e251172c" in text
    assert "8.4" in text
