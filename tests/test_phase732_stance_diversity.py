"""Phase 7.3.2 — Persona Diversity & Stance Reliability (offline only).

Live fixture ``20261028T115954Z-c16c749b`` (Vier-Tage-Woche):
- Betriebsratsvorsitzender labeled as Unternehmensvertretung
- Moderator filler instead of Wirtschaftsexperte (Arbeitsmarkt cue)
- All 3 agent positions ``unknown`` after agreement replies

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
    load_settings,
)
from backend.simulation.agent_generator import (
    extract_stakeholder_roles,
    generate_dynamic_agents,
)
from backend.simulation.discussion_engine import DiscussionEngine
from backend.simulation.engine import SimulationEngine
from backend.simulation.social_dynamics import (
    PositionTracker,
    _is_agreement_reply,
    _polarity,
)
from backend.simulation.topic_context import (
    build_public_seed_content,
    contains_foreign_topic_leak,
    contains_public_feed_leak,
    detect_topic_domain,
    measure_label,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_732 = ROOT / "data" / "results" / "20261028T115954Z-c16c749b.json"
LIVE_FIXTURE_731 = ROOT / "data" / "results" / "20261028T112739Z-bdfb9e9d.json"

# Live scenario that produced the role/stance failures.
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
    ("schulreform", SCHULREFORM, "education", ("schulreform",)),
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
        state = engine.get_state()
        if state["status"] in {"completed", "limit_reached", "failed"}:
            if state.get("result_path") or state["status"] == "failed":
                return state
        time.sleep(0.15)
    return engine.get_state()


def test_live_ceilings_unchanged_phase732():
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    cfg = load_settings()
    assert cfg.max_cost_usd_per_run == 0.20
    payload = cfg.status_payload()
    assert payload["llm_run_requires_approval"] is True
    assert payload["offline_scale"]["paid_llm_unlocked"] is False
    assert payload["society_intelligence"]["quality_fix"] in {"7.3.2", "8.0", "8.1", "8.2", "8.3", "8.4"}


def test_fixture_documents_role_and_stance_failures():
    """Stored live run proves pre-fix persona + all-unknown stance bugs."""
    assert LIVE_FIXTURE_732.is_file()
    data = json.loads(LIVE_FIXTURE_732.read_text(encoding="utf-8"))
    assert data["meta"]["run_id"] == "20261028T115954Z-c16c749b"
    agents = {a["agent_id"]: a for a in data["agents"]}
    assert agents[1]["occupation"] == "Betriebsratsvorsitzender"
    assert agents[1]["role_label"] == "Unternehmensvertretung"
    assert agents[2]["occupation"] == "Moderator"
    assert agents[2]["role_label"] == "Moderator"
    assert data["agent_positions"]["polarity_distribution"] == {"unknown": 3}


def test_live_scenario_roles_include_wirtschaftsexperte_not_moderator():
    roles = extract_stakeholder_roles(LIVE_SCENARIO_732, limit=5)
    assert "Arbeitnehmer" in roles
    assert "Unternehmen" in roles
    assert "Ökonomen" in roles  # Arbeitsmarkt → Ökonomen / Wirtschaftsexperte
    agents = generate_dynamic_agents(LIVE_SCENARIO_732, 3)
    labels = [a.role_label for a in agents]
    occupations = [a.occupation for a in agents]
    assert "Moderator" not in labels
    assert "Moderator" not in occupations
    assert any("Wirtschaftsexperte" in (l or "") or "Ökonom" in (l or "") for l in labels)
    # Company seat must not be works-council occupation.
    company = next(a for a in agents if "Unternehmen" in (a.role_label or ""))
    assert "Betriebsrat" not in (company.occupation or "")
    assert company.occupation in {
        "Betriebsleiter",
        "Personalchefin",
        "Logistikleiter",
    }


def test_explicit_wirtschaftsexperten_still_maps():
    roles = extract_stakeholder_roles(VIER, limit=5)
    assert roles[:3] == ["Arbeitnehmer", "Unternehmer", "Ökonomen"]
    agents = generate_dynamic_agents(VIER, 3)
    assert agents[2].role_label == "Wirtschaftsexperte"
    assert "Betriebsrat" not in agents[1].occupation


def test_persona_diversity_across_three_labor_seats():
    agents = generate_dynamic_agents(LIVE_SCENARIO_732, 3)
    families = set()
    for a in agents:
        low = f"{a.role_label} {a.occupation}".lower()
        if "arbeitnehmer" in low or "industrie" in low or "pflege" in low:
            families.add("worker")
        elif "unternehmen" in low or "betriebsleiter" in low or "personal" in low:
            families.add("company")
        elif "wirtschaft" in low or "ökonom" in low or "finanz" in low or "arbeitsmarkt" in low:
            families.add("economist")
        else:
            families.add(low)
    assert families >= {"worker", "company", "economist"}
    names = [a.name for a in agents]
    assert len(names) == len(set(names))


def test_fixture_comments_have_clear_polarity_after_fix():
    data = json.loads(LIVE_FIXTURE_732.read_text(encoding="utf-8"))
    pols = [_polarity(c["content"]) for c in data["comments"]]
    # First three (Erststatements) must not be unknown.
    assert pols[0] in {"pro", "conditional"}
    assert pols[1] in {"conditional", "contra"}
    assert pols[2] in {"uncertain", "conditional"}
    clear = sum(1 for p in pols if p in {"pro", "contra", "conditional", "exception"})
    assert clear >= 4


def test_agreement_reply_is_zustimmung_not_own_wipe():
    ack = (
        "→Kommentar#2: Deine Sorgen als Betriebsratsvorsitzender sind nachvollziehbar. "
        "Gerade in Branchen mit Schichtarbeit ist eine starre Woche schwer."
    )
    assert _is_agreement_reply(ack) is True
    # Sparse ack alone → unknown; with Lohnausgleich caveat → conditional.
    assert _polarity(ack) in {"unknown", "uncertain", "conditional"}


def test_position_tracker_keeps_own_stance_across_agreement():
    data = json.loads(LIVE_FIXTURE_732.read_text(encoding="utf-8"))
    tracker = PositionTracker()
    tracker.update_from_snapshot(
        agents=data["agents"],
        posts=data["posts"],
        comments=data["comments"],
        reactions=data.get("reactions") or [],
        step=2,
    )
    payload = tracker.positions_payload()
    dist = payload["polarity_distribution"]
    assert dist.get("unknown", 0) == 0
    assert dist.get("conditional", 0) >= 2
    assert payload["change_count"] == 0
    by_id = {p["agent_id"]: p for p in payload["positions"]}
    assert by_id[0]["polarity"] in {"pro", "conditional"}
    assert by_id[1]["polarity"] == "conditional"
    assert by_id[2]["polarity"] in {"uncertain", "conditional"}
    # No evidenced flip from agreement replies.
    for p in payload["positions"]:
        assert p["changed"] is False


def test_evidenced_change_still_recorded_on_explicit_shift():
    tracker = PositionTracker()
    agents = [
        {"agent_id": 0, "goal": "Ich unterstütze das Verbot klar."},
        {"agent_id": 1, "goal": "Ich lehne das Verbot ab."},
    ]
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {
                "post_id": 1,
                "user_id": 0,
                "content": "Ich unterstütze das Verbot klar — sinnvolles Verbot.",
            }
        ],
        comments=[],
        reactions=[],
        step=1,
    )
    changes = tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {
                "post_id": 1,
                "user_id": 0,
                "content": "Ich unterstütze das Verbot klar — sinnvolles Verbot.",
            }
        ],
        comments=[
            {
                "comment_id": 1,
                "post_id": 1,
                "user_id": 0,
                "content": (
                    "→Kommentar#1: Nach den neuen Daten ändere ich meine Position — "
                    "ich lehne das Verbot ab und bin dagegen."
                ),
            }
        ],
        reactions=[],
        step=2,
    )
    assert len(changes) >= 1
    assert changes[0]["from_polarity"] == "pro"
    assert changes[0]["to_polarity"] == "contra"


def test_five_topic_regression_personas_and_seed():
    for _name, scenario, domain, needles in REGRESSION_TOPICS:
        assert detect_topic_domain(scenario) == domain
        measure = measure_label(scenario, domain)
        seed = build_public_seed_content(scenario)
        agents = generate_dynamic_agents(scenario, 3)
        blob = " ".join(
            f"{a.description} {a.goal} {a.role_label} {a.occupation}" for a in agents
        )
        assert not contains_public_feed_leak(seed)
        assert "[id=" not in seed
        for a in agents:
            assert a.age is not None
            assert a.occupation
            assert a.role_label
            assert " " in a.name.strip()
            assert a.role_label != "Moderator" or domain == "generic"
            # Labor/income seats must stay role-diverse (no company=Betriebsrat).
            if "Unternehmen" in (a.role_label or ""):
                assert "Betriebsrat" not in (a.occupation or "")
        if domain == "labor":
            assert "Grundeinkommen" not in blob
            assert "Grundeinkommen" not in seed
            assert any(n in measure.lower() or n in seed.lower() for n in needles)
        if domain == "income":
            assert "grundeinkommen" in measure.lower()
        assert not contains_foreign_topic_leak(scenario, seed)
        # Deterministic.
        again = generate_dynamic_agents(scenario, 3)
        assert [a.user_name for a in agents] == [a.user_name for a in again]


def test_phase731_fixture_reingest_regression():
    """7.3.1 repetition fix must remain green."""
    assert LIVE_FIXTURE_731.is_file()
    data = json.loads(LIVE_FIXTURE_731.read_text(encoding="utf-8"))
    eng = DiscussionEngine()
    for step in range(3):
        eng.ingest_platform(
            posts=data["posts"], comments=data["comments"], step=step, round_n=step
        )
    snap = eng.snapshot()
    assert snap["repetition_blocks"] == 0
    assert snap["argument_count"] == 6
    assert snap["quality_fix"] in {"7.3.2", "8.0", "8.1", "8.2", "8.3", "8.4"}


def test_offline_manual_live_scenario_zero_cost(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=LIVE_SCENARIO_732,
        language="de",
        mode="manual",
        max_steps=3,
        agent_count=3,
        agent_generation="dynamic",
    )
    assert started.get("ok") is True
    state = _wait_terminal(engine)
    assert state["status"] in {"completed", "limit_reached"}
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] in {"7.3.2", "8.0", "8.1", "8.2", "8.3", "8.4"}
    assert int(data["cost_summary"].get("api_calls") or 0) == 0
    assert float(data["cost_summary"].get("estimated_cost_usd") or 0) == 0.0
    agents = data["agents"]
    assert len(agents) == 3
    assert all(a.get("role_label") != "Moderator" for a in agents)
    assert all("Betriebsrat" not in (a.get("occupation") or "") for a in agents)
    assert any(
        "Wirtschaftsexperte" in (a.get("role_label") or "")
        or "Ökonom" in (a.get("role_label") or "")
        for a in agents
    )
    seed = data["posts"][0]["content"]
    assert "Grundeinkommen" not in seed
    assert not contains_public_feed_leak(seed)
    paid = data["meta"].get("society_intelligence", {}).get("paid_llm_unlocked")
    assert paid is False or paid is None
