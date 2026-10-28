"""Phase 7.3.1 — Topic Integrity & Discussion Quality (offline only).

Live fixture ``20261028T112739Z-bdfb9e9d`` (Vier-Tage-Woche) had BGE personas
and a BGE seed because stakeholder role words (Arbeitnehmer) triggered the
income domain and ``measure_label`` invented «bedingungsloses Grundeinkommen».

No provider calls. Ceilings unchanged.
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
from backend.simulation.agent_generator import generate_dynamic_agents
from backend.simulation.discussion_engine import DiscussionEngine
from backend.simulation.engine import SimulationEngine
from backend.simulation.social_dynamics import _polarity
from backend.simulation.topic_context import (
    build_public_seed_content,
    contains_foreign_topic_leak,
    contains_public_feed_leak,
    detect_topic_domain,
    measure_label,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_731 = ROOT / "data" / "results" / "20261028T112739Z-bdfb9e9d.json"

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


def test_live_ceilings_unchanged_phase731():
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    cfg = load_settings()
    assert cfg.max_cost_usd_per_run == 0.20
    payload = cfg.status_payload()
    assert payload["llm_run_requires_approval"] is True
    assert payload["offline_scale"]["paid_llm_unlocked"] is False
    assert payload["society_intelligence"]["quality_fix"] in {"7.3.1", "7.3.2", "8.0", "8.1", "8.2", "8.3"}


def test_fixture_documents_bge_leak_before_fix():
    """Stored live run proves the pre-fix Vier-Tage → BGE contamination."""
    assert LIVE_FIXTURE_731.is_file()
    data = json.loads(LIVE_FIXTURE_731.read_text(encoding="utf-8"))
    assert "Vier-Tage-Woche" in data["meta"]["scenario"]
    seed = data["posts"][0]["content"]
    assert "bedingungsloses Grundeinkommen" in seed
    blob = " ".join(
        f"{a.get('description', '')} {a.get('goal', '')}" for a in data["agents"]
    )
    assert "Grundeinkommen" in blob
    assert int(data["discussion"]["repetition_blocks"]) == 9


def test_vier_tage_no_longer_classified_as_income():
    assert detect_topic_domain(VIER) == "labor"
    assert "Grundeinkommen" not in measure_label(VIER)
    assert "Vier-Tage" in measure_label(VIER)
    seed = build_public_seed_content(VIER)
    assert "Grundeinkommen" not in seed
    assert "Vier-Tage" in seed
    assert not contains_public_feed_leak(seed)
    assert not contains_foreign_topic_leak(VIER, seed)
    agents = generate_dynamic_agents(VIER, 3)
    assert len(agents) == 3
    for a in agents:
        joined = f"{a.description} {a.goal} {a.name}"
        assert "Grundeinkommen" not in joined
        assert a.age is not None and a.occupation
        assert not contains_foreign_topic_leak(VIER, joined)
        # Individual person — not abstract group display name.
        assert a.name.lower() not in {
            "arbeitnehmern",
            "unternehmern",
            "wirtschaftsexperten",
        }


@pytest.mark.parametrize("label,scenario,domain,needles", REGRESSION_TOPICS)
def test_regression_topics_personas_goals_seed(
    label: str, scenario: str, domain: str, needles: tuple[str, ...]
):
    assert detect_topic_domain(scenario) == domain
    measure = measure_label(scenario).lower()
    seed = build_public_seed_content(scenario)
    assert not contains_public_feed_leak(seed)
    assert not contains_foreign_topic_leak(scenario, seed)
    assert any(n in measure or n in seed.lower() for n in needles)
    # Seed is current thesis only.
    assert seed.startswith("[Seed] Diskussionsimpuls:")
    assert "[id=" not in seed.lower()
    assert "stil:" not in seed.lower()

    agents = generate_dynamic_agents(scenario, 6)
    assert len(agents) == 6
    for a in agents:
        joined = f"{a.description} {a.goal}"
        assert not contains_foreign_topic_leak(scenario, joined)
        assert a.age is not None
        assert a.occupation
        # Goals bind to current measure / topic — no invented BGE on non-BGE.
        if label != "bge":
            assert "grundeinkommen" not in joined.lower()
        else:
            assert "grundeinkommen" in joined.lower()
        # Measure appears in description perspective line.
        assert "perspektive" in joined.lower()


def test_no_cross_run_topic_carryover():
    """Generating BGE then Vier-Tage must not leak BGE into the second roster."""
    _ = generate_dynamic_agents(BGE, 4)
    agents = generate_dynamic_agents(VIER, 4)
    seed = build_public_seed_content(VIER)
    assert "grundeinkommen" not in seed.lower()
    for a in agents:
        assert "grundeinkommen" not in f"{a.description} {a.goal}".lower()


def test_argument_history_reingest_does_not_inflate_repetition():
    """Fixture had 9 blocks from re-sync; same evidence must not re-count."""
    data = json.loads(LIVE_FIXTURE_731.read_text(encoding="utf-8"))
    eng = DiscussionEngine()
    for step in range(4):
        eng.ingest_platform(
            posts=data["posts"], comments=data["comments"], step=step, round_n=step
        )
    snap = eng.snapshot()
    assert snap["argument_count"] == 6
    assert snap["repetition_blocks"] == 0
    assert snap["self_replies"] == 0
    assert snap["quality_fix"] in {"7.3.1", "7.3.2", "8.0", "8.1", "8.2", "8.3"}


def test_position_recognition_on_live_fixture_comments():
    """Re-classify stored Vier-Tage comments — clear polarity, not all uncertain."""
    data = json.loads(LIVE_FIXTURE_731.read_text(encoding="utf-8"))
    pols = [_polarity(c["content"]) for c in data["comments"]]
    assert len(pols) == 6
    assert "uncertain" not in pols or pols.count("uncertain") <= 1
    clear = {"pro", "contra", "conditional", "exception"}
    assert sum(1 for p in pols if p in clear) >= 4
    # First worker comment: positive with caveats → conditional/pro.
    assert pols[0] in {"conditional", "pro"}
    # Entrepreneur critique + pilots → conditional (not unknown).
    assert pols[1] in {"conditional", "contra"}


def test_offline_manual_vier_tage_topic_integrity(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=VIER,
        language="de",
        mode="manual",
        max_steps=3,
        agent_count=10,
        agent_generation="dynamic",
    )
    assert started.get("ok") is True
    state = _wait_terminal(engine)
    assert state["status"] in {"completed", "limit_reached"}
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["execution_kind"] == "manual"
    assert data["meta"]["society_intelligence"]["paid_llm_unlocked"] is False
    assert data["cost_summary"]["api_calls"] == 0
    assert float(data["cost_summary"]["estimated_cost_usd"]) == 0.0

    seed = data["posts"][0]["content"]
    assert "Grundeinkommen" not in seed
    assert "Vier-Tage" in seed
    assert not contains_public_feed_leak(seed)

    for a in data["agents"]:
        joined = f"{a.get('description', '')} {a.get('goal', '')}"
        assert "Grundeinkommen" not in joined
        assert a.get("age") is not None
        assert a.get("occupation")

    public = " ".join(
        [p["content"] for p in data["posts"]]
        + [c["content"] for c in data["comments"]]
    )
    assert "grundeinkommen" not in public.lower()
    assert "[id=" not in public.lower()
    assert "stil:" not in public.lower()
    # Discussion engine still present; individuals + threads preserved.
    assert data["discussion"]["self_replies"] == 0
    assert data["discussion"]["argument_count"] >= 1
