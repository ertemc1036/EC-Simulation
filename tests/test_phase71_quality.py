"""Phase 7.1 — Simulation quality fix (offline, no live LLM).

Regression vs ``data/results/20261028T094658Z-63717e3f.json`` (BGE 1500€):
mobility role leftovers, goal/instruction leakage into posts, generic clones,
48/50 uncertain, 3 posts, 0 position_changed.
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
    extract_scenario_keywords,
    extract_stakeholder_roles,
    generate_dynamic_agents,
)
from backend.simulation.engine import SimulationEngine
from backend.simulation.manual_script import build_manual_rounds
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import _polarity
from backend.simulation.topic_context import (
    contains_mobility_leak,
    detect_topic_domain,
    synthetic_tag,
)

ROOT = Path(__file__).resolve().parents[1]
BAD_RUN = ROOT / "data" / "results" / "20261028T094658Z-63717e3f.json"

BGE = (
    "Deutschland führt ab 2030 ein bedingungsloses Grundeinkommen von 1.500 € "
    "monatlich für alle erwachsenen Bürger ein.\n\n"
    "Simuliere die gesellschaftlichen Reaktionen auf diese Entscheidung. "
    "Berücksichtige Arbeitnehmer, Arbeitslose, Studierende, Unternehmer, "
    "Rentner, Selbstständige, Politiker, Ökonomen und Menschen mit "
    "unterschiedlichen Einkommen.\n\n"
    "Untersuche, wie sich Meinungen durch Diskussionen verändern."
)

BERLIN = (
    "Was passiert, wenn Berlin ab 2030 private Autos innerhalb des S-Bahn-Rings "
    "verbietet? Wie reagieren Bürger, Unternehmen, Politiker und Verkehrsbetriebe?"
)

KI_THESE = (
    "Sollten generative KI-Systeme in Schulen ab 2027 als Hilfsmittel erlaubt "
    "werden? Berücksichtige Lehrkräfte, Schülerinnen, Eltern, Schulbehörden "
    "und EdTech-Unternehmen."
)

INSTRUCTION_LEAKS = (
    "zielkonflikt:",
    "stil:",
    "stimme #",
    "[id=",
    "halteeine klare gegenposition",
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


def _wait_terminal(engine: SimulationEngine, *, deadline_s: float = 240.0) -> dict:
    end = time.time() + deadline_s
    while time.time() < end:
        state = engine.get_state()
        if state["status"] in {"completed", "limit_reached", "failed"}:
            if state.get("result_path") or state["status"] == "failed":
                return state
        time.sleep(0.15)
    return engine.get_state()


def test_live_ceilings_unchanged_phase71():
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    cfg = load_settings()
    assert cfg.status_payload()["llm_run_requires_approval"] is True
    assert cfg.max_cost_usd_per_run == 0.20


def test_bad_run_fixture_documents_pre_fix_failure():
    assert BAD_RUN.is_file(), f"missing fixture {BAD_RUN}"
    data = json.loads(BAD_RUN.read_text(encoding="utf-8"))
    assert data["meta"]["phase"] in (7, 8)
    assert len(data["posts"]) <= 3
    pol = data["agent_positions"]["polarity_distribution"]
    assert int(pol.get("uncertain", 0)) >= 40
    assert int(data["agent_positions"]["change_count"]) == 0
    blob = " ".join(
        a.get("description", "") + a.get("goal", "") for a in data["agents"]
    ).lower()
    assert "mobilität" in blob or "stadtbevölkerung" in blob
    comments = " ".join(c["content"] for c in data["comments"][:20]).lower()
    assert "zielkonflikt" in comments


def test_bge_roles_topic_specific_no_mobility_leak():
    assert detect_topic_domain(BGE) == "income"
    roles = extract_stakeholder_roles(BGE, limit=8)
    assert "Arbeitnehmer" in roles
    assert "Arbeitslose" in roles
    assert "Studierende" in roles
    assert "Ökonomen" in roles or "Politiker" in roles
    agents = generate_dynamic_agents(BGE, 12)
    for a in agents:
        joined = f"{a.description} {a.goal} {a.name}"
        assert not contains_mobility_leak(joined), a.name
    kw = {k.lower() for k in extract_scenario_keywords(BGE)}
    assert "simuliere" not in kw
    assert "reaktionen" not in kw
    assert any("grundeinkommen" in k for k in kw)


def test_three_topics_yield_different_discussions():
    """Different theses → different roles / domains / argument markers."""
    specs = [
        (BGE, "income", "grundeinkommen"),
        (BERLIN, "mobility", "verbot"),
        (KI_THESE, "education", "ki"),
    ]
    fingerprints: list[str] = []
    for scenario, domain, needle in specs:
        assert detect_topic_domain(scenario) == domain
        agents = generate_dynamic_agents(scenario, 8)
        rounds = build_manual_rounds(scenario, agents=agents, max_steps=3)
        texts: list[str] = []
        for round_map in rounds:
            for acts in round_map.values():
                for act in acts:
                    c = (act.action_args or {}).get("content")
                    if c:
                        texts.append(str(c))
        blob = " ".join(texts).lower()
        assert synthetic_tag().lower() in blob
        assert needle in blob or needle in " ".join(a.description for a in agents).lower()
        for t in texts:
            low = t.lower()
            # Public content must not dump persona instruction fields.
            assert "stil:" not in low
            assert "[id=" not in low
            # Goal dump pattern from 63717e3f.
            assert not (low.count("zielkonflikt:") >= 1 and "perspektive zu" in low)
        fingerprints.append("|".join(a.name for a in agents[:4]) + "::" + blob[:120])
    assert len(set(fingerprints)) == 3


def test_manual_arguments_have_clear_polarity_and_shifts():
    agents = generate_dynamic_agents(BGE, 10)
    rounds = build_manual_rounds(BGE, agents=agents, max_steps=4)
    contents: list[str] = []
    for round_map in rounds:
        for acts in round_map.values():
            for act in acts:
                c = (act.action_args or {}).get("content")
                if c:
                    contents.append(str(c))
    assert len(contents) >= 12
    pols = {_polarity(c) for c in contents}
    assert "pro" in pols or "contra" in pols or "conditional" in pols or "exception" in pols
    assert any("ändere meine position" in c.lower() for c in contents)
    assert any("→kommentar#" in c.lower() for c in contents)
    # No instruction-style goal dumps.
    for c in contents:
        low = c.lower()
        assert "halteeine klare" not in low.replace(" ", "")
        assert "stimme #" not in low


def test_manual_engine_bge_quality_vs_bad_run(results_dir: Path):
    """Offline Manual BGE run improves posts / polarity / changes vs 63717e3f."""
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=BGE,
        language="de",
        mode="manual",
        max_steps=4,
        agent_count=25,
        agent_generation="dynamic",
    )
    assert started.get("ok") is True
    state = _wait_terminal(engine)
    assert state["status"] in {"completed", "limit_reached"}
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["meta"]["content_synthetic"] is True
    assert data["meta"]["execution_kind"] == "manual"
    assert data["meta"]["live_llm_limits_unchanged"]["max_agents"] == 5
    assert data["meta"]["society_intelligence"]["paid_llm_unlocked"] is False

    # Before (fixture) vs after.
    before = json.loads(BAD_RUN.read_text(encoding="utf-8"))
    assert len(data["posts"]) > len(before["posts"])
    assert len(data["posts"]) >= 4

    pol = data["agent_positions"]["polarity_distribution"]
    uncertain = int(pol.get("uncertain", 0)) + int(pol.get("unknown", 0))
    clear = (
        int(pol.get("pro", 0))
        + int(pol.get("contra", 0))
        + int(pol.get("conditional", 0))
        + int(pol.get("exception", 0))
    )
    assert clear >= 8
    assert uncertain < 20  # was 48/50

    changes = int(data["agent_positions"]["change_count"])
    assert changes >= 1
    assert int(data["interaction_summary"].get("position_changed", 0)) >= 1

    public = " ".join(p["content"] for p in data["posts"])
    public += " ".join(c["content"] for c in data["comments"])
    low = public.lower()
    assert synthetic_tag().lower() in low
    assert "grundeinkommen" in low or "existenzsicherung" in low
    assert "stadtbevölkerung" not in low
    assert "mobilitätseingeschränkt" not in low
    # Instruction dumps from goals must not dominate public feed.
    assert low.count("stil:") == 0
    for a in data["agents"]:
        assert not contains_mobility_leak(
            (a.get("description") or "") + (a.get("goal") or "")
        )


def test_scripted_bge_topic_arguments(results_dir: Path):
    cfg = load_settings()

    def factory(agent_id: int):
        return ScriptedReactiveModel(agent_id=agent_id, label="p71")

    engine = SimulationEngine(cfg, model_factory=factory)
    started = engine.start(
        scenario=BGE,
        language="de",
        mode="llm",
        max_steps=3,
        agent_count=10,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    assert started.get("ok") is True
    state = _wait_terminal(engine, deadline_s=300.0)
    assert state["status"] in {"completed", "limit_reached"}
    data = json.loads(Path(state["result_path"]).read_text(encoding="utf-8"))
    assert data["meta"]["execution_kind"] == "scripted"
    assert data["meta"]["content_synthetic"] is True
    blob = " ".join(p["content"] for p in data["posts"])
    blob += " ".join(c["content"] for c in data["comments"])
    low = blob.lower()
    assert "[synthetisch]" in low
    assert "grundeinkommen" in low or "vollauszahlung" in low or "existenzsicherung" in low
    assert "pauschalem verbot" not in low or "grundeinkommen" in low
    # Thesis-specific: should not be pure Berlin-Verbot copy when BGE scenario set.
    assert "existenzsicherung" in low or "gegenfinanzierung" in low or "pilot" in low
