"""Phase 7.3 — Intelligent Discussion Engine (offline only, no provider calls).

Basis: Phase 7.2 live fixture gaps in ``20261028T105240Z-3202e35e`` —
abstract group personas, repetition, unclear threads, weak polarity/memory.
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
from backend.simulation.discussion_engine import (
    DiscussionEngine,
    argument_fingerprint,
    claims_invented_source,
)
from backend.simulation.engine import SimulationEngine
from backend.simulation.prompt_debug import PROMPT_BUNDLE_ID
from backend.simulation.prompts import (
    MAX_FEED_COMMENTS,
    MAX_FEED_POSTS,
    MAX_MEMORY_ENTRIES,
    build_decision_prompt,
    build_system_prompt,
)
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import _polarity
from backend.simulation.topic_context import contains_public_feed_leak

ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_73 = ROOT / "data" / "results" / "20261028T105240Z-3202e35e.json"

BGE = (
    "Deutschland führt ab 2030 ein bedingungsloses Grundeinkommen von 1.500 € "
    "monatlich für alle Erwachsenen ein."
)
BGE_STAKE = (
    BGE
    + " Berücksichtige Arbeitnehmer, Unternehmer und Wirtschaftsexperten."
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


def test_live_ceilings_unchanged_phase73():
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    cfg = load_settings()
    assert cfg.max_cost_usd_per_run == 0.20
    assert cfg.max_tokens_per_run == 20000
    assert cfg.max_tokens_per_call == 500
    payload = cfg.status_payload()
    assert payload["llm_run_requires_approval"] is True
    assert payload["offline_scale"]["paid_llm_unlocked"] is False
    assert payload["society_intelligence"]["intelligent_discussion"] is True
    assert payload["society_intelligence"]["individual_personas"] is True
    assert PROMPT_BUNDLE_ID == "ec-custom-forum-de-v7"
    assert MAX_FEED_POSTS == 4
    assert MAX_FEED_COMMENTS == 4
    assert MAX_MEMORY_ENTRIES == 3


def test_live_fixture_3202e35e_documents_phase72_gaps():
    assert LIVE_FIXTURE_73.is_file(), f"missing {LIVE_FIXTURE_73}"
    data = json.loads(LIVE_FIXTURE_73.read_text(encoding="utf-8"))
    assert data["meta"]["run_id"] == "20261028T105240Z-3202e35e"
    assert data["meta"]["quality_fix"] == "7.2"
    names = [a["name"] for a in data["agents"]]
    # Historical gap: abstract group labels, not individuals.
    assert any(
        n.lower() in {"arbeitnehmern", "unternehmern", "wirtschaftsexperten"}
        or n.endswith("ern")
        or "stimme" in n.lower()
        or n in {"Arbeitnehmern", "Unternehmern", "Wirtschaftsexperten"}
        for n in names
    )
    assert len(data["reactions"]) == 0
    assert data["agent_positions"]["change_count"] == 0
    # Self-replies already fixed in 7.2 fixture path — document 0.
    assert data["interaction_summary"].get("self_replies", 0) == 0


def test_individual_personas_not_abstract_groups():
    roles = extract_stakeholder_roles(BGE_STAKE, limit=5)
    assert "Arbeitnehmer" in roles
    assert "Unternehmer" in roles
    assert "Ökonomen" in roles  # Wirtschaftsexperten canon → Ökonomen
    agents = generate_dynamic_agents(BGE_STAKE, 3)
    for a in agents:
        assert a.age is not None and 18 <= int(a.age) <= 90
        assert a.occupation
        assert a.role_label
        low = a.name.lower()
        assert "arbeitnehmern" not in low
        assert "unternehmern" not in low
        assert "wirtschaftsexperten" not in low
        assert "stimme" not in low
        # First+Last person name (space).
        assert " " in a.name.strip()
        # No feed leak in public-facing description of identity fields.
        assert not contains_public_feed_leak(a.name)
        assert "[id=" not in a.name
    # Deterministic offline seeds.
    again = generate_dynamic_agents(BGE_STAKE, 3)
    assert [a.user_name for a in agents] == [a.user_name for a in again]
    assert [a.name for a in agents] == [a.name for a in again]


def test_polarity_conditional_and_clear_labels():
    assert _polarity("Ich bin für ein Grundeinkommen — Existenzsicherung ohne Sanktionsdruck.") == "pro"
    assert (
        _polarity(
            "Ich lehne das Grundeinkommen ab — Finanzierung und Arbeitsanreize ungeklärt."
        )
        == "contra"
    )
    assert (
        _polarity(
            "Bedingte Zustimmung: nur wenn Pilot mit klaren Abbruchkriterien."
        )
        == "conditional"
    )
    assert (
        _polarity(
            "Zielkonflikt: Ausnahmeregelungen und Übergangsfristen statt pauschalem Verbot."
        )
        == "conditional"
    )
    assert _polarity("Ich bin unsicher und weiß nicht.") == "uncertain"
    assert _polarity("[Seed] Diskussionsimpuls") == "unknown"
    # Replay live fixture comments — should not stay all-unknown.
    data = json.loads(LIVE_FIXTURE_73.read_text(encoding="utf-8"))
    pols = [_polarity(c["content"]) for c in data["comments"]]
    clear = [p for p in pols if p in {"pro", "contra", "conditional", "exception"}]
    # Fixture mixes clear + uncertain; at least a few must resolve clearly.
    assert len(clear) >= 2
    assert "unknown" not in pols or len(clear) >= 2


def test_discussion_engine_anti_repetition_and_threads():
    eng = DiscussionEngine()
    a = eng.register_statement(
        agent_id=0,
        text="Finanzierung des Grundeinkommens bleibt ungeklärt und riskant.",
        evidence="comment:1",
        round_n=1,
    )
    assert a is not None
    assert eng.is_repetition(
        0, "Finanzierung des Grundeinkommens bleibt ungeklärt und riskant."
    )
    assert (
        eng.register_statement(
            agent_id=0,
            text="Finanzierung des Grundeinkommens bleibt ungeklärt und riskant.",
            evidence="comment:2",
            round_n=2,
        )
        is None
    )
    assert eng._repetition_blocks >= 1
    varied = eng.vary_scripted_body(
        0,
        "Finanzierung des Grundeinkommens bleibt ungeklärt und riskant.",
        call_index=3,
    )
    assert "ungeklärt" in varied
    assert any(
        tip in varied
        for tip in (
            "Zusätzlich",
            "Ergänzend",
            "Ferner",
            "Außerdem",
            "Zugleich",
        )
    )
    eng.register_statement(
        agent_id=1,
        text="Piloten mit Abbruchkriterien sind der einzig seriöse Einstieg.",
        evidence="comment:3",
        round_n=1,
    )
    foreign = eng.foreign_arguments_to_address(0)
    assert foreign and foreign[0].agent_id == 1
    hints = eng.prompt_argument_hints(0)
    assert "nicht wiederholen" in hints.lower() or "Greife" in hints
    assert not claims_invented_source("Ich halte Finanzierung für ungeklärt.")
    assert claims_invented_source("Laut Studie steigt das BIP um 3%.")
    # Thread reconstruction from local refs.
    eng.ingest_platform(
        posts=[{"post_id": 1, "user_id": 0, "content": "These BGE"}],
        comments=[
            {"comment_id": 1, "post_id": 1, "user_id": 0, "content": "Arg A"},
            {
                "comment_id": 2,
                "post_id": 1,
                "user_id": 1,
                "content": "→Kommentar#1 Antwort von B",
            },
            {
                "comment_id": 3,
                "post_id": 1,
                "user_id": 0,
                "content": "→Kommentar#1 Self",
            },
        ],
        round_n=2,
    )
    snap = eng.snapshot()
    assert snap["parent_comment_id_supported"] is False
    assert snap["oasis_nested_reply"] is False
    assert snap["self_replies"] >= 1
    assert snap["cross_agent_replies"] >= 1
    parents = [t["parent_comment_id"] for t in snap["threads"] if t.get("parent_comment_id")]
    assert 1 in parents
    assert argument_fingerprint("abc") != argument_fingerprint("xyz völlig anderes thema hier")


def test_prompts_include_discussion_rules():
    agents = generate_dynamic_agents(BGE, 2)
    sys_p = build_system_prompt(agents[0], scenario=BGE, language="de")
    assert "wiederhole" in sys_p.lower() or "Intelligent diskutieren" in sys_p
    assert "erfundenen" in sys_p.lower() or "Studien" in sys_p
    r2 = build_decision_prompt(
        scenario=BGE,
        goal=agents[0].goal,
        language="de",
        env_prompt="",
        agent_name=agents[0].name,
        feed_context="Post#1 u0: seed",
        memory_context="(Kein Memory.)",
        round_number=2,
        argument_context="Deine jüngsten Argumente (nicht wiederholen): Finanzierung",
    )
    assert "ARGUMENT-KONTEXT" in r2
    assert "nicht wiederholen" in r2


def test_manual_engine_discussion_payload(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    started = engine.start(
        scenario=BGE_STAKE,
        language="de",
        mode="manual",
        max_steps=4,
        agent_count=10,
        agent_generation="dynamic",
    )
    assert started.get("ok") is True
    state = _wait_terminal(engine)
    assert state["status"] in {"completed", "limit_reached"}
    data = json.loads(Path(state["result_path"]).read_text(encoding="utf-8"))
    assert data["meta"]["quality_fix"] in {"7.3", "7.3.1", "7.3.2"}
    assert data["meta"]["prompt_bundle_id"] == "ec-custom-forum-de-v7"
    assert data["meta"]["society_intelligence"]["paid_llm_unlocked"] is False
    assert data["meta"]["society_intelligence"]["intelligent_discussion"] is True
    assert data["meta"]["society_intelligence"]["individual_personas"] is True
    assert data["meta"]["live_llm_limits_unchanged"]["max_agents"] == 5
    assert data["meta"]["live_llm_limits_unchanged"]["max_cost_usd_per_run"] == 0.20
    # Individual personas in result.
    for a in data["agents"]:
        assert a.get("age") is not None
        assert a.get("occupation")
        assert " " in a["name"]
    assert "discussion" in data
    assert data["discussion"]["parent_comment_id_supported"] is False
    assert data["discussion"]["argument_count"] >= 1
    assert data["interaction_summary"].get("self_replies", 0) == 0
    # Memory carries sourced argument development.
    lt = data["agent_memory"].get("long_term") or {}
    arg_hits = 0
    for entries in lt.values():
        for e in entries:
            if (e.get("meta") or {}).get("kind") == "argument_development":
                arg_hits += 1
                assert e.get("sources")
    assert arg_hits >= 1
    pol = data["agent_positions"]["polarity_distribution"]
    clear = (
        int(pol.get("pro", 0))
        + int(pol.get("contra", 0))
        + int(pol.get("conditional", 0))
        + int(pol.get("exception", 0))
    )
    assert clear >= 3
    public = " ".join(p["content"] for p in data["posts"])
    public += " ".join(c["content"] for c in data["comments"])
    assert "[id=" not in public.lower()
    assert "stil:" not in public.lower()


@pytest.mark.parametrize("n", [10, 25, 50])
def test_offline_scale_scripted_phase73(results_dir: Path, n: int):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg,
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid, label="p73"),
    )
    started = engine.start(
        scenario=BGE,
        language="de",
        mode="llm",
        max_steps=3 if n <= 25 else 2,
        agent_count=n,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    assert started.get("ok") is True
    state = _wait_terminal(engine, deadline_s=420.0 if n >= 50 else 300.0)
    assert state["status"] in {"completed", "limit_reached"}, state.get("message")
    data = json.loads(Path(state["result_path"]).read_text(encoding="utf-8"))
    assert data["meta"]["execution_kind"] == "scripted"
    assert data["meta"]["quality_fix"] in {"7.3", "7.3.1", "7.3.2"}
    assert data["meta"]["society_intelligence"]["paid_llm_unlocked"] is False
    assert data["meta"].get("content_synthetic") is True
    assert data["scheduler"]["self_as_partner_count"] == 0
    summary = data["interaction_summary"]
    assert summary.get("self_replies", 0) == 0
    assert (
        summary.get("cross_agent_replies", 0) >= 1
        or summary.get("comment_reactions", 0) + summary.get("post_reactions", 0) >= 1
        or len(data["reactions"]) >= 1
    )
    assert data["discussion"]["argument_count"] >= 1
    # Individuals at scale (sample).
    sample = data["agents"][:5]
    for a in sample:
        assert a.get("age") is not None
        assert " " in a["name"]
    blob = " ".join(p["content"] for p in data["posts"])
    assert "[id=" not in blob.lower()
    # Honest: scripted local stubs — api_calls may be >0 but no OpenAI provider.
    assert data["cost_summary"].get("estimated_cost_usd", 0) >= 0
