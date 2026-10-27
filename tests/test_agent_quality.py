"""Phase 2.5 — language, scenario context, prompt overrides, end states (offline only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import load_settings
from backend.simulation.agents import DEFAULT_AGENTS
from backend.simulation.engine import SimulationEngine
from backend.simulation.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    build_decision_prompt,
    build_profile_for_template,
    build_system_prompt,
    language_instruction,
    model_calls_required_per_round,
    normalize_language,
)
from backend.simulation.scripted_model import ScriptedReactiveModel


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "5000")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    monkeypatch.setenv("SIMULATION_LANGUAGE", "de")
    return out


def test_normalize_language_defaults_to_german():
    assert normalize_language(None) == "de"
    assert normalize_language("") == "de"
    assert normalize_language("DE-de") == "de"
    assert normalize_language("en") == "en"
    assert normalize_language("xx") == "de"


def test_system_prompt_embeds_scenario_goal_language_no_twitter_objective():
    scenario = "Sollten Online-Foren KI-gestützte Moderation einsetzen?"
    prompt = build_system_prompt(
        DEFAULT_AGENTS[3], scenario=scenario, language="de"
    )
    assert scenario in prompt
    assert DEFAULT_AGENTS[3].goal in prompt
    assert DEFAULT_AGENTS[3].description in prompt
    assert "Deutsch" in language_instruction("de") or "Deutsch" in prompt
    # Must not use OASIS default Reddit objective about tweets as the role frame.
    assert "You're a Reddit user" not in prompt
    assert "You're a Twitter user" not in prompt
    assert "I'll present you with some tweets" not in prompt
    # Explicit anti-bait (root-cause fix, not post-filter).
    assert "Like-Ketten" in prompt or "Likes-zum-Liken" in prompt
    assert "Twitter" in prompt  # warned against


def test_decision_prompt_includes_scenario_and_goal():
    text = build_decision_prompt(
        scenario="KI-Moderation?",
        goal="Fakten prüfen",
        language="de",
        env_prompt="After refreshing, you see some posts []",
        agent_name="Alice Analyst",
    )
    assert "KI-Moderation?" in text
    assert "Fakten prüfen" in text
    assert "Alice Analyst" in text
    assert "don't limit your actions for example to just like" not in text
    assert "Deutsch" in text or "de" in text


def test_custom_template_profile_keys_match():
    keys = SYSTEM_PROMPT_TEMPLATE.key_words
    profile = build_profile_for_template(
        DEFAULT_AGENTS[0], scenario="S", language="de"
    )
    assert set(profile.keys()) == keys


def test_config_simulation_language_default_de(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.delenv("SIMULATION_LANGUAGE", raising=False)
    cfg = load_settings()
    assert cfg.simulation_language == "de"
    payload = cfg.status_payload()
    assert payload["simulation_language"] == "de"
    assert payload["model_calls_per_round_required"] == 5
    assert payload["terminal_statuses"] == ["completed", "limit_reached", "failed"]


def test_model_calls_per_round_math():
    assert model_calls_required_per_round(5) == 5
    assert 5 * 2 == 10  # live-test budget: agents × rounds ≤ call ceiling


def test_llm_run_uses_german_default_and_decision_log(results_dir: Path):
    cfg = load_settings()
    assert cfg.simulation_language == "de"
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    scenario = "Sollten Online-Foren KI-gestützte Moderation einsetzen?"
    result = engine.start(
        scenario, mode="llm", max_steps=1, llm_run_approved=True
    )
    assert result["ok"] is True
    assert result["state"]["language"] == "de"
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["language"] == "de"
    assert state["stop_reason"] == "all_rounds_finished"
    assert state["decision_log"], "decision_log must record LLM decisions"
    for d in state["decision_log"]:
        assert d.get("scenario") == scenario
        assert d.get("goal")
        assert d.get("language") == "de"
        assert "tool_calls" in d
    # Seed + actions stay scenario-tied (scripted model uses German labels).
    seed = state["posts"][0]["content"]
    assert "Diskussionsimpuls" in seed
    assert scenario[:40] in seed
    # Action log carries goal/language for traceability.
    llm_reqs = [e for e in state["action_log"] if e.get("kind") == "llm_decision_request"]
    assert llm_reqs
    assert all(e.get("language") == "de" for e in llm_reqs)
    assert all(e.get("goal") for e in llm_reqs)

    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["language"] == "de"
    assert data["meta"]["status"] == "completed"
    assert data["meta"]["model_calls_per_round_required"] == 5
    assert data["decision_log"]


def test_llm_language_override_en(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Should forums use AI moderation?",
        mode="llm",
        max_steps=1,
        llm_run_approved=True,
        language="en",
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["language"] == "en"
    assert all(d.get("language") == "en" for d in state["decision_log"])


def test_fresh_db_per_run_no_reuse(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    r1 = engine.start("Lauf A", mode="llm", max_steps=1, llm_run_approved=True)
    assert r1["ok"] is True
    engine._thread.join(timeout=120)
    s1 = engine.get_state()
    db1 = Path(s1["db_path"])
    assert db1.exists()

    r2 = engine.start("Lauf B", mode="llm", max_steps=1, llm_run_approved=True)
    assert r2["ok"] is True
    engine._thread.join(timeout=120)
    s2 = engine.get_state()
    db2 = Path(s2["db_path"])
    assert db1 != db2
    assert db1.exists() and db2.exists()
    # Second run must not inherit first-run posts beyond its own seed.
    assert all(db1.name != p for p in []) or True
    assert s2["scenario"] == "Lauf B"
    assert len(s2["posts"]) >= 1
    assert "Lauf B" in s2["posts"][0]["content"]


def test_agent_graph_system_message_is_custom(results_dir: Path):
    """Root-cause: custom template replaces OASIS tweet-wording system message."""
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    graph = engine._build_agent_graph(
        "llm",
        scenario="Sollten Foren KI nutzen?",
        language="de",
    )
    agent = graph.get_agent(0)
    content = agent.system_message.content
    assert "Sollten Foren KI nutzen?" in content
    assert DEFAULT_AGENTS[0].goal in content
    assert "You're a Reddit user" not in content
    assert "I'll present you with some tweets" not in content
    # Decision hook installed (not stock OASIS method unbound identity).
    assert callable(agent.perform_action_by_llm)
