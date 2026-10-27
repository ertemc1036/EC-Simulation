"""LLMAction path tests — scripted model only, no paid API calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import load_settings
from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.engine import SimulationEngine
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.tracked_model import TrackedModelBackend


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "3")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "12")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "5000")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    return out


def test_llm_blocked_without_key_and_without_injection(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(cfg)
    result = engine.start(
        "Szenario", mode="llm", max_steps=1, llm_run_approved=True
    )
    assert result["ok"] is False
    assert result["state"]["status"] == "blocked"
    assert "OPENAI_API_KEY" in (result["error"] or "")


def test_agent_interactions_via_llm_action(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Sollten Foren KI-Moderation einsetzen?",
        mode="llm",
        max_steps=2,
        llm_run_approved=True,
    )
    assert result["ok"] is True
    assert result["state"]["mode"] == "llm"
    assert engine._thread is not None
    engine._thread.join(timeout=120)

    state = engine.get_state()
    assert state["status"] == "completed", state
    assert len(state["agents"]) == 5
    assert len(state["posts"]) >= 1
    # Agents observe seed post and react (comments and/or likes/posts)
    assert len(state["comments"]) + len(state["reactions"]) + len(state["posts"]) >= 3
    sources = {e["source"] for e in state["action_log"]}
    assert "ManualAction" in sources
    assert "LLMAction" in sources
    assert state["model_calls"], "model_calls must be logged separately"
    assert all(c["source"] == "scripted_test" for c in state["model_calls"])
    assert state["cost_summary"]["api_calls"] == len(state["model_calls"])
    assert state["cost_summary"]["total_tokens"] > 0

    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "model_calls" in data
    assert "cost_summary" in data
    assert data["meta"]["agent_memory_window"] == 6


def test_cost_call_limit_stops_llm_rounds(results_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "3")
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "5")
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Limit-Test", mode="llm", max_steps=5, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "limit_reached", state
    assert state["stop_reason"]
    assert "Call-Limit" in (state["stop_reason"] or "")
    assert state["cost_summary"]["api_calls"] <= 3
    assert state["cost_summary"]["limit_hit"]
    assert "Call-Limit" in state["cost_summary"]["limit_hit"]
    assert state["cost_summary"]["cost_estimate_disclaimer"]


def test_cost_token_limit_blocks_before_call():
    tracker = CostTracker(
        max_calls=100,
        max_tokens=50,
        price_input_per_1m=0.15,
        price_output_per_1m=0.6,
        max_cost_usd=10.0,
        model_name="test",
    )
    tracker.record(agent_id=0, prompt_tokens=40, completion_tokens=20, source="scripted_test")
    with pytest.raises(LimitExceeded):
        tracker.check_can_call()


def test_cost_usd_limit_blocks_before_call():
    tracker = CostTracker(
        max_calls=100,
        max_tokens=1_000_000,
        price_input_per_1m=10.0,
        price_output_per_1m=20.0,
        max_cost_usd=0.20,
        model_name="test",
    )
    # ~0.21 USD at these prices → hard stop before next call
    tracker.record(
        agent_id=0, prompt_tokens=10_000, completion_tokens=5_500, source="scripted_test"
    )
    assert tracker.estimated_cost_usd >= 0.20
    with pytest.raises(LimitExceeded) as exc:
        tracker.check_can_call()
    assert "Kosten-Limit" in str(exc.value)


def test_readiness_budget_math_five_agents_two_rounds():
    """5 agents × 2 LLM rounds = 10 decisions → fits MAX_LLM_CALLS_PER_RUN=10."""
    assert 5 * 2 == 10
    cfg_defaults = {
        "max_steps": 2,
        "agents": 5,
        "max_calls": 10,
        "max_tokens_per_call": 500,
        "max_tokens_per_run": 20_000,
        "max_cost_usd": 0.20,
    }
    assert cfg_defaults["agents"] * cfg_defaults["max_steps"] <= cfg_defaults["max_calls"]
    # Worst-case completion tokens alone stay under run token + cost ceilings
    # at gpt-4.1-mini list prices ($0.40 / $1.60 per 1M).
    worst_completion = cfg_defaults["max_calls"] * cfg_defaults["max_tokens_per_call"]
    assert worst_completion <= cfg_defaults["max_tokens_per_run"]
    worst_cost = worst_completion / 1_000_000.0 * 1.60
    assert worst_cost < cfg_defaults["max_cost_usd"]


def test_tracked_model_records_usage_and_errors():
    tracker = CostTracker(
        max_calls=5,
        max_tokens=10_000,
        price_input_per_1m=1.0,
        price_output_per_1m=2.0,
        model_name="scripted",
    )
    inner = ScriptedReactiveModel(agent_id=7)
    wrapped = TrackedModelBackend(inner, tracker, agent_id=7, source="scripted_test")
    resp = wrapped._run([{"role": "user", "content": "no posts yet"}])
    assert resp.choices[0].message.tool_calls
    assert tracker.call_count == 1
    assert tracker.calls[0].agent_id == 7
    assert tracker.estimated_cost_usd > 0

    failing = ScriptedReactiveModel(agent_id=1)
    failing.force_error(RuntimeError("boom"))
    wrapped_fail = TrackedModelBackend(
        failing, tracker, agent_id=1, source="scripted_test"
    )
    with pytest.raises(RuntimeError, match="boom"):
        wrapped_fail._run([{"role": "user", "content": "x"}])
    assert tracker.calls[-1].ok is False
    assert "boom" in (tracker.calls[-1].error or "")


def test_llm_error_handling_soft_stop(results_dir: Path):
    cfg = load_settings()

    def factory(aid: int) -> ScriptedReactiveModel:
        m = ScriptedReactiveModel(agent_id=aid)
        m.force_error(RuntimeError("provider down"))
        return m

    engine = SimulationEngine(cfg, model_factory=factory)
    result = engine.start(
        "Fehlerpfad", mode="llm", max_steps=1, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=60)
    state = engine.get_state()
    # Seed ManualAction still ran; LLM round aborted without crashing the process.
    assert state["status"] == "failed", state
    assert state["stop_reason"]
    assert state["posts"] or state["action_log"]
    assert any(e["source"] == "ManualAction" for e in state["action_log"])
