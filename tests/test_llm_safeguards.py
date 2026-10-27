"""Offline safeguards: LLM approval gate, live-test ceilings, limit abort."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app import app
from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    LIVE_TEST_MAX_RETRIES,
    load_settings,
)
from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.engine import SimulationEngine
from backend.simulation.scripted_model import ScriptedReactiveModel


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "20000")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    return out


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_llm_start_blocked_without_approval(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start("Ohne Freigabe", mode="llm", max_steps=1)
    assert result["ok"] is False
    assert result["state"]["status"] == "blocked"
    assert "Freigabe" in (result["error"] or "")
    assert engine._thread is None or not engine._thread.is_alive()


def test_api_llm_start_requires_approval(client):
    # Missing approval must 409 even before key/availability checks.
    res = client.post(
        "/api/simulation/start",
        json={"scenario": "Freigabe-Test", "mode": "llm", "max_steps": 1},
    )
    assert res.status_code == 409
    data = res.get_json()
    assert data["ok"] is False
    assert "Freigabe" in (data["error"] or "")
    assert data["state"]["status"] == "blocked"


def test_api_auto_mode_does_not_enable_llm(client):
    from backend import app as app_module

    # Wait out any leftover worker from earlier API tests.
    if app_module.engine._thread is not None and app_module.engine._thread.is_alive():
        app_module.engine._thread.join(timeout=120)

    res = client.post(
        "/api/simulation/start",
        json={"scenario": "Auto bleibt Manual", "mode": "auto", "max_steps": 1},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["state"]["mode"] == "manual"
    if app_module.engine._thread is not None:
        app_module.engine._thread.join(timeout=120)


def test_live_test_ceilings_clamp_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "ceil"))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "99")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "999")
    monkeypatch.setenv("LLM_MAX_RETRIES", "7")
    cfg = load_settings()
    assert cfg.max_simulation_steps == LIVE_TEST_MAX_ROUNDS
    assert cfg.max_llm_calls_per_run == LIVE_TEST_MAX_LLM_CALLS
    assert cfg.llm_max_retries == LIVE_TEST_MAX_RETRIES
    assert cfg.max_agents == LIVE_TEST_MAX_AGENTS
    payload = cfg.status_payload()
    assert payload["llm_run_requires_approval"] is True
    assert payload["default_mode"] == "manual"
    assert payload["live_test_limits"]["max_model_calls"] == 10
    assert "Provider-Abrechnung" in payload["cost_estimate_disclaimer"]
    assert payload["estimated_max_cost_usd"] <= payload["max_cost_usd_per_run"]


def test_parallel_simulation_rejected(results_dir: Path):
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    first = engine.start(
        "Erster Lauf", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert first["ok"] is True
    second = engine.start(
        "Zweiter Lauf", mode="manual", max_steps=1, llm_run_approved=False
    )
    assert second["ok"] is False
    assert "bereits" in (second["error"] or "").lower()
    assert engine._thread is not None
    engine._thread.join(timeout=120)


def test_conservative_preflight_blocks_before_call():
    tracker = CostTracker(
        max_calls=10,
        max_tokens=100_000,
        price_input_per_1m=100.0,
        price_output_per_1m=200.0,
        max_cost_usd=0.05,
        model_name="test",
        max_tokens_per_call=500,
    )
    # One recorded call already near the soft budget; next conservative estimate must block.
    tracker.record(
        agent_id=0, prompt_tokens=200, completion_tokens=100, source="scripted_test"
    )
    with pytest.raises(LimitExceeded) as exc:
        tracker.check_can_call(conservative=True)
    assert "Kosten-Limit" in str(exc.value)
    assert "Provider-Abrechnung" in str(exc.value) or "konservative" in str(exc.value)


def test_call_limit_aborts_run(results_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "2")
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    cfg = load_settings()
    assert cfg.max_llm_calls_per_run == 2
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Call-Limit-Abbruch", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["cost_summary"]["api_calls"] <= 2
    assert state["cost_summary"]["limit_hit"]
    assert "Call-Limit" in state["cost_summary"]["limit_hit"]


def test_approved_llm_injection_still_offline(results_dir: Path):
    """Approval unlocks LLMAction path; scripted model ensures no provider I/O."""
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Freigegebener Offline-Lauf",
        mode="llm",
        max_steps=1,
        llm_run_approved=True,
    )
    assert result["ok"] is True
    assert "Freigabe" in result["state"]["message"]
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert all(c["source"] == "scripted_test" for c in state["model_calls"])
