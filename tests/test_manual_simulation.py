"""Real OASIS ManualAction run with 5 agents — no LLM API calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import load_settings
from backend.simulation.engine import SimulationEngine


@pytest.fixture()
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimulationEngine:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "4")
    cfg = load_settings()
    return SimulationEngine(cfg)


def test_manual_simulation_persists_results(engine: SimulationEngine, tmp_path: Path):
    result = engine.start(
        "Sollten Foren KI-Moderation einsetzen?",
        mode="manual",
        max_steps=4,
    )
    assert result["ok"] is True
    assert result["state"]["mode"] == "manual"
    assert "OPENAI_API_KEY fehlt" in result["state"]["message"] or "ManualAction" in result[
        "state"
    ]["message"]

    # Wait for background thread
    assert engine._thread is not None
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert len(state["agents"]) == 5
    assert len(state["posts"]) >= 2
    assert len(state["action_log"]) >= 2
    assert state["result_path"]
    path = Path(state["result_path"])
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["mode"] == "manual"
    assert data["meta"]["oasis"] == "camel-oasis==0.2.5"
    assert len(data["posts"]) >= 2


def test_llm_mode_blocked_without_key(engine: SimulationEngine):
    result = engine.start("Test", mode="llm", max_steps=1, llm_run_approved=True)
    assert result["ok"] is False
    assert result["state"]["status"] == "blocked"
    assert "OPENAI_API_KEY" in (result["error"] or "")


def test_default_and_auto_modes_stay_manual(engine: SimulationEngine):
    for mode in (None, "auto", "manual"):
        result = engine.start("Manual default", mode=mode, max_steps=1)
        assert result["ok"] is True
        assert result["state"]["mode"] == "manual"
        assert engine._thread is not None
        engine._thread.join(timeout=120)
        assert engine.get_state()["status"] == "completed"
