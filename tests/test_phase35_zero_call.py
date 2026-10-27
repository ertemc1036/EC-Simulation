"""Phase 3.5 zero-call bug: completion max_tokens must not starve CAMEL context."""
from __future__ import annotations

from pathlib import Path

import pytest
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    load_settings,
)
from backend.simulation.cost_tracker import CostTracker
from backend.simulation.engine import SimulationEngine
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.tracked_model import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    TrackedModelBackend,
    resolve_context_token_limit,
)


BERLIN = (
    "Was passiert, wenn Berlin ab 2030 private Autos innerhalb des S-Bahn-Rings "
    "verbietet? Wie reagieren Bürger, Unternehmen, Politiker und Verkehrsbetriebe?"
)


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_CALL", "500")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "20000")
    monkeypatch.setenv("MAX_COST_USD_PER_RUN", "0.2")
    return out


def test_context_token_limit_ignores_completion_max_tokens():
    """Regression: CAMEL used max_tokens=500 as context → system prompt > 500."""
    inner = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="gpt-4.1-mini",
        api_key="sk-test-not-used",
        url="https://api.openai.com/v1",
        model_config_dict={"temperature": 0.7, "max_tokens": 500},
    )
    # Stock CAMEL conflates completion cap with context window.
    assert inner.token_limit == 500
    # Our resolver must not.
    assert resolve_context_token_limit(inner) >= 4096

    tracker = CostTracker(
        max_calls=10,
        max_tokens=20_000,
        price_input_per_1m=0.4,
        price_output_per_1m=1.6,
        max_cost_usd=0.2,
        max_tokens_per_call=500,
        model_name="gpt-4.1-mini",
    )
    wrapped = TrackedModelBackend(inner, tracker, agent_id=0, source="llm_api")
    assert wrapped.token_limit >= 4096
    assert wrapped.token_limit != 500
    # Completion budget stays in config (API max_tokens unchanged).
    assert wrapped.model_config_dict.get("max_tokens") == 500
    assert tracker.max_tokens_per_call == 500


def test_scripted_model_native_context_is_large():
    inner = ScriptedReactiveModel(agent_id=0)
    assert resolve_context_token_limit(inner) >= 4096
    assert DEFAULT_CONTEXT_TOKEN_LIMIT >= 4096


def test_tiny_context_window_yields_failed_zero_calls(results_dir: Path):
    """Reproduce pre-fix: context budget << system prompt → 0 calls → failed."""
    cfg = load_settings()
    assert cfg.max_tokens_per_call == 500

    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    orig_wrap = engine._wrap_model

    def wrap_tiny(inner, agent_id: int, mode: str):
        if engine._tracker is None:
            engine._tracker = engine._new_tracker()
        return TrackedModelBackend(
            inner,
            engine._tracker,
            agent_id=agent_id,
            source="scripted_test",
            prompt_debug=engine._prompt_debug if mode == "llm" else None,
            context_token_limit=100,  # Phase-3.5 system prompt is ~500+ tokens
        )

    engine._wrap_model = wrap_tiny  # type: ignore[method-assign]
    result = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        llm_run_approved=True,
        agent_count=5,
        agent_generation="dynamic",
    )
    assert result["ok"] is True
    engine._thread.join(timeout=180)
    engine._wrap_model = orig_wrap  # type: ignore[method-assign]

    state = engine.get_state()
    assert state["status"] == "failed", state
    assert state["stop_reason"] == "llm_zero_model_calls"
    assert state["cost_summary"]["api_calls"] == 0
    assert state["cost_summary"]["total_tokens"] == 0
    # Seed may exist; that alone is not success.
    assert any(a.get("action_type") == "create_post" for a in state["action_log"])
    # Decisions must not look like silent successes when context-blocked.
    decisions = state["decision_log"]
    assert decisions
    assert any(d.get("ok") is False for d in decisions) or state["status"] == "failed"


def test_healthy_context_window_scripted_run_still_completes(results_dir: Path):
    """Phase 3.5 interaction preserved: comments/reactions with real model calls."""
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        BERLIN,
        mode="llm",
        max_steps=2,
        llm_run_approved=True,
        agent_count=5,
        agent_generation="dynamic",
    )
    assert result["ok"] is True
    engine._thread.join(timeout=180)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["cost_summary"]["api_calls"] > 0
    assert state["model_calls"]
    interactive = {"create_comment", "like_post", "dislike_post"}
    action_types = {e.get("action_type") for e in state["action_log"]}
    assert action_types & interactive
    assert state["comments"] or state["reactions"]
    assert any(d.get("feed_context_preview") for d in state["decision_log"])
    assert any(d.get("memory_context_preview") for d in state["decision_log"])


def test_limits_unchanged_after_zero_call_fix():
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    cfg = load_settings()
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_llm_calls_per_run == 10
    assert cfg.max_cost_usd_per_run == 0.2
    assert cfg.max_tokens_per_run == 20_000
    # Stub ModelType still has a large native window for scripted tests.
    assert ModelType.GPT_4O_MINI.token_limit >= 4096
