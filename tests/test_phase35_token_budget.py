"""Phase 3.5 — token budget safety (offline, no OpenAI / live LLM)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from backend.config import (
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_ROUNDS,
    load_settings,
)
from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.engine import SimulationEngine
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.tracked_model import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    TrackedModelBackend,
)


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "20000")
    monkeypatch.setenv("MAX_TOKENS_PER_CALL", "500")
    monkeypatch.setenv("MAX_PARALLEL_LLM_CALLS", "1")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    return out


def test_live_ceilings_unchanged(results_dir: Path):
    cfg = load_settings()
    assert cfg.max_tokens_per_run == 20_000
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_llm_calls_per_run == LIVE_TEST_MAX_LLM_CALLS == 10
    assert cfg.max_cost_usd_per_run == 0.20
    assert cfg.max_parallel_llm_calls == LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    assert cfg.max_agents == LIVE_TEST_MAX_AGENTS == 5
    assert cfg.max_simulation_steps == LIVE_TEST_MAX_ROUNDS == 2
    # Env cannot raise the parallel ceiling.
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1


def test_context_token_limit_not_completion_cap(results_dir: Path):
    tracker = CostTracker(
        max_calls=10,
        max_tokens=20_000,
        price_input_per_1m=0.4,
        price_output_per_1m=1.6,
        max_tokens_per_call=500,
    )
    inner = ScriptedReactiveModel(agent_id=0)
    # Completion cap stays in config; context window must stay large.
    inner.model_config_dict = {"max_tokens": 500}
    wrapped = TrackedModelBackend(inner, tracker, agent_id=0, source="scripted_test")
    assert wrapped.token_limit >= 4096
    assert wrapped.token_limit == DEFAULT_CONTEXT_TOKEN_LIMIT or wrapped.token_limit >= 4096
    assert tracker.max_tokens_per_call == 500


def test_atomic_reservation_prevents_parallel_overcommit():
    """Reproduce the 20k→31k race: parallel preflights without shared reservation."""
    tracker = CostTracker(
        max_calls=10,
        max_tokens=10_000,
        price_input_per_1m=0.4,
        price_output_per_1m=1.6,
        max_cost_usd=10.0,
        max_tokens_per_call=500,
        max_parallel_calls=5,
    )
    results: list[str] = []
    barrier = threading.Barrier(5)

    def worker() -> None:
        barrier.wait()
        try:
            rid = tracker.reserve_call(
                estimated_prompt_tokens=4000,
                estimated_completion_tokens=500,
                conservative=False,
            )
            results.append(f"ok:{rid}")
        except LimitExceeded:
            results.append("blocked")

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    ok = [r for r in results if r.startswith("ok:")]
    blocked = [r for r in results if r == "blocked"]
    # 2 × 4500 = 9000 ≤ 10000; a third 4500 would be 13500 > 10000.
    assert len(ok) == 2, results
    assert len(blocked) == 3, results
    assert tracker.reserved_tokens == 9000
    assert tracker.remaining_tokens() == 1000


def test_commit_releases_reservation_and_accounts_usage():
    tracker = CostTracker(
        max_calls=5,
        max_tokens=8_000,
        price_input_per_1m=0.4,
        price_output_per_1m=1.6,
        max_cost_usd=1.0,
        max_tokens_per_call=500,
    )
    rid = tracker.reserve_call(
        estimated_prompt_tokens=3000,
        estimated_completion_tokens=500,
        conservative=False,
    )
    assert tracker.reserved_tokens == 3500
    tracker.commit_reservation(
        rid,
        agent_id=1,
        prompt_tokens=2800,
        completion_tokens=120,
        source="scripted_test",
    )
    assert tracker.reserved_tokens == 0
    assert tracker.total_tokens == 2920
    assert tracker.call_count == 1
    # Next reserve must see committed usage.
    with pytest.raises(LimitExceeded):
        tracker.reserve_call(
            estimated_prompt_tokens=5500,
            estimated_completion_tokens=500,
            conservative=False,
        )


def test_conservative_unknown_prompt_not_just_completion_cap():
    tracker = CostTracker(
        max_calls=10,
        max_tokens=20_000,
        price_input_per_1m=0.4,
        price_output_per_1m=1.6,
        max_tokens_per_call=500,
    )
    prompt, completion = tracker.conservative_call_budget()
    assert completion == 500
    assert prompt >= 4096
    assert prompt > completion


def test_token_overshoot_status_is_limit_reached_not_completed(
    results_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stop ≠ false completed: exceeding MAX_TOKENS_PER_RUN → limit_reached."""
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "5000")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")

    class FatUsageModel(ScriptedReactiveModel):
        def _completion(self, name, args):  # type: ignore[override]
            out = super()._completion(name, args)
            # Force billable usage far above short-prompt estimate.
            out.usage.prompt_tokens = 4000
            out.usage.completion_tokens = 100
            out.usage.total_tokens = 4100
            return out

    # Under-estimate on purpose so commit can overshoot once (post-check path).
    monkeypatch.setattr(
        TrackedModelBackend,
        "_estimate_prompt_tokens",
        lambda self, messages: 80,
    )

    cfg = load_settings()
    assert cfg.max_tokens_per_run == 5000
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: FatUsageModel(agent_id=aid)
    )
    result = engine.start(
        "Token-Overshoot", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "limit_reached", state
    assert state["cost_summary"]["tokens_exceeded"] is True
    assert state["cost_summary"]["total_tokens"] > 5000
    assert "Token" in (state["stop_reason"] or "") or "token" in (
        state["stop_reason"] or ""
    ).lower()
    # Must not look like a clean success.
    assert state["status"] != "completed"
    data = json.loads(Path(state["result_path"]).read_text(encoding="utf-8"))
    assert data["meta"]["status"] == "limit_reached"


def test_reservation_plus_serial_calls_stay_within_budget(
    results_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """With honest estimates, total_tokens must stay ≤ MAX_TOKENS_PER_RUN."""
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "10000")

    class FatUsageModel(ScriptedReactiveModel):
        def _completion(self, name, args):  # type: ignore[override]
            out = super()._completion(name, args)
            out.usage.prompt_tokens = 3500
            out.usage.completion_tokens = 100
            out.usage.total_tokens = 3600
            return out

    # Estimate ≥ actual so pre-reserve blocks further calls in time.
    monkeypatch.setattr(
        TrackedModelBackend,
        "_estimate_prompt_tokens",
        lambda self, messages: 3500,
    )

    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: FatUsageModel(agent_id=aid)
    )
    result = engine.start(
        "Budget-Halt", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    total = state["cost_summary"]["total_tokens"]
    assert total <= 10_000, state["cost_summary"]
    assert state["cost_summary"]["tokens_exceeded"] is False
    # Either clean completed (fit) or limit_reached (stopped before overshoot).
    assert state["status"] in {"completed", "limit_reached"}, state
    assert state["status"] != "failed"


def test_call_budget_exhaust_on_final_round_still_completed(results_dir: Path):
    """Call-ceiling on last round remains completed (not a token overshoot)."""
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Call-Budget-Ende", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["stop_reason"] == "all_rounds_finished_budget_exhausted"
    assert state["cost_summary"]["remaining_calls"] == 0
    assert state["cost_summary"]["total_tokens"] <= state["cost_summary"][
        "max_tokens_per_run"
    ]
