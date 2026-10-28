"""Phase 8.0/8.1 — Cost & Token Planner V3 (offline-safe, paid LLM never unlocked).

Plans call/token/USD budgets for N×R before a run. Live ceilings stay fixed;
offline/scripted plans remain synthetic ($0 provider). Phase 8.1 adds
wave-aware accounting (planned_waves, calls_per_wave).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    OFFLINE_MAX_AGENTS,
    OFFLINE_MAX_ROUNDS,
    OFFLINE_SCALE_PRESETS,
)
from backend.simulation.social_scheduler import DEFAULT_WAVE_SIZE, wave_count_for

ExecutionKind = Literal["live_llm", "manual", "scripted"]

# Live hard ceilings — never raised by the planner.
_LIVE_MAX_TOKENS_RUN = 20_000
_LIVE_MAX_TOKENS_CALL = 500
_LIVE_MAX_COST_USD = 0.20


@dataclass(frozen=True)
class CostTokenPlan:
    """Pre-run budget plan for dashboard + engine gates."""

    agent_count: int
    rounds: int
    execution_kind: ExecutionKind
    planned_calls: int
    planned_tokens: int
    planned_cost_usd: float
    max_tokens_per_call: int
    within_live_ceilings: bool
    live_clamp_applied: bool
    paid_llm_unlocked: bool
    offline_presets: tuple[int, ...]
    note: str
    # Phase 8.1 wave accounting
    wave_size: int = DEFAULT_WAVE_SIZE
    planned_waves: int = 0
    calls_per_wave: int = 0
    planner_version: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_run(
    *,
    agent_count: int,
    rounds: int,
    execution_kind: ExecutionKind = "manual",
    price_input_per_1m: float = 0.40,
    price_output_per_1m: float = 1.60,
    max_tokens_per_call: int = _LIVE_MAX_TOKENS_CALL,
    wave_size: int = DEFAULT_WAVE_SIZE,
) -> CostTokenPlan:
    """Deterministic budget plan — Manual/Scripted = 0 provider cost."""
    kind: ExecutionKind = execution_kind
    if kind not in {"live_llm", "manual", "scripted"}:
        kind = "manual"

    n = max(1, int(agent_count))
    r = max(1, int(rounds))
    w_size = max(1, int(wave_size))
    live_clamp = False

    if kind == "live_llm":
        if n > LIVE_TEST_MAX_AGENTS or r > LIVE_TEST_MAX_ROUNDS:
            live_clamp = True
        n = min(n, LIVE_TEST_MAX_AGENTS)
        r = min(r, LIVE_TEST_MAX_ROUNDS)
        planned_calls = min(n * r, LIVE_TEST_MAX_LLM_CALLS)
        tok_call = min(max(1, int(max_tokens_per_call)), _LIVE_MAX_TOKENS_CALL)
        # Conservative: both sides may hit per-call ceiling.
        planned_tokens = min(planned_calls * tok_call * 2, _LIVE_MAX_TOKENS_RUN)
        per_call_usd = (
            tok_call / 1_000_000.0 * float(price_input_per_1m)
            + tok_call / 1_000_000.0 * float(price_output_per_1m)
        )
        planned_cost = round(min(planned_calls * per_call_usd, _LIVE_MAX_COST_USD), 6)
        within = (
            n <= LIVE_TEST_MAX_AGENTS
            and r <= LIVE_TEST_MAX_ROUNDS
            and planned_calls <= LIVE_TEST_MAX_LLM_CALLS
            and planned_tokens <= _LIVE_MAX_TOKENS_RUN
            and planned_cost <= _LIVE_MAX_COST_USD
        )
        waves_per_round = wave_count_for(n, w_size)
        planned_waves = waves_per_round * r
        calls_per_wave = min(n, w_size)
        note = (
            "Live-LLM-Plan V3 unter festen Ceilings 5/2/10/20k/500/$0.20 — "
            "Freigabe nötig; paid_llm_unlocked=false; Wellen advisory."
        )
    else:
        n = min(n, OFFLINE_MAX_AGENTS)
        r = min(r, OFFLINE_MAX_ROUNDS)
        # Synthetic accounting only — no provider spend.
        planned_calls = n * r if kind == "scripted" else 0
        tok_call = min(max(1, int(max_tokens_per_call)), _LIVE_MAX_TOKENS_CALL)
        planned_tokens = planned_calls * tok_call if kind == "scripted" else 0
        planned_cost = 0.0
        within = True
        waves_per_round = wave_count_for(n, w_size)
        planned_waves = waves_per_round * r
        calls_per_wave = min(n, w_size) if kind == "scripted" else 0
        note = (
            "Offline/Scripted-Plan V3: Provider-Calls=$0; "
            f"Wellen={planned_waves} (size={w_size}). "
            f"Presets {list(OFFLINE_SCALE_PRESETS)}; Live-Ceilings unverändert."
        )

    return CostTokenPlan(
        agent_count=n,
        rounds=r,
        execution_kind=kind,
        planned_calls=planned_calls,
        planned_tokens=planned_tokens,
        planned_cost_usd=planned_cost,
        max_tokens_per_call=min(max(1, int(max_tokens_per_call)), _LIVE_MAX_TOKENS_CALL),
        within_live_ceilings=within,
        live_clamp_applied=live_clamp,
        paid_llm_unlocked=False,
        offline_presets=tuple(OFFLINE_SCALE_PRESETS),
        note=note,
        wave_size=w_size,
        planned_waves=planned_waves,
        calls_per_wave=calls_per_wave,
        planner_version=3,
    )


def planner_status_payload(
    *,
    price_input_per_1m: float = 0.40,
    price_output_per_1m: float = 1.60,
) -> dict[str, Any]:
    """Public /api/health fragment for Cost & Token Planner V3."""
    sample_live = plan_run(
        agent_count=LIVE_TEST_MAX_AGENTS,
        rounds=LIVE_TEST_MAX_ROUNDS,
        execution_kind="live_llm",
        price_input_per_1m=price_input_per_1m,
        price_output_per_1m=price_output_per_1m,
    )
    sample_offline = plan_run(
        agent_count=50,
        rounds=OFFLINE_MAX_ROUNDS,
        execution_kind="scripted",
    )
    return {
        "enabled": True,
        "version": 3,
        "paid_llm_unlocked": False,
        "wave_size": DEFAULT_WAVE_SIZE,
        "live_ceilings": {
            "max_agents": LIVE_TEST_MAX_AGENTS,
            "max_rounds": LIVE_TEST_MAX_ROUNDS,
            "max_model_calls": LIVE_TEST_MAX_LLM_CALLS,
            "max_parallel_llm_calls": LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
            "max_tokens_per_run": _LIVE_MAX_TOKENS_RUN,
            "max_tokens_per_call": _LIVE_MAX_TOKENS_CALL,
            "max_cost_usd_per_run": _LIVE_MAX_COST_USD,
        },
        "offline_scale": {
            "max_agents": OFFLINE_MAX_AGENTS,
            "max_rounds": OFFLINE_MAX_ROUNDS,
            "presets": list(OFFLINE_SCALE_PRESETS),
            "paid_llm_unlocked": False,
            "sample_waves_n50": wave_count_for(50),
        },
        "sample_live_plan": sample_live.to_dict(),
        "sample_offline_plan": sample_offline.to_dict(),
        "note": (
            "Phase-8.1 Cost & Token Planner V3 — Budgets + Wellen vor dem Lauf; "
            "hebt Live-Ceilings nicht an."
        ),
    }
