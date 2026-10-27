"""Hard limits and accounting for real LLM API calls (separate from sim actions)."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.config import COST_ESTIMATE_DISCLAIMER


class LimitExceeded(RuntimeError):
    """Raised before a model call when a hard cost/call/token limit is hit."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelCallRecord:
    call_id: int
    ts: str
    agent_id: int | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    ok: bool = True
    error: str | None = None
    source: str = "llm_api"  # llm_api | scripted_test | stub

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Reservation:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    max_calls: int
    max_tokens: int
    price_input_per_1m: float
    price_output_per_1m: float
    max_cost_usd: float = 0.20
    model_name: str = ""
    max_tokens_per_call: int = 500
    # Max concurrent reserved (in-flight) model calls — complements OASIS semaphore.
    max_parallel_calls: int = 1
    calls: list[ModelCallRecord] = field(default_factory=list)
    limit_hit: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _reservations: dict[int, _Reservation] = field(default_factory=dict, repr=False)
    _next_reservation_id: int = field(default=1, repr=False)
    _reserved_tokens: int = field(default=0, repr=False)
    _reserved_calls: int = field(default=0, repr=False)
    _reserved_cost: float = field(default=0.0, repr=False)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def estimated_cost_usd(self) -> float:
        return round(sum(c.estimated_cost_usd for c in self.calls), 6)

    @property
    def reserved_tokens(self) -> int:
        return self._reserved_tokens

    @property
    def reserved_calls(self) -> int:
        return self._reserved_calls

    @property
    def tokens_exceeded(self) -> bool:
        return self.total_tokens > self.max_tokens

    def remaining_calls(self) -> int:
        with self._lock:
            return max(0, self.max_calls - self.call_count - self._reserved_calls)

    def remaining_tokens(self) -> int:
        with self._lock:
            return max(0, self.max_tokens - self.total_tokens - self._reserved_tokens)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000.0 * self.price_input_per_1m
            + completion_tokens / 1_000_000.0 * self.price_output_per_1m
        )

    def conservative_call_budget(self) -> tuple[int, int]:
        """Assume a large prompt + full completion budget before each call.

        ``max_tokens_per_call`` is the *completion* cap, not a prompt size. Live
        Phase-3.5 prompts were ~1.7k–4.4k tokens — treating 500 as the prompt
        floor let parallel preflights under-reserve and blow past the run cap.
        """
        completion = max(1, int(self.max_tokens_per_call))
        # Prompt floor: several completion-budgets, at least 4k — never the
        # completion cap alone.
        prompt = max(4096, completion * 8)
        return prompt, completion

    def _set_limit_hit(self, message: str) -> None:
        if not self.limit_hit:
            self.limit_hit = message

    def _projected_tokens_locked(
        self, estimated_prompt_tokens: int, estimated_completion_tokens: int
    ) -> int:
        return (
            self.total_tokens
            + self._reserved_tokens
            + int(estimated_prompt_tokens)
            + int(estimated_completion_tokens)
        )

    def _normalize_estimates(
        self,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
        *,
        conservative: bool,
    ) -> tuple[int, int]:
        if conservative and not (
            estimated_prompt_tokens or estimated_completion_tokens
        ):
            return self.conservative_call_budget()
        if conservative:
            c_prompt, c_completion = self.conservative_call_budget()
            if not estimated_prompt_tokens:
                estimated_prompt_tokens = c_prompt
            if not estimated_completion_tokens:
                estimated_completion_tokens = c_completion
        return int(estimated_prompt_tokens), int(estimated_completion_tokens)

    def check_can_call(
        self,
        *,
        estimated_prompt_tokens: int = 0,
        estimated_completion_tokens: int = 0,
        conservative: bool = True,
    ) -> None:
        """Non-reserving probe (tests / readiness). Prefer ``reserve_call`` pre-API."""
        with self._lock:
            self._ensure_can_allocate_locked(
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_completion_tokens=estimated_completion_tokens,
                conservative=conservative,
                for_reservation=False,
            )

    def reserve_call(
        self,
        *,
        estimated_prompt_tokens: int = 0,
        estimated_completion_tokens: int = 0,
        conservative: bool = True,
    ) -> int:
        """Atomically reserve call/token/cost budget before a provider call.

        Parallel agents must not share the same remaining budget snapshot —
        reservation closes the check-then-act race that overshot 20k→31k.
        """
        with self._lock:
            prompt, completion = self._ensure_can_allocate_locked(
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_completion_tokens=estimated_completion_tokens,
                conservative=conservative,
                for_reservation=True,
            )
            cost = self.estimate_cost(prompt, completion)
            rid = self._next_reservation_id
            self._next_reservation_id += 1
            self._reservations[rid] = _Reservation(prompt, completion, cost)
            self._reserved_tokens += prompt + completion
            self._reserved_calls += 1
            self._reserved_cost += cost
            return rid

    def release_reservation(self, reservation_id: int) -> None:
        """Drop a reservation when the call never produced billable usage."""
        with self._lock:
            self._drop_reservation_locked(reservation_id)

    def commit_reservation(
        self,
        reservation_id: int,
        *,
        agent_id: int | None,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        ok: bool = True,
        error: str | None = None,
        source: str = "llm_api",
    ) -> ModelCallRecord:
        """Settle a reservation with actual usage (may be lower or slightly higher)."""
        with self._lock:
            self._drop_reservation_locked(reservation_id)
            return self._record_locked(
                agent_id=agent_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
                ok=ok,
                error=error,
                source=source,
            )

    def _drop_reservation_locked(self, reservation_id: int) -> None:
        reserved = self._reservations.pop(reservation_id, None)
        if reserved is None:
            return
        self._reserved_tokens = max(
            0, self._reserved_tokens - reserved.prompt_tokens - reserved.completion_tokens
        )
        self._reserved_calls = max(0, self._reserved_calls - 1)
        self._reserved_cost = max(0.0, self._reserved_cost - reserved.cost_usd)

    def _ensure_can_allocate_locked(
        self,
        *,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
        conservative: bool,
        for_reservation: bool,
    ) -> tuple[int, int]:
        if self.limit_hit:
            raise LimitExceeded(self.limit_hit)
        if self.call_count + self._reserved_calls >= self.max_calls:
            self._set_limit_hit(
                f"API-Call-Limit erreicht (MAX_LLM_CALLS_PER_RUN={self.max_calls})."
            )
            raise LimitExceeded(self.limit_hit)
        if self.total_tokens >= self.max_tokens:
            self._set_limit_hit(
                f"Token-Limit erreicht (MAX_TOKENS_PER_RUN={self.max_tokens})."
            )
            raise LimitExceeded(self.limit_hit)
        if self.max_cost_usd > 0 and self.estimated_cost_usd >= self.max_cost_usd:
            self._set_limit_hit(
                f"Kosten-Limit erreicht (MAX_COST_USD_PER_RUN={self.max_cost_usd})."
            )
            raise LimitExceeded(self.limit_hit)

        # Concurrency is capped by OasisEnv llm_semaphore (engine passes
        # max_parallel_llm_calls). Budget safety is the atomic reservation below.
        _ = for_reservation  # API symmetry with reserve_call vs check_can_call

        prompt, completion = self._normalize_estimates(
            estimated_prompt_tokens,
            estimated_completion_tokens,
            conservative=conservative,
        )

        if prompt or completion:
            projected_tokens = self._projected_tokens_locked(prompt, completion)
            if projected_tokens > self.max_tokens:
                self._set_limit_hit(
                    f"Token-Limit würde überschritten "
                    f"(MAX_TOKENS_PER_RUN={self.max_tokens}; "
                    f"konservative Schätzung / Reservierung)."
                )
                raise LimitExceeded(self.limit_hit)

        if self.max_cost_usd > 0 and (prompt or completion):
            projected_cost = (
                self.estimated_cost_usd
                + self._reserved_cost
                + self.estimate_cost(prompt, completion)
            )
            if projected_cost > self.max_cost_usd:
                self._set_limit_hit(
                    f"Kosten-Limit würde überschritten "
                    f"(MAX_COST_USD_PER_RUN={self.max_cost_usd}; "
                    f"konservative Schätzung — Provider-Abrechnung kann abweichen)."
                )
                raise LimitExceeded(self.limit_hit)

        return prompt, completion

    def record(
        self,
        *,
        agent_id: int | None,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        ok: bool = True,
        error: str | None = None,
        source: str = "llm_api",
    ) -> ModelCallRecord:
        with self._lock:
            return self._record_locked(
                agent_id=agent_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
                ok=ok,
                error=error,
                source=source,
            )

    def _record_locked(
        self,
        *,
        agent_id: int | None,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        ok: bool = True,
        error: str | None = None,
        source: str = "llm_api",
    ) -> ModelCallRecord:
        total = int(prompt_tokens) + int(completion_tokens)
        rec = ModelCallRecord(
            call_id=len(self.calls) + 1,
            ts=_utc_now(),
            agent_id=agent_id,
            model=model or self.model_name,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total,
            estimated_cost_usd=round(
                self.estimate_cost(int(prompt_tokens), int(completion_tokens)), 8
            ),
            ok=ok,
            error=error,
            source=source,
        )
        self.calls.append(rec)
        if self.total_tokens > self.max_tokens:
            self._set_limit_hit(
                f"Token-Budget überschritten "
                f"(MAX_TOKENS_PER_RUN={self.max_tokens}; "
                f"ist={self.total_tokens})."
            )
        elif self.total_tokens >= self.max_tokens:
            self._set_limit_hit(
                f"Token-Limit erreicht (MAX_TOKENS_PER_RUN={self.max_tokens})."
            )
        if self.call_count >= self.max_calls:
            self._set_limit_hit(
                f"API-Call-Limit erreicht (MAX_LLM_CALLS_PER_RUN={self.max_calls})."
            )
        if (
            self.max_cost_usd > 0
            and self.estimated_cost_usd >= self.max_cost_usd
        ):
            self._set_limit_hit(
                f"Kosten-Limit erreicht (MAX_COST_USD_PER_RUN={self.max_cost_usd})."
            )
        return rec

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "max_llm_calls_per_run": self.max_calls,
            "max_tokens_per_run": self.max_tokens,
            "max_tokens_per_call": self.max_tokens_per_call,
            "max_cost_usd_per_run": self.max_cost_usd,
            "max_parallel_llm_calls": self.max_parallel_calls,
            "remaining_calls": self.remaining_calls(),
            "remaining_tokens": self.remaining_tokens(),
            "reserved_tokens": self.reserved_tokens,
            "reserved_calls": self.reserved_calls,
            "tokens_exceeded": self.tokens_exceeded,
            "limit_hit": self.limit_hit,
            "model": self.model_name,
            "price_input_per_1m_usd": self.price_input_per_1m,
            "price_output_per_1m_usd": self.price_output_per_1m,
            "cost_estimate_disclaimer": COST_ESTIMATE_DISCLAIMER,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "calls": [c.to_dict() for c in self.calls],
        }
