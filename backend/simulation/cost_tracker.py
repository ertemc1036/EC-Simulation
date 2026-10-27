"""Hard limits and accounting for real LLM API calls (separate from sim actions)."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


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
class CostTracker:
    max_calls: int
    max_tokens: int
    price_input_per_1m: float
    price_output_per_1m: float
    model_name: str = ""
    calls: list[ModelCallRecord] = field(default_factory=list)
    limit_hit: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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

    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.call_count)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000.0 * self.price_input_per_1m
            + completion_tokens / 1_000_000.0 * self.price_output_per_1m
        )

    def check_can_call(self, *, estimated_prompt_tokens: int = 0) -> None:
        with self._lock:
            if self.limit_hit:
                raise LimitExceeded(self.limit_hit)
            if self.call_count >= self.max_calls:
                self.limit_hit = (
                    f"API-Call-Limit erreicht (MAX_LLM_CALLS_PER_RUN={self.max_calls})."
                )
                raise LimitExceeded(self.limit_hit)
            if self.total_tokens >= self.max_tokens:
                self.limit_hit = (
                    f"Token-Limit erreicht (MAX_TOKENS_PER_RUN={self.max_tokens})."
                )
                raise LimitExceeded(self.limit_hit)
            if (
                estimated_prompt_tokens
                and self.total_tokens + estimated_prompt_tokens > self.max_tokens
            ):
                self.limit_hit = (
                    f"Token-Limit würde überschritten "
                    f"(MAX_TOKENS_PER_RUN={self.max_tokens})."
                )
                raise LimitExceeded(self.limit_hit)

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
        total = int(prompt_tokens) + int(completion_tokens)
        with self._lock:
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
            if self.total_tokens >= self.max_tokens and not self.limit_hit:
                self.limit_hit = (
                    f"Token-Limit erreicht (MAX_TOKENS_PER_RUN={self.max_tokens})."
                )
            if self.call_count >= self.max_calls and not self.limit_hit:
                self.limit_hit = (
                    f"API-Call-Limit erreicht (MAX_LLM_CALLS_PER_RUN={self.max_calls})."
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
            "remaining_calls": self.remaining_calls(),
            "remaining_tokens": self.remaining_tokens(),
            "limit_hit": self.limit_hit,
            "model": self.model_name,
            "price_input_per_1m_usd": self.price_input_per_1m,
            "price_output_per_1m_usd": self.price_output_per_1m,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "calls": [c.to_dict() for c in self.calls],
        }
