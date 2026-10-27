"""Wrap a CAMEL BaseModelBackend to enforce cost limits and log usage."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from camel.models import BaseModelBackend
from camel.types import ChatCompletion, ChatCompletionChunk
from openai import AsyncStream, Stream
from pydantic import BaseModel

from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.prompt_debug import PromptDebugBuffer


def _usage_from_response(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


class TrackedModelBackend(BaseModelBackend):
    """Delegates inference to an inner backend; never bypasses CostTracker limits."""

    def __init__(
        self,
        inner: BaseModelBackend,
        tracker: CostTracker,
        *,
        agent_id: int | None = None,
        source: str = "llm_api",
        prompt_debug: PromptDebugBuffer | None = None,
    ) -> None:
        # Mirror inner config so CAMEL tooling sees a normal backend.
        super().__init__(
            model_type=inner.model_type,
            model_config_dict=getattr(inner, "model_config_dict", None) or {},
            api_key=getattr(inner, "_api_key", None),
            url=getattr(inner, "_url", None),
            token_counter=getattr(inner, "_token_counter", None),
            timeout=getattr(inner, "_timeout", None),
            # Prefer inner setting; default 0 (no auto-retries) for cost safety.
            max_retries=getattr(inner, "_max_retries", 0),
        )
        self._inner = inner
        self._tracker = tracker
        self._agent_id = agent_id
        self._source = source
        self._prompt_debug = prompt_debug

    def _capture_prompts(self, messages: List[Dict[str, Any]]) -> None:
        if self._prompt_debug is None:
            return
        # Fingerprints + markers only — never persists api_key / Authorization.
        self._prompt_debug.record(
            messages, agent_id=self._agent_id, source=self._source
        )
    @property
    def token_counter(self):
        return self._inner.token_counter

    def _record_ok(self, response: Any) -> None:
        prompt, completion = _usage_from_response(response)
        self._tracker.record(
            agent_id=self._agent_id,
            prompt_tokens=prompt,
            completion_tokens=completion,
            model=str(self.model_type),
            ok=True,
            source=self._source,
        )

    def _record_error(self, exc: BaseException) -> None:
        if isinstance(exc, LimitExceeded):
            return
        self._tracker.record(
            agent_id=self._agent_id,
            prompt_tokens=0,
            completion_tokens=0,
            model=str(self.model_type),
            ok=False,
            error=str(exc),
            source=self._source,
        )

    def _estimate_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Rough char-based prompt estimate; CostTracker still applies full budget."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif content is not None:
                total_chars += len(str(content))
        # ~4 chars/token heuristic, never below 1.
        return max(1, total_chars // 4)

    def _preflight(self, messages: List[Dict[str, Any]]) -> None:
        # Conservative gate: full completion budget + estimated/floor prompt.
        est_prompt = self._estimate_prompt_tokens(messages)
        est_completion = max(1, int(self._tracker.max_tokens_per_call))
        self._tracker.check_can_call(
            estimated_prompt_tokens=est_prompt,
            estimated_completion_tokens=est_completion,
            conservative=True,
        )

    def _run(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        self._preflight(messages)
        self._capture_prompts(messages)
        try:
            response = self._inner._run(messages, response_format, tools)
        except Exception as exc:
            self._record_error(exc)
            raise
        self._record_ok(response)
        # Post-call token ceiling (blocks subsequent calls)
        if self._tracker.total_tokens >= self._tracker.max_tokens:
            self._tracker.limit_hit = (
                f"Token-Limit erreicht "
                f"(MAX_TOKENS_PER_RUN={self._tracker.max_tokens})."
            )
        return response

    async def _arun(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        self._preflight(messages)
        self._capture_prompts(messages)
        try:
            response = await self._inner._arun(messages, response_format, tools)
        except Exception as exc:
            self._record_error(exc)
            raise
        self._record_ok(response)
        if self._tracker.total_tokens >= self._tracker.max_tokens:
            self._tracker.limit_hit = (
                f"Token-Limit erreicht "
                f"(MAX_TOKENS_PER_RUN={self._tracker.max_tokens})."
            )
        return response
