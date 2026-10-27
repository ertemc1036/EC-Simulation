"""Wrap a CAMEL BaseModelBackend to enforce cost limits and log usage."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from camel.models import BaseModelBackend
from camel.types import ChatCompletion, ChatCompletionChunk
from openai import AsyncStream, Stream
from pydantic import BaseModel

from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.prompt_debug import PromptDebugBuffer

# CAMEL BaseModelBackend.token_limit reads model_config_dict["max_tokens"].
# That key is the OpenAI *completion* cap (MAX_TOKENS_PER_CALL=500), not the
# chat context window. Phase-3.5 system prompts exceed 500 tokens, so ChatAgent
# get_context() raises and astep terminates with max_tokens_exceeded — zero API
# calls, empty tool_calls, decision_log still ok. Keep completion max_tokens in
# config; expose a real context window here for memory/context creators.
DEFAULT_CONTEXT_TOKEN_LIMIT = 128_000

# Inflate prompt estimates so parallel/serial pre-reservations stay under the
# run ceiling even when char-heuristics or counters run a bit low.
_PROMPT_ESTIMATE_MARGIN = 1.25
_PROMPT_ESTIMATE_PAD = 32


def _usage_from_response(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


def resolve_context_token_limit(model: BaseModelBackend) -> int:
    """Context window for ChatAgent memory — never the completion max_tokens."""
    mt = getattr(model, "model_type", None)
    native = getattr(mt, "token_limit", None) if mt is not None else None
    if isinstance(native, int) and native >= 4096:
        return int(native)
    return DEFAULT_CONTEXT_TOKEN_LIMIT


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
        context_token_limit: int | None = None,
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
        self._context_token_limit = int(
            context_token_limit
            if context_token_limit is not None
            else resolve_context_token_limit(inner)
        )

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

    @property
    def token_limit(self) -> int:
        """Chat context budget — independent of completion max_tokens."""
        return self._context_token_limit

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
        """Conservative prompt estimate for pre-call reservation."""
        counted: int | None = None
        try:
            counter = self.token_counter
            if counter is not None and hasattr(counter, "count_tokens_from_messages"):
                counted = int(counter.count_tokens_from_messages(messages))
        except Exception:
            counted = None

        total_chars = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif content is not None:
                total_chars += len(str(content))
        # denser than //4 — German Phase-3.5 prompts under-counted at //4
        heuristic = max(1, total_chars // 3)
        base = max(counted or 0, heuristic)
        return max(1, int(base * _PROMPT_ESTIMATE_MARGIN) + _PROMPT_ESTIMATE_PAD)

    def _reserve(self, messages: List[Dict[str, Any]]) -> int:
        # Full completion budget + conservative prompt — atomic with peers.
        est_prompt = self._estimate_prompt_tokens(messages)
        est_completion = max(1, int(self._tracker.max_tokens_per_call))
        return self._tracker.reserve_call(
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
        reservation_id = self._reserve(messages)
        self._capture_prompts(messages)
        try:
            response = self._inner._run(messages, response_format, tools)
        except Exception as exc:
            self._tracker.release_reservation(reservation_id)
            self._record_error(exc)
            raise
        prompt, completion = _usage_from_response(response)
        self._tracker.commit_reservation(
            reservation_id,
            agent_id=self._agent_id,
            prompt_tokens=prompt,
            completion_tokens=completion,
            model=str(self.model_type),
            ok=True,
            source=self._source,
        )
        return response

    async def _arun(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        reservation_id = self._reserve(messages)
        self._capture_prompts(messages)
        try:
            response = await self._inner._arun(messages, response_format, tools)
        except Exception as exc:
            self._tracker.release_reservation(reservation_id)
            self._record_error(exc)
            raise
        prompt, completion = _usage_from_response(response)
        self._tracker.commit_reservation(
            reservation_id,
            agent_id=self._agent_id,
            prompt_tokens=prompt,
            completion_tokens=completion,
            model=str(self.model_type),
            ok=True,
            source=self._source,
        )
        return response
