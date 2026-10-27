"""Deterministic BaseModelBackend for tests — drives real OASIS LLMAction without paid APIs.

NOT a production substitute for OpenAI. Used only when injected (unit/integration tests).
Returns tool_calls so SocialAgent.perform_action_by_llm executes real platform actions
after observing the environment (posts from other agents).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Type, Union

from camel.models import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    ModelType,
)
from camel.utils import OpenAITokenCounter
from openai import AsyncStream, Stream
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel


_POST_ID_RE = re.compile(r'"post_id"\s*:\s*(\d+)')


class ScriptedReactiveModel(BaseModelBackend):
    """Observes prompt content and emits create_post / create_comment / like_post."""

    def __init__(self, *, agent_id: int = 0, label: str = "scripted") -> None:
        super().__init__(model_type=ModelType.GPT_4O_MINI, model_config_dict={})
        self.agent_id = agent_id
        self.label = label
        self.call_count = 0
        self._fail_after: int | None = None
        self._force_error: Exception | None = None

    def fail_after(self, n: int) -> None:
        self._fail_after = n

    def force_error(self, exc: Exception) -> None:
        self._force_error = exc

    @property
    def token_counter(self):
        if self._token_counter is None:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    def _decide(self, messages: List[Dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        blob = " ".join(str(m.get("content") or "") for m in messages)
        ids = [int(x) for x in _POST_ID_RE.findall(blob)]
        self.call_count += 1
        n = self.call_count
        # Rotate actions so multi-agent runs produce varied interactions.
        if not ids:
            return (
                "create_post",
                {
                    "content": (
                        f"[{self.label}:{self.agent_id}] Beitrag #{n} — "
                        f"mein Ziel leitet diese Reaktion auf das Szenario."
                    )
                },
            )
        target = ids[min(self.agent_id, len(ids) - 1)]
        kind = n % 3
        if kind == 1:
            return (
                "create_comment",
                {
                    "post_id": target,
                    "content": (
                        f"[{self.label}:{self.agent_id}] Reaktion #{n} auf Post {target}."
                    ),
                },
            )
        if kind == 2:
            return ("like_post", {"post_id": target})
        return (
            "create_post",
            {
                "content": (
                    f"[{self.label}:{self.agent_id}] Follow-up #{n} nach Beobachtung "
                    f"der Posts {ids[:5]}."
                )
            },
        )

    def _completion(
        self, name: str, args: dict[str, Any]
    ) -> ChatCompletion:
        tool_call = ChatCompletionMessageToolCall(
            id=f"scripted_{self.agent_id}_{self.call_count}",
            type="function",
            function=Function(name=name, arguments=json.dumps(args)),
        )
        # Synthetic but non-zero usage so cost accounting is exercised in tests.
        prompt_tokens = 40 + 10 * self.agent_id
        completion_tokens = 20 + self.call_count
        return ChatCompletion(
            id=f"scripted-{self.agent_id}-{self.call_count}",
            model="scripted-reactive",
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[tool_call],
                    ),
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def _run(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        if self._force_error is not None:
            raise self._force_error
        if self._fail_after is not None and self.call_count >= self._fail_after:
            raise RuntimeError(
                f"Scripted model forced failure after {self._fail_after} calls"
            )
        name, args = self._decide(messages)
        return self._completion(name, args)

    async def _arun(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        return self._run(messages, response_format, tools)
