"""Deterministic BaseModelBackend for tests — drives real OASIS LLMAction without paid APIs.

NOT a production substitute for OpenAI. Used only when injected (unit/integration tests).
Returns tool_calls so SocialAgent.perform_action_by_llm executes real platform actions
after observing the environment (posts from other agents).

Phase 5: Round 2+ prefers comment reactions / local →Kommentar# refs so
agrees_with / disagrees_with / responds_to are evidenced offline.
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
_POST_HASH_RE = re.compile(r"Post#(\d+)", re.IGNORECASE)
_COMMENT_HASH_RE = re.compile(r"Kommentar#(\d+)", re.IGNORECASE)
_COMMENT_ID_RE = re.compile(r'"comment_id"\s*:\s*(\d+)')
_COMMENT_ID_HINT_RE = re.compile(r"comment_id\s*=\s*(\d+)", re.IGNORECASE)
_ROUND_RE = re.compile(r"Runde:\s*(\d+)", re.IGNORECASE)


def _extract_post_ids(blob: str) -> list[int]:
    """Collect post ids from OASIS JSON and our FEED-KONTEXT ``Post#N`` lines."""
    ids: list[int] = []
    seen: set[int] = set()
    for match in _POST_ID_RE.findall(blob):
        pid = int(match)
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
    for match in _POST_HASH_RE.findall(blob):
        pid = int(match)
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


def _extract_comment_ids(blob: str) -> list[int]:
    """Collect comment ids from FEED-KONTEXT / JSON for like_comment tools."""
    ids: list[int] = []
    seen: set[int] = set()
    for pattern in (_COMMENT_ID_RE, _COMMENT_HASH_RE, _COMMENT_ID_HINT_RE):
        for match in pattern.findall(blob):
            cid = int(match)
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


def _extract_round(blob: str) -> int:
    m = _ROUND_RE.search(blob or "")
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            return 1
    return 1


class ScriptedReactiveModel(BaseModelBackend):
    """Observes prompt content; emits real OASIS tools: post/comment/like/dislike."""

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
        ids = _extract_post_ids(blob)
        comment_ids = _extract_comment_ids(blob)
        round_n = _extract_round(blob)
        self.call_count += 1
        n = self.call_count
        # Rotate actions so multi-agent runs produce varied real platform interactions.
        # Prefer comment/like/dislike when feed ids are visible (Phase 3.5 / 4).
        # Reply = create_comment on an observed post (OASIS has no parent_comment_id).
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
        target = ids[min(self.agent_id % len(ids), len(ids) - 1)]

        # Seed a flat thread first so R2+ can react to Kommentar# IDs.
        if not comment_ids:
            return (
                "create_comment",
                {
                    "post_id": target,
                    "content": (
                        f"[{self.label}:{self.agent_id}] Reply/Kommentar #{n} "
                        f"auf Post {target} (kein parent_comment_id)."
                    ),
                },
            )

        # Phase 5: Round 2+ — comment reactions + local refs (Meinungsdynamik).
        if round_n >= 2:
            cid = comment_ids[
                min(self.agent_id % len(comment_ids), len(comment_ids) - 1)
            ]
            kind_r2 = (n + self.agent_id) % 4
            if kind_r2 == 0:
                return ("like_comment", {"comment_id": cid})
            if kind_r2 == 1:
                return ("dislike_comment", {"comment_id": cid})
            if kind_r2 == 2:
                # Local comment reference inside flat create_comment content.
                shift = (
                    "Ich ändere meine Position nach dem Kommentar — Kompromiss."
                    if (n + self.agent_id) % 2 == 0
                    else "Ich halte dagegen und widerspreche klar."
                )
                return (
                    "create_comment",
                    {
                        "post_id": target,
                        "content": (
                            f"[{self.label}:{self.agent_id}] →Kommentar#{cid} "
                            f"{shift} (Runde {round_n}, kein parent_comment_id)."
                        ),
                    },
                )
            return ("like_post", {"post_id": target})

        kind = (n + self.agent_id) % 5
        if kind in {0, 3}:
            return (
                "create_comment",
                {
                    "post_id": target,
                    "content": (
                        f"[{self.label}:{self.agent_id}] Reply/Kommentar #{n} "
                        f"auf Post {target} (kein parent_comment_id)."
                    ),
                },
            )
        if kind == 1:
            return ("like_post", {"post_id": target})
        if kind == 2:
            return ("dislike_post", {"post_id": target})
        # kind == 4 with comments: like/dislike a visible comment.
        cid = comment_ids[min(self.agent_id % len(comment_ids), len(comment_ids) - 1)]
        if (n + self.agent_id) % 2 == 0:
            return ("like_comment", {"comment_id": cid})
        return ("dislike_comment", {"comment_id": cid})

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
