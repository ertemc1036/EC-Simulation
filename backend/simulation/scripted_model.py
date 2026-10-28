"""Deterministic BaseModelBackend for tests — drives real OASIS LLMAction without paid APIs.

NOT a production substitute for OpenAI. Used only when injected (unit/integration tests).
Returns tool_calls so SocialAgent.perform_action_by_llm executes real platform actions
after observing the environment (posts from other agents).

Phase 5: Round 2+ prefers comment reactions / local →Kommentar# refs so
agrees_with / disagrees_with / responds_to are evidenced offline.

Phase 6: SocialInteractionScheduler chooses partners/targets without self-as-partner;
scales to offline 10/25/50 society runs.
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

from backend.simulation.social_scheduler import SocialInteractionScheduler


_POST_ID_RE = re.compile(r'"post_id"\s*:\s*(\d+)')
_POST_HASH_RE = re.compile(r"Post#(\d+)", re.IGNORECASE)
_COMMENT_HASH_RE = re.compile(r"Kommentar#(\d+)", re.IGNORECASE)
_COMMENT_ID_RE = re.compile(r'"comment_id"\s*:\s*(\d+)')
_COMMENT_ID_HINT_RE = re.compile(r"comment_id\s*=\s*(\d+)", re.IGNORECASE)
_USER_ID_RE = re.compile(r'"user_id"\s*:\s*(\d+)')
_ROUND_RE = re.compile(r"Runde:\s*(\d+)", re.IGNORECASE)
_AGENT_HINT_RE = re.compile(r"agent_id\s*[:=]\s*(\d+)", re.IGNORECASE)


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


def _extract_user_ids(blob: str) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for match in _USER_ID_RE.findall(blob):
        uid = int(match)
        if uid not in seen:
            seen.add(uid)
            ids.append(uid)
    return ids


def _extract_round(blob: str) -> int:
    m = _ROUND_RE.search(blob or "")
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            return 1
    return 1


def _posts_from_ids(ids: list[int], authors: list[int]) -> list[dict[str, Any]]:
    """Synthetic post dicts when only ids are visible in the prompt blob."""
    out: list[dict[str, Any]] = []
    for i, pid in enumerate(ids):
        author = authors[i % len(authors)] if authors else i
        out.append({"post_id": pid, "user_id": author})
    return out


def _comments_from_ids(ids: list[int], authors: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        author = authors[i % len(authors)] if authors else i
        out.append({"comment_id": cid, "user_id": author, "post_id": 1})
    return out


class ScriptedReactiveModel(BaseModelBackend):
    """Observes prompt content; emits real OASIS tools: post/comment/like/dislike."""

    # Shared scheduler across agents in one process (tests inject one engine run).
    shared_scheduler: SocialInteractionScheduler | None = None

    def __init__(self, *, agent_id: int = 0, label: str = "scripted") -> None:
        super().__init__(model_type=ModelType.GPT_4O_MINI, model_config_dict={})
        self.agent_id = agent_id
        self.label = label
        self.call_count = 0
        self._fail_after: int | None = None
        self._force_error: Exception | None = None
        self._scheduler = ScriptedReactiveModel.shared_scheduler or SocialInteractionScheduler(
            scenario=label
        )

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
        user_ids = _extract_user_ids(blob)
        round_n = _extract_round(blob)
        self.call_count += 1
        n = self.call_count

        # Candidate agents visible in feed (+ self) for partner selection.
        agent_pool = sorted(set(user_ids) | {self.agent_id})
        if len(agent_pool) < 2:
            # Bootstrap: assume neighboring ids exist in multi-agent society.
            agent_pool = sorted({self.agent_id, (self.agent_id + 1) % max(2, self.agent_id + 2)})

        posts = _posts_from_ids(ids, user_ids or agent_pool)
        comments = _comments_from_ids(comment_ids, user_ids or agent_pool)
        plan = self._scheduler.plan(
            agent_id=self.agent_id,
            round_n=round_n,
            agent_ids=agent_pool,
            posts=posts,
            comments=comments,
            call_index=n,
        )

        # Never treat self as inter-agent partner evidence in content.
        assert plan.partner_agent_id is None or plan.partner_agent_id != self.agent_id

        if not ids or plan.target_post_id is None:
            return (
                "create_post",
                {
                    "content": (
                        f"[{self.label}:{self.agent_id}] Beitrag #{n} — "
                        f"mein Ziel leitet diese Reaktion auf das Szenario."
                    )
                },
            )

        target = int(plan.target_post_id)

        if not comment_ids or plan.target_comment_id is None:
            return (
                "create_comment",
                {
                    "post_id": target,
                    "content": (
                        f"[{self.label}:{self.agent_id}] Reply/Kommentar #{n} "
                        f"auf Post {target} (Partner "
                        f"{plan.partner_agent_id if plan.partner_agent_id is not None else 'n/a'}; "
                        f"kein parent_comment_id)."
                    ),
                },
            )

        cid = int(plan.target_comment_id)
        hint = plan.action_hint

        if round_n >= 2:
            if hint == "like_comment":
                return ("like_comment", {"comment_id": cid})
            if hint == "dislike_comment":
                return ("dislike_comment", {"comment_id": cid})
            if hint == "create_comment_ref":
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
                            f"{shift} (Runde {round_n}, Partner "
                            f"{plan.partner_agent_id}, kein parent_comment_id)."
                        ),
                    },
                )
            return ("like_post", {"post_id": target})

        if hint == "create_comment":
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
        if hint == "like_post":
            return ("like_post", {"post_id": target})
        if hint == "dislike_post":
            return ("dislike_post", {"post_id": target})
        if hint == "like_comment":
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
        # Cap growth so large societies (50) stay within offline token budgets.
        prompt_tokens = 40 + min(12, 2 + self.agent_id % 10)
        completion_tokens = 20 + min(8, self.call_count)
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
