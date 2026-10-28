"""Deterministic BaseModelBackend for tests — drives real OASIS LLMAction without paid APIs.

NOT a production substitute for OpenAI. Used only when injected (unit/integration tests).
Returns tool_calls so SocialAgent.perform_action_by_llm executes real platform actions
after observing the environment (posts from other agents).

Phase 5: Round 2+ prefers comment reactions / local →Kommentar# refs so
agrees_with / disagrees_with / responds_to are evidenced offline.

Phase 6: SocialInteractionScheduler chooses partners/targets without self-as-partner;
scales to offline 10/25/50 society runs.

Phase 7: Multi-round opinion scenarios (bis 4 Runden) — establish → react →
shift → consolidate, with polarity markers for evidenced Meinungsdynamik.
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
from backend.simulation.topic_context import (
    detect_topic_domain,
    measure_label,
    synthetic_tag,
)


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
    # Shared scenario text so offline stances stay thesis-specific (Phase 7.1).
    shared_scenario: str = ""
    # Phase 7.3: shared discussion engine for anti-repetition (optional).
    shared_discussion: Any = None

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

    def _dedupe_content(self, content: str) -> str:
        disc = ScriptedReactiveModel.shared_discussion
        if disc is None or not hasattr(disc, "vary_scripted_body"):
            return content
        varied = disc.vary_scripted_body(
            self.agent_id, content, call_index=self.call_count
        )
        disc.register_statement(
            agent_id=self.agent_id,
            text=varied,
            evidence=f"scripted:{self.agent_id}:{self.call_count}",
            round_n=None,
        )
        return varied

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

        scenario = ScriptedReactiveModel.shared_scenario or self._scheduler.scenario or ""
        stance_seed = _scripted_stance_line(self.agent_id, round_n, scenario=scenario)
        syn = synthetic_tag()

        if not ids or plan.target_post_id is None:
            body = self._dedupe_content(
                f"{syn} [{self.label}:{self.agent_id}] Beitrag #{n} — "
                f"{stance_seed} "
                f"(Runde {round_n}, Szenario-Reaktion)."
            )
            return ("create_post", {"content": body})

        target = int(plan.target_post_id)

        if not comment_ids or plan.target_comment_id is None:
            body = self._dedupe_content(
                f"{syn} [{self.label}:{self.agent_id}] Reply/Kommentar #{n} "
                f"auf Post {target}. {stance_seed} "
                f"(Partner "
                f"{plan.partner_agent_id if plan.partner_agent_id is not None else 'n/a'}; "
                f"kein parent_comment_id)."
            )
            return (
                "create_comment",
                {"post_id": target, "content": body},
            )

        cid = int(plan.target_comment_id)
        hint = plan.action_hint

        # Phase 7 opinion arc: R2 react, R3 shift, R4 consolidate.
        if round_n >= 2:
            if hint == "like_comment":
                return ("like_comment", {"comment_id": cid})
            if hint == "dislike_comment":
                return ("dislike_comment", {"comment_id": cid})
            if hint == "create_comment_ref" or round_n >= 3:
                content = self._dedupe_content(
                    _scripted_ref_comment(
                        label=self.label,
                        agent_id=self.agent_id,
                        comment_id=cid,
                        round_n=round_n,
                        partner=plan.partner_agent_id,
                        call_index=n,
                        scenario=scenario,
                    )
                )
                return (
                    "create_comment",
                    {"post_id": target, "content": content},
                )
            return ("like_post", {"post_id": target})

        if hint == "create_comment":
            body = self._dedupe_content(
                f"{syn} [{self.label}:{self.agent_id}] Reply/Kommentar #{n} "
                f"auf Post {target}. {stance_seed} "
                f"(kein parent_comment_id)."
            )
            return (
                "create_comment",
                {"post_id": target, "content": body},
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


def _scripted_stance_line(agent_id: int, round_n: int, *, scenario: str = "") -> str:
    """Deterministic polarity seed — thesis-specific (mobility / income / generic)."""
    domain = detect_topic_domain(scenario) if scenario else "mobility"
    measure = measure_label(scenario, domain) if scenario else "Auto-Verbot"
    bucket = agent_id % 3

    if domain == "income":
        if round_n <= 1:
            if bucket == 0:
                return (
                    f"Ich bin für ein {measure} — Existenzsicherung ohne "
                    f"Sanktionsdruck stärkt Teilhabe."
                )
            if bucket == 1:
                return (
                    f"Ich lehne das {measure} ab — Finanzierung und Arbeitsanreize "
                    f"ungeklärt; ohne Gegenfinanzierung riskant."
                )
            return (
                "Zielkonflikt: Pilot mit klaren Abbruchkriterien und "
                "Übergangsfristen statt Sofort-Vollauszahlung."
            )
        if round_n == 2:
            if bucket == 0:
                return "stimme zu: ja zum Grundeinkommen, mit Messpunkten."
            if bucket == 1:
                return "widerspreche: gegen das Grundeinkommen ohne Finanzierungsnachweis."
            return (
                "einerseits Absicherung, andererseits Haushaltsgrenzen — "
                "Pilot mit Abbruchkriterien."
            )
        if agent_id % 2 == 0:
            return (
                "Ich ändere meine Position nach dem Kommentar — werde überzeugt: "
                "Piloten und Ausnahmeregelungen statt pauschaler Vollauszahlung."
            )
        return (
            "Ich halte dagegen und widerspreche klar: nein zur ungeprüften "
            "Vollauszahlung ohne Gegenfinanzierung."
        )

    if domain != "mobility":
        if round_n <= 1:
            if bucket == 0:
                return f"Ich bin für {measure} — Vorteile überwiegen bei Messpunkten."
            if bucket == 1:
                return f"Ich lehne {measure} ab — Kosten und Nebenwirkungen unzureichend belegt."
            return (
                "Zielkonflikt: schrittweise Einführung mit Ausnahmen statt Sofort-Vollumsetzung."
            )
        if round_n == 2:
            if bucket == 0:
                return f"stimme zu: ja zu {measure}, mit Messpunkten."
            if bucket == 1:
                return f"widerspreche: gegen {measure} ohne belastbare Evidenz."
            return "einerseits Nutzen, andererseits Risiken — Pilot nötig."
        if agent_id % 2 == 0:
            return (
                "Ich ändere meine Position nach dem Kommentar — werde überzeugt: "
                "Piloten und Ausnahmen statt Sofortumsetzung."
            )
        return "Ich halte dagegen und widerspreche klar: nein zur ungeprüften Sofortumsetzung."

    # Mobility (Berlin / Verbot) — preserve Phase 5–7 markers.
    if round_n <= 1:
        if bucket == 0:
            return "Ich bin für das Verbot — sinnvolles Verbot im Kerngebiet."
        if bucket == 1:
            return (
                "Ich lehne das Verbot ab — pauschale Verbote gefährden Arbeitsplätze; "
                "klare Ausnahmen und Lieferfenster nötig."
            )
        return (
            "Zielkonflikt: Ausnahmeregelungen und Übergangsfristen statt "
            "pauschalem Verbot."
        )
    if round_n == 2:
        if bucket == 0:
            return "stimme zu: ja zum Verbot, mit Messpunkten."
        if bucket == 1:
            return "widerspreche: gegen das Verbot ohne soziale Abfederung."
        return "einerseits Schutz, andererseits Lieferfenster — Ausnahme für Anwohner."
    if agent_id % 2 == 0:
        return (
            "Ich ändere meine Position nach dem Kommentar — werde überzeugt: "
            "Ausnahmeregelungen statt pauschalem Verbot."
        )
    return "Ich halte dagegen und widerspreche klar: nein zum pauschalen Verbot."


def _scripted_ref_comment(
    *,
    label: str,
    agent_id: int,
    comment_id: int,
    round_n: int,
    partner: int | None,
    call_index: int,
    scenario: str = "",
) -> str:
    """Local →Kommentar# content with agree/disagree/shift markers."""
    domain = detect_topic_domain(scenario) if scenario else "mobility"
    syn = synthetic_tag()
    if domain == "income":
        if round_n >= 4:
            if agent_id % 2 == 0:
                body = (
                    "stimme zu und teile diese Einschätzung — "
                    "Piloten und Ausnahmeregelungen bleiben zentral."
                )
            else:
                body = (
                    "widerspreche weiterhin: sehe das anders — "
                    "gegen das Grundeinkommen ohne Gegenfinanzierung."
                )
        elif round_n >= 3:
            if (call_index + agent_id) % 2 == 0:
                body = (
                    "Ich ändere meine Position nach dem Kommentar — "
                    "werde überzeugt von Piloten und Ausnahmeregelungen."
                )
            else:
                body = (
                    "Ich halte dagegen und widerspreche klar — "
                    "lehne das Grundeinkommen ab ohne Finanzierungsnachweis."
                )
        else:
            if (call_index + agent_id) % 2 == 0:
                body = (
                    "Ich ändere meine Position nach dem Kommentar — Kompromiss. "
                    "stimme zu und werde überzeugt von Piloten."
                )
            else:
                body = (
                    "Ich halte dagegen und widerspreche klar — "
                    "lehne das Grundeinkommen ab ohne Gegenfinanzierung."
                )
    else:
        if round_n >= 4:
            if agent_id % 2 == 0:
                body = (
                    "stimme zu und teile diese Einschätzung — "
                    "Ausnahmeregelungen bleiben zentral."
                )
            else:
                body = (
                    "widerspreche weiterhin: sehe das anders — "
                    "gegen das Verbot ohne Lieferfenster."
                )
        elif round_n >= 3:
            if (call_index + agent_id) % 2 == 0:
                body = (
                    "Ich ändere meine Position nach dem Kommentar — "
                    "werde überzeugt von Ausnahmeregelungen."
                )
            else:
                body = (
                    "Ich halte dagegen und widerspreche klar — "
                    "lehne das Verbot ab ohne soziale Abfederung."
                )
        else:
            # Round 2: keep Phase-5 shift markers so 2-round offline runs
            # still evidence position_changed / influenced_by.
            if (call_index + agent_id) % 2 == 0:
                body = (
                    "Ich ändere meine Position nach dem Kommentar — Kompromiss. "
                    "stimme zu und werde überzeugt von Ausnahmeregelungen."
                )
            else:
                body = (
                    "Ich halte dagegen und widerspreche klar — "
                    "lehne das Verbot ab ohne soziale Abfederung."
                )
    return (
        f"{syn} [{label}:{agent_id}] →Kommentar#{comment_id} {body} "
        f"(Runde {round_n}, Partner {partner}, kein parent_comment_id)."
    )
