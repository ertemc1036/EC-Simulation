"""Short research protocol: 3 synthetic personas, 2 interview turns, one synthesis.

Every provider call is counted against the live ceilings (10 calls, 20_000 tokens,
500 output tokens, $0.20). Estimates and prompts do not open a socket.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    settings,
)
from backend.research.integrity import find_banned_claims
from backend.research.provider import Completion, CredentialsMissing, ResearchProvider

PROTOCOL = "short_interview_v1"
SYNTHESIS_MODEL = "model_structured_json"
SYNTHESIS_FIXTURE = "templated_offline_fixture"
AI_PARTICIPANTS = 3
MAX_OUTPUT_CEILING = 500
MAX_TOKEN_CEILING = 20_000
MAX_COST_CEILING = 0.20

_QUESTION_CHARS = 360
_CONTEXT_CHARS = 480
_SOURCE_CHARS = 320
_FIELD_CHARS = 180
_STATEMENT_CHARS = 420

_SYSTEM = (
    "You are a synthetic research stand-in for NanoCrab Lab. "
    "Personas are not human subjects and not a sample. "
    "Reply with one JSON object only. "
    "Do not invent statistics, percentages, sample sizes, significance tests, or market coverage. "
    "Do not claim the persona is a real customer."
)


def live_ceilings() -> dict[str, float | int]:
    """Hard live ceilings. Environment may be lower, never higher."""
    return {
        "max_agents": min(LIVE_TEST_MAX_AGENTS, 5),
        "max_rounds": min(settings.max_simulation_steps, LIVE_TEST_MAX_ROUNDS, 2),
        "max_calls": min(settings.max_llm_calls_per_run, LIVE_TEST_MAX_LLM_CALLS, 10),
        "max_tokens": min(settings.max_tokens_per_run, MAX_TOKEN_CEILING),
        "max_output": min(max(1, settings.max_tokens_per_call), MAX_OUTPUT_CEILING),
        "max_cost": min(settings.max_cost_usd_per_run, MAX_COST_CEILING),
        "participants": AI_PARTICIPANTS,
    }


def output_cap() -> int:
    return int(live_ceilings()["max_output"])


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def estimate_messages(messages: list[dict[str, str]]) -> int:
    return estimate_tokens("\n".join(message.get("content") or "" for message in messages))


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000.0 * settings.price_input_per_1m
        + completion_tokens / 1_000_000.0 * settings.price_output_per_1m
    )


def clip(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


@dataclass
class StudyLedger:
    max_calls: int
    max_tokens: int
    max_output: int
    max_cost: float
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    models: list[str] = field(default_factory=list)
    usage_estimated: bool = False

    @classmethod
    def from_study(cls, study: dict) -> StudyLedger:
        ceilings = live_ceilings()
        prior = study.get("ai_usage") or {}
        return cls(
            max_calls=int(ceilings["max_calls"]),
            max_tokens=int(ceilings["max_tokens"]),
            max_output=int(ceilings["max_output"]),
            max_cost=float(ceilings["max_cost"]),
            calls=int(prior.get("calls") or 0),
            prompt_tokens=int(prior.get("prompt_tokens") or 0),
            completion_tokens=int(prior.get("completion_tokens") or 0),
            cost=float(prior.get("cost_usd") or 0),
            models=list(prior.get("models") or []),
            usage_estimated=bool(prior.get("usage_estimated")),
        )

    def mark(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
        }

    def delta(self, mark: dict[str, float | int]) -> dict[str, float | int]:
        return {
            "calls": self.calls - int(mark["calls"]),
            "prompt_tokens": self.prompt_tokens - int(mark["prompt_tokens"]),
            "completion_tokens": self.completion_tokens - int(mark["completion_tokens"]),
            "total_tokens": (self.prompt_tokens + self.completion_tokens)
            - (int(mark["prompt_tokens"]) + int(mark["completion_tokens"])),
            "cost_usd": round(self.cost - float(mark["cost"]), 6),
        }

    def can_spend(self, prompt_tokens: int, completion_tokens: int) -> str | None:
        if completion_tokens > self.max_output:
            return "budget"
        if self.calls + 1 > self.max_calls:
            return "budget"
        projected_tokens = self.prompt_tokens + self.completion_tokens + prompt_tokens + completion_tokens
        if projected_tokens > self.max_tokens:
            return "budget"
        if self.cost + cost_usd(prompt_tokens, completion_tokens) > self.max_cost + 1e-9:
            return "budget"
        return None

    def commit(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        usage_estimated: bool = False,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))
        self.cost = round(
            self.cost + cost_usd(max(0, int(prompt_tokens)), max(0, int(completion_tokens))),
            6,
        )
        if model and model not in self.models:
            self.models.append(model)
        if usage_estimated:
            self.usage_estimated = True

    def exceeded(self) -> bool:
        return (
            self.calls > self.max_calls
            or self.prompt_tokens + self.completion_tokens > self.max_tokens
            or self.cost > self.max_cost + 1e-9
        )

    def snapshot(self, *, partial: bool, stop_reason: str | None) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost, 6),
            "paid_llm_unlocked": False,
            "partial": partial,
            "stop_reason": stop_reason,
            "models": list(self.models),
            "usage_estimated": self.usage_estimated,
        }


def _messages(task_body: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": task_body},
    ]


def _brief_lines(study: dict) -> str:
    question = ((study.get("research_question") or {}).get("text")) or ""
    context = ((study.get("context") or {}).get("text")) or ""
    audience = study.get("audience") or {}
    sources = (study.get("context") or {}).get("sources") or []
    source_lines = []
    for source in sources[:3]:
        source_lines.append(
            f"Source {source.get('id')}: {clip(source.get('title') or '', 80)} — "
            f"{clip(source.get('text') or '', _SOURCE_CHARS)}"
        )
    source_block = "\n".join(source_lines) if source_lines else "Sources: none supplied."
    return (
        f"Audience: {clip(audience.get('name') or '', 80)}\n"
        f"Audience note: {clip(audience.get('description') or '', _FIELD_CHARS)}\n"
        f"Question: {clip(question, _QUESTION_CHARS)}\n"
        f"Context: {clip(context, _CONTEXT_CHARS)}\n"
        f"{source_block}"
    )


def roster_messages(study: dict) -> list[dict[str, str]]:
    body = (
        "TASK participant_roster\n"
        f"{_brief_lines(study)}\n"
        'Return {"participants":[{"name":"","role":"","perspective":"","tension":"","hope":""}]} '
        "with exactly 3 objects. Each field is one short sentence."
    )
    return _messages(body)


def opening_messages(study: dict, person: dict) -> list[dict[str, str]]:
    body = (
        "TASK interview_opening\n"
        f"You speak only as {clip(person.get('name') or '', 80)}, "
        f"a synthetic {clip(person.get('role') or '', 80)}. Not a real person.\n"
        f"Perspective: {clip(person.get('perspective') or '', _FIELD_CHARS)}\n"
        f"{_brief_lines(study)}\n"
        'Return {"text":"...","source_refs":[]}. '
        "Two or three sentences in the persona's voice. "
        "source_refs may list only source ids shown above."
    )
    return _messages(body)


def follow_up_messages(study: dict, person: dict, opening: dict) -> list[dict[str, str]]:
    body = (
        "TASK interview_follow_up\n"
        f"You still speak only as {clip(person.get('name') or '', 80)}, "
        f"a synthetic {clip(person.get('role') or '', 80)}.\n"
        f"Previous statement {opening.get('id')}: {clip(opening.get('text') or '', _STATEMENT_CHARS)}\n"
        f"{_brief_lines(study)}\n"
        "Name one condition or objection, then answer it in the persona's voice.\n"
        'Return {"text":"...","source_refs":[]}.'
    )
    return _messages(body)


def synthesis_messages(study: dict, statements: list[dict]) -> list[dict[str, str]]:
    question_id = (study.get("research_question") or {}).get("id") or ""
    lines = []
    for statement in statements:
        lines.append(
            f"- {statement.get('id')}: {clip(statement.get('text') or '', _STATEMENT_CHARS)}"
        )
    listed = "\n".join(lines) if lines else "- none"
    body = (
        "TASK findings_synthesis\n"
        f"Question id: {question_id}\n"
        f"Statements:\n{listed}\n"
        'Return {"findings":[{"title":"","text":"","statement_ids":[""],"human_validation":""}]}.\n'
        "Every statement_ids entry must be one of the statement ids listed above. "
        "Each finding is an interpretation of those statements, not a rate and not a market claim."
    )
    return _messages(body)


def _load_object(text: str) -> dict | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data


def _clean_sentence(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > limit:
        return None
    if find_banned_claims(text):
        return None
    return text


def parse_roster(text: str) -> list[dict[str, str]] | None:
    data = _load_object(text)
    if data is None:
        return None
    rows = data.get("participants")
    if not isinstance(rows, list) or len(rows) != AI_PARTICIPANTS:
        return None
    people: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        parsed = {
            "name": _clean_sentence(row.get("name"), 80),
            "role": _clean_sentence(row.get("role"), 80),
            "perspective": _clean_sentence(row.get("perspective"), 240),
            "tension": _clean_sentence(row.get("tension"), 240),
            "hope": _clean_sentence(row.get("hope"), 240),
        }
        if any(value is None for value in parsed.values()):
            return None
        people.append({key: value for key, value in parsed.items() if value is not None})
    return people


def parse_statement(text: str) -> dict[str, Any] | None:
    data = _load_object(text)
    if data is None:
        return None
    sentence = _clean_sentence(data.get("text"), 700)
    if sentence is None:
        return None
    refs = data.get("source_refs") or []
    if not isinstance(refs, list):
        return None
    clean_refs = [ref for ref in refs if isinstance(ref, str) and ref.strip()]
    return {"text": sentence, "source_refs": clean_refs}


@dataclass
class FindingParse:
    findings: list[dict[str, Any]]
    dropped: int


def parse_findings(text: str, statement_ids: set[str]) -> FindingParse | None:
    data = _load_object(text)
    if data is None:
        return None
    rows = data.get("findings")
    if not isinstance(rows, list) or not rows:
        return None
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        if not isinstance(row, dict):
            dropped += 1
            continue
        title = _clean_sentence(row.get("title"), 120)
        body = _clean_sentence(row.get("text"), 700)
        raw_ids = row.get("statement_ids")
        if title is None or body is None or not isinstance(raw_ids, list):
            dropped += 1
            continue
        ids = [sid for sid in raw_ids if isinstance(sid, str) and sid in statement_ids]
        if not ids:
            dropped += 1
            continue
        validation = _clean_sentence(row.get("human_validation"), 400)
        kept.append(
            {
                "title": title,
                "text": body,
                "statement_ids": ids,
                "human_validation": validation,
            }
        )
    if not kept:
        return None
    return FindingParse(findings=kept, dropped=dropped)


def guarded_complete(
    provider: ResearchProvider,
    messages: list[dict[str, str]],
    ledger: StudyLedger,
) -> tuple[Completion | None, str | None]:
    """Spend check happens before ``provider.complete``.

    A budget miss returns (None, "budget") and does not call the provider.
    """
    prompt_tokens = estimate_messages(messages)
    completion_budget = output_cap()
    blocked = ledger.can_spend(prompt_tokens, completion_budget)
    if blocked:
        return None, blocked
    try:
        completion = provider.complete(
            messages=messages,
            max_tokens=completion_budget,
            model=settings.openai_model,
        )
    except CredentialsMissing:
        return None, "missing_credentials"
    except Exception:
        ledger.commit(prompt_tokens=0, completion_tokens=0, model=getattr(provider, "name", ""))
        return None, "provider_error"
    ledger.commit(
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        model=completion.model,
        usage_estimated=completion.usage_estimated,
    )
    if completion.completion_tokens > ledger.max_output or ledger.exceeded():
        return completion, "budget"
    return completion, None


def run_roster(
    study: dict,
    provider: ResearchProvider,
    ledger: StudyLedger,
) -> tuple[list[dict[str, str]] | None, str | None]:
    completion, stop = guarded_complete(provider, roster_messages(study), ledger)
    if completion is None:
        return None, stop or "provider_error"
    people = parse_roster(completion.text)
    if people is None:
        return None, "invalid_model_response"
    return people, stop


def run_interview(
    study: dict,
    provider: ResearchProvider,
    ledger: StudyLedger,
    new_id,
) -> dict[str, Any]:
    statements: list[dict[str, Any]] = []
    partial = False
    stop_reason: str | None = None
    allowed_sources = {
        source.get("id")
        for source in ((study.get("context") or {}).get("sources") or [])
        if source.get("id")
    }
    for person in study.get("participants") or []:
        completion, stop = guarded_complete(provider, opening_messages(study, person), ledger)
        if completion is None:
            partial = True
            stop_reason = stop or "provider_error"
            break
        parsed = parse_statement(completion.text)
        if parsed is None:
            partial = True
            stop_reason = "invalid_model_response"
            break
        opening = {
            "id": new_id("stmt"),
            "kind": "synthetic_statement",
            "participant_id": person.get("id"),
            "text": parsed["text"],
            "source_refs": [ref for ref in parsed["source_refs"] if ref in allowed_sources],
            "prompt_task": "interview_opening",
            "turn": 1,
            "origin": "model",
        }
        statements.append(opening)
        if stop:
            partial = True
            stop_reason = stop
            break
        follow_completion, follow_stop = guarded_complete(
            provider,
            follow_up_messages(study, person, opening),
            ledger,
        )
        if follow_completion is None:
            partial = True
            stop_reason = follow_stop or "provider_error"
            break
        follow_parsed = parse_statement(follow_completion.text)
        if follow_parsed is None:
            partial = True
            stop_reason = "invalid_model_response"
            break
        statements.append(
            {
                "id": new_id("stmt"),
                "kind": "synthetic_statement",
                "participant_id": person.get("id"),
                "text": follow_parsed["text"],
                "source_refs": [ref for ref in follow_parsed["source_refs"] if ref in allowed_sources],
                "prompt_task": "interview_follow_up",
                "turn": 2,
                "origin": "model",
            }
        )
        if follow_stop:
            partial = True
            stop_reason = follow_stop
            break
    return {"statements": statements, "partial": partial, "stop_reason": stop_reason}


def run_synthesis(
    study: dict,
    statements: list[dict],
    provider: ResearchProvider,
    ledger: StudyLedger,
) -> tuple[FindingParse | None, str | None]:
    completion, stop = guarded_complete(
        provider,
        synthesis_messages(study, statements),
        ledger,
    )
    if completion is None:
        return None, stop or "provider_error"
    parsed = parse_findings(completion.text, {row["id"] for row in statements if row.get("id")})
    if parsed is None:
        return None, "invalid_model_response"
    if stop:
        return parsed, stop
    if parsed.dropped:
        return parsed, "invalid_model_response"
    return parsed, None


def _placeholder_people() -> list[dict[str, str]]:
    return [
        {
            "id": f"par_estimate_{index}",
            "name": "Synthetic persona",
            "role": "role named by the audience",
            "perspective": "A short perspective grounded in the brief.",
            "tension": "A tension a human still has to check.",
            "hope": "A condition named in the brief.",
        }
        for index in range(1, AI_PARTICIPANTS + 1)
    ]


def _step(name: str, calls: int, prompt_tokens: int) -> dict[str, Any]:
    completion_tokens = calls * output_cap()
    return {
        "name": name,
        "calls": calls,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": round(cost_usd(prompt_tokens, completion_tokens), 6),
    }


def build_cost_estimate(study: dict) -> dict[str, Any]:
    """Local arithmetic only. Does not construct a provider."""
    ceilings = live_ceilings()
    people = study.get("participants") or []
    if len(people) != AI_PARTICIPANTS:
        people = _placeholder_people()
    roster_prompt = estimate_messages(roster_messages(study))
    opening_prompt = sum(estimate_messages(opening_messages(study, person)) for person in people)
    follow_prompt = 0
    for person in people:
        opening = {
            "id": person.get("id") or "stmt_estimate",
            "text": "A short synthetic opening grounded in the brief.",
        }
        follow_prompt += estimate_messages(follow_up_messages(study, person, opening))
    sample_statements = [
        {
            "id": f"stmt_estimate_{index}",
            "text": "A short synthetic statement grounded in the brief.",
        }
        for index in range(1, 7)
    ]
    synthesis_prompt = estimate_messages(synthesis_messages(study, sample_statements))
    steps = [
        _step("participant_generation", 1, roster_prompt),
        _step("interview_opening", AI_PARTICIPANTS, opening_prompt),
        _step("interview_follow_up", AI_PARTICIPANTS, follow_prompt),
        _step("findings_synthesis", 1, synthesis_prompt),
    ]
    totals = {
        "calls": sum(step["calls"] for step in steps),
        "estimated_prompt_tokens": sum(step["estimated_prompt_tokens"] for step in steps),
        "estimated_completion_tokens": sum(step["estimated_completion_tokens"] for step in steps),
        "estimated_total_tokens": sum(step["estimated_total_tokens"] for step in steps),
        "estimated_cost_usd": round(sum(step["estimated_cost_usd"] for step in steps), 6),
    }
    recorded = study.get("ai_usage") or {}
    recorded_calls = int(recorded.get("calls") or 0)
    recorded_tokens = int(recorded.get("total_tokens") or 0)
    recorded_cost = float(recorded.get("cost_usd") or 0)
    fits = (
        totals["calls"] <= ceilings["max_calls"]
        and totals["estimated_total_tokens"] <= ceilings["max_tokens"]
        and output_cap() <= ceilings["max_output"]
        and totals["estimated_cost_usd"] <= float(ceilings["max_cost"]) + 1e-9
        and AI_PARTICIPANTS <= int(ceilings["max_agents"])
        and 2 <= int(ceilings["max_rounds"])
    )
    remaining_fits = (
        recorded_calls + totals["calls"] <= ceilings["max_calls"]
        and recorded_tokens + totals["estimated_total_tokens"] <= ceilings["max_tokens"]
        and recorded_cost + totals["estimated_cost_usd"] <= float(ceilings["max_cost"]) + 1e-9
    )
    if remaining_fits:
        note = "The full short protocol fits the remaining study budget."
    else:
        note = (
            "The remaining study budget cannot cover the full short protocol. "
            "A run stops at the ceiling and does not invent the missing turns."
        )
    return {
        "ok": True,
        "protocol": PROTOCOL,
        "participants": AI_PARTICIPANTS,
        "rounds": 2,
        "provider_call_made": False,
        "paid_llm_unlocked": False,
        "ceilings": {
            "max_agents": ceilings["max_agents"],
            "max_rounds": ceilings["max_rounds"],
            "max_calls": ceilings["max_calls"],
            "max_tokens": ceilings["max_tokens"],
            "max_output_tokens": ceilings["max_output"],
            "max_cost_usd": ceilings["max_cost"],
        },
        "price_input_per_1m_usd": settings.price_input_per_1m,
        "price_output_per_1m_usd": settings.price_output_per_1m,
        "steps": steps,
        "totals": totals,
        "recorded_usage": {
            "calls": recorded_calls,
            "total_tokens": recorded_tokens,
            "cost_usd": round(recorded_cost, 6),
            "paid_llm_unlocked": False,
        },
        "fits_ceilings": fits,
        "full_protocol_fits_remaining_budget": remaining_fits,
        "budget_note": note,
        "disclaimer": (
            "Estimate from the configured USD per 1M token prices and a 500-token output cap. "
            "Provider billing can differ. No provider call was made."
        ),
    }
