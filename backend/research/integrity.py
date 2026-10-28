"""Provenance labels for NanoCrab Lab research records.

Synthetic fixture text is not human-subject research and not a market measurement.
"""
from __future__ import annotations

import re
from typing import Any, Iterator

PROVENANCE_KINDS = (
    "user_input",
    "source",
    "synthetic_statement",
    "interpretation",
    "human_validation_need",
)

DISCLAIMER: dict[str, Any] = {
    "synthetic_not_human_research": True,
    "not_research_validated": True,
    "no_representativeness_claim": True,
    "no_fabricated_statistics": True,
    "generation": "offline_deterministic_fixture",
    "genuine_ai_research": False,
    "summary": (
        "Synthetic personas and offline fixture statements are not human-subject research, "
        "not a market sample, and not genuine model output. "
        "Interpretations need a human check. This record claims no statistics."
    ),
    "openai_calls": 0,
    "cost_usd": 0,
    "paid_llm_unlocked": False,
}

_BANNED_PROSE = (
    re.compile(r"\d+(?:\.\d+)?\s*%"),
    re.compile(r"\brepresentative sample\b", re.I),
    re.compile(r"\brepräsentativ", re.I),
    re.compile(r"\bstatistically significant\b", re.I),
    re.compile(r"\bstatistisch signifikant\b", re.I),
    re.compile(r"\bconfidence interval\b", re.I),
    re.compile(r"\bp\s*[<=>]\s*0\.\d+", re.I),
    re.compile(r"\bn\s*=\s*\d+", re.I),
    re.compile(r"\bsample size\b", re.I),
)

_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")

PROSE_KEYS = {
    "summary",
    "text",
    "title",
    "description",
    "perspective",
    "name",
    "role",
    "organization",
    "tension",
    "hope",
}


def iter_prose(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in PROSE_KEYS and isinstance(value, str):
                yield value
            else:
                yield from iter_prose(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_prose(item)


def find_banned_claims(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in _BANNED_PROSE:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def prose_banned_hits(obj: Any) -> list[str]:
    hits: list[str] = []
    for text in iter_prose(obj):
        for hit in find_banned_claims(text):
            if hit not in hits:
                hits.append(hit)
    return hits


def user_text_has_percent(text: str) -> bool:
    return bool(_PERCENT.search(text or ""))


def kind_index(study: dict) -> dict[str, str]:
    """Map record ids to provenance kinds. Findings cite these ids."""
    index: dict[str, str] = {}
    question = study.get("research_question") or {}
    if question.get("id"):
        index[question["id"]] = "user_input"
    context = study.get("context") or {}
    if context.get("id"):
        index[context["id"]] = "user_input"
    for source in context.get("sources") or []:
        if source.get("id"):
            index[source["id"]] = "source"
    audience = study.get("audience") or {}
    if audience.get("id"):
        index[audience["id"]] = "user_input"
    for person in study.get("participants") or []:
        if person.get("id"):
            index[person["id"]] = "synthetic_participant"
    for session in study.get("sessions") or []:
        if session.get("id"):
            index[session["id"]] = "synthetic_statement"
        for statement in session.get("statements") or []:
            if statement.get("id"):
                index[statement["id"]] = "synthetic_statement"
    for finding in study.get("findings") or []:
        if finding.get("id"):
            index[finding["id"]] = "interpretation"
        validation = finding.get("human_validation") or {}
        if validation.get("id"):
            index[validation["id"]] = "human_validation_need"
    return index


def unresolved_source_ids(study: dict) -> list[str]:
    known = set(kind_index(study))
    missing: list[str] = []
    for session in study.get("sessions") or []:
        for statement in session.get("statements") or []:
            for ref in statement.get("source_refs") or []:
                if ref not in known and ref not in missing:
                    missing.append(ref)
    for finding in study.get("findings") or []:
        for ref in finding.get("source_ids") or []:
            if ref not in known and ref not in missing:
                missing.append(ref)
        validation = finding.get("human_validation") or {}
        if finding.get("text") and validation.get("kind") != "human_validation_need":
            missing.append(f"bad-kind:{finding.get('id')}")
        if finding.get("text") and finding.get("kind") != "interpretation":
            missing.append(f"bad-finding-kind:{finding.get('id')}")
    return missing


def disclaimer_for_study(study: dict) -> dict[str, Any]:
    """Offline demos keep the fixture disclaimer. Model pathway studies say so."""
    session = ((study.get("sessions") or [None])[0]) or {}
    usage = study.get("ai_usage") or {}
    modelish = (
        session.get("execution") == "model_interview"
        or study.get("participant_generation") == "model"
    )
    if not modelish:
        return dict(DISCLAIMER)
    genuine = (
        session.get("execution") == "model_interview"
        and bool(session.get("statements"))
        and study.get("synthesis_method") == "model_structured_json"
        and not usage.get("partial")
        and not session.get("partial")
    )
    if session.get("execution") == "model_interview":
        generation = "model_pathway"
    else:
        generation = "model_participants"
    return {
        "synthetic_not_human_research": True,
        "not_research_validated": True,
        "no_representativeness_claim": True,
        "no_fabricated_statistics": True,
        "generation": generation,
        "genuine_ai_research": genuine,
        "summary": (
            "Synthetic personas are not human subjects and not a market sample. "
            "Model-pathway text is labeled separately from offline demo fixtures. "
            "Interpretations need a human check. This record claims no statistics."
        ),
        "openai_calls": int(usage.get("calls") or 0),
        "cost_usd": float(usage.get("cost_usd") or 0),
        "paid_llm_unlocked": False,
        "synthesis_method": study.get("synthesis_method") or "not_synthesized",
    }


def provenance_kinds_present(study: dict) -> list[str]:
    present: set[str] = set()
    question = study.get("research_question") or {}
    if question.get("text"):
        present.add("user_input")
    if (study.get("context") or {}).get("text"):
        present.add("user_input")
    if (study.get("context") or {}).get("sources"):
        present.add("source")
    for session in study.get("sessions") or []:
        if session.get("statements"):
            present.add("synthetic_statement")
    for finding in study.get("findings") or []:
        if finding.get("text"):
            present.add("interpretation")
        if (finding.get("human_validation") or {}).get("text"):
            present.add("human_validation_need")
    return [kind for kind in PROVENANCE_KINDS if kind in present]
