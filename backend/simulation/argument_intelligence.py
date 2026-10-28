"""Phase 8.2 — Argument Intelligence V2 (offline-safe, no provider calls).

Classifies claim kinds, scores argument quality, and summarizes debate
argument diversity. Used by Debate Engine V2 / DiscussionEngine snapshots.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from backend.simulation.social_dynamics import (
    classify_textual_agree_disagree,
    extract_local_comment_refs,
)


def _argument_fingerprint(text: str) -> str:
    from backend.simulation.discussion_engine import argument_fingerprint

    return argument_fingerprint(text)


def _claims_invented_source(text: str) -> bool:
    from backend.simulation.discussion_engine import claims_invented_source

    return claims_invented_source(text)

QUALITY_FIX = "8.2"
ARGUMENT_ENGINE = "v2"

_WORD_RE = re.compile(r"[a-zäöüß0-9]{4,}", re.IGNORECASE)

_EVIDENCE_DEMAND = (
    "evidenz",
    "studie",
    "empirisch",
    "evaluation",
    "pilot",
    "belastbar",
    "messbar",
    "datenlage",
    "nachweis",
)
_CAVEAT = (
    "allerdings",
    "jedoch",
    "voraussetzung",
    "bedingung",
    "sofern",
    "nur wenn",
    "nur unter",
    "risiken",
    "datenschutz",
    "datensicherheit",
    "ohne klare",
    "solange",
)
_REBUTTAL = (
    "widerspreche",
    "widersprechen",
    "doch wir dürfen",
    "dürfen nicht",
    "das stimmt so nicht",
    "sehe das anders",
    "allerdings dürfen",
    "nicht automatisch",
    "vergessen, dass",
)
_SUPPORT = (
    "chancen",
    "potenzial",
    "befürwort",
    "unterstütze",
    "grundsätzlich positiv",
    "sinnvoll",
    "entlast",
    "ermöglicht",
)
_ACK = (
    "stimme zu",
    "nachvollziehbar",
    "wichtige punkte",
    "wichtigen punkt",
    "einverstanden",
    "teile diese",
    "wertvoll",
)


def classify_claim_kind(text: str) -> str:
    """Coarse claim kind for one statement (offline lexical heuristics)."""
    low = (text or "").lower()
    if not low.strip():
        return "empty"
    refs = extract_local_comment_refs(text)
    labels = classify_textual_agree_disagree(text)
    rebut = sum(1 for m in _REBUTTAL if m in low)
    ack = sum(1 for m in _ACK if m in low) + (1 if "agrees_with" in labels else 0)
    evid = sum(1 for m in _EVIDENCE_DEMAND if m in low)
    caveat = sum(1 for m in _CAVEAT if m in low)
    support = sum(1 for m in _SUPPORT if m in low)
    if refs and "disagrees_with" in labels:
        return "rebuttal"
    if rebut and refs:
        return "rebuttal"
    if refs and ack and (caveat or evid or support):
        return "develop"
    if refs and ack and not (caveat or evid or support or rebut):
        return "acknowledgment"
    if evid >= 2 or (evid and caveat):
        return "evidence_demand"
    if caveat and support:
        return "conditional_claim"
    if caveat and not support:
        return "caveat"
    if support and not caveat:
        return "support"
    if rebut:
        return "rebuttal"
    if refs:
        return "address"
    return "statement"


def _theme_tokens(text: str, *, limit: int = 8) -> list[str]:
    low = (text or "").lower()
    low = re.sub(r"→\s*kommentar#\d+", " ", low)
    toks = [t.lower() for t in _WORD_RE.findall(low)]
    stop = {
        "dass",
        "aber",
        "oder",
        "auch",
        "eine",
        "einem",
        "einen",
        "einer",
        "nicht",
        "ohne",
        "über",
        "unter",
        "nach",
        "vor",
        "bei",
        "mit",
        "für",
        "aus",
        "zum",
        "zur",
        "vom",
        "ins",
        "dem",
        "den",
        "des",
        "der",
        "die",
        "das",
        "und",
        "als",
        "wie",
        "man",
        "sich",
        "sehr",
        "mehr",
        "kann",
        "muss",
        "soll",
        "nur",
        "schon",
        "hier",
        "dort",
        "also",
        "weil",
        "doch",
        "wenn",
        "dann",
        "wird",
        "werden",
        "sind",
        "haben",
        "diese",
        "dieser",
        "dieses",
        "comment",
        "kommentar",
        "runde",
        "partner",
        "synthetisch",
        "seed",
    }
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in stop or t.isdigit():
            continue
        key = t[:8]
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def score_argument(
    text: str,
    *,
    is_repetition: bool = False,
    addresses_foreign: bool = False,
    claim_kind: str | None = None,
) -> float:
    """0..1 quality heuristic — length, claim kind, no invented sources."""
    raw = (text or "").strip()
    if not raw:
        return 0.0
    kind = claim_kind or classify_claim_kind(raw)
    words = len(raw.split())
    length = min(1.0, words / 40.0)
    kind_bonus = {
        "conditional_claim": 0.95,
        "evidence_demand": 0.9,
        "rebuttal": 0.88,
        "develop": 0.86,
        "caveat": 0.8,
        "support": 0.75,
        "address": 0.7,
        "acknowledgment": 0.55,
        "statement": 0.5,
        "empty": 0.0,
    }.get(kind, 0.5)
    penalty = 0.0
    if is_repetition:
        penalty += 0.35
    if _claims_invented_source(raw):
        penalty += 0.4
    if addresses_foreign:
        kind_bonus = min(1.0, kind_bonus + 0.08)
    themes = _theme_tokens(raw)
    theme_bonus = min(0.15, 0.02 * len(themes))
    score = 0.35 * length + 0.5 * kind_bonus + theme_bonus - penalty
    return round(max(0.0, min(1.0, score)), 4)


def enrich_argument_record(rec: dict[str, Any], *, text: str | None = None) -> dict[str, Any]:
    """Attach V2 claim metadata onto an argument dict (non-destructive copy)."""
    snippet = text or str(rec.get("snippet") or "")
    kind = classify_claim_kind(snippet)
    themes = _theme_tokens(snippet)
    out = dict(rec)
    out["claim_kind"] = kind
    out["themes"] = themes
    out["quality_score"] = score_argument(
        snippet,
        claim_kind=kind,
        addresses_foreign=bool(extract_local_comment_refs(snippet)),
    )
    out["fingerprint_v2"] = _argument_fingerprint(snippet)
    out["engine"] = ARGUMENT_ENGINE
    return out


def build_argument_intelligence(
    *,
    arguments: list[dict[str, Any]],
    threads: list[dict[str, Any]] | None = None,
    repetition_blocks: int = 0,
    agent_count: int = 0,
) -> dict[str, Any]:
    """Aggregate Argument Intelligence V2 payload for discussion / dashboard."""
    enriched = [enrich_argument_record(a) for a in (arguments or [])]
    kinds = Counter(str(a.get("claim_kind") or "statement") for a in enriched)
    theme_counter: Counter[str] = Counter()
    for a in enriched:
        for t in a.get("themes") or []:
            theme_counter[str(t)] += 1
    scores = [float(a.get("quality_score") or 0.0) for a in enriched]
    mean_q = round(sum(scores) / max(1, len(scores)), 4) if scores else 0.0
    unique_kinds = len([k for k, n in kinds.items() if n > 0 and k != "empty"])
    cross = sum(
        1
        for t in (threads or [])
        if t.get("parent_comment_id") is not None and not t.get("self_reply")
    )
    self_replies = sum(1 for t in (threads or []) if t.get("self_reply"))
    n_args = len(enriched)
    diversity = round(
        min(1.0, unique_kinds / 6.0) * 0.5
        + min(1.0, len(theme_counter) / max(8.0, float(n_args or 1))) * 0.5,
        4,
    )
    address_rate = round(
        sum(
            1
            for a in enriched
            if a.get("claim_kind") in {"rebuttal", "develop", "address", "acknowledgment"}
        )
        / max(1, n_args),
        4,
    )
    invented = sum(1 for a in enriched if _claims_invented_source(str(a.get("snippet") or "")))
    return {
        "backend": "local",
        "engine": ARGUMENT_ENGINE,
        "quality_fix": QUALITY_FIX,
        "argument_count": n_args,
        "agents_covered": len({a.get("agent_id") for a in enriched if a.get("agent_id") is not None}),
        "agent_count": agent_count,
        "claim_kinds": dict(sorted(kinds.items())),
        "unique_claim_kinds": unique_kinds,
        "top_themes": [
            {"theme": t, "count": c} for t, c in theme_counter.most_common(12)
        ],
        "mean_quality_score": mean_q,
        "argument_diversity": diversity,
        "address_rate": address_rate,
        "cross_agent_replies": cross,
        "self_replies": self_replies,
        "repetition_blocks": int(repetition_blocks or 0),
        "invented_source_flags": invented,
        "arguments": enriched[-24:],
        "note": (
            "Argument Intelligence V2: Claim-Arten, Themenvielfalt, Quality-Score — "
            "offline lexikalisch; keine Provider-Calls."
        ),
    }


def debate_quality_score(
    *,
    argument_intel: dict[str, Any],
    unknown_stance_share: float = 0.0,
    polarity_distribution: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Composite Debate Quality V2 score for the dashboard."""
    mean_q = float(argument_intel.get("mean_quality_score") or 0.0)
    diversity = float(argument_intel.get("argument_diversity") or 0.0)
    address = float(argument_intel.get("address_rate") or 0.0)
    reps = int(argument_intel.get("repetition_blocks") or 0)
    n_args = max(1, int(argument_intel.get("argument_count") or 1))
    rep_pen = min(0.4, reps / max(4.0, float(n_args)))
    clarity = max(0.0, 1.0 - float(unknown_stance_share or 0.0))
    pol = polarity_distribution or {}
    clear = sum(int(pol.get(k) or 0) for k in ("pro", "contra", "conditional", "exception"))
    total = max(1, sum(int(v) for v in pol.values()) or 1)
    clear_share = clear / total
    # Prefer multi-polar debate over mono-polarity echo chambers.
    nonempty = [int(v) for k, v in pol.items() if k not in {"unknown", "uncertain"} and int(v or 0) > 0]
    balance = 0.0
    if len(nonempty) >= 2:
        mx, mn = max(nonempty), min(nonempty)
        balance = round(mn / max(1, mx), 4)
    elif len(nonempty) == 1:
        balance = 0.35
    score = round(
        max(
            0.0,
            min(
                1.0,
                0.28 * mean_q
                + 0.22 * diversity
                + 0.18 * address
                + 0.18 * clarity
                + 0.08 * clear_share
                + 0.06 * balance
                - rep_pen,
            ),
        ),
        4,
    )
    return {
        "backend": "local",
        "engine": "debate_v2",
        "quality_fix": QUALITY_FIX,
        "score": score,
        "components": {
            "mean_argument_quality": mean_q,
            "argument_diversity": diversity,
            "address_rate": address,
            "stance_clarity": round(clarity, 4),
            "clear_polarity_share": round(clear_share, 4),
            "polarity_balance": balance,
            "repetition_penalty": round(rep_pen, 4),
        },
        "ok": score >= 0.45 and clarity >= 0.5 and reps == 0,
        "note": (
            "Debate Quality V2 aggregiert Argument-Qualität, Adressierung, "
            "Stance-Klarheit und Wiederholungsstrafe — offline."
        ),
    }
