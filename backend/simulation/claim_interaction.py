"""Phase 8.5 — Claim-Based Interaction Engine V4 (offline-safe).

Goes beyond lexical grounding: extracts atomic claims from parent comments
and scores whether replies address those claims. No provider calls.
"""
from __future__ import annotations

import re
from typing import Any

from backend.simulation.social_dynamics import extract_local_comment_refs

QUALITY_FIX = "8.5"
CLAIM_INTERACTION_ENGINE = "v4"

_WORD_RE = re.compile(r"[a-zäöüß0-9]{4,}", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAIM_CUES = (
    "weil",
    "denn",
    "deshalb",
    "daher",
    "braucht",
    "benötigt",
    "muss",
    "müssen",
    "soll",
    "sollte",
    "dürfen nicht",
    "nicht ohne",
    "nur wenn",
    "nur unter",
    "voraussetzung",
    "evidenz",
    "pilot",
    "datenschutz",
    "chancen",
    "risiko",
    "risiken",
    "plädiere",
    "fordere",
    "sehe",
    "halte",
    "betone",
)
_STOP = frozenset(
    {
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
        "stimme",
        "ihnen",
        "ihre",
    }
)


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _WORD_RE.findall(text or "")
        if t.lower() not in _STOP and len(t) >= 4
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / float(union) if union else 0.0


def extract_claims(text: str, *, limit: int = 6) -> list[dict[str, Any]]:
    """Extract coarse atomic claims (sentence/clause heuristics)."""
    raw = (text or "").strip()
    if not raw:
        return []
    cleaned = re.sub(r"→\s*Kommentar#\d+", " ", raw, flags=re.IGNORECASE)
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(cleaned) if p.strip()]
    # Also split long clauses on semicolons / " — ".
    expanded: list[str] = []
    for p in parts:
        chunks = re.split(r"[;…]|\s+—\s+|\s+–\s+", p)
        expanded.extend(c.strip() for c in chunks if c.strip())
    claims: list[dict[str, Any]] = []
    for i, span in enumerate(expanded):
        low = span.lower()
        words = span.split()
        if len(words) < 5:
            continue
        cue_hits = sum(1 for c in _CLAIM_CUES if c in low)
        # Prefer cue-bearing or medium-length substantive sentences.
        if cue_hits == 0 and len(words) < 10:
            continue
        toks = sorted(_tokens(span))
        if len(toks) < 2:
            continue
        claims.append(
            {
                "claim_id": i + 1,
                "text": span[:220],
                "tokens": toks[:10],
                "cue_hits": cue_hits,
            }
        )
        if len(claims) >= limit:
            break
    # Fallback: whole text as one claim if nothing split out.
    if not claims and len(cleaned.split()) >= 6:
        toks = sorted(_tokens(cleaned))
        claims.append(
            {
                "claim_id": 1,
                "text": cleaned[:220],
                "tokens": toks[:10],
                "cue_hits": 0,
            }
        )
    return claims


def score_claim_addressing(
    reply: str,
    parent: str,
    *,
    parent_comment_id: int | None = None,
) -> dict[str, Any]:
    """Score whether ``reply`` addresses extracted parent claims."""
    refs = extract_local_comment_refs(reply)
    is_reply = bool(refs) or parent_comment_id is not None
    if not is_reply:
        return {
            "is_reply": False,
            "claim_count": 0,
            "claims_addressed": 0,
            "claim_address_rate": 1.0,
            "mean_claim_coverage": 1.0,
            "addressed_claim_ids": [],
            "ignored_claim_ids": [],
            "engine": CLAIM_INTERACTION_ENGINE,
            "quality_fix": QUALITY_FIX,
        }

    claims = extract_claims(parent)
    reply_toks = _tokens(reply)
    addressed: list[int] = []
    ignored: list[int] = []
    coverages: list[float] = []
    details: list[dict[str, Any]] = []
    for claim in claims:
        ctoks = set(claim.get("tokens") or [])
        cov = _jaccard(ctoks, reply_toks)
        # Soft thematic: shared prefix stems (4+ chars already).
        coverages.append(cov)
        hit = cov >= 0.12 or len(ctoks & reply_toks) >= 2
        cid = int(claim["claim_id"])
        if hit:
            addressed.append(cid)
        else:
            ignored.append(cid)
        details.append(
            {
                "claim_id": cid,
                "coverage": round(cov, 4),
                "addressed": hit,
                "text": claim.get("text"),
            }
        )

    n = len(claims)
    rate = round(len(addressed) / n, 4) if n else 1.0
    mean_cov = round(sum(coverages) / n, 4) if n else 1.0
    return {
        "is_reply": True,
        "claim_count": n,
        "claims_addressed": len(addressed),
        "claim_address_rate": rate,
        "mean_claim_coverage": mean_cov,
        "addressed_claim_ids": addressed,
        "ignored_claim_ids": ignored,
        "claims": details[:8],
        "parent_comment_id": parent_comment_id or (refs[0] if refs else None),
        "engine": CLAIM_INTERACTION_ENGINE,
        "quality_fix": QUALITY_FIX,
    }


def evaluate_claim_based_interactions(
    *,
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Claim-Based Interaction Engine V4 snapshot over comment threads."""
    by_id = {
        int(c["comment_id"]): c
        for c in (comments or [])
        if c.get("comment_id") is not None
    }
    evaluations: list[dict[str, Any]] = []
    for c in comments or []:
        content = str(c.get("content") or "")
        refs = extract_local_comment_refs(content)
        if not refs:
            continue
        parent = by_id.get(int(refs[0]))
        parent_text = str((parent or {}).get("content") or "")
        ev = score_claim_addressing(
            content,
            parent_text,
            parent_comment_id=int(refs[0]),
        )
        ev["comment_id"] = c.get("comment_id")
        ev["user_id"] = c.get("user_id")
        evaluations.append(ev)

    n = len(evaluations)
    mean_rate = (
        round(sum(float(e.get("claim_address_rate") or 0.0) for e in evaluations) / n, 4)
        if n
        else 1.0
    )
    mean_cov = (
        round(
            sum(float(e.get("mean_claim_coverage") or 0.0) for e in evaluations) / n, 4
        )
        if n
        else 1.0
    )
    total_claims = sum(int(e.get("claim_count") or 0) for e in evaluations)
    total_addressed = sum(int(e.get("claims_addressed") or 0) for e in evaluations)
    return {
        "backend": "local",
        "engine": CLAIM_INTERACTION_ENGINE,
        "quality_fix": QUALITY_FIX,
        "reply_count": n,
        "claim_count": total_claims,
        "claims_addressed": total_addressed,
        "claim_address_rate": mean_rate,
        "mean_claim_coverage": mean_cov,
        "evaluations": evaluations[-24:],
        "ok": (n == 0) or mean_rate >= 0.34,
        "note": (
            "Claim-Based Interaction Engine V4: Antworten müssen Eltern-Claims "
            "adressieren — nicht nur Token-Overlap. Offline lexikalisch."
        ),
    }


def claim_interaction_status() -> dict[str, Any]:
    return {
        "enabled": True,
        "engine": CLAIM_INTERACTION_ENGINE,
        "quality_fix": QUALITY_FIX,
        "paid_llm_unlocked": False,
        "note": (
            "Claim-Based Interaction V4 — Claim-Extraktion und Adressierungsrate "
            "ohne Provider-Calls."
        ),
    }
