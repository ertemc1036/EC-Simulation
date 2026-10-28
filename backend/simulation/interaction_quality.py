"""Phase 8.3–8.5 — Argument-Grounded + Interaction Quality V5.

V3: Antwortbezug via Token-/Themen-Overlap + Attribution-Check.
V4: Grounding-Floor (weak_overlap) + Bildung↔IT bags.
V5: Claim-Based Interaction V4 + Argument Evolution in composite score.
Offline lexical heuristics only — no provider calls.
"""
from __future__ import annotations

import re
from typing import Any

from backend.simulation.claim_interaction import evaluate_claim_based_interactions
from backend.simulation.social_dynamics import extract_local_comment_refs

QUALITY_FIX = "8.5"
GROUNDED_ENGINE = "v4"
INTERACTION_QUALITY_ENGINE = "v5"

# Overlap below this while still grounded → weak floor (Phase 8.4 / C6 class).
GROUNDING_OVERLAP_FLOOR = 0.20

_WORD_RE = re.compile(r"[a-zäöüß0-9]{4,}", re.IGNORECASE)
_ATTR_RE = re.compile(
    r"(?i)(?:stimme\s+(?:ihnen|dir|ihm|ihr)\s+zu,\s*dass|"
    r"ihre\s+sorge,\s*dass|ihr\s+hinweis\s+auf|"
    r"dass\s+die\s+herausforderung\s+(?:der|des)\s+)"
    r"([^.!?]{8,80})"
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
        "noch",
        "wenn",
        "dann",
        "wird",
        "werden",
        "sind",
        "haben",
        "nicht",
        "ohne",
        "mit",
        "für",
        "fuer",
        "dieser",
        "dieses",
        "diese",
        "einem",
        "können",
        "koennen",
        "muss",
        "müssen",
        "muessen",
        "sowie",
        "dabei",
        "darauf",
        "davon",
        "dazu",
        "comment",
        "kommentar",
        "ihnen",
        "ihre",
        "ihren",
        "stimme",
    }
)

# Thematic synonym bags — paraphrases count as grounded Antwortbezug.
_THEME_BAGS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "evidenz",
            "empirisch",
            "studie",
            "studien",
            "datenlage",
            "datenbasis",
            "evaluation",
            "evaluationsstandard",
            "pilot",
            "piloten",
            "pilotprojekt",
            "pilotprojekte",
            "pilotphasen",
            "belastbar",
            "wirksamkeit",
            "lernleistung",
            "lernleistungen",
            "lernfortschritt",
            "lernfortschritte",
            "langfristig",
            "begleitung",
            "wissenschaftlich",
            "nebenwirkungen",
            "motivation",
            "selbststeuerung",
        }
    ),
    frozenset(
        {
            "datenschutz",
            "daten",
            "privatheit",
            "schutzregeln",
            "transparenz",
            "eltern",
            "vertrauen",
        }
    ),
    frozenset(
        {
            "chance",
            "chancen",
            "chancengleichheit",
            "ungleichheit",
            "ungleichheiten",
            "kluft",
            "förder",
            "foerder",
            "förderprogrammen",
            "foerderprogrammen",
            "benachteiligt",
            "sozial",
            "zugang",
            "potenzial",
            "potential",
            "einführung",
            "einfuehrung",
            "vollausrollung",
            "voraussetzung",
            "voraussetzungen",
        }
    ),
    frozenset(
        {
            "lehrkraft",
            "lehrkräfte",
            "lehrkraefte",
            "unterricht",
            "pädagogisch",
            "paedagogisch",
            "entlast",
            "arbeitsbelastung",
            "rolle",
        }
    ),
    # Phase 8.4: Bildung ↔ IT bridge (Hannah C6 ↔ Thomas IT-Eltern).
    frozenset(
        {
            "technologie",
            "technologisch",
            "technisch",
            "ki-lernassistent",
            "lernassistent",
            "lernassistenten",
            "it-projektleiter",
            "projektleiter",
            "digital",
            "digitale",
            "system",
            "systeme",
            "bildungswissenschaftlich",
            "bildungswissenschaft",
        }
    ),
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


def _theme_overlap(a: set[str], b: set[str]) -> float:
    """Share of theme bags that fire in both parent and reply."""
    hits = 0
    active = 0
    for bag in _THEME_BAGS:
        in_a = bool(a & bag)
        in_b = bool(b & bag)
        if in_a or in_b:
            active += 1
        if in_a and in_b:
            hits += 1
    if active <= 0:
        return 0.0
    return hits / float(active)


def _claimed_attributions(reply: str) -> list[str]:
    """Extract 'I agree that X' / 'your concern that X' spans from a reply."""
    out: list[str] = []
    for m in _ATTR_RE.finditer(reply or ""):
        span = (m.group(1) or "").strip().lower()
        if span:
            out.append(span)
    # Explicit invented-topic pattern seen in fixture d1fb1b1c (digitale Kluft).
    low = (reply or "").lower()
    m2 = re.search(
        r"stimme\s+ihnen\s+zu,\s*dass\s+([^.!?]{10,100})",
        low,
        re.IGNORECASE,
    )
    if m2:
        span = m2.group(1).strip()
        if span and span not in out:
            out.append(span)
    return out


def score_reply_grounding(
    reply: str,
    parent: str,
    *,
    parent_comment_id: int | None = None,
) -> dict[str, Any]:
    """Score how well ``reply`` addresses ``parent`` (Antwortbezug)."""
    refs = extract_local_comment_refs(reply)
    is_reply = bool(refs) or parent_comment_id is not None
    if not is_reply:
        return {
            "grounded": True,
            "is_reply": False,
            "score": 1.0,
            "overlap": 0.0,
            "weak_overlap": False,
            "invented_attribution": False,
            "missing_parent_themes": [],
            "claimed_spans": [],
            "engine": GROUNDED_ENGINE,
        }

    parent_toks = _tokens(parent)
    reply_toks = _tokens(reply)
    lexical = _jaccard(parent_toks, reply_toks)
    thematic = _theme_overlap(parent_toks, reply_toks)
    overlap = round(max(lexical, 0.55 * lexical + 0.45 * thematic), 4)

    claimed = _claimed_attributions(reply)
    invented = False
    missing: list[str] = []
    parent_low = (parent or "").lower()
    for span in claimed:
        span_toks = _tokens(span)
        # Attribution is invented when few claimed tokens appear in parent
        # and the span does not share a theme bag with the parent.
        hit = sum(1 for t in span_toks if t in parent_low or t in parent_toks)
        theme_ok = _theme_overlap(span_toks, parent_toks) >= 0.34
        need = max(1, min(2, len(span_toks) // 2))
        if hit < need and not theme_ok:
            invented = True
            theme = " ".join(list(span_toks)[:4]) or span[:40]
            missing.append(theme)

    grounded = (
        (not invented and (overlap >= 0.06 or thematic >= 0.34 or lexical >= 0.05))
        or overlap >= 0.18
    )
    if invented:
        grounded = False
    weak_overlap = bool(grounded and overlap < GROUNDING_OVERLAP_FLOOR)
    score = overlap
    if invented:
        score = max(0.0, score - 0.35)
    elif grounded:
        score = min(1.0, score + 0.15 + 0.1 * thematic)
        if weak_overlap:
            score = max(0.0, score - 0.08)
    score = round(max(0.0, min(1.0, score)), 4)

    return {
        "grounded": grounded,
        "is_reply": True,
        "score": score,
        "overlap": overlap,
        "weak_overlap": weak_overlap,
        "invented_attribution": invented,
        "missing_parent_themes": missing[:6],
        "claimed_spans": claimed[:4],
        "parent_comment_id": parent_comment_id or (refs[0] if refs else None),
        "engine": GROUNDED_ENGINE,
        "quality_fix": QUALITY_FIX,
        "overlap_floor": GROUNDING_OVERLAP_FLOOR,
    }


def evaluate_argument_grounded_interactions(
    *,
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Argument-Grounded Interaction V4 snapshot over a comment thread."""
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
        ev = score_reply_grounding(
            content,
            parent_text,
            parent_comment_id=int(refs[0]),
        )
        ev["comment_id"] = c.get("comment_id")
        ev["user_id"] = c.get("user_id")
        evaluations.append(ev)

    n = len(evaluations)
    grounded_n = sum(1 for e in evaluations if e.get("grounded"))
    invented_n = sum(1 for e in evaluations if e.get("invented_attribution"))
    weak_n = sum(1 for e in evaluations if e.get("weak_overlap"))
    mean_score = (
        round(sum(float(e.get("score") or 0.0) for e in evaluations) / n, 4)
        if n
        else 1.0
    )
    grounded_rate = round(grounded_n / n, 4) if n else 1.0
    invented_rate = round(invented_n / n, 4) if n else 0.0
    weak_rate = round(weak_n / n, 4) if n else 0.0
    return {
        "backend": "local",
        "engine": GROUNDED_ENGINE,
        "quality_fix": QUALITY_FIX,
        "reply_count": n,
        "grounded_replies": grounded_n,
        "grounded_reply_rate": grounded_rate,
        "mean_grounding_score": mean_score,
        "invented_attribution_count": invented_n,
        "invented_attribution_rate": invented_rate,
        "weak_overlap_count": weak_n,
        "weak_overlap_rate": weak_rate,
        "overlap_floor": GROUNDING_OVERLAP_FLOOR,
        "evaluations": evaluations[-24:],
        "ok": (n == 0) or (grounded_rate >= 0.5 and invented_rate <= 0.5),
        "note": (
            "Argument-Grounded Interaction V4: Antwortbezug + Attribution-Check "
            f"+ Grounding-Floor (overlap < {GROUNDING_OVERLAP_FLOOR}) — "
            "erfundene Eltern-Themen und fragile Overlaps senken den Score."
        ),
    }


def build_interaction_quality(
    *,
    comments: list[dict[str, Any]],
    agents: list[dict[str, Any]] | None = None,
    stance_clarity: float = 0.0,
    debate_quality_score: float = 0.0,
    opinion_stability: float = 0.0,
    argument_evolution_score: float | None = None,
) -> dict[str, Any]:
    """Composite Interaction Quality Evaluation for the Quality Dashboard V5."""
    from backend.simulation.persona_motivation import motivation_coverage

    grounded = evaluate_argument_grounded_interactions(comments=comments)
    claims = evaluate_claim_based_interactions(comments=comments)
    mot = motivation_coverage(agents or [])
    g_rate = float(grounded.get("grounded_reply_rate") or 0.0)
    g_score = float(grounded.get("mean_grounding_score") or 0.0)
    invent = float(grounded.get("invented_attribution_rate") or 0.0)
    weak = float(grounded.get("weak_overlap_rate") or 0.0)
    claim_rate = float(claims.get("claim_address_rate") or 0.0)
    claim_cov = float(claims.get("mean_claim_coverage") or 0.0)
    mot_cov = float(mot.get("coverage") or 0.0)
    stake_cov = float(mot.get("stake_coverage") or 0.0)
    clarity = max(0.0, min(1.0, float(stance_clarity or 0.0)))
    debate = max(0.0, min(1.0, float(debate_quality_score or 0.0)))
    stability = max(0.0, min(1.0, float(opinion_stability or 0.0)))
    evo = max(0.0, min(1.0, float(argument_evolution_score or 0.0)))

    composite = round(
        max(
            0.0,
            min(
                1.0,
                0.20 * g_rate
                + 0.12 * g_score
                + 0.14 * claim_rate
                + 0.06 * claim_cov
                + 0.12 * clarity
                + 0.10 * debate
                + 0.08 * mot_cov
                + 0.04 * stake_cov
                + 0.06 * evo
                + 0.06 * stability
                - 0.18 * invent
                - 0.05 * weak,
            ),
        ),
        4,
    )
    return {
        "backend": "local",
        "engine": INTERACTION_QUALITY_ENGINE,
        "quality_fix": QUALITY_FIX,
        "score": composite,
        "components": {
            "grounded_reply_rate": g_rate,
            "mean_grounding_score": g_score,
            "invented_attribution_rate": invent,
            "weak_overlap_rate": weak,
            "claim_address_rate": claim_rate,
            "mean_claim_coverage": claim_cov,
            "stance_clarity": round(clarity, 4),
            "debate_quality_score": round(debate, 4),
            "persona_motivation_coverage": mot_cov,
            "stakeholder_stake_coverage": stake_cov,
            "argument_evolution_score": round(evo, 4),
            "opinion_stability": round(stability, 4),
        },
        "argument_grounded": grounded,
        "claim_based_interaction": claims,
        "persona_motivation": mot,
        "ok": composite >= 0.45 and invent <= 0.5 and clarity >= 0.5,
        "note": (
            "Interaction Quality Evaluation V5 (Phase 8.5): Antwortbezug, "
            "Claim-Adressierung, Stakeholder-Motivation, Argument Evolution, "
            "Stance-Klarheit und Debate Quality — offline."
        ),
    }
