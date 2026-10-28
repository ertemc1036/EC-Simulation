"""Topic/domain helpers for offline persona + argument generation.

Keeps public posts/comments thesis-specific without Berlin-mobility leftovers
when the scenario is about income / BGE / other non-mobility theses.
Phase 7.3.1: never invent BGE from stakeholder role words (Arbeitnehmer…).
No LLM — deterministic keyword heuristics only.
"""
from __future__ import annotations

import re
from typing import Literal

TopicDomain = Literal["mobility", "income", "labor", "education", "generic"]

_MOBILITY_CUES = (
    "verbot",
    "auto",
    "autos",
    "s-bahn",
    "sbahn",
    "verkehr",
    "mobilität",
    "mobilitaet",
    "öpnv",
    "oepnv",
    "parkraum",
    "lieferfenster",
    "innenstadt",
    "ring",
)
# True income/BGE cues only — NEVER stakeholder role nouns (Arbeitnehmer…).
# Role words previously leaked BGE into Vier-Tage-Woche (fixture bdfb9e9d).
_INCOME_CUES = (
    "grundeinkommen",
    "bedingungslos",
    "bge",
    "transfer",
    "sozialhilfe",
    "mindestsicherung",
    "1.500",
    "1500",
)
_LABOR_CUES = (
    "vier-tage-woche",
    "vier-tage",
    "4-tage-woche",
    "lohnausgleich",
    "arbeitszeit",
    "wochenarbeitszeit",
    "vollzeitwoche",
    "work-life",
    "worklife",
)
# Education / school AI theses (Phase 8.0 / fixture ddfedb31).
_EDUCATION_CUES = (
    "schule",
    "schulen",
    "schulisch",
    "lehrkraft",
    "lehrkräfte",
    "lehrkraeften",
    "lehrkräften",
    "bildungsforscher",
    "bildungsforschung",
    "lernassistent",
    "ki-lern",
    "schulreform",
    "digitalunterricht",
    "unterricht",
)

# Instruction / meta verbs that must not become topic keywords or agent names.
INSTRUCTION_TOKEN_STEMS = frozenset(
    {
        "simulier",
        "beruecksichtig",
        "berücksichtig",
        "untersuch",
        "analysier",
        "reagier",
        "reaktion",
        "reaktionen",
        "entscheidung",
        "gesellschaftlich",
        "gesellschaftlichen",
        "beruecksichtige",
        "berücksichtige",
        "untersuche",
        "analysiere",
        "diskutier",
        "entwickel",
        "beruecksichtigen",
        "berücksichtigen",
        "veränder",
        "entsteh",
        "agenten",
        "miteinander",
        "moegliche",
        "mögliche",
        "konkreter",
        "aufgrund",
    }
)

_MOBILITY_LEAK = (
    "mobilität",
    "mobilitaet",
    "stadtbevölkerung",
    "stadtbevoelkerung",
    "anwohnende",
    "mobilitätseingeschränkt",
    "mobilitaetseingeschraenkt",
    "lieferfenster",
    "öpnv",
    "oepnv",
    "stadtentwicklung",
    "durchgangsverkehr",
    "s-bahn",
)

# Markers that must not appear unless present in the scenario thesis.
_BGE_ONLY_MARKERS = (
    "grundeinkommen",
    "bedingungsloses grundeinkommen",
)
_LABOR_ONLY_MARKERS = (
    "vier-tage-woche",
    "lohnausgleich",
)
_MOBILITY_ONLY_MARKERS = (
    "lieferfenster",
    "stadtbevölkerung",
    "stadtbevoelkerung",
    "mobilitätseingeschränkt",
    "mobilitaetseingeschraenkt",
    "pauschale verbote",
    "durchgangsverkehr",
)
_HOUSING_ONLY_MARKERS = ("mietendeckel", "mietpreisbremse")
_EDU_ONLY_MARKERS = (
    "schulreform",
    "ki-lernassistenten",
    "lernassistenten",
    "digitalunterricht",
)
_TAX_ONLY_MARKERS = ("ki-steuer",)


def _cue_hit(low: str, cue: str) -> bool:
    """Substring match with word-ish boundaries for short cues (auto≠automatisiert)."""
    if len(cue) <= 4:
        return bool(
            re.search(rf"(?<![a-zäöüß0-9]){re.escape(cue)}(?![a-zäöüß0-9])", low)
        )
    return cue in low


def detect_topic_domain(scenario: str) -> TopicDomain:
    """Classify scenario domain for persona + argument templates."""
    low = (scenario or "").lower()
    income_hits = sum(1 for c in _INCOME_CUES if _cue_hit(low, c))
    mobility_hits = sum(1 for c in _MOBILITY_CUES if _cue_hit(low, c))
    labor_hits = sum(1 for c in _LABOR_CUES if _cue_hit(low, c))
    education_hits = sum(1 for c in _EDUCATION_CUES if _cue_hit(low, c))
    # Explicit BGE / income theses win even if «Bürger» / labor roles appear.
    if (
        income_hits >= 1
        and ("grundeinkommen" in low or "bedingungslos" in low or "bge" in low)
    ) or income_hits >= 2:
        return "income"
    if labor_hits >= 1:
        return "labor"
    # School / KI-Lernassistenten debates (ddfedb31) before generic mobility.
    if education_hits >= 2 or (
        education_hits >= 1
        and any(
            m in low
            for m in (
                "schule",
                "schulen",
                "lehrkraft",
                "lehrkräfte",
                "lehrkraeften",
                "bildungsforscher",
                "lernassistent",
            )
        )
    ):
        return "education"
    auto_hit = _cue_hit(low, "auto") or _cue_hit(low, "autos")
    if mobility_hits >= 2 or ("verbot" in low and auto_hit):
        return "mobility"
    if income_hits > mobility_hits and income_hits > 0:
        return "income"
    if mobility_hits > 0:
        return "mobility"
    if education_hits > 0:
        return "education"
    return "generic"


def _extract_measure_span(scenario: str) -> str | None:
    """Pull the debated measure phrase from the thesis text (never invent BGE)."""
    text = (scenario or "").strip()
    if not text:
        return None
    patterns = (
        r"(?:bedingungsloses\s+)?[Gg]rundeinkommen(?:\s+von\s+[\d.]+(?:\s*€)?)?",
        r"[Vv]ier-Tage-Woche(?:\s+bei\s+vollem\s+Lohnausgleich)?",
        r"[Mm]ietendeckel(?:\s+für\s+(?:Bestands)?wohnungen)?",
        r"(?:[Ee]inführung\s+einer\s+)?[Kk]I-Steuer(?:\s+auf\s+KI)?",
        r"[Ss]chulreform",
        r"[Kk]I-Lernassistenten",
        r"[Ll]ernassistenten",
        r"[Dd]igitalunterricht",
        r"[Aa]uto-?[Vv]erbot",
    )
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip()[:70]
    return None


def measure_label(scenario: str, domain: TopicDomain | None = None) -> str:
    """Short noun phrase for the debated measure — always from current scenario."""
    domain = domain or detect_topic_domain(scenario)
    text = (scenario or "").strip()
    low = text.lower()
    extracted = _extract_measure_span(text)
    if extracted:
        return extracted
    # Domain fallbacks only when the cue is actually present — never invent BGE.
    if domain == "income" and (
        "grundeinkommen" in low or "bedingungslos" in low or "bge" in low
    ):
        return "bedingungsloses Grundeinkommen"
    if domain == "labor":
        return "Vier-Tage-Woche"
    if domain == "education":
        if "lernassistent" in low or "ki-lern" in low:
            return "KI-Lernassistenten"
        if "schulreform" in low:
            return "Schulreform"
        if "digitalunterricht" in low:
            return "Digitalunterricht"
        return "Bildungsmaßnahme"
    if domain == "mobility":
        if "verbot" in low:
            return "Auto-Verbot"
        return "Verkehrsmaßnahme"
    # First sentence fragment as generic measure label (current thesis only).
    thesis = public_scenario_thesis(text, limit=70)
    return thesis if thesis else "die vorgeschlagene Maßnahme"


def thesis_focus(scenario: str, *, limit: int = 4) -> list[str]:
    """Lightweight content nouns for binding — filters instruction stems."""
    stop = {
        "der",
        "die",
        "das",
        "und",
        "oder",
        "ein",
        "eine",
        "für",
        "alle",
        "mit",
        "von",
        "auf",
        "ab",
        "bis",
        "den",
        "dem",
        "des",
        "wie",
        "sich",
        "durch",
        "oder",
        "welche",
        "ob",
    }
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9.\-]{3,}", scenario or "")
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        low = tok.lower()
        if low in stop or low in seen:
            continue
        if any(stem in low for stem in INSTRUCTION_TOKEN_STEMS):
            continue
        if not tok[:1].isupper() and not any(ch.isdigit() for ch in tok):
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= limit:
            break
    return out or ["Diskussion"]


def contains_mobility_leak(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _MOBILITY_LEAK)


def foreign_topic_markers_for_scenario(scenario: str) -> tuple[str, ...]:
    """Markers forbidden for this scenario (absent from thesis / wrong domain)."""
    low = (scenario or "").lower()
    domain = detect_topic_domain(scenario)
    foreign: list[str] = []

    def _add_if_absent(markers: tuple[str, ...]) -> None:
        for m in markers:
            if m not in low:
                foreign.append(m)

    # BGE vocabulary never on non-income theses (core 7.3.1 leak).
    if domain != "income":
        _add_if_absent(_BGE_ONLY_MARKERS)
    if domain != "labor":
        _add_if_absent(_LABOR_ONLY_MARKERS)
    if domain != "mobility":
        _add_if_absent(_MOBILITY_ONLY_MARKERS)
    if domain != "education":
        _add_if_absent(_EDU_ONLY_MARKERS)
    _add_if_absent(_HOUSING_ONLY_MARKERS)
    _add_if_absent(_TAX_ONLY_MARKERS)
    return tuple(dict.fromkeys(foreign))


def contains_foreign_topic_leak(scenario: str, text: str) -> bool:
    """True if ``text`` injects a topic family absent from ``scenario``."""
    low = (text or "").lower()
    scen = (scenario or "").lower()
    for marker in foreign_topic_markers_for_scenario(scenario):
        if marker in low and marker not in scen:
            return True
    return False


def synthetic_tag() -> str:
    """Public marker for offline Manual/Scripted content (not live LLM)."""
    return "[synthetisch]"


# Public-feed instruction leaks (must never appear in seed / public posts).
# Note: argumentative "Zielkonflikt: …" stance lines are allowed in offline
# Manual/Scripted content; the seed dump pattern is "— Ziel: <persona goal>".
PUBLIC_FEED_LEAK_MARKERS = (
    "stil:",
    "[id=",
    "system prompt",
    "you are",
    "halteeine klare",
)
_GOAL_DUMP_RE = re.compile(
    r"(?i)(?:—|-)\s*ziel\s*:|"
    r"diskussionsimpuls\s*\([^)]*\)\s*:.*ziel\s*:",
)


def contains_public_feed_leak(text: str) -> bool:
    """True if public feed text dumps goals, IDs, style, or prompt residue."""
    raw = text or ""
    low = raw.lower()
    if any(m in low for m in PUBLIC_FEED_LEAK_MARKERS):
        return True
    return bool(_GOAL_DUMP_RE.search(raw))


def public_scenario_thesis(scenario: str, *, limit: int = 280) -> str:
    """First thesis sentence only — strips multi-paragraph instruction blocks."""
    text = " ".join((scenario or "").split()).strip()
    if not text:
        return "Diskussionsimpuls zur aktuellen These."
    # Cut at instruction verbs that often follow the thesis in user prompts.
    cut_re = re.compile(
        r"(?i)\b(?:simuliere|berücksichtige|beruecksichtige|untersuche|"
        r"analysiere|entwickle)\b"
    )
    m = cut_re.search(text)
    if m and m.start() > 40:
        text = text[: m.start()].rstrip(" .,;:")
    # Prefer first sentence.
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    thesis = parts[0].strip() if parts else text
    if len(thesis) > limit:
        thesis = thesis[: limit - 1].rstrip() + "…"
    return thesis or text[:limit]


def build_public_seed_content(
    scenario: str,
    *,
    language: str = "de",
) -> str:
    """Public seed post — current thesis only (no prior-topic / goal / ID leak)."""
    thesis = public_scenario_thesis(scenario)
    measure = measure_label(scenario)
    lang = (language or "de").strip().lower() or "de"
    if lang.startswith("de"):
        ask = f"Wie bewertet ihr «{measure}» — Chancen, Risiken, Bedingungen?"
    else:
        ask = f"How do you assess «{measure}» — upsides, risks, conditions?"
    return f"[Seed] Diskussionsimpuls: {thesis} {ask}"
