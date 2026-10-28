"""Topic/domain helpers for offline persona + argument generation.

Keeps public posts/comments thesis-specific without Berlin-mobility leftovers
when the scenario is about income / BGE / other non-mobility theses.
No LLM — deterministic keyword heuristics only.
"""
from __future__ import annotations

import re
from typing import Literal

TopicDomain = Literal["mobility", "income", "generic"]

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
_INCOME_CUES = (
    "grundeinkommen",
    "bedingungslos",
    "bge",
    "transfer",
    "sozialhilfe",
    "mindestsicherung",
    "arbeitnehmer",
    "arbeitslose",
    "rentner",
    "studierende",
    "einkommen",
    "1.500",
    "1500",
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


def detect_topic_domain(scenario: str) -> TopicDomain:
    """Classify scenario domain for persona + argument templates."""
    low = (scenario or "").lower()
    income_hits = sum(1 for c in _INCOME_CUES if c in low)
    mobility_hits = sum(1 for c in _MOBILITY_CUES if c in low)
    # Explicit BGE / income theses win even if «Bürger» appears.
    if income_hits >= 2 or "grundeinkommen" in low or "bedingungslos" in low:
        return "income"
    if mobility_hits >= 2 or ("verbot" in low and "auto" in low):
        return "mobility"
    if income_hits > mobility_hits and income_hits > 0:
        return "income"
    if mobility_hits > 0:
        return "mobility"
    return "generic"


def measure_label(scenario: str, domain: TopicDomain | None = None) -> str:
    """Short noun phrase for the debated measure."""
    domain = domain or detect_topic_domain(scenario)
    text = (scenario or "").strip()
    if domain == "income":
        # Prefer full BGE phrasing; allow decimals like 1.500 inside the span.
        m = re.search(
            r"(?:bedingungsloses\s+)?[Gg]rundeinkommen(?:\s+von\s+[\d.]+(?:\s*€)?)?",
            text,
        )
        if m:
            snippet = re.sub(r"\s+", " ", m.group(0)).strip()
            return snippet[:70]
        return "bedingungsloses Grundeinkommen"
    if domain == "mobility":
        if "verbot" in text.lower():
            return "Auto-Verbot"
        return "Verkehrsmaßnahme"
    # First sentence fragment as generic measure label.
    first = re.split(r"[.?!\n]", text, maxsplit=1)[0].strip()
    return (first[:70] if first else "die vorgeschlagene Maßnahme")


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


def synthetic_tag() -> str:
    """Public marker for offline Manual/Scripted content (not live LLM)."""
    return "[synthetisch]"
