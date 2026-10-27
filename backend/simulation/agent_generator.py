"""Dynamic agent profiles from a scenario — no LLM required, no clones.

Phase 3 (MiroFish-ähnlich): generate N unique forum personas bound to the
scenario text. Offline/deterministic so tests never call a provider.

Modes:
- ``fixed_test`` — exactly the Phase-1/2 DEFAULT_AGENTS roster (5), for regression.
- ``dynamic`` — scenario-derived unique profiles (default for configurable counts).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from backend.simulation.agents import DEFAULT_AGENTS, AgentProfile

# Hard ceiling kept in sync with Phase-2.x live-test policy (config clamps too).
MAX_GENERATABLE_AGENTS = 5

_ARCHETYPES: list[dict[str, str]] = [
    {
        "slug": "analyst",
        "name": "Analyst",
        "description": "Prüft Annahmen, verlangt Belege und trennt Hype von Substanz.",
        "goal": "Schwachstellen und fehlende Evidenz im Szenario sichtbar machen.",
        "style": "skeptisch, präzise, quellenorientiert",
    },
    {
        "slug": "builder",
        "name": "Builder",
        "description": "Denkt in Experimenten, Prototypen und messbaren Piloten.",
        "goal": "Praktische nächste Schritte und Testkriterien vorschlagen.",
        "style": "enthusiastisch, lösungsorientiert",
    },
    {
        "slug": "moderator",
        "name": "Moderator",
        "description": "Strukturiert die Debatte und dämpft Eskalation.",
        "goal": "Klarheit schaffen und Positionen fair gegenüberstellen.",
        "style": "ruhig, vermittelnd, klar",
    },
    {
        "slug": "contrarian",
        "name": "Contrarian",
        "description": "Greift Konsens an und schärft Gegenpositionen.",
        "goal": "Versteckte Risiken und Gegenargumente zuspitzen.",
        "style": "scharf, ironisch, konfrontativ",
    },
    {
        "slug": "empath",
        "name": "Empath",
        "description": "Betont Betroffene, Nuancen und soziale Kosten.",
        "goal": "Menschliche Auswirkungen und inklusivere Perspektiven einbringen.",
        "style": "warm, reflektiert, inklusiv",
    },
]

_FIRST_NAMES = [
    "Alex",
    "Blake",
    "Casey",
    "Drew",
    "Eden",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
]

_STOP = frozenset(
    {
        "der",
        "die",
        "das",
        "und",
        "oder",
        "ein",
        "eine",
        "einer",
        "einem",
        "den",
        "dem",
        "des",
        "mit",
        "von",
        "zu",
        "im",
        "in",
        "am",
        "auf",
        "für",
        "als",
        "auch",
        "nicht",
        "sich",
        "ist",
        "sind",
        "wird",
        "werden",
        "sollten",
        "sollte",
        "kann",
        "können",
        "wie",
        "was",
        "wer",
        "ob",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "should",
        "would",
        "could",
        "we",
        "our",
        "their",
    }
)


def extract_scenario_keywords(scenario: str, *, limit: int = 4) -> list[str]:
    """Pull distinctive tokens from the scenario for persona binding."""
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9-]{3,}", scenario or "")
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        low = t.lower()
        if low in _STOP or low in seen:
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= limit:
            break
    return out or ["Diskussion"]


def _slugify(text: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (raw or "agent")[:28]


def _topic_phrase(keywords: list[str]) -> str:
    if not keywords:
        return "das vorliegende Szenario"
    if len(keywords) == 1:
        return keywords[0]
    return ", ".join(keywords[:-1]) + f" und {keywords[-1]}"


def _unique_suffix(scenario: str, agent_id: int, slug: str) -> str:
    digest = hashlib.sha1(f"{scenario}|{agent_id}|{slug}".encode("utf-8")).hexdigest()
    return digest[:6]


def generate_dynamic_agents(
    scenario: str,
    count: int,
    *,
    max_agents: int = MAX_GENERATABLE_AGENTS,
) -> list[AgentProfile]:
    """Build ``count`` unique scenario-bound profiles (no clones)."""
    n = max(1, min(int(count), int(max_agents), MAX_GENERATABLE_AGENTS))
    keywords = extract_scenario_keywords(scenario)
    topic = _topic_phrase(keywords)
    short = (scenario or "").strip()[:120] or "Offenes Diskussionsthema"
    profiles: list[AgentProfile] = []
    used_names: set[str] = set()
    used_user_names: set[str] = set()

    for i in range(n):
        arch = _ARCHETYPES[i % len(_ARCHETYPES)]
        # Rotate first names so a second pass over archetypes still differs.
        first = _FIRST_NAMES[(i + len(keywords)) % len(_FIRST_NAMES)]
        suffix = _unique_suffix(short, i, arch["slug"])
        display = f"{first} {arch['name']}"
        # Collision guard (should be rare; force suffix into display if needed).
        if display.lower() in used_names:
            display = f"{first} {arch['name']} {suffix}"
        user_name = f"{_slugify(first)}_{arch['slug']}_{suffix}"
        if user_name in used_user_names:
            user_name = f"{user_name}_{i}"

        description = (
            f"{arch['description']} Bezieht Positionen explizit auf «{topic}» "
            f"im Szenario: {short}."
        )
        goal = f"{arch['goal']} Fokus: {topic}."
        style = arch["style"]

        profile = AgentProfile(
            agent_id=i,
            user_name=user_name,
            name=display,
            description=description,
            goal=goal,
            style=style,
        )
        used_names.add(display.lower())
        used_user_names.add(user_name)
        profiles.append(profile)

    assert_no_clones(profiles)
    return profiles


def fixed_test_agents(count: int = 5) -> list[AgentProfile]:
    """Phase-2 regression roster — fixed five personalities (truncated if count < 5)."""
    n = max(1, min(int(count), len(DEFAULT_AGENTS), MAX_GENERATABLE_AGENTS))
    return [replace(p) for p in DEFAULT_AGENTS[:n]]


def generate_agents(
    scenario: str,
    count: int = 5,
    *,
    mode: str = "dynamic",
    max_agents: int = MAX_GENERATABLE_AGENTS,
) -> list[AgentProfile]:
    """Public entry: ``mode`` is ``dynamic`` or ``fixed_test``."""
    requested = (mode or "dynamic").strip().lower()
    if requested in {"fixed", "fixed_test", "test", "default"}:
        return fixed_test_agents(count)
    return generate_dynamic_agents(scenario, count, max_agents=max_agents)


def assert_no_clones(profiles: list[AgentProfile]) -> None:
    """Fail fast if names, user_names, descriptions or goals collide."""
    if not profiles:
        raise ValueError("Agentenliste ist leer")
    for field in ("user_name", "name", "description", "goal"):
        values = [getattr(p, field) for p in profiles]
        if len(values) != len(set(values)):
            raise ValueError(f"Klon erkannt: doppelte Werte in {field}")


def profiles_payload_from(profiles: list[AgentProfile]) -> list[dict]:
    return [p.to_dict() for p in profiles]
