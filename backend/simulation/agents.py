"""Agent personalities — fixed test roster (Phase 1/2) + Phase-3 generator hooks.

Phase 7.3: optional age / occupation / role_label for realistic individuals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentProfile:
    agent_id: int
    user_name: str
    name: str
    description: str
    goal: str
    style: str
    age: int | None = None
    occupation: str | None = None
    role_label: str | None = None
    # Phase 8.3 Persona Motivation Engine V4 (optional structured drivers).
    motivation: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_AGENTS: list[AgentProfile] = [
    AgentProfile(
        agent_id=0,
        user_name="alice_analyst",
        name="Alice Analyst",
        description="Kritische Tech-Analystin, hinterfragt Hype und verlangt Belege.",
        goal="Fakten prüfen und Schwächen in Argumenten aufdecken.",
        style="skeptisch, präzise, quellenorientiert",
        age=31,
        occupation="Tech-Analystin",
        role_label="Analyst",
    ),
    AgentProfile(
        agent_id=1,
        user_name="bob_builder",
        name="Bob Builder",
        description="Optimistischer Founder, der Open-Source und Experimente pusht.",
        goal="Praktische Lösungen und Experimentierfreude fördern.",
        style="enthusiastisch, lösungsorientiert",
        age=36,
        occupation="Founder",
        role_label="Builder",
    ),
    AgentProfile(
        agent_id=2,
        user_name="cara_community",
        name="Cara Community",
        description="Moderatorin mit Fokus auf zivilen Diskurs und Klarheit.",
        goal="Diskussion strukturieren und Eskalation dämpfen.",
        style="ruhig, vermittelnd, klar",
        age=29,
        occupation="Community-Moderatorin",
        role_label="Moderator",
    ),
    AgentProfile(
        agent_id=3,
        user_name="dan_devil",
        name="Dan Devil",
        description="Contrarian, der jede Behauptung provozierend angreift.",
        goal="Konsens aufbrechen und Gegenpositionen schärfen.",
        style="scharf, ironisch, konfrontativ",
        age=41,
        occupation="Kolumnist",
        role_label="Contrarian",
    ),
    AgentProfile(
        agent_id=4,
        user_name="eve_empath",
        name="Eve Empath",
        description="Empathische Stimme, die Nuancen und Menschen betont.",
        goal="Perspektiven verbinden und emotionale Kosten sichtbar machen.",
        style="warm, reflektiert, inklusiv",
        age=27,
        occupation="Sozialarbeiterin",
        role_label="Empath",
    ),
]


def profiles_payload() -> list[dict]:
    """Default preview roster (fixed test five) for /api/agents before a run."""
    return [p.to_dict() for p in DEFAULT_AGENTS]
