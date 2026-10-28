"""Phase 8.3 — Persona Motivation Engine V4 (offline-safe).

Adds structured motivations (values, fears, drivers, decision style) so
agents behave from goals rather than generic role labels. No provider calls.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

PERSONA_ENGINE = "v4"
QUALITY_FIX = "8.4"

_DOMAIN_MOTIVATIONS: dict[str, dict[str, Any]] = {
    "education": {
        "lehrkraft": {
            "values": ["Lernqualität", "pädagogische Verantwortung", "Entlastung"],
            "fears": ["Technikzwang ohne Konzept", "Mehrbelastung", "Delegieren der Rolle"],
            "drivers": ["Evidenz", "Unterrichtskonzept", "Chancengleichheit im Klassenzimmer"],
            "decision_style": "evidence_first",
        },
        "eltern": {
            "values": ["Chancengleichheit", "Kinderschutz", "Transparenz"],
            "fears": ["digitale Kluft", "ungeprüfte Vollausrollung", "Datenschutzlücken"],
            "drivers": ["Schutzregeln", "Elternbeteiligung", "Förderprogramme"],
            "decision_style": "protection_equity",
        },
        "bildungsforscher": {
            "values": ["Empirie", "Evaluation", "Vergleichbarkeit"],
            "fears": ["anekdotische Erfolgsversprechen", "fehlende Langzeitdaten"],
            "drivers": ["Pilotprojekte", "Evaluationsstandards", "Risikofrüherkennung"],
            "decision_style": "evidence_first",
        },
    },
    "labor": {
        "arbeitnehmer": {
            "values": ["Entlastung", "Lebensqualität", "Planbarkeit"],
            "fears": ["Mehrarbeit", "Scheinreform", "ungesicherte Umsetzung"],
            "drivers": ["volle Entlastung", "verbindliche Übergangsregeln"],
            "decision_style": "worker_protection",
        },
        "unternehmen": {
            "values": ["Machbarkeit", "Kostenkontrolle", "Betriebsabläufe"],
            "fears": ["Personalknappheit", "Kostensteigerung", "Produktionsausfälle"],
            "drivers": ["flexible Modelle", "Übergangsphasen"],
            "decision_style": "feasibility_first",
        },
        "wirtschaftsexperte": {
            "values": ["Produktivität", "Gegenfinanzierung", "Systemeffekte"],
            "fears": ["Makrorisiken", "unklare Finanzierung"],
            "drivers": ["Pilotindikatoren", "branchenspezifische Regeln"],
            "decision_style": "evidence_first",
        },
    },
    "income": {
        "arbeitnehmer": {
            "values": ["Absicherung", "Würde", "Planbarkeit"],
            "fears": ["Lohnentwertung", "unklare Anrechnung", "Sanktionsdruck"],
            "drivers": ["Kompatibilität mit Löhnen", "klare Regeln"],
            "decision_style": "protection_equity",
        },
        "unternehmen": {
            "values": ["Finanzierbarkeit", "Anreize", "Planbarkeit"],
            "fears": ["Steuerlast", "Arbeitsanreizverluste"],
            "drivers": ["Gegenfinanzierung", "Pilotversuche"],
            "decision_style": "feasibility_first",
        },
        "wirtschaftsexperte": {
            "values": ["Fiskalische Tragfähigkeit", "Empirie", "Anreize"],
            "fears": ["Inflation", "unklare Finanzierung"],
            "drivers": ["Evaluationsstandards", "Pilotindikatoren"],
            "decision_style": "evidence_first",
        },
    },
}

_DEFAULT_MOTIVATION = {
    "values": ["Klarheit", "Fairness", "Belastbarkeit"],
    "fears": ["Scheinargumente", "überhastete Vollumsetzung"],
    "drivers": ["Belege", "Voraussetzungen", "messbare Schritte"],
    "decision_style": "balanced",
}


@dataclass
class PersonaMotivation:
    values: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    decision_style: str = "balanced"
    conflict_axis: str = ""
    domain: str = "generic"
    role_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_block(self) -> str:
        """Compact German motivation block for system prompts."""
        vals = ", ".join(self.values[:3]) or "—"
        fears = ", ".join(self.fears[:3]) or "—"
        drivers = ", ".join(self.drivers[:3]) or "—"
        return (
            f"Motivation: Werte [{vals}]; Risiken [{fears}]; "
            f"Treiber [{drivers}]; Entscheidung={self.decision_style}."
        )


def _norm_role(role_label: str | None, occupation: str | None = None) -> str:
    raw = f"{role_label or ''} {occupation or ''}".lower()
    mapping = (
        ("lehr", "lehrkraft"),
        ("eltern", "eltern"),
        ("vater", "eltern"),
        ("mutter", "eltern"),
        ("bildungsforscher", "bildungsforscher"),
        ("digitalisierungsforscher", "bildungsforscher"),
        ("arbeitnehmer", "arbeitnehmer"),
        ("unternehmen", "unternehmen"),
        ("betriebsleiter", "unternehmen"),
        ("wirtschaftsexperte", "wirtschaftsexperte"),
        ("finanzwissenschaft", "wirtschaftsexperte"),
        ("ökonom", "wirtschaftsexperte"),
    )
    for needle, key in mapping:
        if needle in raw:
            return key
    return "generic"


def _detect_domain(scenario: str, role_key: str) -> str:
    """Prefer scenario topic domain so motivations never leak foreign topics."""
    try:
        from backend.simulation.topic_context import detect_topic_domain

        domain = detect_topic_domain(scenario or "")
        if domain and domain != "generic":
            return domain
    except Exception:
        pass
    low = (scenario or "").lower()
    if any(
        t in low
        for t in ("schule", "lehr", "ki-lern", "bildung", "eltern", "schüler")
    ):
        return "education"
    if any(t in low for t in ("grundeinkommen", "bge", "1500", "1.500")):
        return "income"
    if any(t in low for t in ("vier-tage", "vier tage", "lohnausgleich")):
        return "labor"
    # Role fallback only when scenario domain is unknown.
    if role_key in {"lehrkraft", "eltern", "bildungsforscher"}:
        return "education"
    return "generic"


def _extract_conflict_axis(goal: str) -> str:
    m = re.search(r"Zielkonflikt:\s*([^.\[]+)", goal or "", re.IGNORECASE)
    if m:
        return m.group(1).strip()[:160]
    return ""


def build_persona_motivation(
    *,
    role_label: str | None = None,
    occupation: str | None = None,
    goal: str = "",
    scenario: str = "",
    style: str = "",
) -> PersonaMotivation:
    """Derive a V4 motivation profile from role + scenario (deterministic)."""
    role_key = _norm_role(role_label, occupation)
    domain = _detect_domain(scenario, role_key)
    base = dict(_DEFAULT_MOTIVATION)
    domain_map = _DOMAIN_MOTIVATIONS.get(domain) or {}
    if role_key in domain_map:
        base.update(domain_map[role_key])
    # Style nudges decision_style slightly without inventing new facts.
    style_l = (style or "").lower()
    if "evidenz" in style_l or "analytisch" in style_l:
        base["decision_style"] = "evidence_first"
    elif "fürsorglich" in style_l or "pragmatisch" in style_l:
        base["decision_style"] = "protection_equity"
    return PersonaMotivation(
        values=list(base.get("values") or [])[:4],
        fears=list(base.get("fears") or [])[:4],
        drivers=list(base.get("drivers") or [])[:4],
        decision_style=str(base.get("decision_style") or "balanced"),
        conflict_axis=_extract_conflict_axis(goal),
        domain=domain,
        role_key=role_key,
    )


def enrich_goal_with_motivation(goal: str, motivation: PersonaMotivation) -> str:
    """Append a compact motivation cue to the private goal (not public post text)."""
    base = (goal or "").strip()
    if "Motivation:" in base:
        return base
    block = motivation.prompt_block()
    if not base:
        return block
    return f"{base} {block}"


def motivation_coverage(agents: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of agents that carry V4 motivation fields."""
    n = len(agents or [])
    with_mot = 0
    styles: dict[str, int] = {}
    for a in agents or []:
        mot = a.get("motivation") or {}
        if mot.get("values") or mot.get("drivers") or "Motivation:" in str(a.get("goal") or ""):
            with_mot += 1
        style = str(mot.get("decision_style") or "")
        if style:
            styles[style] = styles.get(style, 0) + 1
    return {
        "agent_count": n,
        "with_motivation": with_mot,
        "coverage": round(with_mot / max(1, n), 4),
        "decision_styles": dict(sorted(styles.items())),
        "engine": PERSONA_ENGINE,
        "quality_fix": QUALITY_FIX,
    }


def attach_motivations_to_agents(
    agents: list[Any],
    *,
    scenario: str = "",
) -> list[dict[str, Any]]:
    """Return agent dicts with ``motivation`` and enriched private goals."""
    out: list[dict[str, Any]] = []
    for raw in agents or []:
        if hasattr(raw, "to_dict"):
            d = raw.to_dict()
        elif isinstance(raw, dict):
            d = dict(raw)
        else:
            continue
        mot = build_persona_motivation(
            role_label=d.get("role_label"),
            occupation=d.get("occupation"),
            goal=str(d.get("goal") or ""),
            scenario=scenario,
            style=str(d.get("style") or ""),
        )
        d["motivation"] = mot.to_dict()
        d["goal"] = enrich_goal_with_motivation(str(d.get("goal") or ""), mot)
        out.append(d)
    return out


def persona_motivation_status() -> dict[str, Any]:
    return {
        "enabled": True,
        "engine": PERSONA_ENGINE,
        "quality_fix": QUALITY_FIX,
        "domains": sorted(_DOMAIN_MOTIVATIONS.keys()),
        "paid_llm_unlocked": False,
        "note": (
            "Persona Motivation Engine V4: Werte/Risiken/Treiber steuern "
            "private Ziele und Prompt-Konditionierung — offline."
        ),
    }
