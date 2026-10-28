"""Phase 8.5 — Stakeholder Motivation Engine V5 (offline-safe).

V4: values / fears / drivers / decision_style.
V5: stakes, claim_agenda, engagement_bias, stakeholder_group — so agents
prefer partners and claims from stakeholder interest, not only role labels.
No provider calls.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

PERSONA_ENGINE = "v5"
QUALITY_FIX = "8.5"

_DOMAIN_MOTIVATIONS: dict[str, dict[str, Any]] = {
    "education": {
        "lehrkraft": {
            "values": ["Lernqualität", "pädagogische Verantwortung", "Entlastung"],
            "fears": ["Technikzwang ohne Konzept", "Mehrbelastung", "Delegieren der Rolle"],
            "drivers": ["Evidenz", "Unterrichtskonzept", "Chancengleichheit im Klassenzimmer"],
            "decision_style": "evidence_first",
            "stakes": ["Unterrichtsalltag", "Rollenklarheit", "Arbeitsbelastung"],
            "claim_agenda": ["pädagogische Konzepte", "Evidenz", "keine Technikzwänge"],
            "engagement_bias": "prefer_evidence",
            "stakeholder_group": "practitioners",
        },
        "eltern": {
            "values": ["Chancengleichheit", "Kinderschutz", "Transparenz"],
            "fears": ["digitale Kluft", "ungeprüfte Vollausrollung", "Datenschutzlücken"],
            "drivers": ["Schutzregeln", "Elternbeteiligung", "Förderprogramme"],
            "decision_style": "protection_equity",
            "stakes": ["Kinderschutz", "Zugangsgerechtigkeit", "Datenschutz"],
            "claim_agenda": ["Datenschutz", "schrittweise Einführung", "Elternbeteiligung"],
            "engagement_bias": "prefer_opponents",
            "stakeholder_group": "affected_public",
        },
        "bildungsforscher": {
            "values": ["Empirie", "Evaluation", "Vergleichbarkeit"],
            "fears": ["anekdotische Erfolgsversprechen", "fehlende Langzeitdaten"],
            "drivers": ["Pilotprojekte", "Evaluationsstandards", "Risikofrüherkennung"],
            "decision_style": "evidence_first",
            "stakes": ["Datenqualität", "Evaluierbarkeit", "wissenschaftliche Integrität"],
            "claim_agenda": ["Pilotprojekte", "Evidenzlücken", "Langzeitdaten"],
            "engagement_bias": "prefer_evidence",
            "stakeholder_group": "experts",
        },
    },
    "labor": {
        "arbeitnehmer": {
            "values": ["Entlastung", "Lebensqualität", "Planbarkeit"],
            "fears": ["Mehrarbeit", "Scheinreform", "ungesicherte Umsetzung"],
            "drivers": ["volle Entlastung", "verbindliche Übergangsregeln"],
            "decision_style": "worker_protection",
            "stakes": ["Wochenarbeitszeit", "Lohnsicherheit", "Erholung"],
            "claim_agenda": ["volle Entlastung", "Übergangsregeln", "Belastungsschutz"],
            "engagement_bias": "prefer_opponents",
            "stakeholder_group": "affected_public",
        },
        "unternehmen": {
            "values": ["Machbarkeit", "Kostenkontrolle", "Betriebsabläufe"],
            "fears": ["Personalknappheit", "Kostensteigerung", "Produktionsausfälle"],
            "drivers": ["flexible Modelle", "Übergangsphasen"],
            "decision_style": "feasibility_first",
            "stakes": ["Betriebskontinuität", "Personalkosten", "Wettbewerbsfähigkeit"],
            "claim_agenda": ["Machbarkeit", "flexible Modelle", "Kostenkontrolle"],
            "engagement_bias": "prefer_allies",
            "stakeholder_group": "operators",
        },
        "wirtschaftsexperte": {
            "values": ["Produktivität", "Gegenfinanzierung", "Systemeffekte"],
            "fears": ["Makrorisiken", "unklare Finanzierung"],
            "drivers": ["Pilotindikatoren", "branchenspezifische Regeln"],
            "decision_style": "evidence_first",
            "stakes": ["Makrostabilität", "Produktivitätsmessung"],
            "claim_agenda": ["Produktivität", "Gegenfinanzierung", "Pilotindikatoren"],
            "engagement_bias": "prefer_evidence",
            "stakeholder_group": "experts",
        },
    },
    "income": {
        "arbeitnehmer": {
            "values": ["Absicherung", "Würde", "Planbarkeit"],
            "fears": ["Lohnentwertung", "unklare Anrechnung", "Sanktionsdruck"],
            "drivers": ["Kompatibilität mit Löhnen", "klare Regeln"],
            "decision_style": "protection_equity",
            "stakes": ["Existenzsicherung", "Lohnanreize", "Würde"],
            "claim_agenda": ["Absicherung", "klare Anrechnung", "Würde"],
            "engagement_bias": "prefer_opponents",
            "stakeholder_group": "affected_public",
        },
        "unternehmen": {
            "values": ["Finanzierbarkeit", "Anreize", "Planbarkeit"],
            "fears": ["Steuerlast", "Arbeitsanreizverluste"],
            "drivers": ["Gegenfinanzierung", "Pilotversuche"],
            "decision_style": "feasibility_first",
            "stakes": ["Steuerlast", "Arbeitsangebot", "Planbarkeit"],
            "claim_agenda": ["Finanzierbarkeit", "Anreize", "Pilotversuche"],
            "engagement_bias": "prefer_allies",
            "stakeholder_group": "operators",
        },
        "wirtschaftsexperte": {
            "values": ["Fiskalische Tragfähigkeit", "Empirie", "Anreize"],
            "fears": ["Inflation", "unklare Finanzierung"],
            "drivers": ["Evaluationsstandards", "Pilotindikatoren"],
            "decision_style": "evidence_first",
            "stakes": ["Fiskalstabilität", "Anreizwirkungen"],
            "claim_agenda": ["Tragfähigkeit", "Empirie", "Anreizwirkungen"],
            "engagement_bias": "prefer_evidence",
            "stakeholder_group": "experts",
        },
    },
    "mobility": {
        "buerger": {
            "values": ["Alltagsmobilität", "Erreichbarkeit", "Lebensqualität"],
            "fears": ["Ausgrenzung ohne Auto", "ÖPNV-Überlastung"],
            "drivers": ["verlässiger ÖPNV", "Übergangsfristen", "Ausnahmen"],
            "decision_style": "protection_equity",
            "stakes": ["Alltagsweg", "Erreichbarkeit", "Kosten"],
            "claim_agenda": ["ÖPNV-Kapazität", "Ausnahmen", "Übergangsfristen"],
            "engagement_bias": "prefer_opponents",
            "stakeholder_group": "affected_public",
        },
        "unternehmen": {
            "values": ["Lieferfähigkeit", "Kundenverkehr", "Standortattraktivität"],
            "fears": ["Logistikbruch", "Umsatzverlust"],
            "drivers": ["Ausnahmeregeln", "City-Logistik", "Übergangsphasen"],
            "decision_style": "feasibility_first",
            "stakes": ["Logistik", "Kundenfrequenz", "Standortkosten"],
            "claim_agenda": ["Lieferfenster", "Ausnahmen", "City-Logistik"],
            "engagement_bias": "prefer_allies",
            "stakeholder_group": "operators",
        },
        "politik": {
            "values": ["Umsetzbarkeit", "Akzeptanz", "Klimaziele"],
            "fears": ["Protest", "Umsetzungsstau"],
            "drivers": ["Pilotzonen", "Begleitmaßnahmen", "Kommunikation"],
            "decision_style": "balanced",
            "stakes": ["Akzeptanz", "Zielerreichung", "Verwaltungskapazität"],
            "claim_agenda": ["Pilotzonen", "Begleitmaßnahmen", "Akzeptanz"],
            "engagement_bias": "balanced",
            "stakeholder_group": "decision_makers",
        },
    },
}

_DEFAULT_MOTIVATION = {
    "values": ["Klarheit", "Fairness", "Belastbarkeit"],
    "fears": ["Scheinargumente", "überhastete Vollumsetzung"],
    "drivers": ["Belege", "Voraussetzungen", "messbare Schritte"],
    "decision_style": "balanced",
    "stakes": ["nachvollziehbare Regeln", "messbare Schritte"],
    "claim_agenda": ["Voraussetzungen", "Belege", "Risiken"],
    "engagement_bias": "balanced",
    "stakeholder_group": "generic",
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
    # Phase 8.5 Stakeholder Motivation V5
    stakes: list[str] = field(default_factory=list)
    claim_agenda: list[str] = field(default_factory=list)
    engagement_bias: str = "balanced"
    stakeholder_group: str = "generic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_block(self) -> str:
        """Compact German motivation block for system prompts."""
        vals = ", ".join(self.values[:3]) or "—"
        fears = ", ".join(self.fears[:3]) or "—"
        drivers = ", ".join(self.drivers[:3]) or "—"
        stakes = ", ".join(self.stakes[:3]) or "—"
        agenda = ", ".join(self.claim_agenda[:3]) or "—"
        return (
            f"Motivation: Werte [{vals}]; Risiken [{fears}]; "
            f"Treiber [{drivers}]; Stakes [{stakes}]; Agenda [{agenda}]; "
            f"Entscheidung={self.decision_style}; "
            f"Engagement={self.engagement_bias}; Gruppe={self.stakeholder_group}."
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
        ("bürger", "buerger"),
        ("buerger", "buerger"),
        ("einwohner", "buerger"),
        ("politik", "politik"),
        ("verwaltung", "politik"),
        ("stadt", "politik"),
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
    if any(
        t in low
        for t in ("auto-verbot", "s-bahn", "öpnv", "mobilitaet", "mobilität")
    ):
        return "mobility"
    # Role fallback only when scenario domain is unknown.
    if role_key in {"lehrkraft", "eltern", "bildungsforscher"}:
        return "education"
    if role_key in {"buerger", "politik"}:
        return "mobility"
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
    """Derive a V5 stakeholder motivation profile from role + scenario."""
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
        if base.get("engagement_bias") == "balanced":
            base["engagement_bias"] = "prefer_evidence"
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
        stakes=list(base.get("stakes") or [])[:4],
        claim_agenda=list(base.get("claim_agenda") or [])[:4],
        engagement_bias=str(base.get("engagement_bias") or "balanced"),
        stakeholder_group=str(base.get("stakeholder_group") or "generic"),
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


def stakeholder_affinity(
    a: dict[str, Any] | PersonaMotivation | None,
    b: dict[str, Any] | PersonaMotivation | None,
) -> float:
    """0..1 affinity for multi-agent pairing (same/near groups + shared stakes)."""
    def _as_dict(m: Any) -> dict[str, Any]:
        if m is None:
            return {}
        if isinstance(m, PersonaMotivation):
            return m.to_dict()
        if isinstance(m, dict):
            return m
        return {}

    da, db = _as_dict(a), _as_dict(b)
    if not da or not db:
        return 0.0
    score = 0.0
    if da.get("stakeholder_group") and da.get("stakeholder_group") == db.get(
        "stakeholder_group"
    ):
        score += 0.45
    if da.get("domain") and da.get("domain") == db.get("domain"):
        score += 0.2
    if da.get("decision_style") and da.get("decision_style") == db.get("decision_style"):
        score += 0.1
    stakes_a = {str(x).lower() for x in (da.get("stakes") or [])}
    stakes_b = {str(x).lower() for x in (db.get("stakes") or [])}
    agenda_a = {str(x).lower() for x in (da.get("claim_agenda") or [])}
    agenda_b = {str(x).lower() for x in (db.get("claim_agenda") or [])}
    if stakes_a and stakes_b:
        inter = len(stakes_a & stakes_b)
        union = len(stakes_a | stakes_b) or 1
        score += 0.15 * (inter / union)
    if agenda_a and agenda_b:
        inter = len(agenda_a & agenda_b)
        union = len(agenda_a | agenda_b) or 1
        score += 0.1 * (inter / union)
    return round(min(1.0, score), 4)


def motivation_coverage(agents: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of agents that carry V5 stakeholder motivation fields."""
    n = len(agents or [])
    with_mot = 0
    with_stakes = 0
    styles: dict[str, int] = {}
    groups: dict[str, int] = {}
    biases: dict[str, int] = {}
    for a in agents or []:
        mot = a.get("motivation") or {}
        if mot.get("values") or mot.get("drivers") or "Motivation:" in str(a.get("goal") or ""):
            with_mot += 1
        if mot.get("stakes") or mot.get("claim_agenda"):
            with_stakes += 1
        style = str(mot.get("decision_style") or "")
        if style:
            styles[style] = styles.get(style, 0) + 1
        group = str(mot.get("stakeholder_group") or "")
        if group:
            groups[group] = groups.get(group, 0) + 1
        bias = str(mot.get("engagement_bias") or "")
        if bias:
            biases[bias] = biases.get(bias, 0) + 1
    return {
        "agent_count": n,
        "with_motivation": with_mot,
        "with_stakes": with_stakes,
        "coverage": round(with_mot / max(1, n), 4),
        "stake_coverage": round(with_stakes / max(1, n), 4),
        "decision_styles": dict(sorted(styles.items())),
        "stakeholder_groups": dict(sorted(groups.items())),
        "engagement_biases": dict(sorted(biases.items())),
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
            "Stakeholder Motivation Engine V5: Werte/Risiken/Treiber plus "
            "Stakes, Claim-Agenda und Engagement-Bias — offline."
        ),
    }
