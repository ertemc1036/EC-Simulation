"""Dynamic agent profiles from a scenario — no LLM required, no clones.

Phase 3 (MiroFish-ähnlich): generate N unique forum personas bound to the
scenario text. Offline/deterministic so tests never call a provider.

Modes:
- ``fixed_test`` — exactly the Phase-1/2 DEFAULT_AGENTS roster (5), for regression.
- ``dynamic`` — scenario-derived unique profiles (default for configurable counts).

For scenarios that name stakeholders (e.g. «Bürger, Unternehmen, Politiker und
Verkehrsbetriebe»), those roles become the primary agent identities.
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

# Role → persona framing when the scenario names stakeholders explicitly.
# ``conflict`` is a hard stance that clashes with other roles (Phase 3.5).
_ROLE_PERSONAS: dict[str, dict[str, str]] = {
    "bürger": {
        "name": "Bürgerstimme",
        "slug": "buerger",
        "description": "Spricht für Alltag, Mobilität und Lebensqualität der Stadtbevölkerung.",
        "goal": "Alltagsfolgen und Gerechtigkeit für Anwohnende sichtbar machen.",
        "conflict": (
            "Fordere soziale Abfederung und Ausnahmen für mobilitätseingeschränkte "
            "Menschen; widersprich Wirtschaftsargumenten ohne Alltagsbezug."
        ),
        "style": "direkt, erfahrungsbezogen, pragmatisch",
    },
    "buerger": {
        "name": "Bürgerstimme",
        "slug": "buerger",
        "description": "Spricht für Alltag, Mobilität und Lebensqualität der Stadtbevölkerung.",
        "goal": "Alltagsfolgen und Gerechtigkeit für Anwohnende sichtbar machen.",
        "conflict": (
            "Fordere soziale Abfederung und Ausnahmen für mobilitätseingeschränkte "
            "Menschen; widersprich Wirtschaftsargumenten ohne Alltagsbezug."
        ),
        "style": "direkt, erfahrungsbezogen, pragmatisch",
    },
    "unternehmen": {
        "name": "Unternehmensvertretung",
        "slug": "unternehmen",
        "description": "Vertritt Betriebe, Lieferketten und wirtschaftliche Planungssicherheit.",
        "goal": "Wirtschaftliche Risiken und praktikable Übergänge benennen.",
        "conflict": (
            "Fordere Lieferfenster, Ausnahmen und Übergangsfristen; "
            "widersprich pauschalen Verboten ohne Logistikkonzept."
        ),
        "style": "kostenbewusst, lösungsorientiert, nüchtern",
    },
    "politiker": {
        "name": "Politikstimme",
        "slug": "politiker",
        "description": "Wägt Mehrheiten, Recht und Umsetzbarkeit ab.",
        "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
        "conflict": (
            "Suche Mehrheitsfähigkeit und Kompromissformeln; markiere Konflikte "
            "zwischen Bürgerwünschen und Betriebs-/Kostengrenzen, ohne eine Seite "
            "komplett zu übernehmen."
        ),
        "style": "abwägend, öffentlichkeitsorientiert, strategisch",
    },
    "politik": {
        "name": "Politikstimme",
        "slug": "politik",
        "description": "Wägt Mehrheiten, Recht und Umsetzbarkeit ab.",
        "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
        "conflict": (
            "Suche Mehrheitsfähigkeit und Kompromissformeln; markiere Konflikte "
            "zwischen Bürgerwünschen und Betriebs-/Kostengrenzen, ohne eine Seite "
            "komplett zu übernehmen."
        ),
        "style": "abwägend, öffentlichkeitsorientiert, strategisch",
    },
    "verkehrsbetriebe": {
        "name": "Verkehrsbetrieb",
        "slug": "verkehrsbetriebe",
        "description": "Fokussiert Kapazität, Takt und Betrieb öffentlicher Verkehrsmittel.",
        "goal": "Betriebsfähigkeit und Ausbau des ÖPNV konkretisieren.",
        "conflict": (
            "Fordere Finanzierung und Kapazitätsausbau vor dem Stichtag; "
            "widersprich Terminen ohne belastbaren Betriebs- und Personalplan."
        ),
        "style": "betriebsnah, zahlenorientiert, dienstleistungsbezogen",
    },
    "anwohnende": {
        "name": "Anwohnendenstimme",
        "slug": "anwohnende",
        "description": "Betont Lärm, Sicherheit und Alltag vor der Haustür.",
        "goal": "Lokale Auswirkungen und Mitspracherechte stärken.",
        "conflict": (
            "Priorisiere Lärm-/Sicherheitsgewinne vor Durchgangsverkehr; "
            "widersprich Lieferinteressen, die den Wohnraum belasten."
        ),
        "style": "nahbar, kritisch, lokal",
    },
    "handel": {
        "name": "Handelsstimme",
        "slug": "handel",
        "description": "Vertritt Ladenlokale, Kundenfrequenz und Anlieferung.",
        "goal": "Erreichbarkeit und Umsätze im Blick behalten.",
        "conflict": (
            "Fordere kunden- und lieferfreundliche Zugänge; "
            "widersprich Regeln, die Frequenz und Anlieferung unzumutbar treffen."
        ),
        "style": "praktisch, kundenbezogen",
    },
}

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
        "einen",
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
        "wenn",
        "dann",
        "aber",
        "noch",
        "nur",
        "schon",
        "mehr",
        "sehr",
        "hier",
        "dort",
        "dazu",
        "dabei",
        "ab",
        "bis",
        "aus",
        "nach",
        "vor",
        "über",
        "unter",
        "zwischen",
        "innerhalb",
        "außerhalb",
        "passiert",
        "geschehen",
        "reagieren",
        "reaktion",
        "private",
        "privaten",
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
        "what",
        "when",
        "how",
        "if",
        "happens",
        "react",
        "reacts",
    }
)

# Weak fillers that often win token-order before real content nouns.
_WEAK_LEAD = frozenset(
    {
        "passiert",
        "wenn",
        "wie",
        "was",
        "wer",
        "ob",
        "dann",
        "dazu",
        "dabei",
        "geht",
        "gibt",
        "machen",
        "werden",
        "soll",
        "sollte",
        "kann",
        "muss",
        "private",
        "privaten",
        "innerhalb",
        "außerhalb",
    }
)


# Canonical display names for known stakeholder groups (incl. declined forms).
_ROLE_CANON: dict[str, str] = {
    "buerger": "Bürger",
    "buergern": "Bürger",
    "burger": "Bürger",
    "unternehmen": "Unternehmen",
    "politiker": "Politiker",
    "politik": "Politik",
    "verkehrsbetriebe": "Verkehrsbetriebe",
    "anwohnende": "Anwohnende",
    "anwohner": "Anwohnende",
    "handel": "Handel",
    "gewerbe": "Gewerbe",
    "behoerden": "Behörden",
    "behörden": "Behörden",
    "verwaltung": "Verwaltung",
    "polizei": "Polizei",
    "taxi": "Taxi",
    "lieferdienste": "Lieferdienste",
}

_DEFAULT_STAKEHOLDERS = ("Bürger", "Unternehmen", "Politik")

# Instruction / test-prompt stems that must never become agent names.
_INSTRUCTION_STEMS = (
    "untersuch",
    "simulier",
    "reagier",
    "entsteht",
    "zustimmung",
    "widerspruch",
    "argument",
    "beeinfluss",
    "ursprung",
    "positionen",
    "kompromiss",
    "entwickeln",
    "kontrovers",
    "debatte",
    "akteure",
    "konkret",
    "tragfahig",
    "tragfähig",
    "gemeinsam",
    "obzustimmung",
    "aufkonkret",
)


def _role_key(raw: str) -> str:
    return (
        raw.lower()
        .replace("ü", "ue")
        .replace("ö", "oe")
        .replace("ä", "ae")
        .replace("ß", "ss")
    )


def _is_instruction_fragment(token: str) -> bool:
    """Reject sentence fragments glued from test instructions as role names."""
    cleaned = (token or "").strip()
    if len(cleaned) < 3:
        return True
    if len(cleaned) > 28:
        return True
    # CamelCase / glued multi-word leftovers after stripping spaces.
    if re.search(r"[a-zäöüß][A-ZÄÖÜ]", cleaned):
        return True
    low = _role_key(cleaned)
    if low in _ROLE_CANON:
        return False
    if any(stem in low for stem in _INSTRUCTION_STEMS):
        return True
    # Multiple lowercase "words" glued without separators → prompt debris.
    if len(cleaned) >= 18 and cleaned.islower():
        return True
    return False


def extract_stakeholder_roles(scenario: str, *, limit: int = 5) -> list[str]:
    """Pull named stakeholder groups from the scenario text.

    Prefers lexicon hits (Bürger / Unternehmen / Politik) over free-text
    lists so test-instruction clauses never become agent names.
    """
    text = scenario or ""
    roles: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> bool:
        cleaned = re.sub(r"[^\wÄÖÜäöüß\-]+", "", raw, flags=re.UNICODE).strip("-")
        if len(cleaned) < 3:
            return False
        key = _role_key(cleaned)
        # Decline / alias → canonical display name when known.
        if key in _ROLE_CANON:
            cleaned = _ROLE_CANON[key]
            key = _role_key(cleaned)
        elif _is_instruction_fragment(cleaned):
            return False
        if key in _STOP or key in seen:
            return False
        if key not in _ROLE_CANON and _is_instruction_fragment(cleaned):
            return False
        seen.add(key)
        roles.append(cleaned)
        return True

    # 1) Lexicon first — order of appearance in the scenario.
    lexicon = [
        "Bürgern",
        "Bürger",
        "Buerger",
        "Unternehmen",
        "Politiker",
        "Politik",
        "Verkehrsbetriebe",
        "Anwohnende",
        "Anwohner",
        "Handel",
        "Gewerbe",
        "Behörden",
        "Verwaltung",
        "Polizei",
        "Taxi",
        "Lieferdienste",
    ]
    lower = text.lower()
    # Stable appearance order; word-boundary so «Politik» ≠ prefix of «Politiker».
    found: list[tuple[int, str]] = []
    for role in lexicon:
        pat = re.compile(
            rf"(?<![A-Za-zÄÖÜäöüß]){re.escape(role.lower())}(?![A-Za-zÄÖÜäöüß])",
            flags=re.IGNORECASE,
        )
        m = pat.search(lower)
        if m:
            found.append((m.start(), role))
    found.sort(key=lambda x: x[0])
    for _idx, role in found:
        _add(role)
        if len(roles) >= limit:
            return roles[:limit]

    # 2) «zwischen A, B und C» stakeholder lists (common in DE prompts).
    for match in re.finditer(
        r"zwischen\s+(.+?)(?:\.|\?|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        chunk = match.group(1)
        # Stop at clause verbs that start instruction tails.
        chunk = re.split(
            r"\b(?:untersuche|simuliere|prüfe|analysiere)\b",
            chunk,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        parts = re.split(r"\s*(?:,|/|;|\bund\b|\boder\b|\band\b|\bor\b)\s*", chunk)
        for part in parts:
            _add(part)
            if len(roles) >= limit:
                return roles[:limit]

    # 3) Tight role-list cues only («Wie reagieren A, B und C?» / «Akteure sind: …»).
    # Bare «Akteure …» is intentionally excluded — it matched instruction prose
    # in live run fbcc32b2 («Akteure auf konkrete Argumente …»).
    for match in re.finditer(
        r"(?:wie\s+reagieren|reaktion(?:en)?\s+(?:von|der|des)|"
        r"stimmen\s+(?:von|der)|akteure?\s+sind|stakeholders?\s+are|"
        r"react(?:s|ion)?\s+of)\s*[:\s]+(.+?)(?:\?|$|\.)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        chunk = match.group(1)
        parts = re.split(r"\s*(?:,|/|;|\bund\b|\boder\b|\band\b|\bor\b)\s*", chunk)
        for part in parts:
            _add(part)
            if len(roles) >= limit:
                return roles[:limit]

    # 4) Deterministic fallbacks when the prompt only names the debate topic.
    if not roles:
        for role in _DEFAULT_STAKEHOLDERS:
            _add(role)
            if len(roles) >= limit:
                break

    return roles[:limit]


def extract_content_keywords(scenario: str, *, limit: int = 4) -> list[str]:
    """Topic nouns from the scenario — excludes named stakeholder roles."""
    role_norms = {
        r.lower()
        .replace("ü", "ue")
        .replace("ö", "oe")
        .replace("ä", "ae")
        .replace("ß", "ss")
        for r in extract_stakeholder_roles(scenario, limit=8)
    }
    # Also drop singular/display forms used in role lists.
    role_norms.update(
        {
            "buerger",
            "buergern",
            "burger",
            "unternehmen",
            "politiker",
            "politik",
            "verkehrsbetriebe",
            "anwohnende",
            "handel",
        }
    )
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9-]{3,}", scenario or "")
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for idx, t in enumerate(tokens):
        low = t.lower()
        norm = (
            low.replace("ü", "ue")
            .replace("ö", "oe")
            .replace("ä", "ae")
            .replace("ß", "ss")
        )
        if low in _STOP or low in seen or norm in role_norms:
            continue
        seen.add(low)
        score = 0
        if t[:1].isupper():
            score += 3
        if any(ch.isdigit() for ch in t):
            score += 1
        if "-" in t or len(t) >= 7:
            score += 1
        if low in _WEAK_LEAD:
            score -= 4
        scored.append((score, -idx, t))

    scored.sort(reverse=True)
    out = [t for score, _neg_idx, t in scored if score > 0][:limit]
    if len(out) < limit:
        for _score, _neg_idx, t in scored:
            if t not in out:
                out.append(t)
            if len(out) >= limit:
                break
    return out or ["Diskussion"]


def extract_scenario_keywords(scenario: str, *, limit: int = 4) -> list[str]:
    """Pull distinctive tokens from the scenario for persona binding.

    Prefers content nouns (Berlin, Autos, …). Falls back to stakeholder role
    names only when the scenario has no usable content keywords.
    """
    content = extract_content_keywords(scenario, limit=limit)
    if content and content != ["Diskussion"]:
        return content[:limit]
    roles = extract_stakeholder_roles(scenario, limit=limit)
    if roles:
        return roles[:limit]
    return content


def _slugify(text: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    raw = (
        raw.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
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


def _persona_for_role(role: str, arch_fallback: dict[str, str]) -> dict[str, str]:
    key = (
        role.lower()
        .replace("ü", "ue")
        .replace("ö", "oe")
        .replace("ä", "ae")
        .replace("ß", "ss")
    )
    if key in _ROLE_PERSONAS:
        return dict(_ROLE_PERSONAS[key])
    # Generic role → blend with archetype voice but keep role identity.
    return {
        "name": role[:1].upper() + role[1:],
        "slug": _slugify(role) or arch_fallback["slug"],
        "description": (
            f"Vertritt die Perspektive «{role}» und prüft Folgen für diese Gruppe. "
            f"{arch_fallback['description']}"
        ),
        "goal": f"Interessen von {role} im Szenario klar vertreten. {arch_fallback['goal']}",
        "conflict": (
            f"Vertrete konsequent «{role}» gegen konkurrierende Stakeholder-Interessen; "
            f"keine ausgewogene Gesamtschau aller Seiten."
        ),
        "style": arch_fallback["style"],
    }


def _archetype_conflict(arch: dict[str, str]) -> str:
    """Opposing stance for filler archetypes when roles do not fill the roster."""
    slug = arch.get("slug") or ""
    mapping = {
        "analyst": "Stelle Behauptungen anderer Agenten infrage; verlange Belege statt Konsens.",
        "builder": "Dränge auf Pilotversuche und messbare Schritte; blockiere reine Prinzipiendebatten.",
        "moderator": "Zwinge gegensätzliche Positionen zur Klarheit; verhindere Scheinkonsens.",
        "contrarian": "Spitze Gegenpositionen zu; greife Mehrheitsannahmen direkt an.",
        "empath": "Stelle Betroffene und soziale Kosten über Effizienzargumente.",
    }
    return mapping.get(
        slug,
        "Halte eine klare Gegenposition zu anderen Stimmen; keine Allparteien-Zusammenfassung.",
    )


def generate_dynamic_agents(
    scenario: str,
    count: int,
    *,
    max_agents: int = MAX_GENERATABLE_AGENTS,
) -> list[AgentProfile]:
    """Build ``count`` unique scenario-bound profiles (no clones)."""
    n = max(1, min(int(count), int(max_agents), MAX_GENERATABLE_AGENTS))
    roles = extract_stakeholder_roles(scenario, limit=n)
    keywords = extract_scenario_keywords(scenario)
    topic = _topic_phrase(keywords)
    short = (scenario or "").strip()[:120] or "Offenes Diskussionsthema"
    profiles: list[AgentProfile] = []
    used_names: set[str] = set()
    used_user_names: set[str] = set()

    for i in range(n):
        arch = _ARCHETYPES[i % len(_ARCHETYPES)]
        if i < len(roles):
            persona = _persona_for_role(roles[i], arch)
            display = persona["name"]
            # Disambiguate if the same role maps twice.
            if display.lower() in used_names:
                display = f"{persona['name']} {i + 1}"
            user_name = f"{persona['slug']}_{_unique_suffix(short, i, persona['slug'])}"
            description = (
                f"{persona['description']} Bezieht Positionen explizit auf «{topic}» "
                f"im Szenario: {short}."
            )
            conflict = persona.get("conflict") or _archetype_conflict(arch)
            goal = f"{persona['goal']} Zielkonflikt: {conflict} Thema: {topic}."
            style = persona["style"]
        else:
            first = _FIRST_NAMES[(i + len(keywords)) % len(_FIRST_NAMES)]
            suffix = _unique_suffix(short, i, arch["slug"])
            display = f"{first} {arch['name']}"
            if display.lower() in used_names:
                display = f"{first} {arch['name']} {suffix}"
            user_name = f"{_slugify(first)}_{arch['slug']}_{suffix}"
            description = (
                f"{arch['description']} Bezieht Positionen explizit auf «{topic}» "
                f"im Szenario: {short}."
            )
            goal = (
                f"{arch['goal']} Zielkonflikt: {_archetype_conflict(arch)} "
                f"Thema: {topic}."
            )
            style = arch["style"]

        if user_name in used_user_names:
            user_name = f"{user_name}_{i}"

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
