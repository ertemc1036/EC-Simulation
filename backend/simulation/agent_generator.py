"""Dynamic agent profiles from a scenario — no LLM required, no clones.

Phase 3 (MiroFish-ähnlich): generate N unique forum personas bound to the
scenario text. Offline/deterministic so tests never call a provider.

Phase 8.0 Persona Engine V2: education stakeholders (Lehrkräfte/Eltern/
Bildungsforscher), topic-noun denylist (Datenschutz≠Stakeholder), German
individual pools, declined-form canon — fixes live fixture ddfedb31.

Phase 8.1 Persona Diversity V3: cycle domain stakeholders at 10/25/50,
expanded education individual pools, unique names at scale.

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
from backend.simulation.topic_context import (
    INSTRUCTION_TOKEN_STEMS,
    detect_topic_domain,
    measure_label,
)

# Offline/Scripted society ceiling (Phase 6). Live-LLM still clamps to 5 in engine.
MAX_GENERATABLE_AGENTS = 50
LIVE_LLM_MAX_AGENTS = 5
OFFLINE_SCALE_PRESETS = (10, 25, 50)

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
    {
        "slug": "historian",
        "name": "Historiker",
        "description": "Vergleicht mit früheren Politikzyklen und Reformpfaden.",
        "goal": "Pfadabhängigkeiten und Lernmomente aus der Vergangenheit markieren.",
        "style": "kontextreich, vergleichend",
    },
    {
        "slug": "engineer",
        "name": "Ingenieur",
        "description": "Prüft technische und organisatorische Machbarkeit.",
        "goal": "Engpässe und realistische Umsetzungspfade benennen.",
        "style": "sachlich, spezifikationsnah",
    },
    {
        "slug": "journalist",
        "name": "Journalist",
        "description": "Fragt nach Belegen, Narrativen und öffentlichen Interessen.",
        "goal": "Klärungsfragen stellen und blinde Flecken der Debatte aufzeigen.",
        "style": "nachhakend, öffentlichkeitsorientiert",
    },
    {
        "slug": "activist",
        "name": "Aktivist",
        "description": "Drängt auf ambitionierte Ziele und schnelle Umsetzung.",
        "goal": "Druck für mutige Maßnahmen ohne Verwässerung aufbauen.",
        "style": "engagiert, mobilisierend",
    },
    {
        "slug": "skeptic",
        "name": "Skeptiker",
        "description": "Misstraut Modellannahmen und prognostizierten Nebenwirkungen.",
        "goal": "Überzogene Versprechen und versteckte Kosten entlarven.",
        "style": "kritisch, nüchtern",
    },
]


def _archetype_for_domain(arch: dict[str, str], domain: str) -> dict[str, str]:
    """Topic-specific archetype voice — no mobility leftovers on income theses."""
    out = dict(arch)
    slug = arch.get("slug") or ""
    if domain == "income":
        income_desc = {
            "historian": "Vergleicht mit früheren Sozialreformen und Transfermodellen.",
            "engineer": "Prüft Auszahlungslogistik, Verwaltungskapazität und IT-Systeme.",
            "activist": "Drängt auf zügige Einführung ohne Verwässerung der Maßnahme.",
            "skeptic": "Misstraut Finanzierungsannahmen und Arbeitsanreiz-Prognosen.",
            "builder": "Schlägt Piloten mit klaren Abbruchkriterien für Transfers vor.",
        }
        if slug in income_desc:
            out["description"] = income_desc[slug]
    elif domain == "labor":
        labor_desc = {
            "historian": "Vergleicht mit früheren Arbeitszeit- und Tarifreformen.",
            "engineer": "Prüft Schichtpläne, Kapazität und betriebliche Umsetzung.",
            "activist": "Drängt auf zügige Arbeitszeitverkürzung ohne Lohnverlust.",
            "skeptic": "Misstraut Produktivitäts- und Wettbewerbsversprechen.",
            "builder": "Schlägt Branchenpiloten mit messbaren Produktivitätskriterien vor.",
        }
        if slug in labor_desc:
            out["description"] = labor_desc[slug]
    elif domain == "mobility":
        mobility_desc = {
            "historian": "Vergleicht mit früheren Stadtentwicklungen und Politikzyklen.",
            "engineer": "Prüft technische Machbarkeit, Kapazität und Infrastruktur.",
        }
        if slug in mobility_desc:
            out["description"] = mobility_desc[slug]
    return out


# Role → persona framing when the scenario names stakeholders explicitly.
# Mobility-specific copy lives only under domain=="mobility".
_ROLE_PERSONAS_MOBILITY: dict[str, dict[str, str]] = {
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

# Income/BGE role voices — conflicts use ``{measure}`` (filled at generation).
# Never hardcode "Grundeinkommen" into non-income paths (Phase 7.3.1).
_ROLE_PERSONAS_INCOME: dict[str, dict[str, str]] = {
    "bürger": {
        "name": "Bürgerstimme",
        "slug": "buerger",
        "description": "Spricht für Alltagssicherheit und Teilhabe der Bevölkerung.",
        "goal": "Existenzsicherung und Gerechtigkeit für Haushalte sichtbar machen.",
        "conflict": (
            "Befürworte verlässliche Absicherung über «{measure}»; widersprich "
            "Kürzungsargumenten ohne Alltagsbezug."
        ),
        "style": "direkt, erfahrungsbezogen, pragmatisch",
    },
    "arbeitnehmer": {
        "name": "Arbeitnehmerstimme",
        "slug": "arbeitnehmer",
        "description": "Vertritt abhängig Beschäftigte und Lohn-Anreiz-Fragen.",
        "goal": "Arbeitsanreize und Absicherung fair ausbalancieren.",
        "conflict": (
            "Unterstütze «{measure}» als Absicherung, aber fordere klare "
            "Kompatibilität mit Löhnen; lehne Modelle ab, die Vollzeit entwerten."
        ),
        "style": "klar, lohnbezogen, pragmatisch",
    },
    "arbeitslose": {
        "name": "Arbeitslosenstimme",
        "slug": "arbeitslose",
        "description": "Spricht aus Sicht von Menschen ohne Erwerbsarbeit.",
        "goal": "Würde und Planbarkeit ohne Sanktionsdruck stärken.",
        "conflict": (
            "Befürworte «{measure}» als bedingungslose Absicherung; widersprich "
            "Sanktions- und Zwangsarbeit-Narrativen."
        ),
        "style": "direkt, menschenwürdig, fordernd",
    },
    "studierende": {
        "name": "Studierendenstimme",
        "slug": "studierende",
        "description": "Fokussiert Bildung, Nebenjobs und Studienfinanzierung.",
        "goal": "Studium ohne prekäre Nebenjobs ermöglichen.",
        "conflict": (
            "Unterstütze «{measure}» für Planungssicherheit im Studium; "
            "widersprich Altersgrenzen, die Studierende ausschließen."
        ),
        "style": "sachlich, zukunftsbezogen",
    },
    "unternehmer": {
        "name": "Unternehmerstimme",
        "slug": "unternehmer",
        "description": "Vertritt Gründer und Betriebsrisiken bei Transfers.",
        "goal": "Finanzierbarkeit und Fachkräfteanreize prüfen.",
        "conflict": (
            "Lehne ungeprüfte Vollauszahlung von «{measure}» ab; fordere Piloten "
            "und Gegenfinanzierung; widersprich Versprechen ohne Steuerkonzept."
        ),
        "style": "kostenbewusst, nüchtern",
    },
    "unternehmen": {
        "name": "Unternehmensvertretung",
        "slug": "unternehmen",
        "description": "Vertritt Betriebe bei Lohnkosten und Planungssicherheit.",
        "goal": "Wirtschaftliche Risiken und praktikable Übergänge benennen.",
        "conflict": (
            "Fordere Übergangsregeln und Gegenfinanzierung für «{measure}»; "
            "widersprich pauschalen Transfers ohne Arbeitsmarktanalyse."
        ),
        "style": "kostenbewusst, lösungsorientiert, nüchtern",
    },
    "rentner": {
        "name": "Rentnerstimme",
        "slug": "rentner",
        "description": "Betont Altersarmut und Kaufkraft im Ruhestand.",
        "goal": "Altersarmut mindern ohne Rentenlogik zu zerstören.",
        "conflict": (
            "Befürworte «{measure}» als Aufstockung bei Niedrigrenten; widersprich "
            "Kürzungen bestehender Ansprüche."
        ),
        "style": "ruhig, gerechtigkeitsbezogen",
    },
    "selbststaendige": {
        "name": "Selbstständigenstimme",
        "slug": "selbststaendige",
        "description": "Spricht für Solo-Selbstständige und Auftragsschwankungen.",
        "goal": "Einkommensschwankungen absichern ohne Bürokratiechaos.",
        "conflict": (
            "Unterstütze «{measure}» als stabile Basisabsicherung; fordere klare "
            "Regeln für Selbstständige; lehne Modelle ab, die Beiträge verdoppeln."
        ),
        "style": "praktisch, risikobewusst",
    },
    "politiker": {
        "name": "Politikstimme",
        "slug": "politiker",
        "description": "Wägt Mehrheiten, Recht und Haushaltsregeln ab.",
        "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
        "conflict": (
            "Suche Mehrheitsfähigkeit für «{measure}»; markiere Konflikte "
            "zwischen sozialer Absicherung und Haushaltsgrenzen."
        ),
        "style": "abwägend, öffentlichkeitsorientiert, strategisch",
    },
    "politik": {
        "name": "Politikstimme",
        "slug": "politik",
        "description": "Wägt Mehrheiten, Recht und Haushaltsregeln ab.",
        "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
        "conflict": (
            "Suche Mehrheitsfähigkeit für «{measure}»; markiere Konflikte "
            "zwischen sozialer Absicherung und Haushaltsgrenzen."
        ),
        "style": "abwägend, öffentlichkeitsorientiert, strategisch",
    },
    "oekonomen": {
        "name": "Ökonomenstimme",
        "slug": "oekonomen",
        "description": "Prüft Anreiz-, Inflations- und Finanzierungseffekte.",
        "goal": "Empirische Risiken und Annahmen offenlegen.",
        "conflict": (
            "Stelle Finanzierungs- und Inflationsrisiken von «{measure}» heraus; "
            "unterstütze nur evidenzbasierte Piloten; widersprich utopischen Annahmen."
        ),
        "style": "analytisch, vorsichtig",
    },
    "oekonom": {
        "name": "Ökonomenstimme",
        "slug": "oekonom",
        "description": "Prüft Anreiz-, Inflations- und Finanzierungseffekte.",
        "goal": "Empirische Risiken und Annahmen offenlegen.",
        "conflict": (
            "Stelle Finanzierungs- und Inflationsrisiken von «{measure}» heraus; "
            "unterstütze nur evidenzbasierte Piloten; widersprich utopischen Annahmen."
        ),
        "style": "analytisch, vorsichtig",
    },
}

# Education / school AI roles (Phase 8.0 — ddfedb31).
_ROLE_PERSONAS_EDUCATION: dict[str, dict[str, str]] = {
    "lehrkraefte": {
        "name": "Lehrkraft",
        "slug": "lehrkraefte",
        "description": "Spricht aus Sicht unterrichtender Lehrkräfte an öffentlichen Schulen.",
        "goal": "Lernqualität, Arbeitsbelastung und pädagogische Verantwortung absichern.",
        "conflict": (
            "Unterstütze «{measure}» nur mit belastbarer Evidenz und Entlastung; "
            "widersprich Technikversprechen ohne Unterrichtskonzept."
        ),
        "style": "pädagogisch, evidenzbezogen, klar",
    },
    "eltern": {
        "name": "Elternstimme",
        "slug": "eltern",
        "description": "Vertritt Erziehungsberechtigte zu Chancen, Schutz und Alltag der Kinder.",
        "goal": "Chancengleichheit und kindgerechten Einsatz der Maßnahme fordern.",
        "conflict": (
            "Befürworte «{measure}» bei klaren Qualitäts- und Schutzregeln; "
            "widersprich ungeprüfter Vollausrollung ohne Elternbeteiligung."
        ),
        "style": "fürsorglich, pragmatisch, nachhakend",
    },
    "bildungsforscher": {
        "name": "Bildungsforscher",
        "slug": "bildungsforscher",
        "description": "Prüft Lernwirksamkeit, Evaluation und Systemeffekte empirisch.",
        "goal": "Empirische Risiken und Evaluationsstandards offenlegen.",
        "conflict": (
            "Stelle Evidenzlücken von «{measure}» heraus; unterstütze nur "
            "begleitete Piloten; widersprich anekdotischen Erfolgsversprechen."
        ),
        "style": "analytisch, vorsichtig, methodennah",
    },
    "schueler": {
        "name": "Schülerstimme",
        "slug": "schueler",
        "description": "Spricht aus Alltagssicht Lernender an Schulen.",
        "goal": "Lernalltag, Druck und echte Hilfen sichtbar machen.",
        "conflict": (
            "Bewerte «{measure}» danach, ob Lernen leichter und fairer wird; "
            "widersprich Tools, die nur Kontrolle oder Mehrarbeit bringen."
        ),
        "style": "direkt, alltagsnah",
    },
}

# Labor / working-time roles — no BGE vocabulary (Vier-Tage-Woche etc.).
_ROLE_PERSONAS_LABOR: dict[str, dict[str, str]] = {
    "arbeitnehmer": {
        "name": "Arbeitnehmerstimme",
        "slug": "arbeitnehmer",
        "description": "Vertritt abhängig Beschäftigte zu Arbeitszeit und Lohn.",
        "goal": "Arbeitszeitverkürzung ohne Einkommensverlust absichern.",
        "conflict": (
            "Unterstütze «{measure}» für Lebensqualität, aber fordere echten "
            "Lohnausgleich; lehne Modelle ab, die Leistungsdruck erhöhen."
        ),
        "style": "klar, lohnbezogen, pragmatisch",
    },
    "unternehmer": {
        "name": "Unternehmerstimme",
        "slug": "unternehmer",
        "description": "Vertritt Betriebe zu Produktivität und Personalkosten.",
        "goal": "Wettbewerbsfähigkeit und Fachkräftebindung prüfen.",
        "conflict": (
            "Lehne ungeprüfte Einführung von «{measure}» ab; fordere Branchenpiloten "
            "und Gegenfinanzierung; widersprich Versprechen ohne Produktivitätskonzept."
        ),
        "style": "kostenbewusst, nüchtern",
    },
    "unternehmen": {
        "name": "Unternehmensvertretung",
        "slug": "unternehmen",
        "description": "Vertritt die Unternehmensleitung bei Schichtplanung und Kosten.",
        "goal": "Betriebliche Risiken und praktikable Übergänge benennen.",
        "conflict": (
            "Fordere Übergangsregeln für «{measure}»; widersprich pauschalen "
            "Arbeitszeitregeln ohne Kapazitätsanalyse."
        ),
        "style": "kostenbewusst, lösungsorientiert, nüchtern",
    },
    "betriebsrat": {
        "name": "Betriebsratsstimme",
        "slug": "betriebsrat",
        "description": "Vertritt die Belegschaft in Betriebsvereinbarungen und Schichtfragen.",
        "goal": "Beschäftigtenrechte und Mitbestimmung bei Arbeitszeit sichern.",
        "conflict": (
            "Unterstütze «{measure}» nur mit echtem Lohnausgleich und Mitbestimmung; "
            "widersprich Modellen, die Leistungsdruck ohne Entlastung erhöhen."
        ),
        "style": "mitarbeiterbezogen, tarifklar, pragmatisch",
    },
    "oekonomen": {
        "name": "Wirtschaftsexperte",
        "slug": "oekonomen",
        "description": "Prüft Produktivitäts-, Beschäftigungs- und Kosteneffekte am Arbeitsmarkt.",
        "goal": "Empirische Risiken und Annahmen offenlegen.",
        "conflict": (
            "Stelle Produktivitäts- und Wettbewerbsrisiken von «{measure}» heraus; "
            "unterstütze nur evidenzbasierte Piloten; widersprich utopischen Annahmen."
        ),
        "style": "analytisch, vorsichtig",
    },
    "oekonom": {
        "name": "Wirtschaftsexperte",
        "slug": "oekonom",
        "description": "Prüft Produktivitäts-, Beschäftigungs- und Kosteneffekte am Arbeitsmarkt.",
        "goal": "Empirische Risiken und Annahmen offenlegen.",
        "conflict": (
            "Stelle Produktivitäts- und Wettbewerbsrisiken von «{measure}» heraus; "
            "unterstütze nur evidenzbasierte Piloten; widersprich utopischen Annahmen."
        ),
        "style": "analytisch, vorsichtig",
    },
    "wirtschaftsexperten": {
        "name": "Wirtschaftsexperte",
        "slug": "oekonomen",
        "description": "Prüft Produktivitäts-, Beschäftigungs- und Kosteneffekte am Arbeitsmarkt.",
        "goal": "Empirische Risiken und Annahmen offenlegen.",
        "conflict": (
            "Stelle Produktivitäts- und Wettbewerbsrisiken von «{measure}» heraus; "
            "unterstütze nur evidenzbasierte Piloten; widersprich utopischen Annahmen."
        ),
        "style": "analytisch, vorsichtig",
    },
    "politiker": {
        "name": "Politikstimme",
        "slug": "politiker",
        "description": "Wägt Mehrheiten, Recht und Arbeitsmarktregeln ab.",
        "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
        "conflict": (
            "Suche Mehrheitsfähigkeit für «{measure}»; markiere Konflikte zwischen "
            "Lebensqualität und Wettbewerbsfähigkeit."
        ),
        "style": "abwägend, öffentlichkeitsorientiert, strategisch",
    },
    "politik": {
        "name": "Politikstimme",
        "slug": "politik",
        "description": "Wägt Mehrheiten, Recht und Arbeitsmarktregeln ab.",
        "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
        "conflict": (
            "Suche Mehrheitsfähigkeit für «{measure}»; markiere Konflikte zwischen "
            "Lebensqualität und Wettbewerbsfähigkeit."
        ),
        "style": "abwägend, öffentlichkeitsorientiert, strategisch",
    },
    "bürger": {
        "name": "Bürgerstimme",
        "slug": "buerger",
        "description": "Spricht für Alltag, Freizeit und Lebensqualität.",
        "goal": "Work-Life-Balance und Alltagsfolgen sichtbar machen.",
        "conflict": (
            "Befürworte «{measure}» für mehr freie Zeit; widersprich Argumenten "
            "ohne Alltagsbezug."
        ),
        "style": "direkt, erfahrungsbezogen, pragmatisch",
    },
}

# Backward-compatible alias (mobility defaults for older imports/tests).
_ROLE_PERSONAS: dict[str, dict[str, str]] = {
    **_ROLE_PERSONAS_MOBILITY,
    "buerger": dict(_ROLE_PERSONAS_MOBILITY["bürger"]),
}

# Phase 8.0: German given/family names for unknown-role fallbacks (no EN clones).
_FIRST_NAMES = [
    "Lena",
    "Jonas",
    "Mara",
    "Tim",
    "Ayla",
    "Samir",
    "Clara",
    "Emre",
    "Nora",
    "Lars",
    "Vera",
    "Holger",
    "Ines",
    "Ralf",
    "Silke",
    "Gero",
    "Heike",
    "Udo",
    "Maja",
    "Greta",
    "Paul",
    "Sibel",
    "Uwe",
    "Miriam",
    "Dirk",
    "Beate",
    "Murat",
    "Ilona",
    "Klaus",
    "Sabine",
    "Ahmed",
    "Petra",
    "Martin",
    "Judith",
    "Stefan",
    "Yasmin",
    "Bernd",
    "Hassan",
    "Monika",
    "Igor",
    "Kai",
    "Elena",
    "Ben",
    "Anke",
    "Jan",
    "Lea",
    "Omar",
    "Nina",
    "Felix",
    "Tobias",
]
_LAST_NAMES = [
    "Vogt",
    "Berg",
    "Demir",
    "Keller",
    "Richter",
    "Hahn",
    "Kohn",
    "Neumann",
    "Otto",
    "Yilmaz",
    "Weiss",
    "Krämer",
    "Stein",
    "Brandt",
    "Fuchs",
    "Meissner",
    "Horn",
    "Lang",
    "Braun",
    "Kranz",
    "Vogel",
    "Lind",
    "Mertens",
    "Acar",
    "Hartmann",
    "König",
    "Sommer",
    "Celik",
    "Beyer",
    "Weber",
    "Roth",
    "Farouk",
    "Lange",
    "Koch",
    "Frei",
    "Wolf",
    "Said",
    "Tietz",
    "Ali",
    "Schulz",
    "Petrov",
    "Nguyen",
    "Popov",
    "Adler",
    "Meier",
    "Hofmann",
    "Graf",
    "Busch",
    "Engel",
    "Krug",
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
    "unternehmer": "Unternehmer",
    "unternehmern": "Unternehmer",
    "politiker": "Politiker",
    "politikern": "Politiker",
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
    "arbeitnehmer": "Arbeitnehmer",
    "arbeitnehmern": "Arbeitnehmer",
    "arbeitslose": "Arbeitslose",
    "arbeitslosen": "Arbeitslose",
    "studierende": "Studierende",
    "studierenden": "Studierende",
    "studenten": "Studierende",
    "rentner": "Rentner",
    "rentnern": "Rentner",
    "selbststaendige": "Selbstständige",
    "selbständige": "Selbstständige",
    "oekonomen": "Ökonomen",
    "okonomen": "Ökonomen",
    "oekonom": "Ökonomen",
    "ökonom": "Ökonomen",
    "ökonomen": "Ökonomen",
    # Live-LLM often says «Wirtschaftsexperten» — map to economists, not abstract group.
    "wirtschaftsexperten": "Ökonomen",
    "wirtschaftsexperte": "Ökonomen",
    "wirtschaftswissenschaftler": "Ökonomen",
    # Live labor prompts («… und den Arbeitsmarkt») → economic expert, not Moderator.
    "arbeitsmarkt": "Ökonomen",
    "betriebsrat": "Betriebsrat",
    "betriebsraete": "Betriebsrat",
    "betriebsräte": "Betriebsrat",
    # Phase 8.0 education stakeholders (ddfedb31).
    "lehrkraefte": "Lehrkräfte",
    "lehrkraft": "Lehrkräfte",
    "lehrkraften": "Lehrkräfte",
    "lehrkraeften": "Lehrkräfte",
    "lehrer": "Lehrkräfte",
    "lehrerinnen": "Lehrkräfte",
    "eltern": "Eltern",
    "elternteil": "Eltern",
    "erziehungsberechtigte": "Eltern",
    "bildungsforscher": "Bildungsforscher",
    "bildungsforschern": "Bildungsforscher",
    "bildungsforscherinnen": "Bildungsforscher",
    "bildungsexperten": "Bildungsforscher",
    "schueler": "Schüler",
    "schüler": "Schüler",
    "schuelern": "Schüler",
    "schülern": "Schüler",
}

# Debate themes / metrics — never agent identities (ddfedb31: Datenschutz…).
_TOPIC_THEME_DENY = frozenset(
    {
        "datenschutz",
        "chancengleichheit",
        "lernleistungen",
        "lernleistung",
        "privatheit",
        "privacy",
        "rolle",
        "rollen",
        "qualität",
        "qualitaet",
        "evidenz",
        "potenzial",
        "risiken",
        "chancen",
        "bedingungen",
        "auswirkungen",
        "diskussion",
        "debatte",
        "thema",
        "maßnahme",
        "massnahme",
    }
)

# Phase 7.3 — realistic individuals (name/age/occupation) per role; deterministic picks.
_INDIVIDUALS_BY_ROLE: dict[str, list[dict[str, object]]] = {
    "buerger": [
        {"first": "Lena", "last": "Vogt", "age": 42, "occupation": "Verwaltungsangestellte"},
        {"first": "Jonas", "last": "Berg", "age": 29, "occupation": "Fahrradkurier"},
        {"first": "Ayla", "last": "Demir", "age": 51, "occupation": "Einzelhandelskauffrau"},
    ],
    "arbeitnehmer": [
        {"first": "Mara", "last": "Keller", "age": 34, "occupation": "Pflegefachkraft"},
        {"first": "Tim", "last": "Richter", "age": 45, "occupation": "Industrieelektroniker"},
        {"first": "Nora", "last": "Hahn", "age": 28, "occupation": "Callcenter-Mitarbeiterin"},
    ],
    "arbeitslose": [
        {"first": "Samir", "last": "Kohn", "age": 39, "occupation": "Arbeitssuchender"},
        {"first": "Pia", "last": "Neumann", "age": 33, "occupation": "Umschülerin"},
        {"first": "Lars", "last": "Otto", "age": 52, "occupation": "Langzeitarbeitsloser"},
    ],
    "studierende": [
        {"first": "Emre", "last": "Yilmaz", "age": 22, "occupation": "Politikstudent"},
        {"first": "Clara", "last": "Weiss", "age": 24, "occupation": "Medizinstudentin"},
        {"first": "Jonas", "last": "Krämer", "age": 21, "occupation": "Informatikstudent"},
    ],
    "unternehmer": [
        {"first": "Vera", "last": "Stein", "age": 47, "occupation": "KMU-Gründerin"},
        {"first": "Holger", "last": "Brandt", "age": 54, "occupation": "Handwerksmeister"},
        {"first": "Ines", "last": "Fuchs", "age": 38, "occupation": "Software-Unternehmerin"},
    ],
    # Company-side occupations only — Betriebsrat is employee representation (7.3.2).
    "unternehmen": [
        {"first": "Ralf", "last": "Meissner", "age": 49, "occupation": "Betriebsleiter"},
        {"first": "Silke", "last": "Horn", "age": 43, "occupation": "Personalchefin"},
        {"first": "Gero", "last": "Lang", "age": 56, "occupation": "Logistikleiter"},
    ],
    "betriebsrat": [
        {"first": "Heike", "last": "Braun", "age": 48, "occupation": "Betriebsratsvorsitzende"},
        {"first": "Udo", "last": "Kranz", "age": 55, "occupation": "Betriebsrat"},
        {"first": "Maja", "last": "Vogel", "age": 41, "occupation": "Personalrätin"},
    ],
    "rentner": [
        {"first": "Helga", "last": "Pohl", "age": 71, "occupation": "Rentnerin"},
        {"first": "Werner", "last": "Kraus", "age": 68, "occupation": "Frührentner"},
        {"first": "Inge", "last": "Seidel", "age": 74, "occupation": "Rentnerin"},
    ],
    "selbststaendige": [
        {"first": "Omar", "last": "Said", "age": 36, "occupation": "Freiberuflicher Designer"},
        {"first": "Nina", "last": "Graf", "age": 41, "occupation": "Solo-Selbstständige"},
        {"first": "Felix", "last": "Busch", "age": 32, "occupation": "IT-Freelancer"},
    ],
    "politiker": [
        {"first": "Karin", "last": "Wolff", "age": 48, "occupation": "Stadträtin"},
        {"first": "Tobias", "last": "Engel", "age": 39, "occupation": "Landtagsabgeordneter"},
        {"first": "Svenja", "last": "Krug", "age": 44, "occupation": "Bezirksbürgermeisterin"},
    ],
    "politik": [
        {"first": "Karin", "last": "Wolff", "age": 48, "occupation": "Stadträtin"},
        {"first": "Tobias", "last": "Engel", "age": 39, "occupation": "Landtagsabgeordneter"},
        {"first": "Svenja", "last": "Krug", "age": 44, "occupation": "Bezirksbürgermeisterin"},
    ],
    "oekonomen": [
        {"first": "Dr. Anke", "last": "Meier", "age": 46, "occupation": "Arbeitsmarktforscherin"},
        {"first": "Prof. Jan", "last": "Schulz", "age": 53, "occupation": "Finanzwissenschaftler"},
        {"first": "Lea", "last": "Hofmann", "age": 35, "occupation": "Wirtschaftsanalystin"},
    ],
    "oekonom": [
        {"first": "Dr. Anke", "last": "Meier", "age": 46, "occupation": "Arbeitsmarktforscherin"},
        {"first": "Prof. Jan", "last": "Schulz", "age": 53, "occupation": "Finanzwissenschaftler"},
        {"first": "Lea", "last": "Hofmann", "age": 35, "occupation": "Wirtschaftsanalystin"},
    ],
    "anwohnende": [
        {"first": "Greta", "last": "Lind", "age": 37, "occupation": "Anwohnerin"},
        {"first": "Paul", "last": "Mertens", "age": 58, "occupation": "Anwohner"},
        {"first": "Sibel", "last": "Acar", "age": 31, "occupation": "Mieterin"},
    ],
    "verkehrsbetriebe": [
        {"first": "Uwe", "last": "Hartmann", "age": 50, "occupation": "ÖPNV-Planer"},
        {"first": "Miriam", "last": "König", "age": 34, "occupation": "Fahrdienstleiterin"},
        {"first": "Dirk", "last": "Sommer", "age": 45, "occupation": "Betriebsleiter Nahverkehr"},
    ],
    "handel": [
        {"first": "Beate", "last": "König", "age": 52, "occupation": "Ladeninhaberin"},
        {"first": "Murat", "last": "Celik", "age": 40, "occupation": "Supermarktfilialleiter"},
        {"first": "Ilona", "last": "Beyer", "age": 47, "occupation": "Einzelhändlerin"},
    ],
    "gewerbe": [
        {"first": "Klaus", "last": "Weber", "age": 55, "occupation": "Gewerbetreibender"},
        {"first": "Sabine", "last": "Roth", "age": 42, "occupation": "Handwerksbetriebsleiterin"},
        {"first": "Ahmed", "last": "Farouk", "age": 38, "occupation": "Gastronom"},
    ],
    "behoerden": [
        {"first": "Petra", "last": "Lange", "age": 49, "occupation": "Amtsleiterin"},
        {"first": "Martin", "last": "Koch", "age": 44, "occupation": "Ordnungsamtsmitarbeiter"},
        {"first": "Judith", "last": "Frei", "age": 36, "occupation": "Sachbearbeiterin"},
    ],
    "verwaltung": [
        {"first": "Petra", "last": "Lange", "age": 49, "occupation": "Amtsleiterin"},
        {"first": "Martin", "last": "Koch", "age": 44, "occupation": "Ordnungsamtsmitarbeiter"},
        {"first": "Judith", "last": "Frei", "age": 36, "occupation": "Sachbearbeiterin"},
    ],
    "polizei": [
        {"first": "Stefan", "last": "Wolf", "age": 41, "occupation": "Polizeihauptkommissar"},
        {"first": "Yasmin", "last": "Said", "age": 33, "occupation": "Polizeikommissarin"},
        {"first": "Bernd", "last": "Tietz", "age": 52, "occupation": "Verkehrspolizist"},
    ],
    "taxi": [
        {"first": "Hassan", "last": "Ali", "age": 46, "occupation": "Taxifahrer"},
        {"first": "Monika", "last": "Schulz", "age": 39, "occupation": "Taxiunternehmerin"},
        {"first": "Igor", "last": "Petrov", "age": 51, "occupation": "Taxifahrer"},
    ],
    "lieferdienste": [
        {"first": "Kai", "last": "Nguyen", "age": 26, "occupation": "Fahrradkurier"},
        {"first": "Elena", "last": "Popov", "age": 30, "occupation": "Lieferdienst-Dispatcherin"},
        {"first": "Ben", "last": "Adler", "age": 23, "occupation": "App-Lieferfahrer"},
    ],
    # Phase 8.0/8.1 education (fixture ddfedb31 + scale pools).
    "lehrkraefte": [
        {"first": "Sabine", "last": "Keller", "age": 44, "occupation": "Gymnasiallehrerin"},
        {"first": "Markus", "last": "Brandt", "age": 51, "occupation": "Realschullehrer"},
        {"first": "Fatima", "last": "Hassan", "age": 37, "occupation": "Grundschullehrerin"},
        {"first": "Henning", "last": "Voigt", "age": 48, "occupation": "Berufsschullehrer"},
        {"first": "Aylin", "last": "Öztürk", "age": 33, "occupation": "Förderschullehrerin"},
        {"first": "Claudia", "last": "Mertens", "age": 55, "occupation": "Schulleiterin"},
        {"first": "Björn", "last": "Hansen", "age": 42, "occupation": "Physiklehrer"},
        {"first": "Nadine", "last": "Scholz", "age": 29, "occupation": "Referendarin"},
    ],
    "eltern": [
        {"first": "Thomas", "last": "Bauer", "age": 41, "occupation": "Elternteil / IT-Projektleiter"},
        {"first": "Sandra", "last": "Klein", "age": 38, "occupation": "Mutter / Pflegefachkraft"},
        {"first": "Diego", "last": "Morales", "age": 45, "occupation": "Vater / Handwerksmeister"},
        {"first": "Katrin", "last": "Engel", "age": 36, "occupation": "Mutter / Buchhalterin"},
        {"first": "Omar", "last": "Farouk", "age": 43, "occupation": "Vater / Taxifahrer"},
        {"first": "Birgit", "last": "Lange", "age": 50, "occupation": "Mutter / Verwaltungsangestellte"},
        {"first": "Stefan", "last": "Krüger", "age": 39, "occupation": "Vater / Lagerist"},
        {"first": "Yasemin", "last": "Yilmaz", "age": 34, "occupation": "Mutter / Ärztin"},
    ],
    "bildungsforscher": [
        {
            "first": "Prof. Mira",
            "last": "Hoffmann",
            "age": 52,
            "occupation": "Bildungsforscherin",
        },
        {
            "first": "Dr. Lars",
            "last": "Peters",
            "age": 47,
            "occupation": "Schulforscher",
        },
        {
            "first": "Dr. Yasmin",
            "last": "Kader",
            "age": 39,
            "occupation": "Bildungssoziologin",
        },
        {
            "first": "Prof. Erik",
            "last": "Neumann",
            "age": 58,
            "occupation": "Empirischer Bildungsforscher",
        },
        {
            "first": "Dr. Hannah",
            "last": "Vogel",
            "age": 41,
            "occupation": "Digitalisierungsforscherin Bildung",
        },
        {
            "first": "Dr. Cem",
            "last": "Aydın",
            "age": 44,
            "occupation": "Bildungsökonom",
        },
    ],
    "schueler": [
        {"first": "Lea", "last": "Sommer", "age": 16, "occupation": "Schülerin Klasse 10"},
        {"first": "Jonas", "last": "Krüger", "age": 15, "occupation": "Schüler Klasse 9"},
        {"first": "Amira", "last": "Said", "age": 17, "occupation": "Schülerin Klasse 11"},
        {"first": "Tom", "last": "Becker", "age": 14, "occupation": "Schüler Klasse 8"},
        {"first": "Maya", "last": "Nguyen", "age": 16, "occupation": "Schülerin Klasse 10"},
        {"first": "Luis", "last": "Hoffmann", "age": 15, "occupation": "Schüler Klasse 9"},
    ],
}

_DEFAULT_STAKEHOLDERS = ("Bürger", "Unternehmen", "Politik")
_DEFAULT_STAKEHOLDERS_INCOME = (
    "Arbeitnehmer",
    "Arbeitslose",
    "Studierende",
    "Unternehmer",
    "Rentner",
    "Selbstständige",
    "Politiker",
    "Ökonomen",
)
_DEFAULT_STAKEHOLDERS_LABOR = (
    "Arbeitnehmer",
    "Unternehmer",
    "Ökonomen",
)
_DEFAULT_STAKEHOLDERS_EDUCATION = (
    "Lehrkräfte",
    "Eltern",
    "Bildungsforscher",
    "Schüler",
)

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
        # Debate themes (Datenschutz, Chancengleichheit, …) are never stakeholders.
        if key in _TOPIC_THEME_DENY:
            return False
        # Decline / alias → canonical display name when known.
        if key in _ROLE_CANON:
            cleaned = _ROLE_CANON[key]
            key = _role_key(cleaned)
        elif _is_instruction_fragment(cleaned):
            return False
        if key in _STOP or key in _TOPIC_THEME_DENY or key in seen:
            return False
        if key not in _ROLE_CANON and _is_instruction_fragment(cleaned):
            return False
        # Unknown free-text fragments that look like topic nouns (no person/group cue).
        if key not in _ROLE_CANON and key.endswith(
            ("schutz", "gleichheit", "leistung", "leistungen", "qualität", "qualitaet")
        ):
            return False
        seen.add(key)
        roles.append(cleaned)
        return True

    domain = detect_topic_domain(text)

    # 1) Lexicon first — order of appearance in the scenario.
    # Longer / more specific forms before shorter prefixes (Unternehmer > Unternehmen).
    lexicon = [
        "Selbstständige",
        "Selbständige",
        "Arbeitnehmer",
        "Arbeitslose",
        "Studierende",
        "Studenten",
        "Unternehmer",
        "Unternehmen",
        "Wirtschaftsexperten",
        "Wirtschaftsexperte",
        "Ökonomen",
        "Oekonomen",
        "Ökonom",
        "Arbeitsmarkt",
        "Betriebsrat",
        "Betriebsräte",
        "Rentner",
        "Bürgern",
        "Bürger",
        "Buerger",
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
        # Phase 8.0 education — longer declined forms first.
        "Bildungsforscherinnen",
        "Bildungsforschern",
        "Bildungsforscher",
        "Bildungsexperten",
        "Lehrkräften",
        "Lehrkräfte",
        "Lehrkraft",
        "Lehrerinnen",
        "Erziehungsberechtigte",
        "Elternteil",
        "Eltern",
        "Schülern",
        "Schüler",
    ]
    lower = text.lower()
    # Stable appearance order; word-boundary so «Politik» ≠ prefix of «Politiker».
    found: list[tuple[int, str]] = []
    for role in lexicon:
        # Optional weak German endings so «Arbeitnehmern» / «Unternehmern»
        # keep appearance order vs. nominative «Wirtschaftsexperten» (7.3.2).
        pat = re.compile(
            rf"(?<![A-Za-zÄÖÜäöüß]){re.escape(role.lower())}"
            rf"(?:en|n|e|s|es)?(?![A-Za-zÄÖÜäöüß])",
            flags=re.IGNORECASE,
        )
        m = pat.search(lower)
        if m:
            found.append((m.start(), role))
    found.sort(key=lambda x: x[0])
    for _idx, role in found:
        # On income theses, bare «Bürger» as legal recipient is weak vs. named groups.
        if domain == "income" and _role_key(role) in {"buerger", "buergern", "burger"}:
            continue
        _add(role)
        if len(roles) >= limit:
            return roles[:limit]

    # 2) «zwischen A, B und C» / «Berücksichtige A, B und C» stakeholder lists.
    for match in re.finditer(
        r"(?:zwischen|berücksichtige|beruecksichtige|inkludiere|include)\s+(.+?)(?:\.|\?|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        chunk = match.group(1)
        # Stop at clause verbs that start instruction tails.
        chunk = re.split(
            r"\b(?:untersuche|simuliere|prüfe|analysiere|und\s+menschen)\b",
            chunk,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        # «über Lernleistungen, Datenschutz, …» are themes — not stakeholders (8.0).
        chunk = re.split(
            r"\b(?:über|ueber|about|regarding|zu\s+den\s+themen)\b",
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
        if domain == "income":
            fallback = _DEFAULT_STAKEHOLDERS_INCOME
        elif domain == "labor":
            fallback = _DEFAULT_STAKEHOLDERS_LABOR
        elif domain == "education":
            fallback = _DEFAULT_STAKEHOLDERS_EDUCATION
        else:
            fallback = _DEFAULT_STAKEHOLDERS
        for role in fallback:
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
            "unternehmer",
            "politiker",
            "politik",
            "verkehrsbetriebe",
            "anwohnende",
            "handel",
            "arbeitnehmer",
            "arbeitslose",
            "studierende",
            "rentner",
            "selbststaendige",
            "oekonomen",
            "arbeitsmarkt",
            "wirtschaftsexperten",
            "betriebsrat",
            "lehrkraefte",
            "eltern",
            "bildungsforscher",
            "schueler",
            "datenschutz",
            "chancengleichheit",
            "lernleistungen",
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
        if any(stem in norm for stem in INSTRUCTION_TOKEN_STEMS):
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


def _bind_measure(persona: dict[str, str], measure: str) -> dict[str, str]:
    """Fill ``{measure}`` placeholders — never leave prior-topic defaults."""
    out = dict(persona)
    for key in ("description", "goal", "conflict"):
        raw = out.get(key) or ""
        if "{measure}" in raw:
            out[key] = raw.replace("{measure}", measure)
    return out


def _role_family(key: str) -> str:
    """Collapse near-duplicate stakeholder keys (Unternehmer/Unternehmen, …)."""
    k = _role_key(key)
    if k in {"unternehmer", "unternehmern", "unternehmen"}:
        return "company"
    if k in {
        "oekonom",
        "oekonomen",
        "wirtschaftsexperten",
        "wirtschaftsexperte",
        "arbeitsmarkt",
    }:
        return "economist"
    if k in {"politiker", "politik"}:
        return "politics"
    if k in {"buerger", "buergern", "burger"}:
        return "citizen"
    if k in {"betriebsrat", "betriebsraete", "betriebsräte"}:
        return "works_council"
    if k in {"lehrkraefte", "lehrkraft", "lehrer", "lehrerinnen"}:
        return "teachers"
    if k in {"eltern", "elternteil", "erziehungsberechtigte"}:
        return "parents"
    if k in {
        "bildungsforscher",
        "bildungsforschern",
        "bildungsforscherinnen",
        "bildungsexperten",
    }:
        return "edu_research"
    if k in {"schueler", "schuelern"}:
        return "students"
    return k


def _pad_roles_for_domain(
    roles: list[str],
    *,
    domain: str,
    target: int,
) -> list[str]:
    """Fill missing seats with domain stakeholders before Moderator archetypes.

    Phase 8.1: after unique families are present, cycle domain stakeholders so
    10/25/50 societies keep stakeholder coverage (not only archetypes).
    """
    if target <= len(roles):
        return roles[:target]
    if domain == "income":
        pad = _DEFAULT_STAKEHOLDERS_INCOME
    elif domain == "labor":
        pad = _DEFAULT_STAKEHOLDERS_LABOR
    elif domain == "education":
        pad = _DEFAULT_STAKEHOLDERS_EDUCATION
    else:
        pad = _DEFAULT_STAKEHOLDERS
    out = list(roles)
    seen_families = {_role_family(r) for r in out}
    for role in pad:
        if len(out) >= target:
            break
        fam = _role_family(role)
        if fam in seen_families:
            continue
        out.append(role)
        seen_families.add(fam)
    # V3: cycle stakeholders for remaining scale seats (diversity at 10/25/50).
    if len(out) < target and pad:
        i = 0
        while len(out) < target:
            out.append(pad[i % len(pad)])
            i += 1
    return out


def _persona_table_for_domain(domain: str) -> dict[str, dict[str, str]]:
    if domain == "income":
        table = dict(_ROLE_PERSONAS_INCOME)
        # Alias ü/oe spellings.
        table["buerger"] = dict(table["bürger"])
        table["oekonomen"] = dict(table["oekonomen"])
        return table
    if domain == "labor":
        table = dict(_ROLE_PERSONAS_LABOR)
        table["buerger"] = dict(table["bürger"])
        table["oekonomen"] = dict(table["oekonomen"])
        # Stakeholder plural / Arbeitsmarkt cue → Wirtschaftsexperte (7.3.2).
        table["wirtschaftsexperten"] = dict(table["wirtschaftsexperten"])
        table["arbeitsmarkt"] = dict(table["oekonomen"])
        table["betriebsrat"] = dict(table["betriebsrat"])
        return table
    if domain == "mobility":
        table = dict(_ROLE_PERSONAS_MOBILITY)
        table["buerger"] = dict(table["bürger"])
        return table
    if domain == "education":
        table = dict(_ROLE_PERSONAS_EDUCATION)
        # Declined / alias keys → same persona.
        table["lehrkraft"] = dict(table["lehrkraefte"])
        table["lehrer"] = dict(table["lehrkraefte"])
        table["lehrerinnen"] = dict(table["lehrkraefte"])
        table["elternteil"] = dict(table["eltern"])
        table["erziehungsberechtigte"] = dict(table["eltern"])
        table["bildungsforschern"] = dict(table["bildungsforscher"])
        table["bildungsforscherinnen"] = dict(table["bildungsforscher"])
        table["bildungsexperten"] = dict(table["bildungsforscher"])
        table["schuelern"] = dict(table["schueler"])
        return table
    # Generic: measure-bound citizen/politics without mobility or BGE leaks.
    table = {
        "bürger": {
            "name": "Bürgerstimme",
            "slug": "buerger",
            "description": "Spricht für Alltag und Lebensqualität der Bevölkerung.",
            "goal": "Alltagsfolgen und Gerechtigkeit sichtbar machen.",
            "conflict": (
                "Fordere soziale Abfederung bei «{measure}»; widersprich Argumenten "
                "ohne Alltagsbezug."
            ),
            "style": "direkt, erfahrungsbezogen, pragmatisch",
        },
        "unternehmen": {
            "name": "Unternehmensvertretung",
            "slug": "unternehmen",
            "description": "Vertritt Betriebe bei Kosten und Planungssicherheit.",
            "goal": "Wirtschaftliche Risiken und praktikable Übergänge benennen.",
            "conflict": (
                "Fordere Übergangsregeln für «{measure}»; widersprich pauschalen "
                "Vorgaben ohne Branchenanalyse."
            ),
            "style": "kostenbewusst, lösungsorientiert, nüchtern",
        },
        "unternehmer": {
            "name": "Unternehmerstimme",
            "slug": "unternehmer",
            "description": "Vertritt Gründer und Betriebsrisiken.",
            "goal": "Finanzierbarkeit und Umsetzung prüfen.",
            "conflict": (
                "Lehne ungeprüfte Vollumsetzung von «{measure}» ab; fordere Piloten "
                "und Gegenfinanzierung."
            ),
            "style": "kostenbewusst, nüchtern",
        },
        "politiker": {
            "name": "Politikstimme",
            "slug": "politiker",
            "description": "Wägt Mehrheiten, Recht und Umsetzbarkeit ab.",
            "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
            "conflict": (
                "Suche Mehrheitsfähigkeit für «{measure}»; markiere Konflikte "
                "zwischen Betroffenen und Kostengrenzen."
            ),
            "style": "abwägend, öffentlichkeitsorientiert, strategisch",
        },
        "politik": {
            "name": "Politikstimme",
            "slug": "politik",
            "description": "Wägt Mehrheiten, Recht und Umsetzbarkeit ab.",
            "goal": "Politische Machbarkeit und Zielkonflikte klar machen.",
            "conflict": (
                "Suche Mehrheitsfähigkeit für «{measure}»; markiere Konflikte "
                "zwischen Betroffenen und Kostengrenzen."
            ),
            "style": "abwägend, öffentlichkeitsorientiert, strategisch",
        },
        "arbeitnehmer": {
            "name": "Arbeitnehmerstimme",
            "slug": "arbeitnehmer",
            "description": "Vertritt abhängig Beschäftigte und Alltagsfolgen.",
            "goal": "Beschäftigteninteressen bei der Maßnahme klar vertreten.",
            "conflict": (
                "Unterstütze «{measure}» wo Alltagsvorteile klar sind; fordere "
                "Absicherung gegen Mehrbelastung."
            ),
            "style": "klar, lohnbezogen, pragmatisch",
        },
        "oekonomen": {
            "name": "Ökonomenstimme",
            "slug": "oekonomen",
            "description": "Prüft Kosten-, Anreiz- und Verteilungseffekte.",
            "goal": "Empirische Risiken und Annahmen offenlegen.",
            "conflict": (
                "Stelle Risiken von «{measure}» heraus; unterstütze nur "
                "evidenzbasierte Piloten; widersprich utopischen Annahmen."
            ),
            "style": "analytisch, vorsichtig",
        },
    }
    table["buerger"] = dict(table["bürger"])
    table["oekonom"] = dict(table["oekonomen"])
    return table


def _occupation_for_unknown_role(role_key: str, role_label: str) -> str:
    """Human occupation — never raw ascii keys like ``lehrkraeften``."""
    label = (role_label or "").strip()
    if label and label.lower() not in _TOPIC_THEME_DENY:
        # Singular-ish display for declined plurals.
        if label.endswith("en") and len(label) > 5:
            return label[:-1] if not label.endswith("innen") else label
        return label[:40]
    key = _role_key(role_key)
    if key in _ROLE_CANON:
        return _ROLE_CANON[key]
    return "Diskussionsteilnehmer/in"


def _pick_individual(
    role_key: str,
    *,
    scenario: str,
    agent_id: int,
    role_label: str = "",
) -> dict[str, object]:
    """Deterministic individual (name/age/occupation) for a role key."""
    # Prefer canonical pool keys (Lehrkräften → lehrkraefte).
    pool_key = role_key
    if pool_key not in _INDIVIDUALS_BY_ROLE and pool_key in _ROLE_CANON:
        pool_key = _role_key(_ROLE_CANON[pool_key])
    pool = _INDIVIDUALS_BY_ROLE.get(pool_key) or _INDIVIDUALS_BY_ROLE.get(role_key) or []
    if not pool:
        # Fallback individual from German name tables — never abstract group label.
        first = _FIRST_NAMES[(agent_id + len(role_key)) % len(_FIRST_NAMES)]
        last = _LAST_NAMES[(agent_id * 3 + 7) % len(_LAST_NAMES)]
        if first.lower() == last.lower():
            last = _LAST_NAMES[(agent_id * 5 + 11) % len(_LAST_NAMES)]
        age = 24 + ((agent_id * 11 + len(scenario)) % 40)
        return {
            "first": first,
            "last": last,
            "age": age,
            "occupation": _occupation_for_unknown_role(role_key, role_label),
        }
    digest = hashlib.sha1(
        f"{scenario}|{agent_id}|{pool_key}|indiv".encode("utf-8")
    ).hexdigest()
    return dict(pool[int(digest[:8], 16) % len(pool)])


def _persona_for_role(
    role: str,
    arch_fallback: dict[str, str],
    *,
    domain: str = "generic",
    measure: str = "die vorgeschlagene Maßnahme",
) -> dict[str, str]:
    key = _role_key(role)
    if key in _ROLE_CANON:
        key = _role_key(_ROLE_CANON[key])
    table = _persona_table_for_domain(domain)
    if key in table:
        base = _bind_measure(dict(table[key]), measure)
        base["role_label"] = base.get("name") or role
        return base
    # Unknown role → keep voice, but never use the raw group noun / theme as name.
    if key in _TOPIC_THEME_DENY:
        label = arch_fallback.get("name") or "Teilnehmer/in"
    else:
        label = role[:1].upper() + role[1:]
    return {
        "name": label,
        "slug": _slugify(role) or arch_fallback["slug"],
        "description": (
            f"Spricht aus der Perspektive «{label}». {arch_fallback['description']}"
        ),
        "goal": f"Interessen von {label} zu «{measure}» klar vertreten. {arch_fallback['goal']}",
        "conflict": (
            f"Vertrete konsequent «{label}» zu «{measure}» gegen konkurrierende "
            f"Stakeholder-Interessen; keine ausgewogene Gesamtschau aller Seiten."
        ),
        "style": arch_fallback["style"],
        "role_label": label,
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
    """Build ``count`` unique scenario-bound individual personas (no clones).

    Phase 7.3: every agent is a person (Name, Alter, Beruf) — never an abstract
    group label like «Arbeitnehmern». Scales deterministically to 10/25/50.
    Live-LLM callers must pass ``max_agents=LIVE_LLM_MAX_AGENTS`` (5).
    """
    n = max(1, min(int(count), int(max_agents), MAX_GENERATABLE_AGENTS))
    domain = detect_topic_domain(scenario)
    # Stakeholder roles first; pad with domain defaults before abstract archetypes
    # so labor debates get Wirtschaftsexperte — not Moderator (Phase 7.3.2).
    roles = extract_stakeholder_roles(scenario, limit=min(n, 12))
    roles = _pad_roles_for_domain(roles, domain=domain, target=n)
    keywords = extract_scenario_keywords(scenario)
    topic = _topic_phrase(keywords)
    measure = measure_label(scenario, domain)
    short = (scenario or "").strip()[:120] or "Offenes Diskussionsthema"
    profiles: list[AgentProfile] = []
    used_names: set[str] = set()
    used_user_names: set[str] = set()

    for i in range(n):
        arch = _archetype_for_domain(_ARCHETYPES[i % len(_ARCHETYPES)], domain)
        voice_id = _unique_suffix(short, i, arch["slug"])
        if i < len(roles):
            persona = _persona_for_role(
                roles[i], arch, domain=domain, measure=measure
            )
            role_key = _role_key(roles[i])
            # Prefer canonical key used by individual pools.
            if role_key not in _INDIVIDUALS_BY_ROLE and role_key in _ROLE_CANON:
                role_key = _role_key(_ROLE_CANON[role_key])
            if role_key not in _INDIVIDUALS_BY_ROLE and persona.get("slug"):
                role_key = _role_key(str(persona["slug"]))
            # Canon aliases (Arbeitsmarkt→Ökonomen) must hit the economist pool.
            if role_key == "arbeitsmarkt":
                role_key = "oekonomen"
            # Persona display name (Lehrkraft) beats declined raw (Lehrkräften).
            role_label = str(persona.get("name") or persona.get("role_label") or roles[i])
            if role_key in _ROLE_CANON:
                # Keep singular persona titles; canon is for extraction only.
                pass
            indiv = _pick_individual(
                role_key,
                scenario=short,
                agent_id=i,
                role_label=role_label,
            )
            first = str(indiv["first"])
            last = str(indiv["last"])
            age = int(indiv["age"])  # type: ignore[arg-type]
            occupation = str(indiv["occupation"])
            display = f"{first} {last}"
            if display.lower() in used_names:
                display = f"{first} {last} {voice_id}"
            user_name = f"{_slugify(first)}_{_slugify(last)}_{voice_id}"
            description = (
                f"{age}, {occupation}. {persona['description']} "
                f"Perspektive «{role_label}» zu «{measure}» (Thema: {topic}). "
                f"Stimme #{i + 1}/{n} ({voice_id})."
            )
            conflict = persona.get("conflict") or _archetype_conflict(arch)
            # Goal stays instruction-only (never copied into public posts).
            goal = (
                f"{persona['goal']} Zielkonflikt: {conflict} Thema: {topic}. "
                f"[id={voice_id}]"
            )
            style = persona["style"]
        else:
            first = _FIRST_NAMES[(i + len(keywords)) % len(_FIRST_NAMES)]
            last = _LAST_NAMES[(i * 5 + 11) % len(_LAST_NAMES)]
            if first.lower() == last.lower():
                last = _LAST_NAMES[(i * 7 + 3) % len(_LAST_NAMES)]
            age = 24 + ((i * 13 + len(short)) % 40)
            occupation = arch["name"]
            display = f"{first} {last}"
            if display.lower() in used_names:
                display = f"{first} {last} {voice_id}"
            user_name = f"{_slugify(first)}_{_slugify(last)}_{arch['slug']}_{voice_id}"
            role_label = arch["name"]
            description = (
                f"{age}, {occupation}. {arch['description']} Perspektive zu «{measure}» "
                f"(Thema: {topic}). Stimme #{i + 1}/{n} ({voice_id})."
            )
            goal = (
                f"{arch['goal']} Zielkonflikt: {_archetype_conflict(arch)} "
                f"Thema: {topic}. [id={voice_id}]"
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
            age=age,
            occupation=occupation,
            role_label=role_label,
        )
        used_names.add(display.lower())
        used_user_names.add(user_name)
        profiles.append(profile)

    assert_no_clones(profiles)
    return profiles


def fixed_test_agents(count: int = 5) -> list[AgentProfile]:
    """Phase-2 regression roster — fixed five personalities (truncated if count < 5).

    Never expands beyond the classic five — use ``dynamic`` for 10/25/50.
    """
    n = max(1, min(int(count), len(DEFAULT_AGENTS), LIVE_LLM_MAX_AGENTS))
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
