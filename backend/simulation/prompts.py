"""Scenario-aware agent prompts — replace OASIS Twitter/Reddit English defaults.

OASIS UserInfo.to_reddit_system_message() says \"I'll present you with some tweets\"
and ignores UserInfo.description unless profile['other_info']['user_profile'] is set.
SocialAgent.perform_action_by_llm() injects English \"don't limit … to just like\".
SocialEnvironment.env_template also nudges \"like to like posts\".

This module builds German-default, scenario-bound system + decision prompts and
avoids Twitter/tweet/like-bait wording so agents stay on-topic.
"""
from __future__ import annotations

from camel.prompts import TextPrompt

from backend.simulation.agents import AgentProfile

# Language codes we support; unknown codes fall back to German instructions.
LANGUAGE_NAMES: dict[str, str] = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
}

# Keys required by the TextPrompt template (must match profile dict).
SYSTEM_PROMPT_TEMPLATE = TextPrompt(
    """# ZIEL / OBJECTIVE
Du bist Teilnehmer:in einer Foren-Diskussion (Reddit-ähnliche Plattform, keine Twitter-/X-Timeline).
Diskutiere ausschließlich das Szenario. Erfinde keine themenfremden Tweets, Like-Ketten
oder Engagement-Spam.

# SZENARIO
{scenario}

# ROLLE
Dein Name ist {name}.
Persona: {persona}
Individuelles Ziel: {goal}
Kommunikationsstil: {style}

# SPRACHE
{language_instruction}

# REGELN
1. Alle sichtbaren Beiträge (Posts/Kommentare) in der angegebenen Sprache verfassen.
2. Jede Aktion muss Szenario und dein individuelles Ziel voranbringen.
3. Beziehe dich auf vorhandene Posts im Feed; kein themenfremder Off-Topic-Content.
4. Keine Meta-Posts über Tweets, Likes-zum-Liken oder Plattform-Engagement.
5. Wähle passende Tool-Aktionen (Kommentar, Post, Like) nur wenn sie zur Debatte passen.
6. Antworte nicht als generischer Englisch-Assistent — bleibe in Rolle und Sprache.

# ANTWORT
Aktionen ausschließlich per Tool-Calling ausführen.
"""
)


def normalize_language(code: str | None) -> str:
    raw = (code or "de").strip().lower()
    if not raw:
        return "de"
    # Accept "de-DE" → "de"
    short = raw.split("-", 1)[0]
    return short if short in LANGUAGE_NAMES else "de"


def language_display_name(code: str | None) -> str:
    lang = normalize_language(code)
    return LANGUAGE_NAMES[lang]


def language_instruction(code: str | None) -> str:
    lang = normalize_language(code)
    name = LANGUAGE_NAMES[lang]
    if lang == "de":
        return (
            "Verfassen Sie ALLE Posts und Kommentare auf Deutsch. "
            "Kein Englisch, außer wenn Sie kurz einen Fachbegriff zitieren."
        )
    if lang == "en":
        return "Write ALL posts and comments in English."
    return f"Write ALL posts and comments in {name} (language code: {lang})."


def build_profile_for_template(
    profile: AgentProfile,
    *,
    scenario: str,
    language: str,
) -> dict[str, str]:
    """Profile keys must match SYSTEM_PROMPT_TEMPLATE placeholders."""
    return {
        "name": profile.name,
        "persona": profile.description,
        "goal": profile.goal,
        "style": profile.style,
        "scenario": scenario.strip() or "(kein Szenario)",
        "language_instruction": language_instruction(language),
    }


def build_system_prompt(
    profile: AgentProfile,
    *,
    scenario: str,
    language: str = "de",
) -> str:
    data = build_profile_for_template(profile, scenario=scenario, language=language)
    return SYSTEM_PROMPT_TEMPLATE.format(**data)


def build_decision_prompt(
    *,
    scenario: str,
    goal: str,
    language: str,
    env_prompt: str,
    agent_name: str,
) -> str:
    """Per-decision user message — replaces OASIS English like-bait prompt."""
    lang = normalize_language(language)
    lang_name = LANGUAGE_NAMES[lang]
    return (
        f"Entscheide jetzt als {agent_name} über deine nächste Foren-Aktion.\n"
        f"Szenario (verbindlich): {scenario.strip()}\n"
        f"Dein individuelles Ziel: {goal}\n"
        f"Sprache für alle Inhalte: {lang_name} ({lang}).\n"
        f"Beobachte den Feed und handle szenariokonform. "
        f"Kein themenfremder Content, keine Tweet-/Like-Spam-Meta-Posts.\n"
        f"Plattform-Umgebung:\n{env_prompt}"
    )


def model_calls_required_per_round(agent_count: int) -> int:
    """OASIS LLMAction: one model call per agent per round (perform_action_by_llm → astep)."""
    return max(0, int(agent_count))
