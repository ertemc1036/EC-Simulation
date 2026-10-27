"""Scenario-aware agent prompts — replace OASIS Twitter/Reddit English defaults.

OASIS UserInfo.to_reddit_system_message() says \"I'll present you with some tweets\"
and ignores UserInfo.description unless profile['other_info']['user_profile'] is set.
SocialAgent.perform_action_by_llm() injects English \"don't limit … to just like\".
SocialEnvironment.env_template also nudges \"like to like posts\" and buries the
feed after empty group-chat noise.

This module builds German-default, scenario-bound system + decision prompts,
injects explicit feed + memory context, and steers agents toward comment/like/
dislike (debate tools) instead of search/refresh or duplicate essays.
"""
from __future__ import annotations

import re
from typing import Any

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

# OASIS env_template residue we strip before re-wrapping (wording differs from
# SocialAgent.perform_action_by_llm bait detected in prompt_debug).
_OASIS_ENV_LIKE_BAIT_RE = re.compile(
    r"(?is)pick one you want to perform action that best.*?like posts\s*"
)
_OASIS_GROUPS_NOISE_RE = re.compile(
    r"(?is)And there are many group chat channels.*?already in\s*"
)

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
Individuelles Ziel (inkl. Zielkonflikt): {goal}
Kommunikationsstil: {style}

# SPRACHE
{language_instruction}

# REGELN
1. Alle sichtbaren Beiträge (Posts/Kommentare) in der angegebenen Sprache verfassen.
2. Jede Aktion muss Szenario und dein individuelles Ziel voranbringen — nicht die Ziele fremder Stakeholder mitzusammenfassen.
3. Beziehe dich auf konkrete Posts im Feed (post_id). Wenn Posts sichtbar sind: bevorzugt create_comment, like_post oder dislike_post.
4. Kein neuer create_post, der nur das Szenario erneut paraphrasiert oder alle Stakeholder auflistet — das erzeugt Duplikate.
5. search_posts / refresh vermeiden, wenn der Feed bereits Posts mit IDs zeigt (Umgebung hat den Feed schon geladen).
6. Like/Dislike sind legitime Debatten-Signale auf konkrete Beiträge — kein Like-Spam und keine Meta-Posts über Likes.
7. Antworte nicht als generischer Englisch-Assistent — bleibe in Rolle und Sprache.

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


def sanitize_oasis_env_prompt(env_prompt: str) -> str:
    """Drop group-chat noise and OASIS like-bait tail; keep posts blob if present."""
    text = env_prompt or ""
    text = _OASIS_GROUPS_NOISE_RE.sub("", text)
    text = _OASIS_ENV_LIKE_BAIT_RE.sub("", text)
    return text.strip()


def format_feed_context(
    posts: list[dict[str, Any]],
    comments: list[dict[str, Any]] | None = None,
    *,
    max_posts: int = 8,
    max_comments: int = 6,
) -> str:
    """Compact feed block with explicit post_ids for tool args."""
    if not posts and not comments:
        return (
            "(Noch keine Posts im Feed — du darfst create_post als "
            "Diskussionsimpuls schreiben.)"
        )
    lines: list[str] = []
    recent = posts[-max_posts:] if posts else []
    for p in recent:
        pid = p.get("post_id")
        uid = p.get("user_id")
        content = str(p.get("content") or "").replace("\n", " ").strip()
        lines.append(f"- Post#{pid} von Nutzer {uid}: {content[:220]}")
    for c in (comments or [])[-max_comments:]:
        cid = c.get("comment_id")
        pid = c.get("post_id")
        uid = c.get("user_id")
        content = str(c.get("content") or "").replace("\n", " ").strip()
        lines.append(
            f"- Kommentar#{cid}→Post#{pid} von Nutzer {uid}: {content[:160]}"
        )
    return "\n".join(lines)


def format_memory_context(
    entries: list[dict[str, Any]],
    *,
    max_entries: int = 6,
) -> str:
    """Readable working-memory block for the decision prompt."""
    if not entries:
        return "(Noch kein lokales Arbeitsgedächtnis für dich.)"
    lines: list[str] = []
    for e in entries[-max_entries:]:
        kind = e.get("kind") or "note"
        step = e.get("step")
        text = str(e.get("text") or "").replace("\n", " ").strip()
        prefix = f"s{step}" if step is not None else "—"
        lines.append(f"- [{prefix}|{kind}] {text[:220]}")
    return "\n".join(lines)


def build_decision_prompt(
    *,
    scenario: str,
    goal: str,
    language: str,
    env_prompt: str,
    agent_name: str,
    feed_context: str | None = None,
    memory_context: str | None = None,
) -> str:
    """Per-decision user message — replaces OASIS English like-bait prompt."""
    lang = normalize_language(language)
    lang_name = LANGUAGE_NAMES[lang]
    clean_env = sanitize_oasis_env_prompt(env_prompt)
    feed = (feed_context or "").strip() or "(Feed-Kontext fehlt.)"
    memory = (memory_context or "").strip() or "(Kein Memory-Kontext.)"
    return (
        f"Entscheide jetzt als {agent_name} über deine nächste Foren-Aktion.\n"
        f"Szenario (verbindlich): {scenario.strip()}\n"
        f"Dein individuelles Ziel / Zielkonflikt: {goal}\n"
        f"Sprache für alle Inhalte: {lang_name} ({lang}).\n"
        f"Priorität wenn Posts mit IDs sichtbar sind: "
        f"create_comment, like_post oder dislike_post auf eine konkrete post_id. "
        f"Kein search_posts/refresh und kein doppelter Essay-Post.\n"
        f"FEED-KONTEXT (verbindliche post_ids):\n{feed}\n"
        f"MEMORY-KONTEXT (dein lokales Arbeitsgedächtnis):\n{memory}\n"
        f"Beobachte den Feed und handle szenariokonform. "
        f"Kein themenfremder Content, keine Tweet-/Like-Spam-Meta-Posts.\n"
        f"Plattform-Umgebung (bereinigt):\n{clean_env}"
    )


def model_calls_required_per_round(agent_count: int) -> int:
    """OASIS LLMAction: one model call per agent per round (perform_action_by_llm → astep)."""
    return max(0, int(agent_count))
