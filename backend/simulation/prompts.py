"""Scenario-aware agent prompts — replace OASIS Twitter/Reddit English defaults.

OASIS UserInfo.to_reddit_system_message() says \"I'll present you with some tweets\"
and ignores UserInfo.description unless profile['other_info']['user_profile'] is set.
SocialAgent.perform_action_by_llm() injects English \"don't limit … to just like\".
SocialEnvironment.env_template also nudges \"like to like posts\" and buries the
feed after empty group-chat noise.

Phase 4.1: slim system/decision text, prioritized feed/memory selection with hard
caps, drop redundant OASIS env when FEED-KONTEXT is present, keep post/comment IDs.
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

# Hard caps for decision-prompt context (Phase 4.1 token optimization).
MAX_FEED_POSTS = 4
MAX_FEED_COMMENTS = 4
MAX_MEMORY_ENTRIES = 3
POST_CONTENT_CHARS = 120
COMMENT_CONTENT_CHARS = 100
MEMORY_ENTRY_CHARS = 120

# OASIS env_template residue we strip before re-wrapping (wording differs from
# SocialAgent.perform_action_by_llm bait detected in prompt_debug).
_OASIS_ENV_LIKE_BAIT_RE = re.compile(
    r"(?is)pick one you want to perform action that best.*?like posts\s*"
)
_OASIS_GROUPS_NOISE_RE = re.compile(
    r"(?is)And there are many group chat channels.*?already in\s*"
)
_OASIS_MESSAGES_NOISE_RE = re.compile(
    r"(?is)You receive some messages from them\s*\{.*?\}\s*"
)

# Keys required by the TextPrompt template (must match profile dict).
# Slimmed vs Phase 4: OASIS-reply / anti-spam rules once, no repeated quotas essay.
SYSTEM_PROMPT_TEMPLATE = TextPrompt(
    """# ZIEL / OBJECTIVE
Foren-Diskussion (kein Twitter). Nur das Szenario; keine Tweet-/Like-Spam-Meta-Posts.

# SZENARIO
{scenario}

# ROLLE
Name: {name}
Persona: {persona}
Ziel (inkl. Konflikt): {goal}
Stil: {style}

# SPRACHE
{language_instruction}

# REGELN
1. Inhalte in der angegebenen Sprache.
2. Aktionen dienen deinem Ziel — nicht fremde Stakeholder mitzusammenfassen.
3. Interaktion: create_comment(post_id) als Reply (kein parent_comment_id); like_*/dislike_* auf Post/Kommentar-IDs; kein search/refresh wenn FEED IDs zeigt; kein doppelter Essay-Post.
4. Tool-Calling only; in Rolle bleiben.
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
    text = _OASIS_MESSAGES_NOISE_RE.sub("", text)
    text = _OASIS_ENV_LIKE_BAIT_RE.sub("", text)
    return text.strip()


def _rank_key_foreign_recent(
    item: dict[str, Any],
    *,
    viewer_id: int | None,
    id_key: str,
) -> tuple[int, int]:
    """Sort key: foreign first (0), then higher id (more recent) first."""
    uid = item.get("user_id")
    foreign = 0
    if viewer_id is not None and uid is not None:
        try:
            foreign = 0 if int(uid) != int(viewer_id) else 1
        except (TypeError, ValueError):
            foreign = 0
    try:
        iid = int(item.get(id_key) or 0)
    except (TypeError, ValueError):
        iid = 0
    return (foreign, -iid)


def select_prioritized_posts(
    posts: list[dict[str, Any]],
    *,
    viewer_id: int | None = None,
    max_posts: int = MAX_FEED_POSTS,
) -> list[dict[str, Any]]:
    """Limited, foreign-first, recent-first post selection — IDs preserved."""
    if not posts or max_posts <= 0:
        return []
    ranked = sorted(
        posts,
        key=lambda p: _rank_key_foreign_recent(p, viewer_id=viewer_id, id_key="post_id"),
    )
    return ranked[:max_posts]


def select_prioritized_comments(
    comments: list[dict[str, Any]],
    *,
    viewer_id: int | None = None,
    max_comments: int = MAX_FEED_COMMENTS,
) -> list[dict[str, Any]]:
    """Limited, foreign-first, recent-first comment selection — IDs preserved."""
    if not comments or max_comments <= 0:
        return []
    ranked = sorted(
        comments,
        key=lambda c: _rank_key_foreign_recent(
            c, viewer_id=viewer_id, id_key="comment_id"
        ),
    )
    return ranked[:max_comments]


def select_prioritized_memory(
    entries: list[dict[str, Any]],
    *,
    max_entries: int = MAX_MEMORY_ENTRIES,
) -> list[dict[str, Any]]:
    """Prefer own actions over feed observations (feed is already in FEED-KONTEXT)."""
    if not entries or max_entries <= 0:
        return []
    # Recent-first, then prefer action/statement over observation duplicates.
    indexed = list(enumerate(entries))
    indexed.sort(
        key=lambda pair: (
            0 if (pair[1].get("kind") or "") == "action" else 1,
            -pair[0],
        )
    )
    chosen = [e for _, e in indexed[:max_entries]]
    # Keep chronological order in the prompt for readability.
    chosen.sort(key=lambda e: (e.get("step") is None, e.get("step") or 0))
    return chosen


def format_feed_context(
    posts: list[dict[str, Any]],
    comments: list[dict[str, Any]] | None = None,
    *,
    max_posts: int = MAX_FEED_POSTS,
    max_comments: int = MAX_FEED_COMMENTS,
    viewer_id: int | None = None,
) -> str:
    """Compact prioritized feed with explicit post_ids / comment_ids for tools.

    OASIS reply = create_comment on post_id (no parent_comment_id). Comment
    lines expose comment_id for like_comment / dislike_comment.
    """
    if not posts and not comments:
        return (
            "(Noch keine Posts im Feed — du darfst create_post als "
            "Diskussionsimpuls schreiben.)"
        )
    lines: list[str] = [
        "IDs behalten. Reply=create_comment(post_id); kein parent_comment_id."
    ]
    recent = select_prioritized_posts(
        posts or [], viewer_id=viewer_id, max_posts=max_posts
    )
    for p in recent:
        pid = p.get("post_id")
        uid = p.get("user_id")
        content = str(p.get("content") or "").replace("\n", " ").strip()
        lines.append(
            f"- Post#{pid} u{uid} (reply→post_id={pid}): "
            f"{content[:POST_CONTENT_CHARS]}"
        )
    for c in select_prioritized_comments(
        comments or [], viewer_id=viewer_id, max_comments=max_comments
    ):
        cid = c.get("comment_id")
        pid = c.get("post_id")
        uid = c.get("user_id")
        content = str(c.get("content") or "").replace("\n", " ").strip()
        lines.append(
            f"- Kommentar#{cid}→Post#{pid} u{uid} "
            f"(comment_id={cid}; reply→post_id={pid}): "
            f"{content[:COMMENT_CONTENT_CHARS]}"
        )
    return "\n".join(lines)


def format_memory_context(
    entries: list[dict[str, Any]],
    *,
    max_entries: int = MAX_MEMORY_ENTRIES,
) -> str:
    """Readable working-memory block — prioritized, short, no feed re-dump."""
    selected = select_prioritized_memory(entries, max_entries=max_entries)
    if not selected:
        return "(Kein Memory.)"
    lines: list[str] = []
    for e in selected:
        kind = e.get("kind") or "note"
        step = e.get("step")
        text = str(e.get("text") or "").replace("\n", " ").strip()
        prefix = f"s{step}" if step is not None else "—"
        lines.append(f"- [{prefix}|{kind}] {text[:MEMORY_ENTRY_CHARS]}")
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
    include_env: bool | None = None,
) -> str:
    """Per-decision user message — slim; system prompt already has persona/scenario.

    Phase 4.1: scenario/goal only as short anchors (not full re-essay), drop OASIS
    env body when FEED-KONTEXT already carries IDs (env duplicates posts + noise).
    """
    lang = normalize_language(language)
    lang_name = LANGUAGE_NAMES[lang]
    feed = (feed_context or "").strip() or "(Feed-Kontext fehlt.)"
    memory = (memory_context or "").strip() or "(Kein Memory.)"
    feed_has_ids = "Post#" in feed or "Kommentar#" in feed or "post_id=" in feed
    if include_env is None:
        include_env = not feed_has_ids
    env_block = ""
    if include_env:
        clean_env = sanitize_oasis_env_prompt(env_prompt)
        if clean_env:
            # Hard cap residual env so it cannot balloon with platform noise.
            env_block = f"\nEnv (kurz):\n{clean_env[:400]}"
    # Short anchors — full persona/scenario live in the system prompt.
    scenario_anchor = (scenario or "").strip()
    if len(scenario_anchor) > 160:
        scenario_anchor = scenario_anchor[:157] + "…"
    goal_anchor = (goal or "").strip()
    if len(goal_anchor) > 140:
        goal_anchor = goal_anchor[:137] + "…"
    return (
        f"Entscheide jetzt als {agent_name}.\n"
        f"Szenario: {scenario_anchor}\n"
        f"Dein Ziel: {goal_anchor}\n"
        f"Sprache: {lang_name} ({lang}).\n"
        f"INTERAKTIONSKONTEXT: create_comment(post_id) / like_* / dislike_*; "
        f"kein search/refresh, kein doppelter Essay.\n"
        f"FEED-KONTEXT (verbindliche post_ids / comment_ids):\n{feed}\n"
        f"MEMORY-KONTEXT:\n{memory}"
        f"{env_block}"
    )


def estimate_decision_prompt_chars(
    *,
    scenario: str,
    goal: str,
    language: str,
    env_prompt: str,
    agent_name: str,
    posts: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    memory_entries: list[dict[str, Any]],
    viewer_id: int | None = None,
) -> int:
    """Offline sizing helper — no model calls."""
    feed = format_feed_context(posts, comments, viewer_id=viewer_id)
    memory = format_memory_context(memory_entries)
    text = build_decision_prompt(
        scenario=scenario,
        goal=goal,
        language=language,
        env_prompt=env_prompt,
        agent_name=agent_name,
        feed_context=feed,
        memory_context=memory,
    )
    return len(text)


def model_calls_required_per_round(agent_count: int) -> int:
    """OASIS LLMAction: one model call per agent per round (perform_action_by_llm → astep)."""
    return max(0, int(agent_count))
