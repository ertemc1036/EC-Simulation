"""Safe prompt evidence for LLMAction runs — no API keys, no secret env.

Captures fingerprints + marker flags so we can prove which prompt path reached
the model client without dumping credentials or full paid payloads by default.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

# Stable id for the EC custom prompt bundle (system + decision templates).
PROMPT_BUNDLE_ID = "ec-custom-forum-de-v3"

# Markers that prove OASIS stock Reddit/Twitter framing reached the client.
OASIS_REDDIT_OBJECTIVE = "You're a Reddit user"
OASIS_TWITTER_OBJECTIVE = "You're a Twitter user"
OASIS_TWEET_BAIT = "I'll present you with some tweets"
OASIS_LIKE_BAIT = "don't limit your actions for example to just like"

# Markers that prove our custom prompts reached the client.
CUSTOM_ZIEL_MARKER = "# ZIEL / OBJECTIVE"
CUSTOM_DECISION_MARKER = "Entscheide jetzt"
CUSTOM_NO_TWEET_META = "Tweet-/Like-Spam"
CUSTOM_FEED_MARKER = "FEED-KONTEXT"
CUSTOM_MEMORY_MARKER = "MEMORY-KONTEXT"

_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer\s+[a-z0-9\-._~+/]+=*|sk-[a-z0-9]{10,})"
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _redact(text: str) -> str:
    """Strip anything that looks like a credential before preview storage."""
    return _SECRET_KEY_RE.sub("[REDACTED]", text)


def _preview(text: str, limit: int = 220) -> str:
    clean = _redact(text).replace("\n", " ").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


@dataclass
class PromptEvidence:
    """One model-bound message list — safe to persist in results/decision_log."""

    agent_id: int | None
    source: str
    prompt_bundle_id: str
    content_sha256: str
    roles: list[str]
    message_count: int
    system_preview: str
    user_preview: str
    markers: dict[str, bool]
    custom_prompts_active: bool
    oasis_default_active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_messages(
    messages: list[dict[str, Any]],
    *,
    agent_id: int | None = None,
    source: str = "model",
    prompt_bundle_id: str = PROMPT_BUNDLE_ID,
) -> PromptEvidence:
    """Fingerprint + marker scan over OpenAI-style chat messages."""
    roles: list[str] = []
    parts: list[str] = []
    system_parts: list[str] = []
    user_parts: list[str] = []

    for msg in messages or []:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        if content is None:
            # Tool-call-only assistant turns still count as a role.
            roles.append(role or "?")
            continue
        text = content if isinstance(content, str) else str(content)
        roles.append(role or "?")
        parts.append(text)
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            user_parts.append(text)

    blob = "\n".join(parts)
    markers = {
        "custom_ziel": CUSTOM_ZIEL_MARKER in blob,
        "custom_decision": CUSTOM_DECISION_MARKER in blob,
        "custom_feed_context": CUSTOM_FEED_MARKER in blob,
        "custom_memory_context": CUSTOM_MEMORY_MARKER in blob,
        "custom_anti_tweet_meta": CUSTOM_NO_TWEET_META in blob
        or "Like-Ketten" in blob
        or "Likes-zum-Liken" in blob
        or "Tweet-/Like-Spam-Meta-Posts" in blob,
        "deutsch_instruction": "auf Deutsch" in blob or "Sprache für alle Inhalte: Deutsch" in blob,
        "oasis_reddit_objective": OASIS_REDDIT_OBJECTIVE in blob,
        "oasis_twitter_objective": OASIS_TWITTER_OBJECTIVE in blob,
        "oasis_tweet_bait": OASIS_TWEET_BAIT in blob,
        "oasis_like_bait": OASIS_LIKE_BAIT in blob,
    }
    custom = bool(
        markers["custom_ziel"] or markers["custom_decision"]
    ) and not markers["oasis_reddit_objective"] and not markers["oasis_tweet_bait"]
    oasis_default = bool(
        markers["oasis_reddit_objective"]
        or markers["oasis_twitter_objective"]
        or markers["oasis_tweet_bait"]
        or markers["oasis_like_bait"]
    ) and not markers["custom_ziel"]

    return PromptEvidence(
        agent_id=agent_id,
        source=source,
        prompt_bundle_id=prompt_bundle_id,
        content_sha256=_sha256_text(blob),
        roles=roles,
        message_count=len(messages or []),
        system_preview=_preview("\n".join(system_parts)),
        user_preview=_preview("\n".join(user_parts[-1:])),
        markers=markers,
        custom_prompts_active=custom,
        oasis_default_active=oasis_default,
    )


class PromptDebugBuffer:
    """Thread-safe ring of recent prompt evidence (no keys)."""

    def __init__(self, *, max_entries: int = 64) -> None:
        self._lock = threading.Lock()
        self._entries: list[PromptEvidence] = []
        self._max = max(1, max_entries)

    def record(
        self,
        messages: list[dict[str, Any]],
        *,
        agent_id: int | None,
        source: str,
    ) -> PromptEvidence:
        evidence = analyze_messages(
            messages, agent_id=agent_id, source=source
        )
        with self._lock:
            self._entries.append(evidence)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max :]
        return evidence

    def last_for_agent(self, agent_id: int | None) -> PromptEvidence | None:
        with self._lock:
            for entry in reversed(self._entries):
                if entry.agent_id == agent_id:
                    return entry
        return None

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return {
                "prompt_bundle_id": PROMPT_BUNDLE_ID,
                "captures": 0,
                "custom_prompts_active": False,
                "oasis_default_active": False,
                "unique_content_hashes": 0,
            }
        hashes = {e.content_sha256 for e in entries}
        return {
            "prompt_bundle_id": PROMPT_BUNDLE_ID,
            "captures": len(entries),
            "custom_prompts_active": all(e.custom_prompts_active for e in entries),
            "oasis_default_active": any(e.oasis_default_active for e in entries),
            "unique_content_hashes": len(hashes),
            "marker_counts": {
                key: sum(1 for e in entries if e.markers.get(key))
                for key in entries[0].markers
            },
        }


def _parse_action_args(args: Any) -> Any:
    if args is None:
        return None
    if isinstance(args, str):
        raw = args.strip()
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return args


def normalize_action_args(args: Any) -> str:
    """Stable comparison key for action_log ↔ oasis_trace dedup."""
    parsed = _parse_action_args(args)
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    return str(parsed)


def action_dedup_key(agent_id: Any, action_type: Any, args: Any) -> tuple[Any, ...]:
    """Dedup key that ignores post_id added by OASIS after create_post.

    Seed ManualAction logs ``{\"content\": ...}`` while the DB trace later
    stores ``{\"content\": ..., \"post_id\": N}`` — both must collapse to one.
    """
    parsed = _parse_action_args(args)
    if isinstance(parsed, dict) and action_type in {
        "create_post",
        "create_comment",
    }:
        content = parsed.get("content")
        if content is not None:
            soft: dict[str, Any] = {"content": content}
            if "post_id" in parsed and action_type == "create_comment":
                soft["post_id"] = parsed.get("post_id")
            return (agent_id, action_type, json.dumps(soft, sort_keys=True, ensure_ascii=False))
    return (agent_id, action_type, normalize_action_args(args))
