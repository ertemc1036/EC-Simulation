"""Local bounded per-agent memory — Zep-ready shape, no remote calls.

Phase 3 keeps a short working memory per agent (observations + own actions).
This is NOT Zep: entries stay in-process / result JSON. ``backend/zep`` remains
a stub until explicitly enabled later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryEntry:
    kind: str  # observation | action | statement | relation_hint
    text: str
    step: int | None = None
    ts: str = field(default_factory=_utc_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalAgentMemory:
    """Ring-buffer memory per agent_id with a hard entry cap."""

    def __init__(self, *, max_entries_per_agent: int = 12) -> None:
        self.max_entries_per_agent = max(1, int(max_entries_per_agent))
        self._by_agent: dict[int, list[MemoryEntry]] = {}

    def clear(self) -> None:
        self._by_agent.clear()

    def remember(
        self,
        agent_id: int,
        *,
        kind: str,
        text: str,
        step: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        bucket = self._by_agent.setdefault(int(agent_id), [])
        bucket.append(
            MemoryEntry(
                kind=kind,
                text=text[:800],
                step=step,
                meta=meta or {},
            )
        )
        overflow = len(bucket) - self.max_entries_per_agent
        if overflow > 0:
            del bucket[:overflow]

    def for_agent(self, agent_id: int) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._by_agent.get(int(agent_id), [])]

    def snapshot(self) -> dict[str, Any]:
        agents = {
            str(aid): [e.to_dict() for e in entries]
            for aid, entries in sorted(self._by_agent.items())
        }
        return {
            "backend": "local",
            "zep_integrated": False,
            "max_entries_per_agent": self.max_entries_per_agent,
            "agent_count_with_memory": len(agents),
            "agents": agents,
            "note": (
                "Lokales begrenztes Agentengedächtnis. "
                "Zep ist vorbereitet (Stub), aber nicht integriert."
            ),
        }

    def ingest_action(
        self,
        *,
        agent_id: int,
        action_type: str,
        action_args: Any,
        step: int | None,
        goal: str | None = None,
    ) -> None:
        args = action_args if isinstance(action_args, dict) else {"raw": action_args}
        content = args.get("content")
        target = args.get("post_id") or args.get("comment_id")
        bits = [f"Aktion {action_type}"]
        if target is not None:
            bits.append(f"Ziel={target}")
        if content:
            bits.append(f"Text: {str(content)[:240]}")
        if goal:
            bits.append(f"Zielrahmen: {goal}")
        self.remember(
            agent_id,
            kind="action",
            text=" · ".join(bits),
            step=step,
            meta={"action_type": action_type, "args": args},
        )

    def ingest_feed_snapshot(
        self,
        *,
        agent_ids: list[int],
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        step: int | None,
    ) -> None:
        """Each agent remembers a short view of the latest feed (not full DB)."""
        recent_posts = posts[-3:] if posts else []
        recent_comments = comments[-3:] if comments else []
        preview_parts: list[str] = []
        for p in recent_posts:
            preview_parts.append(
                f"Post#{p.get('post_id')} u{p.get('user_id')}: "
                f"{str(p.get('content') or '')[:120]}"
            )
        for c in recent_comments:
            preview_parts.append(
                f"Kommentar#{c.get('comment_id')}→{c.get('post_id')} "
                f"u{c.get('user_id')}: {str(c.get('content') or '')[:100]}"
            )
        if not preview_parts:
            return
        blob = " | ".join(preview_parts)[:700]
        for aid in agent_ids:
            self.remember(
                aid,
                kind="observation",
                text=f"Feed-Snapshot: {blob}",
                step=step,
                meta={"post_ids": [p.get("post_id") for p in recent_posts]},
            )
