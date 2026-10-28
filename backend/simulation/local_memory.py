"""Local bounded per-agent memory — working + long-term, sourced, no Zep.

Phase 3–6: short working memory (observations + own actions).
Phase 7: long-term tier consolidates sourced facts (post/comment/action refs)
with a hard per-agent cap. Entries stay in-process / result JSON.
``backend/zep`` remains a stub — not integrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryEntry:
    kind: str  # observation | action | statement | relation_hint | long_term
    text: str
    step: int | None = None
    ts: str = field(default_factory=_utc_now)
    meta: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalAgentMemory:
    """Ring-buffer working memory + bounded sourced long-term per agent."""

    def __init__(
        self,
        *,
        max_entries_per_agent: int = 12,
        max_long_term_per_agent: int = 8,
    ) -> None:
        self.max_entries_per_agent = max(1, int(max_entries_per_agent))
        self.max_long_term_per_agent = max(1, int(max_long_term_per_agent))
        self._by_agent: dict[int, list[MemoryEntry]] = {}
        self._long_term: dict[int, list[MemoryEntry]] = {}
        # Dedup keys for long-term promotions: (agent_id, source_key).
        self._lt_keys: dict[int, set[str]] = {}

    def clear(self) -> None:
        self._by_agent.clear()
        self._long_term.clear()
        self._lt_keys.clear()

    def remember(
        self,
        agent_id: int,
        *,
        kind: str,
        text: str,
        step: int | None = None,
        meta: dict[str, Any] | None = None,
        sources: list[str] | None = None,
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        bucket = self._by_agent.setdefault(int(agent_id), [])
        bucket.append(
            MemoryEntry(
                kind=kind,
                text=text[:400],
                step=step,
                meta=meta or {},
                sources=list(sources or []),
            )
        )
        overflow = len(bucket) - self.max_entries_per_agent
        if overflow > 0:
            del bucket[:overflow]

    def remember_long_term(
        self,
        agent_id: int,
        *,
        text: str,
        step: int | None = None,
        sources: list[str],
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """Promote a sourced fact into long-term memory (bounded, deduped)."""
        text = (text or "").strip()
        srcs = [s for s in (sources or []) if s]
        if not text or not srcs:
            return False
        aid = int(agent_id)
        dedup = "|".join(sorted(srcs)[:4])
        keys = self._lt_keys.setdefault(aid, set())
        if dedup in keys:
            return False
        keys.add(dedup)
        bucket = self._long_term.setdefault(aid, [])
        bucket.append(
            MemoryEntry(
                kind="long_term",
                text=text[:320],
                step=step,
                meta=meta or {},
                sources=srcs[:6],
            )
        )
        overflow = len(bucket) - self.max_long_term_per_agent
        if overflow > 0:
            # Drop oldest; free their dedup keys.
            dropped = bucket[:overflow]
            del bucket[:overflow]
            for d in dropped:
                drop_key = "|".join(sorted(d.sources)[:4])
                keys.discard(drop_key)
        return True

    def for_agent(self, agent_id: int) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._by_agent.get(int(agent_id), [])]

    def long_term_for_agent(self, agent_id: int) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._long_term.get(int(agent_id), [])]

    def recent_for_prompt(
        self, agent_id: int, *, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Bounded entries for decision-prompt injection (traceable memory).

        Prefer own actions over feed observations — FEED-KONTEXT already carries
        posts/comments with IDs (Phase 4.1 redundancy cut). Phase 7: append up to
        two long-term sourced facts after working-memory picks.
        """
        entries = self.for_agent(agent_id)
        if limit <= 0:
            return []
        indexed = list(enumerate(entries))
        indexed.sort(
            key=lambda pair: (
                0 if (pair[1].get("kind") or "") == "action" else 1,
                -pair[0],
            )
        )
        work_limit = max(1, limit - 1) if self.long_term_for_agent(agent_id) else limit
        chosen = [e for _, e in indexed[:work_limit]]
        chosen.sort(
            key=lambda e: (
                e.get("step") is None,
                e.get("step") if e.get("step") is not None else 0,
            )
        )
        # Append newest long-term facts (sourced) without exceeding limit.
        lt = self.long_term_for_agent(agent_id)
        for entry in reversed(lt):
            if len(chosen) >= limit:
                break
            chosen.append(entry)
        return chosen[:limit]

    def snapshot(self) -> dict[str, Any]:
        agents = {
            str(aid): [e.to_dict() for e in entries]
            for aid, entries in sorted(self._by_agent.items())
        }
        long_term = {
            str(aid): [e.to_dict() for e in entries]
            for aid, entries in sorted(self._long_term.items())
        }
        sourced = sum(
            1
            for entries in long_term.values()
            for e in entries
            if e.get("sources")
        )
        return {
            "backend": "local",
            "zep_integrated": False,
            "max_entries_per_agent": self.max_entries_per_agent,
            "max_long_term_per_agent": self.max_long_term_per_agent,
            "agent_count_with_memory": len(agents),
            "agent_count_with_long_term": len(long_term),
            "long_term_sourced_entries": sourced,
            "agents": agents,
            "long_term": long_term,
            "note": (
                "Lokales begrenztes Agentengedächtnis (Working + Long-Term). "
                "Long-Term nur mit Quellen (post:/comment:/action:). "
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
        sources: list[str] = [f"action:{action_type}"]
        if args.get("post_id") is not None:
            sources.append(f"post:{args.get('post_id')}")
        if args.get("comment_id") is not None:
            sources.append(f"comment:{args.get('comment_id')}")
        if target is not None:
            bits.append(f"Ziel={target}")
        if content:
            # Short snippet — full text already lives in the feed / DB.
            bits.append(f"Text: {str(content)[:100]}")
        # Do not re-append the full goal (already in system prompt).
        _ = goal
        self.remember(
            agent_id,
            kind="action",
            text=" · ".join(bits),
            step=step,
            meta={"action_type": action_type, "args": args},
            sources=sources,
        )
        # Promote socially meaningful actions into long-term (sourced).
        if action_type in {
            "create_post",
            "create_comment",
            "like_comment",
            "dislike_comment",
            "like_post",
            "dislike_post",
        }:
            self.remember_long_term(
                agent_id,
                text=" · ".join(bits)[:280],
                step=step,
                sources=sources,
                meta={"action_type": action_type, "tier": "long_term"},
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
        sources: list[str] = []
        for p in recent_posts:
            preview_parts.append(
                f"Post#{p.get('post_id')} u{p.get('user_id')}: "
                f"{str(p.get('content') or '')[:120]}"
            )
            if p.get("post_id") is not None:
                sources.append(f"post:{p.get('post_id')}")
        for c in recent_comments:
            preview_parts.append(
                f"Kommentar#{c.get('comment_id')}→{c.get('post_id')} "
                f"u{c.get('user_id')}: {str(c.get('content') or '')[:100]}"
            )
            if c.get("comment_id") is not None:
                sources.append(f"comment:{c.get('comment_id')}")
        if not preview_parts:
            return
        blob = " | ".join(preview_parts)[:280]
        for aid in agent_ids:
            self.remember(
                aid,
                kind="observation",
                text=f"Feed-Snapshot: {blob}",
                step=step,
                meta={"post_ids": [p.get("post_id") for p in recent_posts]},
                sources=list(sources),
            )
            # Long-term: remember one consolidated feed fact per step (sourced).
            if sources:
                self.remember_long_term(
                    aid,
                    text=f"Beobachtung R{step}: {blob[:200]}",
                    step=step,
                    sources=sources[:4],
                    meta={"tier": "long_term", "kind": "observation"},
                )

    def consolidate_from_positions(
        self,
        *,
        positions: list[dict[str, Any]],
        step: int | None,
    ) -> None:
        """Source long-term stance facts from evidenced position payload."""
        for pos in positions:
            aid = pos.get("agent_id")
            if aid is None:
                continue
            evid = str(pos.get("evidence") or "").strip()
            if not evid:
                continue
            sources = [evid] if ":" in evid else [f"stance:{evid}"]
            polarity = pos.get("polarity") or "unknown"
            stance = str(pos.get("stance") or "")[:160]
            self.remember_long_term(
                int(aid),
                text=f"Position ({polarity}): {stance}",
                step=step if step is not None else pos.get("step"),
                sources=sources,
                meta={
                    "tier": "long_term",
                    "kind": "position",
                    "polarity": polarity,
                    "changed": bool(pos.get("changed")),
                },
            )

    def remember_argument_development(
        self,
        agent_id: int,
        *,
        snippet: str,
        evidence: str,
        step: int | None = None,
        addressed_evidence: str | None = None,
        polarity: str | None = None,
    ) -> bool:
        """Phase 7.3: sourced argument-development fact (no invented memories)."""
        text = (snippet or "").strip()
        evid = (evidence or "").strip()
        if not text or not evid or ":" not in evid:
            return False
        sources = [evid]
        if addressed_evidence and ":" in addressed_evidence:
            sources.append(addressed_evidence)
        bits = [f"Argument: {text[:160]}"]
        if polarity:
            bits.append(f"Polarität={polarity}")
        if addressed_evidence:
            bits.append(f"greift {addressed_evidence} auf")
        return self.remember_long_term(
            int(agent_id),
            text=" · ".join(bits)[:280],
            step=step,
            sources=sources[:6],
            meta={
                "tier": "long_term",
                "kind": "argument_development",
                "polarity": polarity,
            },
        )
