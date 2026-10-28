"""Phase 5 — Social Dynamics: persistent positions + local comment refs.

OASIS has no ``parent_comment_id``. Local references (``→Kommentar#N``,
``Kommentar#N``, ``comment_id=N`` in comment content) plus like/dislike
evidence drive social relations. No invented edges.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Phase-5 social relations — only emit when evidenced.
SOCIAL_RELATIONS = frozenset(
    {
        "agrees_with",
        "disagrees_with",
        "responds_to",
        "influenced_by",
        "position_changed",
    }
)

_COMMENT_REF_RE = re.compile(
    r"(?:→\s*)?(?:Kommentar#|comment_id\s*=\s*)(\d+)",
    re.IGNORECASE,
)

# Lightweight polarity lexicon for offline stance keys (DE + a few EN stems).
_PRO_MARKERS = (
    "dafür",
    "unterstütz",
    "befürwort",
    "sinnvoll",
    "notwendig",
    "ja zum",
    "pro ",
    "support",
    "agree",
    "förder",
)
_CONTRA_MARKERS = (
    "dagegen",
    "ablehn",
    "widersprich",
    "nein zum",
    "gegen ",
    "unsinn",
    "schädlich",
    "oppose",
    "disagree",
    "protest",
)
_SHIFT_MARKERS = (
    "umdenken",
    "revidier",
    "änder meine",
    "ändere meine",
    "neue position",
    "umstimme",
    "überzeugt",
    "einlenk",
    "kompromiss",
)


def extract_local_comment_refs(content: str) -> list[int]:
    """Parse local comment references from free text (OASIS has no parent_id)."""
    ids: list[int] = []
    seen: set[int] = set()
    for match in _COMMENT_REF_RE.findall(content or ""):
        cid = int(match)
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _polarity(text: str) -> str:
    low = (text or "").lower()
    pro = sum(1 for m in _PRO_MARKERS if m in low)
    contra = sum(1 for m in _CONTRA_MARKERS if m in low)
    if pro > contra:
        return "pro"
    if contra > pro:
        return "contra"
    if any(m in low for m in _SHIFT_MARKERS):
        return "shifting"
    return "neutral"


def _stance_key(polarity: str, snippet: str) -> str:
    digest = hashlib.sha1((snippet or "").encode("utf-8")).hexdigest()[:8]
    return f"{polarity}:{digest}"


def _short(text: str, limit: int = 140) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


@dataclass
class AgentPosition:
    agent_id: int
    stance: str
    stance_key: str
    polarity: str
    step: int | None = None
    evidence: str = ""
    evidence_kind: str = ""  # post | comment | goal | reaction
    changed: bool = False
    previous_stance: str | None = None
    previous_stance_key: str | None = None
    influenced_by_agent_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PositionTracker:
    """Persistent per-agent positions across simulation rounds."""

    def __init__(self) -> None:
        self._current: dict[int, AgentPosition] = {}
        self._history: dict[int, list[AgentPosition]] = {}
        self._change_events: list[dict[str, Any]] = []

    def clear(self) -> None:
        self._current.clear()
        self._history.clear()
        self._change_events.clear()

    def seed_from_goals(
        self, agents: list[Any], *, step: int | None = 0
    ) -> None:
        """Initial positions from goals — before any platform content."""
        for raw in agents:
            if hasattr(raw, "agent_id"):
                aid = int(raw.agent_id)
                goal = str(getattr(raw, "goal", "") or "")
            else:
                aid = int(raw.get("agent_id", 0))
                goal = str(raw.get("goal") or "")
            if aid in self._current:
                continue
            pol = _polarity(goal)
            stance = _short(goal) or f"Agent {aid}"
            pos = AgentPosition(
                agent_id=aid,
                stance=stance,
                stance_key=_stance_key(pol, stance),
                polarity=pol,
                step=step,
                evidence=stance,
                evidence_kind="goal",
            )
            self._current[aid] = pos
            self._history.setdefault(aid, []).append(pos)

    def update_from_snapshot(
        self,
        *,
        agents: list[Any],
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        reactions: list[dict[str, Any]],
        step: int | None,
    ) -> list[dict[str, Any]]:
        """Recompute positions from latest content; record evidenced changes."""
        self.seed_from_goals(agents, step=step if step is not None else 0)
        # Latest statement per agent (comment preferred over older post).
        latest: dict[int, tuple[str, str, str]] = {}
        for p in posts:
            uid = p.get("user_id")
            if uid is None:
                continue
            content = str(p.get("content") or "").strip()
            if not content:
                continue
            latest[int(uid)] = (content, "post", f"post:{p.get('post_id')}")
        for c in comments:
            uid = c.get("user_id")
            if uid is None:
                continue
            content = str(c.get("content") or "").strip()
            if not content:
                continue
            latest[int(uid)] = (
                content,
                "comment",
                f"comment:{c.get('comment_id')}",
            )

        # Who did this agent interact with recently (for influenced_by)?
        interacted: dict[int, set[int]] = {aid: set() for aid in self._current}
        author_of_post = {
            int(p["post_id"]): int(p["user_id"])
            for p in posts
            if p.get("post_id") is not None and p.get("user_id") is not None
        }
        author_of_comment = {
            int(c["comment_id"]): int(c["user_id"])
            for c in comments
            if c.get("comment_id") is not None and c.get("user_id") is not None
        }
        for r in reactions:
            uid = r.get("user_id")
            if uid is None:
                continue
            aid = int(uid)
            target_author: int | None = None
            if r.get("post_id") is not None:
                target_author = author_of_post.get(int(r["post_id"]))
            elif r.get("comment_id") is not None:
                target_author = author_of_comment.get(int(r["comment_id"]))
            if target_author is not None and target_author != aid:
                interacted.setdefault(aid, set()).add(target_author)
        for c in comments:
            uid = c.get("user_id")
            if uid is None:
                continue
            aid = int(uid)
            for ref in extract_local_comment_refs(str(c.get("content") or "")):
                target_author = author_of_comment.get(ref)
                if target_author is not None and target_author != aid:
                    interacted.setdefault(aid, set()).add(target_author)
            # Flat OASIS reply under post still evidences response to post author.
            pid = c.get("post_id")
            if pid is not None:
                target_author = author_of_post.get(int(pid))
                if target_author is not None and target_author != aid:
                    interacted.setdefault(aid, set()).add(target_author)

        changes: list[dict[str, Any]] = []
        for aid, (content, kind, evid) in latest.items():
            pol = _polarity(content)
            # Explicit shift markers force a polarity flip marker for change detect.
            if any(m in content.lower() for m in _SHIFT_MARKERS) and pol == "neutral":
                pol = "shifting"
            stance = _short(content)
            key = _stance_key(pol, stance)
            prev = self._current.get(aid)
            changed = bool(prev and prev.stance_key != key)
            influencers: list[int] = []
            if changed:
                influencers = sorted(interacted.get(aid) or [])
            pos = AgentPosition(
                agent_id=aid,
                stance=stance,
                stance_key=key,
                polarity=pol,
                step=step,
                evidence=evid,
                evidence_kind=kind,
                changed=changed,
                previous_stance=prev.stance if changed and prev else None,
                previous_stance_key=prev.stance_key if changed and prev else None,
                influenced_by_agent_ids=influencers,
            )
            if changed and prev:
                event = {
                    "agent_id": aid,
                    "from_stance": prev.stance,
                    "to_stance": stance,
                    "from_key": prev.stance_key,
                    "to_key": key,
                    "step": step,
                    "evidence": evid,
                    "influenced_by_agent_ids": influencers,
                }
                changes.append(event)
                self._change_events.append(event)
            self._current[aid] = pos
            hist = self._history.setdefault(aid, [])
            if not hist or hist[-1].stance_key != key:
                hist.append(pos)
        return changes

    def positions_payload(self) -> dict[str, Any]:
        items = [p.to_dict() for p in sorted(self._current.values(), key=lambda x: x.agent_id)]
        return {
            "backend": "local",
            "agent_count": len(items),
            "change_count": len(self._change_events),
            "positions": items,
            "changes": list(self._change_events),
            "note": (
                "Persistente Agentenpositionen aus Goal-Seed + Posts/Kommentaren. "
                "position_changed / influenced_by nur bei belegtem Stance-Wechsel."
            ),
        }

    def social_edge_specs(self) -> list[dict[str, Any]]:
        """Evidence-gated edge specs for the knowledge graph (no invention)."""
        specs: list[dict[str, Any]] = []
        for ev in self._change_events:
            aid = int(ev["agent_id"])
            specs.append(
                {
                    "relation": "position_changed",
                    "source": f"actor:{aid}",
                    "target": f"topic:scenario",
                    "meta": {
                        "from_stance": ev.get("from_stance"),
                        "to_stance": ev.get("to_stance"),
                        "step": ev.get("step"),
                        "evidence": ev.get("evidence"),
                    },
                }
            )
            for other in ev.get("influenced_by_agent_ids") or []:
                specs.append(
                    {
                        "relation": "influenced_by",
                        "source": f"actor:{aid}",
                        "target": f"actor:{int(other)}",
                        "meta": {
                            "step": ev.get("step"),
                            "evidence": "position_change_after_interaction",
                        },
                    }
                )
        return specs


def collect_local_responds_to(
    comments: list[dict[str, Any]],
    *,
    comment_statement_ids: dict[int, str],
    post_statement_ids: dict[int, str],
    agent_nodes: dict[int, str],
) -> list[dict[str, Any]]:
    """Build responds_to edges from local comment refs + flat post replies."""
    specs: list[dict[str, Any]] = []
    for c in comments:
        cid = c.get("comment_id")
        uid = c.get("user_id")
        if cid is None or uid is None:
            continue
        aid = int(uid)
        if aid not in agent_nodes:
            continue
        content = str(c.get("content") or "")
        refs = extract_local_comment_refs(content)
        for ref in refs:
            if ref not in comment_statement_ids:
                continue
            specs.append(
                {
                    "relation": "responds_to",
                    "source": agent_nodes[aid],
                    "target": comment_statement_ids[ref],
                    "meta": {
                        "comment_id": cid,
                        "target_comment_id": ref,
                        "evidence": "local_comment_ref",
                        "note": "OASIS has no parent_comment_id",
                    },
                }
            )
        # Flat create_comment under a post = responds_to that post statement.
        pid = c.get("post_id")
        if pid is not None and int(pid) in post_statement_ids:
            specs.append(
                {
                    "relation": "responds_to",
                    "source": agent_nodes[aid],
                    "target": post_statement_ids[int(pid)],
                    "meta": {
                        "comment_id": cid,
                        "post_id": pid,
                        "evidence": "create_comment",
                    },
                }
            )
    return specs
