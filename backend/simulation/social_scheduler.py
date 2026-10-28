"""Phase 6 + 8.0 — Scalable Social Interaction Scheduler (deterministic).

Chooses what each agent reads and whom they interact with. Inter-agent
partners are never the agent itself. Phase 8.0 adds wave/batch round
planning for offline 10/25/50 without unlocking paid Live-LLM.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


OFFLINE_SCALE_PRESETS = (10, 25, 50)
# Wave size for scalable offline rounds (Live still serializes via semaphore=1).
DEFAULT_WAVE_SIZE = 10


def _stable_rank(seed: str, item: int) -> int:
    digest = hashlib.sha1(f"{seed}|{item}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass
class InteractionPlan:
    """One agent's scheduled actions for a round."""

    agent_id: int
    round: int
    read_post_ids: list[int] = field(default_factory=list)
    read_comment_ids: list[int] = field(default_factory=list)
    partner_agent_id: int | None = None
    target_post_id: int | None = None
    target_comment_id: int | None = None
    action_hint: str = "create_post"
    self_reference_blocked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SocialInteractionScheduler:
    """Deterministic partner / target selection without self-as-partner."""

    def __init__(self, *, scenario: str = "", wave_size: int = DEFAULT_WAVE_SIZE) -> None:
        self.scenario = scenario or ""
        self.wave_size = max(1, int(wave_size))
        self._plans: list[dict[str, Any]] = []
        self._pair_counts: dict[tuple[int, int], int] = {}
        self._waves: list[dict[str, Any]] = []

    def clear(self) -> None:
        self._plans.clear()
        self._pair_counts.clear()
        self._waves.clear()

    def _seed(self, agent_id: int, round_n: int, kind: str) -> str:
        return f"{self.scenario}|{agent_id}|{round_n}|{kind}"

    def choose_partner(
        self,
        agent_id: int,
        candidates: list[int],
        *,
        round_n: int = 1,
    ) -> int | None:
        """Pick another agent as interaction partner — never self."""
        others = sorted({int(c) for c in candidates if int(c) != int(agent_id)})
        if not others:
            return None
        # Prefer less-used pairs, then stable hash for ties.
        ranked = sorted(
            others,
            key=lambda other: (
                self._pair_counts.get((agent_id, other), 0),
                _stable_rank(self._seed(agent_id, round_n, "partner"), other),
            ),
        )
        partner = ranked[0]
        self._pair_counts[(agent_id, partner)] = (
            self._pair_counts.get((agent_id, partner), 0) + 1
        )
        return partner

    def choose_post_target(
        self,
        agent_id: int,
        posts: list[dict[str, Any]],
        *,
        round_n: int = 1,
        prefer_partner: int | None = None,
    ) -> int | None:
        """Pick a post to read/reply to; prefer partner's post, never own-only trap."""
        usable: list[tuple[int, int]] = []  # (post_id, author)
        for p in posts:
            pid = p.get("post_id")
            uid = p.get("user_id")
            if pid is None or uid is None:
                continue
            usable.append((int(pid), int(uid)))
        if not usable:
            return None
        # Prefer partner-authored posts when available.
        if prefer_partner is not None:
            partner_posts = [pid for pid, uid in usable if uid == prefer_partner]
            if partner_posts:
                partner_posts.sort(
                    key=lambda pid: _stable_rank(
                        self._seed(agent_id, round_n, "post"), pid
                    )
                )
                return partner_posts[0]
        # Otherwise any post not authored solely as self-loop preference:
        # still allow own posts if they are the only ones (feed bootstrap).
        foreign = [pid for pid, uid in usable if uid != agent_id]
        pool = foreign or [pid for pid, _uid in usable]
        pool.sort(
            key=lambda pid: _stable_rank(self._seed(agent_id, round_n, "post"), pid)
        )
        return pool[0]

    def choose_comment_target(
        self,
        agent_id: int,
        comments: list[dict[str, Any]],
        *,
        round_n: int = 1,
        prefer_partner: int | None = None,
    ) -> int | None:
        """Pick a comment id to like/ref — never own; None if no foreign comments."""
        usable: list[tuple[int, int]] = []
        for c in comments:
            cid = c.get("comment_id")
            uid = c.get("user_id")
            if cid is None or uid is None:
                continue
            usable.append((int(cid), int(uid)))
        if not usable:
            return None
        # Phase 7.2: refuse self-targets entirely (blocks self-replies / self-likes).
        foreign = [(cid, uid) for cid, uid in usable if uid != agent_id]
        if not foreign:
            return None
        if prefer_partner is not None:
            partner_cs = [cid for cid, uid in foreign if uid == prefer_partner]
            if partner_cs:
                partner_cs.sort(
                    key=lambda cid: _stable_rank(
                        self._seed(agent_id, round_n, "comment"), cid
                    )
                )
                return partner_cs[0]
        pool = [cid for cid, _uid in foreign]
        pool.sort(
            key=lambda cid: _stable_rank(
                self._seed(agent_id, round_n, "comment"), cid
            )
        )
        return pool[0]

    def plan(
        self,
        *,
        agent_id: int,
        round_n: int,
        agent_ids: list[int],
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        call_index: int = 1,
    ) -> InteractionPlan:
        """Build a full interaction plan for one agent in one round."""
        partner = self.choose_partner(agent_id, agent_ids, round_n=round_n)
        target_post = self.choose_post_target(
            agent_id, posts, round_n=round_n, prefer_partner=partner
        )
        target_comment = self.choose_comment_target(
            agent_id, comments, round_n=round_n, prefer_partner=partner
        )
        read_posts = []
        if target_post is not None:
            read_posts.append(target_post)
        # Extra reads: up to 2 other foreign posts.
        extras = [
            int(p["post_id"])
            for p in posts
            if p.get("post_id") is not None
            and int(p["post_id"]) not in read_posts
            and p.get("user_id") is not None
            and int(p["user_id"]) != agent_id
        ]
        extras.sort(
            key=lambda pid: _stable_rank(self._seed(agent_id, round_n, "read"), pid)
        )
        read_posts.extend(extras[:2])

        read_comments = []
        if target_comment is not None:
            read_comments.append(target_comment)

        if target_post is None:
            hint = "create_post"
        elif target_comment is None:
            hint = "create_comment"
        elif round_n >= 2:
            kind = (call_index + agent_id) % 4
            hint = (
                "like_comment",
                "dislike_comment",
                "create_comment_ref",
                "like_post",
            )[kind]
        else:
            kind = (call_index + agent_id) % 5
            hint = (
                "create_comment",
                "like_post",
                "dislike_post",
                "like_comment",
                "dislike_comment",
            )[kind]

        plan = InteractionPlan(
            agent_id=agent_id,
            round=round_n,
            read_post_ids=read_posts,
            read_comment_ids=read_comments,
            partner_agent_id=partner,
            target_post_id=target_post,
            target_comment_id=target_comment,
            action_hint=hint,
            self_reference_blocked=True,
        )
        self._plans.append(plan.to_dict())
        return plan

    def plan_round(
        self,
        *,
        round_n: int,
        agent_ids: list[int],
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
    ) -> list[InteractionPlan]:
        """Batch-plan an entire round in deterministic waves (Phase 8.0)."""
        ids = sorted({int(a) for a in agent_ids})
        plans: list[InteractionPlan] = []
        wave_idx = 0
        for start in range(0, len(ids), self.wave_size):
            wave = ids[start : start + self.wave_size]
            wave_plans: list[dict[str, Any]] = []
            for offset, aid in enumerate(wave):
                p = self.plan(
                    agent_id=aid,
                    round_n=round_n,
                    agent_ids=ids,
                    posts=posts,
                    comments=comments,
                    call_index=offset + 1,
                )
                plans.append(p)
                wave_plans.append(p.to_dict())
            self._waves.append(
                {
                    "round": round_n,
                    "wave": wave_idx,
                    "agent_ids": wave,
                    "plans": len(wave_plans),
                }
            )
            wave_idx += 1
        return plans

    def summary(self) -> dict[str, Any]:
        self_hits = sum(
            1
            for p in self._plans
            if p.get("partner_agent_id") is not None
            and p.get("partner_agent_id") == p.get("agent_id")
        )
        agents_planned = {int(p["agent_id"]) for p in self._plans if "agent_id" in p}
        pair_loads = list(self._pair_counts.values()) if self._pair_counts else [0]
        return {
            "backend": "local_scheduler_v2",
            "scalable": True,
            "wave_size": self.wave_size,
            "waves": len(self._waves),
            "recent_waves": self._waves[-12:],
            "plans": len(self._plans),
            "unique_pairs": len(self._pair_counts),
            "pair_load_max": max(pair_loads),
            "pair_load_mean": round(sum(pair_loads) / max(1, len(pair_loads)), 3),
            "agent_coverage": len(agents_planned),
            "self_as_partner_count": self_hits,
            "self_reference_blocked": True,
            "offline_scale_presets": list(OFFLINE_SCALE_PRESETS),
            "recent": self._plans[-24:],
            "note": (
                "Scalable Social Interaction Scheduler (Phase 8.0): "
                "Wellenplanung, Partner-Balance, keine Selbstreferenz — "
                "Live-Ceilings unverändert."
            ),
        }
