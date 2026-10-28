"""Phase 6 + 8.0/8.1 — Scalable Social Interaction Scheduler (deterministic).

Chooses what each agent reads and whom they interact with. Inter-agent
partners are never the agent itself. Phase 8.0 added wave/batch round
planning. Phase 8.1 wires waves into real execution via pending plans
(``take_plan``) and executed-wave markers.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any


OFFLINE_SCALE_PRESETS = (10, 25, 50)
# Wave size for scalable offline rounds (Live still serializes via semaphore=1).
DEFAULT_WAVE_SIZE = 10

# Which run paths consume scheduler waves in execution (Phase 8.1).
SCHEDULER_WAVE_PATHS = {
    "scripted": {
        "plan_round": True,
        "wave_env_step": True,
        "consume_plan": True,
        "note": "Offline ScriptedReactiveModel: plan_round → wave env.step → take_plan",
    },
    "manual": {
        "plan_round": True,
        "wave_env_step": True,
        "consume_plan": False,
        "note": "ManualAction: plan_round for coverage; wave-batched env.step when N>wave_size",
    },
    "live_llm": {
        "plan_round": True,
        "wave_env_step": True,
        "consume_plan": False,
        "note": (
            "Live-LLM: plan_round + wave env.step (usually 1 wave ≤5); "
            "OpenAI decides freely — plans are advisory/metrics only"
        ),
    },
}


def _stable_rank(seed: str, item: int) -> int:
    digest = hashlib.sha1(f"{seed}|{item}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def wave_count_for(agent_count: int, wave_size: int = DEFAULT_WAVE_SIZE) -> int:
    n = max(0, int(agent_count))
    w = max(1, int(wave_size))
    return int(math.ceil(n / w)) if n else 0


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
    wave: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SocialInteractionScheduler:
    """Deterministic partner / target selection without self-as-partner.

    Phase 8.1 (V3): ``plan_round`` stores pending plans; execution consumes
    them via ``take_plan`` and records ``mark_wave_executed``.
    """

    def __init__(self, *, scenario: str = "", wave_size: int = DEFAULT_WAVE_SIZE) -> None:
        self.scenario = scenario or ""
        self.wave_size = max(1, int(wave_size))
        self._plans: list[dict[str, Any]] = []
        self._pair_counts: dict[tuple[int, int], int] = {}
        self._waves: list[dict[str, Any]] = []
        self._pending: dict[tuple[int, int], InteractionPlan] = {}
        self._executed_waves: list[dict[str, Any]] = []
        self._paths_used: set[str] = set()

    def clear(self) -> None:
        self._plans.clear()
        self._pair_counts.clear()
        self._waves.clear()
        self._pending.clear()
        self._executed_waves.clear()
        self._paths_used.clear()

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
        wave: int | None = None,
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
            wave=wave,
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
        """Batch-plan an entire round in deterministic waves (Phase 8.0/8.1).

        Stores pending plans for ``take_plan`` during execution.
        """
        ids = sorted({int(a) for a in agent_ids})
        plans: list[InteractionPlan] = []
        wave_idx = 0
        for start in range(0, len(ids), self.wave_size):
            wave = ids[start : start + self.wave_size]
            wave_plans: list[dict[str, Any]] = []
            for offset, aid in enumerate(wave):
                # call_index ≈ round so R2+ keeps create_comment_ref / stance-shift
                # hints (historical per-agent call_count matched the round index).
                p = self.plan(
                    agent_id=aid,
                    round_n=round_n,
                    agent_ids=ids,
                    posts=posts,
                    comments=comments,
                    call_index=max(1, int(round_n)),
                    wave=wave_idx,
                )
                self._pending[(round_n, aid)] = p
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

    def take_plan(self, agent_id: int, round_n: int) -> InteractionPlan | None:
        """Consume a pending wave plan (Phase 8.1 execution path)."""
        return self._pending.pop((int(round_n), int(agent_id)), None)

    def peek_plan(self, agent_id: int, round_n: int) -> InteractionPlan | None:
        return self._pending.get((int(round_n), int(agent_id)))

    def mark_wave_executed(
        self,
        *,
        round_n: int,
        wave: int,
        agent_ids: list[int],
        path: str,
    ) -> None:
        """Record that a wave actually ran via env.step (not just planned)."""
        path_key = (path or "unknown").strip().lower() or "unknown"
        self._paths_used.add(path_key)
        self._executed_waves.append(
            {
                "round": int(round_n),
                "wave": int(wave),
                "agent_ids": [int(a) for a in agent_ids],
                "agents": len(agent_ids),
                "path": path_key,
            }
        )

    def wave_slices(self, agent_ids: list[int]) -> list[list[int]]:
        """Split agent ids into execution waves of ``wave_size``."""
        ids = sorted({int(a) for a in agent_ids})
        return [
            ids[start : start + self.wave_size]
            for start in range(0, len(ids), self.wave_size)
        ]

    def summary(self) -> dict[str, Any]:
        self_hits = sum(
            1
            for p in self._plans
            if p.get("partner_agent_id") is not None
            and p.get("partner_agent_id") == p.get("agent_id")
        )
        agents_planned = {int(p["agent_id"]) for p in self._plans if "agent_id" in p}
        pair_loads = list(self._pair_counts.values()) if self._pair_counts else [0]
        paths = sorted(self._paths_used)
        return {
            "backend": "local_scheduler_v3",
            "version": 3,
            "scalable": True,
            "wave_size": self.wave_size,
            "waves": len(self._waves),
            "waves_planned": len(self._waves),
            "waves_executed": len(self._executed_waves),
            "recent_waves": self._waves[-12:],
            "recent_executed_waves": self._executed_waves[-12:],
            "plans": len(self._plans),
            "pending_plans": len(self._pending),
            "unique_pairs": len(self._pair_counts),
            "pair_load_max": max(pair_loads),
            "pair_load_mean": round(sum(pair_loads) / max(1, len(pair_loads)), 3),
            "agent_coverage": len(agents_planned),
            "self_as_partner_count": self_hits,
            "self_reference_blocked": True,
            "offline_scale_presets": list(OFFLINE_SCALE_PRESETS),
            "execution_paths_used": paths,
            "wave_path_matrix": SCHEDULER_WAVE_PATHS,
            "recent": self._plans[-24:],
            "note": (
                "Scalable Social Interaction Scheduler V3 (Phase 8.1): "
                "Wellenplanung + Execution (take_plan / mark_wave_executed), "
                "Partner-Balance, keine Selbstreferenz — Live-Ceilings unverändert."
            ),
        }
