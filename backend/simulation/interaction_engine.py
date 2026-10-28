"""Phase 8.1 — Multi-Agent Interaction Engine (offline-safe).

Coordinates scheduler wave planning with execution batches so scalable
10/25/50 societies actually run in waves — not only unit-test plan_round.
No provider calls. Live ceilings unchanged.
"""
from __future__ import annotations

import math
from typing import Any

from backend.simulation.social_scheduler import (
    DEFAULT_WAVE_SIZE,
    InteractionPlan,
    SCHEDULER_WAVE_PATHS,
    SocialInteractionScheduler,
    wave_count_for,
)


QUALITY_FIX = "8.3"


class MultiAgentInteractionEngine:
    """Wave-aware interaction coordinator bound to one simulation run."""

    def __init__(
        self,
        scheduler: SocialInteractionScheduler | None = None,
        *,
        scenario: str = "",
        wave_size: int = DEFAULT_WAVE_SIZE,
    ) -> None:
        self.scheduler = scheduler or SocialInteractionScheduler(
            scenario=scenario, wave_size=wave_size
        )
        self._rounds: list[dict[str, Any]] = []
        self._path: str | None = None

    def bind_scheduler(self, scheduler: SocialInteractionScheduler) -> None:
        self.scheduler = scheduler

    def clear(self) -> None:
        self._rounds.clear()
        self._path = None

    def prepare_round(
        self,
        *,
        round_n: int,
        agent_ids: list[int],
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        path: str,
    ) -> list[InteractionPlan]:
        """Plan the full round as waves before env.step execution."""
        path_key = (path or "unknown").strip().lower() or "unknown"
        self._path = path_key
        ids = sorted({int(a) for a in agent_ids})
        plans = self.scheduler.plan_round(
            round_n=round_n,
            agent_ids=ids,
            posts=posts,
            comments=comments,
        )
        self_hits = sum(
            1
            for p in plans
            if p.partner_agent_id is not None and p.partner_agent_id == p.agent_id
        )
        waves = wave_count_for(len(ids), self.scheduler.wave_size)
        partners = {
            p.partner_agent_id
            for p in plans
            if p.partner_agent_id is not None
        }
        self._rounds.append(
            {
                "round": int(round_n),
                "path": path_key,
                "agents": len(ids),
                "plans": len(plans),
                "waves": waves,
                "unique_partners": len(partners),
                "self_as_partner_count": self_hits,
                "consume_plan": bool(
                    (SCHEDULER_WAVE_PATHS.get(path_key) or {}).get("consume_plan")
                ),
            }
        )
        return plans

    def wave_agent_groups(
        self, agents: list[Any], *, id_attr: str = "social_agent_id"
    ) -> list[list[Any]]:
        """Split live agent objects into wave batches preserving id order."""
        indexed = sorted(
            agents,
            key=lambda a: int(getattr(a, id_attr, getattr(a, "agent_id", 0))),
        )
        w = self.scheduler.wave_size
        return [indexed[i : i + w] for i in range(0, len(indexed), w)]

    def mark_wave(
        self,
        *,
        round_n: int,
        wave: int,
        agent_ids: list[int],
        path: str | None = None,
    ) -> None:
        self.scheduler.mark_wave_executed(
            round_n=round_n,
            wave=wave,
            agent_ids=agent_ids,
            path=path or self._path or "unknown",
        )

    def summary(self) -> dict[str, Any]:
        sched = self.scheduler.summary()
        rounds = len(self._rounds)
        planned_waves = sum(int(r.get("waves") or 0) for r in self._rounds)
        return {
            "backend": "multi_agent_interaction_v3",
            "quality_fix": QUALITY_FIX,
            "version": 3,
            "path": self._path,
            "rounds_prepared": rounds,
            "plans_total": sum(int(r.get("plans") or 0) for r in self._rounds),
            "waves_planned": planned_waves,
            "waves_executed": sched.get("waves_executed", 0),
            "self_as_partner_count": sum(
                int(r.get("self_as_partner_count") or 0) for r in self._rounds
            ),
            "wave_size": self.scheduler.wave_size,
            "recent_rounds": self._rounds[-8:],
            "scheduler": {
                "backend": sched.get("backend"),
                "waves_planned": sched.get("waves_planned"),
                "waves_executed": sched.get("waves_executed"),
                "execution_paths_used": sched.get("execution_paths_used"),
            },
            "wave_path_matrix": SCHEDULER_WAVE_PATHS,
            "note": (
                "Multi-Agent Interaction Engine V3 (Phase 8.1): "
                "plan_round vor env.step; Wellen in Execution; "
                "Scripted konsumiert take_plan; Live/Manual advisory/batch."
            ),
        }


def interaction_status_payload() -> dict[str, Any]:
    """Public /api/health fragment."""
    return {
        "enabled": True,
        "backend": "multi_agent_interaction_v3",
        "quality_fix": QUALITY_FIX,
        "wave_size": DEFAULT_WAVE_SIZE,
        "offline_presets": [10, 25, 50],
        "wave_path_matrix": SCHEDULER_WAVE_PATHS,
        "sample_waves": {
            "n10": wave_count_for(10),
            "n25": wave_count_for(25),
            "n50": wave_count_for(50),
        },
        "paid_llm_unlocked": False,
        "note": (
            "Phase 8.1 Interaction Engine — Scheduler-Wellen in Execution; "
            "Live-Ceilings unverändert."
        ),
    }


def expected_waves(agent_count: int, rounds: int, wave_size: int = DEFAULT_WAVE_SIZE) -> int:
    return wave_count_for(agent_count, wave_size) * max(1, int(rounds))
