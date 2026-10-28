"""Phase 7 — Opinion dynamics chronology (round snapshots + shift metrics).

Builds on PositionTracker change events. Does not invent polarity flips.
"""
from __future__ import annotations

from typing import Any


class OpinionDynamics:
    """Round-by-round polarity chronology and shift analytics."""

    def __init__(self, *, max_rounds_tracked: int = 16) -> None:
        self.max_rounds_tracked = max(1, int(max_rounds_tracked))
        self._round_snapshots: list[dict[str, Any]] = []
        self._seen_steps: set[int] = set()

    def clear(self) -> None:
        self._round_snapshots.clear()
        self._seen_steps.clear()

    def record_round(
        self,
        *,
        step: int | None,
        polarity_distribution: dict[str, int],
        change_events: list[dict[str, Any]],
        agent_count: int,
    ) -> None:
        """Append a chronology frame for this sync step (idempotent per step)."""
        if step is None:
            return
        step_i = int(step)
        if step_i in self._seen_steps:
            # Refresh the existing frame for the same step (latest sync wins).
            for frame in self._round_snapshots:
                if frame.get("step") == step_i:
                    frame["polarity_distribution"] = dict(polarity_distribution)
                    frame["agent_count"] = agent_count
                    frame["changes_this_step"] = sum(
                        1 for e in change_events if e.get("step") == step_i
                    )
                    frame["cumulative_changes"] = len(change_events)
                    return
            return

        self._seen_steps.add(step_i)
        changes_here = [e for e in change_events if e.get("step") == step_i]
        frame = {
            "step": step_i,
            "agent_count": agent_count,
            "polarity_distribution": dict(polarity_distribution),
            "changes_this_step": len(changes_here),
            "cumulative_changes": len(change_events),
            "change_samples": [
                {
                    "agent_id": e.get("agent_id"),
                    "from_polarity": e.get("from_polarity"),
                    "to_polarity": e.get("to_polarity"),
                    "evidence": e.get("evidence"),
                    "influenced_by_agent_ids": e.get("influenced_by_agent_ids") or [],
                }
                for e in changes_here[:12]
            ],
        }
        self._round_snapshots.append(frame)
        overflow = len(self._round_snapshots) - self.max_rounds_tracked
        if overflow > 0:
            dropped = self._round_snapshots[:overflow]
            del self._round_snapshots[:overflow]
            for d in dropped:
                self._seen_steps.discard(int(d.get("step") or -1))

    def snapshot(self) -> dict[str, Any]:
        frames = list(self._round_snapshots)
        total_changes = frames[-1]["cumulative_changes"] if frames else 0
        rounds = len(frames)
        shift_rate = (
            round(total_changes / max(1, frames[-1].get("agent_count", 1) * rounds), 4)
            if frames
            else 0.0
        )
        # Transition matrix counts (from→to) across chronology samples.
        transitions: dict[str, int] = {}
        for frame in frames:
            for sample in frame.get("change_samples") or []:
                key = f"{sample.get('from_polarity')}→{sample.get('to_polarity')}"
                transitions[key] = transitions.get(key, 0) + 1

        return {
            "backend": "local",
            "rounds_tracked": rounds,
            "total_evidenced_changes": total_changes,
            "shift_rate_per_agent_round": shift_rate,
            "transitions": dict(sorted(transitions.items())),
            "chronology": frames,
            "note": (
                "Meinungsdynamik-Chronologie aus belegten Positionswechseln. "
                "Keine erfundenen Flips; Scripted-Szenarien liefern Offline-Evidenz."
            ),
        }
