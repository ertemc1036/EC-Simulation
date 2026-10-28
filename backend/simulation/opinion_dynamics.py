"""Phase 7 + 8.2 — Opinion dynamics chronology (V2).

Builds on PositionTracker change events. Does not invent polarity flips.
V2 adds stance clarity, stability, polarization, and unknown-share series.
"""
from __future__ import annotations

import math
from typing import Any

QUALITY_FIX = "8.2"
OPINION_ENGINE = "v2"


def _unknown_share(polarity_distribution: dict[str, int]) -> float:
    total = sum(int(v) for v in (polarity_distribution or {}).values()) or 0
    if total <= 0:
        return 1.0
    unk = int((polarity_distribution or {}).get("unknown") or 0) + int(
        (polarity_distribution or {}).get("uncertain") or 0
    )
    return round(unk / float(total), 4)


def _clarity(polarity_distribution: dict[str, int]) -> float:
    return round(max(0.0, 1.0 - _unknown_share(polarity_distribution)), 4)


def _polarization(polarity_distribution: dict[str, int]) -> float:
    """Simple pro↔contra tension in [0,1]; conditional counts as half-weight."""
    pol = polarity_distribution or {}
    total = max(1, sum(int(v) for v in pol.values()))
    pro = int(pol.get("pro") or 0)
    contra = int(pol.get("contra") or 0)
    cond = int(pol.get("conditional") or 0) + int(pol.get("exception") or 0)
    # Tension rises with both poles; pure mono-polarity → low.
    poles = (pro + 0.5 * cond) / total
    antipoles = (contra + 0.5 * cond) / total
    return round(min(1.0, 2.0 * min(poles, antipoles) + 0.15 * (cond / total)), 4)


def _entropy(polarity_distribution: dict[str, int]) -> float:
    total = sum(int(v) for v in (polarity_distribution or {}).values()) or 0
    if total <= 0:
        return 0.0
    h = 0.0
    for v in polarity_distribution.values():
        p = int(v) / total
        if p > 0:
            h -= p * math.log(p + 1e-12, 2)
    # Normalize by log2(5) ≈ clear label set size.
    return round(min(1.0, h / math.log(5, 2)), 4)


class OpinionDynamics:
    """Round-by-round polarity chronology and shift analytics (V2)."""

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
        dist = dict(polarity_distribution)
        clarity = _clarity(dist)
        unk = _unknown_share(dist)
        frame_fields = {
            "polarity_distribution": dist,
            "agent_count": agent_count,
            "stance_clarity": clarity,
            "unknown_share": unk,
            "polarization": _polarization(dist),
            "entropy": _entropy(dist),
        }
        if step_i in self._seen_steps:
            # Refresh the existing frame for the same step (latest sync wins).
            for frame in self._round_snapshots:
                if frame.get("step") == step_i:
                    frame.update(frame_fields)
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
            **frame_fields,
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

        clarities = [float(f.get("stance_clarity") or 0.0) for f in frames]
        unknown_series = [float(f.get("unknown_share") or 0.0) for f in frames]
        # Stability: 1 - mean absolute clarity delta across rounds.
        if len(clarities) >= 2:
            deltas = [abs(clarities[i] - clarities[i - 1]) for i in range(1, len(clarities))]
            stability = round(max(0.0, 1.0 - (sum(deltas) / len(deltas))), 4)
        else:
            stability = 1.0 if clarities else 0.0
        final_dist = (frames[-1].get("polarity_distribution") or {}) if frames else {}
        final_clarity = clarities[-1] if clarities else 0.0
        final_unk = unknown_series[-1] if unknown_series else 1.0

        return {
            "backend": "local",
            "engine": OPINION_ENGINE,
            "version": 2,
            "quality_fix": QUALITY_FIX,
            "rounds_tracked": rounds,
            "total_evidenced_changes": total_changes,
            "shift_rate_per_agent_round": shift_rate,
            "transitions": dict(sorted(transitions.items())),
            "stance_clarity": final_clarity,
            "unknown_share": final_unk,
            "stability_index": stability,
            "polarization": _polarization(final_dist),
            "entropy": _entropy(final_dist),
            "unknown_share_by_round": [
                {"step": f.get("step"), "unknown_share": f.get("unknown_share")}
                for f in frames
            ],
            "chronology": frames,
            "note": (
                "Meinungsdynamik V2: Chronologie + Stance-Klarheit, Stabilität, "
                "Polarisierung. Keine erfundenen Flips; Scripted liefert Offline-Evidenz."
            ),
        }
