"""Phase 7 + 8.2–8.4 — Opinion dynamics chronology (V4).

Builds on PositionTracker change events. Does not invent polarity flips.
V2 adds stance clarity, stability, polarization, and unknown-share series.
V3 adds grounded-reply share and motivation-linked stability signals.
V4 treats ``shifting`` as unclear for clarity math and tracks evidenced
conditional targets (scripted mind-change without bare shifting polarity).
"""
from __future__ import annotations

import math
from typing import Any

QUALITY_FIX = "8.4"
OPINION_ENGINE = "v4"

# Polarities that dilute stance clarity (unknown-like for Opinion V4).
_UNCLEAR_POLARITIES = frozenset({"unknown", "uncertain", "shifting"})


def _unknown_share(polarity_distribution: dict[str, int]) -> float:
    total = sum(int(v) for v in (polarity_distribution or {}).values()) or 0
    if total <= 0:
        return 1.0
    unk = sum(int((polarity_distribution or {}).get(k) or 0) for k in _UNCLEAR_POLARITIES)
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


def _shifting_share(polarity_distribution: dict[str, int]) -> float:
    total = sum(int(v) for v in (polarity_distribution or {}).values()) or 0
    if total <= 0:
        return 0.0
    return round(int((polarity_distribution or {}).get("shifting") or 0) / float(total), 4)


class OpinionDynamics:
    """Round-by-round polarity chronology and shift analytics (V4)."""

    def __init__(self, *, max_rounds_tracked: int = 16) -> None:
        self.max_rounds_tracked = max(1, int(max_rounds_tracked))
        self._round_snapshots: list[dict[str, Any]] = []
        self._seen_steps: set[int] = set()
        self._grounded_reply_share: float | None = None
        self._motivation_coverage: float | None = None
        self._weak_overlap_rate: float | None = None

    def clear(self) -> None:
        self._round_snapshots.clear()
        self._seen_steps.clear()
        self._grounded_reply_share = None
        self._motivation_coverage = None
        self._weak_overlap_rate = None

    def set_interaction_signals(
        self,
        *,
        grounded_reply_share: float | None = None,
        motivation_coverage: float | None = None,
        weak_overlap_rate: float | None = None,
    ) -> None:
        """Optional V3/V4 signals from Interaction Quality / Persona Motivation."""
        if grounded_reply_share is not None:
            self._grounded_reply_share = max(
                0.0, min(1.0, float(grounded_reply_share))
            )
        if motivation_coverage is not None:
            self._motivation_coverage = max(
                0.0, min(1.0, float(motivation_coverage))
            )
        if weak_overlap_rate is not None:
            self._weak_overlap_rate = max(0.0, min(1.0, float(weak_overlap_rate)))

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
            "shifting_share": _shifting_share(dist),
            "polarization": _polarization(dist),
            "entropy": _entropy(dist),
            "grounded_reply_share": self._grounded_reply_share,
            "motivation_coverage": self._motivation_coverage,
            "weak_overlap_rate": self._weak_overlap_rate,
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
        # V3/V4: boost stability slightly when motivation coverage is high and
        # grounded replies dominate; soft penalty for weak overlaps.
        if self._motivation_coverage is not None and self._grounded_reply_share is not None:
            weak_pen = 0.04 * float(self._weak_overlap_rate or 0.0)
            stability = round(
                min(
                    1.0,
                    max(
                        0.0,
                        stability * 0.85
                        + 0.10 * float(self._motivation_coverage)
                        + 0.05 * float(self._grounded_reply_share)
                        - weak_pen,
                    ),
                ),
                4,
            )
        final_dist = (frames[-1].get("polarity_distribution") or {}) if frames else {}
        final_clarity = clarities[-1] if clarities else 0.0
        final_unk = unknown_series[-1] if unknown_series else 1.0

        return {
            "backend": "local",
            "engine": OPINION_ENGINE,
            "version": 4,
            "quality_fix": QUALITY_FIX,
            "rounds_tracked": rounds,
            "total_evidenced_changes": total_changes,
            "shift_rate_per_agent_round": shift_rate,
            "transitions": dict(sorted(transitions.items())),
            "stance_clarity": final_clarity,
            "unknown_share": final_unk,
            "shifting_share": _shifting_share(final_dist),
            "stability_index": stability,
            "polarization": _polarization(final_dist),
            "entropy": _entropy(final_dist),
            "grounded_reply_share": self._grounded_reply_share,
            "motivation_coverage": self._motivation_coverage,
            "weak_overlap_rate": self._weak_overlap_rate,
            "unknown_share_by_round": [
                {"step": f.get("step"), "unknown_share": f.get("unknown_share")}
                for f in frames
            ],
            "chronology": frames,
            "note": (
                "Meinungsdynamik V4: Chronologie + Stance-Klarheit (shifting "
                "zählt als unklar), Stabilität, Polarisierung, Grounding-Floor. "
                "Keine erfundenen Flips; Scripted liefert Offline-Evidenz."
            ),
        }
