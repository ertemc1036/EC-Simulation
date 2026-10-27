"""Deterministic ManualAction scripts derived from agent personalities.

These are NOT fake DB writes: every action is executed via real oasis.ManualAction
inside OasisEnv.step. Content is personality-templated because LLM mode is off.
Works with dynamic agent counts (Phase 3), not only the fixed five.
"""
from __future__ import annotations

from oasis import ActionType, ManualAction

from .agents import AgentProfile, DEFAULT_AGENTS


def build_manual_rounds(
    scenario: str,
    agents: list[AgentProfile] | None = None,
    max_steps: int = 4,
) -> list[dict[int, list[ManualAction]]]:
    """Return a list of action maps: agent_id -> list[ManualAction]."""
    agents = list(agents or DEFAULT_AGENTS)
    if not agents:
        return []
    by_id = {a.agent_id: a for a in agents}
    ids = [a.agent_id for a in agents]
    scenario_short = scenario.strip()[:280] or "Offenes Diskussionsthema"

    rounds: list[dict[int, list[ManualAction]]] = []

    # Round 0 — seed posts from first one/two agents
    seed: dict[int, list[ManualAction]] = {}
    seed_authors = ids[: min(2, len(ids))]
    for aid in seed_authors:
        a = by_id[aid]
        seed[aid] = [
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content": (
                        f"[{a.name}] Szenario: «{scenario_short}». "
                        f"{a.description[:120]} Ziel: {a.goal}"
                    )
                },
            )
        ]
    rounds.append(seed)

    # Round 1 — reactions + comments on earliest posts
    react: dict[int, list[ManualAction]] = {}
    reactors = ids[2:] if len(ids) > 2 else ids[1:]
    if not reactors and len(ids) >= 1:
        reactors = [ids[0]]
    for i, aid in enumerate(reactors):
        a = by_id[aid]
        target_post = 1 if i % 2 == 0 else min(2, len(seed_authors))
        target_post = max(1, target_post)
        acts: list[ManualAction] = []
        if i % 3 == 0:
            acts.append(
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": target_post},
                )
            )
        elif i % 3 == 1:
            acts.append(
                ManualAction(
                    action_type=ActionType.DISLIKE_POST,
                    action_args={"post_id": target_post},
                )
            )
        else:
            acts.append(
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": 1},
                )
            )
        acts.append(
            ManualAction(
                action_type=ActionType.CREATE_COMMENT,
                action_args={
                    "post_id": target_post,
                    "content": (
                        f"[{a.name}] Bezug zu «{scenario_short[:80]}»: "
                        f"{a.goal} (Stil: {a.style})."
                    ),
                },
            )
        )
        react[aid] = acts
    if react:
        rounds.append(react)

    # Round 2 — follow-ups / summary post
    follow: dict[int, list[ManualAction]] = {}
    if len(ids) >= 1:
        a0 = by_id[ids[0]]
        follow[ids[0]] = [
            ManualAction(
                action_type=ActionType.CREATE_COMMENT,
                action_args={
                    "post_id": min(2, len(seed_authors)) if len(seed_authors) > 1 else 1,
                    "content": (
                        f"[{a0.name}] Nachfassen: Welche Annahmen zu «{scenario_short[:60]}» "
                        f"sind belegt?"
                    ),
                },
            )
        ]
    if len(ids) >= 2:
        a1 = by_id[ids[1]]
        follow[ids[1]] = [
            ManualAction(
                action_type=ActionType.CREATE_COMMENT,
                action_args={
                    "post_id": 1,
                    "content": (
                        f"[{a1.name}] Vorschlag: kleiner Pilot mit klaren "
                        f"Abbruchkriterien zu «{scenario_short[:60]}»."
                    ),
                },
            )
        ]
    if len(ids) >= 3:
        a2 = by_id[ids[2]]
        follow[ids[2]] = [
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content": (
                        f"[{a2.name} / Struktur] Zwischenstand zu «{scenario_short}»: "
                        "Positionen sammeln, dann bewerten."
                    )
                },
            )
        ]
    if len(ids) >= 4:
        follow[ids[3]] = [
            ManualAction(
                action_type=ActionType.FOLLOW,
                action_args={"followee_id": ids[0]},
            )
        ]
    if len(ids) >= 5:
        follow[ids[4]] = [
            ManualAction(
                action_type=ActionType.FOLLOW,
                action_args={"followee_id": ids[min(2, len(ids) - 1)]},
            )
        ]
    if follow:
        rounds.append(follow)

    # Round 3 — closing reactions on latest summary-ish post (best-effort post_id 3)
    close: dict[int, list[ManualAction]] = {}
    summary_post = 3 if len(ids) >= 3 else 1
    for aid in ids[: min(3, len(ids))]:
        a = by_id[aid]
        if aid == ids[0] or (len(ids) > 1 and aid == ids[1]):
            close[aid] = [
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": summary_post},
                )
            ]
        else:
            close[aid] = [
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": summary_post,
                        "content": (
                            f"[{a.name}] Abschlussgedanke: {a.goal}"
                        ),
                    },
                )
            ]
    if close:
        rounds.append(close)

    return rounds[: max(1, max_steps)]
