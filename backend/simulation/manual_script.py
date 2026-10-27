"""Deterministic ManualAction scripts derived from agent personalities.

These are NOT fake DB writes: every action is executed via real oasis.ManualAction
inside OasisEnv.step. Content is personality-templated because LLM mode is off.
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
    agents = agents or DEFAULT_AGENTS
    by_id = {a.agent_id: a for a in agents}
    scenario_short = scenario.strip()[:280] or "Offenes Diskussionsthema"

    rounds: list[dict[int, list[ManualAction]]] = []

    # Round 0 — seed posts from Alice + Bob
    rounds.append(
        {
            0: [
                ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={
                        "content": (
                            f"[Alice / Analyse] Szenario: «{scenario_short}». "
                            f"Bevor wir urteilen: Welche Annahmen sind belegt? "
                            f"Ziel: {by_id[0].goal}"
                        )
                    },
                )
            ],
            1: [
                ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={
                        "content": (
                            f"[Bob / Builder] Zu «{scenario_short}»: Lasst uns "
                            f"etwas Konkretes ausprobieren statt nur debattieren. "
                            f"Ziel: {by_id[1].goal}"
                        )
                    },
                )
            ],
        }
    )

    # Round 1 — reactions + comments (post_id 1 = Alice, 2 = Bob)
    rounds.append(
        {
            2: [
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": 1},
                ),
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": 1,
                        "content": (
                            "[Cara] Bitte bleibt sachlich. Klärt Begriffe zuerst, "
                            "dann bewertet ihr Optionen."
                        ),
                    },
                ),
            ],
            3: [
                ManualAction(
                    action_type=ActionType.DISLIKE_POST,
                    action_args={"post_id": 1},
                ),
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": 2,
                        "content": (
                            "[Dan] Typischer Optimismus. Was, wenn der «praktische» "
                            "Weg die eigentlichen Risiken verdeckt?"
                        ),
                    },
                ),
            ],
            4: [
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": 2},
                ),
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": 1,
                        "content": (
                            "[Eve] Beide Seiten berühren echte Menschen. "
                            "Welche Gruppen tragen die Kosten?"
                        ),
                    },
                ),
            ],
        }
    )

    # Round 2 — follow-ups / second wave posts
    rounds.append(
        {
            0: [
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": 2,
                        "content": (
                            "[Alice] Bob, ohne Messkriterien bleibt «ausprobieren» "
                            "Anekdote. Welche Metrik zählt?"
                        ),
                    },
                )
            ],
            1: [
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": 1,
                        "content": (
                            "[Bob] Fair. Vorschlag: kleiner Pilot, klare Abbruchkriterien, "
                            "öffentliche Auswertung."
                        ),
                    },
                )
            ],
            2: [
                ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={
                        "content": (
                            f"[Cara / Moderation] Thread-Zusammenfassung zu «{scenario_short}»: "
                            "Belege (Alice), Experiment (Bob), Risiko (Dan), Betroffene (Eve)."
                        )
                    },
                )
            ],
            3: [
                ManualAction(
                    action_type=ActionType.FOLLOW,
                    action_args={"followee_id": 0},
                )
            ],
            4: [
                ManualAction(
                    action_type=ActionType.FOLLOW,
                    action_args={"followee_id": 2},
                )
            ],
        }
    )

    # Round 3 — closing reactions on Cara's summary (post_id 3)
    rounds.append(
        {
            0: [
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": 3},
                )
            ],
            1: [
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": 3},
                )
            ],
            3: [
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": 3,
                        "content": (
                            "[Dan] Zusammenfassung ok — fehlt aber der Punkt, "
                            "wer bei Fehlern haftet."
                        ),
                    },
                )
            ],
            4: [
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": 3,
                        "content": (
                            "[Eve] Danke Cara. Lasst uns Betroffene früh einbinden, "
                            "nicht erst nach dem Pilot."
                        ),
                    },
                )
            ],
        }
    )

    return rounds[: max(1, max_steps)]
