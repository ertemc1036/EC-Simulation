"""Deterministic ManualAction scripts derived from agent personalities.

These are NOT fake DB writes: every action is executed via real oasis.ManualAction
inside OasisEnv.step. Content is personality-templated because LLM mode is off.
Works with dynamic agent counts (Phase 3), not only the fixed five.

Phase 7.1: public posts/comments carry thesis-specific arguments — never raw
persona goals / Zielkonflikt / style instructions. Manual runs are marked
``[synthetisch]``. Position markers enable evidenced polarity + shifts.
"""
from __future__ import annotations

from oasis import ActionType, ManualAction

from .agents import AgentProfile, DEFAULT_AGENTS
from .topic_context import (
    detect_topic_domain,
    measure_label,
    synthetic_tag,
)


def _stance_bucket(agent_id: int) -> str:
    """Deterministic initial polarity bucket per agent."""
    return ("pro", "contra", "exception")[agent_id % 3]


def _argument_line(
    agent: AgentProfile,
    *,
    scenario: str,
    round_n: int,
    partner_hint: int | None = None,
) -> str:
    """Public contribution — arguments only, no instruction leakage."""
    domain = detect_topic_domain(scenario)
    measure = measure_label(scenario, domain)
    tag = synthetic_tag()
    bucket = _stance_bucket(agent.agent_id)
    name = agent.name

    if domain == "income":
        pro = (
            f"Ich bin für ein {measure} — Existenzsicherung ohne Sanktionsdruck "
            f"stärkt Teilhabe und Planungssicherheit."
        )
        contra = (
            f"Ich lehne das {measure} ab — Finanzierung und Arbeitsanreize sind "
            f"ungeklärt; Inflations- und Steuerlast-Risiken überwiegen."
        )
        exception = (
            f"Zielkonflikt: Pilot mit klaren Abbruchkriterien statt Sofort-Vollauszahlung; "
            f"Übergangsfristen und soziale Abfederung nötig."
        )
        shift_to_exception = (
            f"Ich ändere meine Position nach dem Kommentar — werde überzeugt: "
            f"Piloten und Ausnahmeregelungen statt pauschaler Vollauszahlung."
        )
        shift_hold = (
            f"Ich halte dagegen und widerspreche klar: nein zur ungeprüften "
            f"Vollauszahlung ohne Gegenfinanzierung."
        )
        agree = (
            f"stimme zu: ja zum Grundeinkommen als Absicherung, mit Messpunkten."
        )
        disagree = (
            f"widerspreche: gegen das Grundeinkommen ohne Finanzierungsnachweis."
        )
    elif domain == "mobility":
        pro = (
            f"Ich bin für das Verbot — sinnvolles Verbot im Kerngebiet schützt "
            f"Luft und Lebensqualität."
        )
        contra = (
            f"Ich lehne das Verbot ab — pauschale Verbote gefährden Arbeitsplätze; "
            f"klare Ausnahmen und Lieferfenster nötig."
        )
        exception = (
            f"Zielkonflikt: Ausnahmeregelungen und Übergangsfristen statt "
            f"pauschalem Verbot."
        )
        shift_to_exception = (
            f"Ich ändere meine Position nach dem Kommentar — werde überzeugt: "
            f"Ausnahmeregelungen statt pauschalem Verbot."
        )
        shift_hold = (
            f"Ich halte dagegen und widerspreche klar: nein zum pauschalen Verbot."
        )
        agree = "stimme zu: ja zum Verbot, mit Messpunkten."
        disagree = "widerspreche: gegen das Verbot ohne soziale Abfederung."
    else:
        pro = f"Ich bin für {measure} — die Vorteile überwiegen bei klaren Messpunkten."
        contra = (
            f"Ich lehne {measure} ab — Kosten und Nebenwirkungen sind unzureichend belegt."
        )
        exception = (
            f"Zielkonflikt: schrittweise Einführung mit Ausnahmen statt Sofort-Vollumsetzung."
        )
        shift_to_exception = (
            f"Ich ändere meine Position nach dem Kommentar — werde überzeugt: "
            f"Piloten und Ausnahmen statt Sofortumsetzung."
        )
        shift_hold = (
            f"Ich halte dagegen und widerspreche klar: nein zur ungeprüften Sofortumsetzung."
        )
        agree = f"stimme zu: ja zu {measure}, mit Messpunkten."
        disagree = f"widerspreche: gegen {measure} ohne belastbare Evidenz."

    # Round arc: establish → react → shift/hold → consolidate.
    if round_n <= 1:
        body = {"pro": pro, "contra": contra, "exception": exception}[bucket]
    elif round_n == 2:
        body = agree if bucket == "pro" else (disagree if bucket == "contra" else exception)
    elif round_n == 3:
        # Even agents shift toward exception (evidenced change); odd hold contra/pro.
        body = shift_to_exception if agent.agent_id % 2 == 0 else shift_hold
    else:
        if agent.agent_id % 2 == 0:
            body = (
                "stimme zu und teile diese Einschätzung — "
                "Ausnahmeregelungen und Piloten bleiben zentral."
            )
        else:
            body = (
                "widerspreche weiterhin: sehe das anders — "
                f"gegen ungeprüfte Vollumsetzung von {measure}."
            )

    ref = ""
    if partner_hint is not None and round_n >= 2:
        # Local OASIS reply ref (no parent_comment_id).
        ref = f" →Kommentar#{partner_hint}"

    return f"{tag} [{name}]{ref} {body}"


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
    tag = synthetic_tag()
    measure = measure_label(scenario)

    rounds: list[dict[int, list[ManualAction]]] = []

    # Round 0 — seed posts from several agents (richer discussion than 1–2 posts).
    seed: dict[int, list[ManualAction]] = {}
    seed_n = min(max(3, len(ids) // 8), min(8, len(ids)))
    seed_authors = ids[:seed_n]
    for aid in seed_authors:
        a = by_id[aid]
        seed[aid] = [
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content": _argument_line(a, scenario=scenario, round_n=1)
                },
            )
        ]
    rounds.append(seed)

    # Round 1 — reactions + argumentative comments on seed posts.
    react: dict[int, list[ManualAction]] = {}
    reactors = [aid for aid in ids if aid not in seed_authors] or ids[1:]
    if not reactors and ids:
        reactors = [ids[0]]
    for i, aid in enumerate(reactors):
        a = by_id[aid]
        target_post = (i % seed_n) + 1 if seed_n else 1
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
                    "content": _argument_line(a, scenario=scenario, round_n=1),
                },
            )
        )
        react[aid] = acts
    if react:
        rounds.append(react)

    # Round 2 — follow-ups with →Kommentar# refs + agree/disagree markers.
    follow: dict[int, list[ManualAction]] = {}
    # comment_id roughly maps 1..len(reactors) from round 1; use modular refs.
    comment_count = max(1, len(reactors))
    for i, aid in enumerate(ids[: min(len(ids), max(6, seed_n + 4))]):
        a = by_id[aid]
        ref_cid = (i % comment_count) + 1
        target_post = (i % seed_n) + 1 if seed_n else 1
        follow[aid] = [
            ManualAction(
                action_type=ActionType.CREATE_COMMENT,
                action_args={
                    "post_id": target_post,
                    "content": _argument_line(
                        a,
                        scenario=scenario,
                        round_n=2,
                        partner_hint=ref_cid,
                    ),
                },
            )
        ]
    # Extra summary post to keep discussion multi-threaded.
    if len(ids) >= 3:
        a2 = by_id[ids[2]]
        follow.setdefault(ids[2], [])
        follow[ids[2]].append(
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content": (
                        f"{tag} [{a2.name} / Struktur] Zwischenstand zu «{measure}»: "
                        f"Positionen sammeln, dann bewerten — "
                        f"{_argument_line(a2, scenario=scenario, round_n=2)}"
                    )
                },
            )
        )
    if len(ids) >= 4:
        follow[ids[3]] = follow.get(ids[3], []) + [
            ManualAction(
                action_type=ActionType.FOLLOW,
                action_args={"followee_id": ids[0]},
            )
        ]
    if follow:
        rounds.append(follow)

    # Round 3 — evidenced shifts for even agents; hold for odd.
    shift: dict[int, list[ManualAction]] = {}
    summary_post = seed_n + 1 if len(ids) >= 3 else 1
    for i, aid in enumerate(ids[: min(len(ids), max(8, seed_n + 2))]):
        a = by_id[aid]
        ref_cid = (i % comment_count) + 1
        shift[aid] = [
            ManualAction(
                action_type=ActionType.CREATE_COMMENT,
                action_args={
                    "post_id": max(1, min(summary_post, seed_n + 1)),
                    "content": _argument_line(
                        a,
                        scenario=scenario,
                        round_n=3,
                        partner_hint=ref_cid,
                    ),
                },
            )
        ]
    if shift:
        rounds.append(shift)

    # Round 4 — consolidate + light reactions.
    close: dict[int, list[ManualAction]] = {}
    for i, aid in enumerate(ids[: min(6, len(ids))]):
        a = by_id[aid]
        if i < 2:
            close[aid] = [
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": max(1, summary_post)},
                )
            ]
        else:
            ref_cid = (i % comment_count) + 1
            close[aid] = [
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={
                        "post_id": max(1, min(summary_post, seed_n)),
                        "content": _argument_line(
                            a,
                            scenario=scenario,
                            round_n=4,
                            partner_hint=ref_cid,
                        ),
                    },
                )
            ]
    if close:
        rounds.append(close)

    return rounds[: max(1, max_steps)]
