"""Phase 8.0–8.3 — Simulation Intelligence (scaling + debate + realistic agents).

Aggregates persona motivation V4, scheduler coverage/waves-in-execution,
cost plan V3, interaction engine, debate quality V2, stance V3, opinion V3,
argument-grounded interaction V3, and interaction quality. No provider calls.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    OFFLINE_MAX_AGENTS,
    OFFLINE_MAX_ROUNDS,
    OFFLINE_SCALE_PRESETS,
)
from backend.simulation.cost_token_planner import CostTokenPlan


QUALITY_FIX = "8.3"
PERSONA_ENGINE = "v4"
DEBATE_ENGINE = "v2"
STANCE_CLASSIFIER = "v3"
OPINION_ENGINE = "v3"
ARGUMENT_ENGINE = "v2"
GROUNDED_ENGINE = "v3"
INTERACTION_QUALITY_ENGINE = "v3"


def _unique_occupations(agents: list[dict[str, Any]]) -> int:
    occ = {
        str(a.get("occupation") or "").strip().lower()
        for a in agents
        if str(a.get("occupation") or "").strip()
    }
    return len(occ)


def _unique_role_labels(agents: list[dict[str, Any]]) -> int:
    labels = {
        str(a.get("role_label") or "").strip().lower()
        for a in agents
        if str(a.get("role_label") or "").strip()
    }
    return len(labels)


def _unique_names(agents: list[dict[str, Any]]) -> int:
    names = {
        str(a.get("name") or "").strip().lower()
        for a in agents
        if str(a.get("name") or "").strip()
    }
    return len(names)


def _role_balance(agents: list[dict[str, Any]]) -> float:
    """1.0 = perfectly even role distribution; lower = concentrated."""
    labels = [
        str(a.get("role_label") or "").strip().lower()
        for a in agents
        if str(a.get("role_label") or "").strip()
    ]
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    k = len(counts)
    if k <= 1:
        return 1.0 if n <= 1 else 0.0
    ideal = n / k
    # Mean absolute deviation from ideal, normalized.
    mad = sum(abs(c - ideal) for c in counts.values()) / n
    return round(max(0.0, 1.0 - mad), 4)


def _persona_issues(agents: list[dict[str, Any]]) -> list[str]:
    """Heuristic flags for broken V1 personas (ddfedb31-class)."""
    issues: list[str] = []
    topicish = {
        "datenschutz",
        "chancengleichheit",
        "lernleistungen",
        "privatheit",
        "privacy",
    }
    seen_names: set[str] = set()
    for a in agents:
        occ = str(a.get("occupation") or "").strip().lower()
        role = str(a.get("role_label") or "").strip().lower()
        name = str(a.get("name") or "")
        if " " in occ and any(ch.isdigit() is False and ord(ch) < 128 for ch in occ):
            # "lehrkr ften" style ascii hole from stripped umlauts
            if "  " in occ or "kr f" in occ or occ.endswith(" ften"):
                issues.append(f"broken_occupation:{a.get('agent_id')}")
        if occ in topicish or role in topicish:
            issues.append(f"topic_as_stakeholder:{a.get('agent_id')}")
        if occ == role and occ in {"eltern", "lehrkraeften", "lehrkräften"}:
            issues.append(f"group_label_as_occupation:{a.get('agent_id')}")
        # Duplicate first/last like "Noa Noa"
        parts = name.split()
        if len(parts) >= 2 and parts[0].lower() == parts[1].lower():
            issues.append(f"duplicate_name_token:{a.get('agent_id')}")
        key = name.strip().lower()
        if key and key in seen_names:
            issues.append(f"duplicate_display_name:{a.get('agent_id')}")
        if key:
            seen_names.add(key)
    return issues


def build_simulation_intelligence(
    *,
    agents: list[dict[str, Any]],
    execution_kind: str,
    agent_count: int,
    max_steps: int,
    step: int | None,
    cost_plan: CostTokenPlan | dict[str, Any] | None,
    scheduler: dict[str, Any] | None,
    society_analytics: dict[str, Any] | None,
    discussion: dict[str, Any] | None,
    agent_positions: dict[str, Any] | None,
    interaction_engine: dict[str, Any] | None = None,
    opinion_dynamics: dict[str, Any] | None = None,
    interaction_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact Phase-8.3 intelligence snapshot for status + result JSON."""
    agents = agents or []
    n = max(1, int(agent_count) or len(agents) or 1)
    issues = _persona_issues(agents)
    role_n = _unique_role_labels(agents)
    occ_n = _unique_occupations(agents)
    name_n = _unique_names(agents)
    balance = _role_balance(agents)
    diversity = round(
        min(
            1.0,
            (
                (name_n / max(1.0, float(len(agents) or n))) * 0.4
                + (occ_n / max(1.0, float(len(agents) or n))) * 0.3
                + min(1.0, role_n / max(3.0, min(n, 10))) * 0.2
                + balance * 0.1
            ),
        ),
        4,
    )

    plan = cost_plan.to_dict() if isinstance(cost_plan, CostTokenPlan) else (cost_plan or {})
    sched = scheduler or {}
    analytics = society_analytics or {}
    disc = discussion or {}
    positions = agent_positions or {}
    iex = interaction_engine or {}
    opinion = opinion_dynamics or {}
    iq = interaction_quality or disc.get("interaction_quality") or {}
    pol = positions.get("polarity_distribution") or {}
    unknown_share = float(pol.get("unknown") or 0) / max(
        1.0, float(sum(int(v) for v in pol.values()) or 1)
    )
    uncertain_share = float(pol.get("uncertain") or 0) / max(
        1.0, float(sum(int(v) for v in pol.values()) or 1)
    )
    unclear_share = round(unknown_share + uncertain_share, 4)

    scale_ready = (
        execution_kind in {"manual", "scripted"}
        and n <= OFFLINE_MAX_AGENTS
        and int(max_steps) <= OFFLINE_MAX_ROUNDS
        and plan.get("paid_llm_unlocked") is not True
    )
    live_safe = (
        plan.get("paid_llm_unlocked") is False
        or plan.get("paid_llm_unlocked") is None
    ) and bool(plan.get("within_live_ceilings", True))

    waves_executed = int(sched.get("waves_executed") or iex.get("waves_executed") or 0)
    waves_planned = int(
        sched.get("waves_planned") or sched.get("waves") or iex.get("waves_planned") or 0
    )

    arg_intel = disc.get("argument_intelligence") or {}
    debate_q = disc.get("debate_quality") or {}
    debate_score = float(debate_q.get("score") or 0.0)
    stance_clarity = float(
        opinion.get("stance_clarity")
        if opinion.get("stance_clarity") is not None
        else max(0.0, 1.0 - unclear_share)
    )
    grounded = iq.get("argument_grounded") or disc.get("argument_grounded") or {}
    mot_cov = float(
        (iq.get("components") or {}).get("persona_motivation_coverage")
        or (opinion.get("motivation_coverage") or 0.0)
        or 0.0
    )
    if mot_cov <= 0.0:
        with_mot = sum(
            1
            for a in agents
            if (a.get("motivation") or {}).get("values")
            or "Motivation:" in str(a.get("goal") or "")
        )
        mot_cov = round(with_mot / max(1, len(agents) or 1), 4)

    return {
        "phase": 8,
        "quality_fix": QUALITY_FIX,
        "persona_engine": PERSONA_ENGINE,
        "debate_engine": disc.get("debate_engine") or DEBATE_ENGINE,
        "stance_classifier": positions.get("stance_classifier") or STANCE_CLASSIFIER,
        "opinion_engine": opinion.get("engine") or OPINION_ENGINE,
        "argument_engine": arg_intel.get("engine") or ARGUMENT_ENGINE,
        "argument_grounded_engine": grounded.get("engine") or GROUNDED_ENGINE,
        "interaction_quality_engine": iq.get("engine") or INTERACTION_QUALITY_ENGINE,
        "backend": "local",
        "execution_kind": execution_kind,
        "paid_llm_unlocked": False,
        "persona": {
            "engine": PERSONA_ENGINE,
            "agent_count": len(agents),
            "unique_role_labels": role_n,
            "unique_occupations": occ_n,
            "unique_names": name_n,
            "role_balance": balance,
            "diversity_score": diversity,
            "motivation_coverage": mot_cov,
            "issue_flags": issues,
            "issue_count": len(issues),
            "ok": len(issues) == 0 and role_n >= min(3, n) and name_n == len(agents),
        },
        "scheduler": {
            "backend": sched.get("backend"),
            "version": sched.get("version", 3),
            "plans": sched.get("plans", 0),
            "waves": sched.get("waves", 0),
            "waves_planned": waves_planned,
            "waves_executed": waves_executed,
            "unique_pairs": sched.get("unique_pairs", 0),
            "self_as_partner_count": sched.get("self_as_partner_count", 0),
            "coverage": sched.get("agent_coverage"),
            "scalable": bool(sched.get("scalable", True)),
            "execution_paths_used": sched.get("execution_paths_used") or [],
        },
        "interaction_engine": {
            "backend": iex.get("backend"),
            "quality_fix": iex.get("quality_fix") or QUALITY_FIX,
            "rounds_prepared": iex.get("rounds_prepared", 0),
            "waves_planned": iex.get("waves_planned", 0),
            "waves_executed": iex.get("waves_executed", 0),
            "path": iex.get("path"),
        },
        "cost_plan": plan,
        "discussion": {
            "argument_count": disc.get("argument_count", 0),
            "repetition_blocks": disc.get("repetition_blocks", 0),
            "cross_agent_replies": disc.get("cross_agent_replies", 0),
            "self_replies": disc.get("self_replies", 0),
            "quality_fix": disc.get("quality_fix") or QUALITY_FIX,
            "debate_engine": disc.get("debate_engine") or DEBATE_ENGINE,
            "debate_quality_score": debate_score,
            "mean_argument_quality": arg_intel.get("mean_quality_score", 0.0),
            "argument_diversity": arg_intel.get("argument_diversity", 0.0),
            "claim_kinds": arg_intel.get("claim_kinds") or {},
            "grounded_reply_rate": grounded.get("grounded_reply_rate"),
            "invented_attribution_rate": grounded.get("invented_attribution_rate"),
        },
        "stance": {
            "classifier": positions.get("stance_classifier") or STANCE_CLASSIFIER,
            "polarity_distribution": pol,
            "change_count": positions.get("change_count", 0),
            "unknown_share": round(unknown_share, 4),
            "unclear_share": unclear_share,
            "clarity": stance_clarity,
            "quality_fix": positions.get("quality_fix") or QUALITY_FIX,
        },
        "opinion": {
            "engine": opinion.get("engine") or OPINION_ENGINE,
            "version": opinion.get("version", 3),
            "stance_clarity": stance_clarity,
            "stability_index": opinion.get("stability_index"),
            "polarization": opinion.get("polarization"),
            "shift_rate_per_agent_round": opinion.get("shift_rate_per_agent_round", 0.0),
            "rounds_tracked": opinion.get("rounds_tracked", 0),
            "grounded_reply_share": opinion.get("grounded_reply_share"),
            "motivation_coverage": opinion.get("motivation_coverage", mot_cov),
        },
        "interaction_quality": iq
        or {
            "score": 0.0,
            "engine": INTERACTION_QUALITY_ENGINE,
            "quality_fix": QUALITY_FIX,
        },
        "debate_quality": debate_q
        or {
            "score": debate_score,
            "engine": DEBATE_ENGINE,
            "quality_fix": QUALITY_FIX,
        },
        "society_kpis": (analytics.get("kpis") or {}),
        "scale": {
            "agent_count": n,
            "step": step,
            "max_steps": max_steps,
            "offline_presets": list(OFFLINE_SCALE_PRESETS),
            "offline_max_agents": OFFLINE_MAX_AGENTS,
            "offline_max_rounds": OFFLINE_MAX_ROUNDS,
            "live_llm_max_agents": LIVE_TEST_MAX_AGENTS,
            "live_llm_max_rounds": LIVE_TEST_MAX_ROUNDS,
            "live_llm_max_calls": LIVE_TEST_MAX_LLM_CALLS,
            "offline_scale_ready": scale_ready,
            "live_ceilings_safe": live_safe,
            "paid_llm_unlocked": False,
            "waves_in_execution": waves_executed > 0,
        },
        "kpis": {
            "persona_diversity": diversity,
            "persona_issues": len(issues),
            "persona_motivation_coverage": mot_cov,
            "scheduler_self_hits": int(sched.get("self_as_partner_count") or 0),
            "waves_executed": waves_executed,
            "planned_waves": int(plan.get("planned_waves") or waves_planned or 0),
            "planned_cost_usd": float(plan.get("planned_cost_usd") or 0.0),
            "unknown_stance_share": round(unknown_share, 4),
            "stance_clarity": stance_clarity,
            "repetition_blocks": int(disc.get("repetition_blocks") or 0),
            "debate_quality_score": debate_score,
            "argument_diversity": float(arg_intel.get("argument_diversity") or 0.0),
            "mean_argument_quality": float(arg_intel.get("mean_quality_score") or 0.0),
            "opinion_stability": float(opinion.get("stability_index") or 0.0),
            "grounded_reply_rate": float(grounded.get("grounded_reply_rate") or 0.0),
            "invented_attribution_rate": float(
                grounded.get("invented_attribution_rate") or 0.0
            ),
            "interaction_quality_score": float(iq.get("score") or 0.0),
        },
        "note": (
            "Simulation Intelligence (Phase 8.3): Persona Motivation V4, "
            "Argument-Grounded Interaction V3, Stance Classification V3, "
            "Opinion Dynamics V3, Interaction Quality — offline; "
            "Live-Ceilings unverändert; paid_llm_unlocked=false."
        ),
    }
