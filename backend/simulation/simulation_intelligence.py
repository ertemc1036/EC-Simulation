"""Phase 8.0 — Simulation Intelligence (offline foundation layer).

Aggregates persona diversity, scheduler coverage, cost plan, and society
signals into one dashboard-ready payload. No provider calls.
"""
from __future__ import annotations

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


QUALITY_FIX = "8.0"
PERSONA_ENGINE = "v2"


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
) -> dict[str, Any]:
    """Compact Phase-8 intelligence snapshot for status + result JSON."""
    agents = agents or []
    n = max(1, int(agent_count) or len(agents) or 1)
    issues = _persona_issues(agents)
    role_n = _unique_role_labels(agents)
    occ_n = _unique_occupations(agents)
    diversity = round(min(1.0, (role_n + occ_n) / max(2.0, 2.0 * min(n, 5))), 4)

    plan = cost_plan.to_dict() if isinstance(cost_plan, CostTokenPlan) else (cost_plan or {})
    sched = scheduler or {}
    analytics = society_analytics or {}
    disc = discussion or {}
    positions = agent_positions or {}
    pol = positions.get("polarity_distribution") or {}
    unknown_share = float(pol.get("unknown") or 0) / max(
        1.0, float(sum(int(v) for v in pol.values()) or 1)
    )

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

    return {
        "phase": 8,
        "quality_fix": QUALITY_FIX,
        "persona_engine": PERSONA_ENGINE,
        "backend": "local",
        "execution_kind": execution_kind,
        "paid_llm_unlocked": False,
        "persona": {
            "agent_count": len(agents),
            "unique_role_labels": role_n,
            "unique_occupations": occ_n,
            "diversity_score": diversity,
            "issue_flags": issues,
            "issue_count": len(issues),
            "ok": len(issues) == 0 and role_n >= min(3, n),
        },
        "scheduler": {
            "backend": sched.get("backend"),
            "plans": sched.get("plans", 0),
            "waves": sched.get("waves", 0),
            "unique_pairs": sched.get("unique_pairs", 0),
            "self_as_partner_count": sched.get("self_as_partner_count", 0),
            "coverage": sched.get("agent_coverage"),
            "scalable": bool(sched.get("scalable", True)),
        },
        "cost_plan": plan,
        "discussion": {
            "argument_count": disc.get("argument_count", 0),
            "repetition_blocks": disc.get("repetition_blocks", 0),
            "cross_agent_replies": disc.get("cross_agent_replies", 0),
            "self_replies": disc.get("self_replies", 0),
            "quality_fix": disc.get("quality_fix") or QUALITY_FIX,
        },
        "stance": {
            "polarity_distribution": pol,
            "change_count": positions.get("change_count", 0),
            "unknown_share": round(unknown_share, 4),
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
        },
        "kpis": {
            "persona_diversity": diversity,
            "persona_issues": len(issues),
            "scheduler_self_hits": int(sched.get("self_as_partner_count") or 0),
            "planned_cost_usd": float(plan.get("planned_cost_usd") or 0.0),
            "unknown_stance_share": round(unknown_share, 4),
            "repetition_blocks": int(disc.get("repetition_blocks") or 0),
        },
        "note": (
            "Simulation Intelligence (Phase 8.0): Persona Engine V2, scalable "
            "scheduler, cost/token plan, society signals — offline foundation; "
            "Live-Ceilings unverändert; paid_llm_unlocked=false."
        ),
    }
