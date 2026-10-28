"""Phase 7 — Society Analytics aggregation for dashboard + offline metrics."""
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


def build_society_analytics(
    *,
    agent_count: int,
    max_steps: int,
    step: int | None,
    execution_kind: str,
    posts: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    agent_memory: dict[str, Any],
    agent_positions: dict[str, Any],
    relationships: dict[str, Any],
    opinion: dict[str, Any],
    knowledge_graph: dict[str, Any],
    interaction_summary: dict[str, Any],
    scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact analytics payload — offline 10/25/50 × ≤4 rounds friendly."""
    pol = agent_positions.get("polarity_distribution") or {}
    social = (knowledge_graph or {}).get("social_relations") or {}
    network = (knowledge_graph or {}).get("network") or {}
    mem_agents = int(agent_memory.get("agent_count_with_memory") or 0)
    lt_agents = int(agent_memory.get("agent_count_with_long_term") or 0)
    lt_sourced = int(agent_memory.get("long_term_sourced_entries") or 0)
    bond_count = int(relationships.get("bond_count") or 0)
    mean_strength = float(relationships.get("mean_strength") or 0.0)
    change_count = int(agent_positions.get("change_count") or 0)
    rounds_tracked = int(opinion.get("rounds_tracked") or 0)
    shift_rate = float(opinion.get("shift_rate_per_agent_round") or 0.0)

    engagement = {
        "posts": len(posts),
        "comments": len(comments),
        "reactions": len(reactions),
        "posts_per_agent": round(len(posts) / max(1, agent_count), 3),
        "comments_per_agent": round(len(comments) / max(1, agent_count), 3),
        "reactions_per_agent": round(len(reactions) / max(1, agent_count), 3),
    }

    return {
        "phase": 7,
        "backend": "local",
        "execution_kind": execution_kind,
        "scale": {
            "agent_count": agent_count,
            "step": step,
            "max_steps": max_steps,
            "offline_presets": list(OFFLINE_SCALE_PRESETS),
            "offline_max_agents": OFFLINE_MAX_AGENTS,
            "offline_max_rounds": OFFLINE_MAX_ROUNDS,
            "live_llm_max_agents": LIVE_TEST_MAX_AGENTS,
            "live_llm_max_rounds": LIVE_TEST_MAX_ROUNDS,
            "live_llm_max_calls": LIVE_TEST_MAX_LLM_CALLS,
            "paid_llm_unlocked": False,
        },
        "memory": {
            "agents_with_working": mem_agents,
            "agents_with_long_term": lt_agents,
            "long_term_sourced_entries": lt_sourced,
            "max_working": agent_memory.get("max_entries_per_agent"),
            "max_long_term": agent_memory.get("max_long_term_per_agent"),
            "zep_integrated": False,
        },
        "relationships": {
            "bond_count": bond_count,
            "mean_strength": mean_strength,
            "by_relation": relationships.get("by_relation") or {},
            "evidenced_only": True,
            "chronology_events": len(relationships.get("chronology") or []),
        },
        "opinion": {
            "change_count": change_count,
            "polarity_distribution": pol,
            "rounds_tracked": rounds_tracked,
            "shift_rate_per_agent_round": shift_rate,
            "transitions": opinion.get("transitions") or {},
            "chronology_len": len(opinion.get("chronology") or []),
        },
        "engagement": engagement,
        "social_relations": social,
        "network": network,
        "interaction_summary": {
            k: interaction_summary.get(k)
            for k in (
                "agrees_with",
                "disagrees_with",
                "responds_to",
                "influenced_by",
                "position_changed",
                "comments",
                "post_reactions",
                "comment_reactions",
            )
        },
        "scheduler": {
            "plans": (scheduler or {}).get("plans", 0),
            "self_as_partner_count": (scheduler or {}).get("self_as_partner_count", 0),
        },
        "kpis": {
            "memory_coverage": round(lt_agents / max(1, agent_count), 4),
            "relationship_density": round(
                bond_count / max(1, agent_count * max(1, agent_count - 1)), 4
            ),
            "opinion_volatility": shift_rate,
            "social_edge_total": sum(int(v or 0) for v in social.values()),
        },
        "note": (
            "Society Analytics (Phase 7). Offline 10/25/50 × ≤4 Runden. "
            "Live-Ceilings unverändert; paid_llm_unlocked=false; kein Zep."
        ),
    }
