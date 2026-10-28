"""Local knowledge graph: actors, topics, statements, relationships.

Built offline from simulation state (agents + posts + comments + reactions).
No external graph DB — dashboard-ready JSON only.

Phase 5 social relations (only when evidenced):
- ``agrees_with`` / ``disagrees_with`` — like_* / dislike_*
- ``responds_to`` — create_comment on post + local Kommentar# refs
  (OASIS has no parent_comment_id)
- ``influenced_by`` / ``position_changed`` — from PositionTracker changes
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.simulation.agents import AgentProfile
from backend.simulation.agent_generator import extract_scenario_keywords
from backend.simulation.social_dynamics import (
    SOCIAL_RELATIONS,
    collect_local_responds_to,
)

# Phase-5 social + legacy replied_to (create_comment → post) for dashboards.
INTERACTION_RELATIONS = frozenset(
    {"replied_to", *SOCIAL_RELATIONS}
)


@dataclass
class GraphNode:
    id: str
    kind: str  # actor | topic | statement
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    relation: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalKnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()

    def upsert_node(
        self, node_id: str, kind: str, label: str, **meta: Any
    ) -> GraphNode:
        existing = self.nodes.get(node_id)
        if existing:
            existing.label = label or existing.label
            existing.meta.update(meta)
            return existing
        node = GraphNode(id=node_id, kind=kind, label=label, meta=dict(meta))
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        edge_id: str | None = None,
        **meta: Any,
    ) -> GraphEdge:
        eid = edge_id or f"{relation}:{source}->{target}"
        if eid in self.edges:
            self.edges[eid].meta.update(meta)
            return self.edges[eid]
        edge = GraphEdge(
            id=eid, source=source, target=target, relation=relation, meta=dict(meta)
        )
        self.edges[eid] = edge
        return edge

    def rebuild(
        self,
        *,
        scenario: str,
        agents: list[AgentProfile] | list[dict[str, Any]],
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        reactions: list[dict[str, Any]],
        action_log: list[dict[str, Any]] | None = None,
        social_edge_specs: list[dict[str, Any]] | None = None,
        agent_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Replace graph contents from the latest simulation snapshot."""
        self.clear()
        keywords = extract_scenario_keywords(scenario, limit=6)
        scenario_node = self.upsert_node(
            "topic:scenario",
            "topic",
            (scenario or "Szenario")[:120],
            keywords=keywords,
        )
        for kw in keywords:
            tid = f"topic:{_slug(kw)}"
            self.upsert_node(tid, "topic", kw, from_scenario=True)
            self.add_edge(scenario_node.id, tid, "includes_topic")

        agent_nodes: dict[int, str] = {}
        for raw in agents:
            if isinstance(raw, AgentProfile):
                aid = raw.agent_id
                name = raw.name
                goal = raw.goal
                style = raw.style
                user_name = raw.user_name
            else:
                aid = int(raw.get("agent_id", 0))
                name = str(raw.get("name") or f"Agent {aid}")
                goal = str(raw.get("goal") or "")
                style = str(raw.get("style") or "")
                user_name = str(raw.get("user_name") or "")
            nid = f"actor:{aid}"
            pos_meta: dict[str, Any] = {}
            for p in agent_positions or []:
                if int(p.get("agent_id", -1)) == aid:
                    pos_meta = {
                        "stance": p.get("stance"),
                        "polarity": p.get("polarity"),
                        "stance_key": p.get("stance_key"),
                        "position_changed": bool(p.get("changed")),
                    }
                    break
            self.upsert_node(
                nid,
                "actor",
                name,
                agent_id=aid,
                user_name=user_name,
                goal=goal,
                style=style,
                **pos_meta,
            )
            agent_nodes[aid] = nid
            self.add_edge(nid, scenario_node.id, "discusses")

        post_statement: dict[int, str] = {}
        for p in posts:
            pid = p.get("post_id")
            if pid is None:
                continue
            sid = f"statement:post:{pid}"
            content = str(p.get("content") or "")
            self.upsert_node(
                sid,
                "statement",
                _short(content),
                post_id=pid,
                user_id=p.get("user_id"),
                statement_kind="post",
            )
            post_statement[int(pid)] = sid
            uid = p.get("user_id")
            if uid is not None and int(uid) in agent_nodes:
                self.add_edge(
                    agent_nodes[int(uid)],
                    sid,
                    "posted",
                    post_id=pid,
                )
            self.add_edge(sid, scenario_node.id, "about")

        comment_statement: dict[int, str] = {}
        for c in comments:
            cid = c.get("comment_id")
            if cid is None:
                continue
            sid = f"statement:comment:{cid}"
            content = str(c.get("content") or "")
            self.upsert_node(
                sid,
                "statement",
                _short(content),
                comment_id=cid,
                post_id=c.get("post_id"),
                user_id=c.get("user_id"),
                statement_kind="comment",
            )
            comment_statement[int(cid)] = sid
            uid = c.get("user_id")
            if uid is not None and int(uid) in agent_nodes:
                self.add_edge(
                    agent_nodes[int(uid)],
                    sid,
                    "commented",
                    comment_id=cid,
                )
                # Legacy Phase-4: evidenced reply (OASIS = comment under post_id).
                parent_pid = c.get("post_id")
                if parent_pid is not None and int(parent_pid) in post_statement:
                    self.add_edge(
                        agent_nodes[int(uid)],
                        post_statement[int(parent_pid)],
                        "replied_to",
                        comment_id=cid,
                        post_id=parent_pid,
                        evidence="create_comment",
                    )
            parent_pid = c.get("post_id")
            if parent_pid is not None and int(parent_pid) in post_statement:
                # Structural statement→statement link (flat OASIS thread).
                self.add_edge(
                    sid,
                    post_statement[int(parent_pid)],
                    "replies_to",
                    comment_id=cid,
                )
            self.add_edge(sid, scenario_node.id, "about")

        # Phase 5: local comment refs + flat post replies → responds_to.
        for spec in collect_local_responds_to(
            comments,
            comment_statement_ids=comment_statement,
            post_statement_ids=post_statement,
            agent_nodes=agent_nodes,
        ):
            self.add_edge(
                spec["source"],
                spec["target"],
                spec["relation"],
                **(spec.get("meta") or {}),
            )

        for r in reactions:
            rtype = str(r.get("type") or "like").lower()
            uid = r.get("user_id")
            if uid is None or int(uid) not in agent_nodes:
                continue
            target: str | None = None
            evidence = rtype
            meta: dict[str, Any] = {"reaction": rtype}
            pid = r.get("post_id")
            cid = r.get("comment_id")
            if pid is not None and int(pid) in post_statement:
                target = post_statement[int(pid)]
                meta["post_id"] = pid
                evidence = f"{rtype}_post"
            elif cid is not None and int(cid) in comment_statement:
                target = comment_statement[int(cid)]
                meta["comment_id"] = cid
                evidence = f"{rtype}_comment"
            if not target:
                continue
            if "dislike" in rtype:
                relation = "disagrees_with"
            else:
                relation = "agrees_with"
            meta["evidence"] = evidence
            self.add_edge(
                agent_nodes[int(uid)],
                target,
                relation,
                **meta,
            )

        # Optional: follow edges + interaction edges from action_log when DB
        # reactions lagged (still evidence-gated by concrete tool args).
        for entry in action_log or []:
            action = str(entry.get("action_type") or "")
            aid = entry.get("agent_id")
            args = entry.get("action_args") or {}
            if not isinstance(args, dict) or aid is None:
                continue
            if int(aid) not in agent_nodes:
                continue
            if action == "follow":
                followee = args.get("followee_id")
                if followee is not None and int(followee) in agent_nodes:
                    self.add_edge(
                        agent_nodes[int(aid)],
                        agent_nodes[int(followee)],
                        "follows",
                    )
                continue
            # Only add interaction edges if not already present from DB rows.
            if action == "create_comment":
                pid = args.get("post_id")
                if pid is not None and int(pid) in post_statement:
                    eid = (
                        f"replied_to:{agent_nodes[int(aid)]}->"
                        f"{post_statement[int(pid)]}"
                    )
                    if eid not in self.edges:
                        self.add_edge(
                            agent_nodes[int(aid)],
                            post_statement[int(pid)],
                            "replied_to",
                            edge_id=eid,
                            post_id=pid,
                            evidence="action_log:create_comment",
                        )
                    # Phase 5 also maps create_comment → responds_to.
                    reid = (
                        f"responds_to:{agent_nodes[int(aid)]}->"
                        f"{post_statement[int(pid)]}"
                    )
                    if reid not in self.edges:
                        self.add_edge(
                            agent_nodes[int(aid)],
                            post_statement[int(pid)],
                            "responds_to",
                            edge_id=reid,
                            post_id=pid,
                            evidence="action_log:create_comment",
                        )
            elif action in {"like_post", "dislike_post"}:
                pid = args.get("post_id")
                if pid is None or int(pid) not in post_statement:
                    continue
                relation = (
                    "agrees_with" if action == "like_post" else "disagrees_with"
                )
                eid = f"{relation}:{agent_nodes[int(aid)]}->{post_statement[int(pid)]}"
                if eid not in self.edges:
                    self.add_edge(
                        agent_nodes[int(aid)],
                        post_statement[int(pid)],
                        relation,
                        edge_id=eid,
                        post_id=pid,
                        evidence=f"action_log:{action}",
                    )
            elif action in {"like_comment", "dislike_comment"}:
                cid = args.get("comment_id")
                if cid is None or int(cid) not in comment_statement:
                    continue
                relation = (
                    "agrees_with"
                    if action == "like_comment"
                    else "disagrees_with"
                )
                eid = (
                    f"{relation}:{agent_nodes[int(aid)]}->"
                    f"{comment_statement[int(cid)]}"
                )
                if eid not in self.edges:
                    self.add_edge(
                        agent_nodes[int(aid)],
                        comment_statement[int(cid)],
                        relation,
                        edge_id=eid,
                        comment_id=cid,
                        evidence=f"action_log:{action}",
                    )

        # Phase 5: position_changed / influenced_by from PositionTracker.
        for spec in social_edge_specs or []:
            rel = str(spec.get("relation") or "")
            src = str(spec.get("source") or "")
            tgt = str(spec.get("target") or "")
            if rel not in SOCIAL_RELATIONS or not src or not tgt:
                continue
            if src not in self.nodes or tgt not in self.nodes:
                continue
            meta = dict(spec.get("meta") or {})
            self.add_edge(src, tgt, rel, **meta)

    def interaction_counts(self) -> dict[str, int]:
        counts = {r: 0 for r in sorted(INTERACTION_RELATIONS)}
        for e in self.edges.values():
            if e.relation in counts:
                counts[e.relation] += 1
        return counts

    def social_relation_counts(self) -> dict[str, int]:
        counts = {r: 0 for r in sorted(SOCIAL_RELATIONS)}
        for e in self.edges.values():
            if e.relation in counts:
                counts[e.relation] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        nodes = [n.to_dict() for n in self.nodes.values()]
        edges = [e.to_dict() for e in self.edges.values()]
        by_kind: dict[str, int] = {}
        for n in nodes:
            by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
        by_rel: dict[str, int] = {}
        for e in edges:
            by_rel[e["relation"]] = by_rel.get(e["relation"], 0) + 1
        interaction = self.interaction_counts()
        social = self.social_relation_counts()
        return {
            "backend": "local",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "counts_by_kind": by_kind,
            "counts_by_relation": by_rel,
            "interaction_relations": interaction,
            "social_relations": social,
            "nodes": nodes,
            "edges": edges,
            "note": (
                "Lokaler Wissensgraph (Akteure / Themen / Aussagen). "
                "Phase-5 Social-Kanten agrees_with / disagrees_with / "
                "responds_to / influenced_by / position_changed nur belegt. "
                "OASIS: kein parent_comment_id — lokale Kommentar-Refs."
            ),
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")[:40] or "x"


def _short(text: str, limit: int = 160) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"
