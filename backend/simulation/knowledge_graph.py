"""Local knowledge graph: actors, topics, statements, relationships.

Built offline from simulation state (agents + posts + comments + reactions).
No external graph DB — dashboard-ready JSON only.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.simulation.agents import AgentProfile
from backend.simulation.agent_generator import extract_scenario_keywords


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
            self.upsert_node(
                nid,
                "actor",
                name,
                agent_id=aid,
                user_name=user_name,
                goal=goal,
                style=style,
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
            uid = c.get("user_id")
            if uid is not None and int(uid) in agent_nodes:
                self.add_edge(
                    agent_nodes[int(uid)],
                    sid,
                    "commented",
                    comment_id=cid,
                )
            parent_pid = c.get("post_id")
            if parent_pid is not None and int(parent_pid) in post_statement:
                self.add_edge(
                    sid,
                    post_statement[int(parent_pid)],
                    "replies_to",
                    comment_id=cid,
                )
            self.add_edge(sid, scenario_node.id, "about")

        for r in reactions:
            rtype = str(r.get("type") or "like").lower()
            uid = r.get("user_id")
            pid = r.get("post_id")
            if uid is None or pid is None:
                continue
            if int(uid) not in agent_nodes:
                continue
            target = post_statement.get(int(pid))
            if not target:
                # Still record actor→topic signal even if post missing.
                target = scenario_node.id
            relation = "liked" if "dislike" not in rtype else "disliked"
            self.add_edge(
                agent_nodes[int(uid)],
                target,
                relation,
                reaction=rtype,
                post_id=pid,
            )

        # Optional: follow edges from action log
        for entry in action_log or []:
            if entry.get("action_type") != "follow":
                continue
            aid = entry.get("agent_id")
            args = entry.get("action_args") or {}
            if not isinstance(args, dict):
                continue
            followee = args.get("followee_id")
            if aid is None or followee is None:
                continue
            if int(aid) in agent_nodes and int(followee) in agent_nodes:
                self.add_edge(
                    agent_nodes[int(aid)],
                    agent_nodes[int(followee)],
                    "follows",
                )

    def to_dict(self) -> dict[str, Any]:
        nodes = [n.to_dict() for n in self.nodes.values()]
        edges = [e.to_dict() for e in self.edges.values()]
        by_kind: dict[str, int] = {}
        for n in nodes:
            by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
        by_rel: dict[str, int] = {}
        for e in edges:
            by_rel[e["relation"]] = by_rel.get(e["relation"], 0) + 1
        return {
            "backend": "local",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "counts_by_kind": by_kind,
            "counts_by_relation": by_rel,
            "nodes": nodes,
            "edges": edges,
            "note": (
                "Lokaler Wissensgraph (Akteure / Themen / Aussagen / Beziehungen). "
                "Kein externes Graph-Backend."
            ),
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")[:40] or "x"


def _short(text: str, limit: int = 160) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"
