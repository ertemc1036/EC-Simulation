"""Phase 7 — Dynamic agent↔agent relationships (evidenced-only).

Strength and chronology grow only from proven social edges / interactions.
No invented ties. Self-pairs are discarded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from backend.simulation.social_dynamics import SOCIAL_RELATIONS, extract_local_comment_refs


# Actor↔actor relations that may form a dynamic bond.
_PAIR_RELATIONS = frozenset(
    {
        "agrees_with",
        "disagrees_with",
        "responds_to",
        "influenced_by",
    }
)

_STRENGTH_CAP = 1.0
_EVIDENCE_PER_TICK = 0.15
_MAX_EVIDENCES_PER_PAIR = 12
_MAX_CHRONOLOGY = 200


@dataclass
class RelationshipBond:
    source_agent_id: int
    target_agent_id: int
    relations: list[str] = field(default_factory=list)
    evidence_count: int = 0
    strength: float = 0.0
    first_step: int | None = None
    last_step: int | None = None
    evidences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RelationshipDynamics:
    """Bounded, evidenced-only relationship graph over rounds."""

    def __init__(self, *, max_bonds: int = 500) -> None:
        self.max_bonds = max(1, int(max_bonds))
        self._bonds: dict[tuple[int, int], RelationshipBond] = {}
        self._chronology: list[dict[str, Any]] = []
        self._chronology_steps: set[int] = set()

    def clear(self) -> None:
        self._bonds.clear()
        self._chronology.clear()
        self._chronology_steps.clear()

    def _reset_bonds(self) -> None:
        self._bonds.clear()

    @staticmethod
    def _pair_key(a: int, b: int) -> tuple[int, int] | None:
        if a == b:
            return None
        # Directed: source observes/acts toward target.
        return (int(a), int(b))

    def _touch(
        self,
        *,
        source: int,
        target: int,
        relation: str,
        step: int | None,
        evidence: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if relation not in _PAIR_RELATIONS and relation not in SOCIAL_RELATIONS:
            return
        if relation not in _PAIR_RELATIONS:
            return
        key = self._pair_key(source, target)
        if key is None:
            return
        bond = self._bonds.get(key)
        if bond is None:
            if len(self._bonds) >= self.max_bonds:
                # Drop weakest / oldest when at capacity.
                weakest = min(
                    self._bonds.items(),
                    key=lambda kv: (kv[1].strength, kv[1].evidence_count),
                )[0]
                del self._bonds[weakest]
            bond = RelationshipBond(
                source_agent_id=key[0],
                target_agent_id=key[1],
                first_step=step,
            )
            self._bonds[key] = bond

        if relation not in bond.relations:
            bond.relations.append(relation)
        bond.evidence_count += 1
        bond.strength = min(
            _STRENGTH_CAP,
            round(bond.strength + _EVIDENCE_PER_TICK, 4),
        )
        if bond.first_step is None:
            bond.first_step = step
        bond.last_step = step
        evid = {
            "relation": relation,
            "step": step,
            "evidence": evidence,
            **(meta or {}),
        }
        bond.evidences.append(evid)
        overflow = len(bond.evidences) - _MAX_EVIDENCES_PER_PAIR
        if overflow > 0:
            del bond.evidences[:overflow]

    def ingest_platform(
        self,
        *,
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        reactions: list[dict[str, Any]],
        social_edge_specs: list[dict[str, Any]] | None = None,
        step: int | None = None,
    ) -> None:
        """Rebuild bonds from full platform evidence (idempotent per sync)."""
        # Full rebuild avoids double-counting across repeated syncs.
        self._reset_bonds()
        author_of_post = {
            int(p["post_id"]): int(p["user_id"])
            for p in posts
            if p.get("post_id") is not None and p.get("user_id") is not None
        }
        author_of_comment = {
            int(c["comment_id"]): int(c["user_id"])
            for c in comments
            if c.get("comment_id") is not None and c.get("user_id") is not None
        }

        for r in reactions:
            uid = r.get("user_id")
            if uid is None:
                continue
            aid = int(uid)
            target_author: int | None = None
            evid = "reaction"
            if r.get("comment_id") is not None:
                cid = int(r["comment_id"])
                target_author = author_of_comment.get(cid)
                evid = f"reaction:comment:{cid}"
            elif r.get("post_id") is not None:
                pid = int(r["post_id"])
                target_author = author_of_post.get(pid)
                evid = f"reaction:post:{pid}"
            if target_author is None or target_author == aid:
                continue
            rtype = str(r.get("type") or r.get("reaction_type") or "").lower()
            relation = "agrees_with"
            if "dislike" in rtype or rtype in {"down", "dislike", "-1"}:
                relation = "disagrees_with"
            self._touch(
                source=aid,
                target=target_author,
                relation=relation,
                step=step,
                evidence=evid,
            )

        for c in comments:
            uid = c.get("user_id")
            cid = c.get("comment_id")
            if uid is None or cid is None:
                continue
            aid = int(uid)
            content = str(c.get("content") or "")
            # Flat reply under a post → responds_to post author.
            pid = c.get("post_id")
            if pid is not None:
                post_author = author_of_post.get(int(pid))
                if post_author is not None and post_author != aid:
                    self._touch(
                        source=aid,
                        target=post_author,
                        relation="responds_to",
                        step=step,
                        evidence=f"create_comment:{cid}->post:{pid}",
                    )
            for ref in extract_local_comment_refs(content):
                target_author = author_of_comment.get(ref)
                if target_author is None or target_author == aid:
                    continue
                self._touch(
                    source=aid,
                    target=target_author,
                    relation="responds_to",
                    step=step,
                    evidence=f"local_ref:comment:{cid}->comment:{ref}",
                )

        for spec in social_edge_specs or []:
            rel = str(spec.get("relation") or "")
            if rel not in _PAIR_RELATIONS:
                continue
            src = str(spec.get("source") or "")
            tgt = str(spec.get("target") or "")
            if not src.startswith("actor:") or not tgt.startswith("actor:"):
                continue
            try:
                source = int(src.split(":", 1)[1])
                target = int(tgt.split(":", 1)[1])
            except (ValueError, IndexError):
                continue
            meta = dict(spec.get("meta") or {})
            evid = str(meta.get("evidence") or f"graph:{rel}")
            self._touch(
                source=source,
                target=target,
                relation=rel,
                step=step if step is not None else meta.get("step"),
                evidence=evid,
                meta={"from_graph": True},
            )

        # One chronology frame per step (society-level), not per edge touch.
        if step is not None and int(step) not in self._chronology_steps:
            self._chronology_steps.add(int(step))
            bonds = list(self._bonds.values())
            self._chronology.append(
                {
                    "step": int(step),
                    "bond_count": len(bonds),
                    "mean_strength": (
                        round(sum(b.strength for b in bonds) / len(bonds), 4)
                        if bonds
                        else 0.0
                    ),
                    "evidence_total": sum(b.evidence_count for b in bonds),
                }
            )
            chron_overflow = len(self._chronology) - _MAX_CHRONOLOGY
            if chron_overflow > 0:
                dropped = self._chronology[:chron_overflow]
                del self._chronology[:chron_overflow]
                for d in dropped:
                    self._chronology_steps.discard(int(d.get("step") or -1))

    def snapshot(self) -> dict[str, Any]:
        bonds = sorted(
            self._bonds.values(),
            key=lambda b: (-b.strength, -b.evidence_count, b.source_agent_id),
        )
        by_relation: dict[str, int] = {r: 0 for r in sorted(_PAIR_RELATIONS)}
        for b in bonds:
            for r in b.relations:
                if r in by_relation:
                    by_relation[r] += 1
        return {
            "backend": "local",
            "evidenced_only": True,
            "bond_count": len(bonds),
            "max_bonds": self.max_bonds,
            "by_relation": by_relation,
            "mean_strength": (
                round(sum(b.strength for b in bonds) / len(bonds), 4) if bonds else 0.0
            ),
            "bonds": [b.to_dict() for b in bonds[:80]],
            "chronology": list(self._chronology[-80:]),
            "note": (
                "Dynamische Beziehungen nur aus belegten Interaktionen "
                "(Reaktionen, →Kommentar#, influenced_by). Keine erfundenen Kanten. "
                "Zep nicht involviert."
            ),
        }
