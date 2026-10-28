"""Phase 7.3 + 8.2 — Intelligent Discussion / Debate Engine V2 (offline-safe).

Tracks argument fingerprints, bounded per-agent history, logical thread
metadata (parent/author/recipient/round), and anti-repetition hints for
prompts / scripted content. OASIS still has no ``parent_comment_id`` —
nested structure is reconstructed from local ``→Kommentar#`` refs only.

Phase 8.2 Debate Engine V2: Argument Intelligence + debate quality score
in snapshots (no provider calls).

Phase 8.3: discussion snapshots may attach Argument-Grounded Interaction V3
and Interaction Quality Evaluation (wired by the engine).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.simulation.social_dynamics import extract_local_comment_refs

# Hard caps — keep prompt/token budgets stable (Phase 4.1 ceilings).
MAX_ARGUMENT_HISTORY_PER_AGENT = 6
MAX_GLOBAL_ARGUMENTS = 48
MAX_THREAD_EVENTS = 80
ARGUMENT_FP_CHARS = 160
PROMPT_HINT_CHARS = 220
QUALITY_FIX = "8.3"
DEBATE_ENGINE = "v2"

_WORD_RE = re.compile(r"[a-zäöüß0-9]{4,}", re.IGNORECASE)
_SOURCE_CLAIM_RE = re.compile(
    r"(?i)\b(?:laut\s+studie|einer\s+studie\s+zufolge|studie\s+zeigt|"
    r"laut\s+\[\d+\]|http[s]?://|doi:\s*\S+)\b"
)
_STOP = frozenset(
    {
        "dass",
        "aber",
        "oder",
        "auch",
        "eine",
        "einem",
        "einen",
        "einer",
        "noch",
        "wenn",
        "dann",
        "wird",
        "werden",
        "sind",
        "haben",
        "diese",
        "dieser",
        "dieses",
        "nicht",
        "ohne",
        "über",
        "unter",
        "nach",
        "vor",
        "bei",
        "mit",
        "für",
        "aus",
        "zum",
        "zur",
        "vom",
        "ins",
        "dem",
        "den",
        "des",
        "der",
        "die",
        "das",
        "und",
        "als",
        "wie",
        "man",
        "sich",
        "sehr",
        "mehr",
        "kann",
        "muss",
        "soll",
        "nur",
        "schon",
        "hier",
        "dort",
        "also",
        "weil",
        "doch",
        "comment",
        "kommentar",
        "runde",
        "partner",
        "synthetisch",
        "seed",
    }
)


def argument_fingerprint(text: str) -> str:
    """Stable coarse fingerprint — content words, order-insensitive."""
    low = (text or "").lower()
    # Drop local reply prefix noise.
    low = re.sub(r"→\s*kommentar#\d+", " ", low)
    low = re.sub(r"\[synthetisch\]", " ", low)
    low = re.sub(r"\[[^\]]{0,40}\]", " ", low)
    tokens = [
        t
        for t in _WORD_RE.findall(low)
        if t.lower() not in _STOP and not t.isdigit()
    ]
    # Keep most frequent distinctive stems (first occurrence order preserved).
    uniq: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        key = t[:8].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
        if len(uniq) >= 12:
            break
    blob = " ".join(sorted(uniq))[:ARGUMENT_FP_CHARS]
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]
    return f"{digest}:{blob[:48]}"


def claims_invented_source(text: str) -> bool:
    """Heuristic: explicit study/URL claims without local evidence markers."""
    return bool(_SOURCE_CLAIM_RE.search(text or ""))


@dataclass
class ArgumentRecord:
    agent_id: int
    fingerprint: str
    snippet: str
    step: int | None
    evidence: str  # comment:N | post:N
    round_n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThreadEvent:
    comment_id: int
    post_id: int | None
    author_id: int | None
    parent_comment_id: int | None  # local reconstruction only
    recipient_id: int | None
    round_n: int | None
    self_reply: bool
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscussionEngine:
    """Bounded argument/thread tracker for one simulation run."""

    max_history_per_agent: int = MAX_ARGUMENT_HISTORY_PER_AGENT
    max_global: int = MAX_GLOBAL_ARGUMENTS
    _by_agent: dict[int, list[ArgumentRecord]] = field(default_factory=dict)
    _global: list[ArgumentRecord] = field(default_factory=list)
    _fps_global: set[str] = field(default_factory=set)
    _threads: list[ThreadEvent] = field(default_factory=list)
    _repetition_blocks: int = 0
    _addressed: dict[int, list[str]] = field(default_factory=dict)
    # Evidence keys already ingested (comment:N / post:N) — re-sync must not
    # inflate repetition_blocks (live fixture bdfb9e9d: 9 false blocks).
    _seen_evidence: set[str] = field(default_factory=set)

    def clear(self) -> None:
        self._by_agent.clear()
        self._global.clear()
        self._fps_global.clear()
        self._threads.clear()
        self._repetition_blocks = 0
        self._addressed.clear()
        self._seen_evidence.clear()

    def is_repetition(self, agent_id: int, text: str) -> bool:
        fp = argument_fingerprint(text)
        own = self._by_agent.get(int(agent_id), [])
        if any(r.fingerprint == fp for r in own):
            return True
        # Near-global duplicate of own prior theme is also repetition.
        return False

    def register_statement(
        self,
        *,
        agent_id: int,
        text: str,
        evidence: str,
        step: int | None = None,
        round_n: int | None = None,
    ) -> ArgumentRecord | None:
        text = (text or "").strip()
        if not text:
            return None
        ev = (evidence or "").strip()
        # Re-ingest of the same platform row is not a repetition block.
        if ev and ev in self._seen_evidence:
            return None
        fp = argument_fingerprint(text)
        aid = int(agent_id)
        if any(r.fingerprint == fp for r in self._by_agent.get(aid, [])):
            self._repetition_blocks += 1
            if ev:
                self._seen_evidence.add(ev)
            return None
        rec = ArgumentRecord(
            agent_id=aid,
            fingerprint=fp,
            snippet=" ".join(text.split())[:140],
            step=step,
            evidence=evidence,
            round_n=round_n,
        )
        if ev:
            self._seen_evidence.add(ev)
        bucket = self._by_agent.setdefault(aid, [])
        bucket.append(rec)
        overflow = len(bucket) - self.max_history_per_agent
        if overflow > 0:
            del bucket[:overflow]
        if fp not in self._fps_global:
            self._fps_global.add(fp)
            self._global.append(rec)
            g_overflow = len(self._global) - self.max_global
            if g_overflow > 0:
                dropped = self._global[:g_overflow]
                del self._global[:g_overflow]
                for d in dropped:
                    self._fps_global.discard(d.fingerprint)
        return rec

    def mark_addressed(self, agent_id: int, fingerprint: str) -> None:
        aid = int(agent_id)
        lst = self._addressed.setdefault(aid, [])
        if fingerprint not in lst:
            lst.append(fingerprint)
            if len(lst) > self.max_history_per_agent:
                del lst[: len(lst) - self.max_history_per_agent]

    def foreign_arguments_to_address(
        self, agent_id: int, *, limit: int = 3
    ) -> list[ArgumentRecord]:
        """Recent foreign arguments not yet addressed by this agent."""
        aid = int(agent_id)
        done = set(self._addressed.get(aid, []))
        own_fps = {r.fingerprint for r in self._by_agent.get(aid, [])}
        out: list[ArgumentRecord] = []
        for rec in reversed(self._global):
            if rec.agent_id == aid:
                continue
            if rec.fingerprint in done or rec.fingerprint in own_fps:
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def prompt_argument_hints(self, agent_id: int) -> str:
        """Short decision-prompt block: avoid repeats; pick up foreign points."""
        aid = int(agent_id)
        lines: list[str] = []
        own = self._by_agent.get(aid, [])
        if own:
            recent = "; ".join(r.snippet[:60] for r in own[-2:])
            lines.append(f"Deine jüngsten Argumente (nicht wiederholen): {recent}")
        foreign = self.foreign_arguments_to_address(aid, limit=2)
        if foreign:
            bits = []
            for r in foreign:
                bits.append(f"u{r.agent_id}/{r.evidence}: {r.snippet[:70]}")
            lines.append("Greife auf: " + " | ".join(bits))
        lines.append(
            "Kein Self-Reply; keine erfundenen Studien/URLs; neues Argument oder "
            "begründete Positionsänderung."
        )
        blob = "\n".join(lines)
        return blob[:PROMPT_HINT_CHARS]

    def ingest_platform(
        self,
        *,
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        step: int | None = None,
        round_n: int | None = None,
    ) -> None:
        """Rebuild argument + thread state from platform snapshots (evidenced)."""
        comment_author: dict[int, int] = {}
        for c in comments:
            try:
                cid = int(c.get("comment_id"))
                uid = int(c.get("user_id"))
            except (TypeError, ValueError):
                continue
            comment_author[cid] = uid

        for p in posts:
            try:
                uid = int(p.get("user_id"))
                pid = int(p.get("post_id"))
            except (TypeError, ValueError):
                continue
            content = str(p.get("content") or "")
            if content.lstrip().startswith("[Seed"):
                continue
            self.register_statement(
                agent_id=uid,
                text=content,
                evidence=f"post:{pid}",
                step=step,
                round_n=round_n,
            )

        threads: list[ThreadEvent] = []
        for c in comments:
            try:
                cid = int(c.get("comment_id"))
                uid = int(c.get("user_id"))
            except (TypeError, ValueError):
                continue
            content = str(c.get("content") or "")
            pid = c.get("post_id")
            try:
                post_id = int(pid) if pid is not None else None
            except (TypeError, ValueError):
                post_id = None
            refs = extract_local_comment_refs(content)
            parent = refs[0] if refs else None
            recipient = comment_author.get(parent) if parent is not None else None
            self_reply = bool(
                parent is not None
                and recipient is not None
                and int(recipient) == int(uid)
            )
            self.register_statement(
                agent_id=uid,
                text=content,
                evidence=f"comment:{cid}",
                step=step,
                round_n=round_n,
            )
            if parent is not None:
                parent_fp = None
                # Mark addressing the parent's argument fingerprint if known.
                for rec in self._global:
                    if rec.evidence == f"comment:{parent}":
                        parent_fp = rec.fingerprint
                        break
                if parent_fp:
                    self.mark_addressed(uid, parent_fp)
            threads.append(
                ThreadEvent(
                    comment_id=cid,
                    post_id=post_id,
                    author_id=uid,
                    parent_comment_id=parent,
                    recipient_id=recipient,
                    round_n=round_n if round_n is not None else step,
                    self_reply=self_reply,
                    snippet=" ".join(content.split())[:120],
                )
            )
        # Keep newest thread events bounded.
        if len(threads) > MAX_THREAD_EVENTS:
            threads = threads[-MAX_THREAD_EVENTS:]
        self._threads = threads

    def vary_scripted_body(
        self,
        agent_id: int,
        body: str,
        *,
        call_index: int = 0,
    ) -> str:
        """Offline helper: append a small unique angle if body would repeat."""
        if not self.is_repetition(agent_id, body):
            return body
        self._repetition_blocks += 1
        angles = (
            "Zusätzlich: messbare Pilotindikatoren vor Vollausrollung.",
            "Ergänzend: Verteilungswirkung nach Betroffenheit prüfen.",
            "Ferner: Umsetzungskapazität und Betriebsabläufe klären.",
            "Außerdem: Wechselwirkungen mit bestehenden Regeln benennen.",
            "Zugleich: regionale Unterschiede nicht pauschal ignorieren.",
        )
        extra = angles[(int(agent_id) + int(call_index)) % len(angles)]
        return f"{body.rstrip()} {extra}"

    def snapshot(
        self,
        *,
        polarity_distribution: dict[str, int] | None = None,
        unknown_stance_share: float | None = None,
        agent_count: int = 0,
    ) -> dict[str, Any]:
        self_replies = sum(1 for t in self._threads if t.self_reply)
        cross = sum(
            1
            for t in self._threads
            if t.parent_comment_id is not None and not t.self_reply
        )
        args = [r.to_dict() for r in self._global[-24:]]
        threads = [t.to_dict() for t in self._threads[-40:]]
        # Lazy import avoids cycles with argument_intelligence → discussion.
        from backend.simulation.argument_intelligence import (
            build_argument_intelligence,
            debate_quality_score,
        )

        pol = polarity_distribution or {}
        if unknown_stance_share is None:
            total = sum(int(v) for v in pol.values()) or 0
            unk = int(pol.get("unknown") or 0) + int(pol.get("uncertain") or 0)
            unknown_stance_share = (unk / float(total)) if total else 0.0
        arg_intel = build_argument_intelligence(
            arguments=args,
            threads=threads,
            repetition_blocks=self._repetition_blocks,
            agent_count=agent_count or len(self._by_agent),
        )
        debate_q = debate_quality_score(
            argument_intel=arg_intel,
            unknown_stance_share=float(unknown_stance_share or 0.0),
            polarity_distribution=pol,
        )
        return {
            "backend": "local",
            "quality_fix": QUALITY_FIX,
            "debate_engine": DEBATE_ENGINE,
            "max_argument_history_per_agent": self.max_history_per_agent,
            "max_global_arguments": self.max_global,
            "argument_count": len(self._global),
            "agents_with_arguments": len(self._by_agent),
            "repetition_blocks": self._repetition_blocks,
            "thread_events": len(self._threads),
            "self_replies": self_replies,
            "cross_agent_replies": cross,
            "parent_comment_id_supported": False,
            "oasis_nested_reply": False,
            "arguments": arg_intel.get("arguments") or args,
            "by_agent": {
                str(aid): [r.to_dict() for r in recs]
                for aid, recs in sorted(self._by_agent.items())
            },
            "threads": threads,
            "argument_intelligence": arg_intel,
            "debate_quality": debate_q,
            "note": (
                "Debate Engine V2 (Phase 8.2/8.3): Argument-History + Claim-Intelligence "
                "+ Quality-Score. Threads aus lokalen →Kommentar#-Refs. "
                "OASIS: kein parent_comment_id. Keine erfundenen Quellen. "
                "Interaction Quality / Argument-Grounded V3 werden vom Engine angehängt."
            ),
        }
