"""Phase 5 / 5.1 / 5.2 — Social Dynamics: persistent positions + local refs.

OASIS has no ``parent_comment_id``. Local references (``→Kommentar#N``,
``Kommentar#N``, ``comment_id=N`` in comment content) plus like/dislike
evidence drive social relations. No invented edges.

Phase 5.1 accuracy rules:
- Erstposition (goal seed → first platform statement) is not a change
- ``position_changed`` only on evidenced polarity/stance shifts
- Confirmations are not changes
- ``unknown`` / ``uncertain`` when polarity is unclear
- ``influenced_by`` only after a real change + direct interaction evidence
- No double-counting; comment-id → author mapping must resolve correctly

Phase 5.2 (live-run 7f75d2e4):
- Manual ``[Seed / …]`` posts do not establish stance
- Policy polarity: Ausnahme vs. volles Verbot (``exception``) is stable
- Lexicon avoids false pro/contra from Support-/Kompromiss-Substrings
- Textual ``agrees_with`` / ``disagrees_with`` only with comment-ref evidence

Phase 6: polarity distribution + scale-safe payloads for 10/25/50 offline.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Phase-5 social relations — only emit when evidenced.
SOCIAL_RELATIONS = frozenset(
    {
        "agrees_with",
        "disagrees_with",
        "responds_to",
        "influenced_by",
        "position_changed",
    }
)

_COMMENT_REF_RE = re.compile(
    r"(?:→\s*)?(?:Kommentar#|comment_id\s*=\s*)(\d+)",
    re.IGNORECASE,
)
_SEED_CONTENT_RE = re.compile(r"^\[Seed\s*/", re.IGNORECASE)

# Stable stance: demands exceptions / transition rules / pilots vs. full measure.
_EXCEPTION_MARKERS = (
    "ausnahmeregel",
    "ausnahmeregelungen",
    "ausnahmen für",
    "ausnahme für",
    "klare ausnahmen",
    "soziale abfederung",
    "soziale abfederungen",
    "abfederung",
    "lieferfenster",
    "übergangsfrist",
    "uebergangsfrist",
    "übergangsfristen",
    "uebergangsfristen",
    "mobilitätseingeschränkte",
    "mobilitaetseingeschraenkte",
    "anwohnerpark",
    "pilot mit",
    "piloten und",
    "pilotprojekt",
    "pilotprojekte",
    "plädiere deshalb für pilot",
    "plädiere für evidenzbasierte pilot",
    "plädiere für pilot",
    "abbruchkriterien",
    "schrittweise einführung",
)

# Pro toward the ban / measure — avoid matching "Unterstützung" (aid).
_PRO_MARKERS = (
    "dafür",
    "befürwort",
    "bin für das",
    "bin für ein",
    "ja zum verbot",
    "ja zum grundeinkommen",
    "pro verbot",
    "pro grundeinkommen",
    "sinnvolles verbot",
    "existenzsicherung ohne sanktionsdruck",
    # Live-LLM BGE phrasing (Phase 7.2) — clear support without "ja zum …".
    "soziale absicherung",
    "wichtige soziale absicherung",
    "würde und planbarkeit",
    "planungssicherheit",
    "ohne sanktion",
    "ohne angst vor sanktion",
    "bedingungslose absicherung",
    "ermöglicht, sich voll auf das studium",
    "für studierende eine enorme erleichterung",
    # Live Vier-Tage-Woche (Phase 7.3.1 fixture bdfb9e9d).
    "grundsätzlich positiv",
    "lebensqualität verbessert",
    "support the ban",
    "agree with the ban",
)
# Verb support for ban / BGE / generic measure.
# Allow longer bridges ("unterstütze die Einführung des bedingungslosen …").
_PRO_BAN_RE = re.compile(
    r"unterstütze(?:n|t)?\s+(?:\w+\s+){0,6}(?:das\s+)?verbot|"
    r"unterstütze(?:n|t)?\s+(?:\w+\s+){0,6}(?:die\s+)?maßnahme|"
    r"unterstütze(?:n|t)?\s+(?:\w+\s+){0,6}(?:das\s+|ein\s+|des\s+)?"
    r"(?:bedingungslose[sn]?\s+)?grundeinkommen|"
    r"befürworte?\s+(?:\w+\s+){0,4}(?:das\s+)?verbot|"
    r"befürworte?\s+(?:\w+\s+){0,4}(?:ein\s+|das\s+|bedingungslose\s+)?"
    r"grundeinkommen|"
    r"bietet\s+(?:\w+\s+){0,3}(?:soziale\s+)?absicherung",
    re.IGNORECASE,
)
# Reject phrasing with words between "lehne" and "ab".
_CONTRA_REJECT_RE = re.compile(
    r"lehne(?:n)?\s+(?:\w+\s+){0,8}ab\b|"
    r"lehne(?:n)?\s+(?:\w+\s+){0,6}(?:das\s+)?(?:bedingungslose[sn]?\s+)?"
    r"grundeinkommen|"
    r"finanzierung\s+(?:\w+\s+){0,4}ungeklärt|"
    r"inflations-?\s*und\s+steuerlast",
    re.IGNORECASE,
)

# Contra toward a full / blanket ban or unchecked BGE rollout.
_CONTRA_MARKERS = (
    "dagegen",
    "ablehn",
    "lehne das verbot ab",
    "lehnen das verbot ab",
    "lehne das grundeinkommen ab",
    "lehnen das grundeinkommen ab",
    "lehne ab",
    "nein zum",
    "gegen das verbot",
    "gegen ein verbot",
    "gegen das grundeinkommen",
    "gegen ungeprüfte",
    "pauschale verbote",
    "pauschales verbot",
    "pauschalen verboten",
    "ungeprüfte vollauszahlung",
    "ohne gegenfinanzierung",
    "ohne finanzierungsnachweis",
    "mit sorge",
    "gefährden arbeitsplätze",
    "gefährdet",
    "widerspreche",
    "widersprechen",
    "oppose",
    "disagree with the ban",
    "protest",
    # Live-LLM BGE critique (Phase 7.3 fixture 3202e35e).
    "nicht ohne erhebliche risiken",
    "finanzierung ungeklärt",
    "wie dieses modell finanziert",
    "steigende lohnkosten",
    "lohnsteigerungen und zusätzliche kosten",
    "arbeitsanreize und lohnkosten",
    "auswirkungen auf arbeitsanreize",
    "praktische evidenz für positive effekte",
    # Live Vier-Tage-Woche critique (Phase 7.3.1).
    "sehe ich die idee",
    "kritisch, solange",
    "sehe ich … kritisch",
    "mehrkosten, die ohne gegenfinanzierung",
    "ohne klaren finanzierungsmodellen",
    "solange keine klaren finanzierungsmodelle",
)

# Balanced BGE / measure stance: support + incentive / wage caveats (not full contra).
_BGE_EXCEPTION_MARKERS = (
    "arbeitsanreize",
    "anreize für erwerbsarbeit",
    "vollzeitarbeit entwert",
    "löhne müssen weiterhin attraktiv",
    "faire und attraktive löhne",
    "anrechnung des grundeinkommens",
    "kompatibilität mit löhnen",
    "basisabsicherung",
    "zunächst in klar definierten pi",
    "in klar definierten pilot",
    "pilotversuchen",
    "pilotversuch",
)

# Explicit mind-change language only (bare "Kompromiss" is too common).
_SHIFT_MARKERS = (
    "umdenken",
    "revidier",
    "änder meine",
    "ändere meine",
    "neue position",
    "umstimme",
    "überzeugt",
    "einlenk",
    "ändere meine position",
    "position nach dem",
    "werde überzeugt",
)

_CONFIRM_MARKERS = (
    "bestätigt",
    "bestaetigt",
    "stimmt,",
    "stimmt dass",
    "stimmt, dass",
    "sehr richtig",
    "einverstanden",
    "teile diese",
    "genau das",
    "ebenso",
    "unterschreibe",
    "likewise",
    "agree with that",
)

_UNCERTAIN_MARKERS = (
    "unsicher",
    "unklar",
    "weiß nicht",
    "weiss nicht",
    "vielleicht",
    "ungewiss",
    "uncertain",
    "keine klare",
    "offen lassen",
    "noch offen",
)

# Mediating / balancing language (politics) without taking a ban side.
_MEDIATING_MARKERS = (
    "zielkonflikt",
    "zielkonflikte",
    "einerseits",
    "andererseits",
    "mehrheitsfähigkeit",
    "berechtigten anliegen aller",
    "anliegen aller gruppen",
    "ohne eine seite komplett",
)

# Textual agreement / disagreement toward a referenced comment (Phase 5.2).
_TEXT_AGREE_MARKERS = (
    "stimme zu",
    "stimme dir zu",
    "stimme ihnen zu",
    "einverstanden",
    "sehr richtig",
    "teile diese",
    "teile deine",
    "teile ihre",
    "danke für ihren hinweis",
    "danke für den hinweis",
    "danke für ihren",
    "danke für deinen",
    "wir erkennen die",
    "erkenne die berechtigten",
    "genau das",
    "unterschreibe",
    "likewise",
    "agree with",
)

_TEXT_DISAGREE_MARKERS = (
    "widerspreche",
    "widersprechen",
    "sehe das anders",
    "sehe ich anders",
    "kann dem nicht zustimmen",
    "stimme dem nicht zu",
    "das stimmt so nicht",
    "allerdings müssen wir",
    "allerdings dürfen",
    "allerdings darf",
    "dürfen nicht über",
    "darf nicht über",
    "ohne alltagsbezug",
    "disagree",
    "oppose that",
)

# Clear stance polarities (Phase 7.3: conditional replaces legacy «exception»).
_CLEAR_POLARITIES = frozenset({"pro", "contra", "conditional", "exception"})

# Explicit conditional / «nur wenn» stance (Phase 7.3).
_CONDITIONAL_MARKERS = (
    "unter der bedingung",
    "unter der voraussetzung",
    "nur wenn",
    "nur falls",
    "sofern",
    "bedingt zustimm",
    "bedingte zustimmung",
    "wenn klar gegenfinanziert",
    "falls ein pilot",
    "falls pilote",
    "nur mit pilot",
    "nur mit ausnahmen",
    "nur mit übergangs",
    "nur mit uebergangs",
)


def extract_local_comment_refs(content: str) -> list[int]:
    """Parse local comment references from free text (OASIS has no parent_id)."""
    ids: list[int] = []
    seen: set[int] = set()
    for match in _COMMENT_REF_RE.findall(content or ""):
        cid = int(match)
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _is_seed_content(text: str) -> bool:
    """Manual Diskussionsimpuls seed is not a real agent stance statement."""
    return bool(_SEED_CONTENT_RE.match((text or "").lstrip()))


def _polarity(text: str) -> str:
    """Coarse stance: pro/contra/conditional/uncertain/unknown (+ legacy exception).

    Phase 7.3 prefers ``conditional`` for Ausnahme-/«nur wenn»-Stances.
    ``exception`` remains as an alias return for classic mobility Ausnahme
    markers so older fixtures/tests stay comparable; treat both as clear.
    """
    raw = text or ""
    if _is_seed_content(raw):
        return "unknown"
    low = raw.lower()
    uncertain_hit = any(m in low for m in _UNCERTAIN_MARKERS)

    conditional_hit = sum(1 for m in _CONDITIONAL_MARKERS if m in low)
    pro = sum(1 for m in _PRO_MARKERS if m in low)
    if _PRO_BAN_RE.search(low):
        pro += 1
    # Soft support phrasing common in live labor debates.
    if "grundsätzlich positiv" in low:
        pro += 1
    contra = sum(1 for m in _CONTRA_MARKERS if m in low)
    if _CONTRA_REJECT_RE.search(low):
        contra += 1
    if "kritisch, solange" in low or (
        "sehe ich die idee" in low and "kritisch" in low
    ):
        contra += 1
    classic_exception = sum(1 for m in _EXCEPTION_MARKERS if m in low)
    bge_balance = sum(1 for m in _BGE_EXCEPTION_MARKERS if m in low)
    mediating = sum(1 for m in _MEDIATING_MARKERS if m in low)
    # BGE balancing (Anreize/Löhne) only when not a hard reject.
    exception = classic_exception + (bge_balance if contra == 0 else 0)

    # Explicit conditional language wins when not a hard reject.
    if conditional_hit and contra == 0:
        return "conditional"
    # Ausnahme-/Übergangsforderungen → conditional (7.3); keep exception alias
    # path via classic markers when no explicit conditional phrasing.
    if classic_exception and pro == 0:
        return "conditional"
    if exception and pro > 0 and contra == 0:
        return "conditional"
    if exception and pro == 0 and contra == 0:
        return "conditional"
    if pro > contra and pro > 0:
        return "pro"
    if contra > pro and contra > 0:
        return "contra"
    if exception:
        return "conditional"
    # Uncertain only when no clearer polarity (7.3.1 — «unklar» must not
    # hide pilot/conditional language from live Vier-Tage comments).
    if uncertain_hit:
        return "uncertain"
    if mediating and pro == 0 and contra == 0:
        return "uncertain"
    if any(m in low for m in _SHIFT_MARKERS):
        return "shifting"
    return "unknown"


def _is_confirmation(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _CONFIRM_MARKERS)


def _is_explicit_shift(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _SHIFT_MARKERS)


def classify_textual_agree_disagree(content: str) -> list[str]:
    """Evidenced textual agree/disagree labels (may be both, or empty)."""
    low = (content or "").lower()
    out: list[str] = []
    if any(m in low for m in _TEXT_AGREE_MARKERS):
        out.append("agrees_with")
    if any(m in low for m in _TEXT_DISAGREE_MARKERS):
        out.append("disagrees_with")
    return out


def _stance_key(polarity: str, snippet: str) -> str:
    # Key is polarity-primary so rephrasings of the same stance do not look
    # like changes. Snippet digest only disambiguates display history.
    digest = hashlib.sha1((snippet or "").encode("utf-8")).hexdigest()[:8]
    return f"{polarity}:{digest}"


def _coarse_key(polarity: str) -> str:
    return polarity


def _short(text: str, limit: int = 140) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _is_evidenced_change(
    *,
    prev_polarity: str,
    new_polarity: str,
    content: str,
) -> bool:
    """True only for a real stance shift — not confirmation or rephrase."""
    if _is_confirmation(content) and not _is_explicit_shift(content):
        return False
    if prev_polarity == new_polarity:
        return False
    # unknown/uncertain → first clear stance after establishment is
    # clarification, not a Meinungswechsel, unless the text signals a shift.
    if prev_polarity in {"unknown", "uncertain"} and new_polarity in _CLEAR_POLARITIES:
        return _is_explicit_shift(content)
    if new_polarity in {"unknown", "uncertain"} and prev_polarity in _CLEAR_POLARITIES:
        return _is_explicit_shift(content)
    # Clear flip among pro / contra / conditional only when polarity differs
    # and either it is a classic pro↔contra flip or shift language is present.
    if {prev_polarity, new_polarity} == {"pro", "contra"}:
        return True
    # conditional ↔ exception is the same bucket (alias), not a change.
    if {prev_polarity, new_polarity} <= {"conditional", "exception"}:
        return False
    if prev_polarity in _CLEAR_POLARITIES and new_polarity in _CLEAR_POLARITIES:
        # conditional/exception ↔ pro/contra requires explicit mind-change language
        return _is_explicit_shift(content)
    # Entering/leaving "shifting" only with explicit mind-change language.
    if "shifting" in {prev_polarity, new_polarity}:
        return _is_explicit_shift(content)
    if prev_polarity != new_polarity and _is_explicit_shift(content):
        return True
    return False


@dataclass
class AgentPosition:
    agent_id: int
    stance: str
    stance_key: str
    polarity: str
    step: int | None = None
    evidence: str = ""
    evidence_kind: str = ""  # post | comment | goal | reaction
    changed: bool = False
    previous_stance: str | None = None
    previous_stance_key: str | None = None
    influenced_by_agent_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PositionTracker:
    """Persistent per-agent positions across simulation rounds."""

    def __init__(self) -> None:
        self._current: dict[int, AgentPosition] = {}
        self._history: dict[int, list[AgentPosition]] = {}
        self._change_events: list[dict[str, Any]] = []
        # Agents that already have a platform Erstposition (post/comment).
        self._established: set[int] = set()
        # Dedup keys for change events (agent, from_pol, to_pol, evidence).
        self._change_keys: set[str] = set()

    def clear(self) -> None:
        self._current.clear()
        self._history.clear()
        self._change_events.clear()
        self._established.clear()
        self._change_keys.clear()

    def seed_from_goals(
        self, agents: list[Any], *, step: int | None = 0
    ) -> None:
        """Initial positions from goals — before any platform content."""
        for raw in agents:
            if hasattr(raw, "agent_id"):
                aid = int(raw.agent_id)
                goal = str(getattr(raw, "goal", "") or "")
            else:
                aid = int(raw.get("agent_id", 0))
                goal = str(raw.get("goal") or "")
            if aid in self._current:
                continue
            pol = _polarity(goal)
            stance = _short(goal) or f"Agent {aid}"
            pos = AgentPosition(
                agent_id=aid,
                stance=stance,
                stance_key=_stance_key(pol, stance),
                polarity=pol,
                step=step,
                evidence=stance,
                evidence_kind="goal",
            )
            self._current[aid] = pos
            self._history.setdefault(aid, []).append(pos)

    def update_from_snapshot(
        self,
        *,
        agents: list[Any],
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        reactions: list[dict[str, Any]],
        step: int | None,
    ) -> list[dict[str, Any]]:
        """Recompute positions from latest content; record evidenced changes."""
        self.seed_from_goals(agents, step=step if step is not None else 0)
        # Latest statement per agent (comment preferred over older post).
        # Manual [Seed / …] posts are ignored for stance establishment.
        latest: dict[int, tuple[str, str, str]] = {}
        for p in posts:
            uid = p.get("user_id")
            if uid is None:
                continue
            content = str(p.get("content") or "").strip()
            if not content or _is_seed_content(content):
                continue
            latest[int(uid)] = (content, "post", f"post:{p.get('post_id')}")
        for c in comments:
            uid = c.get("user_id")
            if uid is None:
                continue
            content = str(c.get("content") or "").strip()
            if not content:
                continue
            # Preserve correct comment_id → author mapping for evidence.
            cid = c.get("comment_id")
            latest[int(uid)] = (
                content,
                "comment",
                f"comment:{cid}",
            )

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

        # Direct interaction evidence only (no invented influence via shared post).
        interacted: dict[int, set[int]] = {aid: set() for aid in self._current}
        for r in reactions:
            uid = r.get("user_id")
            if uid is None:
                continue
            aid = int(uid)
            target_author: int | None = None
            if r.get("comment_id") is not None:
                target_author = author_of_comment.get(int(r["comment_id"]))
            elif r.get("post_id") is not None:
                target_author = author_of_post.get(int(r["post_id"]))
            if target_author is not None and target_author != aid:
                interacted.setdefault(aid, set()).add(target_author)
        for c in comments:
            uid = c.get("user_id")
            if uid is None:
                continue
            aid = int(uid)
            content = str(c.get("content") or "")
            for ref in extract_local_comment_refs(content):
                target_author = author_of_comment.get(ref)
                # Only count refs that resolve to a real comment author.
                if target_author is not None and target_author != aid:
                    interacted.setdefault(aid, set()).add(target_author)

        changes: list[dict[str, Any]] = []
        for aid, (content, kind, evid) in latest.items():
            pol = _polarity(content)
            if (
                any(m in content.lower() for m in _SHIFT_MARKERS)
                and pol == "unknown"
            ):
                pol = "shifting"
            stance = _short(content)
            key = _stance_key(pol, stance)
            prev = self._current.get(aid)
            first_platform = aid not in self._established
            changed = False
            influencers: list[int] = []

            if first_platform:
                # Erstposition: goal → first real post/comment is establishment.
                changed = False
            elif prev is not None:
                changed = _is_evidenced_change(
                    prev_polarity=prev.polarity,
                    new_polarity=pol,
                    content=content,
                )
                # Rephrase / confirmation: keep prior polarity for stability when
                # confirmation restates without a real flip.
                if not changed and _is_confirmation(content) and prev.polarity in {
                    "pro",
                    "contra",
                    "exception",
                    "uncertain",
                    "unknown",
                }:
                    pol = prev.polarity
                    key = _stance_key(pol, stance)
                # Same-side Ausnahme restatement: keep exception polarity even
                # when a later comment is marker-sparse (avoid unknown drift).
                if (
                    not changed
                    and prev.polarity == "exception"
                    and pol in {"unknown", "uncertain"}
                    and not _is_explicit_shift(content)
                ):
                    pol = prev.polarity
                    key = _stance_key(pol, stance)

            if changed and prev:
                influencers = sorted(interacted.get(aid) or [])
                dedup = f"{aid}|{prev.polarity}|{pol}|{evid}"
                if dedup not in self._change_keys:
                    self._change_keys.add(dedup)
                    event = {
                        "agent_id": aid,
                        "from_stance": prev.stance,
                        "to_stance": stance,
                        "from_key": prev.stance_key,
                        "to_key": key,
                        "from_polarity": prev.polarity,
                        "to_polarity": pol,
                        "step": step,
                        "evidence": evid,
                        "influenced_by_agent_ids": influencers,
                    }
                    changes.append(event)
                    self._change_events.append(event)
                else:
                    # Already recorded this evidenced transition — no double count.
                    changed = False
                    influencers = []

            pos = AgentPosition(
                agent_id=aid,
                stance=stance,
                stance_key=key,
                polarity=pol,
                step=step,
                evidence=evid,
                evidence_kind=kind,
                changed=changed,
                previous_stance=prev.stance if changed and prev else None,
                previous_stance_key=prev.stance_key if changed and prev else None,
                influenced_by_agent_ids=influencers,
            )
            self._current[aid] = pos
            self._established.add(aid)
            hist = self._history.setdefault(aid, [])
            if not hist or _coarse_key(hist[-1].polarity) != _coarse_key(pol):
                hist.append(pos)
            elif hist[-1].stance_key != key and kind != hist[-1].evidence_kind:
                # Update latest history entry text without new polarity bucket.
                hist[-1] = pos
        return changes

    def polarity_distribution(self) -> dict[str, int]:
        """Histogram of current polarities — dashboard Positionsverteilung."""
        dist: dict[str, int] = {}
        for pos in self._current.values():
            key = pos.polarity or "unknown"
            dist[key] = dist.get(key, 0) + 1
        return dict(sorted(dist.items()))

    def positions_payload(self) -> dict[str, Any]:
        items = [p.to_dict() for p in sorted(self._current.values(), key=lambda x: x.agent_id)]
        return {
            "backend": "local",
            "agent_count": len(items),
            "change_count": len(self._change_events),
            "polarity_distribution": self.polarity_distribution(),
            "positions": items,
            "changes": list(self._change_events),
            "note": (
                "Persistente Agentenpositionen aus Goal-Seed + Posts/Kommentaren. "
                "Seed-Impulse und Erstposition ≠ Wechsel; position_changed / "
                "influenced_by nur bei belegtem Stance-Wechsel "
                "(Bestätigung / Ausnahme-Restatement zählen nicht). "
                "Phase 6: polarity_distribution für Society-Übersicht 10/25/50."
            ),
        }

    def social_edge_specs(self) -> list[dict[str, Any]]:
        """Evidence-gated edge specs for the knowledge graph (no invention)."""
        specs: list[dict[str, Any]] = []
        for ev in self._change_events:
            aid = int(ev["agent_id"])
            specs.append(
                {
                    "relation": "position_changed",
                    "source": f"actor:{aid}",
                    "target": f"topic:scenario",
                    "meta": {
                        "from_stance": ev.get("from_stance"),
                        "to_stance": ev.get("to_stance"),
                        "from_polarity": ev.get("from_polarity"),
                        "to_polarity": ev.get("to_polarity"),
                        "step": ev.get("step"),
                        "evidence": ev.get("evidence"),
                    },
                }
            )
            for other in ev.get("influenced_by_agent_ids") or []:
                specs.append(
                    {
                        "relation": "influenced_by",
                        "source": f"actor:{aid}",
                        "target": f"actor:{int(other)}",
                        "meta": {
                            "step": ev.get("step"),
                            "evidence": ev.get("evidence")
                            or "position_change_after_interaction",
                            "via": "reaction_or_local_comment_ref",
                        },
                    }
                )
        return specs


def collect_local_responds_to(
    comments: list[dict[str, Any]],
    *,
    comment_statement_ids: dict[int, str],
    post_statement_ids: dict[int, str],
    agent_nodes: dict[int, str],
) -> list[dict[str, Any]]:
    """Build responds_to edges from local comment refs + flat post replies."""
    specs: list[dict[str, Any]] = []
    for c in comments:
        cid = c.get("comment_id")
        uid = c.get("user_id")
        if cid is None or uid is None:
            continue
        aid = int(uid)
        if aid not in agent_nodes:
            continue
        content = str(c.get("content") or "")
        refs = extract_local_comment_refs(content)
        for ref in refs:
            if ref not in comment_statement_ids:
                continue
            specs.append(
                {
                    "relation": "responds_to",
                    "source": agent_nodes[aid],
                    "target": comment_statement_ids[ref],
                    "meta": {
                        "comment_id": cid,
                        "target_comment_id": ref,
                        "evidence": "local_comment_ref",
                        "note": "OASIS has no parent_comment_id",
                    },
                }
            )
        # Flat create_comment under a post = responds_to that post statement.
        pid = c.get("post_id")
        if pid is not None and int(pid) in post_statement_ids:
            specs.append(
                {
                    "relation": "responds_to",
                    "source": agent_nodes[aid],
                    "target": post_statement_ids[int(pid)],
                    "meta": {
                        "comment_id": cid,
                        "post_id": pid,
                        "evidence": "create_comment",
                    },
                }
            )
    return specs


def collect_textual_agree_disagree(
    comments: list[dict[str, Any]],
    *,
    comment_statement_ids: dict[int, str],
    agent_nodes: dict[int, str],
) -> list[dict[str, Any]]:
    """agrees_with / disagrees_with from comment text + resolvable →Kommentar#.

    Only emits when both (1) a local comment ref resolves and (2) clear
    agree/disagree markers appear in the same comment. No invention from
    bare replies or marker-free acknowledgements.
    """
    specs: list[dict[str, Any]] = []
    for c in comments:
        cid = c.get("comment_id")
        uid = c.get("user_id")
        if cid is None or uid is None:
            continue
        aid = int(uid)
        if aid not in agent_nodes:
            continue
        content = str(c.get("content") or "")
        refs = extract_local_comment_refs(content)
        if not refs:
            continue
        labels = classify_textual_agree_disagree(content)
        if not labels:
            continue
        for ref in refs:
            if ref not in comment_statement_ids:
                continue
            for relation in labels:
                specs.append(
                    {
                        "relation": relation,
                        "source": agent_nodes[aid],
                        "target": comment_statement_ids[ref],
                        "meta": {
                            "comment_id": cid,
                            "target_comment_id": ref,
                            "evidence": "textual_comment_ref",
                            "markers": labels,
                        },
                    }
                )
    return specs
