"""OASIS simulation runner — ManualAction and real LLMAction modes."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import oasis
from camel.memories import ChatHistoryMemory
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import (
    ActionType,
    AgentGraph,
    DefaultPlatformType,
    LLMAction,
    ManualAction,
    SocialAgent,
    UserInfo,
)

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    OFFLINE_MAX_AGENTS,
    OFFLINE_MAX_ROUNDS,
    OFFLINE_SCALE_PRESETS,
    Settings,
    settings as default_settings,
)
from backend.simulation.agent_generator import generate_agents
from backend.simulation.agents import DEFAULT_AGENTS, AgentProfile
from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.cost_token_planner import plan_run
from backend.simulation.discussion_engine import DiscussionEngine
from backend.simulation.interaction_engine import MultiAgentInteractionEngine
from backend.simulation.knowledge_graph import LocalKnowledgeGraph
from backend.simulation.local_memory import LocalAgentMemory
from backend.simulation.manual_script import build_manual_rounds
from backend.simulation.oasis_reply import oasis_reply_status
from backend.simulation.opinion_dynamics import OpinionDynamics
from backend.simulation.prompt_debug import (
    PROMPT_BUNDLE_ID,
    PromptDebugBuffer,
    action_dedup_key,
)
from backend.simulation.relationship_dynamics import RelationshipDynamics
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.simulation_intelligence import build_simulation_intelligence
from backend.simulation.society_analytics import build_society_analytics
from backend.simulation.social_scheduler import SocialInteractionScheduler
from backend.simulation.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    build_decision_prompt,
    build_profile_for_template,
    format_feed_context,
    format_memory_context,
    language_display_name,
    model_calls_required_per_round,
    normalize_language,
)
from backend.simulation.social_dynamics import PositionTracker
from backend.simulation.tracked_model import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    TrackedModelBackend,
    resolve_context_token_limit,
)
from backend.zep.client import ZepMemoryStub


def _unwrap_model_backend(model: Any) -> Any:
    """CAMEL wraps single backends in ModelManager — unwrap for token_limit."""
    if model is None:
        return None
    models = getattr(model, "models", None)
    if models:
        return models[0]
    return model


def _context_limit_for_agent_model(model: Any) -> int:
    inner = _unwrap_model_backend(model)
    if isinstance(inner, TrackedModelBackend):
        return int(inner.token_limit)
    if inner is not None:
        return resolve_context_token_limit(inner)
    return DEFAULT_CONTEXT_TOKEN_LIMIT


def _empty_interaction_summary() -> dict[str, Any]:
    return {
        "replied_to": 0,
        "agrees_with": 0,
        "disagrees_with": 0,
        "responds_to": 0,
        "influenced_by": 0,
        "position_changed": 0,
        "comments": 0,
        "post_reactions": 0,
        "comment_reactions": 0,
        "self_replies": 0,
        "cross_agent_replies": 0,
        "thread_refs": 0,
        "note": (
            "Nur belegte Social-Relations (like_*/dislike_*/create_comment/"
            "lokale Kommentar-Refs/Positionswechsel). OASIS: kein parent_comment_id."
        ),
    }


def _comment_thread_metrics(comments: list[dict[str, Any]]) -> dict[str, int]:
    """Count →Kommentar# self vs cross-agent refs (evidence only, no invention)."""
    from backend.simulation.social_dynamics import extract_local_comment_refs

    by_id: dict[int, int] = {}
    for c in comments:
        cid = c.get("comment_id")
        uid = c.get("user_id")
        if cid is None or uid is None:
            continue
        try:
            by_id[int(cid)] = int(uid)
        except (TypeError, ValueError):
            continue
    self_replies = 0
    cross_replies = 0
    thread_refs = 0
    for c in comments:
        uid = c.get("user_id")
        if uid is None:
            continue
        try:
            author = int(uid)
        except (TypeError, ValueError):
            continue
        refs = extract_local_comment_refs(str(c.get("content") or ""))
        for ref in refs:
            thread_refs += 1
            target_uid = by_id.get(ref)
            if target_uid is None:
                continue
            if target_uid == author:
                self_replies += 1
            else:
                cross_replies += 1
    return {
        "self_replies": self_replies,
        "cross_agent_replies": cross_replies,
        "thread_refs": thread_refs,
    }


def _build_interaction_summary(
    *,
    comments: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    knowledge_graph: dict[str, Any],
) -> dict[str, Any]:
    """Dashboard-facing counts; graph social/interaction_relations are source of truth."""
    ir = knowledge_graph.get("interaction_relations") or {}
    sr = knowledge_graph.get("social_relations") or {}
    post_rx = sum(1 for r in reactions if r.get("post_id") is not None)
    comment_rx = sum(1 for r in reactions if r.get("comment_id") is not None)
    threads = _comment_thread_metrics(comments)
    return {
        "replied_to": int(ir.get("replied_to") or 0),
        "agrees_with": int(sr.get("agrees_with") or ir.get("agrees_with") or 0),
        "disagrees_with": int(
            sr.get("disagrees_with") or ir.get("disagrees_with") or 0
        ),
        "responds_to": int(sr.get("responds_to") or 0),
        "influenced_by": int(sr.get("influenced_by") or 0),
        "position_changed": int(sr.get("position_changed") or 0),
        "comments": len(comments),
        "post_reactions": post_rx,
        "comment_reactions": comment_rx,
        "self_replies": threads["self_replies"],
        "cross_agent_replies": threads["cross_agent_replies"],
        "thread_refs": threads["thread_refs"],
        "note": (
            "Nur belegte Social-Relations. OASIS: kein parent_comment_id — "
            "Reply = create_comment(post_id); Kommentar→Kommentar via →Kommentar#id; "
            "self_replies/cross_agent_replies aus belegten →Kommentar#-Refs."
        ),
    }


# Explicit OASIS Reddit interaction set for Phase-4 LLMAction runs.
# Reply = CREATE_COMMENT on an observed post_id — OASIS has no parent_comment_id
# and no separate reply ActionType (see oasis_reply.py).
# SEARCH_POSTS / REFRESH removed: OASIS already refreshes inside to_text_prompt,
# and those tools produced empty interaction (no comments/reactions) in live runs.
# FOLLOW removed in Phase 4.1 (tool-schema cost without debate value).
# comment / like / dislike stay available without quotas.
LLM_INTERACTION_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.DO_NOTHING,
]


EventCallback = Callable[[dict[str, Any]], None]

# Terminal run outcomes (plus lifecycle: idle|running|blocked).
STATUS_COMPLETED = "completed"
STATUS_LIMIT_REACHED = "limit_reached"
STATUS_FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SimulationState:
    run_id: str
    status: str = "idle"  # idle|running|completed|limit_reached|failed|blocked
    mode: str = "manual"  # manual|llm
    scenario: str = ""
    language: str = "de"
    step: int = 0
    max_steps: int = 4
    agent_count: int = 5
    agent_generation: str = "dynamic"
    llm_configured: bool = False
    message: str = ""
    stop_reason: str | None = None
    agents: list[dict] = field(default_factory=list)
    posts: list[dict] = field(default_factory=list)
    comments: list[dict] = field(default_factory=list)
    reactions: list[dict] = field(default_factory=list)
    action_log: list[dict] = field(default_factory=list)
    decision_log: list[dict] = field(default_factory=list)
    model_calls: list[dict] = field(default_factory=list)
    cost_summary: dict[str, Any] = field(default_factory=dict)
    agent_memory: dict[str, Any] = field(default_factory=dict)
    agent_positions: dict[str, Any] = field(default_factory=dict)
    relationships: dict[str, Any] = field(default_factory=dict)
    opinion_dynamics: dict[str, Any] = field(default_factory=dict)
    society_analytics: dict[str, Any] = field(default_factory=dict)
    knowledge_graph: dict[str, Any] = field(default_factory=dict)
    interaction_summary: dict[str, Any] = field(default_factory=dict)
    oasis_reply: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    interaction_engine: dict[str, Any] = field(default_factory=dict)
    society_overview: dict[str, Any] = field(default_factory=dict)
    discussion: dict[str, Any] = field(default_factory=dict)
    cost_plan: dict[str, Any] = field(default_factory=dict)
    simulation_intelligence: dict[str, Any] = field(default_factory=dict)
    execution_kind: str = "manual"  # manual | scripted | live_llm
    error: str | None = None
    result_path: str | None = None
    db_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "execution_kind": self.execution_kind,
            "scenario": self.scenario,
            "language": self.language,
            "step": self.step,
            "max_steps": self.max_steps,
            "agent_count": self.agent_count,
            "agent_generation": self.agent_generation,
            "llm_configured": self.llm_configured,
            "message": self.message,
            "stop_reason": self.stop_reason,
            "agents": self.agents,
            "posts": self.posts,
            "comments": self.comments,
            "reactions": self.reactions,
            "action_log": self.action_log,
            "decision_log": self.decision_log,
            "model_calls": self.model_calls,
            "cost_summary": self.cost_summary,
            "agent_memory": self.agent_memory,
            "agent_positions": self.agent_positions,
            "relationships": self.relationships,
            "opinion_dynamics": self.opinion_dynamics,
            "society_analytics": self.society_analytics,
            "knowledge_graph": self.knowledge_graph,
            "interaction_summary": self.interaction_summary,
            "oasis_reply": self.oasis_reply,
            "scheduler": self.scheduler,
            "interaction_engine": self.interaction_engine,
            "society_overview": self.society_overview,
            "discussion": self.discussion,
            "cost_plan": self.cost_plan,
            "simulation_intelligence": self.simulation_intelligence,
            "error": self.error,
            "result_path": self.result_path,
            "db_path": self.db_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class SimulationEngine:
    """Thread-safe façade around OASIS OasisEnv."""

    def __init__(
        self,
        cfg: Settings | None = None,
        *,
        model_factory: Callable[[int], BaseModelBackend] | None = None,
    ) -> None:
        self.cfg = cfg or default_settings
        self._lock = threading.Lock()
        self.state = SimulationState(run_id="none")
        self._thread: threading.Thread | None = None
        self._listeners: list[EventCallback] = []
        self.zep = ZepMemoryStub(self.cfg)
        # Optional injectable model builder for tests (no paid API).
        # Signature: agent_id -> BaseModelBackend (untracked; engine wraps it).
        self._model_factory = model_factory
        self._tracker: CostTracker | None = None
        self._prompt_debug: PromptDebugBuffer = PromptDebugBuffer()
        self._run_agents: list[AgentProfile] = list(
            DEFAULT_AGENTS[:LIVE_TEST_MAX_AGENTS]
        )
        self._agent_profiles: dict[int, AgentProfile] = {
            p.agent_id: p for p in self._run_agents
        }
        self._local_memory = LocalAgentMemory(
            max_entries_per_agent=self.cfg.agent_local_memory_entries,
            max_long_term_per_agent=self.cfg.agent_long_term_memory_entries,
        )
        self._knowledge_graph = LocalKnowledgeGraph()
        self._positions = PositionTracker()
        self._relationships = RelationshipDynamics()
        self._opinion = OpinionDynamics()
        self._scheduler = SocialInteractionScheduler()
        self._interaction = MultiAgentInteractionEngine(self._scheduler)
        self._discussion = DiscussionEngine()
        self._run_language: str = "de"
        self._run_scenario: str = ""
        self._agent_generation: str = "dynamic"
        self._execution_kind: str = "manual"
        self._limit_stop: bool = False
        self._failed: bool = False
        self._custom_prompts_verified: bool = False

    def _is_offline_scripted(self) -> bool:
        """Test injection / ScriptedReactiveModel — never paid OpenAI."""
        return self._model_factory is not None

    def _agent_ceiling(self, *, mode: str) -> int:
        """Live-LLM stays at 5; manual + scripted may scale to OFFLINE_MAX_AGENTS."""
        if mode == "llm" and not self._is_offline_scripted():
            return LIVE_TEST_MAX_AGENTS
        return OFFLINE_MAX_AGENTS

    def _round_ceiling(self, *, mode: str) -> int:
        if mode == "llm" and not self._is_offline_scripted():
            return LIVE_TEST_MAX_ROUNDS
        return max(LIVE_TEST_MAX_ROUNDS, OFFLINE_MAX_ROUNDS)

    def add_listener(self, cb: EventCallback) -> None:
        self._listeners.append(cb)

    def _emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        data = {"event": event, "state": self.state.to_dict(), "payload": payload or {}}
        for cb in list(self._listeners):
            try:
                cb(data)
            except Exception:
                pass

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return self.state.to_dict()

    def _new_tracker(
        self, *, n_agents: int = 5, steps: int = 2
    ) -> CostTracker:
        n = max(1, int(n_agents))
        steps_n = max(1, int(steps))
        # Live paid LLM: hard ceilings unchanged (5×2≤10 / 20k / $0.20).
        if not self._is_offline_scripted():
            return CostTracker(
                max_calls=min(self.cfg.max_llm_calls_per_run, LIVE_TEST_MAX_LLM_CALLS),
                max_tokens=self.cfg.max_tokens_per_run,
                price_input_per_1m=self.cfg.price_input_per_1m,
                price_output_per_1m=self.cfg.price_output_per_1m,
                max_cost_usd=self.cfg.max_cost_usd_per_run,
                model_name=self.cfg.openai_model,
                max_tokens_per_call=self.cfg.max_tokens_per_call,
                max_parallel_calls=min(
                    self.cfg.max_parallel_llm_calls,
                    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
                ),
            )
        # Scripted with N≤5: keep configured ceilings so Approval/Reserve/Limit
        # regressions (Call-Limit=2, Token-Overshoot, …) still fire.
        if n <= LIVE_TEST_MAX_AGENTS:
            return CostTracker(
                max_calls=min(self.cfg.max_llm_calls_per_run, LIVE_TEST_MAX_LLM_CALLS),
                max_tokens=self.cfg.max_tokens_per_run,
                price_input_per_1m=self.cfg.price_input_per_1m,
                price_output_per_1m=self.cfg.price_output_per_1m,
                max_cost_usd=self.cfg.max_cost_usd_per_run,
                model_name="scripted-reactive",
                max_tokens_per_call=self.cfg.max_tokens_per_call,
                max_parallel_calls=min(
                    self.cfg.max_parallel_llm_calls,
                    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
                ),
            )
        # Scripted society scale (10/25/50): budget for full N×rounds, no paid API.
        needed_calls = max(1, n * steps_n)
        needed_tokens = max(
            self.cfg.max_tokens_per_run,
            needed_calls * max(80, self.cfg.max_tokens_per_call),
        )
        return CostTracker(
            max_calls=needed_calls,
            max_tokens=needed_tokens,
            price_input_per_1m=self.cfg.price_input_per_1m,
            price_output_per_1m=self.cfg.price_output_per_1m,
            max_cost_usd=max(self.cfg.max_cost_usd_per_run, 999.0),
            model_name="scripted-reactive",
            max_tokens_per_call=self.cfg.max_tokens_per_call,
            max_parallel_calls=min(max(8, self.cfg.max_parallel_llm_calls), n),
        )

    def _is_busy(self) -> bool:
        """One simulation at a time — running state or live worker thread."""
        if self.state.status == "running":
            return True
        if self._thread is not None and self._thread.is_alive():
            return True
        return False

    def _sync_cost_to_state(self) -> None:
        if self._tracker is None:
            return
        with self._lock:
            self.state.model_calls = [c.to_dict() for c in self._tracker.calls]
            self.state.cost_summary = self._tracker.summary()

    def start(
        self,
        scenario: str,
        *,
        mode: str | None = None,
        max_steps: int | None = None,
        llm_run_approved: bool = False,
        language: str | None = None,
        agent_count: int | None = None,
        agent_generation: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._is_busy():
                return {
                    "ok": False,
                    "error": "Simulation läuft bereits — parallele Läufe sind deaktiviert.",
                    "state": self.state.to_dict(),
                }

            # Default is always ManualAction. "auto" must never silently enable LLM.
            requested = (mode or "manual").lower().strip()
            if requested in {"", "auto"}:
                requested = "manual"
            if requested not in {"manual", "llm"}:
                return {
                    "ok": False,
                    "error": f"Unbekannter Modus '{mode}'. Erlaubt: manual | llm.",
                    "state": self.state.to_dict(),
                }

            sim_lang = normalize_language(
                language if language is not None else self.cfg.simulation_language
            )

            gen_mode = (
                agent_generation
                if agent_generation is not None
                else self.cfg.default_agent_generation
            ).strip().lower() or "dynamic"
            if gen_mode in {"fixed", "test", "default"}:
                gen_mode = "fixed_test"
            if gen_mode not in {"dynamic", "fixed_test"}:
                return {
                    "ok": False,
                    "error": (
                        f"Unbekannte agent_generation '{agent_generation}'. "
                        "Erlaubt: dynamic | fixed_test."
                    ),
                    "state": self.state.to_dict(),
                }

            raw_count = (
                agent_count
                if agent_count is not None
                else self.cfg.default_agent_count
            )
            try:
                n_agents = int(raw_count)
            except (TypeError, ValueError):
                return {
                    "ok": False,
                    "error": "agent_count muss eine Zahl sein.",
                    "state": self.state.to_dict(),
                }
            # Ceiling depends on path: Live-LLM ≤5; Manual/Scripted ≤50.
            agent_cap = self._agent_ceiling(mode=requested)
            n_agents = max(1, min(n_agents, agent_cap))
            # Live paid LLM never exceeds LIVE_TEST_MAX_AGENTS even if mis-set.
            if requested == "llm" and not self._is_offline_scripted():
                n_agents = min(n_agents, LIVE_TEST_MAX_AGENTS)
            # fixed_test always uses the classic 5-roster truncated to n_agents;
            # dashboard "Testmodus" typically requests count=5.
            # dynamic may scale to offline presets 10/25/50.
            run_agents = generate_agents(
                scenario.strip(),
                n_agents,
                mode=gen_mode,
                max_agents=agent_cap,
            )
            agent_dicts = [a.to_dict() for a in run_agents]

            # Approval is enforced before key/availability checks so the gate
            # cannot be bypassed by frontend-only confirmation.
            if requested == "llm" and not llm_run_approved:
                msg = (
                    "LLM-Lauf ohne explizite Freigabe blockiert. "
                    "Sende llm_run_approved=true nach Bestätigung im Dashboard "
                    "(Modell, Agenten, Runden, Call-Limit, geschätzte max. Kosten)."
                )
                self.state = SimulationState(
                    run_id=str(uuid.uuid4())[:8],
                    status="blocked",
                    mode="llm",
                    scenario=scenario,
                    language=sim_lang,
                    agent_count=len(run_agents),
                    agent_generation=gen_mode,
                    llm_configured=self.cfg.llm_configured,
                    message=msg,
                    agents=agent_dicts,
                    cost_summary={},
                )
                blocked = self.state.to_dict()
                self._emit("blocked")
                return {"ok": False, "error": msg, "state": blocked}

            # Test injection may enable llm without a real key.
            llm_ok = self.cfg.llm_mode_available or self._model_factory is not None
            if requested == "llm" and not llm_ok:
                self.state = SimulationState(
                    run_id=str(uuid.uuid4())[:8],
                    status="blocked",
                    mode="llm",
                    scenario=scenario,
                    language=sim_lang,
                    agent_count=len(run_agents),
                    agent_generation=gen_mode,
                    llm_configured=self.cfg.llm_configured,
                    message=self.cfg.status_payload()["llm_mode_blocked_reason"]
                    or "LLM-Modus nicht verfügbar.",
                    agents=agent_dicts,
                    cost_summary={},
                )
                blocked = self.state.to_dict()
                self._emit("blocked")
                return {"ok": False, "error": self.state.message, "state": blocked}

            effective = requested

            steps = max_steps or self.cfg.max_simulation_steps
            round_cap = self._round_ceiling(mode=effective)
            # Live paid LLM always ≤ LIVE_TEST_MAX_ROUNDS; offline may use OFFLINE_MAX_ROUNDS.
            if effective == "llm" and not self._is_offline_scripted():
                steps = max(1, min(steps, self.cfg.max_simulation_steps, LIVE_TEST_MAX_ROUNDS))
            else:
                steps = max(1, min(steps, round_cap))

            run_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + str(uuid.uuid4())[:8]
            )
            self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.cfg.results_dir / f"{run_id}.db"

            calls_per_round = model_calls_required_per_round(len(run_agents))
            if self._is_offline_scripted():
                exec_kind = "scripted"
            elif effective == "llm":
                exec_kind = "live_llm"
            else:
                exec_kind = "manual"
            self._execution_kind = exec_kind

            if effective == "llm":
                if self._model_factory is not None:
                    message = (
                        "ScriptedModel (Offline): echte OASIS LLMAction, "
                        "kein kostenpflichtiger Provider-Call. Freigabe bestätigt. "
                        f"Sprache={sim_lang}. Agenten={len(run_agents)} ({gen_mode}). "
                        f"Scale-Presets={list(OFFLINE_SCALE_PRESETS)}."
                    )
                else:
                    est = self.cfg.estimate_max_run_cost_usd(rounds=steps)
                    message = (
                        "LLM-Modus: echte OASIS LLMAction über OpenAI-kompatibles Modell. "
                        f"Freigabe bestätigt. Sprache={sim_lang} "
                        f"({language_display_name(sim_lang)}). "
                        f"Agenten={len(run_agents)} ({gen_mode}). Limits: "
                        f"{min(self.cfg.max_llm_calls_per_run, LIVE_TEST_MAX_LLM_CALLS)} Calls / "
                        f"{self.cfg.max_tokens_per_run} Tokens / "
                        f"${self.cfg.max_cost_usd_per_run:.2f} / "
                        f"{steps} Runden / retries={self.cfg.llm_max_retries} / "
                        f"~{calls_per_round} Modellaufrufe/Runde / "
                        f"geschätzte max. Kosten ~${est:.4f} "
                        "(Provider-Abrechnung kann abweichen). "
                        "Live-LLM-Ceiling: max 5 Agenten."
                    )
            else:
                message = (
                    "ManualAction-Modus: echte OASIS-Umgebung mit StubModel — "
                    "keine LLM-API-Calls. "
                    f"Sprache={sim_lang}. Agenten={len(run_agents)} ({gen_mode}). "
                    f"Offline-Scale bis {OFFLINE_MAX_AGENTS}. "
                    + (
                        "OPENAI_API_KEY fehlt."
                        if not self.cfg.llm_configured
                        else "Manuell gewählt."
                    )
                )

            self._tracker = self._new_tracker(n_agents=len(run_agents), steps=steps)
            self._prompt_debug = PromptDebugBuffer()
            self._local_memory = LocalAgentMemory(
                max_entries_per_agent=self.cfg.agent_local_memory_entries,
                max_long_term_per_agent=self.cfg.agent_long_term_memory_entries,
            )
            self._knowledge_graph = LocalKnowledgeGraph()
            self._positions = PositionTracker()
            self._positions.seed_from_goals(run_agents, step=0)
            self._relationships = RelationshipDynamics()
            self._opinion = OpinionDynamics()
            self._scheduler = SocialInteractionScheduler(scenario=scenario.strip())
            self._interaction = MultiAgentInteractionEngine(
                self._scheduler, scenario=scenario.strip()
            )
            self._discussion = DiscussionEngine()
            ScriptedReactiveModel.shared_scheduler = self._scheduler
            ScriptedReactiveModel.shared_scenario = scenario.strip()
            ScriptedReactiveModel.shared_discussion = self._discussion
            self._limit_stop = False
            self._failed = False
            self._custom_prompts_verified = False
            self._run_language = sim_lang
            self._run_scenario = scenario.strip()
            self._agent_generation = gen_mode
            self._run_agents = run_agents
            self._agent_profiles = {p.agent_id: p for p in run_agents}
            reply_meta = oasis_reply_status()
            overview = {
                "agent_count": len(run_agents),
                "presets": list(OFFLINE_SCALE_PRESETS),
                "execution_kind": exec_kind,
                "live_llm_max_agents": LIVE_TEST_MAX_AGENTS,
                "offline_max_agents": OFFLINE_MAX_AGENTS,
                "paid_llm_unlocked_for_scale": False,
                "persona_engine": "v3",
                "quality_fix": "8.2",
            }
            seed_positions = self._positions.positions_payload()
            self.state = SimulationState(
                run_id=run_id,
                status="running",
                mode=effective,
                execution_kind=exec_kind,
                scenario=scenario.strip(),
                language=sim_lang,
                max_steps=steps,
                agent_count=len(run_agents),
                agent_generation=gen_mode,
                llm_configured=self.cfg.llm_configured,
                message=message,
                agents=agent_dicts,
                agent_memory=self._local_memory.snapshot(),
                agent_positions=seed_positions,
                relationships=self._relationships.snapshot(),
                opinion_dynamics=self._opinion.snapshot(),
                society_analytics={},
                knowledge_graph=self._knowledge_graph.to_dict(),
                interaction_summary=_empty_interaction_summary(),
                oasis_reply=reply_meta,
                scheduler=self._scheduler.summary(),
                interaction_engine=self._interaction.summary(),
                society_overview=overview,
                discussion=self._discussion.snapshot(
                    polarity_distribution=seed_positions.get("polarity_distribution")
                    or {},
                    agent_count=len(run_agents),
                ),
                db_path=str(db_path),
                started_at=_utc_now(),
                cost_summary=self._tracker.summary(),
            )
            self._emit("started")

            self._thread = threading.Thread(
                target=self._run_thread,
                args=(scenario.strip(), effective, steps, db_path, sim_lang),
                daemon=True,
            )
            self._thread.start()
            return {"ok": True, "state": self.state.to_dict()}

    def _run_thread(
        self, scenario: str, mode: str, max_steps: int, db_path: Path, language: str
    ) -> None:
        try:
            asyncio.run(self._run_async(scenario, mode, max_steps, db_path, language))
        except Exception as exc:
            with self._lock:
                self.state.status = STATUS_FAILED
                self.state.error = str(exc)
                self.state.stop_reason = "exception"
                self.state.finished_at = _utc_now()
                self.state.message = f"Fehlgeschlagen: {exc}"
            self._sync_cost_to_state()
            self._emit("error", {"error": str(exc)})

    async def _run_async(
        self,
        scenario: str,
        mode: str,
        max_steps: int,
        db_path: Path,
        language: str,
    ) -> None:
        # Fresh DB per run — never reuse prior simulation SQLite files.
        if db_path.exists():
            db_path.unlink()
        # SocialEnvironment follower helpers read OASIS_DB_PATH; bind to this run.
        os.environ["OASIS_DB_PATH"] = str(db_path)

        graph = self._build_agent_graph(mode, scenario=scenario, language=language)
        # Live-LLM / small scripted: serialize (parallel=1). Large scripted scale: higher.
        n_run = len(self._run_agents) or 1
        if self._is_offline_scripted() and n_run > LIVE_TEST_MAX_AGENTS:
            parallel = max(1, min(8, n_run))
        else:
            parallel = max(
                1,
                min(
                    self.cfg.max_parallel_llm_calls,
                    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
                    LIVE_TEST_MAX_AGENTS,
                ),
            )
        env = oasis.make(
            agent_graph=graph,
            platform=DefaultPlatformType.REDDIT,
            database_path=str(db_path),
            semaphore=parallel,
        )

        self.zep.maybe_remember_session(
            self.state.run_id,
            scenario=scenario,
            mode=mode,
            agents=self._run_agents,
        )
        self.zep.prepare_local_memory_handoff(
            self.state.run_id, self._local_memory.snapshot()
        )

        await env.reset()
        self._refresh_from_db(db_path)
        self._emit("reset")

        try:
            if mode == "manual":
                await self._run_manual(env, scenario, max_steps, db_path)
            else:
                await self._run_llm(env, scenario, max_steps, db_path, language)
        finally:
            await env.close()

        self._sync_cost_to_state()
        # LLM mode with zero model calls is never a success (seed ≠ LLM work).
        # Wins over earlier soft-fail flags so stop_reason stays explicit.
        if mode == "llm" and self._tracker is not None and self._tracker.call_count == 0:
            self._failed = True
            with self._lock:
                self.state.stop_reason = "llm_zero_model_calls"
                self.state.error = (
                    "LLM-Lauf ohne Modellaufrufe (0 API-Calls/Tokens). "
                    "Seed-Post allein ist kein Erfolg."
                )
                self.state.message = self.state.error

        # Token overshoot is never a successful completed run (false completed).
        # Call-budget emptying on the final round without overshoot stays completed.
        rounds_finished = self.state.step >= self.state.max_steps
        token_exceeded = bool(self._tracker and self._tracker.tokens_exceeded)
        early_limit_stop = bool(
            (self._limit_stop or (self._tracker and self._tracker.limit_hit))
            and not rounds_finished
            and not token_exceeded
        )
        if self._failed:
            status = STATUS_FAILED
            stop_reason = self.state.stop_reason or "llm_error"
            msg_prefix = self.state.message or "Simulation fehlgeschlagen."
        elif token_exceeded:
            status = STATUS_LIMIT_REACHED
            stop_reason = (
                (self._tracker.limit_hit if self._tracker else None)
                or "token_budget_exceeded"
            )
            msg_prefix = (
                f"Token-Budget überschritten ({self.state.mode}). {stop_reason}"
            )
        elif early_limit_stop:
            status = STATUS_LIMIT_REACHED
            stop_reason = (
                (self._tracker.limit_hit if self._tracker else None)
                or self.state.stop_reason
                or "limit"
            )
            msg_prefix = f"Limit erreicht ({self.state.mode}). {stop_reason}"
        else:
            status = STATUS_COMPLETED
            stop_reason = "all_rounds_finished"
            if self._tracker and self._tracker.limit_hit and rounds_finished:
                stop_reason = "all_rounds_finished_budget_exhausted"
            msg_prefix = f"Abgeschlossen ({self.state.mode})."

        # Write JSON with the final status, then publish status+result_path together
        # so pollers never observe a terminal status without a result file (N=50 race).
        with self._lock:
            self.state.stop_reason = stop_reason
            self.state.finished_at = _utc_now()
        result_path = self._persist_results(db_path, status=status, stop_reason=stop_reason)
        with self._lock:
            self.state.result_path = str(result_path)
            self.state.message = f"{msg_prefix} Ergebnisse: {result_path.name}."
            self.state.status = status
        self._emit("completed" if status == STATUS_COMPLETED else status)

    def _build_inner_model(self, mode: str, agent_id: int) -> BaseModelBackend:
        if self._model_factory is not None:
            return self._model_factory(agent_id)
        if mode == "llm":
            if self.cfg.openai_api_key:
                os.environ["OPENAI_API_KEY"] = self.cfg.openai_api_key
            if self.cfg.openai_base_url:
                os.environ.setdefault("OPENAI_API_BASE_URL", self.cfg.openai_base_url)
            return ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type=self.cfg.openai_model,
                api_key=self.cfg.openai_api_key,
                url=self.cfg.openai_base_url,
                timeout=float(self.cfg.request_timeout_seconds),
                # Live-test: never auto-retry paid calls (ceiling is 0).
                max_retries=0,
                model_config_dict={
                    "temperature": 0.7,
                    "max_tokens": self.cfg.max_tokens_per_call,
                },
            )
        return ModelFactory.create(
            model_platform=ModelPlatformType.STUB,
            model_type=ModelType.STUB,
        )

    def _wrap_model(self, inner: BaseModelBackend, agent_id: int, mode: str):
        if self._tracker is None:
            self._tracker = self._new_tracker()
        source = "scripted_test" if self._model_factory is not None else (
            "llm_api" if mode == "llm" else "stub"
        )
        return TrackedModelBackend(
            inner,
            self._tracker,
            agent_id=agent_id,
            source=source,
            prompt_debug=self._prompt_debug if mode == "llm" else None,
        )

    def _apply_memory_window(self, agent: SocialAgent) -> None:
        """Limit chat history each agent carries into the next LLM step.

        Rebuild ScoreBasedContextCreator from the wrapped model's context
        window — never reuse a creator that baked completion max_tokens (500)
        in as the chat token_limit.
        """
        from camel.memories.context_creators.score_based import (
            ScoreBasedContextCreator,
        )

        window = max(1, self.cfg.agent_memory_window)
        model = getattr(agent, "model_backend", None) or getattr(agent, "model", None)
        token_counter = (
            model.token_counter
            if model is not None
            else agent.memory.get_context_creator().token_counter
        )
        context_limit = _context_limit_for_agent_model(model)
        creator = ScoreBasedContextCreator(token_counter, context_limit)
        agent.memory = ChatHistoryMemory(
            context_creator=creator,
            window_size=window,
            agent_id=getattr(agent, "agent_id", None),
        )

    def _install_decision_hook(
        self,
        agent: SocialAgent,
        profile: AgentProfile,
        *,
        scenario: str,
        language: str,
    ) -> None:
        """Replace OASIS English like-bait perform_action_by_llm with scenario context."""
        engine = self

        async def perform_action_by_llm_bound():
            # Phase 4.1: reset chat to system-only before each decision so the
            # previous round's long user/tool transcript is not re-sent (~2k→4k).
            # Local memory + FEED-KONTEXT already carry actions / IDs.
            try:
                agent.init_messages()
            except Exception:
                mem = getattr(agent, "memory", None)
                if mem is not None and hasattr(mem, "clear"):
                    mem.clear()

            env_prompt = await agent.env.to_text_prompt()
            # Engine-owned prioritized feed + memory — OASIS env buries posts
            # after empty group-chat noise and ends with like-bait.
            with engine._lock:
                posts_snap = list(engine.state.posts)
                comments_snap = list(engine.state.comments)
                round_n = max(1, int(engine.state.step) + 1)
                pos_payload = engine.state.agent_positions or {}
            feed_context = format_feed_context(
                posts_snap,
                comments_snap,
                viewer_id=profile.agent_id,
            )
            memory_entries = engine._local_memory.recent_for_prompt(
                profile.agent_id, limit=3
            )
            memory_context = format_memory_context(memory_entries)
            position_context = None
            for p in pos_payload.get("positions") or []:
                if int(p.get("agent_id", -1)) == profile.agent_id:
                    position_context = (
                        f"{p.get('polarity') or '?'}: {p.get('stance') or ''}"
                    )
                    break
            argument_context = engine._discussion.prompt_argument_hints(
                profile.agent_id
            )
            user_content = build_decision_prompt(
                scenario=scenario,
                goal=profile.goal,
                language=language,
                env_prompt=env_prompt,
                agent_name=profile.name,
                feed_context=feed_context,
                memory_context=memory_context,
                round_number=round_n,
                position_context=position_context,
                argument_context=argument_context,
            )
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=user_content,
            )
            decision_base = {
                "agent_id": profile.agent_id,
                "agent_name": profile.name,
                "goal": profile.goal,
                "scenario": scenario,
                "language": language,
                "env_preview": (env_prompt or "")[:400],
                "feed_context_preview": feed_context[:500],
                "memory_context_preview": memory_context[:500],
                "ts": _utc_now(),
            }
            try:
                response = await agent.astep(user_msg)
                tool_calls = []
                info = getattr(response, "info", None) or {}
                terminated = bool(getattr(response, "terminated", False))
                termination_reasons = list(
                    info.get("termination_reasons") or []
                )
                # CAMEL returns terminated+max_tokens_exceeded without calling
                # the model when system prompt > context token_limit.
                context_blocked = terminated and any(
                    "max_tokens" in str(r).lower() for r in termination_reasons
                )
                for tool_call in info.get("tool_calls") or []:
                    action_name = getattr(tool_call, "tool_name", None) or getattr(
                        tool_call, "name", None
                    )
                    args = getattr(tool_call, "args", None) or {}
                    tool_calls.append({"action": action_name, "args": args})
                    engine._append_action_log(
                        {
                            "step": engine.state.step + 1,
                            "agent_id": profile.agent_id,
                            "action_type": action_name,
                            "action_args": args,
                            "source": "LLMAction",
                            "kind": "simulated_platform_action",
                            "via": "decision_hook",
                            "goal": profile.goal,
                            "language": language,
                            "ts": _utc_now(),
                        }
                    )
                    engine._local_memory.ingest_action(
                        agent_id=profile.agent_id,
                        action_type=str(action_name or "unknown"),
                        action_args=args,
                        step=engine.state.step + 1,
                        goal=profile.goal,
                    )
                evidence = engine._prompt_debug.last_for_agent(profile.agent_id)
                decision_ok = not context_blocked
                note = (
                    "scenario+goal+language+feed+memory injected; "
                    "OASIS default decision prompt bypassed"
                )
                if context_blocked:
                    note = (
                        "astep terminated before model call "
                        f"(reasons={termination_reasons}); "
                        "context token_limit vs completion max_tokens mismatch"
                    )
                    engine._failed = True
                engine._append_decision_log(
                    {
                        **decision_base,
                        "step": engine.state.step + 1,
                        "ok": decision_ok,
                        "tool_calls": tool_calls,
                        "terminated": terminated,
                        "termination_reasons": termination_reasons,
                        "prompt_bundle_id": PROMPT_BUNDLE_ID,
                        "prompt_evidence": evidence.to_dict() if evidence else None,
                        "note": note,
                        **(
                            {"error": "max_tokens_exceeded_before_model_call"}
                            if context_blocked
                            else {}
                        ),
                    }
                )
                return response
            except Exception as exc:
                evidence = engine._prompt_debug.last_for_agent(profile.agent_id)
                engine._append_decision_log(
                    {
                        **decision_base,
                        "step": engine.state.step + 1,
                        "ok": False,
                        "error": str(exc),
                        "tool_calls": [],
                        "prompt_bundle_id": PROMPT_BUNDLE_ID,
                        "prompt_evidence": evidence.to_dict() if evidence else None,
                    }
                )
                raise

        agent.perform_action_by_llm = perform_action_by_llm_bound  # type: ignore[method-assign]

    def _build_agent_graph(
        self, mode: str, *, scenario: str, language: str
    ) -> AgentGraph:
        graph = AgentGraph()
        # Prefer explicit interaction set for LLM; full Reddit defaults for manual.
        available = (
            list(LLM_INTERACTION_ACTIONS)
            if mode == "llm"
            else ActionType.get_default_reddit_actions()
        )
        for profile in self._run_agents:
            inner = self._build_inner_model(mode, profile.agent_id)
            model = (
                self._wrap_model(inner, profile.agent_id, mode)
                if mode == "llm"
                else inner
            )
            # Custom TextPrompt — OASIS default Reddit prompt says "tweets" and
            # ignores UserInfo.description unless other_info.user_profile is set.
            profile_dict = build_profile_for_template(
                profile, scenario=scenario, language=language
            )
            agent = SocialAgent(
                agent_id=profile.agent_id,
                user_info=UserInfo(
                    user_name=profile.user_name,
                    name=profile.name,
                    # Persona/goal/style live in SYSTEM_PROMPT_TEMPLATE via
                    # profile_dict — avoid duplicating them in description.
                    description=profile.style[:80],
                    profile=profile_dict,
                    recsys_type="reddit",
                ),
                user_info_template=SYSTEM_PROMPT_TEMPLATE,
                agent_graph=graph,
                model=model,
                available_actions=available,
            )
            if mode == "llm":
                self._apply_memory_window(agent)
                self._install_decision_hook(
                    agent, profile, scenario=scenario, language=language
                )
                # Prove custom path is bound on the live agent instance.
                sys_content = getattr(agent.system_message, "content", "") or ""
                hook_is_custom = (
                    agent.perform_action_by_llm
                    is not SocialAgent.perform_action_by_llm
                )
                self._custom_prompts_verified = bool(
                    hook_is_custom
                    and "# ZIEL / OBJECTIVE" in sys_content
                    and "You're a Reddit user" not in sys_content
                    and "I'll present you with some tweets" not in sys_content
                )
            graph.add_agent(agent)
        return graph

    def _interaction_path(self) -> str:
        if self._is_offline_scripted():
            return "scripted"
        if self._execution_kind == "live_llm":
            return "live_llm"
        return "manual"

    async def _run_manual(
        self, env, scenario: str, max_steps: int, db_path: Path
    ) -> None:
        rounds = build_manual_rounds(
            scenario, agents=self._run_agents, max_steps=max_steps
        )
        path = "manual"
        for idx, round_map in enumerate(rounds):
            round_n = idx + 1
            agent_ids = sorted(
                aid for aid in round_map.keys() if aid in self._agent_profiles
            )
            # Phase 8.1: plan_round before ManualAction waves (coverage metrics).
            self._interaction.prepare_round(
                round_n=round_n,
                agent_ids=agent_ids,
                posts=list(self.state.posts or []),
                comments=list(self.state.comments or []),
                path=path,
            )
            wave_groups = self._scheduler.wave_slices(agent_ids)
            for wave_idx, wave_ids in enumerate(wave_groups):
                actions: dict[SocialAgent, list[ManualAction]] = {}
                for agent_id in wave_ids:
                    acts = round_map.get(agent_id) or []
                    if not acts:
                        continue
                    agent = env.agent_graph.get_agent(agent_id)
                    actions[agent] = acts
                    profile = self._agent_profiles.get(agent_id)
                    for act in acts:
                        entry = {
                            "step": round_n,
                            "agent_id": agent_id,
                            "action_type": act.action_type.value,
                            "action_args": act.action_args,
                            "source": "ManualAction",
                            "kind": "simulated_platform_action",
                            "goal": profile.goal if profile else None,
                            "language": self._run_language,
                            "wave": wave_idx,
                            "ts": _utc_now(),
                        }
                        self._append_action_log(entry)
                        self._local_memory.ingest_action(
                            agent_id=agent_id,
                            action_type=act.action_type.value,
                            action_args=act.action_args,
                            step=round_n,
                            goal=profile.goal if profile else None,
                        )
                if actions:
                    await env.step(actions)
                    self._interaction.mark_wave(
                        round_n=round_n,
                        wave=wave_idx,
                        agent_ids=wave_ids,
                        path=path,
                    )
                    self._refresh_from_db(db_path)
            with self._lock:
                self.state.step = round_n
            self._refresh_from_db(db_path)
            self._emit("step", {"step": round_n})

    async def _run_llm(
        self, env, scenario: str, max_steps: int, db_path: Path, language: str
    ) -> None:
        # Seed so agents have content to perceive (ManualAction, not billed).
        # Public feed must not leak goals, IDs, style, or prompt instructions.
        from backend.simulation.topic_context import build_public_seed_content

        seed_profile = self._run_agents[0]
        seed_agent = env.agent_graph.get_agent(seed_profile.agent_id)
        seed_content = build_public_seed_content(scenario, language=language)
        seed = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": seed_content},
        )
        self._append_action_log(
            {
                "step": 0,
                "agent_id": seed_profile.agent_id,
                "action_type": "create_post",
                "action_args": seed.action_args,
                "source": "ManualAction",
                "kind": "simulated_platform_action",
                "note": "seed_before_llm",
                "goal": seed_profile.goal,
                "language": language,
                "ts": _utc_now(),
            }
        )
        self._local_memory.ingest_action(
            agent_id=seed_profile.agent_id,
            action_type="create_post",
            action_args=seed.action_args,
            step=0,
            goal=seed_profile.goal,
        )
        await env.step({seed_agent: seed})
        self._refresh_from_db(db_path)
        self._emit("step", {"step": 0})

        path = self._interaction_path()
        for step in range(1, max_steps + 1):
            assert self._tracker is not None
            remaining = self._tracker.remaining_calls()
            if remaining <= 0 or self._tracker.limit_hit:
                self._limit_stop = True
                with self._lock:
                    self.state.message = self._tracker.limit_hit or (
                        "LLM-Call-Limit erreicht."
                    )
                    self.state.stop_reason = self.state.message
                self._sync_cost_to_state()
                break
            # Live-LLM: cap to 5 (5×2≤10). Offline scripted: full society roster.
            agent_slice_cap = (
                len(self._run_agents)
                if self._is_offline_scripted()
                else LIVE_TEST_MAX_AGENTS
            )
            agents = [
                a for _, a in env.agent_graph.get_agents()
            ][: min(remaining, len(self._run_agents), agent_slice_cap)]
            agent_ids = [int(a.social_agent_id) for a in agents]

            # Phase 8.1: plan_round before wave execution (Scheduler V3 in execution).
            self._interaction.prepare_round(
                round_n=step,
                agent_ids=agent_ids,
                posts=list(self.state.posts or []),
                comments=list(self.state.comments or []),
                path=path,
            )

            wave_groups = self._interaction.wave_agent_groups(agents)
            aborted = False
            for wave_idx, wave_agents in enumerate(wave_groups):
                if self._tracker.remaining_calls() <= 0 or self._tracker.limit_hit:
                    self._limit_stop = True
                    break
                # Respect remaining call budget per wave.
                rem = self._tracker.remaining_calls()
                if rem < len(wave_agents):
                    wave_agents = wave_agents[: max(0, rem)]
                if not wave_agents:
                    break
                actions = {agent: LLMAction() for agent in wave_agents}
                wave_ids = [int(a.social_agent_id) for a in wave_agents]
                for agent in wave_agents:
                    profile = self._agent_profiles.get(agent.social_agent_id)
                    self._append_action_log(
                        {
                            "step": step,
                            "agent_id": agent.social_agent_id,
                            "action_type": "LLMAction",
                            "action_args": {
                                "scenario": scenario,
                                "goal": profile.goal if profile else None,
                                "language": language,
                                "wave": wave_idx,
                            },
                            "source": "LLMAction",
                            "kind": "llm_decision_request",
                            "goal": profile.goal if profile else None,
                            "language": language,
                            "wave": wave_idx,
                            "ts": _utc_now(),
                        }
                    )
                try:
                    await env.step(actions)
                    self._interaction.mark_wave(
                        round_n=step,
                        wave=wave_idx,
                        agent_ids=wave_ids,
                        path=path,
                    )
                except LimitExceeded as exc:
                    self._limit_stop = True
                    with self._lock:
                        self.state.message = str(exc)
                        self.state.stop_reason = str(exc)
                    self._sync_cost_to_state()
                    aborted = True
                    break
                except Exception as exc:
                    # Soft-fail a round: record and stop rather than crash unpaid tests.
                    self._failed = True
                    with self._lock:
                        self.state.message = f"LLM-Runde abgebrochen: {exc}"
                        self.state.error = str(exc)
                        self.state.stop_reason = "llm_round_error"
                    self._sync_cost_to_state()
                    self._refresh_from_db(db_path, merge_llm_traces=True, step=step)
                    aborted = True
                    break

                # Later waves see earlier wave content (controlled multi-agent scaling).
                self._refresh_from_db(db_path, merge_llm_traces=True, step=step)

            if aborted and self._failed:
                break
            if aborted and self._limit_stop:
                break

            with self._lock:
                self.state.step = step
            self._sync_cost_to_state()
            self._refresh_from_db(db_path, merge_llm_traces=True, step=step)
            self._emit("step", {"step": step})

            if self._tracker.limit_hit:
                with self._lock:
                    self.state.message = self._tracker.limit_hit
                    self.state.stop_reason = self._tracker.limit_hit
                # Budget exhausted after a finished round: early-stop only if
                # more rounds were planned (otherwise status stays completed).
                if step < max_steps:
                    self._limit_stop = True
                break

    def _append_action_log(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.state.action_log.append(entry)

    def _append_decision_log(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.state.decision_log.append(entry)

    def _refresh_from_db(
        self,
        db_path: Path,
        *,
        merge_llm_traces: bool = False,
        step: int | None = None,
    ) -> None:
        if not db_path.exists():
            return
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        try:
            posts = [
                dict(r)
                for r in con.execute(
                    "SELECT post_id, user_id, content, created_at, num_likes "
                    "FROM post ORDER BY post_id"
                )
            ]
            try:
                comments = [
                    dict(r)
                    for r in con.execute(
                        "SELECT comment_id, post_id, user_id, content, created_at "
                        "FROM comment ORDER BY comment_id"
                    )
                ]
            except sqlite3.OperationalError:
                comments = []
            try:
                likes = [
                    {"type": "like", "target": "post", **dict(r)}
                    for r in con.execute(
                        "SELECT like_id, user_id, post_id, created_at FROM like"
                    )
                ]
            except sqlite3.OperationalError:
                likes = []
            try:
                dislikes = [
                    {"type": "dislike", "target": "post", **dict(r)}
                    for r in con.execute(
                        "SELECT dislike_id, user_id, post_id, created_at FROM dislike"
                    )
                ]
            except sqlite3.OperationalError:
                dislikes = []
            try:
                comment_likes = [
                    {
                        "type": "like",
                        "target": "comment",
                        "comment_like_id": r["comment_like_id"],
                        "user_id": r["user_id"],
                        "comment_id": r["comment_id"],
                        "created_at": r["created_at"],
                    }
                    for r in con.execute(
                        "SELECT comment_like_id, user_id, comment_id, created_at "
                        "FROM comment_like"
                    )
                ]
            except sqlite3.OperationalError:
                comment_likes = []
            try:
                comment_dislikes = [
                    {
                        "type": "dislike",
                        "target": "comment",
                        "comment_dislike_id": r["comment_dislike_id"],
                        "user_id": r["user_id"],
                        "comment_id": r["comment_id"],
                        "created_at": r["created_at"],
                    }
                    for r in con.execute(
                        "SELECT comment_dislike_id, user_id, comment_id, created_at "
                        "FROM comment_dislike"
                    )
                ]
            except sqlite3.OperationalError:
                comment_dislikes = []
            try:
                traces = [
                    dict(r)
                    for r in con.execute(
                        "SELECT user_id, created_at, action, info FROM trace "
                        "ORDER BY created_at"
                    )
                ]
            except sqlite3.OperationalError:
                traces = []
        finally:
            con.close()

        with self._lock:
            self.state.posts = posts
            self.state.comments = comments
            self.state.reactions = (
                likes + dislikes + comment_likes + comment_dislikes
            )
            if merge_llm_traces and traces:
                # Normalize dict vs JSON-string args (and create_post content) so
                # seed ManualAction / decision_hook entries are not re-imported.
                known = {
                    action_dedup_key(
                        e.get("agent_id"), e.get("action_type"), e.get("action_args")
                    )
                    for e in self.state.action_log
                    if e.get("kind") == "simulated_platform_action"
                }
                for t in traces:
                    action = t.get("action")
                    if action in {"sign_up", "refresh", "search_posts"}:
                        continue
                    info = t.get("info")
                    key = action_dedup_key(t.get("user_id"), action, info)
                    if key in known:
                        continue
                    # Prefer decision_hook entries (carry goal/language).
                    already_hook = any(
                        e.get("via") == "decision_hook"
                        and e.get("agent_id") == t.get("user_id")
                        and e.get("action_type") == action
                        and e.get("step") == step
                        for e in self.state.action_log
                    )
                    if already_hook:
                        continue
                    profile = self._agent_profiles.get(t.get("user_id"))
                    entry = {
                        "step": step,
                        "agent_id": t.get("user_id"),
                        "action_type": action,
                        "action_args": info,
                        "source": "LLMAction",
                        "kind": "simulated_platform_action",
                        "via": "oasis_trace",
                        "goal": profile.goal if profile else None,
                        "language": self._run_language,
                        "ts": t.get("created_at"),
                    }
                    self.state.action_log.append(entry)
                    known.add(key)
                    self._local_memory.ingest_action(
                        agent_id=int(t.get("user_id") or -1),
                        action_type=str(action),
                        action_args=info,
                        step=step,
                        goal=profile.goal if profile else None,
                    )
            elif traces and not self.state.action_log:
                # Fallback only for unexpected empty logs — skip signup/noise.
                self.state.action_log = [
                    {
                        "step": None,
                        "agent_id": t.get("user_id"),
                        "action_type": t.get("action"),
                        "action_args": t.get("info"),
                        "source": "oasis_trace",
                        "kind": "simulated_platform_action",
                        "ts": t.get("created_at"),
                    }
                    for t in traces
                    if t.get("action")
                    not in {"sign_up", "refresh", "search_posts"}
                ]
            self._sync_memory_and_graph_locked(step=step)

    def _sync_memory_and_graph_locked(self, *, step: int | None = None) -> None:
        """Caller must hold ``self._lock`` (or accept best-effort race on idle)."""
        agent_ids = [p.agent_id for p in self._run_agents]
        sync_step = step if step is not None else self.state.step
        self._local_memory.ingest_feed_snapshot(
            agent_ids=agent_ids,
            posts=self.state.posts,
            comments=self.state.comments,
            step=sync_step,
        )
        self._positions.update_from_snapshot(
            agents=self._run_agents,
            posts=self.state.posts,
            comments=self.state.comments,
            reactions=self.state.reactions,
            step=sync_step,
        )
        positions_payload = self._positions.positions_payload()
        # Phase 7: promote evidenced stances into sourced long-term memory.
        self._local_memory.consolidate_from_positions(
            positions=positions_payload.get("positions") or [],
            step=sync_step,
        )
        # Phase 7.3: argument history + logical threads (local refs only).
        self._discussion.ingest_platform(
            posts=self.state.posts,
            comments=self.state.comments,
            step=sync_step,
            round_n=sync_step,
        )
        for pos in positions_payload.get("positions") or []:
            aid = pos.get("agent_id")
            evid = str(pos.get("evidence") or "")
            stance = str(pos.get("stance") or "")
            if aid is None or not evid or not stance:
                continue
            # Distinct source key so argument-dev does not collide with
            # consolidate_from_positions dedup on the raw evidence string.
            arg_evid = (
                evid if evid.startswith("argument:") else f"argument:{evid}"
            )
            self._local_memory.remember_argument_development(
                int(aid),
                snippet=stance,
                evidence=arg_evid,
                step=sync_step,
                polarity=str(pos.get("polarity") or "") or None,
            )
        edge_specs = self._positions.social_edge_specs()
        self._knowledge_graph.rebuild(
            scenario=self._run_scenario or self.state.scenario,
            agents=self._run_agents,
            posts=self.state.posts,
            comments=self.state.comments,
            reactions=self.state.reactions,
            action_log=self.state.action_log,
            social_edge_specs=edge_specs,
            agent_positions=positions_payload.get("positions") or [],
        )
        # Phase 7: dynamic relationships + opinion chronology (evidenced-only).
        self._relationships.ingest_platform(
            posts=self.state.posts,
            comments=self.state.comments,
            reactions=self.state.reactions,
            social_edge_specs=edge_specs,
            step=sync_step,
        )
        self._opinion.record_round(
            step=sync_step,
            polarity_distribution=positions_payload.get("polarity_distribution")
            or {},
            change_events=positions_payload.get("changes") or [],
            agent_count=len(self._run_agents),
        )
        self.state.agent_memory = self._local_memory.snapshot()
        self.state.agent_positions = positions_payload
        self.state.relationships = self._relationships.snapshot()
        self.state.opinion_dynamics = self._opinion.snapshot()
        self.state.knowledge_graph = self._knowledge_graph.to_dict()
        pol_tmp = positions_payload.get("polarity_distribution") or {}
        unk_tmp = float(pol_tmp.get("unknown") or 0) + float(
            pol_tmp.get("uncertain") or 0
        )
        pol_total = max(1.0, float(sum(int(v) for v in pol_tmp.values()) or 1))
        self.state.discussion = self._discussion.snapshot(
            polarity_distribution=pol_tmp,
            unknown_stance_share=unk_tmp / pol_total,
            agent_count=len(self._run_agents),
        )
        self.state.interaction_summary = _build_interaction_summary(
            comments=self.state.comments,
            reactions=self.state.reactions,
            knowledge_graph=self.state.knowledge_graph,
        )
        # Merge discussion thread metrics into interaction summary (honest OASIS).
        disc = self.state.discussion or {}
        self.state.interaction_summary = {
            **self.state.interaction_summary,
            "discussion_arguments": disc.get("argument_count", 0),
            "discussion_repetition_blocks": disc.get("repetition_blocks", 0),
            "discussion_thread_events": disc.get("thread_events", 0),
        }
        self.state.oasis_reply = oasis_reply_status()
        self.state.scheduler = self._scheduler.summary()
        self.state.interaction_engine = self._interaction.summary()
        pol = positions_payload.get("polarity_distribution") or {}
        social = (self.state.knowledge_graph or {}).get("social_relations") or {}
        self.state.society_analytics = build_society_analytics(
            agent_count=len(self._run_agents),
            max_steps=self.state.max_steps,
            step=sync_step,
            execution_kind=self._execution_kind,
            posts=self.state.posts,
            comments=self.state.comments,
            reactions=self.state.reactions,
            agent_memory=self.state.agent_memory,
            agent_positions=positions_payload,
            relationships=self.state.relationships,
            opinion=self.state.opinion_dynamics,
            knowledge_graph=self.state.knowledge_graph,
            interaction_summary=self.state.interaction_summary,
            scheduler=self.state.scheduler,
        )
        kind = self._execution_kind
        plan_kind = (
            "live_llm"
            if kind == "live_llm"
            else ("scripted" if kind == "scripted" else "manual")
        )
        cost_plan = plan_run(
            agent_count=len(self._run_agents),
            rounds=int(self.state.max_steps or 1),
            execution_kind=plan_kind,  # type: ignore[arg-type]
            price_input_per_1m=self.cfg.price_input_per_1m,
            price_output_per_1m=self.cfg.price_output_per_1m,
            max_tokens_per_call=self.cfg.max_tokens_per_call,
        )
        self.state.cost_plan = cost_plan.to_dict()
        self.state.simulation_intelligence = build_simulation_intelligence(
            agents=self.state.agents,
            execution_kind=kind,
            agent_count=len(self._run_agents),
            max_steps=self.state.max_steps,
            step=sync_step,
            cost_plan=cost_plan,
            scheduler=self.state.scheduler,
            society_analytics=self.state.society_analytics,
            discussion=self.state.discussion,
            agent_positions=positions_payload,
            interaction_engine=self.state.interaction_engine,
            opinion_dynamics=self.state.opinion_dynamics,
        )
        self.state.society_overview = {
            "agent_count": len(self._run_agents),
            "step": sync_step,
            "posts": len(self.state.posts),
            "comments": len(self.state.comments),
            "reactions": len(self.state.reactions),
            "polarity_distribution": pol,
            "change_count": positions_payload.get("change_count", 0),
            "social_relations": social,
            "network": (self.state.knowledge_graph or {}).get("network") or {},
            "execution_kind": self._execution_kind,
            "presets": list(OFFLINE_SCALE_PRESETS),
            "live_llm_max_agents": LIVE_TEST_MAX_AGENTS,
            "offline_max_agents": OFFLINE_MAX_AGENTS,
            "paid_llm_unlocked_for_scale": False,
            "scheduler_plans": (self.state.scheduler or {}).get("plans", 0),
            "scheduler_waves": (self.state.scheduler or {}).get("waves", 0),
            "scheduler_waves_executed": (self.state.scheduler or {}).get(
                "waves_executed", 0
            ),
            "self_as_partner_count": (self.state.scheduler or {}).get(
                "self_as_partner_count", 0
            ),
            "long_term_memory_agents": self.state.agent_memory.get(
                "agent_count_with_long_term", 0
            ),
            "relationship_bonds": self.state.relationships.get("bond_count", 0),
            "opinion_rounds_tracked": self.state.opinion_dynamics.get(
                "rounds_tracked", 0
            ),
            "analytics_kpis": (self.state.society_analytics or {}).get("kpis") or {},
            "persona_engine": "v3",
            "quality_fix": "8.2",
            "planned_cost_usd": cost_plan.planned_cost_usd,
            "planned_waves": cost_plan.planned_waves,
            "intelligence_kpis": (self.state.simulation_intelligence or {}).get("kpis")
            or {},
            "interaction_engine": (self.state.interaction_engine or {}).get("backend"),
            "debate_quality_score": (
                (self.state.discussion or {}).get("debate_quality") or {}
            ).get("score"),
            "stance_clarity": (self.state.opinion_dynamics or {}).get("stance_clarity"),
        }

    def _persist_results(
        self,
        db_path: Path,
        *,
        status: str | None = None,
        stop_reason: str | None = None,
    ) -> Path:
        self._refresh_from_db(db_path)
        self._sync_cost_to_state()
        out = self.cfg.results_dir / f"{self.state.run_id}.json"
        agents_n = len(self._run_agents) or LIVE_TEST_MAX_AGENTS
        prompt_summary = self._prompt_debug.summary()
        # Prefer live message captures; fall back to graph-install verification.
        custom_active = bool(
            prompt_summary.get("custom_prompts_active")
            or (
                self._custom_prompts_verified
                and not prompt_summary.get("oasis_default_active")
            )
        )
        meta_status = status if status is not None else self.state.status
        meta_stop = stop_reason if stop_reason is not None else self.state.stop_reason
        payload = {
            "meta": {
                "run_id": self.state.run_id,
                "scenario": self.state.scenario,
                "language": self.state.language,
                "mode": self.state.mode,
                "status": meta_status,
                "stop_reason": meta_stop,
                "llm_configured": self.state.llm_configured,
                "started_at": self.state.started_at,
                "finished_at": _utc_now(),
                "oasis": "camel-oasis==0.2.5",
                "mcp_pin": ">=1.3.0,<2",
                "phase": 8,
                "quality_fix": "8.2",
                "persona_engine": "v3",
                "execution_kind": self._execution_kind,
                "content_synthetic": self._execution_kind in {"manual", "scripted"},
                "agent_count": agents_n,
                "agent_generation": self._agent_generation,
                "offline_scale_presets": list(OFFLINE_SCALE_PRESETS),
                "live_llm_limits_unchanged": {
                    "max_agents": LIVE_TEST_MAX_AGENTS,
                    "max_rounds": LIVE_TEST_MAX_ROUNDS,
                    "max_model_calls": LIVE_TEST_MAX_LLM_CALLS,
                    "max_tokens_per_run": 20_000,
                    "max_tokens_per_call": 500,
                    "max_cost_usd_per_run": 0.20,
                },
                "agent_memory_window": self.cfg.agent_memory_window,
                "agent_local_memory_entries": self.cfg.agent_local_memory_entries,
                "agent_long_term_memory_entries": (
                    self.cfg.agent_long_term_memory_entries
                ),
                "zep_integrated": False,
                "knowledge_graph_backend": "local",
                "oasis_reply": oasis_reply_status(),
                "model_calls_per_round_required": model_calls_required_per_round(
                    agents_n
                ),
                "db_isolation": "per_run_sqlite + OASIS_DB_PATH",
                "prompt_bundle_id": PROMPT_BUNDLE_ID,
                "custom_prompts_active": custom_active,
                "custom_prompts_verified_at_build": self._custom_prompts_verified,
                "prompt_debug": prompt_summary,
                "token_optimization": {
                    "chat_history_reset_per_decision": True,
                    "prioritized_feed_memory": True,
                    "ids_retained": True,
                    "oasis_env_omitted_when_feed_present": True,
                    "follow_tool_removed": True,
                    "approval_gate": True,
                    "atomic_reserve": True,
                },
                "social_dynamics": {
                    "persistent_positions": True,
                    "local_comment_refs": True,
                    "round2_comment_reactions": True,
                    "scheduler": True,
                    "scalable_scheduler": True,
                    "relations_evidenced_only": [
                        "agrees_with",
                        "disagrees_with",
                        "responds_to",
                        "influenced_by",
                        "position_changed",
                    ],
                    "polarity_labels": [
                        "pro",
                        "contra",
                        "conditional",
                        "uncertain",
                        "unknown",
                    ],
                },
                "society_intelligence": {
                    "long_term_memory_sourced": True,
                    "relationship_dynamics": True,
                    "opinion_chronology": True,
                    "society_analytics": True,
                    "scripted_scenarios_max_rounds": OFFLINE_MAX_ROUNDS,
                    "zep_integrated": False,
                    "paid_llm_unlocked": False,
                    "quality_fix": "8.2",
                    "persona_engine": "v3",
                    "topic_aware_personas": True,
                    "individual_personas": True,
                    "synthetic_offline_content": True,
                    "cross_agent_threads": True,
                    "public_seed_no_goal_leak": True,
                    "intelligent_discussion": True,
                    "argument_history_bounded": True,
                    "argument_intelligence_v2": True,
                    "debate_engine_v2": True,
                    "stance_classification_v2": True,
                    "opinion_dynamics_v2": True,
                    "quality_dashboard": True,
                    "cost_token_planner": True,
                    "simulation_intelligence": True,
                    "multi_agent_interaction_engine": True,
                    "scheduler_waves_in_execution": True,
                },
            },
            "agents": self.state.agents,
            "posts": self.state.posts,
            "comments": self.state.comments,
            "reactions": self.state.reactions,
            "action_log": self.state.action_log,
            "decision_log": self.state.decision_log,
            "model_calls": self.state.model_calls,
            "cost_summary": self.state.cost_summary,
            "cost_plan": self.state.cost_plan,
            "agent_memory": self.state.agent_memory,
            "agent_positions": self.state.agent_positions,
            "relationships": self.state.relationships,
            "opinion_dynamics": self.state.opinion_dynamics,
            "society_analytics": self.state.society_analytics,
            "simulation_intelligence": self.state.simulation_intelligence,
            "knowledge_graph": self.state.knowledge_graph,
            "interaction_summary": self.state.interaction_summary,
            "oasis_reply": self.state.oasis_reply or oasis_reply_status(),
            "scheduler": self.state.scheduler,
            "interaction_engine": self.state.interaction_engine,
            "society_overview": self.state.society_overview,
            "discussion": self.state.discussion,
            "db_path": str(db_path),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.zep.maybe_store_summary(self.state.run_id, payload)
        self.zep.prepare_local_memory_handoff(
            self.state.run_id, payload.get("agent_memory") or {}
        )
        return out
