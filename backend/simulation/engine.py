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
    LIVE_TEST_MAX_ROUNDS,
    Settings,
    settings as default_settings,
)
from backend.simulation.agents import DEFAULT_AGENTS, AgentProfile
from backend.simulation.cost_tracker import CostTracker, LimitExceeded
from backend.simulation.manual_script import build_manual_rounds
from backend.simulation.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    build_decision_prompt,
    build_profile_for_template,
    language_display_name,
    model_calls_required_per_round,
    normalize_language,
)
from backend.simulation.tracked_model import TrackedModelBackend
from backend.zep.client import ZepMemoryStub


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
            "scenario": self.scenario,
            "language": self.language,
            "step": self.step,
            "max_steps": self.max_steps,
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
        self._agent_profiles: dict[int, AgentProfile] = {
            p.agent_id: p for p in DEFAULT_AGENTS[:LIVE_TEST_MAX_AGENTS]
        }
        self._run_language: str = "de"
        self._run_scenario: str = ""
        self._limit_stop: bool = False
        self._failed: bool = False

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

    def _new_tracker(self) -> CostTracker:
        return CostTracker(
            max_calls=min(self.cfg.max_llm_calls_per_run, LIVE_TEST_MAX_LLM_CALLS),
            max_tokens=self.cfg.max_tokens_per_run,
            price_input_per_1m=self.cfg.price_input_per_1m,
            price_output_per_1m=self.cfg.price_output_per_1m,
            max_cost_usd=self.cfg.max_cost_usd_per_run,
            model_name=self.cfg.openai_model,
            max_tokens_per_call=self.cfg.max_tokens_per_call,
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
                    llm_configured=self.cfg.llm_configured,
                    message=msg,
                    agents=[a.to_dict() for a in DEFAULT_AGENTS[:LIVE_TEST_MAX_AGENTS]],
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
                    llm_configured=self.cfg.llm_configured,
                    message=self.cfg.status_payload()["llm_mode_blocked_reason"]
                    or "LLM-Modus nicht verfügbar.",
                    agents=[a.to_dict() for a in DEFAULT_AGENTS[:LIVE_TEST_MAX_AGENTS]],
                    cost_summary={},
                )
                blocked = self.state.to_dict()
                self._emit("blocked")
                return {"ok": False, "error": self.state.message, "state": blocked}

            effective = requested

            steps = max_steps or self.cfg.max_simulation_steps
            steps = max(1, min(steps, self.cfg.max_simulation_steps, LIVE_TEST_MAX_ROUNDS))

            run_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + str(uuid.uuid4())[:8]
            )
            self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.cfg.results_dir / f"{run_id}.db"

            calls_per_round = model_calls_required_per_round(
                min(len(DEFAULT_AGENTS), LIVE_TEST_MAX_AGENTS)
            )
            if effective == "llm":
                if self._model_factory is not None:
                    message = (
                        "LLM-Modus (Test-Injection): echte OASIS LLMAction, "
                        "kein kostenpflichtiger Provider-Call. Freigabe bestätigt. "
                        f"Sprache={sim_lang}."
                    )
                else:
                    est = self.cfg.estimate_max_run_cost_usd(rounds=steps)
                    message = (
                        "LLM-Modus: echte OASIS LLMAction über OpenAI-kompatibles Modell. "
                        f"Freigabe bestätigt. Sprache={sim_lang} "
                        f"({language_display_name(sim_lang)}). "
                        f"Limits: "
                        f"{min(self.cfg.max_llm_calls_per_run, LIVE_TEST_MAX_LLM_CALLS)} Calls / "
                        f"{self.cfg.max_tokens_per_run} Tokens / "
                        f"${self.cfg.max_cost_usd_per_run:.2f} / "
                        f"{steps} Runden / retries={self.cfg.llm_max_retries} / "
                        f"~{calls_per_round} Modellaufrufe/Runde / "
                        f"geschätzte max. Kosten ~${est:.4f} "
                        "(Provider-Abrechnung kann abweichen)."
                    )
            else:
                message = (
                    "ManualAction-Modus: echte OASIS-Umgebung mit StubModel — "
                    "keine LLM-API-Calls. "
                    f"Sprache={sim_lang}. "
                    + (
                        "OPENAI_API_KEY fehlt."
                        if not self.cfg.llm_configured
                        else "Manuell gewählt."
                    )
                )

            self._tracker = self._new_tracker()
            self._limit_stop = False
            self._failed = False
            self._run_language = sim_lang
            self._run_scenario = scenario.strip()
            self.state = SimulationState(
                run_id=run_id,
                status="running",
                mode=effective,
                scenario=scenario.strip(),
                language=sim_lang,
                max_steps=steps,
                llm_configured=self.cfg.llm_configured,
                message=message,
                agents=[a.to_dict() for a in DEFAULT_AGENTS[:LIVE_TEST_MAX_AGENTS]],
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
        env = oasis.make(
            agent_graph=graph,
            platform=DefaultPlatformType.REDDIT,
            database_path=str(db_path),
        )

        self.zep.maybe_remember_session(
            self.state.run_id, scenario=scenario, mode=mode, agents=DEFAULT_AGENTS
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
        # limit_reached only if we aborted before finishing planned rounds.
        # Exhausting the call budget on the final round still counts as completed.
        rounds_finished = self.state.step >= self.state.max_steps
        early_limit_stop = bool(
            (self._limit_stop or (self._tracker and self._tracker.limit_hit))
            and not rounds_finished
        )
        if self._failed:
            status = STATUS_FAILED
            stop_reason = self.state.stop_reason or "llm_error"
            msg_prefix = self.state.message or "Simulation fehlgeschlagen."
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

        with self._lock:
            self.state.status = status
            self.state.stop_reason = stop_reason
            self.state.finished_at = _utc_now()

        result_path = self._persist_results(db_path)
        with self._lock:
            self.state.result_path = str(result_path)
            self.state.message = f"{msg_prefix} Ergebnisse: {result_path.name}."
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
            inner, self._tracker, agent_id=agent_id, source=source
        )

    def _apply_memory_window(self, agent: SocialAgent) -> None:
        """Limit chat history each agent carries into the next LLM step."""
        window = max(1, self.cfg.agent_memory_window)
        creator = agent.memory.get_context_creator()
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
            env_prompt = await agent.env.to_text_prompt()
            # Strip OASIS env_template like-bait residue from the observed blob
            # by wrapping it — our decision prompt forbids tweet/like spam.
            user_content = build_decision_prompt(
                scenario=scenario,
                goal=profile.goal,
                language=language,
                env_prompt=env_prompt,
                agent_name=profile.name,
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
                "ts": _utc_now(),
            }
            try:
                response = await agent.astep(user_msg)
                tool_calls = []
                info = getattr(response, "info", None) or {}
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
                engine._append_decision_log(
                    {
                        **decision_base,
                        "step": engine.state.step + 1,
                        "ok": True,
                        "tool_calls": tool_calls,
                        "note": "scenario+goal injected; OASIS default decision prompt bypassed",
                    }
                )
                return response
            except Exception as exc:
                engine._append_decision_log(
                    {
                        **decision_base,
                        "step": engine.state.step + 1,
                        "ok": False,
                        "error": str(exc),
                        "tool_calls": [],
                    }
                )
                raise

        agent.perform_action_by_llm = perform_action_by_llm_bound  # type: ignore[method-assign]

    def _build_agent_graph(
        self, mode: str, *, scenario: str, language: str
    ) -> AgentGraph:
        graph = AgentGraph()
        available = ActionType.get_default_reddit_actions()
        for profile in DEFAULT_AGENTS[:LIVE_TEST_MAX_AGENTS]:
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
                    description=(
                        f"{profile.description} Ziel: {profile.goal}. "
                        f"Stil: {profile.style}."
                    ),
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
            graph.add_agent(agent)
        return graph

    async def _run_manual(
        self, env, scenario: str, max_steps: int, db_path: Path
    ) -> None:
        rounds = build_manual_rounds(scenario, max_steps=max_steps)
        for idx, round_map in enumerate(rounds):
            actions: dict[SocialAgent, list[ManualAction]] = {}
            for agent_id, acts in round_map.items():
                agent = env.agent_graph.get_agent(agent_id)
                actions[agent] = acts
                profile = self._agent_profiles.get(agent_id)
                for act in acts:
                    self._append_action_log(
                        {
                            "step": idx + 1,
                            "agent_id": agent_id,
                            "action_type": act.action_type.value,
                            "action_args": act.action_args,
                            "source": "ManualAction",
                            "kind": "simulated_platform_action",
                            "goal": profile.goal if profile else None,
                            "language": self._run_language,
                            "ts": _utc_now(),
                        }
                    )
            await env.step(actions)
            with self._lock:
                self.state.step = idx + 1
            self._refresh_from_db(db_path)
            self._emit("step", {"step": idx + 1})

    async def _run_llm(
        self, env, scenario: str, max_steps: int, db_path: Path, language: str
    ) -> None:
        # Seed so agents have content to perceive (ManualAction, not billed).
        seed_agent = env.agent_graph.get_agent(0)
        seed_content = (
            f"[Seed / {DEFAULT_AGENTS[0].name}] Diskussionsimpuls ({language}): "
            f"{scenario[:500]} — Ziel: {DEFAULT_AGENTS[0].goal}"
        )
        seed = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": seed_content},
        )
        self._append_action_log(
            {
                "step": 0,
                "agent_id": 0,
                "action_type": "create_post",
                "action_args": seed.action_args,
                "source": "ManualAction",
                "kind": "simulated_platform_action",
                "note": "seed_before_llm",
                "goal": DEFAULT_AGENTS[0].goal,
                "language": language,
                "ts": _utc_now(),
            }
        )
        await env.step({seed_agent: seed})
        self._refresh_from_db(db_path)
        self._emit("step", {"step": 0})

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
            # One parallel sim only; within a round, cap agents by remaining calls
            # and live-test agent ceiling (5 × 2 ≤ 10 calls).
            agents = [
                a for _, a in env.agent_graph.get_agents()
            ][: min(remaining, LIVE_TEST_MAX_AGENTS)]
            actions = {agent: LLMAction() for agent in agents}
            for agent in agents:
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
                        },
                        "source": "LLMAction",
                        "kind": "llm_decision_request",
                        "goal": profile.goal if profile else None,
                        "language": language,
                        "ts": _utc_now(),
                    }
                )
            try:
                await env.step(actions)
            except LimitExceeded as exc:
                self._limit_stop = True
                with self._lock:
                    self.state.message = str(exc)
                    self.state.stop_reason = str(exc)
                self._sync_cost_to_state()
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
                    {"type": "like", **dict(r)}
                    for r in con.execute(
                        "SELECT like_id, user_id, post_id, created_at FROM like"
                    )
                ]
            except sqlite3.OperationalError:
                likes = []
            try:
                dislikes = [
                    {"type": "dislike", **dict(r)}
                    for r in con.execute(
                        "SELECT dislike_id, user_id, post_id, created_at FROM dislike"
                    )
                ]
            except sqlite3.OperationalError:
                dislikes = []
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
            self.state.reactions = likes + dislikes
            if merge_llm_traces and traces:
                known = {
                    (
                        e.get("agent_id"),
                        e.get("action_type"),
                        str(e.get("action_args")),
                    )
                    for e in self.state.action_log
                    if e.get("kind") == "simulated_platform_action"
                }
                for t in traces:
                    action = t.get("action")
                    if action in {"sign_up", "refresh", "search_posts"}:
                        continue
                    info = t.get("info")
                    key = (t.get("user_id"), action, str(info))
                    if key in known:
                        continue
                    # Skip duplicates already logged via decision_hook (args may
                    # differ in serialization); prefer hook entries for goals.
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

    def _persist_results(self, db_path: Path) -> Path:
        self._refresh_from_db(db_path)
        self._sync_cost_to_state()
        out = self.cfg.results_dir / f"{self.state.run_id}.json"
        agents_n = min(len(DEFAULT_AGENTS), LIVE_TEST_MAX_AGENTS)
        payload = {
            "meta": {
                "run_id": self.state.run_id,
                "scenario": self.state.scenario,
                "language": self.state.language,
                "mode": self.state.mode,
                "status": self.state.status,
                "stop_reason": self.state.stop_reason,
                "llm_configured": self.state.llm_configured,
                "started_at": self.state.started_at,
                "finished_at": _utc_now(),
                "oasis": "camel-oasis==0.2.5",
                "mcp_pin": ">=1.3.0,<2",
                "agent_memory_window": self.cfg.agent_memory_window,
                "model_calls_per_round_required": model_calls_required_per_round(
                    agents_n
                ),
                "db_isolation": "per_run_sqlite + OASIS_DB_PATH",
            },
            "agents": self.state.agents,
            "posts": self.state.posts,
            "comments": self.state.comments,
            "reactions": self.state.reactions,
            "action_log": self.state.action_log,
            "decision_log": self.state.decision_log,
            "model_calls": self.state.model_calls,
            "cost_summary": self.state.cost_summary,
            "db_path": str(db_path),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.zep.maybe_store_summary(self.state.run_id, payload)
        return out
