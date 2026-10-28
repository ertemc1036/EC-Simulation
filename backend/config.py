"""Runtime configuration from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Absolute live-test ceilings — env may be lower, never higher.
# Phase 6–7: these stay fixed for paid Live-LLM; 10/25/50 is Offline/Scripted only.
LIVE_TEST_MAX_AGENTS = 5
LIVE_TEST_MAX_ROUNDS = 2
LIVE_TEST_MAX_LLM_CALLS = 10
LIVE_TEST_MAX_RETRIES = 0
# Concurrent in-flight LLM calls inside one round (OASIS gather + semaphore).
# Default 1: serialize model calls so token reservations cannot race.
LIVE_TEST_MAX_PARALLEL_LLM_CALLS = 1

# Offline / Scripted society scale (Phase 6+) — never unlocks paid Live-LLM.
OFFLINE_MAX_AGENTS = 50
OFFLINE_SCALE_PRESETS = (10, 25, 50)
# Manual/scripted may use more rounds than live; live still clamps to 2.
OFFLINE_MAX_ROUNDS = 4
# Phase 7 long-term local memory cap (sourced entries; not Zep).
DEFAULT_LONG_TERM_MEMORY_ENTRIES = 8

COST_ESTIMATE_DISCLAIMER = (
    "Kostenschätzung auf Basis der konfigurierten USD/1M-Preise; "
    "die Provider-Abrechnung kann abweichen."
)


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


_KNOWN_SIM_LANGUAGES = frozenset({"de", "en", "fr", "es", "it"})


def _normalize_sim_language(code: str | None) -> str:
    """Default German; keep independent of simulation.prompts (avoid import cycles)."""
    raw = (code or "de").strip().lower()
    if not raw:
        return "de"
    short = raw.split("-", 1)[0]
    return short if short in _KNOWN_SIM_LANGUAGES else "de"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    max_simulation_steps: int
    max_llm_calls_per_run: int
    max_tokens_per_call: int
    max_tokens_per_run: int
    max_cost_usd_per_run: float
    max_parallel_llm_calls: int
    llm_max_retries: int
    agent_memory_window: int
    agent_local_memory_entries: int
    agent_long_term_memory_entries: int
    max_agents: int
    offline_max_agents: int
    default_agent_count: int
    default_agent_generation: str
    price_input_per_1m: float
    price_output_per_1m: float
    request_timeout_seconds: int
    allow_llm_mode: bool
    simulation_language: str
    flask_host: str
    flask_port: int
    flask_debug: bool
    cors_origins: list[str]
    zep_enabled: bool
    zep_api_key: str
    zep_api_url: str
    zep_project_name: str
    results_dir: Path
    root: Path

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def llm_mode_available(self) -> bool:
        return self.allow_llm_mode and self.llm_configured

    def estimate_max_run_cost_usd(self, *, rounds: int | None = None) -> float:
        """Conservative upper bound for a full LLM run (may exceed provider bill)."""
        steps = rounds if rounds is not None else self.max_simulation_steps
        steps = max(1, min(steps, self.max_simulation_steps, LIVE_TEST_MAX_ROUNDS))
        # Budget: one model call per agent per round, capped by call limit.
        planned_calls = min(
            self.max_agents * steps,
            self.max_llm_calls_per_run,
            LIVE_TEST_MAX_LLM_CALLS,
        )
        # Conservative: each call may use max_tokens_per_call for BOTH sides.
        per_call = (
            self.max_tokens_per_call / 1_000_000.0 * self.price_input_per_1m
            + self.max_tokens_per_call / 1_000_000.0 * self.price_output_per_1m
        )
        estimated = planned_calls * per_call
        # Soft display ceiling: never advertise above configured hard cost gate.
        if self.max_cost_usd_per_run > 0:
            return round(min(estimated, self.max_cost_usd_per_run), 6)
        return round(estimated, 6)

    def status_payload(self) -> dict:
        """Public status — never includes API keys."""
        return {
            "llm_configured": self.llm_configured,
            "llm_mode_available": self.llm_mode_available,
            "llm_mode_blocked_reason": (
                None
                if self.llm_mode_available
                else (
                    "OPENAI_API_KEY fehlt — Simulation läuft nur im ManualAction-Modus "
                    "(echte OASIS-Umgebung mit StubModel, keine LLM-Calls)."
                    if not self.llm_configured
                    else "ALLOW_LLM_MODE=false"
                )
            ),
            "llm_run_requires_approval": True,
            "default_mode": "manual",
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "max_agents": self.max_agents,
            "offline_max_agents": self.offline_max_agents,
            "default_agent_count": self.default_agent_count,
            "default_agent_generation": self.default_agent_generation,
            "agent_generation_modes": ["dynamic", "fixed_test"],
            "agent_local_memory_entries": self.agent_local_memory_entries,
            "agent_long_term_memory_entries": self.agent_long_term_memory_entries,
            "max_simulation_steps": self.max_simulation_steps,
            "max_llm_calls_per_run": self.max_llm_calls_per_run,
            "max_tokens_per_call": self.max_tokens_per_call,
            "max_tokens_per_run": self.max_tokens_per_run,
            "max_cost_usd_per_run": self.max_cost_usd_per_run,
            "max_parallel_llm_calls": self.max_parallel_llm_calls,
            "estimated_max_cost_usd": self.estimate_max_run_cost_usd(),
            "cost_estimate_disclaimer": COST_ESTIMATE_DISCLAIMER,
            "llm_max_retries": self.llm_max_retries,
            "agent_memory_window": self.agent_memory_window,
            "simulation_language": self.simulation_language,
            "price_input_per_1m_usd": self.price_input_per_1m,
            "price_output_per_1m_usd": self.price_output_per_1m,
            # OASIS LLMAction: 1 model call per agent per round (astep once).
            "model_calls_per_round_required": self.max_agents,
            "model_calls_budget_note": (
                f"{self.max_agents} Agenten × Runden ≤ "
                f"MAX_LLM_CALLS_PER_RUN ({self.max_llm_calls_per_run}); "
                "Seed-ManualAction kostet 0 Calls. "
                f"Offline/Scripted Society: bis {self.offline_max_agents} Agenten "
                f"(Presets {list(OFFLINE_SCALE_PRESETS)}) — nicht für Live-LLM."
            ),
            "live_test_limits": {
                "max_agents": LIVE_TEST_MAX_AGENTS,
                "max_rounds": LIVE_TEST_MAX_ROUNDS,
                "max_model_calls": LIVE_TEST_MAX_LLM_CALLS,
                "max_retries": LIVE_TEST_MAX_RETRIES,
                "max_parallel_llm_calls": LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
                "max_tokens_per_run": 20_000,
                "max_tokens_per_call": 500,
                "max_cost_usd_per_run": 0.20,
            },
            "offline_scale": {
                "max_agents": OFFLINE_MAX_AGENTS,
                "presets": list(OFFLINE_SCALE_PRESETS),
                "max_rounds": OFFLINE_MAX_ROUNDS,
                "paid_llm_unlocked": False,
                "note": (
                    "10/25/50 nur Manual/Scripted offline — "
                    "Live-LLM-Ceilings bleiben 5/2/10/20k/500/$0.20."
                ),
            },
            "terminal_statuses": ["completed", "limit_reached", "failed"],
            "zep_enabled": self.zep_enabled,
            "zep_configured": bool(self.zep_api_key.strip()),
            "zep_integrated": False,
            "knowledge_graph_backend": "local",
            "oasis_package": "camel-oasis",
            "mcp_pin": ">=1.3.0,<2",
            "api_key_exposed": False,
            "phase": "7",
            "oasis_reply": {
                "parent_comment_id_supported": False,
                "reply_via": "create_comment(post_id, content)",
                "documented_alternative": (
                    "Reply = CREATE_COMMENT on post_id; nested comment→comment "
                    "not supported — use local →Kommentar#id. "
                    "Agreement/disagreement = like_*/dislike_* without quotas. "
                    "Round 2+: prefer like_comment/dislike_comment."
                ),
            },
            "social_dynamics": {
                "persistent_positions": True,
                "relations_evidenced_only": [
                    "agrees_with",
                    "disagrees_with",
                    "responds_to",
                    "influenced_by",
                    "position_changed",
                ],
                "scheduler": True,
                "offline_scale_presets": list(OFFLINE_SCALE_PRESETS),
                "long_term_memory": True,
                "relationship_dynamics": True,
                "opinion_chronology": True,
                "society_analytics": True,
            },
            "society_intelligence": {
                "long_term_memory_sourced": True,
                "relationship_dynamics_evidenced_only": True,
                "opinion_chronology": True,
                "scripted_scenarios_max_rounds": OFFLINE_MAX_ROUNDS,
                "analytics_dashboard": True,
                "zep_integrated": False,
            },
        }


def load_settings() -> Settings:
    results = Path(os.getenv("RESULTS_DIR", "data/results"))
    if not results.is_absolute():
        results = ROOT / results
    origins = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
        ).split(",")
        if o.strip()
    ]
    # Clamp to live-test ceilings so a loose .env cannot widen the first run.
    max_steps = min(
        max(1, _int("MAX_SIMULATION_STEPS", LIVE_TEST_MAX_ROUNDS)),
        LIVE_TEST_MAX_ROUNDS,
    )
    max_calls = min(
        max(1, _int("MAX_LLM_CALLS_PER_RUN", LIVE_TEST_MAX_LLM_CALLS)),
        LIVE_TEST_MAX_LLM_CALLS,
    )
    max_retries = min(
        max(0, _int("LLM_MAX_RETRIES", LIVE_TEST_MAX_RETRIES)),
        LIVE_TEST_MAX_RETRIES,
    )
    # Parallel LLM calls: env may lower further; never above live-test ceiling.
    max_parallel = min(
        max(1, _int("MAX_PARALLEL_LLM_CALLS", LIVE_TEST_MAX_PARALLEL_LLM_CALLS)),
        LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
        LIVE_TEST_MAX_AGENTS,
    )
    # Default German — live run showed English drift from OASIS defaults.
    sim_lang = _normalize_sim_language(os.getenv("SIMULATION_LANGUAGE", "de"))

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        max_simulation_steps=max_steps,
        max_llm_calls_per_run=max_calls,
        max_tokens_per_call=max(1, _int("MAX_TOKENS_PER_CALL", 500)),
        max_tokens_per_run=max(1, _int("MAX_TOKENS_PER_RUN", 20000)),
        max_cost_usd_per_run=_float("MAX_COST_USD_PER_RUN", 0.20),
        max_parallel_llm_calls=max_parallel,
        llm_max_retries=max_retries,
        agent_memory_window=_int("AGENT_MEMORY_WINDOW", 6),
        agent_local_memory_entries=max(1, _int("AGENT_LOCAL_MEMORY_ENTRIES", 12)),
        agent_long_term_memory_entries=max(
            1,
            _int(
                "AGENT_LONG_TERM_MEMORY_ENTRIES",
                DEFAULT_LONG_TERM_MEMORY_ENTRIES,
            ),
        ),
        # max_agents = Live-LLM ceiling (unchanged). Offline scale is separate.
        max_agents=LIVE_TEST_MAX_AGENTS,
        offline_max_agents=OFFLINE_MAX_AGENTS,
        default_agent_count=min(
            max(1, _int("DEFAULT_AGENT_COUNT", LIVE_TEST_MAX_AGENTS)),
            LIVE_TEST_MAX_AGENTS,
        ),
        default_agent_generation=(
            os.getenv("AGENT_GENERATION_MODE", "dynamic").strip().lower() or "dynamic"
        ),
        price_input_per_1m=_float("LLM_PRICE_INPUT_PER_1M", 0.40),
        price_output_per_1m=_float("LLM_PRICE_OUTPUT_PER_1M", 1.60),
        request_timeout_seconds=_int("REQUEST_TIMEOUT_SECONDS", 60),
        allow_llm_mode=_bool("ALLOW_LLM_MODE", True),
        simulation_language=sim_lang,
        flask_host=os.getenv("FLASK_HOST", "127.0.0.1"),
        flask_port=_int("FLASK_PORT", 5001),
        flask_debug=_bool("FLASK_DEBUG", False),
        cors_origins=origins,
        zep_enabled=_bool("ZEP_ENABLED", False),
        zep_api_key=os.getenv("ZEP_API_KEY", ""),
        zep_api_url=os.getenv("ZEP_API_URL", "https://api.getzep.com"),
        zep_project_name=os.getenv("ZEP_PROJECT_NAME", "ec-simulation"),
        results_dir=results,
        root=ROOT,
    )


settings = load_settings()
