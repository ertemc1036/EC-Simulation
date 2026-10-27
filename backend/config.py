"""Runtime configuration from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


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


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    max_simulation_steps: int
    max_llm_calls_per_run: int
    max_tokens_per_call: int
    request_timeout_seconds: int
    allow_llm_mode: bool
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

    def status_payload(self) -> dict:
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
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "max_simulation_steps": self.max_simulation_steps,
            "max_llm_calls_per_run": self.max_llm_calls_per_run,
            "max_tokens_per_call": self.max_tokens_per_call,
            "zep_enabled": self.zep_enabled,
            "zep_configured": bool(self.zep_api_key.strip()),
            "oasis_package": "camel-oasis",
            "mcp_pin": ">=1.3.0,<2",
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
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        max_simulation_steps=_int("MAX_SIMULATION_STEPS", 8),
        max_llm_calls_per_run=_int("MAX_LLM_CALLS_PER_RUN", 40),
        max_tokens_per_call=_int("MAX_TOKENS_PER_CALL", 512),
        request_timeout_seconds=_int("REQUEST_TIMEOUT_SECONDS", 60),
        allow_llm_mode=_bool("ALLOW_LLM_MODE", True),
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
