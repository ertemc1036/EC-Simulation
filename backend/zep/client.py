"""Zep memory stub — prepared for later wiring, no mandatory network calls."""
from __future__ import annotations

import logging
from typing import Any

from backend.config import Settings
from backend.simulation.agents import AgentProfile

log = logging.getLogger("ec.zep")


class ZepMemoryStub:
    """No-op unless ZEP_ENABLED=true and ZEP_API_KEY is set.

    Phase 1 never requires Zep. When enabled later, swap `_client` for a real
    zep-cloud SDK client without changing call sites.
    """

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.zep_enabled and cfg.zep_api_key.strip())
        self._client = None
        if self.enabled:
            log.info(
                "Zep flagged enabled for project=%s url=%s — SDK not auto-imported "
                "in Phase 1; calls remain local stubs.",
                cfg.zep_project_name,
                cfg.zep_api_url,
            )

    @property
    def status(self) -> dict[str, Any]:
        return {
            "enabled_flag": self.cfg.zep_enabled,
            "api_key_present": bool(self.cfg.zep_api_key.strip()),
            "active": self.enabled,
            "api_url": self.cfg.zep_api_url,
            "project": self.cfg.zep_project_name,
            "note": (
                "Stub only — no Zep network calls in Phase 1. "
                "Install zep-cloud and set ZEP_ENABLED=true to activate later."
            ),
        }

    def maybe_remember_session(
        self,
        run_id: str,
        *,
        scenario: str,
        mode: str,
        agents: list[AgentProfile],
    ) -> None:
        if not self.enabled:
            return
        log.info(
            "zep stub remember_session run_id=%s mode=%s agents=%d",
            run_id,
            mode,
            len(agents),
        )

    def maybe_store_summary(self, run_id: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        log.info(
            "zep stub store_summary run_id=%s posts=%d",
            run_id,
            len(payload.get("posts", [])),
        )
