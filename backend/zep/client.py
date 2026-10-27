"""Zep memory preparation — stub only, no 'Zep integrated' claim.

Phase 1–3 never require Zep network calls. Local agent memory is the active
store; this module records handoff-shaped payloads when enabled later.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.config import Settings
from backend.simulation.agents import AgentProfile

log = logging.getLogger("ec.zep")


class ZepMemoryStub:
    """No-op unless ZEP_ENABLED=true and ZEP_API_KEY is set.

    Even when the flag is on, Phase 3 does not import a Zep SDK or claim
    integration — calls stay local stubs until a later phase wires the client.
    """

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.zep_enabled and cfg.zep_api_key.strip())
        self._client = None
        self._last_handoff: dict[str, Any] | None = None
        if self.enabled:
            log.info(
                "Zep flagged enabled for project=%s url=%s — SDK not auto-imported; "
                "Phase 3 remains local-memory + stub (not integrated).",
                cfg.zep_project_name,
                cfg.zep_api_url,
            )

    @property
    def status(self) -> dict[str, Any]:
        return {
            "enabled_flag": self.cfg.zep_enabled,
            "api_key_present": bool(self.cfg.zep_api_key.strip()),
            "active": self.enabled,
            "integrated": False,
            "api_url": self.cfg.zep_api_url,
            "project": self.cfg.zep_project_name,
            "last_handoff_run_id": (
                (self._last_handoff or {}).get("run_id") if self._last_handoff else None
            ),
            "note": (
                "Zep-Vorbereitung nur — nicht integriert. "
                "Aktives Gedächtnis ist lokal (LocalAgentMemory). "
                "Später: zep-cloud + ZEP_ENABLED=true."
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

    def prepare_local_memory_handoff(
        self, run_id: str, memory_snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """Shape a future Zep payload from local memory — never sends network I/O."""
        handoff = {
            "run_id": run_id,
            "project": self.cfg.zep_project_name,
            "zep_integrated": False,
            "source": "local_agent_memory",
            "agent_count_with_memory": memory_snapshot.get("agent_count_with_memory"),
            "max_entries_per_agent": memory_snapshot.get("max_entries_per_agent"),
            "ready_for_future_sdk": True,
            "would_call_network": False,
        }
        self._last_handoff = handoff
        if self.enabled:
            log.info(
                "zep stub prepare_handoff run_id=%s agents_with_memory=%s",
                run_id,
                handoff.get("agent_count_with_memory"),
            )
        return handoff
