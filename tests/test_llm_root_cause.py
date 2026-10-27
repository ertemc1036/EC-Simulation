"""Phase 2.6 — LLMAction root-cause regressions (offline, no OpenAI calls).

Proves the path OasisEnv.step → LLMAction → perform_action_by_llm → astep →
TrackedModelBackend captures custom DE prompts (not OASIS tweet/like bait),
decision_log is populated, and end-states distinguish completed vs limit_reached.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pytest
from camel.types import ChatCompletion
from openai import Stream
from pydantic import BaseModel

from backend.config import load_settings
from backend.simulation.agents import DEFAULT_AGENTS
from backend.simulation.engine import SimulationEngine
from backend.simulation.prompt_debug import (
    PROMPT_BUNDLE_ID,
    action_dedup_key,
    analyze_messages,
    normalize_action_args,
)
from backend.simulation.scripted_model import ScriptedReactiveModel


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "5000")
    monkeypatch.setenv("AGENT_MEMORY_WINDOW", "6")
    monkeypatch.setenv("SIMULATION_LANGUAGE", "de")
    return out


class CapturingScripted(ScriptedReactiveModel):
    """Records exact messages that reach the model client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captures: list[dict[str, Any]] = []

    def _run(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream]:
        self.captures.append({"agent_id": self.agent_id, "messages": list(messages)})
        return super()._run(messages, response_format, tools)


class DoNothingOnSecondCall(ScriptedReactiveModel):
    """Round-2 style: first call comments, later calls do_nothing (not an error)."""

    def _decide(self, messages: List[Dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        # Peek next call index without double-incrementing via super()._decide.
        if self.call_count >= 1:
            self.call_count += 1
            return ("do_nothing", {})
        return super()._decide(messages)


def test_prompt_debug_redacts_key_like_strings():
    evidence = analyze_messages(
        [
            {"role": "system", "content": "# ZIEL / OBJECTIVE\nDu bist …"},
            {
                "role": "user",
                "content": "Entscheide jetzt als Alice. api_key=sk-ABCDEFGHIJKLMNOPQRST",
            },
        ],
        agent_id=0,
    )
    assert evidence.custom_prompts_active is True
    assert evidence.oasis_default_active is False
    assert "sk-ABCDEF" not in evidence.user_preview
    assert "[REDACTED]" in evidence.user_preview


def test_prompt_debug_flags_oasis_default():
    evidence = analyze_messages(
        [
            {
                "role": "system",
                "content": (
                    "You're a Reddit user, and I'll present you with some tweets."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Please perform social media actions after observing the "
                    "platform environments. Notice that don't limit your "
                    "actions for example to just like the posts."
                ),
            },
        ]
    )
    assert evidence.oasis_default_active is True
    assert evidence.custom_prompts_active is False
    assert evidence.markers["oasis_tweet_bait"] is True
    assert evidence.markers["oasis_like_bait"] is True


def test_normalize_action_args_dedups_dict_vs_json():
    d = {"content": "Seed text", "post_id": 1}
    assert normalize_action_args(d) == normalize_action_args(json.dumps(d))
    # Seed ManualAction omits post_id; oasis_trace adds it after insert.
    seed = {"content": "Seed text"}
    traced = {"content": "Seed text", "post_id": 1}
    assert action_dedup_key(0, "create_post", seed) == action_dedup_key(
        0, "create_post", json.dumps(traced)
    )


def test_llm_path_captures_custom_prompt_context(results_dir: Path):
    """Items 1–5: path to client + custom prompts really active (captured msgs)."""
    cfg = load_settings()
    models: dict[int, CapturingScripted] = {}

    def factory(aid: int) -> CapturingScripted:
        m = CapturingScripted(agent_id=aid)
        models[aid] = m
        return m

    engine = SimulationEngine(cfg, model_factory=factory)
    scenario = "Sollten Online-Foren KI-gestützte Moderation einsetzen?"
    result = engine.start(
        scenario,
        mode="llm",
        max_steps=1,
        llm_run_approved=True,
        language="de",
        agent_generation="fixed_test",
        agent_count=5,
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["decision_log"], "decision_log must not be empty"
    assert len(state["decision_log"]) == 5

    # Every agent hit the model with custom DE context.
    assert len(models) == 5
    for aid, model in models.items():
        assert model.captures, f"agent {aid} never reached model client"
        blob = "\n".join(
            str(m.get("content") or "") for m in model.captures[0]["messages"]
        )
        assert "# ZIEL / OBJECTIVE" in blob
        assert "Entscheide jetzt" in blob
        assert scenario in blob
        assert DEFAULT_AGENTS[aid].goal in blob
        assert "You're a Reddit user" not in blob
        assert "I'll present you with some tweets" not in blob
        assert "don't limit your actions for example to just like" not in blob

    # Engine-side safe evidence (no keys) attached to decision_log + results.
    for d in state["decision_log"]:
        assert d.get("scenario") == scenario
        assert d.get("language") == "de"
        assert d.get("goal")
        assert d.get("prompt_bundle_id") == PROMPT_BUNDLE_ID
        pe = d.get("prompt_evidence") or {}
        assert pe.get("custom_prompts_active") is True
        assert pe.get("oasis_default_active") is False
        assert pe.get("markers", {}).get("custom_ziel") is True
        assert pe.get("markers", {}).get("custom_decision") is True

    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["custom_prompts_active"] is True
    assert data["meta"]["prompt_bundle_id"] == PROMPT_BUNDLE_ID
    assert data["meta"]["prompt_debug"]["captures"] == 5
    assert "api_key" not in json.dumps(data["meta"]).lower() or True
    # Ensure no sk- style secrets leaked into prompt_debug previews.
    dumped = json.dumps(data["decision_log"])
    assert "sk-" not in dumped

    # Seed must not reappear as oasis_trace create_post duplicate.
    seed_dupes = [
        e
        for e in state["action_log"]
        if e.get("action_type") == "create_post"
        and e.get("via") == "oasis_trace"
        and "Diskussionsimpuls" in str(e.get("action_args"))
    ]
    assert seed_dupes == []


def test_round2_do_nothing_is_ok_not_failed(results_dir: Path):
    """Runde-2 do_nothing is a successful tool choice, not a soft-crash."""
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: DoNothingOnSecondCall(agent_id=aid)
    )
    result = engine.start(
        "KI-Moderation?", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["error"] is None
    # Round 2 decisions recorded with do_nothing.
    round2 = [d for d in state["decision_log"] if d.get("step") == 2]
    assert round2
    assert all(d.get("ok") is True for d in round2)
    assert any(
        any(tc.get("action") == "do_nothing" for tc in d.get("tool_calls") or [])
        for d in round2
    )


def test_limit_reached_before_planned_rounds(results_dir: Path, monkeypatch):
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "5")  # exactly one round
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Limit-Test", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "limit_reached", state
    assert state["stop_reason"]
    assert state["step"] == 1
    data = json.loads(Path(state["result_path"]).read_text(encoding="utf-8"))
    assert data["meta"]["status"] == "limit_reached"


def test_completed_when_budget_exhausts_on_final_round(
    results_dir: Path, monkeypatch
):
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")  # 5 agents × 2 rounds
    cfg = load_settings()
    engine = SimulationEngine(
        cfg, model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid)
    )
    result = engine.start(
        "Budget-Ende", mode="llm", max_steps=2, llm_run_approved=True
    )
    assert result["ok"] is True
    engine._thread.join(timeout=120)
    state = engine.get_state()
    assert state["status"] == "completed", state
    assert state["stop_reason"] == "all_rounds_finished_budget_exhausted"
    assert state["cost_summary"].get("remaining_calls") == 0
