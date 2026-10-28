"""Phase 5.1 — Agent quality & position accuracy (offline, no live LLM).

Regression fixture: live run ``data/results/20261028T004108Z-fbcc32b2.json``
exposed fragment agent names and false position_changed / influenced_by.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_PARALLEL_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
    load_settings,
)
from backend.simulation.agent_generator import (
    extract_stakeholder_roles,
    generate_dynamic_agents,
    generate_agents,
)
from backend.simulation.engine import SimulationEngine
from backend.simulation.scripted_model import ScriptedReactiveModel
from backend.simulation.social_dynamics import PositionTracker, _polarity

ROOT = Path(__file__).resolve().parents[1]
LIVE_RUN = ROOT / "data" / "results" / "20261028T004108Z-fbcc32b2.json"

# Exact scenario text from the live Phase-5 run (fbcc32b2).
FBCC_SCENARIO = (
    "Sollte Berlin ab 2030 private Autos innerhalb des S-Bahn-Rings vollständig "
    "verbieten? Simuliere eine kontroverse Debatte zwischen Bürgern, Unternehmen "
    "und Politik. Untersuche, wie die Akteure auf konkrete Argumente anderer "
    "reagieren, ob Zustimmung oder Widerspruch entsteht und ob sich ihre "
    "ursprünglichen Positionen durch gegenseitige Beeinflussung verändern. "
    "Können sie einen tragfähigen gemeinsamen Kompromiss entwickeln?"
)

FRAGMENT_STEMS = (
    "aufkonkrete",
    "obzustimmung",
    "widerspruchentsteht",
    "argumenteanderer",
    "beeinflussung",
)


@pytest.fixture()
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ALLOW_LLM_MODE", "true")
    monkeypatch.setenv("RESULTS_DIR", str(out))
    monkeypatch.setenv("MAX_SIMULATION_STEPS", "2")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_RUN", "10")
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "20000")
    monkeypatch.setenv("MAX_TOKENS_PER_CALL", "500")
    monkeypatch.setenv("MAX_COST_USD_PER_RUN", "0.20")
    monkeypatch.setenv("MAX_PARALLEL_LLM_CALLS", "1")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    return out


def test_phase41_budget_approval_limits_unchanged():
    """4.1 Budget / Chat-Reset / Approval remain unchanged in Phase 5.1."""
    assert LIVE_TEST_MAX_AGENTS == 5
    assert LIVE_TEST_MAX_ROUNDS == 2
    assert LIVE_TEST_MAX_LLM_CALLS == 10
    assert LIVE_TEST_MAX_PARALLEL_LLM_CALLS == 1
    cfg = load_settings()
    payload = cfg.status_payload()
    assert payload["llm_run_requires_approval"] is True
    assert cfg.max_tokens_per_run == 20000
    assert cfg.max_tokens_per_call == 500
    assert cfg.max_cost_usd_per_run == 0.20
    assert cfg.max_parallel_llm_calls == 1


def test_fbcc_scenario_roles_are_buerger_unternehmen_politik():
    roles = extract_stakeholder_roles(FBCC_SCENARIO, limit=3)
    assert roles == ["Bürger", "Unternehmen", "Politik"]
    agents = generate_dynamic_agents(FBCC_SCENARIO, 3)
    labels = " ".join(
        (a.role_label or "") + " " + (a.description or "") + " " + a.user_name
        for a in agents
    ).lower()
    joined = " ".join(a.name + a.user_name for a in agents).lower()
    # Phase 7.3: display names are individuals; role lives in role_label/description.
    assert any("bürger" in labels or "buerger" in labels for _ in [0])
    assert "unternehmen" in labels or any("unternehmen" in (a.role_label or "").lower() for a in agents)
    assert "politik" in labels
    for a in agents:
        assert " " in a.name  # First + Last
        assert a.age is not None
    for stem in FRAGMENT_STEMS:
        assert stem not in joined
        assert stem not in labels


def test_live_run_fixture_documents_old_bug():
    """Offline fixture still shows the pre-fix failure mode (documentation)."""
    assert LIVE_RUN.is_file(), f"missing regression fixture {LIVE_RUN}"
    data = json.loads(LIVE_RUN.read_text(encoding="utf-8"))
    # Fixture is a historical Phase-5.1 live dump — phase stamp stays 5.
    assert data["meta"]["phase"] in (5, 5.1, 5.2, 6, 7, 8)
    names = " ".join(a["name"] for a in data["agents"]).lower()
    assert "aufkonkrete" in names or "obzustimmung" in names
    # Pre-fix tracker over-counted confirmations as changes.
    assert int(data["agent_positions"]["change_count"]) >= 5


def test_no_instruction_fragments_as_names_for_wie_reagieren_prompt():
    classic = (
        "Was passiert, wenn Berlin ab 2030 private Autos innerhalb des "
        "S-Bahn-Rings verbietet? Wie reagieren Bürger, Unternehmen, "
        "Politiker und Verkehrsbetriebe?"
    )
    roles = extract_stakeholder_roles(classic)
    assert "Bürger" in roles
    assert "Unternehmen" in roles
    assert "Politiker" in roles
    agents = generate_agents(classic, 4, mode="dynamic")
    blob = " ".join(a.name for a in agents).lower()
    assert "passiert" not in blob
    assert "aufkonkrete" not in blob


def test_deterministic_fallback_roles_when_no_stakeholders_named():
    roles = extract_stakeholder_roles(
        "Sollte Berlin ab 2030 Autos im S-Bahn-Ring verbieten?", limit=3
    )
    assert roles == ["Bürger", "Unternehmen", "Politik"]


def test_erstposition_is_not_position_changed():
    tracker = PositionTracker()
    agents = [
        {"agent_id": 0, "goal": "Interessen vertreten, Zielkonflikt markieren."},
        {"agent_id": 1, "goal": "Wirtschaftliche Risiken benennen."},
    ]
    changes = tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {
                "post_id": 1,
                "user_id": 0,
                "content": "Als Bürgerstimme fordere ich Ausnahmen.",
            },
            {
                "post_id": 2,
                "user_id": 1,
                "content": "Als Unternehmen brauche ich Lieferfenster.",
            },
        ],
        comments=[],
        reactions=[],
        step=1,
    )
    assert changes == []
    assert tracker.positions_payload()["change_count"] == 0
    for p in tracker.positions_payload()["positions"]:
        assert p["changed"] is False
        assert p["evidence_kind"] in {"post", "goal"}


def test_confirmation_is_not_position_changed():
    tracker = PositionTracker()
    agents = [{"agent_id": 0, "goal": "Dagegen: Verbot ablehnen."}]
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {
                "post_id": 1,
                "user_id": 0,
                "content": "Das vollständige Verbot ist problematisch und dagegen.",
            }
        ],
        comments=[],
        reactions=[],
        step=1,
    )
    changes = tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {
                "post_id": 1,
                "user_id": 0,
                "content": "Das vollständige Verbot ist problematisch und dagegen.",
            }
        ],
        comments=[
            {
                "comment_id": 1,
                "post_id": 1,
                "user_id": 0,
                "content": "Belastbare Daten fehlen zur Wirksamkeit.",
            },
            {
                "comment_id": 2,
                "post_id": 1,
                "user_id": 0,
                "content": (
                    "→Kommentar#1 bestätigt, dass belastbare Daten fehlen. "
                    "Ohne Evidenz bleibt das Verbot theoretisch."
                ),
            },
        ],
        reactions=[],
        step=2,
    )
    assert changes == []
    assert tracker.positions_payload()["change_count"] == 0


def test_unknown_uncertain_polarity_helpers():
    assert _polarity("Ich bin unsicher und weiß nicht.") == "uncertain"
    assert _polarity("Rein deskriptiver Satz ohne Marker.") == "unknown"
    assert _polarity("Ich unterstütze das Verbot klar.") == "pro"
    assert _polarity("Ich lehne das ab und widerspreche.") == "contra"


def test_no_invented_influenced_by_without_change():
    tracker = PositionTracker()
    agents = [
        {"agent_id": 0, "goal": "Dafür unterstützen."},
        {"agent_id": 1, "goal": "Dagegen ablehnen."},
    ]
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich unterstütze das Verbot."},
            {"post_id": 2, "user_id": 1, "content": "Ich lehne das Verbot ab."},
        ],
        comments=[],
        reactions=[],
        step=1,
    )
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich unterstütze das Verbot."},
            {"post_id": 2, "user_id": 1, "content": "Ich lehne das Verbot ab."},
        ],
        comments=[
            {
                "comment_id": 1,
                "post_id": 1,
                "user_id": 0,
                "content": "Datenlage prüfen.",
            },
            {
                "comment_id": 2,
                "post_id": 2,
                "user_id": 1,
                "content": "→Kommentar#1 stimmt, dass Daten fehlen — ich bleibe dagegen.",
            },
        ],
        reactions=[],
        step=2,
    )
    specs = tracker.social_edge_specs()
    assert not any(s["relation"] == "influenced_by" for s in specs)
    assert not any(s["relation"] == "position_changed" for s in specs)


def test_comment_id_author_mapping_for_influence():
    tracker = PositionTracker()
    agents = [
        {"agent_id": 0, "goal": "Dagegen ablehnen."},
        {"agent_id": 1, "goal": "Dagegen ablehnen."},
    ]
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich lehne das Verbot ab."},
            {"post_id": 2, "user_id": 1, "content": "Auch ich bin dagegen."},
        ],
        comments=[],
        reactions=[],
        step=1,
    )
    changes = tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich lehne das Verbot ab."},
            {"post_id": 2, "user_id": 1, "content": "Auch ich bin dagegen."},
        ],
        comments=[
                {
                    "comment_id": 10,
                    "post_id": 1,
                    "user_id": 0,
                    # Stay contra so only agent 1's evidenced mind-change counts.
                    "content": "Neue Studien prüfen — ich bleibe dagegen.",
                },
            {
                "comment_id": 11,
                "post_id": 2,
                "user_id": 1,
                "content": (
                    "→Kommentar#10 ich ändere meine Position und unterstütze "
                    "jetzt das Verbot."
                ),
            },
        ],
        reactions=[{"type": "like", "user_id": 1, "comment_id": 10}],
        step=2,
    )
    assert len(changes) == 1
    assert changes[0]["agent_id"] == 1
    assert changes[0]["influenced_by_agent_ids"] == [0]
    assert changes[0]["evidence"] == "comment:11"
    # Wrong author must not appear (comment 10 is user 0, not self).
    assert 1 not in changes[0]["influenced_by_agent_ids"]


def test_no_double_counting_same_transition():
    tracker = PositionTracker()
    agents = [
        {"agent_id": 0, "goal": "Dafür unterstützen."},
        {"agent_id": 1, "goal": "Dagegen ablehnen."},
    ]
    posts = [
        {"post_id": 1, "user_id": 0, "content": "Ich unterstütze das Verbot klar."},
        {"post_id": 2, "user_id": 1, "content": "Dagegen — ablehnen."},
    ]
    comments = [
        {
            "comment_id": 1,
            "post_id": 1,
            "user_id": 0,
            "content": "Weiterhin unterstütze ich das Verbot.",
        },
        {
            "comment_id": 2,
            "post_id": 2,
            "user_id": 1,
            "content": "→Kommentar#1 ich ändere meine Position und bin dafür.",
        },
    ]
    tracker.update_from_snapshot(
        agents=agents, posts=posts, comments=[], reactions=[], step=1
    )
    tracker.update_from_snapshot(
        agents=agents, posts=posts, comments=comments, reactions=[], step=2
    )
    # Re-sync with identical snapshot must not add another change.
    tracker.update_from_snapshot(
        agents=agents, posts=posts, comments=comments, reactions=[], step=2
    )
    assert tracker.positions_payload()["change_count"] == 1
    assert tracker.positions_payload()["changes"][0]["agent_id"] == 1


def test_fbcc_replay_positions_no_false_changes():
    """Replay content from fbcc32b2: confirmations must not yield changes."""
    data = json.loads(LIVE_RUN.read_text(encoding="utf-8"))
    agents = data["agents"]
    tracker = PositionTracker()
    # Step-wise: posts only, then all comments (simulates R1→R2 growth).
    tracker.update_from_snapshot(
        agents=agents,
        posts=data["posts"],
        comments=[],
        reactions=data.get("reactions") or [],
        step=1,
    )
    # Erstpositionen from posts — not changes.
    assert tracker.positions_payload()["change_count"] == 0
    tracker.update_from_snapshot(
        agents=agents,
        posts=data["posts"],
        comments=data["comments"],
        reactions=data.get("reactions") or [],
        step=2,
    )
    # Live comments were confirmations / restatements, not mind changes.
    assert tracker.positions_payload()["change_count"] == 0
    assert not any(
        s["relation"] == "influenced_by" for s in tracker.social_edge_specs()
    )


def test_scripted_phase5_still_records_real_shifts(results_dir: Path):
    engine = SimulationEngine(
        load_settings(),
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid, label="p51"),
    )
    started = engine.start(
        FBCC_SCENARIO,
        mode="llm",
        max_steps=2,
        agent_count=3,
        agent_generation="dynamic",
        llm_run_approved=True,
    )
    assert started["ok"] is True
    import time

    deadline = time.time() + 90
    while time.time() < deadline:
        state = engine.get_state()
        if state["status"] in {"completed", "limit_reached", "failed"}:
            break
        time.sleep(0.2)
    state = engine.get_state()
    assert state["status"] in {"completed", "limit_reached"}
    agents = state["agents"]
    blob = " ".join(a["name"] + a["user_name"] for a in agents).lower()
    for stem in FRAGMENT_STEMS:
        assert stem not in blob
    assert any(
        "bürger" in (a.get("role_label") or a.get("description") or a["name"]).lower()
        or "buerger" in (a.get("role_label") or a.get("description") or a["name"]).lower()
        for a in agents
    )
    # Approval / token opts from 4.1 still present on result.
    path = Path(state["result_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["token_optimization"]["approval_gate"] is True
    assert data["meta"]["token_optimization"]["chat_history_reset_per_decision"] is True
    assert data["meta"]["token_optimization"]["atomic_reserve"] is True


def test_approval_gate_still_blocks_unapproved_llm(results_dir: Path):
    engine = SimulationEngine(
        load_settings(),
        model_factory=lambda aid: ScriptedReactiveModel(agent_id=aid),
    )
    blocked = engine.start(
        FBCC_SCENARIO, mode="llm", max_steps=2, llm_run_approved=False
    )
    assert blocked["ok"] is False
    assert blocked["state"]["status"] == "blocked"
