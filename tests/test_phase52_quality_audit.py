"""Phase 5.2 — Live-run quality audit (offline, no live LLM).

Regression fixture: ``data/results/20261028T081810Z-7f75d2e4.json``
exposed false position_changed (Seed contra→pro via „Unterstützung“),
Bürgerstimme unknown, Agent1/2 false pro, and agrees/disagrees=0 despite
evidenced textual Zustimmung/Widerspruch on →Kommentar# refs.
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
from backend.simulation.knowledge_graph import LocalKnowledgeGraph
from backend.simulation.social_dynamics import (
    PositionTracker,
    _is_seed_content,
    _polarity,
    classify_textual_agree_disagree,
    collect_textual_agree_disagree,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE_RUN = ROOT / "data" / "results" / "20261028T081810Z-7f75d2e4.json"
# Exact scenario from the Phase-5.2 live run.
RUN_SCENARIO = (
    "Berlin plant, ab 2030 private Autos innerhalb des S-Bahn-Rings vollständig "
    "zu verbieten. Drei Interessengruppen diskutieren die Entscheidung: "
    "Bürgerinnen und Bürger, Unternehmen sowie die Politik. Welche "
    "wirtschaftlichen, sozialen und politischen Konflikte entstehen? Wie "
    "reagieren die Gruppen auf konkrete Argumente der anderen? Können sie einen "
    "gemeinsamen Kompromiss entwickeln? Untersuche, ob sich ihre ursprünglichen "
    "Positionen tatsächlich verändern, und erfasse Meinungsänderungen nur bei "
    "nachvollziehbarer Begründung."
)


@pytest.fixture()
def run_data() -> dict:
    assert LIVE_RUN.is_file(), f"missing regression fixture {LIVE_RUN}"
    return json.loads(LIVE_RUN.read_text(encoding="utf-8"))


def test_limits_and_approval_unchanged():
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


def test_live_fixture_documents_old_bugs(run_data: dict):
    """Offline fixture still shows the pre-fix failure mode."""
    assert run_data["meta"]["run_id"] == "20261028T081810Z-7f75d2e4"
    assert int(run_data["interaction_summary"]["position_changed"]) == 1
    assert int(run_data["interaction_summary"]["agrees_with"]) == 0
    assert int(run_data["interaction_summary"]["disagrees_with"]) == 0
    assert int(run_data["interaction_summary"]["responds_to"]) == 5
    by_id = {p["agent_id"]: p for p in run_data["agent_positions"]["positions"]}
    assert by_id[0]["polarity"] == "unknown"
    assert by_id[1]["polarity"] == "pro"
    assert by_id[2]["polarity"] == "pro"
    ch = run_data["agent_positions"]["changes"][0]
    assert ch["from_polarity"] == "contra" and ch["to_polarity"] == "pro"


def test_seed_content_is_not_stance():
    seed = (
        "[Seed / Bürgerstimme] Diskussionsimpuls (de): Berlin plant … "
        "— Ziel: widersprich Wirtschaftsargumenten ohne Alltagsbezug."
    )
    assert _is_seed_content(seed) is True
    assert _polarity(seed) == "unknown"
    assert _is_seed_content("Als Bürgerstimme fordere ich Ausnahmen.") is False


def test_polarity_exception_vs_full_ban_not_false_pro():
    # Live comment#1: "finanzielle Unterstützung" must not flip to pro.
    c1 = (
        "Als Bürgerstimme möchte ich betonen, dass das geplante Auto-Verbot "
        "enorme Auswirkungen hat. Daher fordere ich dringend soziale "
        "Abfederungen wie Ausnahmen für mobilitätseingeschränkte Menschen, "
        "finanzielle Unterstützung für Betroffene."
    )
    assert _polarity(c1) in {"conditional", "exception"}
    # Live comment#3: Ausnahme vs. volles Verbot, not unknown.
    c3 = (
        "→Kommentar#2: Ich verstehe die wirtschaftlichen Bedenken, aber für "
        "viele Anwohnende bedeutet das Verbot eine erhebliche Einschränkung. "
        "Wir brauchen daher klare Ausnahmeregelungen für "
        "mobilitätseingeschränkte Menschen und eine soziale Abfederung."
    )
    assert _polarity(c3) in {"conditional", "exception"}
    # Live Unternehmens-Post: Sorge + Lieferfenster, not pro via „unterstützen wir das Ziel“.
    firm = (
        "Als Unternehmensvertretung sehen wir die geplante vollständige "
        "Sperrung mit Sorge. Zwar unterstützen wir das Ziel, die Stadt "
        "umweltfreundlicher zu gestalten. Wir fordern Lieferfenster, "
        "Ausnahmen und Übergangsfristen. Pauschale Verbote ohne Logistik "
        "gefährden Arbeitsplätze."
    )
    assert _polarity(firm) in {"conditional", "exception"}
    # Clear pro/contra still work.
    assert _polarity("Ich unterstütze das Verbot klar.") == "pro"
    assert _polarity("Ich lehne das Verbot ab — dagegen.") == "contra"


def test_textual_agree_disagree_markers_only_with_evidence():
    assert classify_textual_agree_disagree("nur eine Antwort ohne Marker") == []
    labels = classify_textual_agree_disagree(
        "→Kommentar#1 Danke für Ihren Hinweis auf die sozialen Vorteile. "
        "Allerdings müssen wir sicherstellen, dass Unternehmen nicht "
        "unverhältnismäßig belastet werden."
    )
    assert "agrees_with" in labels
    assert "disagrees_with" in labels
    # No invent without comment ref target in collector.
    specs = collect_textual_agree_disagree(
        [
            {
                "comment_id": 4,
                "user_id": 1,
                "post_id": 1,
                "content": (
                    "Danke für Ihren Hinweis — allerdings müssen wir "
                    "Lieferfenster sichern."
                ),
            }
        ],
        comment_statement_ids={1: "statement:comment:1"},
        agent_nodes={1: "actor:1"},
    )
    assert specs == []  # no →Kommentar# ref


def test_7f75_replay_no_false_position_changed(run_data: dict):
    """Chronological replay of the live run: no artificial Seed→pro switch."""
    agents = run_data["agents"]
    tracker = PositionTracker()
    posts: list[dict] = []
    comments: list[dict] = []
    post_id = 0
    comment_id = 0
    for entry in run_data["action_log"]:
        at = entry.get("action_type")
        args = entry.get("action_args") or {}
        aid = entry.get("agent_id")
        step = entry.get("step")
        if at == "create_post":
            post_id += 1
            posts.append(
                {
                    "post_id": post_id,
                    "user_id": aid,
                    "content": args.get("content") or "",
                }
            )
            tracker.update_from_snapshot(
                agents=agents,
                posts=posts,
                comments=comments,
                reactions=[],
                step=step,
            )
        elif at == "create_comment":
            comment_id += 1
            comments.append(
                {
                    "comment_id": comment_id,
                    "user_id": aid,
                    "post_id": args.get("post_id") or 1,
                    "content": args.get("content") or "",
                }
            )
            tracker.update_from_snapshot(
                agents=agents,
                posts=posts,
                comments=comments,
                reactions=[],
                step=step,
            )

    payload = tracker.positions_payload()
    assert payload["change_count"] == 0
    assert payload["changes"] == []
    by_id = {p["agent_id"]: p for p in payload["positions"]}
    # Bürgerstimme / Unternehmen: Ausnahme vs. volles Verbot — stable.
    assert by_id[0]["polarity"] in {"conditional", "exception"}
    assert by_id[1]["polarity"] in {"conditional", "exception"}
    # Politik: mediating / uncertain — not false pro.
    assert by_id[2]["polarity"] in {"uncertain", "conditional", "exception", "unknown"}
    assert by_id[2]["polarity"] != "pro"
    assert not any(
        s["relation"] == "position_changed" for s in tracker.social_edge_specs()
    )


def test_7f75_textual_relations_evidenced(run_data: dict):
    """Comment#4 thanks + allerdings → agrees_with and disagrees_with."""
    kg = LocalKnowledgeGraph()
    kg.rebuild(
        scenario=RUN_SCENARIO,
        agents=run_data["agents"],
        posts=run_data["posts"],
        comments=run_data["comments"],
        reactions=run_data.get("reactions") or [],
        action_log=run_data.get("action_log") or [],
    )
    data = kg.to_dict()
    sr = data["social_relations"]
    assert sr["responds_to"] >= 1
    assert sr["agrees_with"] >= 1
    assert sr["disagrees_with"] >= 1
    assert sr.get("position_changed", 0) == 0
    # Edges must point at the referenced comment statement, not invent actors.
    agree_edges = [e for e in data["edges"] if e["relation"] == "agrees_with"]
    disagree_edges = [e for e in data["edges"] if e["relation"] == "disagrees_with"]
    assert any(
        e["source"] == "actor:1" and e["target"] == "statement:comment:1"
        for e in agree_edges
    )
    assert any(
        e["source"] == "actor:1" and e["target"] == "statement:comment:1"
        for e in disagree_edges
    )
    for e in agree_edges + disagree_edges:
        assert e["meta"].get("evidence") in {
            "textual_comment_ref",
            "like",
            "dislike",
            "like_post",
            "dislike_post",
            "like_comment",
            "dislike_comment",
            "action_log:like_post",
            "action_log:dislike_post",
            "action_log:like_comment",
            "action_log:dislike_comment",
        } or str(e["meta"].get("evidence", "")).startswith("action_log:")


def test_no_artificial_exception_to_pro_without_shift():
    tracker = PositionTracker()
    agents = [{"agent_id": 0, "goal": "Fordere Ausnahmen und Abfederung."}]
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {
                "post_id": 1,
                "user_id": 0,
                "content": "Wir brauchen Ausnahmen für mobilitätseingeschränkte Menschen.",
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
                "content": "Wir brauchen Ausnahmen für mobilitätseingeschränkte Menschen.",
            }
        ],
        comments=[
            {
                "comment_id": 1,
                "post_id": 1,
                "user_id": 0,
                "content": (
                    "Finanzielle Unterstützung und soziale Abfederung bleiben "
                    "zentral — Ausnahmeregelungen jetzt."
                ),
            }
        ],
        reactions=[],
        step=2,
    )
    assert changes == []
    assert tracker.positions_payload()["positions"][0]["polarity"] in {
        "conditional",
        "exception",
    }


def test_real_shift_still_recorded():
    tracker = PositionTracker()
    agents = [
        {"agent_id": 0, "goal": "Dafür: Autoverbot unterstützen."},
        {"agent_id": 1, "goal": "Dagegen: Autoverbot ablehnen."},
    ]
    tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich unterstütze das Verbot klar."},
            {"post_id": 2, "user_id": 1, "content": "Ich lehne das Verbot ab — dagegen."},
        ],
        comments=[],
        reactions=[],
        step=1,
    )
    changes = tracker.update_from_snapshot(
        agents=agents,
        posts=[
            {"post_id": 1, "user_id": 0, "content": "Ich unterstütze das Verbot klar."},
            {"post_id": 2, "user_id": 1, "content": "Ich lehne das Verbot ab — dagegen."},
        ],
        comments=[
            {
                "comment_id": 1,
                "post_id": 1,
                "user_id": 0,
                "content": "Luftqualität im Ring.",
            },
            {
                "comment_id": 2,
                "post_id": 1,
                "user_id": 1,
                "content": (
                    "→Kommentar#1 ich ändere meine Position und unterstütze "
                    "jetzt das Verbot."
                ),
            },
        ],
        reactions=[{"type": "like", "user_id": 1, "comment_id": 1}],
        step=2,
    )
    assert len(changes) == 1
    assert changes[0]["agent_id"] == 1
    assert changes[0]["from_polarity"] == "contra"
    assert changes[0]["to_polarity"] == "pro"
