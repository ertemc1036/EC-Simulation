"""NanoCrab Lab research MVP — offline fixtures, no provider calls."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from backend.app import app
from backend.config import (
    LIVE_TEST_MAX_AGENTS,
    LIVE_TEST_MAX_LLM_CALLS,
    LIVE_TEST_MAX_ROUNDS,
)
from backend.research import integrity
from backend.research.routes import set_service
from backend.research.service import ResearchService

EXAMPLES = ("saas-pricing", "banking-onboarding", "restaurant-software")
KINDS = {
    "user_input",
    "source",
    "synthetic_statement",
    "interpretation",
    "human_validation_need",
}


@pytest.fixture()
def research_client(tmp_path: Path):
    service = ResearchService(tmp_path / "store.json", seed_examples=True)
    set_service(service)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, service
    set_service(None)


def test_banned_detector_catches_percent_and_spares_disclaimer():
    assert integrity.find_banned_claims("about 12% of buyers")
    assert integrity.find_banned_claims("a representative sample of CFOs")
    assert not integrity.find_banned_claims(integrity.DISCLAIMER["summary"])


def test_examples_export_complete_with_provenance(research_client):
    client, _service = research_client
    for slug in EXAMPLES:
        loaded = client.post(f"/api/v1/research/examples/{slug}/load")
        assert loaded.status_code == 200
        study_id = loaded.get_json()["study"]["id"]
        res = client.get(f"/api/v1/research/studies/{study_id}/export")
        assert res.status_code == 200
        data = res.get_json()
        assert data["schema"] == "nanocrab.research.export.v1"
        assert data["disclaimer"]["genuine_ai_research"] is False
        assert data["disclaimer"]["not_research_validated"] is True
        assert data["disclaimer"]["synthetic_not_human_research"] is True
        assert data["integrity"]["ok"] is True
        assert data["integrity"]["openai_calls"] == 0
        assert data["integrity"]["cost_usd"] == 0
        assert data["integrity"]["paid_llm_unlocked"] is False
        assert data["integrity"]["banned_claim_patterns"] == []
        assert data["integrity"]["unresolved_source_ids"] == []
        assert 3 <= data["integrity"]["participant_count"] <= 5
        assert set(data["integrity"]["provenance_kinds_present"]) == KINDS
        brief = data["brief"]
        assert brief["research_question"]["kind"] == "user_input"
        assert brief["research_question"]["text"]
        assert brief["context"]["kind"] == "user_input"
        assert brief["context"]["sources"]
        assert all(src["kind"] == "source" for src in brief["context"]["sources"])
        session = data["study"]["sessions"][0]
        assert session["execution"] == "offline_deterministic_fixture"
        assert session["genuine_ai"] is False
        assert session["provider_calls"] == 0
        assert session["statements"]
        assert all(s["kind"] == "synthetic_statement" for s in session["statements"])
        index = data["source_index"]
        for finding in data["study"]["findings"]:
            assert finding["kind"] == "interpretation"
            assert finding["source_ids"]
            for source_id in finding["source_ids"]:
                assert source_id in index
            validation = finding["human_validation"]
            assert validation["kind"] == "human_validation_need"
            assert validation["id"] in index
        assert data["study"]["audience"]["representativeness"] is None


def test_example_roster_sizes(research_client):
    client, _service = research_client
    expected = {
        "saas-pricing": 4,
        "banking-onboarding": 5,
        "restaurant-software": 3,
    }
    dashboard = client.get("/api/v1/research/dashboard")
    assert dashboard.status_code == 200
    body = dashboard.get_json()
    by_slug = {row["slug"]: row for row in body["examples"]}
    for slug, count in expected.items():
        assert by_slug[slug]["participant_count"] == count
        assert by_slug[slug]["complete"] is True
    ceilings = body["ceilings"]["live_test_limits"]
    assert ceilings["max_agents"] == 5
    assert ceilings["max_rounds"] == 2
    assert ceilings["max_model_calls"] == 10
    assert ceilings["max_tokens_per_run"] == 20_000
    assert ceilings["max_tokens_per_call"] == 500
    assert ceilings["max_cost_usd_per_run"] == 0.20
    assert body["ceilings"]["paid_llm_unlocked"] is False
    assert body["disclaimer"]["openai_calls"] == 0


def test_custom_workflow_and_export(research_client):
    client, _service = research_client
    created = client.post(
        "/api/v1/research/projects",
        json={"name": "Field kit", "organization": "Classroom exercise"},
    )
    assert created.status_code == 200
    project_id = created.get_json()["project"]["id"]
    study = client.post(
        "/api/v1/research/studies",
        json={
            "project_id": project_id,
            "title": "Shift handover notes",
            "research_question": "How do leads describe a handover tool change?",
            "context": "User-supplied note: one team shares a paper log.",
            "audience_name": "Shift leads",
            "audience_description": "Synthetic roles for a classroom exercise.",
            "participant_count": 3,
            "sources": [
                {
                    "title": "Paper log note",
                    "text": "The user wrote that the log lives at the desk.",
                }
            ],
        },
    )
    assert study.status_code == 200
    study_id = study.get_json()["study"]["id"]
    people = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"count": 3},
    )
    assert people.status_code == 200
    assert len(people.get_json()["participants"]) == 3
    assert all(p["synthetic"] is True and p["human_subject"] is False for p in people.get_json()["participants"])
    session = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "offline_fixture"},
    )
    assert session.status_code == 200
    payload = session.get_json()
    assert payload["openai_calls"] == 0
    assert payload["session"]["genuine_ai"] is False
    assert payload["session"]["provider"] is None
    findings = client.post(f"/api/v1/research/studies/{study_id}/findings")
    assert findings.status_code == 200
    assert findings.get_json()["findings"]
    exported = client.get(f"/api/v1/research/studies/{study_id}/export")
    assert exported.status_code == 200
    data = exported.get_json()
    assert data["integrity"]["ok"] is True
    assert set(data["integrity"]["provenance_kinds_present"]) == KINDS
    assert data["integrity"]["genuine_ai_research"] is False
    assert data["integrity"]["openai_calls"] == 0
    assert data["integrity"]["cost_usd"] == 0


def test_participant_bounds(research_client):
    client, _service = research_client
    created = client.post("/api/v1/research/projects", json={"name": "Bounds"})
    project_id = created.get_json()["project"]["id"]
    study = client.post(
        "/api/v1/research/studies",
        json={
            "project_id": project_id,
            "title": "Bounds study",
            "research_question": "How do people describe the change?",
            "context": "A short user-supplied scene with no figures.",
            "audience_name": "Operators",
            "participant_count": 4,
        },
    )
    study_id = study.get_json()["study"]["id"]
    low = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"count": 2},
    )
    high = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"count": 6},
    )
    assert low.status_code == 400
    assert high.status_code == 400


def test_llm_session_is_409_and_makes_no_provider_call(research_client, monkeypatch):
    client, _service = research_client
    called = {"n": 0}

    class Boom:
        def __init__(self, *args, **kwargs):
            called["n"] += 1
            raise AssertionError("OpenAI client constructed")

    try:
        import openai

        monkeypatch.setattr(openai, "OpenAI", Boom, raising=False)
    except ImportError:
        pass

    study_id = client.post("/api/v1/research/examples/saas-pricing/load").get_json()["study"]["id"]
    blocked = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "llm"},
    )
    assert blocked.status_code == 409
    body = blocked.get_json()
    assert body["ok"] is False
    assert body["status"] == "blocked"
    assert body["paid_llm_unlocked"] is False
    assert body["openai_calls"] == 0
    assert "Freigabe" in body["error"]

    refused = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "llm", "llm_run_approved": True},
    )
    assert refused.status_code == 409
    refused_body = refused.get_json()
    assert refused_body["paid_llm_unlocked"] is False
    assert refused_body["openai_calls"] == 0
    assert "paid_llm_unlocked=false" in refused_body["error"]
    assert called["n"] == 0


def test_research_module_does_not_reference_openai():
    import backend.research.service as service
    import backend.research.routes as routes
    import backend.research.fixtures as fixtures

    for module in (service, routes, fixtures):
        text = inspect.getsource(module)
        assert "import openai" not in text.lower()
        assert "api.openai.com" not in text
        assert "OpenAI(" not in text


def test_health_ceilings_and_simulation_gate_unchanged(research_client):
    client, _service = research_client
    health = client.get("/api/health")
    assert health.status_code == 200
    data = health.get_json()
    live = data["config"]["live_test_limits"]
    assert live["max_agents"] == LIVE_TEST_MAX_AGENTS == 5
    assert live["max_rounds"] == LIVE_TEST_MAX_ROUNDS == 2
    assert live["max_model_calls"] == LIVE_TEST_MAX_LLM_CALLS == 10
    assert live["max_tokens_per_run"] == 20_000
    assert live["max_tokens_per_call"] == 500
    assert live["max_cost_usd_per_run"] == 0.20
    assert data["config"]["offline_scale"]["paid_llm_unlocked"] is False
    assert data["config"]["scalable_foundation"]["paid_llm_unlocked"] is False
    lab = data["research_lab"]
    assert lab["genuine_ai_research"] is False
    assert lab["openai_calls"] == 0
    assert lab["paid_llm_unlocked"] is False
    assert lab["simulation_engine_preserved"] is True
    blocked = client.post(
        "/api/simulation/start",
        json={"scenario": "Research gate", "mode": "llm", "max_steps": 1},
    )
    assert blocked.status_code == 409
    assert blocked.get_json()["ok"] is False


def test_meta_has_no_sso_and_does_not_touch_other_apps(research_client):
    client, _service = research_client
    res = client.get("/api/v1/research/meta")
    assert res.status_code == 200
    meta = res.get_json()
    assert meta["api_version"] == "v1"
    assert meta["sso"] is False
    assert meta["authentication"] == "none"
    assert meta["modifies_other_apps"] is False
    assert meta["research_sessions_call_openai"] is False
    assert meta["simulation_engine_preserved"] is True


def test_audiences_and_reports_list_examples(research_client):
    client, _service = research_client
    audiences = client.get("/api/v1/research/audiences")
    reports = client.get("/api/v1/research/reports")
    assert audiences.status_code == 200
    assert reports.status_code == 200
    assert len(audiences.get_json()["audiences"]) >= 3
    assert all(row["synthetic"] is True for row in audiences.get_json()["audiences"])
    assert all(row["representativeness"] is None for row in audiences.get_json()["audiences"])
    titles = {row["title"] for row in reports.get_json()["reports"]}
    assert "Seat price to usage price" in titles
    assert "Retail checking onboarding" in titles
    assert "POS and reservations in one dining room" in titles
    assert all(row["genuine_ai_research"] is False for row in reports.get_json()["reports"])


def test_example_patch_is_rejected(research_client):
    client, _service = research_client
    study_id = client.post("/api/v1/research/examples/restaurant-software/load").get_json()["study"]["id"]
    res = client.patch(
        f"/api/v1/research/studies/{study_id}",
        json={"research_question": "A different question that should not replace the fixture."},
    )
    assert res.status_code == 400
