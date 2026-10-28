"""AI research pathway with a mocked provider. No network and no OASIS start."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.app import app, engine
from backend.config import settings
from backend.research import ai_protocol
from backend.research.ai_protocol import (
    follow_up_messages,
    live_ceilings,
    opening_messages,
    parse_findings,
    parse_roster,
    parse_statement,
    roster_messages,
)
from backend.research.integrity import find_banned_claims
from backend.research.provider import (
    Completion,
    CredentialsMissing,
    OpenAICompatibleProvider,
    credential_block_reason,
    http_attempts,
    reset_http_attempts,
    set_research_provider,
)
from backend.research.routes import set_service
from backend.research.service import ResearchService
from backend.simulation.engine import SimulationEngine

QUESTION = "How would small-business buyers describe a SaaS list price moving from €29 to €39?"
CONTEXT = "The vendor note says the list price moves from €29 to €39 for the small-business plan."


class ScriptedProvider:
    name = "mock"

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def complete(self, *, messages, max_tokens, model):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "model": model})
        text = self.handler("\n".join(message["content"] for message in messages))
        return Completion(text=text, prompt_tokens=120, completion_tokens=80, model="mock-model")


class BoomProvider:
    name = "boom"

    def complete(self, *, messages, max_tokens, model):
        raise AssertionError("provider was called")


@pytest.fixture()
def research_client(tmp_path: Path, monkeypatch):
    reset_http_attempts()
    set_research_provider(None)

    def explode(*_args, **_kwargs):
        raise AssertionError("real HTTP is forbidden in research tests")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    service = ResearchService(tmp_path / "store.json", seed_examples=True)
    set_service(service)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, service
    set_service(None)
    set_research_provider(None)
    assert http_attempts == 0


def _create_pricing_study(client) -> str:
    project = client.post(
        "/api/v1/research/projects",
        json={"name": "Pricing desk", "organization": "Classroom exercise"},
    )
    assert project.status_code == 200
    study = client.post(
        "/api/v1/research/studies",
        json={
            "project_id": project.get_json()["project"]["id"],
            "title": "List price €29 to €39",
            "research_question": QUESTION,
            "context": CONTEXT,
            "audience_name": "Small-business software buyers",
            "audience_description": "Synthetic buyers for a classroom exercise.",
            "participant_count": 3,
            "sources": [
                {
                    "title": "Price note",
                    "text": "List price moves from €29 to €39.",
                }
            ],
        },
    )
    assert study.status_code == 200
    return study.get_json()["study"]["id"]


def _pricing_handler(blob: str) -> str:
    if "TASK participant_roster" in blob:
        return json.dumps(
            {
                "participants": [
                    {
                        "name": "Mara Keller",
                        "role": "Studio owner",
                        "perspective": "Reads a renewal month against the price on the invoice.",
                        "tension": "The list price change lands in the same month as a renewal.",
                        "hope": "The current term keeps the price already on the invoice.",
                    },
                    {
                        "name": "Jonas Feld",
                        "role": "Finance lead",
                        "perspective": "Matches the invoice line to the price the team was shown.",
                        "tension": "An invoice that does not match the price shown to the team.",
                        "hope": "A written price for the current term.",
                    },
                    {
                        "name": "Priya Shah",
                        "role": "Operations lead",
                        "perspective": "Asks whether the current plan stays available through the term.",
                        "tension": "Staff explaining a price they cannot find in the product.",
                        "hope": "The current plan stays available until the term ends.",
                    },
                ]
            }
        )
    source = re.search(r"Source (src_[0-9a-f]+):", blob)
    refs = [source.group(1)] if source else []
    if "TASK interview_opening" in blob:
        name = "Mara Keller" if "Mara Keller" in blob else "Jonas Feld" if "Jonas Feld" in blob else "Priya Shah"
        return json.dumps(
            {
                "text": (
                    f"{name} would describe the move from €29 to €39 as something the team "
                    "has to explain at renewal, and would want the current term left as written."
                ),
                "source_refs": refs,
            }
        )
    if "TASK interview_follow_up" in blob:
        return json.dumps(
            {
                "text": (
                    "The condition is a written price for the current term. "
                    "Without that note, the buyer would pause the change and ask finance to confirm the invoice."
                ),
                "source_refs": refs,
            }
        )
    if "TASK findings_synthesis" in blob:
        ids = re.findall(r"^- (stmt_[0-9a-f]+):", blob, re.M)
        assert ids, blob
        findings = []
        labels = (
            "Renewal month",
            "Invoice match",
            "Current plan",
        )
        for index, start in enumerate(range(0, len(ids), 2)):
            group = ids[start : start + 2]
            findings.append(
                {
                    "title": labels[index] if index < len(labels) else "Reading",
                    "text": (
                        "Interpretation of the cited synthetic statements: the persona wants "
                        "the current term kept distinct from the later list price. "
                        "This is not a rate and not a market claim."
                    ),
                    "statement_ids": group,
                    "human_validation": (
                        "A person must check whether real buyers recognize this condition. "
                        "These statements are synthetic model output."
                    ),
                }
            )
        return json.dumps({"findings": findings})
    raise AssertionError(f"unexpected prompt: {blob[:240]}")


def test_live_ceilings_cannot_rise_above_the_existing_caps():
    ceilings = live_ceilings()
    assert ceilings["max_agents"] <= 5
    assert ceilings["max_rounds"] <= 2
    assert ceilings["max_calls"] <= 10
    assert ceilings["max_tokens"] <= 20_000
    assert ceilings["max_output"] <= 500
    assert ceilings["max_cost"] <= 0.20
    assert ceilings["participants"] == 3


def test_prompts_name_the_task_and_refuse_statistics():
    study = {
        "research_question": {"id": "q_test", "text": QUESTION},
        "context": {
            "id": "ctx_test",
            "text": CONTEXT,
            "sources": [{"id": "src_test", "title": "Price note", "text": "List price moves from €29 to €39."}],
        },
        "audience": {"name": "Small-business software buyers", "description": "Synthetic buyers."},
    }
    person = {
        "id": "par_test",
        "name": "Mara Keller",
        "role": "Studio owner",
        "perspective": "Reads the invoice.",
    }
    roster = roster_messages(study)
    opening = opening_messages(study, person)
    follow = follow_up_messages(study, person, {"id": "stmt_test", "text": "The renewal month matters."})
    assert "TASK participant_roster" in roster[1]["content"]
    assert QUESTION in roster[1]["content"]
    assert "TASK interview_opening" in opening[1]["content"]
    assert "Mara Keller" in opening[1]["content"]
    assert "TASK interview_follow_up" in follow[1]["content"]
    assert "stmt_test" in follow[1]["content"]
    for messages in (roster, opening, follow):
        assert "not human subjects" in messages[0]["content"]
        assert find_banned_claims(messages[0]["content"]) == []
        assert find_banned_claims(messages[1]["content"]) == []


def test_schema_rejects_bad_json_and_statistics():
    roster = parse_roster(_pricing_handler("TASK participant_roster\n"))
    assert roster is not None and len(roster) == 3
    assert parse_roster("not json") is None
    assert parse_roster('{"participants": []}') is None
    assert parse_statement('{"text": "About 12% would leave."}') is None
    assert parse_statement("sure") is None
    parsed = parse_statement('{"text": "The current term should stay written down.", "source_refs": ["src_a"]}')
    assert parsed is not None and parsed["source_refs"] == ["src_a"]
    findings = parse_findings(
        json.dumps(
            {
                "findings": [
                    {
                        "title": "Renewal",
                        "text": "One synthetic persona wants the current term kept.",
                        "statement_ids": ["stmt_known", "stmt_missing"],
                        "human_validation": "A person still has to check this reading.",
                    }
                ]
            }
        ),
        {"stmt_known"},
    )
    assert findings is not None
    assert findings.findings[0]["statement_ids"] == ["stmt_known"]
    assert findings.dropped == 0
    assert parse_findings('{"findings": [{"title": "X", "text": "Y", "statement_ids": ["nope"]}]}', {"stmt_known"}) is None


def test_missing_credentials_make_no_socket(research_client, monkeypatch):
    client, _service = research_client
    study_id = _create_pricing_study(client)
    original = settings.openai_api_key
    object.__setattr__(settings, "openai_api_key", "")
    try:
        reason = credential_block_reason()
        assert reason and "OPENAI_API_KEY" in reason
        provider = OpenAICompatibleProvider()
        with pytest.raises(CredentialsMissing):
            provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
                model="mock",
            )
        blocked = client.post(
            f"/api/v1/research/studies/{study_id}/participants",
            json={"mode": "ai", "count": 3, "llm_run_approved": True},
        )
    finally:
        object.__setattr__(settings, "openai_api_key", original)
    assert blocked.status_code == 400
    body = blocked.get_json()
    assert body["missing_credentials"] is True
    assert body["provider_call_made"] is False
    assert body["openai_calls"] == 0
    assert http_attempts == 0


def test_approval_409_before_provider_and_ignores_simulation_state(research_client, monkeypatch):
    client, _service = research_client
    set_research_provider(BoomProvider())
    engine.llm_run_approved = True
    monkeypatch.setattr(
        SimulationEngine,
        "start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("engine start")),
    )
    study_id = client.post("/api/v1/research/examples/saas-pricing/load").get_json()["study"]["id"]
    blocked = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "llm"},
    )
    assert blocked.status_code == 409
    body = blocked.get_json()
    assert body["status"] == "blocked"
    assert "Freigabe" in body["error"]
    assert "Simulation Lab" in body["error"]
    assert body["provider_call_made"] is False
    assert body["openai_calls"] == 0
    people = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"mode": "ai", "llm_run_approved": False},
    )
    assert people.status_code == 409
    assert http_attempts == 0


def test_offline_demo_never_calls_and_is_not_labeled_as_model(research_client):
    client, _service = research_client
    provider = ScriptedProvider(_pricing_handler)
    set_research_provider(provider)
    study_id = _create_pricing_study(client)
    estimate = client.get(f"/api/v1/research/studies/{study_id}/cost-estimate")
    assert estimate.status_code == 200
    estimate_body = estimate.get_json()
    assert estimate_body["provider_call_made"] is False
    assert estimate_body["paid_llm_unlocked"] is False
    assert estimate_body["totals"]["calls"] == 8
    assert estimate_body["fits_ceilings"] is True
    people = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"count": 3, "mode": "offline_fixture", "llm_run_approved": True},
    )
    assert people.status_code == 200
    assert people.get_json()["provider_call_made"] is False
    session = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "offline_fixture", "llm_run_approved": True},
    )
    assert session.status_code == 200
    payload = session.get_json()
    assert payload["provider_call_made"] is False
    assert payload["session"]["execution"] == "offline_deterministic_fixture"
    assert payload["session"]["genuine_ai"] is False
    assert payload["session"]["provider"] is None
    assert "Offline demo" in payload["session"]["label"]
    assert all(row["origin"] == "offline_fixture" for row in payload["session"]["statements"])
    assert provider.calls == []
    pages = [
        "/api/v1/research/meta",
        "/api/v1/research/dashboard",
        "/api/v1/research/studies",
        f"/api/v1/research/studies/{study_id}",
        f"/api/v1/research/studies/{study_id}/cost-estimate",
        f"/api/v1/research/studies/{study_id}/export",
    ]
    for path in pages:
        assert client.get(path).status_code == 200
    assert provider.calls == []


def test_pricing_acceptance_mocked_pathway(research_client, monkeypatch):
    client, _service = research_client
    started = {"n": 0}

    def refuse_start(*_args, **_kwargs):
        started["n"] += 1
        raise AssertionError("OASIS start")

    monkeypatch.setattr(SimulationEngine, "start", refuse_start)
    provider = ScriptedProvider(_pricing_handler)
    set_research_provider(provider)
    study_id = _create_pricing_study(client)
    people = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"mode": "ai", "count": 3, "llm_run_approved": True},
    )
    assert people.status_code == 200
    roster = people.get_json()["participants"]
    assert [person["name"] for person in roster] == ["Mara Keller", "Jonas Feld", "Priya Shah"]
    assert all(person["generation"] == "model" and person["human_subject"] is False for person in roster)
    session = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "llm", "llm_run_approved": True},
    )
    assert session.status_code == 200
    session_body = session.get_json()
    assert session_body["partial"] is False
    assert session_body["session"]["execution"] == "model_interview"
    assert session_body["session"]["protocol"] == "short_interview_v1"
    assert session_body["session"]["provider"] == "mock"
    assert session_body["session"]["provider_calls"] == 6
    assert len(session_body["session"]["statements"]) == 6
    assert all(row["origin"] == "model" for row in session_body["session"]["statements"])
    assert all("offline demo" not in row["text"].lower() for row in session_body["session"]["statements"])
    findings = client.post(
        f"/api/v1/research/studies/{study_id}/findings",
        json={"llm_run_approved": True},
    )
    assert findings.status_code == 200
    finding_rows = findings.get_json()["findings"]
    assert finding_rows
    assert findings.get_json()["synthesis_method"] == "model_structured_json"
    exported = client.get(f"/api/v1/research/studies/{study_id}/export")
    assert exported.status_code == 200
    data = exported.get_json()
    assert data["integrity"]["ok"] is True
    assert data["integrity"]["genuine_ai_research"] is True
    assert data["integrity"]["synthesis_method"] == "model_structured_json"
    assert data["integrity"]["partial"] is False
    assert data["integrity"]["openai_calls"] == 8
    assert data["integrity"]["prompt_tokens"] == 8 * 120
    assert data["integrity"]["completion_tokens"] == 8 * 80
    assert data["integrity"]["cost_usd"] > 0
    assert data["integrity"]["paid_llm_unlocked"] is False
    assert data["integrity"]["banned_claim_patterns"] == []
    assert data["integrity"]["unresolved_source_ids"] == []
    assert data["integrity"]["not_research_validated"] is True
    index = data["source_index"]
    question_id = data["brief"]["research_question"]["id"]
    source_id = data["brief"]["context"]["sources"][0]["id"]
    statement_ids = {row["id"] for row in data["study"]["sessions"][0]["statements"]}
    assert statement_ids
    for finding in data["study"]["findings"]:
        assert finding["synthesis_method"] == "model_structured_json"
        assert finding["source_ids"]
        assert question_id in finding["source_ids"]
        assert any(sid in statement_ids for sid in finding["source_ids"])
        for source in finding["source_ids"]:
            assert source in index
        assert finding["human_validation"]["kind"] == "human_validation_need"
    assert source_id in index
    assert any(source_id in finding["source_ids"] for finding in data["study"]["findings"])
    assert all(call["max_tokens"] <= 500 for call in provider.calls)
    tasks = []
    for call in provider.calls:
        blob = "\n".join(message["content"] for message in call["messages"])
        for task in (
            "participant_roster",
            "interview_opening",
            "interview_follow_up",
            "findings_synthesis",
        ):
            if f"TASK {task}" in blob:
                tasks.append(task)
    assert tasks == [
        "participant_roster",
        "interview_opening",
        "interview_follow_up",
        "interview_opening",
        "interview_follow_up",
        "interview_opening",
        "interview_follow_up",
        "findings_synthesis",
    ]
    assert started["n"] == 0
    assert "SimulationEngine" not in Path("backend/research/service.py").read_text(encoding="utf-8")
    assert "SimulationEngine" not in Path("backend/research/ai_protocol.py").read_text(encoding="utf-8")


def test_invalid_model_response_is_not_stored(research_client):
    client, _service = research_client

    def handler(blob: str) -> str:
        if "TASK participant_roster" in blob:
            return "not json"
        return _pricing_handler(blob)

    set_research_provider(ScriptedProvider(handler))
    study_id = _create_pricing_study(client)
    rejected = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"mode": "ai", "count": 3, "llm_run_approved": True},
    )
    assert rejected.status_code == 422
    body = rejected.get_json()
    assert body["status"] == "invalid_model_response"
    assert body["provider_call_made"] is True
    assert body["openai_calls"] == 1
    study = client.get(f"/api/v1/research/studies/{study_id}").get_json()["study"]
    assert study["participants"] == []

    def percent_handler(blob: str) -> str:
        if "TASK participant_roster" in blob:
            return _pricing_handler(blob)
        return json.dumps({"text": "About 12% would leave at renewal."})

    set_research_provider(ScriptedProvider(percent_handler))
    other = _create_pricing_study(client)
    assert client.post(
        f"/api/v1/research/studies/{other}/participants",
        json={"mode": "ai", "count": 3, "llm_run_approved": True},
    ).status_code == 200
    session = client.post(
        f"/api/v1/research/studies/{other}/sessions",
        json={"mode": "llm", "llm_run_approved": True},
    )
    assert session.status_code == 200
    payload = session.get_json()
    assert payload["partial"] is True
    assert payload["stop_reason"] == "invalid_model_response"
    assert payload["session"]["statements"] == []


def test_budget_returns_partial_without_inventing_turns(research_client, monkeypatch):
    client, _service = research_client

    def tight():
        return {
            "max_agents": 5,
            "max_rounds": 2,
            "max_calls": 2,
            "max_tokens": 20_000,
            "max_output": 500,
            "max_cost": 0.20,
            "participants": 3,
        }

    monkeypatch.setattr(ai_protocol, "live_ceilings", tight)
    provider = ScriptedProvider(_pricing_handler)
    set_research_provider(provider)
    study_id = _create_pricing_study(client)
    people = client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"mode": "ai", "count": 3, "llm_run_approved": True},
    )
    assert people.status_code == 200
    session = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "llm", "llm_run_approved": True},
    )
    assert session.status_code == 200
    body = session.get_json()
    assert body["partial"] is True
    assert body["stop_reason"] == "budget"
    assert len(body["session"]["statements"]) == 1
    assert body["session"]["statements"][0]["prompt_task"] == "interview_opening"
    spoken = {row["participant_id"] for row in body["session"]["statements"]}
    assert len(spoken) == 1
    findings = client.post(
        f"/api/v1/research/studies/{study_id}/findings",
        json={"llm_run_approved": True},
    )
    assert findings.status_code == 200
    assert findings.get_json()["partial"] is True
    assert findings.get_json()["findings"] == []
    assert findings.get_json()["provider_call_made"] is False
    exported = client.get(f"/api/v1/research/studies/{study_id}/export").get_json()
    assert exported["integrity"]["ok"] is False
    assert exported["integrity"]["partial"] is True
    assert exported["integrity"]["genuine_ai_research"] is False
    assert len(provider.calls) == 2


def test_older_study_still_exports_and_offline_session_works(research_client):
    client, service = research_client
    set_research_provider(BoomProvider())
    study_id = _create_pricing_study(client)
    assert client.post(
        f"/api/v1/research/studies/{study_id}/participants",
        json={"count": 3},
    ).status_code == 200
    assert client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "offline_fixture"},
    ).status_code == 200
    assert client.post(f"/api/v1/research/studies/{study_id}/findings").status_code == 200
    study = service.studies[study_id]
    for key in ("participant_generation", "ai_usage", "synthesis_method"):
        study.pop(key, None)
    for person in study["participants"]:
        person.pop("generation", None)
    for session in study["sessions"]:
        session.pop("protocol", None)
        session.pop("partial", None)
        session.pop("prompt_tokens", None)
        for statement in session["statements"]:
            statement.pop("origin", None)
            statement.pop("prompt_task", None)
    service._persist()
    reloaded = ResearchService(service.path, seed_examples=True)
    set_service(reloaded)
    exported = client.get(f"/api/v1/research/studies/{study_id}/export")
    assert exported.status_code == 200
    data = exported.get_json()
    assert data["integrity"]["ok"] is True
    assert data["integrity"]["openai_calls"] == 0
    assert data["integrity"]["genuine_ai_research"] is False
    assert data["integrity"]["synthesis_method"] == "templated_offline_fixture"
    again = client.post(
        f"/api/v1/research/studies/{study_id}/sessions",
        json={"mode": "offline_fixture"},
    )
    assert again.status_code == 200
    assert again.get_json()["session"]["execution"] == "offline_deterministic_fixture"
    assert again.get_json()["provider_call_made"] is False


def test_frontend_does_not_auto_run_paid_research_on_load():
    text = Path("frontend/src/research/ResearchViews.vue").read_text(encoding="utf-8")
    assert "onMounted(refresh)" in text
    refresh = text[text.index("async function refresh") : text.index("async function openStudy")]
    assert "llm_run_approved" not in refresh
    assert "method: 'POST'" not in refresh
    assert "Offline demo fixture" in text
    assert "model_interview" in text
    assert "Approve this research run only" in text
