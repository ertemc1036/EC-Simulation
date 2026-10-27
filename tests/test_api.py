"""Flask API smoke tests (no LLM)."""
from __future__ import annotations

import pytest

from backend.app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["oasis_import_ok"] is True
    assert data["versions"]["camel-oasis"] == "0.2.5"
    major = int(data["versions"]["mcp"].split(".")[0])
    assert major == 1
    assert data["config"]["llm_configured"] is False or isinstance(
        data["config"]["llm_configured"], bool
    )


def test_agents(client):
    res = client.get("/api/agents")
    assert res.status_code == 200
    agents = res.get_json()["agents"]
    assert len(agents) == 5
    names = {a["user_name"] for a in agents}
    assert "alice_analyst" in names
    assert "eve_empath" in names


def test_start_requires_scenario(client):
    res = client.post("/api/simulation/start", json={})
    assert res.status_code == 400
