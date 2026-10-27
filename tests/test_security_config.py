"""Ensure secrets never leak via API status payloads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app import app
from backend.config import load_settings


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-never-leak")
    monkeypatch.setenv("ZEP_API_KEY", "zep-secret-should-never-leak")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_does_not_expose_keys(client, monkeypatch: pytest.MonkeyPatch):
    # Reload settings used by app would still be module-level; inspect payload keys.
    res = client.get("/api/health")
    data = res.get_json()
    blob = json.dumps(data)
    assert "sk-test-should-never-leak" not in blob
    assert "zep-secret-should-never-leak" not in blob
    assert "OPENAI_API_KEY" not in blob or data["config"].get("api_key_exposed") is False
    assert data["config"]["api_key_exposed"] is False


def test_status_payload_never_includes_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-fake")
    cfg = load_settings()
    payload = cfg.status_payload()
    assert "openai_api_key" not in payload
    assert "sk-live-fake" not in json.dumps(payload)
    assert payload["api_key_exposed"] is False
    assert payload["llm_configured"] is True


def test_frontend_sources_have_no_secrets():
    root = Path(__file__).resolve().parents[1] / "frontend" / "src"
    for path in root.rglob("*"):
        if path.suffix not in {".ts", ".vue", ".js", ".css"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" not in text
        assert "sk-proj-" not in text
        assert "sk-live-" not in text
        # No env wiring or secret fields in the browser bundle sources.
        assert "process.env" not in text
        assert "VITE_OPENAI" not in text


def test_dotenv_gitignored_and_not_tracked():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.splitlines() or any(
        line.strip() == ".env" for line in gitignore.splitlines()
    )
    import subprocess

    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, ".env is tracked by git — remove it from the index"


def test_example_env_has_no_real_key():
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(
        encoding="utf-8"
    )
    for line in example.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            assert line.split("=", 1)[1].strip() == ""
        if line.startswith("ZEP_API_KEY="):
            assert line.split("=", 1)[1].strip() == ""
    assert "sk-" not in example
