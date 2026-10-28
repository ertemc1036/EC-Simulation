"""Versioned research API. No SSO. No provider calls."""
from __future__ import annotations

import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

from backend.research.service import ResearchError, ResearchService

research_bp = Blueprint("nanocrab_research", __name__)

_service: ResearchService | None = None
_service_lock = threading.Lock()


def default_store_path() -> Path:
    from backend.config import settings

    return settings.root / "data" / "research" / "store.json"


def get_service() -> ResearchService:
    global _service
    with _service_lock:
        if _service is None:
            _service = ResearchService(default_store_path(), seed_examples=True)
        return _service


def set_service(service: ResearchService | None) -> None:
    global _service
    with _service_lock:
        _service = service


def _call(fn, *args, **kwargs):
    try:
        return jsonify(fn(*args, **kwargs))
    except ResearchError as exc:
        body = {
            "ok": False,
            "error": exc.message,
            "paid_llm_unlocked": False,
            "openai_calls": 0,
            **exc.extra,
        }
        return jsonify(body), exc.http_status


@research_bp.get("/api/v1/research/meta")
def research_meta():
    return _call(get_service().meta)


@research_bp.get("/api/v1/research/dashboard")
def research_dashboard():
    return _call(get_service().dashboard)


@research_bp.get("/api/v1/research/projects")
def research_projects():
    return _call(get_service().list_projects)


@research_bp.post("/api/v1/research/projects")
def research_create_project():
    body = request.get_json(silent=True) or {}
    return _call(get_service().create_project, body)


@research_bp.get("/api/v1/research/projects/<project_id>")
def research_get_project(project_id: str):
    return _call(get_service().get_project, project_id)


@research_bp.get("/api/v1/research/studies")
def research_studies():
    return _call(get_service().list_studies)


@research_bp.post("/api/v1/research/studies")
def research_create_study():
    body = request.get_json(silent=True) or {}
    return _call(get_service().create_study, body)


@research_bp.get("/api/v1/research/studies/<study_id>")
def research_get_study(study_id: str):
    return _call(get_service().get_study, study_id)


@research_bp.patch("/api/v1/research/studies/<study_id>")
def research_patch_study(study_id: str):
    body = request.get_json(silent=True) or {}
    return _call(get_service().update_study, study_id, body)


@research_bp.post("/api/v1/research/studies/<study_id>/participants")
def research_participants(study_id: str):
    body = request.get_json(silent=True) or {}
    return _call(get_service().generate_participants, study_id, body)


@research_bp.post("/api/v1/research/studies/<study_id>/sessions")
def research_session(study_id: str):
    body = request.get_json(silent=True) or {}
    return _call(get_service().run_session, study_id, body)


@research_bp.post("/api/v1/research/studies/<study_id>/findings")
def research_findings(study_id: str):
    return _call(get_service().derive_findings, study_id)


@research_bp.get("/api/v1/research/studies/<study_id>/export")
def research_export(study_id: str):
    return _call(get_service().export_study, study_id)


@research_bp.get("/api/v1/research/audiences")
def research_audiences():
    return _call(get_service().list_audiences)


@research_bp.get("/api/v1/research/reports")
def research_reports():
    return _call(get_service().list_reports)


@research_bp.post("/api/v1/research/examples/<slug>/load")
def research_load_example(slug: str):
    return _call(get_service().load_example, slug)
