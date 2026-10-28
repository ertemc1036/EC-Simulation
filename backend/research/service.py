"""Research study workflow stored as JSON.

Offline demo fixtures never call a provider. The model pathway uses the
OpenAI-compatible transport only after a run-specific approval on that request.
This module does not start the simulation engine and does not read its approval.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.research.ai_protocol import (
    AI_PARTICIPANTS,
    PROTOCOL,
    SYNTHESIS_FIXTURE,
    SYNTHESIS_MODEL,
    StudyLedger,
    build_cost_estimate,
    run_interview,
    run_roster,
    run_synthesis,
)
from backend.research.fixtures import EXAMPLE_ORDER, get_example
from backend.research.integrity import (
    DISCLAIMER,
    disclaimer_for_study,
    kind_index,
    prose_banned_hits,
    provenance_kinds_present,
    unresolved_source_ids,
    user_text_has_percent,
)
from backend.research.provider import credential_block_reason, get_research_provider

_AI_MODES = {"ai", "llm", "model", "live", "live_llm", "openai"}
_OFFLINE_MODES = {"offline_fixture", "offline", "fixture", "offline_demo"}
_MODEL_VALIDATION = (
    "A person must check whether real customers recognize this reading. "
    "The cited statements are synthetic model output, not human-subject evidence."
)


def _stored_synthesis_method(study: dict) -> str:
    explicit = study.get("synthesis_method")
    if explicit:
        return str(explicit)
    session = (study.get("sessions") or [{}])[0]
    if session.get("execution") == "model_interview":
        return "not_synthesized"
    if study.get("findings"):
        return SYNTHESIS_FIXTURE
    return "not_synthesized"


def _stored_genuine(study: dict) -> bool:
    session = (study.get("sessions") or [{}])[0]
    usage = study.get("ai_usage") or {}
    partial = bool(usage.get("partial") or session.get("partial"))
    return bool(
        session.get("execution") == "model_interview"
        and session.get("statements")
        and _stored_synthesis_method(study) == SYNTHESIS_MODEL
        and not partial
    )

SCHEMA = "nanocrab.research.v1"
EXPORT_SCHEMA = "nanocrab.research.export.v1"

_CUSTOM_ROSTER = (
    {
        "name": "Avery Quinn",
        "slug": "avery_quinn",
        "role": "Finance lead",
        "perspective": "Reads the brief through budget signature and invoice control.",
        "tension": "a change that can move the invoice after a budget is already signed",
        "hope": "a written boundary for what can change during the term",
    },
    {
        "name": "Noah Berger",
        "slug": "noah_berger",
        "role": "Operations manager",
        "perspective": "Owns the rollout on a day that is already full.",
        "tension": "extra steps for staff who already have a full shift",
        "hope": "a short pilot with a clear stop condition",
    },
    {
        "name": "Sofia Almeida",
        "slug": "sofia_almeida",
        "role": "Frontline user",
        "perspective": "Uses the tool while talking with a customer or guest.",
        "tension": "a screen that hides the one fact needed during a conversation",
        "hope": "the current workaround staying available until the new path is reliable",
    },
    {
        "name": "Kenji Mori",
        "slug": "kenji_mori",
        "role": "IT or security reviewer",
        "perspective": "Asks who can see customer data before a rollout.",
        "tension": "a new vendor path that is unclear about who can see customer data",
        "hope": "an admin control and a named data boundary before rollout",
    },
    {
        "name": "Leah Okoye",
        "slug": "leah_okoye",
        "role": "Customer-facing lead",
        "perspective": "Has to explain the change without inventing a figure.",
        "tension": "having to explain a policy they cannot see in the product",
        "hope": "a plain-language line they can repeat without inventing a number",
    },
)


class ResearchError(Exception):
    def __init__(self, message: str, http_status: int = 400, **extra: Any) -> None:
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.extra = extra


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


class ResearchService:
    def __init__(self, path: Path, *, seed_examples: bool = True) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.projects: dict[str, dict] = {}
        self.studies: dict[str, dict] = {}
        if path.exists():
            self._load()
        if seed_examples:
            for slug in EXAMPLE_ORDER:
                if self._study_for_slug(slug) is None:
                    self._insert_example(slug)
            self._persist()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.projects = raw.get("projects") or {}
        self.studies = raw.get("studies") or {}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "projects": self.projects,
            "studies": self.studies,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _study_for_slug(self, slug: str) -> dict | None:
        for study in self.studies.values():
            if study.get("example_slug") == slug:
                return study
        return None

    def _insert_example(self, slug: str) -> dict:
        spec = get_example(slug)
        if spec is None:
            raise ResearchError(f"Unknown example '{slug}'.", 404)
        project = dict(spec["project"])
        project["kind"] = "user_input"
        project["created_at"] = "2026-10-28T00:00:00+00:00"
        project["example_slug"] = slug
        study = spec["study"]
        study["project_id"] = project["id"]
        study["created_at"] = "2026-10-28T00:00:00+00:00"
        study["updated_at"] = "2026-10-28T00:00:00+00:00"
        study["read_only_fixture"] = True
        self._assert_example_clean(study)
        self.projects[project["id"]] = project
        self.studies[study["id"]] = study
        return study

    def _assert_example_clean(self, study: dict) -> None:
        missing = unresolved_source_ids(study)
        banned = prose_banned_hits(study)
        count = len(study.get("participants") or [])
        kinds = provenance_kinds_present(study)
        if missing or banned or not (3 <= count <= 5) or len(kinds) < 5:
            raise ResearchError(
                "Example fixture failed integrity checks.",
                500,
                unresolved_source_ids=missing,
                banned=banned,
                participant_count=count,
                kinds=kinds,
            )

    def _require_study(self, study_id: str) -> dict:
        study = self.studies.get(study_id)
        if study is None:
            raise ResearchError("Study not found.", 404)
        return study

    def _public_study(self, study: dict) -> dict:
        project = self.projects.get(study.get("project_id") or "", {})
        out = dict(study)
        out["project_name"] = project.get("name")
        out["organization"] = project.get("organization")
        out["disclaimer"] = disclaimer_for_study(study)
        out["provenance_kinds"] = provenance_kinds_present(study)
        out["ai_cost_estimate"] = build_cost_estimate(study)
        return out

    def ceilings(self) -> dict[str, Any]:
        payload = settings.status_payload()
        live = dict(payload["live_test_limits"])
        offline = payload["offline_scale"]
        return {
            "live_test_limits": live,
            "paid_llm_unlocked": False,
            "offline_paid_llm_unlocked": offline.get("paid_llm_unlocked") is True,
            "openai_calls": 0,
            "cost_usd": 0,
        }

    def meta(self) -> dict[str, Any]:
        return {
            "ok": True,
            "product": "NanoCrab Lab",
            "mvp": "1.0",
            "api_version": "v1",
            "schema": SCHEMA,
            "export_schema": EXPORT_SCHEMA,
            "authentication": "none",
            "sso": False,
            "modifies_other_apps": False,
            "simulation_engine_preserved": True,
            "research_sessions_call_openai": False,
            "research_sessions_call_simulation_engine": False,
            "ai_research_pathway": True,
            "ai_research_protocol": PROTOCOL,
            "ai_research_participants": AI_PARTICIPANTS,
            "ai_research_requires_run_specific_approval": True,
            "ai_research_inherits_simulation_approval": False,
            "page_load_makes_provider_calls": False,
            "paid_llm_unlocked": False,
            "disclaimer": dict(DISCLAIMER),
            "ceilings": self.ceilings(),
            "provenance_kinds": [
                "user_input",
                "source",
                "synthetic_statement",
                "interpretation",
                "human_validation_need",
            ],
            "examples": list(EXAMPLE_ORDER),
        }

    def dashboard(self) -> dict[str, Any]:
        with self._lock:
            studies = list(self.studies.values())
            reports = [s for s in studies if s.get("findings")]
            return {
                "ok": True,
                "product": "NanoCrab Lab",
                "mvp": "1.0",
                "disclaimer": dict(DISCLAIMER),
                "ceilings": self.ceilings(),
                "recorded_provider_usage": self._recorded_usage(),
                "page_load_makes_provider_calls": False,
                "counts": {
                    "projects": len(self.projects),
                    "studies": len(studies),
                    "audiences": sum(1 for s in studies if s.get("audience")),
                    "reports": len(reports),
                },
                "examples": [
                    {
                        "slug": s.get("example_slug"),
                        "title": s.get("title"),
                        "study_id": s.get("id"),
                        "project_id": s.get("project_id"),
                        "project_name": self.projects.get(s.get("project_id") or "", {}).get("name"),
                        "participant_count": len(s.get("participants") or []),
                        "complete": s.get("status") == "findings_ready",
                    }
                    for s in studies
                    if s.get("example_slug")
                ],
                "workflow": [
                    "research_project",
                    "research_question",
                    "context",
                    "audience",
                    "synthetic_participants",
                    "research_session",
                    "findings",
                    "export_json",
                ],
            }

    def list_projects(self) -> dict[str, Any]:
        with self._lock:
            rows = sorted(self.projects.values(), key=lambda p: p.get("name") or "")
            return {"ok": True, "projects": rows, "disclaimer": dict(DISCLAIMER)}

    def create_project(self, body: dict) -> dict[str, Any]:
        name = (body.get("name") or "").strip()
        if len(name) < 2:
            raise ResearchError("Project name is required.")
        organization = (body.get("organization") or "").strip()
        project = {
            "id": _new_id("prj"),
            "name": name,
            "organization": organization,
            "kind": "user_input",
            "created_at": _now(),
            "example_slug": None,
        }
        with self._lock:
            self.projects[project["id"]] = project
            self._persist()
        return {"ok": True, "project": project}

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            project = self.projects.get(project_id)
            if project is None:
                raise ResearchError("Project not found.", 404)
            studies = [
                self._summary(s)
                for s in self.studies.values()
                if s.get("project_id") == project_id
            ]
        return {"ok": True, "project": project, "studies": studies}

    def _summary(self, study: dict) -> dict[str, Any]:
        project = self.projects.get(study.get("project_id") or "", {})
        return {
            "id": study.get("id"),
            "project_id": study.get("project_id"),
            "project_name": project.get("name"),
            "title": study.get("title"),
            "status": study.get("status"),
            "example_slug": study.get("example_slug"),
            "participant_count": len(study.get("participants") or []),
            "session_count": len(study.get("sessions") or []),
            "finding_count": len(study.get("findings") or []),
            "audience_name": (study.get("audience") or {}).get("name"),
            "read_only_fixture": bool(study.get("read_only_fixture")),
        }

    def list_studies(self) -> dict[str, Any]:
        with self._lock:
            rows = [self._summary(s) for s in self.studies.values()]
            rows.sort(key=lambda r: (r.get("project_name") or "", r.get("title") or ""))
        return {"ok": True, "studies": rows, "disclaimer": dict(DISCLAIMER)}

    def create_study(self, body: dict) -> dict[str, Any]:
        project_id = (body.get("project_id") or "").strip()
        title = (body.get("title") or "").strip()
        question = (body.get("research_question") or body.get("question") or "").strip()
        context_text = (body.get("context") or body.get("context_text") or "").strip()
        audience_name = (body.get("audience_name") or "").strip()
        audience_description = (body.get("audience_description") or "").strip()
        if project_id not in self.projects:
            raise ResearchError("Project not found.", 404)
        if len(title) < 2:
            raise ResearchError("Study title is required.")
        if len(question) < 8:
            raise ResearchError("Research question is required.")
        if len(context_text) < 8:
            raise ResearchError("Context is required.")
        if len(audience_name) < 2:
            raise ResearchError("Audience name is required.")
        try:
            count = int(body.get("participant_count") or 4)
        except (TypeError, ValueError):
            raise ResearchError("participant_count must be a number from 3 to 5.")
        if count < 3 or count > 5:
            raise ResearchError("Synthetic participants must be between 3 and 5.")
        sources = self._parse_sources(body.get("sources"))
        study = {
            "id": _new_id("stu"),
            "project_id": project_id,
            "title": title,
            "example_slug": None,
            "status": "draft",
            "read_only_fixture": False,
            "requested_participant_count": count,
            "created_at": _now(),
            "updated_at": _now(),
            "research_question": {
                "id": _new_id("q"),
                "kind": "user_input",
                "text": question,
            },
            "context": {
                "id": _new_id("ctx"),
                "kind": "user_input",
                "text": context_text,
                "sources": sources,
            },
            "audience": {
                "id": _new_id("aud"),
                "name": audience_name,
                "description": audience_description
                or "Synthetic roster supplied for this study. Not a market sample.",
                "synthetic": True,
                "representativeness": None,
            },
            "participants": [],
            "sessions": [],
            "findings": [],
            "user_input_contains_percent_pattern": user_text_has_percent(question)
            or user_text_has_percent(context_text),
        }
        with self._lock:
            if project_id not in self.projects:
                raise ResearchError("Project not found.", 404)
            self.studies[study["id"]] = study
            self._persist()
        return {"ok": True, "study": self._public_study(study)}

    def _parse_sources(self, raw: Any) -> list[dict]:
        if not raw:
            return []
        if not isinstance(raw, list):
            raise ResearchError("sources must be a list.")
        sources: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ResearchError("Each source needs a title and text.")
            title = (item.get("title") or "").strip()
            text = (item.get("text") or "").strip()
            if not title or not text:
                continue
            sources.append(
                {
                    "id": _new_id("src"),
                    "kind": "source",
                    "title": title,
                    "text": text,
                    "supplied_by": "user",
                }
            )
        return sources

    def get_study(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            study = self._require_study(study_id)
            return {
                "ok": True,
                "study": self._public_study(study),
                "source_index": kind_index(study),
            }

    def update_study(self, study_id: str, body: dict) -> dict[str, Any]:
        with self._lock:
            study = self._require_study(study_id)
            if study.get("read_only_fixture"):
                raise ResearchError(
                    "Example fixtures are read-only. Create a new study to change the brief.",
                    400,
                )
            if "research_question" in body or "question" in body:
                text = (body.get("research_question") or body.get("question") or "").strip()
                if len(text) < 8:
                    raise ResearchError("Research question is required.")
                study["research_question"]["text"] = text
            if "context" in body or "context_text" in body:
                text = (body.get("context") or body.get("context_text") or "").strip()
                if len(text) < 8:
                    raise ResearchError("Context is required.")
                study["context"]["text"] = text
            if "sources" in body:
                study["context"]["sources"] = self._parse_sources(body.get("sources"))
            if "audience_name" in body:
                name = (body.get("audience_name") or "").strip()
                if len(name) < 2:
                    raise ResearchError("Audience name is required.")
                study["audience"]["name"] = name
            if "audience_description" in body:
                study["audience"]["description"] = (body.get("audience_description") or "").strip()
            if "participant_count" in body:
                try:
                    count = int(body.get("participant_count"))
                except (TypeError, ValueError):
                    raise ResearchError("participant_count must be a number from 3 to 5.")
                if count < 3 or count > 5:
                    raise ResearchError("Synthetic participants must be between 3 and 5.")
                study["requested_participant_count"] = count
            study["participants"] = []
            study["sessions"] = []
            study["findings"] = []
            study["participant_generation"] = None
            study["synthesis_method"] = None
            study["status"] = "draft"
            study["updated_at"] = _now()
            question = study["research_question"]["text"]
            context_text = study["context"]["text"]
            study["user_input_contains_percent_pattern"] = user_text_has_percent(
                question
            ) or user_text_has_percent(context_text)
            self._persist()
            return {
                "ok": True,
                "study": self._public_study(study),
                "invalidated": ["participants", "sessions", "findings"],
            }

    def _recorded_usage(self) -> dict[str, Any]:
        calls = prompt = completion = 0
        cost = 0.0
        for study in self.studies.values():
            usage = study.get("ai_usage") or {}
            calls += int(usage.get("calls") or 0)
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
            cost += float(usage.get("cost_usd") or 0)
        return {
            "calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cost_usd": round(cost, 6),
            "paid_llm_unlocked": False,
            "provider_call_made_by_this_response": False,
        }

    def cost_estimate(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            study = self._require_study(study_id)
            return build_cost_estimate(study)

    def _require_approval(self, body: dict) -> None:
        approved = _truthy(body.get("llm_run_approved", body.get("confirm_llm_spend", False)))
        if approved:
            return
        raise ResearchError(
            "Approval required (Freigabe). Send llm_run_approved=true on this research request. "
            "Simulation Lab approval is not used. No provider call was made.",
            409,
            status="blocked",
            paid_llm_unlocked=False,
            openai_calls=0,
            provider_call_made=False,
        )

    def _block_if_no_credentials(self, study: dict) -> None:
        reason = credential_block_reason()
        if not reason:
            return
        usage = study.get("ai_usage") or {}
        raise ResearchError(
            reason,
            400,
            status="missing_credentials",
            missing_credentials=True,
            paid_llm_unlocked=False,
            openai_calls=int(usage.get("calls") or 0),
            cost_usd=float(usage.get("cost_usd") or 0),
            provider_call_made=False,
        )

    def _apply_usage(self, study: dict, ledger: StudyLedger, *, partial: bool, stop_reason: str | None) -> dict:
        snap = ledger.snapshot(partial=partial, stop_reason=stop_reason)
        study["ai_usage"] = snap
        study["updated_at"] = _now()
        return snap

    def _usage_payload(self, snap: dict) -> dict[str, Any]:
        return {
            "openai_calls": snap["calls"],
            "prompt_tokens": snap["prompt_tokens"],
            "completion_tokens": snap["completion_tokens"],
            "total_tokens": snap["total_tokens"],
            "cost_usd": snap["cost_usd"],
            "paid_llm_unlocked": False,
            "partial": snap["partial"],
            "stop_reason": snap["stop_reason"],
        }

    def _reject_demo_ai(self) -> None:
        raise ResearchError(
            "Offline demo fixtures are not live AI research. "
            "Create a study and choose AI Research. No provider call was made.",
            400,
            status="offline_demo",
            paid_llm_unlocked=False,
            openai_calls=0,
            provider_call_made=False,
        )

    def generate_participants(self, study_id: str, body: dict | None) -> dict[str, Any]:
        body = body or {}
        mode = (body.get("mode") or body.get("generation") or "offline_fixture").strip().lower()
        if mode in _AI_MODES:
            self._require_approval(body)
            return self._generate_model_participants(study_id, body)
        if mode not in _OFFLINE_MODES:
            raise ResearchError(
                "Unknown participant mode. Use offline_fixture or ai.",
                400,
                paid_llm_unlocked=False,
                openai_calls=0,
                provider_call_made=False,
            )
        with self._lock:
            study = self._require_study(study_id)
            if study.get("read_only_fixture"):
                roster = study.get("participants") or []
                if body.get("count") is not None or body.get("participant_count") is not None:
                    try:
                        asked = int(body.get("count", body.get("participant_count")))
                    except (TypeError, ValueError):
                        raise ResearchError("participant_count must be a number from 3 to 5.")
                    if asked != len(roster):
                        raise ResearchError(
                            "Example studies keep their fixture roster "
                            f"({len(roster)} synthetic personas)."
                        )
                return {
                    "ok": True,
                    "participants": roster,
                    "study": self._public_study(study),
                    "note": "Example fixture roster kept. Offline fixture, not generated by a model.",
                }
            if body.get("count") is None and body.get("participant_count") is None:
                count = int(study.get("requested_participant_count") or 4)
            else:
                try:
                    count = int(body.get("count", body.get("participant_count")))
                except (TypeError, ValueError):
                    raise ResearchError("participant_count must be a number from 3 to 5.")
            if count < 3 or count > 5:
                raise ResearchError("Synthetic participants must be between 3 and 5.")
            people = []
            for row in _CUSTOM_ROSTER[:count]:
                people.append(
                    {
                        "id": _new_id("par"),
                        "kind": "synthetic_participant",
                        "name": row["name"],
                        "role": row["role"],
                        "perspective": row["perspective"],
                        "tension": row["tension"],
                        "hope": row["hope"],
                        "synthetic": True,
                        "human_subject": False,
                        "generation": "offline_deterministic_fixture",
                    }
                )
            study["participants"] = people
            study["participant_generation"] = "offline_deterministic_fixture"
            study["requested_participant_count"] = count
            study["sessions"] = []
            study["findings"] = []
            study["synthesis_method"] = None
            study["status"] = "participants_ready"
            study["updated_at"] = _now()
            self._persist()
            return {
                "ok": True,
                "participants": people,
                "study": self._public_study(study),
                "note": "Offline demo roster. Not a model transcript and not human subjects.",
                "provider_call_made": False,
                "openai_calls": 0,
                "paid_llm_unlocked": False,
            }

    def _generate_model_participants(self, study_id: str, body: dict) -> dict[str, Any]:
        with self._lock:
            study = self._require_study(study_id)
            if study.get("read_only_fixture"):
                self._reject_demo_ai()
            if body.get("count") is not None or body.get("participant_count") is not None:
                try:
                    asked = int(body.get("count", body.get("participant_count")))
                except (TypeError, ValueError):
                    raise ResearchError("AI research interviews use 3 synthetic participants.")
                if asked != AI_PARTICIPANTS:
                    raise ResearchError(
                        "AI research interviews use 3 synthetic participants.",
                        400,
                        paid_llm_unlocked=False,
                        openai_calls=0,
                        provider_call_made=False,
                    )
            self._block_if_no_credentials(study)
            ledger = StudyLedger.from_study(study)
            calls_before = ledger.calls
            people_rows, stop = run_roster(study, get_research_provider(), ledger)
            if people_rows is None and stop == "budget":
                snap = self._apply_usage(study, ledger, partial=True, stop_reason="budget")
                self._persist()
                return {
                    "ok": True,
                    "partial": True,
                    "participants": study.get("participants") or [],
                    "study": self._public_study(study),
                    "note": "The study budget cannot cover another provider call. No personas were invented.",
                    "provider_call_made": False,
                    **self._usage_payload(snap),
                }
            if people_rows is None and stop == "missing_credentials":
                raise ResearchError(
                    "OPENAI_API_KEY is missing. No provider call was made.",
                    400,
                    status="missing_credentials",
                    missing_credentials=True,
                    paid_llm_unlocked=False,
                    openai_calls=ledger.calls,
                    provider_call_made=False,
                )
            if people_rows is None:
                snap = self._apply_usage(
                    study,
                    ledger,
                    partial=True,
                    stop_reason=stop or "invalid_model_response",
                )
                self._persist()
                status = 422 if stop == "invalid_model_response" else 502
                raise ResearchError(
                    "The model response could not be stored. No personas were invented.",
                    status,
                    status=stop or "invalid_model_response",
                    **self._usage_payload(snap),
                    provider_call_made=ledger.calls > calls_before,
                )
            people = []
            for row in people_rows:
                people.append(
                    {
                        "id": _new_id("par"),
                        "kind": "synthetic_participant",
                        "name": row["name"],
                        "role": row["role"],
                        "perspective": row["perspective"],
                        "tension": row["tension"],
                        "hope": row["hope"],
                        "synthetic": True,
                        "human_subject": False,
                        "generation": "model",
                    }
                )
            study["participants"] = people
            study["participant_generation"] = "model"
            study["requested_participant_count"] = AI_PARTICIPANTS
            study["sessions"] = []
            study["findings"] = []
            study["synthesis_method"] = None
            study["status"] = "participants_ready"
            snap = self._apply_usage(study, ledger, partial=bool(stop), stop_reason=stop)
            self._persist()
            return {
                "ok": True,
                "participants": people,
                "study": self._public_study(study),
                "note": "Model pathway roster. Synthetic personas, not human subjects, not an offline demo fixture.",
                "provider_call_made": ledger.calls > calls_before,
                **self._usage_payload(snap),
            }

    def run_session(self, study_id: str, body: dict | None) -> dict[str, Any]:
        body = body or {}
        mode = (body.get("mode") or body.get("execution") or "offline_fixture").strip().lower()
        if mode in _AI_MODES:
            self._require_approval(body)
            return self._run_model_session(study_id)
        if mode not in _OFFLINE_MODES:
            raise ResearchError(
                "Unknown research session mode. Use offline_fixture or llm. "
                "This endpoint does not start the OASIS engine.",
                400,
                paid_llm_unlocked=False,
                openai_calls=0,
                provider_call_made=False,
            )
        with self._lock:
            study = self._require_study(study_id)
            if study.get("read_only_fixture"):
                return {
                    "ok": True,
                    "session": (study.get("sessions") or [None])[0],
                    "study": self._public_study(study),
                    "openai_calls": 0,
                    "cost_usd": 0,
                    "paid_llm_unlocked": False,
                }
            people = study.get("participants") or []
            if not (3 <= len(people) <= 5):
                raise ResearchError("Add 3 to 5 synthetic participants before the session.")
            statements = []
            for person in people:
                first = _new_id("stmt")
                second = _new_id("stmt")
                tension = person.get("tension") or "a tension named in the role perspective"
                hope = person.get("hope") or "a condition a human still has to check"
                statements.append(
                    {
                        "id": first,
                        "kind": "synthetic_statement",
                        "participant_id": person["id"],
                        "text": (
                            f"{person['name']} is a synthetic {person['role']} persona "
                            f"and not a human subject. In this offline demo fixture, the tension "
                            f"they voice is {tension}. This is not a model transcript and "
                            f"not a measured frequency."
                        ),
                        "source_refs": [],
                        "origin": "offline_fixture",
                        "prompt_task": "offline_template",
                    }
                )
                refs = []
                sources = (study.get("context") or {}).get("sources") or []
                if sources:
                    refs.append(sources[0]["id"])
                statements.append(
                    {
                        "id": second,
                        "kind": "synthetic_statement",
                        "participant_id": person["id"],
                        "text": (
                            f"{person['name']} would want the study record to capture this hope: "
                            f"{hope}. They point at the user-supplied brief rather than inventing a figure."
                        ),
                        "source_refs": refs,
                        "origin": "offline_fixture",
                        "prompt_task": "offline_template",
                    }
                )
            session = {
                "id": _new_id("ses"),
                "kind": "research_session",
                "execution": "offline_deterministic_fixture",
                "protocol": "offline_demo",
                "genuine_ai": False,
                "provider": None,
                "provider_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0,
                "rounds": 1,
                "live_llm": False,
                "partial": False,
                "label": "Offline demo fixture — not live AI",
                "statements": statements,
            }
            study["sessions"] = [session]
            study["findings"] = []
            study["synthesis_method"] = None
            study["status"] = "session_ready"
            study["updated_at"] = _now()
            self._persist()
            return {
                "ok": True,
                "session": session,
                "study": self._public_study(study),
                "openai_calls": 0,
                "cost_usd": 0,
                "paid_llm_unlocked": False,
                "provider_call_made": False,
            }

    def _run_model_session(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            study = self._require_study(study_id)
            if study.get("read_only_fixture"):
                self._reject_demo_ai()
            if study.get("participant_generation") != "model":
                usage = study.get("ai_usage") or {}
                raise ResearchError(
                    "Generate model participants before an AI research interview. "
                    "Offline demo personas stay on the offline path. No provider call was made.",
                    400,
                    status="offline_demo",
                    paid_llm_unlocked=False,
                    openai_calls=int(usage.get("calls") or 0),
                    provider_call_made=False,
                )
            people = study.get("participants") or []
            if len(people) != AI_PARTICIPANTS:
                raise ResearchError(
                    "AI research interviews use 3 synthetic participants.",
                    400,
                    paid_llm_unlocked=False,
                    openai_calls=0,
                    provider_call_made=False,
                )
            self._block_if_no_credentials(study)
            ledger = StudyLedger.from_study(study)
            mark = ledger.mark()
            provider = get_research_provider()
            result = run_interview(study, provider, ledger, _new_id)
            statements = result["statements"]
            partial = bool(result["partial"])
            stop = result["stop_reason"]
            spent = ledger.delta(mark)
            rounds = max((row.get("turn") or 1) for row in statements) if statements else 0
            session = {
                "id": _new_id("ses"),
                "kind": "research_session",
                "execution": "model_interview",
                "protocol": PROTOCOL,
                "genuine_ai": bool(statements),
                "provider": getattr(provider, "name", "model"),
                "provider_calls": spent["calls"],
                "prompt_tokens": spent["prompt_tokens"],
                "completion_tokens": spent["completion_tokens"],
                "total_tokens": spent["total_tokens"],
                "cost_usd": spent["cost_usd"],
                "rounds": rounds,
                "live_llm": False,
                "partial": partial,
                "stop_reason": stop,
                "label": "Model pathway transcript — synthetic personas, not an offline demo",
                "statements": statements,
            }
            study["sessions"] = [session]
            study["findings"] = []
            study["synthesis_method"] = None
            study["status"] = "session_partial" if partial or not statements else "session_ready"
            snap = self._apply_usage(study, ledger, partial=partial, stop_reason=stop)
            self._persist()
            note = (
                "Partial model transcript. Missing turns were not invented."
                if partial
                else "Model pathway transcript. Not an offline demo fixture."
            )
            return {
                "ok": True,
                "session": session,
                "study": self._public_study(study),
                "note": note,
                "provider_call_made": int(spent["calls"]) > 0,
                **self._usage_payload(snap),
            }

    def derive_findings(self, study_id: str, body: dict | None = None) -> dict[str, Any]:
        body = body or {}
        with self._lock:
            study = self._require_study(study_id)
            if study.get("read_only_fixture") and study.get("findings"):
                return {
                    "ok": True,
                    "findings": study["findings"],
                    "study": self._public_study(study),
                    "note": "Fixture interpretations kept. Not model output.",
                    "provider_call_made": False,
                    "openai_calls": 0,
                    "paid_llm_unlocked": False,
                }
            sessions = study.get("sessions") or []
            session = sessions[0] if sessions else None
            if session and session.get("execution") == "model_interview":
                if not session.get("statements"):
                    usage = study.get("ai_usage") or {}
                    raise ResearchError(
                        "The model session has no statements. Nothing was invented.",
                        400,
                        paid_llm_unlocked=False,
                        openai_calls=int(usage.get("calls") or 0),
                        provider_call_made=False,
                    )
                self._require_approval(body)
                return self._synthesize_model_findings(study)
            if not session or not session.get("statements"):
                raise ResearchError("Run a session before findings.")
            people = {p["id"]: p for p in study.get("participants") or []}
            sources = (study.get("context") or {}).get("sources") or []
            findings = []
            seen_participants: set[str] = set()
            for statement in sessions[0]["statements"]:
                pid = statement.get("participant_id")
                if pid in seen_participants:
                    continue
                seen_participants.add(pid)
                person = people.get(pid) or {}
                role = person.get("role") or "synthetic persona"
                tension = person.get("tension") or "the tension in the first synthetic statement"
                hope = person.get("hope") or "a hope a human still has to check"
                source_ids = [statement["id"]]
                if sources:
                    source_ids.append(sources[0]["id"])
                source_ids.append(study["research_question"]["id"])
                findings.append(
                    {
                        "id": _new_id("find"),
                        "kind": "interpretation",
                        "title": f"Fixture reading for the {role}",
                        "text": (
                            f"Interpretation of one synthetic persona ({role}): the fixture "
                            f"voices “{tension}” and the hope “{hope}”. This is not a rate, "
                            f"a market share, or evidence from human subjects."
                        ),
                        "source_ids": source_ids,
                        "synthesis_method": SYNTHESIS_FIXTURE,
                        "human_validation": {
                            "id": _new_id("hv"),
                            "kind": "human_validation_need",
                            "text": (
                                f"A person must check whether real people in the {role} role "
                                f"recognize this tension. Statement {statement['id']} is synthetic "
                                f"and is not evidence of customer behavior."
                            ),
                        },
                    }
                )
            study["findings"] = findings
            study["synthesis_method"] = SYNTHESIS_FIXTURE
            study["status"] = "findings_ready"
            study["updated_at"] = _now()
            banned = prose_banned_hits({"findings": findings, "sessions": study["sessions"]})
            if banned:
                raise ResearchError(
                    "Findings contained a disallowed claim pattern and were not stored.",
                    500,
                    banned=banned,
                )
            self._persist()
            return {
                "ok": True,
                "findings": findings,
                "study": self._public_study(study),
                "synthesis_method": SYNTHESIS_FIXTURE,
                "provider_call_made": False,
                "openai_calls": int((study.get("ai_usage") or {}).get("calls") or 0),
                "paid_llm_unlocked": False,
            }

    def _synthesize_model_findings(self, study: dict) -> dict[str, Any]:
        """Caller holds the study lock. Does not start the simulation engine."""
        self._block_if_no_credentials(study)
        session = study["sessions"][0]
        statements = session.get("statements") or []
        ledger = StudyLedger.from_study(study)
        calls_before = ledger.calls
        parsed, stop = run_synthesis(study, statements, get_research_provider(), ledger)
        if parsed is None and stop == "budget":
            snap = self._apply_usage(study, ledger, partial=True, stop_reason="budget")
            self._persist()
            return {
                "ok": True,
                "partial": True,
                "findings": study.get("findings") or [],
                "study": self._public_study(study),
                "note": "The study budget cannot cover synthesis. No findings were invented.",
                "synthesis_method": study.get("synthesis_method") or "not_synthesized",
                "provider_call_made": False,
                **self._usage_payload(snap),
            }
        if parsed is None:
            snap = self._apply_usage(
                study,
                ledger,
                partial=True,
                stop_reason=stop or "invalid_model_response",
            )
            self._persist()
            code = 422 if stop == "invalid_model_response" else 502
            raise ResearchError(
                "The model synthesis could not be stored. No findings were invented.",
                code,
                status=stop or "invalid_model_response",
                provider_call_made=ledger.calls > calls_before,
                **self._usage_payload(snap),
            )
        findings = []
        question_id = (study.get("research_question") or {}).get("id")
        for row in parsed.findings:
            source_ids: list[str] = []
            for statement_id in row["statement_ids"]:
                if statement_id not in source_ids:
                    source_ids.append(statement_id)
            if question_id and question_id not in source_ids:
                source_ids.append(question_id)
            cited = set(row["statement_ids"])
            for statement in statements:
                if statement.get("id") not in cited:
                    continue
                for ref in statement.get("source_refs") or []:
                    if ref not in source_ids:
                        source_ids.append(ref)
            findings.append(
                {
                    "id": _new_id("find"),
                    "kind": "interpretation",
                    "title": row["title"],
                    "text": row["text"],
                    "source_ids": source_ids,
                    "synthesis_method": SYNTHESIS_MODEL,
                    "human_validation": {
                        "id": _new_id("hv"),
                        "kind": "human_validation_need",
                        "text": row["human_validation"] or _MODEL_VALIDATION,
                    },
                }
            )
        banned = prose_banned_hits({"findings": findings})
        if banned:
            snap = self._apply_usage(
                study,
                ledger,
                partial=True,
                stop_reason="invalid_model_response",
            )
            self._persist()
            raise ResearchError(
                "Synthesis contained a disallowed claim and was not stored.",
                422,
                status="invalid_model_response",
                banned=banned,
                provider_call_made=True,
                **self._usage_payload(snap),
            )
        partial = bool(stop) or bool(session.get("partial")) or bool(parsed.dropped)
        reason = stop or (session.get("stop_reason") if partial else None)
        study["findings"] = findings
        study["synthesis_method"] = SYNTHESIS_MODEL
        study["status"] = "findings_partial" if partial else "findings_ready"
        snap = self._apply_usage(study, ledger, partial=partial, stop_reason=reason)
        self._persist()
        note = (
            "Partial synthesis. Missing pieces were not invented."
            if partial
            else "Model structured synthesis. Not an offline demo reading."
        )
        return {
            "ok": True,
            "partial": partial,
            "findings": findings,
            "study": self._public_study(study),
            "note": note,
            "synthesis_method": SYNTHESIS_MODEL,
            "provider_call_made": ledger.calls > calls_before,
            **self._usage_payload(snap),
        }

    def export_study(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            study = self._require_study(study_id)
            project = self.projects.get(study.get("project_id") or "")
            if project is None:
                raise ResearchError("Project not found.", 404)
            missing = unresolved_source_ids(study)
            kinds = provenance_kinds_present(study)
            banned = prose_banned_hits(
                {
                    "disclaimer": DISCLAIMER,
                    "participants": study.get("participants"),
                    "sessions": study.get("sessions"),
                    "findings": study.get("findings"),
                    "audience": {
                        "name": (study.get("audience") or {}).get("name"),
                        "description": (study.get("audience") or {}).get("description"),
                    },
                }
            )
            count = len(study.get("participants") or [])
            usage = study.get("ai_usage") or {}
            session = (study.get("sessions") or [{}])[0]
            partial = bool(usage.get("partial") or session.get("partial"))
            method = _stored_synthesis_method(study)
            complete = (
                not missing
                and not banned
                and 3 <= count <= 5
                and bool(study.get("sessions"))
                and bool(study.get("findings"))
                and bool((study.get("research_question") or {}).get("text"))
                and bool((study.get("context") or {}).get("text"))
                and not partial
            )
            return {
                "schema": EXPORT_SCHEMA,
                "exported_at": _now(),
                "product": "NanoCrab Lab",
                "mvp": "1.0",
                "api_version": "v1",
                "disclaimer": disclaimer_for_study(study),
                "project": {
                    "id": project.get("id"),
                    "name": project.get("name"),
                    "organization": project.get("organization"),
                    "kind": "user_input",
                },
                "brief": {
                    "research_question": study.get("research_question"),
                    "context": study.get("context"),
                },
                "study": {
                    "id": study.get("id"),
                    "title": study.get("title"),
                    "status": study.get("status"),
                    "example_slug": study.get("example_slug"),
                    "audience": study.get("audience"),
                    "participant_generation": study.get("participant_generation"),
                    "synthesis_method": method,
                    "participants": study.get("participants"),
                    "sessions": study.get("sessions"),
                    "findings": study.get("findings"),
                },
                "source_index": kind_index(study),
                "integrity": {
                    "ok": complete,
                    "provenance_kinds_present": kinds,
                    "unresolved_source_ids": missing,
                    "banned_claim_patterns": banned,
                    "participant_count": count,
                    "roster_count_is_not_a_sample_size": True,
                    "synthetic_not_human_research": True,
                    "not_research_validated": True,
                    "no_representativeness_claim": True,
                    "genuine_ai_research": _stored_genuine(study),
                    "synthesis_method": method,
                    "partial": partial,
                    "openai_calls": int(usage.get("calls") or 0),
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                    "cost_usd": float(usage.get("cost_usd") or 0),
                    "paid_llm_unlocked": False,
                    "user_input_contains_percent_pattern": bool(
                        study.get("user_input_contains_percent_pattern")
                    ),
                },
            }

    def list_audiences(self) -> dict[str, Any]:
        with self._lock:
            rows = []
            for study in self.studies.values():
                audience = study.get("audience")
                if not audience:
                    continue
                project = self.projects.get(study.get("project_id") or "", {})
                rows.append(
                    {
                        "id": audience.get("id"),
                        "name": audience.get("name"),
                        "description": audience.get("description"),
                        "synthetic": True,
                        "representativeness": None,
                        "study_id": study.get("id"),
                        "study_title": study.get("title"),
                        "project_name": project.get("name"),
                        "participant_count": len(study.get("participants") or []),
                    }
                )
        return {
            "ok": True,
            "audiences": rows,
            "disclaimer": dict(DISCLAIMER),
            "note": "Synthetic rosters are not a market sample.",
        }

    def list_reports(self) -> dict[str, Any]:
        with self._lock:
            rows = []
            for study in self.studies.values():
                if not study.get("findings"):
                    continue
                project = self.projects.get(study.get("project_id") or "", {})
                rows.append(
                    {
                        "study_id": study.get("id"),
                        "title": study.get("title"),
                        "project_name": project.get("name"),
                        "example_slug": study.get("example_slug"),
                        "finding_count": len(study.get("findings") or []),
                        "participant_count": len(study.get("participants") or []),
                        "export_path": f"/api/v1/research/studies/{study.get('id')}/export",
                        "genuine_ai_research": _stored_genuine(study),
                        "synthesis_method": _stored_synthesis_method(study),
                        "not_research_validated": True,
                    }
                )
        return {"ok": True, "reports": rows, "disclaimer": dict(DISCLAIMER)}

    def load_example(self, slug: str) -> dict[str, Any]:
        with self._lock:
            existing = self._study_for_slug(slug)
            if existing is None:
                if get_example(slug) is None:
                    raise ResearchError(f"Unknown example '{slug}'.", 404)
                existing = self._insert_example(slug)
                self._persist()
            return {"ok": True, "study": self._public_study(existing)}
