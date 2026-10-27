"""Flask API + Socket.IO live updates for EC-Simulation Phase 1."""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import settings  # noqa: E402
from backend.simulation.agents import profiles_payload  # noqa: E402
from backend.simulation.engine import SimulationEngine  # noqa: E402

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": settings.cors_origins}})
socketio = SocketIO(app, cors_allowed_origins=settings.cors_origins, async_mode="threading")

engine = SimulationEngine(settings)


def _broadcast(data: dict) -> None:
    socketio.emit("simulation_update", data)


engine.add_listener(_broadcast)


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@app.get("/api/health")
def health():
    oasis_ok = False
    oasis_error = None
    try:
        import oasis  # noqa: F401

        oasis_ok = True
    except Exception as exc:  # pragma: no cover
        oasis_error = str(exc)

    return jsonify(
        {
            "ok": True,
            "oasis_import_ok": oasis_ok,
            "oasis_error": oasis_error,
            "versions": {
                "camel-oasis": _pkg_version("camel-oasis"),
                "camel-ai": _pkg_version("camel-ai"),
                "mcp": _pkg_version("mcp"),
                "flask": _pkg_version("flask"),
                "openai": _pkg_version("openai"),
            },
            "config": settings.status_payload(),
            "zep": engine.zep.status,
        }
    )


@app.get("/api/agents")
def agents():
    return jsonify({"agents": profiles_payload()})


@app.get("/api/status")
def status():
    return jsonify(
        {
            "config": settings.status_payload(),
            "zep": engine.zep.status,
            "simulation": engine.get_state(),
        }
    )


@app.get("/api/simulation")
def simulation_state():
    return jsonify(engine.get_state())


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


@app.post("/api/simulation/start")
def simulation_start():
    body = request.get_json(silent=True) or {}
    scenario = (body.get("scenario") or "").strip()
    if not scenario:
        return jsonify({"ok": False, "error": "scenario ist erforderlich"}), 400
    mode = body.get("mode")  # manual|llm (auto → manual; never silent LLM)
    max_steps = body.get("max_steps")
    try:
        max_steps_i = int(max_steps) if max_steps is not None else None
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "max_steps muss eine Zahl sein"}), 400

    # Explicit per-run approval required for LLM (backend-enforced, not UI-only).
    llm_run_approved = _truthy(
        body.get("llm_run_approved", body.get("confirm_llm_spend", False))
    )

    result = engine.start(
        scenario,
        mode=mode,
        max_steps=max_steps_i,
        llm_run_approved=llm_run_approved,
    )
    code = 200 if result.get("ok") else 409
    return jsonify(result), code


@app.get("/api/results")
def list_results():
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(settings.results_dir.glob("*.json"), reverse=True)
    return jsonify(
        {
            "results": [
                {"name": f.name, "path": str(f), "size": f.stat().st_size}
                for f in files
            ]
        }
    )


@app.get("/api/results/<name>")
def get_result(name: str):
    if "/" in name or "\\" in name or not name.endswith(".json"):
        return jsonify({"error": "invalid name"}), 400
    path = settings.results_dir / name
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(settings.results_dir, name)


# Serve built frontend if present
FRONTEND_DIST = ROOT / "frontend" / "dist"


@app.get("/")
def index():
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify(
        {
            "message": "EC-Simulation API. Frontend: cd frontend && npm run dev",
            "health": "/api/health",
        }
    )


@app.get("/assets/<path:filename>")
def frontend_assets(filename: str):
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        return send_from_directory(assets, filename)
    return jsonify({"error": "not found"}), 404


def main() -> None:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    socketio.run(
        app,
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
