# EC-Simulation (Phase 1)

Multi-Agent Social Simulation auf **camel-oasis 0.2.5** mit Flask-Backend und Vue/TypeScript-Dashboard.

## Voraussetzungen

- Python **3.11** (OASIS: `>=3.10,<3.12`)
- Node.js 20+ (für Frontend)
- Lokales `.venv` im Projektroot (bereits vorhanden)

## Wichtige Package-Pins

| Paket | Version | Hinweis |
|-------|---------|---------|
| `camel-oasis` | `0.2.5` | OASIS-Runtime |
| `camel-ai` | `0.2.78` | von oasis gepinnt |
| **`mcp`** | **`>=1.3.0,<2`** (getestet: **1.30.0**) | mcp 2.x bricht `import oasis` (`FastMCP`) |
| `flask` | `3.1.x` | API |
| `vue` / `vite` | siehe `frontend/package.json` | Dashboard |

## Setup

```bash
cd ~/Desktop/EC-Simulation
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Optional: OPENAI_API_KEY setzen für LLM-Modus
# Ohne Key: ManualAction-Modus (echte OASIS-Env, StubModel, keine API-Calls)

cd frontend && npm install && cd ..
```

## Start

Terminal 1 — Backend:

```bash
cd ~/Desktop/EC-Simulation
source .venv/bin/activate
python -m backend.app
# → http://127.0.0.1:5001/api/health
```

Terminal 2 — Frontend:

```bash
cd ~/Desktop/EC-Simulation/frontend
npm run dev
# → http://127.0.0.1:5173
```

## Verhalten ohne API-Key

- Dashboard zeigt klar: **Kein API-Key / ManualAction-Modus**
- Simulation startet mit `mode=manual` oder `auto` über echte `oasis.ManualAction` + `OasisEnv.step`
- Keine Fake-DB, keine vorgetäuschten LLM-Antworten
- `mode=llm` ohne Key → Status `blocked` mit Begründung

## API (Kurz)

- `GET /api/health` — OASIS-Import, Versionen, LLM/Zep-Status
- `GET /api/agents` — fünf Agentenprofile
- `POST /api/simulation/start` — `{ "scenario", "mode": "auto|manual|llm", "max_steps" }`
- `GET /api/simulation` — Live-State
- `GET /api/results` — gespeicherte JSON-Läufe unter `data/results/`
- Socket.IO Event `simulation_update` für Live-Updates

## Tests

```bash
cd ~/Desktop/EC-Simulation
source .venv/bin/activate
pytest -q
```

## Zep

Stub unter `backend/zep/` — keine Pflicht-Calls. Aktivierung später via `ZEP_ENABLED=true` + `ZEP_API_KEY` (optional `pip install zep-cloud`).

## Kosten-/Sicherheitslimits (`.env`)

- `MAX_SIMULATION_STEPS` (Default 8)
- `MAX_LLM_CALLS_PER_RUN` (Default 40)
- `MAX_TOKENS_PER_CALL` (Default 512)
- `REQUEST_TIMEOUT_SECONDS`
- `ALLOW_LLM_MODE`
