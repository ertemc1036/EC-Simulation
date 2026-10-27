# EC-Simulation (Phase 2)

Multi-Agent Social Simulation auf **camel-oasis 0.2.5** mit Flask-Backend und Vue/TypeScript-Dashboard.

Phase 1: echte `ManualAction`-Läufe. Phase 2: echte OASIS `LLMAction` mit OpenAI-kompatiblem Modell, Zielen, begrenztem Erinnerungskontext, Kostenlimits und getrennter Protokollierung von Modellaufrufen vs. Platform-Aktionen.

## Voraussetzungen

- Python **3.11** (OASIS: `>=3.10,<3.12`)
- Node.js 20+ (für Frontend)
- Lokales `.venv` im Projektroot

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
# Optional: OPENAI_API_KEY setzen für echten LLM-Modus
# Ohne Key: ManualAction-Modus (echte OASIS-Env, StubModel, keine API-Calls)
# LLM-Modus ohne Key → blocked (keine Fake-KI)

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

## ManualAction vs LLMAction

| Modus | Engine | Modell | Kosten |
|-------|--------|--------|--------|
| `manual` | echte `OasisEnv` + `ManualAction` | CAMEL StubModel | keine API-Calls |
| `llm` | echte `OasisEnv` + `LLMAction` | OpenAI-kompatibel (`OPENAI_*`) | Calls/Tokens limitiert |
| `auto` | LLM falls Key + `ALLOW_LLM_MODE`, sonst Manual | — | — |

Dashboard kennzeichnet Quellen im Aktionsprotokoll und führt **Modellaufrufe/Kosten** getrennt.

## Verhalten ohne API-Key

- Dashboard: **Kein API-Key / ManualAction-Modus**
- `mode=manual` / `auto` → echte ManualAction-Simulation
- `mode=llm` → Status `blocked`, keine Simulation, keine Fake-Antworten
- API-Keys erscheinen nie im Frontend und nicht in `/api/health`

## API (Kurz)

- `GET /api/health` — OASIS-Import, Versionen, LLM/Zep-Status (ohne Secrets)
- `GET /api/agents` — fünf Agentenprofile inkl. Ziele
- `POST /api/simulation/start` — `{ "scenario", "mode": "auto|manual|llm", "max_steps" }`
- `GET /api/simulation` — Live-State inkl. `action_log`, `model_calls`, `cost_summary`
- `GET /api/results` — JSON-Läufe unter `data/results/`
- Socket.IO Event `simulation_update`

## Tests

```bash
cd ~/Desktop/EC-Simulation
source .venv/bin/activate
pytest -q
```

LLM-Pfad-Tests nutzen eine injizierte `ScriptedReactiveModel` (echte `LLMAction`, **keine** Provider-Calls). Echter OpenAI-Lauf braucht Key + explizite Freigabe.

## Kosten-/Sicherheitslimits (`.env`)

- `MAX_SIMULATION_STEPS` (Default 8)
- `MAX_LLM_CALLS_PER_RUN` (Default 40)
- `MAX_TOKENS_PER_CALL` (Default 512)
- `MAX_TOKENS_PER_RUN` (Default 20000)
- `AGENT_MEMORY_WINDOW` (Default 8)
- `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` (Dashboard-Schätzung)
- `REQUEST_TIMEOUT_SECONDS`
- `ALLOW_LLM_MODE`

## Zep

Stub unter `backend/zep/` — keine Pflicht-Calls. Aktivierung später via `ZEP_ENABLED=true` + `ZEP_API_KEY`.
