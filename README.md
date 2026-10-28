# EC-Simulation (Phase 6)

Multi-Agent Social Simulation auf **camel-oasis 0.2.5** mit Flask-Backend und Vue/TypeScript-Dashboard.

Phase 1–5.2: Manual/LLMAction, Freigabe, Token-Gates, Social Dynamics. **Phase 6:** skalierbare Offline-Gesellschaft **10/25/50** (Manual/Scripted). Live-LLM-Ceilings bleiben **5/2/10/20k/500/$0.20** — Scale schaltet kein bezahltes LLM frei. Architektur: `docs/architecture-phase6.md`.

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

## Agenten — Live vs Offline-Scale

| Modus | Bedeutung |
|-------|-----------|
| `dynamic` | Einzigartige Personas aus Szenario-Keywords (keine Klone) |
| `fixed_test` | Klassisches Alice…Eve-Roster (Testmodus, max. 5) |

| Pfad | Agenten |
|------|---------|
| Live-LLM | 1–**5** (unverändert) |
| Manual / Scripted Offline | 1–**50**, Presets **10/25/50** |

Start-Body: `agent_count`, `agent_generation`. Siehe `docs/architecture-phase6.md`.

## ManualAction vs LLMAction

| Modus | Engine | Modell | Kosten |
|-------|--------|--------|--------|
| `manual` | echte `OasisEnv` + `ManualAction` | CAMEL StubModel | keine API-Calls |
| `llm` | echte `OasisEnv` + `LLMAction` | OpenAI-kompatibel (`OPENAI_*`) | Calls/Tokens limitiert |
| `auto` | wird wie `manual` behandelt (kein stilles LLM) | — | — |

LLM-Interaktionen (keine Fake-Entscheidungen): Agenten lesen den Feed und wählen echte OASIS-Tools — `create_post`, `create_comment` (Reply auf `post_id`; OASIS hat **kein** `parent_comment_id`), `like_post` / `dislike_post`, `like_comment` / `dislike_comment`, …

**Modellaufrufe pro Runde:** 1 Call pro Agent. Bei 5 Agenten × 2 Runden = 10 Calls (Ceiling). Seed-`ManualAction` kostet 0 Calls.

**Endzustände:** `completed` | `limit_reached` | `failed` (+ Lifecycle `idle` / `running` / `blocked`).

## Memory & Wissensgraph

- **Lokal:** `LocalAgentMemory` (Ringpuffer je Agent) + CAMEL `ChatHistoryMemory(window_size=AGENT_MEMORY_WINDOW)`
- **Zep:** Stub unter `backend/zep/` — **nicht integriert**; Handoff-Form vorbereitet
- **Graph / Social Dynamics (Phase 5):** persistente Agentenpositionen + Social-Kanten nur belegt (`agrees_with`, `disagrees_with`, `responds_to`, `influenced_by`, `position_changed`); lokale `→Kommentar#id`-Refs (OASIS hat kein `parent_comment_id`)

## Verhalten ohne API-Key

- Dashboard: **Kein API-Key / ManualAction-Modus**
- `mode=manual` / `auto` → echte ManualAction-Simulation
- `mode=llm` → Status `blocked`, keine Simulation, keine Fake-Antworten
- API-Keys erscheinen nie im Frontend und nicht in `/api/health`

## API (Kurz)

- `GET /api/health` — OASIS-Import, Versionen, LLM/Zep-Status (ohne Secrets)
- `GET /api/agents?scenario=&count=&generation=dynamic|fixed_test` — Profil-Preview
- `POST /api/simulation/start` — `{ scenario, mode, max_steps, language?, agent_count?, agent_generation?, llm_run_approved? }`
- `GET /api/simulation` — Live-State inkl. Feed, Memory, Knowledge-Graph, Kosten
- `GET /api/results` — JSON-Läufe unter `data/results/`
- Socket.IO Event `simulation_update`

## Tests

```bash
cd ~/Desktop/EC-Simulation
source .venv/bin/activate
pytest -q
```

LLM-Pfad-Tests nutzen `ScriptedReactiveModel` (echte `LLMAction`, **keine** Provider-Calls).

## Kosten-/Sicherheitslimits (`.env`)

Phase-2.x-Ceilings bleiben:

- `MAX_SIMULATION_STEPS` (Default 2)
- `MAX_LLM_CALLS_PER_RUN` (Default 10)
- `MAX_TOKENS_PER_CALL` / `MAX_TOKENS_PER_RUN` / `MAX_COST_USD_PER_RUN`
- `LLM_MAX_RETRIES` (Default 0)
- `AGENT_MEMORY_WINDOW` / `AGENT_LOCAL_MEMORY_ENTRIES`
- `DEFAULT_AGENT_COUNT` / `AGENT_GENERATION_MODE`
- `ALLOW_LLM_MODE` / `SIMULATION_LANGUAGE`
- Freigabe: `llm_run_approved=true` für LLM-Starts

Parallel: nur eine Simulation gleichzeitig. Pro Lauf frische SQLite-DB — kein DB-Reuse.

## Zep

Stub unter `backend/zep/` — **kein „Zep integriert“-Claim**. Aktivierung später via `ZEP_ENABLED=true` + `ZEP_API_KEY` + SDK-Wiring.
