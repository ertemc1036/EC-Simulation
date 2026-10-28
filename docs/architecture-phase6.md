# Phase 6 — Scalable Multi-Agent Society (Architecture)

Basis: Phase 5.2 (PR #11). Offline/Scripted scale **10 / 25 / 50**.  
**Live-LLM-Ceilings unverändert:** 5 Agenten / 2 Runden / 10 Calls / 20k Tokens / 500 Tokens/Call / $0.20.

## Trennung Live vs Offline

| Pfad | Agenten | Runden | Calls/Budget | Zweck |
|------|---------|--------|--------------|-------|
| Live-LLM (OpenAI) | ≤ **5** | ≤ **2** | ≤ **10**, 20k, $0.20 | kostenpflichtig, Freigabe nötig |
| ManualAction | ≤ **50** | ≤ **4** | 0 Provider-Calls | Offline-Demo |
| ScriptedModel (Tests) | ≤ **50** | ≤ **4** | N×Runden (kein Paid API) | Regression 10/25/50 |

`LIVE_TEST_MAX_*` werden **nicht** angehoben. `OFFLINE_MAX_AGENTS=50` und Presets `(10,25,50)` gelten nur Manual/Scripted.

## Komponenten

### OASIS
- Unverändert: Reddit-Platform, flache Comments, **kein** `parent_comment_id`.
- Semaphore: Live=1; Scripted parallel höher für Scale-Laufzeiten.
- Pro Run frische SQLite-DB.

### Agentengenerator
- `MAX_GENERATABLE_AGENTS=50`, deterministisch, keine Klone.
- `fixed_test` bleibt klassisches 5er-Roster.
- `dynamic` füllt Stakeholder + Archetypen bis N.

### Memory
- `LocalAgentMemory` Ringpuffer unverändert; bei N=50 mehr Keys, gleiches Cap/Agent.
- ChatHistory-Fenster & Phase-4.1 Caps bleiben.
- Zep weiterhin nur Stub.

### Social Dynamics / PositionTracker (5.2 +)
- Seed-Posts ignorieren, `exception`-Polarität, textuelle Agree/Disagree nur belegt.
- Neu: `polarity_distribution` für Dashboard-Histogramm.
- Relationen weiterhin evidenced-only.

### Social Interaction Scheduler
- Modul `social_scheduler.py`: Lesen, Partnerwahl, Post-/Kommentar-Ziele.
- **Keine Selbstreferenz** als Inter-Agent-Partner.
- ScriptedModel nutzt den Scheduler für deterministische Aktionen.

### Knowledge Graph
- Skaliert mit Actor-/Statement-Knoten; `network`-Summary (actors/statements/social_edges).
- Keine erfundenen Kanten.

### Token-Limits / CostTracker
- Live: Reserve/Approval/Ceilings wie Phase 3.5–5.2.
- Scripted: Call-/Token-Budget = N×Runden (synthetisch), **ohne** Paid-LLM-Freischaltung.

### Dashboard
- Offline-Presets 10/25/50 (nur Manual).
- Society-Übersicht, Positionsverteilung, Netzwerk-Sample, Progress.
- Klarer Modus: Manual / Scripted / Live-LLM.

## API / Status
- `config.phase = "6"`
- `config.live_test_limits` + `config.offline_scale`
- State: `execution_kind`, `scheduler`, `society_overview`

## Nicht im Scope
- Kein Merge in main durch diesen Agenten.
- Kein OpenAI-/Live-LLM-Lauf in Phase 6.
- 10/25/50 **nicht** für kostenpflichtiges LLM freischalten.
