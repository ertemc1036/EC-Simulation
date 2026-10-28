# Phase 8.0 — Scalable Simulation Foundation (Architecture)

Basis: Phase 7.3.2 (PR #18, `cursor/phase7.3.2-stance-diversity-610e` @ `9f6ba22`).  
**Offline Foundation.** Kein OpenAI-/Live-LLM, kein Merge.

Live-LLM-Ceilings **unverändert:** 5 / 2 / 10 / 20k / 500 / $0.20 · `paid_llm_unlocked=false`.

## Trennung Live vs Offline

| Pfad | Agenten | Runden | Paid LLM |
|------|---------|--------|----------|
| Live-LLM | ≤ **5** | ≤ **2** | ja (Freigabe) |
| Manual / Scripted | ≤ **50** (10/25/50) | ≤ **4** | nein |

## Komponenten

### Persona Engine V2
- Education-Domain: Lehrkräfte / Eltern / Bildungsforscher
- Topic-Theme-Denylist (Datenschutz, Chancengleichheit, … ≠ Stakeholder)
- Declined-Form-Canon; deutsche Individual-Pools; Occupation ≠ Gruppenlabel
- Fix Live-Fixture `20261028T122857Z-ddfedb31`
- 7.3.2 Labor-Rollen (Arbeitnehmer / Unternehmen / Wirtschaftsexperte) bleiben

### Scalable Agent Scheduler
- `SocialInteractionScheduler` V2: Wellen (`plan_round`), Partner-Balance
- Keine Selbstreferenz; Coverage-/Wave-Metriken im Dashboard

### Cost & Token Planner
- `cost_token_planner.plan_run` — Budgets vor dem Lauf
- Live-Plan unter festen Ceilings; Offline/Scripted Provider=$0
- `paid_llm_unlocked` immer **false**

### Simulation Intelligence
- Aggregat: Persona-Diversity, Scheduler, Cost-Plan, Stance, Society-KPIs
- `quality_fix=8.0`, `persona_engine=v2`

### Dashboard
- Phase-8 Banner, Intelligence- & Cost-Plan-Karten, stale-Backend-Check auf Phase 8

## API / Status
- `config.phase = "8"`
- `config.scalable_foundation`, `society_intelligence.quality_fix=8.0`
- State: `cost_plan`, `simulation_intelligence`
- Health: `cost_token_planner` Sample-Pläne

## Nicht im Scope
- Kein Merge, kein Live-LLM, keine Limit-Erhöhung für Paid-LLM, kein Zep.
