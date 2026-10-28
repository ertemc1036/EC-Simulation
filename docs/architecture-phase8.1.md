# Phase 8.1 — Controlled Multi-Agent Scaling (Architecture)

Basis: Phase 8.0 (PR #19, `cursor/phase8.0-scalable-foundation-3f95` @ `20fb6da`).  
**Offline Scaling.** Kein OpenAI-/Live-LLM, kein Merge.

Live-LLM-Ceilings **unverändert:** 5 / 2 / 10 / 20k / 500 / $0.20 · `paid_llm_unlocked=false`.

## Trennung Live vs Offline

| Pfad | Agenten | Runden | Paid LLM | Scheduler-Wellen in Execution |
|------|---------|--------|----------|-------------------------------|
| Live-LLM | ≤ **5** | ≤ **2** | ja (Freigabe) | `plan_round` + 1 Wave `env.step` (Plans advisory) |
| Manual | ≤ **50** | ≤ **4** | nein | `plan_round` + wave-batched ManualAction |
| Scripted | ≤ **50** | ≤ **4** | nein | `plan_round` + wave `env.step` + **`take_plan`** |

## Komponenten

### Scalable Scheduler V3
- Modul: `social_scheduler.py` (`local_scheduler_v3`)
- `plan_round` speichert Pending-Pläne; Execution via `take_plan` / `mark_wave_executed`
- Wave-Size default **10** → 10→1, 25→3, 50→5 Wellen/Runde

### Multi-Agent Interaction Engine
- Modul: `interaction_engine.py` (`multi_agent_interaction_v3`)
- `prepare_round` vor `env.step`; `wave_agent_groups` für Batch-Execution
- Spätere Wellen sehen frühere Feed-Inhalte (kontrolliertes Scaling)

### Persona Diversity V3
- Stakeholder-Cycling bei 10/25/50; erweiterte Education-Pools
- Diversity-Score: unique names/occupations/roles + role_balance

### Cost & Token Planner V3
- Zusätzlich: `planned_waves`, `calls_per_wave`, `planner_version=3`
- Offline/Scripted Provider=$0; Live-Ceilings unverändert

### Dashboard V3
- Waves Planned / Executed, Interaction Engine Path, Persona V3, Cost V3

## Welche Pfade nutzen Scheduler-Wellen wirklich?

| Path | `plan_round` | Wave `env.step` | `take_plan` konsumiert |
|------|--------------|-----------------|------------------------|
| **scripted** | ja | ja | **ja** (ScriptedReactiveModel) |
| **manual** | ja | ja (bei N>wave_size mehrere) | nein (ManualAction-Script) |
| **live_llm** | ja | ja (meist 1 Wave) | nein (OpenAI entscheidet frei) |

## Nicht im Scope
- Kein Merge, kein Live-LLM-Test, keine Limit-Erhöhung, kein Zep.
