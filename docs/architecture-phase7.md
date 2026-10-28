# Phase 7 — Society Intelligence (Architecture)

Basis: Phase 6 (PR #12, `cursor/phase6-scalable-society-d029`).  
**Offline Society Intelligence.** Kein OpenAI-/Live-LLM, kein Merge.

Live-LLM-Ceilings **unverändert:** 5 / 2 / 10 / 20k / 500 / $0.20 · `paid_llm_unlocked=false`.

## Trennung Live vs Offline

| Pfad | Agenten | Runden | Paid LLM |
|------|---------|--------|----------|
| Live-LLM | ≤ **5** | ≤ **2** | ja (Freigabe) |
| Manual / Scripted | ≤ **50** (10/25/50) | ≤ **4** | nein |

## Komponenten

### Langfristiges lokales Agentengedächtnis
- `LocalAgentMemory`: Working-Ringpuffer + Long-Term-Tier.
- Long-Term nur mit **Sources** (`post:`, `comment:`, `action:`).
- Caps: `AGENT_LOCAL_MEMORY_ENTRIES`, `AGENT_LONG_TERM_MEMORY_ENTRIES`.
- **Kein Zep** (`zep_integrated=false`).

### Dynamische Beziehungen (evidenced-only)
- `RelationshipDynamics`: Actor↔Actor-Bonds aus Reaktionen, →Kommentar#, `influenced_by`.
- Strength wächst nur mit Belegen; Self-Pairs verworfen; Rebuild pro Sync.
- Chronologie der Bond-Events fürs Dashboard.

### Meinungsdynamik + Chronologie
- `OpinionDynamics` auf `PositionTracker`-Changes.
- Round-Snapshots: Polaritätsverteilung, Δ/Σ Changes, Transitionen.
- Scripted-Szenarien (R1 establish → R2 react → R3 shift → R4 consolidate).

### Society Analytics Dashboard
- `society_analytics` Payload + Frontend-Abschnitt.
- KPIs: memory_coverage, relationship_density, opinion_volatility, social_edge_total.
- Persistiert in Result-JSON.

### OASIS / Scale / Tokens
- OASIS unverändert (kein `parent_comment_id`).
- Phase-6 Scheduler + Generator + Live-Ceilings bleiben.
- Scripted N×Runden-Budget ohne Paid-LLM-Freischaltung.

## API / Status
- `config.phase = "7"`
- `config.society_intelligence`, `offline_scale.paid_llm_unlocked=false`
- State: `relationships`, `opinion_dynamics`, `society_analytics`

## Nicht im Scope
- Kein Merge, kein Live-LLM, kein Zep-SDK, keine Limit-Erhöhung für Paid-LLM.
