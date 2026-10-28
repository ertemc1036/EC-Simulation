# Phase 8.5 — Realistic Multi-Agent Behavior (Architecture)

Basis: Phase 8.4 (PR #24, `cursor/phase8.4-simulation-reliability-17fa` @ `c804fd6`).  
**Offline Realistic Multi-Agent.** Kein OpenAI-/Live-LLM, kein Merge.

Live-LLM-Ceilings **unverändert:** 5 / 2 / 10 / 20k / 500 / $0.20 · `paid_llm_unlocked=false`.

## Komponenten

### Stakeholder Motivation Engine V5
- Modul: `persona_motivation.py` (`persona_engine=v5`)
- Neu: `stakes`, `claim_agenda`, `engagement_bias`, `stakeholder_group`
- Domains: education / labor / income / **mobility**
- `stakeholder_affinity` für Multi-Agent-Paarungs-Signale
- Prompt-Block enthält Stakes + Agenda

### Claim-Based Interaction Engine V4
- Modul: `claim_interaction.py` (`engine=v4`)
- Extrahiert atomare Claims aus Parent-Kommentaren
- Misst `claim_address_rate` / `mean_claim_coverage` pro Reply
- Ergänzt Argument-Grounded V4 (Token-/Themen-Overlap bleibt)

### Argument Evolution V4
- Modul: `argument_intelligence.py` (`engine=v4`)
- Evolution-Stages: introduce → support/develop → challenge → refine
- Themen-Trajektorien, `evolution_rate`, `challenge_chain_count`
- Debate Quality nutzt Evolution-Boost

### Interaction Quality V5
- Modul: `interaction_quality.py` (`engine=v5`)
- Composite: Grounding + Claim-Address + Motivation/Stakes + Evolution + Stance/Debate
- Quality Dashboard V5 (Claim Address %, Arg. Evolution, Stake Coverage)

## Offline Scale
- Presets 10 / 25 / 50 × ≤4 Runden (Scripted/Manual)
- Szenarien A–E offline; adversarial Grounding (digitale Kluft)
- Fixtures: `e251172c`, `d1fb1b1c`, `ab541048`
- Live-Ceilings unberührt; Provider-USD $0 offline

## Nicht im Scope
- Kein Merge, kein Live-LLM-Test, keine Limit-Erhöhung, kein Zep.
