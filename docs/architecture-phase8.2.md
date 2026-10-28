# Phase 8.2 — Simulation Intelligence & Debate Quality (Architecture)

Basis: Phase 8.1 (PR #20, `cursor/phase8.1-multi-agent-scaling-1ee3` @ `032ffe6`).  
**Offline Debate Quality.** Kein OpenAI-/Live-LLM, kein Merge.

Live-LLM-Ceilings **unverändert:** 5 / 2 / 10 / 20k / 500 / $0.20 · `paid_llm_unlocked=false`.

## Komponenten

### Argument Intelligence V2
- Modul: `argument_intelligence.py`
- Claim-Arten (support, caveat, conditional_claim, evidence_demand, rebuttal, …)
- Quality-Score, Themenvielfalt, Address-Rate

### Debate Engine V2
- Modul: `discussion_engine.py` (`debate_engine=v2`)
- Snapshot inkl. `argument_intelligence` + `debate_quality`
- Composite Score: Argument-Qualität × Diversity × Stance-Klarheit − Repetition

### Stance Classification V2
- Modul: `social_dynamics.py` (`stance_classifier=v2`)
- Soft-Reject flächendeckend/ungeprüft + Chancen/Voraussetzungen → `conditional`
- Fixture `ab541048`: Lehrerin/Eltern nicht länger contra/unknown

### Opinion Dynamics V2
- Modul: `opinion_dynamics.py` (`engine=v2`)
- Stance-Klarheit, Stabilität, Polarisierung, Unknown-Share-Serie

### Quality Dashboard
- Frontend: Debate Quality, Stance Clarity, Claim-Arten, Argument Diversity
- Simulation Intelligence KPIs (`debate_quality_score`, `stance_clarity`, …)

## Offline Scale
- Presets 10 / 25 / 50 × ≤4 Runden (Scripted/Manual)
- Live-Ceilings unberührt; Provider-USD $0 offline

## Nicht im Scope
- Kein Merge, kein Live-LLM-Test, keine Limit-Erhöhung, kein Zep.
