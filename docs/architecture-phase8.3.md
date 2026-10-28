# Phase 8.3 — Realistic Agent Behavior (Architecture)

Basis: Phase 8.2 (PR #21, `cursor/phase8.2-debate-quality-891b` @ `d4c6f0b`).  
**Offline Realistic Agents.** Kein OpenAI-/Live-LLM, kein Merge.

Live-LLM-Ceilings **unverändert:** 5 / 2 / 10 / 20k / 500 / $0.20 · `paid_llm_unlocked=false`.

## Komponenten

### Persona Motivation Engine V4
- Modul: `persona_motivation.py` (`persona_engine=v4`)
- Werte / Risiken / Treiber / decision_style je Rolle+Domain
- Goals tragen `Motivation:`-Block; Prompt-Konditionierung

### Argument-Grounded Interaction V3
- Modul: `interaction_quality.py`
- Antwortbezug: Token-/Themen-Overlap + Attribution-Check
- Erfundene Eltern-Themen (Fixture Thomas Bauer «digitale Kluft») → `invented_attribution`

### Stance Classification V3
- Modul: `social_dynamics.py` (`stance_classifier=v3`)
- Potenzial + Voraussetzungen / Evidenzlücke / Vollausrollung riskant → `conditional`
- Fixture `d1fb1b1c`: Thomas Bauer unknown → conditional

### Opinion Dynamics V3
- Modul: `opinion_dynamics.py` (`engine=v3`)
- V2-Metriken + `grounded_reply_share` / `motivation_coverage`

### Interaction Quality Evaluation
- Composite Score: Grounding × Stance-Klarheit × Debate × Motivation − Invented Attr.
- Quality Dashboard V3 im Frontend

## Offline Scale
- Presets 10 / 25 / 50 × ≤4 Runden (Scripted/Manual)
- Live-Ceilings unberührt; Provider-USD $0 offline

## Nicht im Scope
- Kein Merge, kein Live-LLM-Test, keine Limit-Erhöhung, kein Zep.
