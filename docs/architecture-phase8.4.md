# Phase 8.4 — Simulation Reliability (Architecture)

Basis: Phase 8.3.1 (PR #23, `cursor/phase8.3.1-quality-audit-bf02` @ `3abc376`).  
**Offline Reliability.** Kein OpenAI-/Live-LLM, kein Merge.

Live-LLM-Ceilings **unverändert:** 5 / 2 / 10 / 20k / 500 / $0.20 · `paid_llm_unlocked=false`.

## Komponenten

### Stance Classification V4
- Modul: `social_dynamics.py` (`stance_classifier=v4`)
- Bare `dafür` entfernt (False-pro «plädiere ich dafür, dass … nicht»)
- Conditional-Lexikon: schrittweise, Elternbeteiligung, Datenschutzregeln, pädagogische Konzepte
- False-pro Guard: Soft-Support + Caveat-Lexikon → `conditional`
- Shift + klares Ziel → Ziel-Polarität (nicht bare `shifting`)
- Fixture `e251172c`: Thomas C2 / Fatima C4 → conditional; unknown=0

### Argument-Grounded Interaction V4
- Modul: `interaction_quality.py` (`engine=v4`)
- Grounding-Floor: `weak_overlap` bei overlap < 0.20 trotz grounded
- Bildung↔IT Themen-Bag (C6-Klasse)
- Adversarial «digitale Kluft» bleibt invented (d1fb1b1c)

### Opinion Dynamics V4
- Modul: `opinion_dynamics.py` (`engine=v4`, `version=4`)
- `shifting` zählt als unklar für Stance-Klarheit
- `shifting_share` + `weak_overlap_rate` Signale
- Scripted Mind-Change-Templates mit klarem Conditional-Ziel

### Quality Dashboard V4
- Interaction Quality V4 (inkl. Weak Overlap %)
- Frontend: `Quality Dashboard V4`
- Simulation Intelligence KPIs: `weak_overlap_rate`

## Offline Scale
- Presets 10 / 25 / 50 × ≤4 Runden (Scripted/Manual)
- Szenarien A–E offline; adversarial Grounding
- Live-Ceilings unberührt; Provider-USD $0 offline

## Nicht im Scope
- Kein Merge, kein Live-LLM-Test, keine Limit-Erhöhung, kein Zep.
