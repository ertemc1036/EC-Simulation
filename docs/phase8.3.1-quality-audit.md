# Phase 8.3.1 — Multi-Scenario Quality Audit

Pfad: `/Users/ertemmac/Desktop/EC-Simulation`  
Basis: Phase 8.3 PR [#22](https://github.com/ertemc1036/EC-Simulation/pull/22) @ `ad4fb63` (`cursor/phase8.3-realistic-agents-383f`)  
Branch: `cursor/phase8.3.1-quality-audit-bf02`  
Fixture: `data/results/20261028T145656Z-e251172c.json`  
**Keine großen Features. Keine OpenAI-/Provider-Calls. Keine Live-Simulation. Kein Merge. Live-Ceilings unverändert. `paid_llm_unlocked=false`. $0.**

## PHASE 8.3.1 QUALITY AUDIT COMPLETE

Offline Multi-Scenario-Audit A–E + Stance-/Grounding-Messung gegen Phase-8.3-Code; Befunde und 8.4-Empfehlungen dokumentiert; Regressionstests; 0 Provider-Calls / $0.

---

## Scope

| In | Out |
|----|-----|
| Messen (Stance V3, Grounding V3, IQ, Motivation, Topic-Domain) | Feature-Fixes / Classifier-Tuning |
| 5 Szenarien A–E offline | Live-LLM / OpenAI |
| Fixture `e251172c` Reclassify | Merge, Ceiling-Änderungen |
| Empfehlungen für Phase 8.4 | Zep / OASIS `parent_comment_id` |

---

## Szenarien A–E (Ergebnisse)

| ID | Szenario | Domain | Art | N | IQ | Grounded % | Invented | Unknown | Motivation | Topic-Leaks |
|----|----------|--------|-----|---|-----|------------|----------|---------|------------|-------------|
| **A** | KI-Lernassistenten (Live-Fixture `e251172c`) | education | live JSON replay | 3 | **0.814** | **100%** | **0** | **0** | 1.0 | — |
| **B** | KI-Lernassistenten | education | scripted | 10 | 0.736 | 100% | 0 | 0 | 1.0 | keine |
| **C** | Vier-Tage-Woche | labor | scripted | 10 | 0.729 | 100% | 0 | 0 | 1.0 | keine |
| **D** | Bedingungsloses Grundeinkommen | income | scripted | 10 | 0.734 | 100% | 0 | 0 | 1.0 | keine |
| **E** | Berlin Auto-Verbot (S-Bahn-Ring) | mobility | scripted | 10 | 0.742 | 100% | 0 | 0 | 1.0 | keine |

Scripted-Läufe: `execution_kind=scripted`, `model=scripted-reactive`, `source=scripted_test`, `cost_plan.planned_cost_usd=0.0`, `paid_llm_unlocked=false`. Provider-USD **$0** (keine OpenAI-Calls in dieser Phase).

---

## Fixture A — `e251172c` (Phase 8.3 Live)

### Meta / Engines

| Feld | Wert |
|------|------|
| `quality_fix` / `persona_engine` | 8.3 / v4 |
| Stance / Opinion / Arg-Grounded / IQ | v3 / v3 / v3 / v3 |
| Agents / Comments / Replies | 3 / 6 / 3 (cross-agent, 0 self-replies) |
| Historische Live-Kosten (nur Fixture) | 6 Calls · ~$0.0057 — **nicht** dieser Audit |

### Stance (V3 Reclassify)

| Kommentar | Agent | V3 | Semantik (Audit) | Befund |
|-----------|-------|----|------------------|--------|
| C1 | Fatima (Lehrkraft) | conditional | Potenzial + Evidenz/Konzept | OK |
| C2 | Thomas (Eltern) | **pro** | Potenzial **wenn** richtig + Datenschutz + schrittweise + Elternbeteiligung | **False pro** |
| C3 | Hannah (Forschung) | conditional | Evidenzlücken / Piloten | OK |
| C4 | Fatima | **pro** | Evidenz begrenzt → kein Technikzwang; pädagogische Konzepte | **False pro** (`dafür`-Marker) |
| C5 | Thomas | conditional | gegen unreflektierte Vollausrollung | OK |
| C6 | Hannah | conditional | Chancen, aber fehlende Datenbasis | OK |

Finale Agenten-Positionen (stored = replay): Fatima **pro**, Thomas/Hannah **conditional**, `unknown=0`.  
Gegen `d1fb1b1c` (8.2): Thomas-unknown und invented «digitale Kluft» sind in `e251172c` behoben — aber neue False-**pro**-Fälle.

### Grounding V3

| Reply | Parent | Grounded | Score | Overlap | Invented |
|-------|--------|----------|-------|---------|----------|
| C4 → C3 | Evidenz | true | 0.560 | 0.343 | false |
| C5 → C3 | Evidenz/Vollausrollung | true | 0.538 | 0.321 | false |
| C6 → C2 | Potenzial/Zugang | true | **0.308** | **0.133** | false |

`grounded_reply_rate=1.0`, `invented_attribution_rate=0.0`, `mean_grounding_score≈0.47`. C6 ist lexikalisch schwach geerdet (Threshold knapp).

### Interaction Quality

`score=0.814` · Komponenten: grounded 1.0 · stance_clarity 1.0 · debate 0.724 · motivation 1.0 · invented 0 · opinion_stability 0.858.

---

## Stance-Tests (gezielte Fälle)

| Fall | Erwartung | Ist (V3) | Ursache |
|------|-----------|----------|---------|
| Thomas C2 (`e251172c`) | conditional | pro | Soft-Support «großes Potenzial» ohne Conditional-/Soft-Reject-Treffer («schrittweise», «Beteiligung der Eltern», «wenn sie richtig» fehlen im Lexikon) |
| Fatima C4 | conditional | pro | Bare `_PRO_MARKERS` **`dafür`** trifft «plädiere ich **dafür**, dass … **nicht** … technikgetrieben» |
| Soft-Support + Evidenz/Vollausrollung (C1/C5) | conditional | conditional | 8.3-Pfad OK |
| Hard pro / hard contra | pro / contra | OK | Regression |
| `d1fb1b1c` Thomas | conditional | conditional | 8.3-Fix hält |

---

## Grounding-Tests

| Fall | Erwartung | Ist |
|------|-----------|-----|
| `e251172c` alle 3 Replies | grounded, invented=0 | OK |
| Adversarial «digitale Kluft» vs. Evidenz-Parent (`d1fb1b1c` / synthetisch) | invented, ungrounded | OK (8.3 V3) |
| Scripted B–E replies | grounded_rate=1.0, invented=0 | OK |
| C6 schwacher Overlap | dokumentiert | mean≈0.31 — lexikalisch fragil |

---

## Weitere Messbefunde (B–E)

1. **Domain-Routing OK** — education / labor / income / mobility korrekt; keine BGE-/Mobility-/Edu-Leaks in Fremddomains.
2. **Motivation V4** — Coverage 1.0 auf allen fünf Läufen.
3. **Unknown-Share** — 0 auf A–E (Stance-Klarheit hoch).
4. **Income (D) `shifting`×3** — Scripted-Templates «Ich ändere meine Position … werde überzeugt» → Polarity `shifting`; senkt nicht `unknown`, aber Opinion-/Clarity-Semantik unklar für 8.4.
5. **Debate-Quality Scripted ≪ Live** — B–E ≈0.23–0.26 vs. A 0.72 (erwartete Scripted-Schwäche, kein Live-Regression).
6. **IQ Scripted** stabil ~0.73–0.74 bei N=10; Live-Edu höher durch bessere Debate/Grounding-Scores.

---

## Empfehlungen für Phase 8.4 (kein Fix hier)

1. **Stance V4 / Marker-Hygiene**
   - `_PRO_MARKERS`: bare `dafür` entfernen oder nur in positiver Zustimmungsphrase (nicht «dafür, dass … nicht»).
   - Conditional-Lexikon: `schrittweise`, `wenn sie richtig eingesetzt`, `beteiligung der eltern`, `nicht überhastet`, `klare datenschutzregeln` (education/eltern).
2. **False-pro Guard** — Soft-Support ohne Soft-Reject/Conditional → eher `conditional`/`uncertain` statt hard `pro`, wenn Caveat-Lexik (Datenschutz, schrittweise, Beteiligung) vorliegt.
3. **Grounding-Floor** — Warnung/Score-Druck bei `overlap < ~0.20` trotz `grounded=true` (C6-Klasse); optional thematische Bags für Bildung↔IT-Brücken.
4. **Scripted `shifting`** — Templates ohne generisches «ändere meine Position», oder Stance-Map `shifting`→evidenced change only when prior polarity klar.
5. **Debate Scripted** — Argument-Diversity/Claim-Kinds anreichern, damit Offline-IQ nicht fast nur von Grounding×Motivation getragen wird.
6. **Weiter messen, nicht mergen** — Frischer Live-Lauf erst nach Stance-V4; Ceilings 5/2/10/20k/500/$0.20 unverändert lassen.

---

## Regression / Tests

```text
.venv/bin/pytest -q tests/test_phase831_quality_audit.py
# plus full suite in PR CI / local
```

Neu: `tests/test_phase831_quality_audit.py` — Ceilings, Fixture-Doku False-pro, Stance-Fälle, Grounding `e251172c` + adversarial, Offline-Szenarien B–E (scripted N=10, $0 plan, no leaks).

---

## Sicherheit / Kosten (diese Phase)

| Check | Status |
|-------|--------|
| OpenAI/Provider-Calls | **0** |
| Provider-USD | **$0** |
| Live-Simulation gestartet | **nein** |
| `paid_llm_unlocked` | **false** |
| Live-Ceilings | unverändert 5/2/10/20k/500/$0.20 |
| Merge | **nein** |
| Feature-Code geändert | **nein** (Audit + Tests + Doc) |

---

## Fertig

**PHASE 8.3.1 QUALITY AUDIT COMPLETE** — Multi-Scenario A–E offline; Stance-/Grounding-Befunde (False-pro `dafür` / Thomas C2; Grounding OK außer schwachem C6); 8.4-Empfehlungen; 0 Calls / $0; PR gegen Phase 8.3.
