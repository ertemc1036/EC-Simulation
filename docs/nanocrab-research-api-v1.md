# NanoCrab Research API v1

Contract for NanoCrab Lab Research MVP 1.0 and for a future NanoCrab AI adapter that speaks the same resources.

This document is an interface. It does not change the simulation engine, API keys, or any other application.

## Boundary

- Base path: `/api/v1/research`
- Schema: `nanocrab.research.v1`
- Export schema: `nanocrab.research.export.v1`
- Authentication: none
- SSO: not part of this contract
- This API does not write into or modify other applications
- Offline Demo sessions do not call a provider and do not start the OASIS simulation engine
- AI Research calls an OpenAI-compatible chat endpoint only after `llm_run_approved=true` on that same request
- Research approval is not read from Simulation Lab
- Page loads and GET routes do not call a provider
- `paid_llm_unlocked` stays `false`
- Live ceilings stay 5 agents / 2 rounds / 10 calls / 20_000 tokens / 500 output tokens / $0.20
- A paid simulation start without `llm_run_approved=true` still returns HTTP 409 from `/api/simulation/start`
- A research session with `mode=llm` returns HTTP 409 whether or not approval is set, and makes no provider call

Simulation Lab remains `POST /api/simulation/start` on the existing engine.

## Provenance

Every study distinguishes:

| Kind | Meaning |
| --- | --- |
| `user_input` | Question, context, or audience text a person typed |
| `source` | A document the user supplied |
| `synthetic_statement` | Fixture speech from a synthetic persona |
| `interpretation` | A reading of those statements |
| `human_validation_need` | What a person still has to check |

Findings cite `source_ids`. Export includes `source_index` so a client can see the kind of each id. Roster length is a count of personas, not a sample size. Exports claim no percentages, no significance tests, and no market coverage.

`user_input_contains_percent_pattern` flags a percent-like token in the brief. That token stays user input. The MVP does not turn it into a finding.

## Resources

### `GET /api/v1/research/meta`

Product, schema, ceilings, `sso: false`, `modifies_other_apps: false`.

### `GET /api/v1/research/dashboard`

Counts, the three example studies, workflow order, disclaimer, ceilings.

### `GET /api/v1/research/projects`

### `POST /api/v1/research/projects`

```json
{ "name": "Field kit", "organization": "Optional" }
```

### `GET /api/v1/research/projects/{project_id}`

### `GET /api/v1/research/studies`

### `POST /api/v1/research/studies`

```json
{
  "project_id": "prj_…",
  "title": "Shift handover notes",
  "research_question": "How do leads describe a handover tool change?",
  "context": "User-supplied note: one team shares a paper log.",
  "audience_name": "Shift leads",
  "audience_description": "Synthetic roles for this study.",
  "participant_count": 4,
  "sources": [{ "title": "Paper log note", "text": "The log lives at the desk." }]
}
```

`participant_count` must be 3, 4, or 5. `sources` may be empty. A missing source is not invented.

### `GET /api/v1/research/studies/{study_id}`

### `PATCH /api/v1/research/studies/{study_id}`

Updates the brief and clears participants, sessions, and findings. Example fixtures are read-only and return 400.

### `POST /api/v1/research/studies/{study_id}/participants`

```json
{ "count": 4 }
```

Custom studies receive a deterministic offline roster. Example studies keep their authored roster. Count outside 3–5 returns 400.

### `POST /api/v1/research/studies/{study_id}/sessions`

```json
{ "mode": "offline_fixture" }
```

Response session fields:

- `execution`: `offline_deterministic_fixture`
- `genuine_ai`: false
- `provider`: null
- `provider_calls`: 0
- `cost_usd`: 0

`mode=offline_fixture` (the default) writes an offline demo transcript. It makes no provider call, including when `llm_run_approved` is also set. Demo transcripts stay labeled `offline_deterministic_fixture`.

`mode=llm` without `llm_run_approved` returns **409** with `status: blocked` and an approval (`Freigabe`) message. No provider object is used.

`mode=llm` with `llm_run_approved: true` on an offline demo example returns **400** (`status: offline_demo`) and makes no provider call. Demo transcripts are not relabeled as model output.

`mode=llm` with approval on a custom study requires model participants (`POST .../participants` with `mode=ai` and the same approval flag), a configured key or an injected provider, and room under the ceilings. The short protocol is 3 participants and 2 turns. Every provider call counts toward 10 calls / 20_000 tokens / 500 output tokens / $0.20. `paid_llm_unlocked` stays false. If the budget cannot cover the next call, the response is a partial result and the missing turns are not invented.

Other modes return 400. This resource does not start the OASIS engine.

### `GET /api/v1/research/studies/{study_id}/cost-estimate`

Local arithmetic for the short protocol. `provider_call_made` is false.

### `POST /api/v1/research/studies/{study_id}/findings`

Builds interpretations that cite statement ids and, when present, a user-supplied source id and the question id. Each finding includes `human_validation`. Requires a session. Returns 400 otherwise.

### `GET /api/v1/research/studies/{study_id}/export`

`nanocrab.research.export.v1` body:

- `disclaimer`
- `project` (`kind: user_input`)
- `brief.research_question`, `brief.context` (context may include `sources`)
- `study.audience`, `study.participants`, `study.sessions`, `study.findings`
- `source_index`
- `integrity.ok`, `provenance_kinds_present`, `unresolved_source_ids`, `banned_claim_patterns`
- `integrity.synthesis_method`
- `integrity.genuine_ai_research` is false for offline demos and for partial model runs
- `integrity.openai_calls`, `integrity.prompt_tokens`, `integrity.completion_tokens`, `integrity.cost_usd` come from stored provider usage (0 for a demo that never called)
- `integrity.paid_llm_unlocked: false`
- `integrity.not_research_validated: true`

### `GET /api/v1/research/audiences`

### `GET /api/v1/research/reports`

Reports are studies that already have findings. `export_path` points at the export route.

### `POST /api/v1/research/examples/{slug}/load`

Idempotent. Slugs:

- `saas-pricing`
- `banking-onboarding`
- `restaurant-software`

Unknown slug: 404.

## AI Research pathway

`POST .../participants` with `{ "mode": "ai", "count": 3, "llm_run_approved": true }` asks the provider for 3 personas. The offline roster is a different code path.

`POST .../sessions` with `{ "mode": "llm", "llm_run_approved": true }` runs `short_interview_v1` (opening and one follow-up per persona).

`POST .../findings` with `{ "llm_run_approved": true }` on a model session synthesizes interpretations that cite statement ids. `synthesis_method` is `model_structured_json` when that parse succeeds, and `templated_offline_fixture` for offline demo findings. Invalid provider JSON is not stored. Percentages, sample sizes, and significance language are rejected.

`genuine_ai_research` is true only for a complete model interview plus model synthesis. Partial runs stay false. Offline exports stay false. `not_research_validated` stays true.

A missing key returns HTTP 400 with `missing_credentials: true` and does not open a socket. Simulation approval is not consulted.

## Licensing

- Installed `camel-oasis` 0.2.5 and `camel-ai` 0.2.78 both report Apache-2.0 in package metadata. This MVP does not change those pins.
- MiroFish source is not included. The existing persona generator is described in code as similar in spirit to MiroFish. That comment is not a copy of MiroFish.
- MiroFish is commonly published under the GNU Affero General Public License v3.0. Serving or combining AGPL-covered code with this application can impose AGPL source-offer duties on the combined work. Do not vendor it without a separate license review.
- This note is not legal advice.

## Examples in this build

| Slug | Roster | Session | Findings cite source ids |
| --- | --- | --- | --- |
| `saas-pricing` | 4 | offline fixture | yes |
| `banking-onboarding` | 5 | offline fixture | yes |
| `restaurant-software` | 3 | offline fixture | yes |
