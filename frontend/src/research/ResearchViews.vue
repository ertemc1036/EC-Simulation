<script setup lang="ts">
import { onMounted, ref } from 'vue'
import type { ProductView } from '../product/types'
import { api } from './api'

defineProps<{ view: ProductView }>()
const emit = defineEmits<{ navigate: [view: ProductView] }>()

interface Source {
  id: string
  kind: string
  title: string
  text: string
}

interface Participant {
  id: string
  name: string
  role: string
  perspective?: string
  synthetic?: boolean
  human_subject?: boolean
  generation?: string
}

interface Statement {
  id: string
  kind: string
  participant_id: string
  text: string
  source_refs?: string[]
  origin?: string
}

interface Usage {
  calls?: number
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  cost_usd?: number
  partial?: boolean
  stop_reason?: string | null
  paid_llm_unlocked?: boolean
}

interface CostEstimate {
  provider_call_made?: boolean
  paid_llm_unlocked?: boolean
  protocol?: string
  totals?: {
    calls: number
    estimated_prompt_tokens: number
    estimated_completion_tokens: number
    estimated_total_tokens: number
    estimated_cost_usd: number
  }
  full_protocol_fits_remaining_budget?: boolean
  budget_note?: string
  disclaimer?: string
  ceilings?: Record<string, number>
  recorded_usage?: Usage
}

interface Finding {
  id: string
  kind: string
  title: string
  text: string
  source_ids: string[]
  human_validation: { id: string; kind: string; text: string }
}

interface Study {
  id: string
  title: string
  status: string
  project_id: string
  project_name?: string
  example_slug?: string | null
  read_only_fixture?: boolean
  research_question?: { id: string; kind: string; text: string }
  context?: { id: string; kind: string; text: string; sources?: Source[] }
  audience?: { id: string; name: string; description?: string; synthetic?: boolean }
  participants?: Participant[]
  sessions?: Array<{
    id: string
    execution: string
    label?: string
    genuine_ai?: boolean
    provider?: string | null
    provider_calls?: number
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
    cost_usd?: number
    partial?: boolean
    stop_reason?: string | null
    protocol?: string
    statements: Statement[]
  }>
  findings?: Finding[]
  disclaimer?: { summary?: string; genuine_ai_research?: boolean }
  user_input_contains_percent_pattern?: boolean
  participant_generation?: string | null
  synthesis_method?: string | null
  ai_usage?: Usage
  ai_cost_estimate?: CostEstimate
}

interface StudySummary {
  id: string
  title: string
  status: string
  project_name?: string
  example_slug?: string | null
  participant_count: number
  finding_count: number
  audience_name?: string
}

interface Dashboard {
  disclaimer?: { summary?: string }
  ceilings?: {
    paid_llm_unlocked?: boolean
    live_test_limits?: Record<string, number>
  }
  recorded_provider_usage?: Usage
  page_load_makes_provider_calls?: boolean
  counts?: Record<string, number>
  examples?: Array<{
    slug: string
    title: string
    study_id: string
    project_name?: string
    participant_count: number
  }>
  workflow?: string[]
}

const dashboard = ref<Dashboard | null>(null)
const projects = ref<Array<{ id: string; name: string; organization?: string }>>([])
const studies = ref<StudySummary[]>([])
const audiences = ref<
  Array<{
    id: string
    name: string
    description?: string
    study_id: string
    study_title?: string
    project_name?: string
    participant_count: number
    synthetic?: boolean
  }>
>([])
const reports = ref<
  Array<{
    study_id: string
    title: string
    project_name?: string
    finding_count: number
    participant_count: number
    genuine_ai_research?: boolean
  }>
>([])
const active = ref<Study | null>(null)
const tab = ref<'overview' | 'participants' | 'sessions' | 'findings' | 'report'>('overview')
const exportPayload = ref<unknown>(null)
const error = ref<string | null>(null)
const busy = ref(false)
const researchMode = ref<'offline_demo' | 'ai_research'>('offline_demo')
const aiApproved = ref(false)

const projectForm = ref({ name: '', organization: '' })
const studyForm = ref({
  project_id: '',
  title: '',
  research_question: '',
  context: '',
  audience_name: '',
  audience_description: '',
  source_title: '',
  source_text: '',
  participant_count: 4,
})

const workflowLabels: Record<string, string> = {
  research_project: 'Research project',
  research_question: 'Research question',
  context: 'Context',
  audience: 'Audience',
  synthetic_participants: 'Synthetic participants (3–5)',
  research_session: 'Offline research session',
  findings: 'Findings with source ids',
  export_json: 'Export JSON',
}

function personName(study: Study, id: string) {
  return study.participants?.find((p) => p.id === id)?.name || id
}

function money(value: number | undefined) {
  return (value ?? 0).toFixed(6)
}

function isModelSession(session: { execution?: string }) {
  return session.execution === 'model_interview'
}

async function refresh() {
  error.value = null
  try {
    const [dash, proj, stud, aud, rep] = await Promise.all([
      api<Dashboard>('/api/v1/research/dashboard'),
      api<{ projects: Array<{ id: string; name: string; organization?: string }> }>(
        '/api/v1/research/projects',
      ),
      api<{ studies: StudySummary[] }>('/api/v1/research/studies'),
      api<{ audiences: typeof audiences.value }>('/api/v1/research/audiences'),
      api<{ reports: typeof reports.value }>('/api/v1/research/reports'),
    ])
    dashboard.value = dash
    projects.value = proj.projects
    studies.value = stud.studies
    audiences.value = aud.audiences
    reports.value = rep.reports
    if (!studyForm.value.project_id && proj.projects[0]) {
      studyForm.value.project_id = proj.projects[0].id
    }
  } catch (err) {
    error.value =
      err instanceof Error
        ? err.message
        : 'Research API is not reachable. Start the Flask app on port 5001.'
  }
}

async function openStudy(id: string) {
  busy.value = true
  error.value = null
  exportPayload.value = null
  tab.value = 'overview'
  emit('navigate', 'studies')
  try {
    const data = await api<{ study: Study }>(`/api/v1/research/studies/${id}`)
    active.value = data.study
    aiApproved.value = false
    if (data.study.read_only_fixture) researchMode.value = 'offline_demo'
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Could not open the study.'
  } finally {
    busy.value = false
  }
}

async function createProject() {
  busy.value = true
  error.value = null
  try {
    await api('/api/v1/research/projects', {
      method: 'POST',
      body: JSON.stringify(projectForm.value),
    })
    projectForm.value = { name: '', organization: '' }
    await refresh()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Could not create the project.'
  } finally {
    busy.value = false
  }
}

async function createStudy() {
  busy.value = true
  error.value = null
  try {
    const sources =
      studyForm.value.source_title.trim() && studyForm.value.source_text.trim()
        ? [
            {
              title: studyForm.value.source_title.trim(),
              text: studyForm.value.source_text.trim(),
            },
          ]
        : []
    const created = await api<{ study: Study }>('/api/v1/research/studies', {
      method: 'POST',
      body: JSON.stringify({
        project_id: studyForm.value.project_id,
        title: studyForm.value.title,
        research_question: studyForm.value.research_question,
        context: studyForm.value.context,
        audience_name: studyForm.value.audience_name,
        audience_description: studyForm.value.audience_description,
        participant_count: Number(studyForm.value.participant_count),
        sources,
      }),
    })
    studyForm.value.title = ''
    studyForm.value.research_question = ''
    studyForm.value.context = ''
    studyForm.value.audience_name = ''
    studyForm.value.audience_description = ''
    studyForm.value.source_title = ''
    studyForm.value.source_text = ''
    await refresh()
    await openStudy(created.study.id)
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Could not create the study.'
  } finally {
    busy.value = false
  }
}

async function postStep(path: string, body: Record<string, unknown> = {}) {
  if (!active.value) return
  busy.value = true
  error.value = null
  try {
    await api(path, { method: 'POST', body: JSON.stringify(body) })
    const data = await api<{ study: Study }>(`/api/v1/research/studies/${active.value.id}`)
    active.value = data.study
    exportPayload.value = null
    await refresh()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'The step failed.'
  } finally {
    busy.value = false
  }
}

async function runApproved(path: string, extra: Record<string, unknown>) {
  if (!aiApproved.value) {
    error.value = 'Approval required (Freigabe) for this research run. No request was sent.'
    return
  }
  const approved = aiApproved.value
  aiApproved.value = false
  await postStep(path, { ...extra, llm_run_approved: approved })
}

async function loadExport(download: boolean) {
  if (!active.value) return
  busy.value = true
  error.value = null
  try {
    const data = await api<Record<string, unknown>>(
      `/api/v1/research/studies/${active.value.id}/export`,
    )
    exportPayload.value = data
    if (download) {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${active.value.id}-export.json`
      link.click()
      URL.revokeObjectURL(url)
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Export failed.'
  } finally {
    busy.value = false
  }
}

onMounted(refresh)
</script>

<template>
  <div class="nc-research">
    <p v-if="error" class="nc-error" role="alert">{{ error }}</p>

    <section v-if="view === 'dashboard'">
      <p class="nc-kicker">Dashboard</p>
      <h1>Offline Demo and AI Research</h1>
      <p class="nc-lede">
        Offline Demo uses labeled fixtures and makes no provider call. AI Research is a separate
        short interview of 3 synthetic personas. It shows a cost estimate first and calls a model
        only after you approve that run.
      </p>
      <div class="nc-banner">
        <strong>Not human research.</strong>
        {{ dashboard?.disclaimer?.summary || 'Synthetic statements are not interviews, and this build does not call a model.' }}
      </div>
      <div class="nc-chips" aria-label="Live ceilings">
        <span class="nc-chip">Agents {{ dashboard?.ceilings?.live_test_limits?.max_agents ?? 5 }}</span>
        <span class="nc-chip">Rounds {{ dashboard?.ceilings?.live_test_limits?.max_rounds ?? 2 }}</span>
        <span class="nc-chip">Calls {{ dashboard?.ceilings?.live_test_limits?.max_model_calls ?? 10 }}</span>
        <span class="nc-chip">Tokens {{ dashboard?.ceilings?.live_test_limits?.max_tokens_per_run ?? 20000 }}</span>
        <span class="nc-chip">Out {{ dashboard?.ceilings?.live_test_limits?.max_tokens_per_call ?? 500 }}</span>
        <span class="nc-chip">$ {{ dashboard?.ceilings?.live_test_limits?.max_cost_usd_per_run ?? 0.2 }}</span>
        <span class="nc-chip">paid_llm_unlocked=false</span>
        <span class="nc-chip">Page load calls 0</span>
        <span class="nc-chip">
          Recorded calls {{ dashboard?.recorded_provider_usage?.calls ?? 0 }}
        </span>
      </div>
      <div class="nc-grid">
        <article class="nc-card">
          <h3>Workflow</h3>
          <ol class="nc-steps">
            <li v-for="step in dashboard?.workflow || []" :key="step">
              {{ workflowLabels[step] || step }}
            </li>
          </ol>
        </article>
        <article class="nc-card">
          <h3>In this workspace</h3>
          <p>Projects {{ dashboard?.counts?.projects ?? 0 }}</p>
          <p>Studies {{ dashboard?.counts?.studies ?? 0 }}</p>
          <p>Audiences {{ dashboard?.counts?.audiences ?? 0 }}</p>
          <p>Reports {{ dashboard?.counts?.reports ?? 0 }}</p>
        </article>
      </div>
      <h2>B2B offline demo examples</h2>
      <div class="nc-grid">
        <article v-for="example in dashboard?.examples || []" :key="example.slug" class="nc-card">
          <p class="nc-kicker">{{ example.slug }}</p>
          <h3>{{ example.title }}</h3>
          <p class="nc-muted">{{ example.project_name }}</p>
          <p>Roster {{ example.participant_count }} synthetic personas</p>
          <button type="button" class="nc-btn" @click="openStudy(example.study_id)">Open study</button>
        </article>
      </div>
    </section>

    <section v-else-if="view === 'projects'">
      <p class="nc-kicker">Projects</p>
      <h1>Research projects</h1>
      <form class="nc-panel nc-form" @submit.prevent="createProject">
        <div class="nc-split">
          <label>
            Name
            <input v-model="projectForm.name" required minlength="2" maxlength="120" />
          </label>
          <label>
            Organization
            <input v-model="projectForm.organization" maxlength="160" placeholder="Optional" />
          </label>
        </div>
        <div>
          <button class="nc-btn" type="submit" :disabled="busy">Create project</button>
        </div>
      </form>
      <div class="nc-list">
        <article v-for="project in projects" :key="project.id" class="nc-card">
          <h3>{{ project.name }}</h3>
          <p class="nc-muted">{{ project.organization || 'No organization' }}</p>
          <p class="nc-muted">{{ project.id }}</p>
        </article>
      </div>
    </section>

    <section v-else-if="view === 'studies' && active">
      <p class="nc-kicker">Research study</p>
      <div class="nc-actions">
        <button type="button" class="nc-btn-ghost" @click="active = null">All studies</button>
        <span v-if="active.example_slug" class="nc-kind kind-source">Offline demo</span>
        <span class="nc-kind kind-synthetic_participant">Synthetic</span>
        <span class="nc-muted">{{ active.status }}</span>
      </div>
      <div class="nc-modes" role="radiogroup" aria-label="Research mode">
        <button
          type="button"
          :class="{ active: researchMode === 'offline_demo' }"
          @click="researchMode = 'offline_demo'"
        >
          Offline Demo
        </button>
        <button
          type="button"
          :class="{ active: researchMode === 'ai_research' }"
          :disabled="active.read_only_fixture"
          @click="researchMode = 'ai_research'"
        >
          AI Research
        </button>
      </div>
      <h1>{{ active.title }}</h1>
      <p class="nc-muted">{{ active.project_name }} · {{ active.id }}</p>
      <div class="nc-banner">
        <strong>Not research-validated.</strong>
        {{ active.disclaimer?.summary }}
      </div>
      <p v-if="active.user_input_contains_percent_pattern" class="nc-error">
        The user-supplied brief contains a percent pattern. It stays labeled as user input and is not a finding.
      </p>
      <div class="nc-tabs" role="tablist">
        <button type="button" :class="{ active: tab === 'overview' }" @click="tab = 'overview'">Overview</button>
        <button type="button" :class="{ active: tab === 'participants' }" @click="tab = 'participants'">Participants</button>
        <button type="button" :class="{ active: tab === 'sessions' }" @click="tab = 'sessions'">Sessions</button>
        <button type="button" :class="{ active: tab === 'findings' }" @click="tab = 'findings'">Findings</button>
        <button type="button" :class="{ active: tab === 'report' }" @click="tab = 'report'">Report</button>
      </div>

      <div v-if="tab === 'overview'" class="nc-grid">
        <article class="nc-panel">
          <span class="nc-kind kind-user_input">User input</span>
          <h3>Research question</h3>
          <p>{{ active.research_question?.text }}</p>
          <p class="nc-muted">{{ active.research_question?.id }}</p>
        </article>
        <article class="nc-panel">
          <span class="nc-kind kind-user_input">User input</span>
          <h3>Context</h3>
          <p>{{ active.context?.text }}</p>
          <p class="nc-muted">{{ active.context?.id }}</p>
        </article>
        <article v-for="source in active.context?.sources || []" :key="source.id" class="nc-panel">
          <span class="nc-kind kind-source">Source</span>
          <h3>{{ source.title }}</h3>
          <p>{{ source.text }}</p>
          <p class="nc-muted">{{ source.id }}</p>
        </article>
        <article class="nc-panel">
          <span class="nc-kind kind-user_input">User input</span>
          <h3>Audience · {{ active.audience?.name }}</h3>
          <p>{{ active.audience?.description }}</p>
          <p class="nc-muted">Synthetic roster. Not a market sample. {{ active.audience?.id }}</p>
        </article>
      </div>

      <div v-else-if="tab === 'participants'">
        <article v-if="researchMode === 'ai_research'" class="nc-panel nc-estimate">
          <h3>Cost estimate before any model call</h3>
          <p>
            Full short protocol:
            {{ active.ai_cost_estimate?.totals?.calls ?? '—' }} calls ·
            {{ active.ai_cost_estimate?.totals?.estimated_prompt_tokens ?? '—' }} prompt tokens ·
            {{ active.ai_cost_estimate?.totals?.estimated_completion_tokens ?? '—' }} output tokens ·
            {{ active.ai_cost_estimate?.totals?.estimated_total_tokens ?? '—' }} total tokens ·
            ${{ money(active.ai_cost_estimate?.totals?.estimated_cost_usd) }}
          </p>
          <p class="nc-muted">{{ active.ai_cost_estimate?.disclaimer }}</p>
          <p class="nc-muted">
            Ceilings stay 5 agents / 2 rounds / 10 calls / 20k tokens / 500 output / $0.20.
            paid_llm_unlocked=false. This estimate did not call a provider.
          </p>
          <p v-if="active.ai_cost_estimate?.full_protocol_fits_remaining_budget === false" class="nc-error">
            {{ active.ai_cost_estimate?.budget_note }}
          </p>
          <label class="nc-check">
            <input v-model="aiApproved" type="checkbox" />
            Approve this research run only. Simulation Lab approval is not used.
          </label>
        </article>
        <div class="nc-actions">
          <button
            v-if="researchMode === 'offline_demo'"
            type="button"
            class="nc-btn"
            :disabled="busy || active.read_only_fixture"
            @click="postStep(`/api/v1/research/studies/${active.id}/participants`, { mode: 'offline_fixture' })"
          >
            Generate offline demo personas
          </button>
          <button
            v-else
            type="button"
            class="nc-btn"
            :disabled="busy || !aiApproved"
            @click="runApproved(`/api/v1/research/studies/${active.id}/participants`, { mode: 'ai', count: 3 })"
          >
            Generate 3 model personas
          </button>
        </div>
        <p v-if="active.read_only_fixture" class="nc-muted">
          This example keeps its authored roster. These personas are an offline demo, not model output.
        </p>
        <p v-else-if="researchMode === 'offline_demo'" class="nc-muted">
          Offline demo roster. This button does not call a provider.
        </p>
        <div class="nc-grid">
          <article v-for="person in active.participants || []" :key="person.id" class="nc-card">
            <span class="nc-kind kind-synthetic_participant">Synthetic participant</span>
            <h3>{{ person.name }}</h3>
            <p>{{ person.role }}</p>
            <p class="nc-muted">{{ person.perspective }}</p>
            <p class="nc-muted">
              {{ person.id }} · human subject {{ person.human_subject ? 'yes' : 'no' }}
              · {{ person.generation === 'model' ? 'model persona' : 'offline demo persona' }}
            </p>
          </article>
        </div>
        <p v-if="!(active.participants || []).length" class="nc-muted">No personas yet.</p>
      </div>

      <div v-else-if="tab === 'sessions'">
        <article v-if="researchMode === 'ai_research'" class="nc-panel nc-estimate">
          <h3>Cost estimate before the interview</h3>
          <p>
            {{ active.ai_cost_estimate?.totals?.calls ?? '—' }} calls ·
            {{ active.ai_cost_estimate?.totals?.estimated_total_tokens ?? '—' }} tokens ·
            ${{ money(active.ai_cost_estimate?.totals?.estimated_cost_usd) }}
          </p>
          <p class="nc-muted">{{ active.ai_cost_estimate?.disclaimer }}</p>
          <p class="nc-muted">{{ active.ai_cost_estimate?.budget_note }}</p>
          <label class="nc-check">
            <input v-model="aiApproved" type="checkbox" />
            Approve this research run only. Simulation Lab approval is not used.
          </label>
        </article>
        <div class="nc-actions">
          <button
            v-if="researchMode === 'offline_demo'"
            type="button"
            class="nc-btn"
            :disabled="busy || !(active.participants || []).length"
            @click="postStep(`/api/v1/research/studies/${active.id}/sessions`, { mode: 'offline_fixture' })"
          >
            Run offline demo session
          </button>
          <button
            v-else
            type="button"
            class="nc-btn"
            :disabled="busy || !aiApproved || active.participant_generation !== 'model'"
            @click="runApproved(`/api/v1/research/studies/${active.id}/sessions`, { mode: 'llm' })"
          >
            Run AI interview (3 participants)
          </button>
        </div>
        <p v-if="researchMode === 'offline_demo'" class="nc-muted">
          Offline demo fixture. This button does not call a provider and does not start Simulation Lab.
          Demo transcripts are not live AI.
        </p>
        <p v-else class="nc-muted">
          Short protocol: two turns for each of 3 model personas. Offline demo personas are not sent
          through this path. Without the checkbox the server returns HTTP 409 and makes no call.
        </p>
        <article v-for="session in active.sessions || []" :key="session.id" class="nc-panel">
          <span v-if="isModelSession(session)" class="nc-kind kind-interpretation">Model transcript</span>
          <span v-else class="nc-kind kind-synthetic_statement">Offline demo fixture</span>
          <p>{{ session.label || session.execution }}</p>
          <p v-if="!isModelSession(session)" class="nc-muted">
            This transcript is an offline demo. It is not live AI.
          </p>
          <p class="nc-muted">
            {{ session.id }} · calls {{ session.provider_calls ?? 0 }} ·
            prompt {{ session.prompt_tokens ?? 0 }} · output {{ session.completion_tokens ?? 0 }} ·
            total {{ session.total_tokens ?? 0 }} · ${{ money(session.cost_usd) }}
            <span v-if="session.partial"> · partial {{ session.stop_reason || 'incomplete' }}</span>
          </p>
          <div v-for="statement in session.statements" :key="statement.id" class="nc-card">
            <span v-if="isModelSession(session)" class="nc-kind kind-interpretation">Model statement</span>
            <span v-else class="nc-kind kind-synthetic_statement">Offline demo statement</span>
            <p>{{ statement.text }}</p>
            <p class="nc-muted">
              {{ statement.id }} · {{ personName(active, statement.participant_id) }}
              <span v-if="statement.source_refs?.length"> · refs {{ statement.source_refs.join(', ') }}</span>
            </p>
          </div>
        </article>
        <p v-if="active.ai_usage" class="nc-muted">
          Study provider usage: {{ active.ai_usage.calls ?? 0 }} calls ·
          {{ active.ai_usage.prompt_tokens ?? 0 }} prompt tokens ·
          {{ active.ai_usage.completion_tokens ?? 0 }} output tokens ·
          ${{ money(active.ai_usage.cost_usd) }} · paid_llm_unlocked=false
          <span v-if="active.ai_usage.partial"> · partial result, missing pieces were not invented</span>
        </p>
      </div>

      <div v-else-if="tab === 'findings'">
        <article v-if="researchMode === 'ai_research'" class="nc-panel nc-estimate">
          <h3>Synthesis uses one model call</h3>
          <p class="nc-muted">
            The estimate above is the full protocol, including this call.
            ${{ money(active.ai_cost_estimate?.totals?.estimated_cost_usd) }} estimated ·
            {{ active.ai_cost_estimate?.totals?.calls ?? '—' }} calls.
            No call happens until you approve.
          </p>
          <label class="nc-check">
            <input v-model="aiApproved" type="checkbox" />
            Approve this synthesis only. Simulation Lab approval is not used.
          </label>
        </article>
        <div class="nc-actions">
          <button
            v-if="researchMode === 'offline_demo'"
            type="button"
            class="nc-btn"
            :disabled="busy || !(active.sessions || []).length"
            @click="postStep(`/api/v1/research/studies/${active.id}/findings`)"
          >
            Derive offline demo findings
          </button>
          <button
            v-else
            type="button"
            class="nc-btn"
            :disabled="busy || !aiApproved"
            @click="runApproved(`/api/v1/research/studies/${active.id}/findings`, {})"
          >
            Synthesize findings
          </button>
        </div>
        <p v-if="active.synthesis_method" class="nc-muted">Synthesis method: {{ active.synthesis_method }}</p>
        <article v-for="finding in active.findings || []" :key="finding.id" class="nc-panel">
          <span class="nc-kind kind-interpretation">Interpretation</span>
          <h3>{{ finding.title }}</h3>
          <p>{{ finding.text }}</p>
          <p class="nc-muted">Source ids: {{ finding.source_ids.join(', ') }}</p>
          <div class="nc-validate">
            <span class="nc-kind kind-human_validation_need">Human validation</span>
            <p>{{ finding.human_validation.text }}</p>
            <p class="nc-muted">{{ finding.human_validation.id }}</p>
          </div>
        </article>
        <p v-if="!(active.findings || []).length" class="nc-muted">No findings yet.</p>
      </div>

      <div v-else class="nc-panel">
        <h3>Export JSON</h3>
        <p class="nc-muted">
          Schema nanocrab.research.export.v1. Roster counts are not sample sizes. Nothing here is research-validated.
          Synthesis method: {{ active.synthesis_method || 'not synthesized yet' }}.
        </p>
        <p v-if="active.ai_usage" class="nc-muted">
          Recorded after runs: {{ active.ai_usage.calls ?? 0 }} calls ·
          {{ active.ai_usage.total_tokens ?? 0 }} tokens ·
          ${{ money(active.ai_usage.cost_usd) }} · paid_llm_unlocked=false
        </p>
        <div class="nc-actions">
          <button type="button" class="nc-btn" :disabled="busy" @click="loadExport(false)">Preview</button>
          <button type="button" class="nc-btn-ghost" :disabled="busy" @click="loadExport(true)">Download JSON</button>
        </div>
        <pre v-if="exportPayload" class="nc-pre">{{ JSON.stringify(exportPayload, null, 2) }}</pre>
      </div>
    </section>

    <section v-else-if="view === 'studies'">
      <p class="nc-kicker">Research studies</p>
      <h1>Question, context, audience</h1>
      <form class="nc-panel nc-form" @submit.prevent="createStudy">
        <label>
          Project
          <select v-model="studyForm.project_id" required>
            <option v-for="project in projects" :key="project.id" :value="project.id">
              {{ project.name }}
            </option>
          </select>
        </label>
        <label>
          Title
          <input v-model="studyForm.title" required minlength="2" />
        </label>
        <label>
          Research question
          <textarea v-model="studyForm.research_question" required minlength="8" rows="3" />
        </label>
        <label>
          Context
          <textarea v-model="studyForm.context" required minlength="8" rows="3" />
        </label>
        <div class="nc-split">
          <label>
            Audience name
            <input v-model="studyForm.audience_name" required minlength="2" />
          </label>
          <label>
            Synthetic personas
            <select v-model.number="studyForm.participant_count">
              <option :value="3">3</option>
              <option :value="4">4</option>
              <option :value="5">5</option>
            </select>
          </label>
        </div>
        <label>
          Audience description
          <textarea v-model="studyForm.audience_description" rows="2" />
        </label>
        <div class="nc-split">
          <label>
            Optional source title
            <input v-model="studyForm.source_title" />
          </label>
          <label>
            Optional source text
            <textarea v-model="studyForm.source_text" rows="2" />
          </label>
        </div>
        <div>
          <button class="nc-btn" type="submit" :disabled="busy || !projects.length">Create study</button>
        </div>
      </form>
      <div class="nc-list">
        <button
          v-for="study in studies"
          :key="study.id"
          type="button"
          class="nc-link"
          @click="openStudy(study.id)"
        >
          <strong>{{ study.title }}</strong>
          <span class="nc-muted">
            {{ study.project_name }} · {{ study.status }} · roster {{ study.participant_count }} ·
            findings {{ study.finding_count }}
          </span>
        </button>
      </div>
    </section>

    <section v-else-if="view === 'audiences'">
      <p class="nc-kicker">Synthetic audiences</p>
      <h1>Rosters, not samples</h1>
      <div class="nc-banner">
        <strong>No market claim.</strong>
        These audiences are synthetic personas attached to a study. They do not represent a population.
      </div>
      <div class="nc-grid">
        <article v-for="audience in audiences" :key="audience.id" class="nc-card">
          <span class="nc-kind kind-synthetic_participant">Synthetic</span>
          <h3>{{ audience.name }}</h3>
          <p>{{ audience.description }}</p>
          <p class="nc-muted">
            {{ audience.project_name }} · {{ audience.study_title }} · roster {{ audience.participant_count }}
          </p>
          <button type="button" class="nc-btn-ghost" @click="openStudy(audience.study_id)">Open study</button>
        </article>
      </div>
    </section>

    <section v-else-if="view === 'reports'">
      <p class="nc-kicker">Reports</p>
      <h1>Exports with source ids</h1>
      <p class="nc-lede">
        A report is a study that has findings. Each export labels user input, sources, synthetic
        statements, interpretation, and human-validation needs.
      </p>
      <div class="nc-list">
        <article v-for="report in reports" :key="report.study_id" class="nc-card">
          <h3>{{ report.title }}</h3>
          <p class="nc-muted">
            {{ report.project_name }} · findings {{ report.finding_count }} · roster
            {{ report.participant_count }} · genuine model research
            {{ report.genuine_ai_research ? 'yes' : 'no' }}
          </p>
          <button type="button" class="nc-btn" @click="openStudy(report.study_id)">Open report</button>
        </article>
      </div>
    </section>
  </div>
</template>
