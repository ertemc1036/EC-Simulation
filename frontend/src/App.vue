<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { io, Socket } from 'socket.io-client'
import type { AgentProfile, HealthResponse, SimulationState } from './types'

const API = ''

const health = ref<HealthResponse | null>(null)
const agents = ref<AgentProfile[]>([])
const sim = ref<SimulationState | null>(null)
const scenario = ref('Sollten Online-Foren KI-gestützte Moderation einsetzen?')
/** Default always ManualAction — never auto-enable LLM. */
const mode = ref<'manual' | 'llm'>('manual')
/** Content language for LLM agents (default DE; overrides OASIS English prompts). */
const language = ref('de')
const maxSteps = ref(2)
/** Live-LLM ceiling 5; Offline/Manual presets 10/25/50. */
const agentCount = ref(5)
/** dynamic = scenario-derived; fixed_test = classic 5-roster (truncated). */
const agentGeneration = ref<'dynamic' | 'fixed_test'>('dynamic')
const busy = ref(false)
const error = ref<string | null>(null)
const showLlmConfirm = ref(false)
let socket: Socket | null = null

const offlinePresets = computed(() => {
  const p = health.value?.config?.offline_scale?.presets
  return Array.isArray(p) && p.length ? p : [10, 25, 50]
})

const agentCountMax = computed(() => {
  if (mode.value === 'llm') {
    return health.value?.config?.live_test_limits?.max_agents ?? 5
  }
  return health.value?.config?.offline_max_agents ?? 50
})

function statusClass(status?: string) {
  if (status === 'limit_reached') return 'status-limit'
  if (status === 'failed' || status === 'error' || status === 'blocked') return 'status-fail'
  if (status === 'completed') return 'status-ok'
  if (status === 'running') return 'status-run'
  return ''
}

const llmBanner = computed(() => {
  const cfg = health.value?.config
  if (!cfg) return null
  if (cfg.llm_mode_available) {
    return {
      tone: 'ok' as const,
      text: `LLM bereit (${cfg.openai_model}). Start nur nach expliziter Freigabe. Limits: ${cfg.max_llm_calls_per_run ?? '—'} Calls, ${cfg.max_tokens_per_run ?? '—'} Tokens/Run, $${cfg.max_cost_usd_per_run ?? '—'} USD, retries=${cfg.llm_max_retries ?? '—'}, Memory-Fenster ${cfg.agent_memory_window ?? '—'}.`,
    }
  }
  return {
    tone: 'warn' as const,
    text:
      cfg.llm_mode_blocked_reason ||
      'Kein API-Key konfiguriert — Simulation nur im ManualAction-Modus (echte OASIS-Umgebung, StubModel, keine API-Calls).',
  }
})

/** Stale Flask on :5001 (pre-Phase-7) misses society intelligence fields. */
const staleBackendBanner = computed(() => {
  const h = health.value
  if (!h) return null
  const phase = String(h.config?.phase ?? h.runtime?.phase ?? '')
  const phaseOk =
    phase === '7' ||
    phase.startsWith('7') ||
    phase === '6' ||
    phase.startsWith('6') ||
    phase.startsWith('5')
  const hasRevision = Boolean(h.runtime?.code_revision)
  const hasGenModes = Array.isArray(h.config?.agent_generation_modes)
  const hasOasisReply = Boolean(h.config?.oasis_reply)
  const hasSocial = Boolean(h.config?.social_dynamics)
  const hasOffline = Boolean(h.config?.offline_scale)
  const hasIntel =
    Boolean(h.config?.society_intelligence) ||
    Boolean(h.config?.social_dynamics?.society_analytics) ||
    phase.startsWith('7')
  if (
    phaseOk &&
    hasRevision &&
    hasGenModes &&
    hasOasisReply &&
    hasSocial &&
    hasOffline &&
    (phase.startsWith('7') ? hasIntel : true)
  ) {
    return null
  }
  return {
    tone: 'warn' as const,
    text:
      'Backend wirkt veraltet (Phase ≠ 7 oder fehlende Society-Intelligence-Felder). ' +
      'Flask auf Port 5001 neu starten, sonst fehlen Analytics-/Memory-Ansichten.',
  }
})

const executionKindLabel = computed(() => {
  const kind = sim.value?.execution_kind
  if (kind === 'scripted') return 'ScriptedModel (Offline)'
  if (kind === 'live_llm') return 'Live-LLM (kostenpflichtig)'
  if (kind === 'manual' || sim.value?.mode === 'manual') return 'ManualAction (Offline)'
  if (sim.value?.mode === 'llm') {
    const src = sim.value?.model_calls?.[0]?.source
    if (src === 'scripted_test') return 'ScriptedModel (Offline)'
    return 'LLMAction'
  }
  return '—'
})

const societyOverview = computed(() => {
  const o = sim.value?.society_overview
  if (o && typeof o.agent_count === 'number') return o
  const pos = sim.value?.agent_positions
  const g = sim.value?.knowledge_graph
  if (!pos && !g) return null
  return {
    agent_count: pos?.agent_count ?? sim.value?.agent_count ?? 0,
    step: sim.value?.step,
    posts: sim.value?.posts?.length ?? 0,
    comments: sim.value?.comments?.length ?? 0,
    reactions: sim.value?.reactions?.length ?? 0,
    polarity_distribution: pos?.polarity_distribution || {},
    change_count: pos?.change_count ?? 0,
    social_relations: g?.social_relations || {},
    network: g?.network || {},
    execution_kind: sim.value?.execution_kind,
    live_llm_max_agents: 5,
    offline_max_agents: 50,
    paid_llm_unlocked_for_scale: false,
    scheduler_plans: sim.value?.scheduler?.plans ?? 0,
    self_as_partner_count: sim.value?.scheduler?.self_as_partner_count ?? 0,
    long_term_memory_agents: sim.value?.agent_memory?.agent_count_with_long_term ?? 0,
    relationship_bonds: sim.value?.relationships?.bond_count ?? 0,
    opinion_rounds_tracked: sim.value?.opinion_dynamics?.rounds_tracked ?? 0,
    analytics_kpis: sim.value?.society_analytics?.kpis || {},
  }
})

const polarityBars = computed(() => {
  const dist = societyOverview.value?.polarity_distribution || {}
  const entries = Object.entries(dist)
  const total = entries.reduce((s, [, n]) => s + n, 0) || 1
  return entries.map(([label, n]) => ({
    label,
    count: n,
    pct: Math.round((n / total) * 100),
  }))
})

const networkActors = computed(() => {
  const nodes = sim.value?.knowledge_graph?.nodes || []
  return nodes.filter((n) => n.kind === 'actor').slice(0, 40)
})

const networkEdges = computed(() => {
  const edges = sim.value?.knowledge_graph?.edges || []
  return edges
    .filter((e) =>
      ['agrees_with', 'disagrees_with', 'responds_to', 'influenced_by', 'position_changed'].includes(
        e.relation,
      ),
    )
    .slice(0, 48)
})

const progressPct = computed(() => {
  const s = sim.value
  if (!s?.max_steps) return 0
  return Math.min(100, Math.round((100 * (s.step || 0)) / s.max_steps))
})

const societyAnalytics = computed(() => sim.value?.society_analytics || null)

const analyticsKpis = computed(() => {
  const k = societyAnalytics.value?.kpis || societyOverview.value?.analytics_kpis
  if (!k || typeof k !== 'object') return null
  return k as Record<string, number>
})

const opinionChronology = computed(() => {
  const frames = sim.value?.opinion_dynamics?.chronology
  return Array.isArray(frames) ? frames.slice(-8) : []
})

const relationshipBonds = computed(() => {
  const bonds = sim.value?.relationships?.bonds
  return Array.isArray(bonds) ? bonds.slice(0, 24) : []
})

const longTermMemoryAgents = computed(() => {
  const lt = sim.value?.agent_memory?.long_term
  if (!lt || typeof lt !== 'object') return []
  return Object.entries(lt)
    .map(([id, entries]) => ({
      id,
      entries: Array.isArray(entries) ? entries.slice(-4) : [],
    }))
    .filter((m) => m.entries.length)
    .slice(0, 12)
})

/** Explicit stop reason for limit_reached / failed — always visible when set. */
const stopReasonDisplay = computed(() => {
  const s = sim.value
  if (!s?.stop_reason) return null
  const isLimit = s.status === 'limit_reached' || Boolean(s.cost_summary?.limit_hit)
  return {
    reason: s.stop_reason,
    limitHit: s.cost_summary?.limit_hit || null,
    isLimit,
  }
})

const interactionSummary = computed(() => {
  const s = sim.value?.interaction_summary
  if (!s) return null
  return s
})

const oasisReplyNote = computed(() => {
  const r = sim.value?.oasis_reply || health.value?.config?.oasis_reply
  if (!r) return null
  return {
    parent: r.parent_comment_id_supported === false ? 'nein' : '?',
    via: r.reply_via || 'create_comment(post_id, content)',
    alt: r.documented_alternative || '',
  }
})

function applyPreset(n: number) {
  const max = agentCountMax.value
  agentCount.value = Math.max(1, Math.min(n, max))
  if (mode.value === 'llm' && agentCount.value > 5) {
    agentCount.value = 5
  }
  if (n > 5) {
    agentGeneration.value = 'dynamic'
  }
  void refreshAgents()
}

function onModeChange() {
  if (mode.value === 'llm' && agentCount.value > 5) {
    agentCount.value = 5
    void refreshAgents()
  }
}

const llmConfirmSummary = computed(() => {
  const cfg = health.value?.config
  const nAgents = Math.min(
    sim.value?.agent_count ||
      agentCount.value ||
      agents.value.length ||
      cfg?.max_agents ||
      5,
    cfg?.live_test_limits?.max_agents ?? 5,
  )
  const rounds = Math.min(maxSteps.value || 2, cfg?.max_simulation_steps ?? 2)
  const callLimit = cfg?.max_llm_calls_per_run ?? 10
  const model = cfg?.openai_model ?? '—'
  const est =
    cfg?.estimated_max_cost_usd ??
    cfg?.max_cost_usd_per_run ??
    0.2
  const disclaimer =
    cfg?.cost_estimate_disclaimer ||
    'Kostenschätzung auf Basis der konfigurierten Preise; die Provider-Abrechnung kann abweichen.'
  return {
    model,
    agentCount: nAgents,
    rounds,
    callLimit,
    estimatedMaxCost: Number(est),
    hardCostCap: cfg?.max_cost_usd_per_run ?? 0.2,
    disclaimer,
    generation: agentGeneration.value,
  }
})

const displayAgents = computed(() => {
  if (sim.value?.agents?.length) return sim.value.agents
  return agents.value
})

const memoryAgents = computed(() => {
  const snap = sim.value?.agent_memory?.agents
  if (!snap) return [] as Array<{ id: string; entries: Array<{ kind?: string; text?: string; step?: number | null }> }>
  return Object.entries(snap).map(([id, entries]) => ({ id, entries: entries || [] }))
})

const graphSummary = computed(() => {
  const g = sim.value?.knowledge_graph
  if (!g) return null
  const social = g.social_relations || {}
  const socialEdges = (g.edges || []).filter((e) =>
    ['agrees_with', 'disagrees_with', 'responds_to', 'influenced_by', 'position_changed'].includes(
      e.relation,
    ),
  )
  return {
    nodes: g.node_count ?? 0,
    edges: g.edge_count ?? 0,
    kinds: g.counts_by_kind || {},
    relations: g.counts_by_relation || {},
    social,
    sampleEdges: socialEdges.slice(0, 16),
    note: g.note,
  }
})

const positionSummary = computed(() => {
  const snap = sim.value?.agent_positions
  if (!snap?.positions?.length) return null
  return {
    count: snap.agent_count ?? snap.positions.length,
    changes: snap.change_count ?? 0,
    positions: snap.positions,
    note: snap.note,
  }
})

function sourceClass(source?: string) {
  if (source === 'LLMAction') return 'src-llm'
  if (source === 'ManualAction') return 'src-manual'
  return 'src-other'
}

async function refreshHealth() {
  const res = await fetch(`${API}/api/health`)
  health.value = await res.json()
}

async function refreshAgents() {
  const params = new URLSearchParams({
    count: String(agentCount.value),
    generation: agentGeneration.value,
  })
  if (scenario.value.trim()) params.set('scenario', scenario.value.trim())
  const res = await fetch(`${API}/api/agents?${params}`)
  const data = await res.json()
  agents.value = data.agents || []
}

async function refreshSim() {
  const res = await fetch(`${API}/api/simulation`)
  sim.value = await res.json()
}

function requestStart() {
  error.value = null
  if (mode.value === 'llm') {
    showLlmConfirm.value = true
    return
  }
  void startSimulation(false)
}

function cancelLlmConfirm() {
  showLlmConfirm.value = false
}

function confirmLlmStart() {
  showLlmConfirm.value = false
  void startSimulation(true)
}

async function startSimulation(llmRunApproved: boolean) {
  error.value = null
  busy.value = true
  try {
    const body: Record<string, unknown> = {
      scenario: scenario.value,
      mode: mode.value,
      max_steps: maxSteps.value,
      language: language.value,
      agent_count: agentCount.value,
      agent_generation: agentGeneration.value,
    }
    if (mode.value === 'llm') {
      body.llm_run_approved = llmRunApproved === true
    }
    const res = await fetch(`${API}/api/simulation/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    if (!res.ok || data.ok === false) {
      error.value = data.error || 'Start fehlgeschlagen'
    }
    if (data.state) sim.value = data.state
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

onMounted(async () => {
  await Promise.all([refreshHealth(), refreshAgents(), refreshSim()])
  socket = io(window.location.origin, { transports: ['websocket', 'polling'] })
  socket.on('simulation_update', (payload: { state: SimulationState }) => {
    if (payload?.state) sim.value = payload.state
  })
})

onUnmounted(() => {
  socket?.disconnect()
})
</script>

<template>
  <div class="shell">
    <header class="hero">
      <p class="brand">EC-Simulation</p>
      <h1>OASIS Multi-Agent Forum</h1>
      <p class="lede">
        Phase-7 Society Intelligence — lokales Langzeitgedächtnis, dynamische
        Beziehungen, Meinungs-Chronologie und Analytics (Offline 10/25/50 × ≤4).
        Live-LLM bleibt ≤5.
      </p>
    </header>

    <section v-if="llmBanner" class="banner" :class="llmBanner.tone" aria-live="polite">
      <strong>{{ llmBanner.tone === 'ok' ? 'LLM' : 'Kein API-Key' }}</strong>
      <span>{{ llmBanner.text }}</span>
    </section>

    <section
      v-if="staleBackendBanner"
      class="banner"
      :class="staleBackendBanner.tone"
      aria-live="assertive"
    >
      <strong>Stale Backend</strong>
      <span>{{ staleBackendBanner.text }}</span>
    </section>

    <section class="controls">
      <label>
        Szenario
        <textarea
          v-model="scenario"
          rows="3"
          maxlength="800"
          @change="refreshAgents"
        />
      </label>
      <div class="row">
        <label>
          Modus
          <select v-model="mode" @change="onModeChange">
            <option value="manual">manual (StubModel, Offline-Scale) — Standard</option>
            <option value="llm">llm (Freigabe + API-Key; Ceiling 5 Agenten)</option>
          </select>
        </label>
        <label>
          Sprache
          <select v-model="language">
            <option value="de">Deutsch (Standard)</option>
            <option value="en">English</option>
            <option value="fr">Français</option>
            <option value="es">Español</option>
            <option value="it">Italiano</option>
          </select>
        </label>
        <label>
          Agenten
          <input
            v-model.number="agentCount"
            type="number"
            min="1"
            :max="agentCountMax"
            @change="refreshAgents"
          />
        </label>
        <label>
          Generierung
          <select v-model="agentGeneration" @change="refreshAgents">
            <option value="dynamic">dynamic (aus Szenario, keine Klone)</option>
            <option value="fixed_test">fixed_test (feste 5 / Testmodus)</option>
          </select>
        </label>
        <label>
          Schritte
          <input
            v-model.number="maxSteps"
            type="number"
            min="1"
            :max="mode === 'llm' ? 2 : 4"
          />
        </label>
        <button class="cta" :disabled="busy || sim?.status === 'running'" @click="requestStart">
          {{ sim?.status === 'running' ? 'Läuft…' : 'Simulation starten' }}
        </button>
      </div>
      <div class="preset-row" v-if="mode === 'manual'">
        <span class="k">Offline-Presets</span>
        <button
          v-for="n in offlinePresets"
          :key="n"
          type="button"
          class="preset"
          :class="{ active: agentCount === n }"
          @click="applyPreset(n)"
        >
          {{ n }}
        </button>
        <span class="muted lede-sm">nur Manual/Scripted — nicht für Live-LLM</span>
      </div>
      <p v-if="mode === 'llm'" class="lede-sm muted">
        Live-LLM-Ceilings unverändert: 5 Agenten / 2 Runden / 10 Calls / 20k Tokens / $0.20.
      </p>
      <p v-if="error" class="err">{{ error }}</p>
    </section>

    <div
      v-if="showLlmConfirm"
      class="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="llm-confirm-title"
      @click.self="cancelLlmConfirm"
    >
      <div class="modal">
        <h2 id="llm-confirm-title">LLM-Lauf freigeben?</h2>
        <p class="lede-sm">
          Es werden echte Provider-Calls ausgelöst. Ohne diese Freigabe blockiert das Backend den Start.
        </p>
        <dl class="confirm-grid">
          <div>
            <dt>Modell</dt>
            <dd class="mono">{{ llmConfirmSummary.model }}</dd>
          </div>
          <div>
            <dt>Agenten</dt>
            <dd>{{ llmConfirmSummary.agentCount }} ({{ llmConfirmSummary.generation }})</dd>
          </div>
          <div>
            <dt>Runden</dt>
            <dd>{{ llmConfirmSummary.rounds }}</dd>
          </div>
          <div>
            <dt>Call-Limit</dt>
            <dd>{{ llmConfirmSummary.callLimit }}</dd>
          </div>
          <div>
            <dt>Geschätzte max. Kosten</dt>
            <dd class="mono">
              ~${{ llmConfirmSummary.estimatedMaxCost.toFixed(4) }}
              <span class="muted"> (Hard-Cap ${{ llmConfirmSummary.hardCostCap }})</span>
            </dd>
          </div>
        </dl>
        <p class="disclaimer">{{ llmConfirmSummary.disclaimer }}</p>
        <div class="modal-actions">
          <button type="button" class="btn-secondary" @click="cancelLlmConfirm">Abbrechen</button>
          <button type="button" class="cta" :disabled="busy" @click="confirmLlmStart">
            Freigeben &amp; starten
          </button>
        </div>
      </div>
    </div>

    <section class="status-strip" v-if="sim">
      <div>
        <span class="k">Status</span>
        <span class="v" :class="statusClass(sim.status)">{{ sim.status }}</span>
      </div>
      <div>
        <span class="k">Modus</span>
        <span
          class="v mode-pill"
          :class="
            sim.execution_kind === 'scripted'
              ? 'mode-scripted'
              : sim.mode === 'llm'
                ? 'mode-llm'
                : 'mode-manual'
          "
        >
          {{ executionKindLabel }}
        </span>
      </div>
      <div>
        <span class="k">Progress</span>
        <span class="v">{{ progressPct }}%</span>
      </div>
      <div>
        <span class="k">Sprache</span>
        <span class="v">{{ sim.language || language }}</span>
      </div>
      <div>
        <span class="k">Schritt</span>
        <span class="v">{{ sim.step }} / {{ sim.max_steps }}</span>
      </div>
      <div>
        <span class="k">Agenten</span>
        <span class="v">{{ sim.agent_count ?? displayAgents.length }} · {{ sim.agent_generation || agentGeneration }}</span>
      </div>
      <div>
        <span class="k">Run</span>
        <span class="v mono">{{ sim.run_id }}</span>
      </div>
      <p class="msg">{{ sim.message }}</p>
      <div
        v-if="stopReasonDisplay"
        class="stop-reason"
        :class="stopReasonDisplay.isLimit ? 'stop-limit' : 'stop-other'"
      >
        <span class="k">Stop-Grund</span>
        <span class="v">{{ stopReasonDisplay.reason }}</span>
        <p
          v-if="stopReasonDisplay.limitHit && stopReasonDisplay.limitHit !== stopReasonDisplay.reason"
          class="msg muted"
        >
          Limit-Detail: {{ stopReasonDisplay.limitHit }}
        </p>
      </div>
      <p v-if="sim.error" class="err">{{ sim.error }}</p>
    </section>

    <section class="cost-strip" v-if="sim && (sim.mode === 'llm' || sim.model_calls?.length)">
      <h2>Modellaufrufe & Kosten</h2>
      <p class="lede-sm">
        Getrennt vom Aktionsprotokoll: echte/injizierte Modellaufrufe, Tokens und
        Kostenschätzung — nicht die simulierten Platform-Aktionen.
        {{ sim.cost_summary?.cost_estimate_disclaimer || 'Provider-Abrechnung kann abweichen.' }}
      </p>
      <div class="cost-grid">
        <div>
          <span class="k">API-Calls</span>
          <span class="v">{{ sim.cost_summary?.api_calls ?? 0 }} / {{ sim.cost_summary?.max_llm_calls_per_run ?? '—' }}</span>
        </div>
        <div>
          <span class="k">Tokens</span>
          <span class="v">{{ sim.cost_summary?.total_tokens ?? 0 }} / {{ sim.cost_summary?.max_tokens_per_run ?? '—' }}</span>
        </div>
        <div>
          <span class="k">Schätzung USD</span>
          <span class="v mono">
            {{ (sim.cost_summary?.estimated_cost_usd ?? 0).toFixed(6) }}
            / {{ sim.cost_summary?.max_cost_usd_per_run ?? '—' }}
          </span>
        </div>
        <div>
          <span class="k">Stop-Grund</span>
          <span class="v">{{ sim.stop_reason || sim.cost_summary?.limit_hit || '—' }}</span>
        </div>
      </div>
      <ol class="log calls" v-if="sim.model_calls?.length">
        <li v-for="c in sim.model_calls" :key="c.call_id">
          <span class="badge src-llm">model</span>
          #{{ c.call_id }} · agent {{ c.agent_id ?? '-' }} ·
          {{ c.prompt_tokens }}+{{ c.completion_tokens }} tok ·
          ${{ c.estimated_cost_usd.toFixed(6) }}
          <span class="src">({{ c.source }}{{ c.ok ? '' : ' · fail' }})</span>
        </li>
      </ol>
      <p v-else class="empty">Keine Modellaufrufe in diesem Lauf.</p>
    </section>

    <section class="grid">
      <div>
        <h2>Profile</h2>
        <p class="lede-sm">
          {{ displayAgents.length }} Agenten ·
          {{ sim?.agent_generation || agentGeneration }}
        </p>
        <ul class="agents">
          <li v-for="a in displayAgents" :key="a.agent_id">
            <span class="name">{{ a.name }}</span>
            <span class="desc">{{ a.description }}</span>
            <span class="goal">Ziel: {{ a.goal }}</span>
            <span class="desc">Stil: {{ a.style }}</span>
          </li>
        </ul>
      </div>

      <div>
        <h2>Live-Feed</h2>
        <ul class="feed" v-if="sim?.posts?.length">
          <li v-for="p in sim.posts" :key="String(p.post_id)">
            <span class="meta">#{{ p.post_id }} · user {{ p.user_id }}</span>
            <p>{{ p.content }}</p>
          </li>
        </ul>
        <p v-else class="empty">Noch keine Posts.</p>

        <h2>Kommentare & Reaktionen</h2>
        <ul class="feed compact" v-if="sim && (sim.comments.length || sim.reactions.length)">
          <li v-for="c in sim.comments" :key="'c' + c.comment_id">
            <span class="meta">Kommentar/Reply #{{ c.comment_id }} → Post {{ c.post_id }} · user {{ c.user_id }}</span>
            <p>{{ c.content }}</p>
          </li>
          <li v-for="(r, i) in sim.reactions" :key="'r' + i">
            <span class="meta">
              {{ r.type }} · user {{ r.user_id }}
              <span v-if="r.comment_id != null"> → comment {{ r.comment_id }}</span>
              <span v-else> → post {{ r.post_id }}</span>
            </span>
          </li>
        </ul>
        <p v-else class="empty">Noch keine Reaktionen.</p>
      </div>

      <div>
        <h2>Aktionsprotokoll</h2>
        <p class="lede-sm">
          <span class="badge src-manual">ManualAction</span>
          skriptierte Platform-Aktion ·
          <span class="badge src-llm">LLMAction</span>
          KI-Entscheidung (OASIS)
        </p>
        <ol class="log" v-if="sim?.action_log?.length">
          <li v-for="(e, i) in sim.action_log" :key="i">
            <span class="badge" :class="sourceClass(e.source)">{{ e.source || '?' }}</span>
            <span class="mono">s{{ e.step ?? '-' }}</span>
            agent {{ e.agent_id }} · {{ e.action_type }}
            <span class="src" v-if="e.kind">· {{ e.kind }}</span>
            <span class="src" v-if="e.goal">· Ziel: {{ e.goal }}</span>
          </li>
        </ol>
        <p v-else class="empty">Noch keine Aktionen.</p>

        <h2>Entscheidungen</h2>
        <p class="lede-sm">Szenario + Ziel + Sprache je LLM-Entscheidung.</p>
        <ol class="log" v-if="sim?.decision_log?.length">
          <li v-for="(d, i) in sim.decision_log" :key="'d' + i">
            <span class="badge src-llm">decision</span>
            <span class="mono">s{{ d.step ?? '-' }}</span>
            {{ d.agent_name || ('agent ' + d.agent_id) }}
            · {{ d.ok === false ? 'fail' : 'ok' }}
            <span class="src" v-if="d.tool_calls?.length">
              · {{ d.tool_calls.map((t) => t.action).join(', ') }}
            </span>
            <span class="src" v-if="d.goal">· {{ d.goal }}</span>
          </li>
        </ol>
        <p v-else class="empty">Noch keine LLM-Entscheidungen.</p>

        <h2>Runtime</h2>
        <ul class="meta-list" v-if="health">
          <li>
            Phase:
            {{ health.config.phase || health.runtime?.phase || '—' }}
            <span v-if="health.runtime?.code_revision" class="mono">
              · rev {{ health.runtime.code_revision }}
            </span>
          </li>
          <li>oasis import: {{ health.oasis_import_ok ? 'ok' : 'fail' }}</li>
          <li>camel-oasis {{ health.versions['camel-oasis'] }}</li>
          <li>mcp {{ health.versions.mcp }}</li>
          <li>API-Key im Frontend: {{ health.config.api_key_exposed === false ? 'nein' : '?' }}</li>
          <li>LLM-Freigabe nötig: {{ health.config.llm_run_requires_approval ? 'ja' : '—' }}</li>
          <li>Zep integriert: {{ health.config.zep_integrated === false ? 'nein (Vorbereitung)' : health.zep?.integrated === false ? 'nein' : '—' }}</li>
          <li>KG-Backend: {{ health.config.knowledge_graph_backend || '—' }}</li>
          <li>Agent-Gen-Default: {{ health.config.default_agent_generation || '—' }}</li>
          <li>Ergebnis: {{ sim?.result_path || '—' }}</li>
        </ul>
      </div>
    </section>

    <section class="society" v-if="societyOverview">
      <h2>Society-Übersicht</h2>
      <p class="lede-sm">
        {{ societyOverview.agent_count }} Agenten ·
        Ausführung: {{ executionKindLabel }} ·
        Live-LLM-Max {{ societyOverview.live_llm_max_agents ?? 5 }} ·
        Offline-Max {{ societyOverview.offline_max_agents ?? 50 }}
        <span v-if="societyOverview.paid_llm_unlocked_for_scale === false">
          · Scale ≠ bezahltes LLM
        </span>
      </p>
      <div class="progress-track" aria-hidden="true">
        <div class="progress-fill" :style="{ width: progressPct + '%' }" />
      </div>
      <div class="society-grid">
        <div>
          <span class="k">Posts</span>
          <span class="v">{{ societyOverview.posts ?? 0 }}</span>
        </div>
        <div>
          <span class="k">Kommentare</span>
          <span class="v">{{ societyOverview.comments ?? 0 }}</span>
        </div>
        <div>
          <span class="k">Reaktionen</span>
          <span class="v">{{ societyOverview.reactions ?? 0 }}</span>
        </div>
        <div>
          <span class="k">Positionswechsel</span>
          <span class="v">{{ societyOverview.change_count ?? 0 }}</span>
        </div>
        <div>
          <span class="k">Scheduler-Pläne</span>
          <span class="v">{{ societyOverview.scheduler_plans ?? sim?.scheduler?.plans ?? 0 }}</span>
        </div>
        <div>
          <span class="k">Selbst-als-Partner</span>
          <span class="v">{{ societyOverview.self_as_partner_count ?? 0 }}</span>
        </div>
      </div>

      <h3>Positionsverteilung</h3>
      <ul class="polarity-bars" v-if="polarityBars.length">
        <li v-for="b in polarityBars" :key="b.label">
          <span class="pol-label">{{ b.label }}</span>
          <span class="pol-track">
            <span class="pol-fill" :style="{ width: b.pct + '%' }" />
          </span>
          <span class="pol-count mono">{{ b.count }} ({{ b.pct }}%)</span>
        </li>
      </ul>
      <p v-else class="empty">Noch keine Polaritätsverteilung.</p>

      <h3>Netzwerk</h3>
      <p class="lede-sm" v-if="sim?.knowledge_graph?.network">
        {{ sim.knowledge_graph.network.actors ?? 0 }} Akteure ·
        {{ sim.knowledge_graph.network.statements ?? 0 }} Aussagen ·
        {{ sim.knowledge_graph.network.social_edges ?? 0 }} Social-Kanten
      </p>
      <div class="network-panel">
        <ul class="network-actors">
          <li v-for="a in networkActors" :key="a.id">{{ a.label }}</li>
        </ul>
        <ol class="log network-edges" v-if="networkEdges.length">
          <li v-for="e in networkEdges" :key="e.id">
            <span class="mono">{{ e.relation }}</span>
            · {{ e.source }} → {{ e.target }}
          </li>
        </ol>
        <p v-else class="empty">Noch keine Social-Kanten.</p>
      </div>
    </section>

    <section class="mem-grid">
      <div>
        <h2>Memory</h2>
        <p class="lede-sm">
          Lokales begrenztes Agentengedächtnis
          <span v-if="sim?.agent_memory?.max_entries_per_agent">
            (max {{ sim.agent_memory.max_entries_per_agent }}/Agent)
          </span>
          — Zep vorbereitet, nicht integriert.
        </p>
        <div v-if="memoryAgents.length" class="mem-list">
          <article v-for="m in memoryAgents" :key="m.id" class="mem-card">
            <h3>Agent {{ m.id }}</h3>
            <ul class="feed compact">
              <li v-for="(e, i) in m.entries.slice(-5)" :key="i">
                <span class="meta">{{ e.kind }} · s{{ e.step ?? '-' }}</span>
                <p>{{ e.text }}</p>
              </li>
            </ul>
          </article>
        </div>
        <p v-else class="empty">Noch keine Memory-Einträge.</p>
      </div>
      <div>
        <h2>Positionen</h2>
        <p class="lede-sm" v-if="positionSummary">
          {{ positionSummary.count }} Agenten ·
          {{ positionSummary.changes }} Positionswechsel (belegt)
        </p>
        <ul class="feed compact" v-if="positionSummary?.positions?.length">
          <li v-for="p in positionSummary.positions" :key="'pos' + p.agent_id">
            <span class="meta">
              Agent {{ p.agent_id }} · {{ p.polarity || '?' }}
              <span v-if="p.changed"> · geändert</span>
              <span v-if="p.influenced_by_agent_ids?.length">
                · beeinflusst von {{ p.influenced_by_agent_ids.join(', ') }}
              </span>
            </span>
            <p>{{ p.stance }}</p>
            <p v-if="p.previous_stance" class="meta">vorher: {{ p.previous_stance }}</p>
          </li>
        </ul>
        <p v-else class="empty">Positionen nach dem Lauf sichtbar.</p>

        <h2>Beziehungen</h2>
        <p class="lede-sm" v-if="graphSummary">
          {{ graphSummary.nodes }} Knoten · {{ graphSummary.edges }} Kanten —
          Social nur belegt: agrees / disagrees / responds_to / influenced_by /
          position_changed.
        </p>
        <ul class="meta-list" v-if="interactionSummary">
          <li>agrees_with: {{ interactionSummary.agrees_with }}</li>
          <li>disagrees_with: {{ interactionSummary.disagrees_with }}</li>
          <li>responds_to: {{ interactionSummary.responds_to }}</li>
          <li>influenced_by: {{ interactionSummary.influenced_by }}</li>
          <li>position_changed: {{ interactionSummary.position_changed }}</li>
          <li>replied_to (legacy): {{ interactionSummary.replied_to }}</li>
          <li>Kommentare: {{ interactionSummary.comments }}</li>
          <li>
            Reaktionen: Post {{ interactionSummary.post_reactions }} ·
            Kommentar {{ interactionSummary.comment_reactions }}
          </li>
        </ul>
        <p class="lede-sm" v-if="oasisReplyNote">
          OASIS Reply-API: parent_comment_id={{ oasisReplyNote.parent }} ·
          Reply via {{ oasisReplyNote.via }}
        </p>
        <ol class="log" v-if="graphSummary?.sampleEdges?.length">
          <li v-for="e in graphSummary.sampleEdges" :key="e.id">
            <span class="mono">{{ e.relation }}</span>
            · {{ e.source }} → {{ e.target }}
          </li>
        </ol>
        <p v-else class="empty">Beziehungen leer — nach dem Lauf sichtbar.</p>
      </div>
    </section>

    <section class="analytics" v-if="societyAnalytics || analyticsKpis">
      <h2>Society Analytics</h2>
      <p class="lede-sm">
        Phase-7 Kennzahlen — Memory, Beziehungen, Meinungsdynamik.
        Offline-Scale {{ societyAnalytics?.scale?.agent_count ?? societyOverview?.agent_count ?? '—' }}
        Agenten × ≤{{ societyAnalytics?.scale?.offline_max_rounds ?? 4 }} Runden ·
        paid_llm_unlocked={{ societyAnalytics?.scale?.paid_llm_unlocked === false ? 'false' : '—' }}
      </p>
      <div class="society-grid" v-if="analyticsKpis">
        <div>
          <span class="k">Memory-Coverage</span>
          <span class="v mono">{{ analyticsKpis.memory_coverage ?? '—' }}</span>
        </div>
        <div>
          <span class="k">Relationship-Density</span>
          <span class="v mono">{{ analyticsKpis.relationship_density ?? '—' }}</span>
        </div>
        <div>
          <span class="k">Opinion-Volatility</span>
          <span class="v mono">{{ analyticsKpis.opinion_volatility ?? '—' }}</span>
        </div>
        <div>
          <span class="k">Social-Edges</span>
          <span class="v mono">{{ analyticsKpis.social_edge_total ?? '—' }}</span>
        </div>
        <div>
          <span class="k">LT-Memory Agents</span>
          <span class="v">{{ societyAnalytics?.memory?.agents_with_long_term ?? societyOverview?.long_term_memory_agents ?? 0 }}</span>
        </div>
        <div>
          <span class="k">Bonds</span>
          <span class="v">{{ societyAnalytics?.relationships?.bond_count ?? societyOverview?.relationship_bonds ?? 0 }}</span>
        </div>
      </div>

      <h3>Meinungs-Chronologie</h3>
      <ol class="log" v-if="opinionChronology.length">
        <li v-for="f in opinionChronology" :key="'op' + f.step">
          <span class="mono">R{{ f.step }}</span>
          · Δ {{ f.changes_this_step ?? 0 }} /
          Σ {{ f.cumulative_changes ?? 0 }}
          · Polarität
          <span class="mono">{{ JSON.stringify(f.polarity_distribution || {}) }}</span>
        </li>
      </ol>
      <p v-else class="empty">Chronologie nach Offline-Lauf sichtbar.</p>

      <h3>Dynamische Beziehungen (evidenced)</h3>
      <ol class="log" v-if="relationshipBonds.length">
        <li v-for="b in relationshipBonds" :key="b.source_agent_id + '-' + b.target_agent_id">
          <span class="mono">u{{ b.source_agent_id }}→u{{ b.target_agent_id }}</span>
          · strength {{ b.strength }} · evid {{ b.evidence_count }}
          · {{ (b.relations || []).join(', ') }}
        </li>
      </ol>
      <p v-else class="empty">Noch keine belegten Bonds.</p>

      <h3>Long-Term Memory (sourced)</h3>
      <div v-if="longTermMemoryAgents.length" class="mem-list">
        <article v-for="m in longTermMemoryAgents" :key="'lt' + m.id" class="mem-card">
          <h3>Agent {{ m.id }}</h3>
          <ul class="feed compact">
            <li v-for="(e, i) in m.entries" :key="i">
              <span class="meta">
                {{ e.kind }} · s{{ e.step ?? '-' }}
                <span v-if="e.sources?.length"> · {{ e.sources.join(', ') }}</span>
              </span>
              <p>{{ e.text }}</p>
            </li>
          </ul>
        </article>
      </div>
      <p v-else class="empty">Long-Term Memory nach dem Lauf sichtbar (kein Zep).</p>
    </section>
  </div>
</template>

<style scoped>
.shell {
  max-width: 1120px;
  margin: 0 auto;
  padding: 2.5rem 1.25rem 4rem;
}

.hero {
  margin-bottom: 1.5rem;
  animation: rise 0.7s ease both;
}

.brand {
  font-family: var(--font-display);
  font-size: clamp(2.4rem, 6vw, 3.6rem);
  margin: 0;
  letter-spacing: -0.02em;
  line-height: 1;
}

h1 {
  font-family: var(--font-body);
  font-weight: 500;
  font-size: 1.05rem;
  margin: 0.65rem 0 0.4rem;
  color: var(--muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.lede {
  max-width: 36rem;
  color: var(--ink);
  opacity: 0.88;
  line-height: 1.5;
  margin: 0;
}

.banner {
  display: flex;
  gap: 0.85rem;
  align-items: baseline;
  flex-wrap: wrap;
  padding: 0.9rem 1rem;
  border: 1px solid var(--line);
  background: var(--panel);
  backdrop-filter: blur(8px);
  margin-bottom: 1.25rem;
  animation: rise 0.8s ease both 0.08s;
}

.banner.warn {
  border-color: rgba(255, 180, 84, 0.45);
}

.banner.ok {
  border-color: rgba(196, 240, 97, 0.35);
}

.banner strong {
  color: var(--accent);
}

.banner.warn strong {
  color: var(--warn);
}

.controls {
  border-top: 1px solid var(--line);
  padding-top: 1.25rem;
  margin-bottom: 1.25rem;
  animation: rise 0.85s ease both 0.12s;
}

.controls label {
  display: grid;
  gap: 0.4rem;
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 0.9rem;
}

textarea,
select,
input {
  width: 100%;
  background: rgba(8, 16, 14, 0.65);
  border: 1px solid var(--line);
  color: var(--ink);
  padding: 0.7rem 0.8rem;
  border-radius: 2px;
}

.row {
  display: grid;
  grid-template-columns: 1.2fr 0.55fr 0.45fr 1fr 0.45fr auto;
  gap: 0.8rem;
  align-items: end;
}

.cta {
  background: var(--accent);
  color: var(--accent-ink);
  border: 0;
  padding: 0.75rem 1.1rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  transition: transform 0.15s ease, filter 0.15s ease;
}

.cta:hover:not(:disabled) {
  transform: translateY(-1px);
  filter: brightness(1.05);
}

.cta:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.btn-secondary {
  background: transparent;
  color: var(--ink);
  border: 1px solid var(--line);
  padding: 0.75rem 1.1rem;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  z-index: 40;
  background: rgba(6, 12, 10, 0.72);
  display: grid;
  place-items: center;
  padding: 1.25rem;
  animation: fade 0.2s ease both;
}

.modal {
  width: min(32rem, 100%);
  background: linear-gradient(165deg, #1a3229, #101c18 70%);
  border: 1px solid var(--line);
  padding: 1.35rem 1.4rem 1.2rem;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
  animation: rise 0.25s ease both;
}

.modal h2 {
  margin: 0 0 0.5rem;
  font-size: 1.45rem;
}

.confirm-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem 1rem;
  margin: 1rem 0;
}

.confirm-grid dt {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0;
}

.confirm-grid dd {
  margin: 0.15rem 0 0;
  font-size: 1rem;
}

.disclaimer {
  font-size: 0.86rem;
  color: var(--warn);
  line-height: 1.4;
  margin: 0 0 1rem;
}

.muted {
  color: var(--muted);
  font-size: 0.85rem;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.65rem;
  flex-wrap: wrap;
}

.status-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem 1rem;
  padding: 1rem 0 1.4rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.25rem;
}

.cost-strip {
  border-bottom: 1px solid var(--line);
  padding-bottom: 1.25rem;
  margin-bottom: 1.5rem;
  animation: rise 0.9s ease both 0.16s;
}

.cost-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 0.75rem 0;
}

.lede-sm {
  color: var(--muted);
  font-size: 0.88rem;
  margin: 0 0 0.5rem;
  line-height: 1.4;
}

.mode-pill.mode-llm,
.badge.src-llm {
  color: var(--accent);
}

.mode-pill.mode-manual,
.badge.src-manual {
  color: var(--warn);
}

.mode-pill.mode-scripted {
  color: #9fd4ff;
}

.preset-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.preset {
  border: 1px solid var(--line);
  background: transparent;
  color: var(--ink);
  padding: 0.25rem 0.7rem;
  border-radius: 4px;
}

.preset.active,
.preset:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.analytics {
  margin: 1.75rem 0 2rem;
  padding: 1.1rem 1.15rem 1.25rem;
  border: 1px solid var(--line);
  background: linear-gradient(
    160deg,
    rgba(255, 255, 255, 0.04),
    rgba(196, 240, 97, 0.05)
  );
  animation: rise 0.85s ease both 0.12s;
}

.analytics h2 {
  margin: 0 0 0.35rem;
  font-family: var(--font-display);
}

.analytics h3 {
  margin: 1.1rem 0 0.4rem;
  font-size: 0.95rem;
}

.society {
  margin: 1.5rem 0;
  padding: 1.1rem 1.2rem;
  border: 1px solid var(--line);
  background: var(--panel);
  animation: rise 0.55s ease both;
}

.society h2 {
  margin: 0 0 0.35rem;
}

.society h3 {
  margin: 1.1rem 0 0.45rem;
  font-size: 0.95rem;
  font-weight: 600;
}

.society-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.65rem;
  margin-top: 0.75rem;
}

.progress-track {
  height: 6px;
  margin-top: 0.65rem;
  background: rgba(232, 242, 236, 0.08);
  border-radius: 999px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.35s ease;
}

.polarity-bars {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.4rem;
}

.polarity-bars li {
  display: grid;
  grid-template-columns: 7rem 1fr 5.5rem;
  gap: 0.5rem;
  align-items: center;
}

.pol-track {
  height: 8px;
  background: rgba(232, 242, 236, 0.08);
  border-radius: 999px;
  overflow: hidden;
}

.pol-fill {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #7fb89a, var(--accent));
}

.pol-label {
  font-size: 0.85rem;
  color: var(--muted);
}

.pol-count {
  font-size: 0.8rem;
  text-align: right;
}

.network-panel {
  display: grid;
  grid-template-columns: minmax(140px, 0.35fr) 1fr;
  gap: 0.85rem;
}

.network-actors {
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 220px;
  overflow: auto;
  font-size: 0.82rem;
  color: var(--muted);
}

.network-actors li {
  padding: 0.15rem 0;
  border-bottom: 1px solid var(--line);
}

.network-edges {
  max-height: 220px;
  overflow: auto;
}

@media (max-width: 720px) {
  .network-panel {
    grid-template-columns: 1fr;
  }
  .polarity-bars li {
    grid-template-columns: 1fr;
  }
}

.badge {
  display: inline-block;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-right: 0.35rem;
}

.badge.src-other {
  color: var(--muted);
}

.k {
  display: block;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.v {
  font-size: 1rem;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85rem;
}

.msg {
  grid-column: 1 / -1;
  margin: 0.2rem 0 0;
  color: var(--muted);
}

.stop-reason {
  grid-column: 1 / -1;
  margin: 0.35rem 0 0;
  padding: 0.55rem 0.7rem;
  border-left: 3px solid var(--muted);
  background: color-mix(in srgb, var(--panel, #1a1f2a) 92%, transparent);
}
.stop-reason.stop-limit {
  border-left-color: var(--warn, #e0b35a);
}
.stop-reason.stop-other {
  border-left-color: var(--danger, #e07a7a);
}
.stop-reason .v {
  display: block;
  margin-top: 0.15rem;
  font-size: 0.95rem;
  line-height: 1.35;
}

.status-ok {
  color: var(--ok, #7dcea0);
}
.status-limit {
  color: var(--warn, #e0b35a);
}
.status-fail {
  color: var(--danger, #e07a7a);
}
.status-run {
  color: var(--accent, #8ecae6);
}

.err {
  color: var(--danger);
  margin: 0.35rem 0 0;
}

.grid {
  display: grid;
  grid-template-columns: 0.9fr 1.2fr 0.9fr;
  gap: 1.5rem;
}

h2 {
  font-family: var(--font-display);
  font-weight: 400;
  font-size: 1.55rem;
  margin: 0 0 0.75rem;
}

.agents,
.feed,
.log,
.meta-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.agents li {
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--line);
  display: grid;
  gap: 0.2rem;
}

.agents .name {
  font-weight: 700;
}

.agents .desc,
.agents .goal,
.empty,
.src {
  color: var(--muted);
  font-size: 0.9rem;
}

.feed li {
  padding: 0.7rem 0;
  border-bottom: 1px solid var(--line);
}

.feed .meta {
  display: block;
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 0.25rem;
}

.feed p {
  margin: 0;
  line-height: 1.45;
}

.feed.compact li {
  padding: 0.45rem 0;
}

.log {
  max-height: 22rem;
  overflow: auto;
  font-size: 0.88rem;
}

.log li {
  padding: 0.35rem 0;
  border-bottom: 1px dashed var(--line);
}

.meta-list li {
  padding: 0.3rem 0;
  color: var(--muted);
  font-size: 0.9rem;
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@keyframes fade {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.mem-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-top: 1.75rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--line);
  animation: rise 0.95s ease both 0.2s;
}

.mem-list {
  display: grid;
  gap: 0.85rem;
}

.mem-card h3 {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  font-weight: 600;
}

@media (max-width: 900px) {
  .grid,
  .row,
  .status-strip,
  .cost-grid,
  .confirm-grid,
  .mem-grid {
    grid-template-columns: 1fr;
  }
}
</style>
