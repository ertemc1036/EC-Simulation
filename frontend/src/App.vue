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
const maxSteps = ref(2)
const busy = ref(false)
const error = ref<string | null>(null)
const showLlmConfirm = ref(false)
let socket: Socket | null = null

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

const llmConfirmSummary = computed(() => {
  const cfg = health.value?.config
  const agentCount = agents.value.length || cfg?.max_agents || 5
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
    agentCount,
    rounds,
    callLimit,
    estimatedMaxCost: Number(est),
    hardCostCap: cfg?.max_cost_usd_per_run ?? 0.2,
    disclaimer,
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
  const res = await fetch(`${API}/api/agents`)
  const data = await res.json()
  agents.value = data.agents
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
        Fünf Persönlichkeiten diskutieren dein Szenario in einer echten
        camel-oasis-Umgebung — live im Dashboard.
      </p>
    </header>

    <section v-if="llmBanner" class="banner" :class="llmBanner.tone" aria-live="polite">
      <strong>{{ llmBanner.tone === 'ok' ? 'LLM' : 'Kein API-Key' }}</strong>
      <span>{{ llmBanner.text }}</span>
    </section>

    <section class="controls">
      <label>
        Szenario
        <textarea v-model="scenario" rows="3" maxlength="800" />
      </label>
      <div class="row">
        <label>
          Modus
          <select v-model="mode">
            <option value="manual">manual (StubModel, keine LLM-Calls) — Standard</option>
            <option value="llm">llm (benötigt Freigabe + API-Key im Backend-.env)</option>
          </select>
        </label>
        <label>
          Schritte
          <input v-model.number="maxSteps" type="number" min="1" max="2" />
        </label>
        <button class="cta" :disabled="busy || sim?.status === 'running'" @click="requestStart">
          {{ sim?.status === 'running' ? 'Läuft…' : 'Simulation starten' }}
        </button>
      </div>
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
            <dd>{{ llmConfirmSummary.agentCount }}</dd>
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
        <span class="v">{{ sim.status }}</span>
      </div>
      <div>
        <span class="k">Modus</span>
        <span class="v mode-pill" :class="sim.mode === 'llm' ? 'mode-llm' : 'mode-manual'">
          {{ sim.mode === 'llm' ? 'LLMAction' : 'ManualAction' }}
        </span>
      </div>
      <div>
        <span class="k">Schritt</span>
        <span class="v">{{ sim.step }} / {{ sim.max_steps }}</span>
      </div>
      <div>
        <span class="k">Run</span>
        <span class="v mono">{{ sim.run_id }}</span>
      </div>
      <p class="msg">{{ sim.message }}</p>
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
          <span class="k">Limit</span>
          <span class="v">{{ sim.cost_summary?.limit_hit || '—' }}</span>
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
        <h2>Agenten</h2>
        <ul class="agents">
          <li v-for="a in agents" :key="a.agent_id">
            <span class="name">{{ a.name }}</span>
            <span class="desc">{{ a.description }}</span>
            <span class="goal">Ziel: {{ a.goal }}</span>
          </li>
        </ul>
      </div>

      <div>
        <h2>Beiträge</h2>
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
            <span class="meta">Kommentar #{{ c.comment_id }} → Post {{ c.post_id }} · user {{ c.user_id }}</span>
            <p>{{ c.content }}</p>
          </li>
          <li v-for="(r, i) in sim.reactions" :key="'r' + i">
            <span class="meta">{{ r.type }} · user {{ r.user_id }} → post {{ r.post_id }}</span>
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
          </li>
        </ol>
        <p v-else class="empty">Noch keine Aktionen.</p>

        <h2>Runtime</h2>
        <ul class="meta-list" v-if="health">
          <li>oasis import: {{ health.oasis_import_ok ? 'ok' : 'fail' }}</li>
          <li>camel-oasis {{ health.versions['camel-oasis'] }}</li>
          <li>camel-ai {{ health.versions['camel-ai'] }}</li>
          <li>mcp {{ health.versions.mcp }}</li>
          <li>API-Key im Frontend: {{ health.config.api_key_exposed === false ? 'nein' : '?' }}</li>
          <li>LLM-Freigabe nötig: {{ health.config.llm_run_requires_approval ? 'ja' : '—' }}</li>
          <li>Ergebnis: {{ sim?.result_path || '—' }}</li>
        </ul>
      </div>
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
  grid-template-columns: 1.4fr 0.5fr auto;
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

@media (max-width: 900px) {
  .grid,
  .row,
  .status-strip,
  .cost-grid,
  .confirm-grid {
    grid-template-columns: 1fr;
  }
}
</style>
