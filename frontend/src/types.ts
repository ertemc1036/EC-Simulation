export interface AgentProfile {
  agent_id: number
  user_name: string
  name: string
  description: string
  goal: string
  style: string
}

export interface ActionLogEntry {
  step?: number | null
  agent_id?: number | null
  action_type?: string
  action_args?: unknown
  source?: string
  kind?: string
  note?: string
  via?: string
  ts?: string
}

export interface ModelCallEntry {
  call_id: number
  ts: string
  agent_id: number | null
  model: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  estimated_cost_usd: number
  ok: boolean
  error: string | null
  source: string
}

export interface CostSummary {
  api_calls?: number
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  estimated_cost_usd?: number
  max_llm_calls_per_run?: number
  max_tokens_per_run?: number
  remaining_calls?: number
  remaining_tokens?: number
  limit_hit?: string | null
  model?: string
}

export interface SimulationState {
  run_id: string
  status: string
  mode: string
  scenario: string
  step: number
  max_steps: number
  llm_configured: boolean
  message: string
  agents: AgentProfile[]
  posts: Array<Record<string, unknown>>
  comments: Array<Record<string, unknown>>
  reactions: Array<Record<string, unknown>>
  action_log: ActionLogEntry[]
  model_calls: ModelCallEntry[]
  cost_summary: CostSummary
  error: string | null
  result_path: string | null
  db_path: string | null
  started_at: string | null
  finished_at: string | null
}

export interface HealthResponse {
  ok: boolean
  oasis_import_ok: boolean
  versions: Record<string, string | null>
  config: {
    llm_configured: boolean
    llm_mode_available: boolean
    llm_mode_blocked_reason: string | null
    openai_model: string
    openai_base_url?: string
    max_simulation_steps: number
    max_llm_calls_per_run?: number
    max_tokens_per_run?: number
    agent_memory_window?: number
    zep_enabled: boolean
    zep_configured: boolean
    api_key_exposed?: boolean
  }
  zep: Record<string, unknown>
}
