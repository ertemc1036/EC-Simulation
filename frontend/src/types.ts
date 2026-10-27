export interface AgentProfile {
  agent_id: number
  user_name: string
  name: string
  description: string
  goal: string
  style: string
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
  action_log: Array<Record<string, unknown>>
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
    max_simulation_steps: number
    zep_enabled: boolean
    zep_configured: boolean
  }
  zep: Record<string, unknown>
}
