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
  goal?: string
  language?: string
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
  max_tokens_per_call?: number
  max_cost_usd_per_run?: number
  remaining_calls?: number
  remaining_tokens?: number
  limit_hit?: string | null
  model?: string
  cost_estimate_disclaimer?: string
}

export interface DecisionLogEntry {
  step?: number | null
  agent_id?: number | null
  agent_name?: string
  goal?: string
  scenario?: string
  language?: string
  ok?: boolean
  tool_calls?: Array<{ action?: string; args?: unknown }>
  error?: string
  note?: string
  env_preview?: string
  ts?: string
}

export interface KnowledgeGraph {
  backend?: string
  node_count?: number
  edge_count?: number
  counts_by_kind?: Record<string, number>
  counts_by_relation?: Record<string, number>
  interaction_relations?: {
    replied_to?: number
    agrees_with?: number
    disagrees_with?: number
    responds_to?: number
    influenced_by?: number
    position_changed?: number
  }
  social_relations?: {
    agrees_with?: number
    disagrees_with?: number
    responds_to?: number
    influenced_by?: number
    position_changed?: number
  }
  nodes?: Array<{
    id: string
    kind: string
    label: string
    meta?: Record<string, unknown>
  }>
  edges?: Array<{
    id: string
    source: string
    target: string
    relation: string
    meta?: Record<string, unknown>
  }>
  note?: string
}

export interface InteractionSummary {
  replied_to?: number
  agrees_with?: number
  disagrees_with?: number
  responds_to?: number
  influenced_by?: number
  position_changed?: number
  comments?: number
  post_reactions?: number
  comment_reactions?: number
  note?: string
}

export interface AgentPosition {
  agent_id: number
  stance?: string
  stance_key?: string
  polarity?: string
  step?: number | null
  evidence?: string
  evidence_kind?: string
  changed?: boolean
  previous_stance?: string | null
  influenced_by_agent_ids?: number[]
}

export interface AgentPositionsSnapshot {
  backend?: string
  agent_count?: number
  change_count?: number
  positions?: AgentPosition[]
  changes?: Array<Record<string, unknown>>
  note?: string
}

export interface OasisReplyInfo {
  parent_comment_id_supported?: boolean
  separate_reply_action_type?: boolean
  reply_via?: string
  documented_alternative?: string
  graph_relations_evidenced_only?: string[]
}

export interface AgentMemorySnapshot {
  backend?: string
  zep_integrated?: boolean
  max_entries_per_agent?: number
  agent_count_with_memory?: number
  agents?: Record<
    string,
    Array<{ kind?: string; text?: string; step?: number | null; ts?: string }>
  >
  note?: string
}

export interface SimulationState {
  run_id: string
  status: string
  mode: string
  scenario: string
  language?: string
  step: number
  max_steps: number
  agent_count?: number
  agent_generation?: string
  llm_configured: boolean
  message: string
  stop_reason?: string | null
  agents: AgentProfile[]
  posts: Array<Record<string, unknown>>
  comments: Array<Record<string, unknown>>
  reactions: Array<Record<string, unknown>>
  action_log: ActionLogEntry[]
  decision_log?: DecisionLogEntry[]
  model_calls: ModelCallEntry[]
  cost_summary: CostSummary
  agent_memory?: AgentMemorySnapshot
  agent_positions?: AgentPositionsSnapshot
  knowledge_graph?: KnowledgeGraph
  interaction_summary?: InteractionSummary
  oasis_reply?: OasisReplyInfo
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
    llm_run_requires_approval?: boolean
    default_mode?: string
    openai_model: string
    openai_base_url?: string
    max_agents?: number
    default_agent_count?: number
    default_agent_generation?: string
    agent_generation_modes?: string[]
    agent_local_memory_entries?: number
    max_simulation_steps: number
    max_llm_calls_per_run?: number
    max_tokens_per_call?: number
    max_tokens_per_run?: number
    max_cost_usd_per_run?: number
    estimated_max_cost_usd?: number
    cost_estimate_disclaimer?: string
    llm_max_retries?: number
    agent_memory_window?: number
    simulation_language?: string
    model_calls_per_round_required?: number
    model_calls_budget_note?: string
    terminal_statuses?: string[]
    live_test_limits?: {
      max_agents: number
      max_rounds: number
      max_model_calls: number
      max_retries: number
    }
    zep_enabled: boolean
    zep_configured: boolean
    zep_integrated?: boolean
    knowledge_graph_backend?: string
    phase?: string
    oasis_reply?: OasisReplyInfo
    social_dynamics?: {
      persistent_positions?: boolean
      relations_evidenced_only?: string[]
    }
    api_key_exposed?: boolean
  }
  runtime?: {
    phase?: string
    code_revision?: string
    expects_phase?: string
    port?: number
  }
  zep: Record<string, unknown>
}
