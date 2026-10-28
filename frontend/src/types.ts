export interface AgentProfile {
  agent_id: number
  user_name: string
  name: string
  description: string
  goal: string
  style: string
  age?: number | null
  occupation?: string | null
  role_label?: string | null
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
  network?: {
    actors?: number
    statements?: number
    topics?: number
    social_edges?: number
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
  polarity_distribution?: Record<string, number>
  positions?: AgentPosition[]
  changes?: Array<Record<string, unknown>>
  note?: string
}

export interface SocietyOverview {
  agent_count?: number
  step?: number | null
  posts?: number
  comments?: number
  reactions?: number
  polarity_distribution?: Record<string, number>
  change_count?: number
  social_relations?: Record<string, number>
  network?: {
    actors?: number
    statements?: number
    topics?: number
    social_edges?: number
  }
  execution_kind?: string
  presets?: number[]
  live_llm_max_agents?: number
  offline_max_agents?: number
  paid_llm_unlocked_for_scale?: boolean
  scheduler_plans?: number
  self_as_partner_count?: number
  long_term_memory_agents?: number
  relationship_bonds?: number
  opinion_rounds_tracked?: number
  analytics_kpis?: Record<string, number>
}

export interface SchedulerSnapshot {
  backend?: string
  plans?: number
  unique_pairs?: number
  self_as_partner_count?: number
  self_reference_blocked?: boolean
  recent?: Array<Record<string, unknown>>
  note?: string
}

export interface OasisReplyInfo {
  parent_comment_id_supported?: boolean
  separate_reply_action_type?: boolean
  reply_via?: string
  documented_alternative?: string
  graph_relations_evidenced_only?: string[]
}

export interface MemoryEntry {
  kind?: string
  text?: string
  step?: number | null
  ts?: string
  sources?: string[]
  meta?: Record<string, unknown>
}

export interface AgentMemorySnapshot {
  backend?: string
  zep_integrated?: boolean
  max_entries_per_agent?: number
  max_long_term_per_agent?: number
  agent_count_with_memory?: number
  agent_count_with_long_term?: number
  long_term_sourced_entries?: number
  agents?: Record<string, MemoryEntry[]>
  long_term?: Record<string, MemoryEntry[]>
  note?: string
}

export interface RelationshipBond {
  source_agent_id: number
  target_agent_id: number
  relations?: string[]
  evidence_count?: number
  strength?: number
  first_step?: number | null
  last_step?: number | null
  evidences?: Array<Record<string, unknown>>
}

export interface RelationshipsSnapshot {
  backend?: string
  evidenced_only?: boolean
  bond_count?: number
  mean_strength?: number
  by_relation?: Record<string, number>
  bonds?: RelationshipBond[]
  chronology?: Array<Record<string, unknown>>
  note?: string
}

export interface OpinionDynamicsSnapshot {
  backend?: string
  rounds_tracked?: number
  total_evidenced_changes?: number
  shift_rate_per_agent_round?: number
  transitions?: Record<string, number>
  chronology?: Array<{
    step?: number
    agent_count?: number
    polarity_distribution?: Record<string, number>
    changes_this_step?: number
    cumulative_changes?: number
    change_samples?: Array<Record<string, unknown>>
  }>
  note?: string
}

export interface SocietyAnalytics {
  phase?: number
  execution_kind?: string
  scale?: {
    agent_count?: number
    step?: number | null
    max_steps?: number
    offline_presets?: number[]
    offline_max_rounds?: number
    paid_llm_unlocked?: boolean
  }
  memory?: Record<string, unknown>
  relationships?: Record<string, unknown>
  opinion?: Record<string, unknown>
  engagement?: Record<string, unknown>
  kpis?: {
    memory_coverage?: number
    relationship_density?: number
    opinion_volatility?: number
    social_edge_total?: number
  }
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
  execution_kind?: string
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
  relationships?: RelationshipsSnapshot
  opinion_dynamics?: OpinionDynamicsSnapshot
  society_analytics?: SocietyAnalytics
  knowledge_graph?: KnowledgeGraph
  interaction_summary?: InteractionSummary
  oasis_reply?: OasisReplyInfo
  scheduler?: SchedulerSnapshot
  society_overview?: SocietyOverview
  discussion?: DiscussionSnapshot
  error: string | null
  result_path: string | null
  db_path: string | null
  started_at: string | null
  finished_at: string | null
}

export interface DiscussionThreadEvent {
  comment_id: number
  post_id?: number | null
  author_id?: number | null
  parent_comment_id?: number | null
  recipient_id?: number | null
  round_n?: number | null
  self_reply?: boolean
  snippet?: string
}

export interface DiscussionSnapshot {
  backend?: string
  quality_fix?: string
  max_argument_history_per_agent?: number
  max_global_arguments?: number
  argument_count?: number
  agents_with_arguments?: number
  repetition_blocks?: number
  thread_events?: number
  self_replies?: number
  cross_agent_replies?: number
  parent_comment_id_supported?: boolean
  oasis_nested_reply?: boolean
  arguments?: Array<Record<string, unknown>>
  threads?: DiscussionThreadEvent[]
  note?: string
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
    offline_max_agents?: number
    default_agent_count?: number
    default_agent_generation?: string
    agent_generation_modes?: string[]
    agent_local_memory_entries?: number
    agent_long_term_memory_entries?: number
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
      max_tokens_per_run?: number
      max_tokens_per_call?: number
      max_cost_usd_per_run?: number
    }
    offline_scale?: {
      max_agents?: number
      presets?: number[]
      max_rounds?: number
      paid_llm_unlocked?: boolean
      note?: string
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
      scheduler?: boolean
      offline_scale_presets?: number[]
      long_term_memory?: boolean
      relationship_dynamics?: boolean
      opinion_chronology?: boolean
      society_analytics?: boolean
    }
    society_intelligence?: {
      long_term_memory_sourced?: boolean
      relationship_dynamics_evidenced_only?: boolean
      opinion_chronology?: boolean
      scripted_scenarios_max_rounds?: number
      analytics_dashboard?: boolean
      zep_integrated?: boolean
      intelligent_discussion?: boolean
      individual_personas?: boolean
      quality_fix?: string
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
