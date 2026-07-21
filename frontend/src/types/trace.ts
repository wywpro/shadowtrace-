/** Trace / audit models — matching backend agent trace + openapi.json */

export interface AgentTrace {
  trace_id: string;
  event_id: string;
  agent_name: string;
  status: "completed" | "failed" | "processing";
  input_data: Record<string, unknown> | null;
  output_data: Record<string, unknown> | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  error_detail: string | null;
  llm_model: string | null;
  llm_tokens_used: number | null;
}

export interface DecisionTrace {
  decision_id: string;
  event_id: string;
  agent_name: string;
  decision_type: string;
  observation_summary: string;
  evidence_refs: string[];
  candidate_actions: string[];
  selected_action: string | null;
  confidence: number | null;
  timestamp: string;
}

export interface AuditLog {
  log_id: string;
  event_id: string;
  actor: string;
  action: string;
  detail: string;
  timestamp: string;
}
