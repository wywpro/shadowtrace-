/** Core enumerations — must stay in sync with backend app/models/enums.py */

export type EventStatus =
  | "new"
  | "triaging"
  | "collecting_evidence"
  | "analyzing"
  | "scoring"
  | "planning_response"
  | "waiting_approval"
  | "executing_response"
  | "verifying"
  | "replanning"
  | "contained"
  | "failed"
  | "reporting"
  | "closed";

export type Severity = "low" | "medium" | "high" | "critical";

export type FinalVerdict =
  | "none"
  | "possible_false_positive"
  | "false_positive"
  | "confirmed_threat";

export type EventType =
  | "account_anomaly"
  | "host_compromise"
  | "data_exfiltration"
  | "insider_threat"
  | "malicious_process"
  | "suspicious_domain"
  | "lateral_movement"
  | "other";

export type DispositionPolicy = "required" | "not_required";

export type WritebackReadiness =
  | "not_required"
  | "ready"
  | "source_unresolved"
  | "not_configured"
  | "capability_unknown"
  | "capability_unsupported"
  | "permission_denied"
  | "connector_unavailable";

export type WritebackStatus =
  | "pending"
  | "sending"
  | "accepted"
  | "confirmed"
  | "partial"
  | "failed"
  | "conflict"
  | "unknown";

export type ActionStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "scheduled"
  | "dispatched"
  | "executing"
  | "success"
  | "failed"
  | "rolled_back"
  | "cancelled"
  | "unknown";

export type ActionLevel = "immediate" | "deferred";

export type ExecutionOwner = "xdr_managed" | "direct_tool";

export type ActionCategory =
  | "entity_containment"
  | "evidence_collection"
  | "event_status_update"
  | "notification"
  | "custom";

export type EvidenceSource =
  | "identity"
  | "endpoint"
  | "network_flow"
  | "data_security"
  | "dns"
  | "threat_intel"
  | "external_feed";

export type CollectionStatus =
  | "completed"
  | "partial_done"
  | "degraded"
  | "failed";

export type VerificationOverallStatus =
  | "success"
  | "partial"
  | "failed"
  | "waiting"
  | "manual_resolution";

export type ScoringMode = "llm_and_rule" | "rule_only";

export type ResponsePlanGeneratedBy = "llm" | "template";

export type EffectStatus = "verified" | "failed" | "skipped" | "unverifiable";

/* ------------------------------------------------------------------ */
/*  Entity models                                                     */
/* ------------------------------------------------------------------ */

export interface AccountEntity {
  entity_type: "account";
  accounts: string[];
}

export interface HostEntity {
  entity_type: "host";
  hosts: string[];
}

export interface IpEntity {
  entity_type: "ip";
  ips: string[];
}

export interface DomainEntity {
  entity_type: "domain";
  domains: string[];
}

export interface ProcessEntity {
  entity_type: "process";
  processes: string[];
}

export interface FileEntity {
  entity_type: "file";
  files: string[];
}

export type EntityItem =
  | AccountEntity
  | HostEntity
  | IpEntity
  | DomainEntity
  | ProcessEntity
  | FileEntity;

export interface EntitySet {
  accounts: string[];
  hosts: string[];
  ips: string[];
  domains: string[];
  processes: string[];
  files: string[];
}

/* ------------------------------------------------------------------ */
/*  Source / disposition models                                       */
/* ------------------------------------------------------------------ */

export interface SourceReference {
  source_id: string;
  source_type: string;
  object_kind: string;
  object_id: string;
  source_status_raw: string;
}

export interface SourceObjectLocator {
  source_id: string;
  source_type: string;
  object_kind: string;
  object_id: string;
}

export interface DispositionReceipt {
  disposition_id: string;
  event_id: string;
  source_locator: SourceObjectLocator;
  writeback_id: string | null;
  writeback_status: WritebackStatus | null;
  confirmation_evidence: string | null;
  submitted_at: string | null;
  confirmed_at: string | null;
}

/* ------------------------------------------------------------------ */
/*  Evidence models                                                   */
/* ------------------------------------------------------------------ */

export interface Evidence {
  evidence_id: string;
  event_id: string;
  source: EvidenceSource;
  evidence_type: string;
  description: string;
  confidence: number;
  timestamp: string;
  raw_data: Record<string, unknown> | null;
  is_conflicting: boolean;
}

export interface EvidenceConflict {
  conflict_id: string;
  evidence_a_id: string;
  evidence_b_id: string;
  description: string;
}

export interface EvidenceGap {
  gap_id: string;
  description: string;
  severity: "low" | "medium" | "high";
}

/* ------------------------------------------------------------------ */
/*  Risk models                                                       */
/* ------------------------------------------------------------------ */

export interface RiskFactor {
  factor_name: string;
  weight: number;
  raw_score: number;
  weighted_score: number;
  reasoning: string;
}

export interface RiskAssessment {
  risk_score: number;
  severity: Severity;
  confidence: number;
  risk_factors: RiskFactor[];
  possible_false_positive: boolean;
  scoring_mode: ScoringMode;
}

/* ------------------------------------------------------------------ */
/*  API response models                                               */
/* ------------------------------------------------------------------ */

export interface EventListItem {
  event_id: string;
  event_type: EventType;
  title: string;
  status: EventStatus;
  severity: Severity;
  risk_score: number;
  final_verdict: FinalVerdict;
  writeback_required: boolean;
  writeback_readiness: WritebackReadiness;
  writeback_overall_status: WritebackStatus | null;
  pending_writeback_count: number;
  created_at: string | null;
  updated_at: string | null;
  occurred_at: string | null;
}

export interface EventListResponse {
  total: number;
  page: number;
  page_size: number;
  items: EventListItem[];
}

export interface EventListParams {
  page?: number;
  page_size?: number;
  status?: EventStatus;
  severity?: Severity;
  event_type?: EventType;
  final_verdict?: FinalVerdict;
  keyword?: string;
  start_time?: string;
  end_time?: string;
  sort_by?: string;
  sort_order?: "asc" | "desc";
}

export interface SecurityEvent {
  event_id: string;
  event_type: EventType;
  title: string;
  description: string;
  status: EventStatus;
  severity: Severity;
  risk_score: number;
  confidence: number;
  final_verdict: FinalVerdict;
  entities: EntitySet;
  creation_source_ref: SourceReference;
  source_reference_snapshots: SourceReference[];
  current_primary_source_record_id: string | null;
  disposition_source_ref: SourceObjectLocator | null;
  disposition_policy: DispositionPolicy;
  raw_alert_ids: string[];
  raw_alert_snapshot: Record<string, unknown> | null;
  source_type: string | null;
  occurred_at: string | null;
  created_at: string | null;
  updated_at: string | null;
  closed_at: string | null;
  replan_count: number;
  degraded_flags: string[];
  escalated: boolean;
  external_unsynced: boolean;
  event_context_snapshot: Record<string, unknown> | null;
  row_version: number;
}

export interface EventDetailResponse {
  event: SecurityEvent;
  writeback_required: boolean;
  writeback_readiness: WritebackReadiness;
  writeback_overall_status: WritebackStatus | null;
  pending_writeback_count: number;
}

export interface InvestigationResult {
  event_id: string;
  final_status: EventStatus;
  final_verdict: FinalVerdict;
  escalated: boolean;
  external_unsynced: boolean;
  report_id: string | null;
  writeback_required: boolean;
  writeback_readiness: WritebackReadiness;
  writeback_overall_status: WritebackStatus | null;
  pending_writeback_ids: string[];
}

export interface ExecutionJob {
  job_id: string;
  event_id: string;
  action_id: string;
  status: string;
  result: Record<string, unknown> | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface WritebackRecord {
  writeback_id: string;
  event_id: string;
  disposition_id: string;
  status: WritebackStatus;
  confirmation_evidence: string | null;
  submitted_at: string | null;
  confirmed_at: string | null;
  retry_count: number;
  error_detail: string | null;
}
