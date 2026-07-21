/** Report models — matching backend app/models/report.py + openapi.json */

import type { EventType, FinalVerdict, Severity } from "./event";

export interface InvestigationReport {
  report_id: string;
  event_id: string;
  title: string;
  event_type: EventType;
  severity: Severity;
  final_verdict: FinalVerdict;
  risk_score: number;
  confidence: number;
  executive_summary: string;
  attack_storyline: string;
  evidence_summary: string;
  actions_summary: string;
  writeback_summary: string;
  generated_at: string | null;
  updated_at: string | null;
}
