/** Action models — matching backend app/models/action.py + openapi.json */

import type { ActionCategory, ActionLevel, ActionStatus, ExecutionOwner } from "./event";

export interface Action {
  action_id: string;
  event_id: string;
  action_level: ActionLevel;
  category: ActionCategory;
  owner: ExecutionOwner;
  tool_name: string;
  tool_params: Record<string, unknown>;
  status: ActionStatus;
  rationale: string;
  created_at: string | null;
  updated_at: string | null;
  approved_by: string | null;
  approved_at: string | null;
  dispatched_at: string | null;
  completed_at: string | null;
  error_detail: string | null;
  retry_count: number;
  max_retries: number;
  rollback_action_id: string | null;
  is_rollback: boolean;
}

export interface ActionListResponse {
  total: number;
  page: number;
  page_size: number;
  items: Action[];
}
