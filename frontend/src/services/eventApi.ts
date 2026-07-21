/** Event API — all /api/v1/events endpoints (ISSUE-067). */

import apiClient from "./apiClient";
import type {
  EventDetailResponse,
  EventListParams,
  EventListResponse,
} from "../types/event";
import type { ActionListResponse } from "../types/action";
import type { InvestigationReport } from "../types/report";
import type { AgentTrace, AuditLog, DecisionTrace } from "../types/trace";

// ------------------------------------------------------------------ //
// Events
// ------------------------------------------------------------------ //

export function listEvents(params?: EventListParams) {
  return apiClient.get<EventListResponse>("/events", { params });
}

export function getEvent(eventId: string) {
  return apiClient.get<EventDetailResponse>(`/events/${eventId}`);
}

export function triggerInvestigation(eventId: string) {
  return apiClient.post<{ event_id: string; status: string }>(
    `/events/${eventId}/investigate`,
  );
}

export function closeEvent(eventId: string) {
  return apiClient.post<{ event_id: string; status: string }>(
    `/events/${eventId}/close`,
  );
}

// ------------------------------------------------------------------ //
// Report & traces
// ------------------------------------------------------------------ //

export function getReport(eventId: string) {
  return apiClient.get<InvestigationReport>(`/events/${eventId}/report`);
}

export function getTraces(eventId: string) {
  return apiClient.get<AgentTrace[]>(`/events/${eventId}/traces`);
}

export function getAuditLogs(eventId: string) {
  return apiClient.get<AuditLog[]>(`/events/${eventId}/audit-logs`);
}

export function getToolCalls(eventId: string) {
  return apiClient.get<unknown[]>(`/events/${eventId}/tool-calls`);
}

export function getDecisionTrace(eventId: string) {
  return apiClient.get<DecisionTrace[]>(`/events/${eventId}/decision-trace`);
}

// ------------------------------------------------------------------ //
// Actions
// ------------------------------------------------------------------ //

export function listActions(
  eventId: string,
  params?: { page?: number; page_size?: number },
) {
  return apiClient.get<ActionListResponse>(`/events/${eventId}/actions`, {
    params,
  });
}

export function approveAction(actionId: string) {
  return apiClient.post(`/actions/${actionId}/approve`);
}

// ------------------------------------------------------------------ //
// Source records & connectors
// ------------------------------------------------------------------ //

export function getSourceRecord(eventId: string) {
  return apiClient.get<unknown>(`/events/${eventId}/source-record`);
}

export function listConnectors() {
  return apiClient.get<unknown[]>("/connectors");
}

// ------------------------------------------------------------------ //
// Execution jobs
// ------------------------------------------------------------------ //

export function getExecutionJob(eventId: string, jobId: string) {
  return apiClient.get<unknown>(`/events/${eventId}/execution-jobs/${jobId}`);
}

// ------------------------------------------------------------------ //
// Dispositions
// ------------------------------------------------------------------ //

export function listDispositions(eventId: string) {
  return apiClient.get<unknown[]>(`/events/${eventId}/dispositions`);
}

export function getDisposition(dispositionId: string) {
  return apiClient.get<unknown>(`/dispositions/${dispositionId}`);
}

export function selectDispositionSource(
  eventId: string,
  sourceLocator: Record<string, unknown>,
) {
  return apiClient.put(`/events/${eventId}/disposition-source`, sourceLocator);
}

// ------------------------------------------------------------------ //
// Writebacks
// ------------------------------------------------------------------ //

export function getWriteback(writebackId: string) {
  return apiClient.get<unknown>(`/writebacks/${writebackId}`);
}

export function retryWriteback(writebackId: string) {
  return apiClient.post(`/writebacks/${writebackId}/retry`);
}

// ------------------------------------------------------------------ //
// Admin-only resolve actions
// ------------------------------------------------------------------ //

export function resolveUnknownAction(actionId: string, resolution: string) {
  return apiClient.post(`/actions/${actionId}/resolve`, { resolution });
}

export function resolveWriteback(writebackId: string, resolution: string) {
  return apiClient.post(`/writebacks/${writebackId}/resolve`, { resolution });
}
