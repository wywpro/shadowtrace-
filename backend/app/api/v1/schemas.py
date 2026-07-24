"""API-layer request/response models (ISSUE-004).

Response models embed the ISSUE-002 core models. The unified error body and
pagination body follow intro §4.2. Request models are ``extra="forbid"`` so the
client can never smuggle server-controlled fields (e.g. ``operator``).

These placeholder responses return static, schema-valid example data so the
frontend and backend can develop in parallel against stable contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.action import Action
from app.models.disposition import (
    DispositionCommand,
    SetEventDispositionParams,
    SourceObjectLocator,
    TargetWritebackResult,
)
from app.models.enums import (
    ConfirmationEvidence,
    DispositionIntentKind,
    DispositionPolicy,
    EventStatus,
    EventType,
    ExecutionOwner,
    FinalVerdict,
    Severity,
    SourceDisposition,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.report import InvestigationReport, ReportSection

# EventListItem / EventSummary live in app.models.security_event (ISSUE-094 §2)
# so EventContext.event can be typed without the models layer depending on the
# API layer; re-exported here for backward-compatible ``from
# app.api.v1.schemas import EventSummary`` call sites.
from app.models.security_event import EventListItem as EventListItem
from app.models.security_event import EventSummary as EventSummary
from app.models.security_event import SecurityEvent
from app.models.source import SourceReference


class _StrictRequest(BaseModel):
    """Base for request bodies: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Error + pagination envelopes (intro §4.2.3)
# --------------------------------------------------------------------------- #
class ErrorResponse(BaseModel):
    error_code: str
    error_message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PageMeta(BaseModel):
    total: int = 0
    page: int = 1
    page_size: int = 20


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class EventCreateRequest(_StrictRequest):
    event_type: EventType
    title: str
    description: str = ""
    severity: Severity = Severity.LOW
    creation_source_ref: SourceReference


class InvestigateRequest(_StrictRequest):
    force_replan: bool = False


class EventCloseRequest(_StrictRequest):
    reason: str
    final_verdict: FinalVerdict | None = None
    need_investigation: bool | None = None
    force_local_close: bool = False


class ActionApproveRequest(_StrictRequest):
    comment: str | None = None
    decision_id: str | None = None


class ActionRejectRequest(_StrictRequest):
    comment: str | None = None
    decision_id: str | None = None


class ResolveUnknownRequest(_StrictRequest):
    resolution: Literal[
        "success",
        "partial_success",
        "failed",
        "mark_success",
        "mark_failed",
        "manual_confirmed",
    ]
    comment: str
    evidence_ref: str | None = None


class ResolveWritebackRequest(_StrictRequest):
    resolution: Literal["manual_confirmed", "mark_failed", "abandon"]
    comment: str
    evidence_ref: str | None = None


class SelectDispositionSourceRequest(_StrictRequest):
    source_record_id: str
    expected_event_version: int
    comment: str | None = None


class RecheckDispositionReadinessRequest(_StrictRequest):
    expected_event_version: int


class IngestSourceRecordRequest(_StrictRequest):
    reference: SourceReference
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Event responses
# --------------------------------------------------------------------------- #
# EventListItem / EventSummary are defined in app.models.security_event and
# imported above.


class EventListResponse(PageMeta):
    items: list[EventListItem] = Field(default_factory=list)


class EventDetailResponse(BaseModel):
    event: SecurityEvent
    writeback_required: bool
    writeback_readiness: WritebackReadiness
    writeback_overall_status: WritebackStatus | None = None
    pending_writeback_count: int = 0


class InvestigateResponse(BaseModel):
    event_id: str
    task_id: str
    status: EventStatus


class EventCloseResponse(BaseModel):
    event_id: str
    status: EventStatus
    final_verdict: FinalVerdict
    external_unsynced: bool = False


class ReportResponse(BaseModel):
    report: InvestigationReport


class TraceItem(BaseModel):
    trace_id: str
    agent_name: str
    status: str
    duration_ms: int | None = None
    started_at: datetime | None = None


class TracesResponse(PageMeta):
    items: list[TraceItem] = Field(default_factory=list)


class AuditLogItem(BaseModel):
    id: int
    from_status: str | None = None
    to_status: str | None = None
    operator: str | None = None
    reason: str | None = None
    created_at: datetime | None = None


class AuditLogsResponse(PageMeta):
    items: list[AuditLogItem] = Field(default_factory=list)


class ToolCallItem(BaseModel):
    call_id: str
    event_id: str
    action_id: str | None = None
    tool_name: str
    tool_category: str
    status: str
    duration_ms: int | None = None


class ToolCallsResponse(PageMeta):
    items: list[ToolCallItem] = Field(default_factory=list)


class TimelineItem(BaseModel):
    timestamp: datetime
    kind: str
    summary: str


class TimelineResponse(BaseModel):
    event_id: str
    items: list[TimelineItem] = Field(default_factory=list)


class GraphResponse(BaseModel):
    event_id: str
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class DecisionTraceResponse(BaseModel):
    event_id: str
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ActionListResponse(PageMeta):
    items: list[Action] = Field(default_factory=list)


class ActionOperationResponse(BaseModel):
    action_id: str
    status: str
    decision_id: str | None = None
    message: str = ""


# --------------------------------------------------------------------------- #
# Source / connector responses
# --------------------------------------------------------------------------- #
class IngestSourceRecordResponse(BaseModel):
    source_record_id: str
    event_id: str | None = None
    accepted: bool = True


class SourceRecordResponse(BaseModel):
    source_record_id: str
    reference: SourceReference
    normalized: dict[str, Any] = Field(default_factory=dict)
    current_source_disposition: SourceDisposition = SourceDisposition.UNKNOWN
    source_sync_state: str | None = None


class ConnectorPublic(BaseModel):
    """Connector view exposing capability/health but NEVER credential refs."""

    connector_id: str
    source_product: str
    display_name: str
    device_type: str | None = None
    status: str
    capabilities: dict[str, str] = Field(default_factory=dict)
    # None means "not provisioned" — live connectors must fail closed rather than
    # silently reporting NOT_REQUIRED to API clients.
    disposition_policy_default: DispositionPolicy | None = None
    last_sync_at: datetime | None = None


class ConnectorsResponse(BaseModel):
    items: list[ConnectorPublic] = Field(default_factory=list)


class DispositionSourceSelectResponse(BaseModel):
    event_id: str
    disposition_source_ref: SourceObjectLocator
    event_version: int


class ReadinessRecheckResponse(BaseModel):
    event_id: str
    writeback_readiness: WritebackReadiness
    blocked_reason: str | None = None
    event_version: int


# --------------------------------------------------------------------------- #
# Disposition / writeback responses (redacted; no raw_result exposed)
# --------------------------------------------------------------------------- #
class DispositionResponse(BaseModel):
    disposition: DispositionCommand
    writeback_status: WritebackStatus | None = None


class DispositionListResponse(BaseModel):
    event_id: str
    items: list[DispositionResponse] = Field(default_factory=list)


class WritebackResponse(BaseModel):
    """Redacted writeback view. ``raw_result`` is intentionally not exposed."""

    writeback_id: str
    disposition_id: str
    action_id: str
    status: WritebackStatus
    confirmation_evidence: ConfirmationEvidence | None = None
    evidence_tier: Literal["strong", "medium", "weak"] | None = None
    provider_code: str | None = None
    message_code: str | None = None
    target_results: list[TargetWritebackResult] = Field(default_factory=list)


class WritebackOperationResponse(BaseModel):
    writeback_id: str
    status: WritebackStatus
    message: str = ""


# --------------------------------------------------------------------------- #
# Platform responses
# --------------------------------------------------------------------------- #
class ExecutionJobResponse(BaseModel):
    job_id: str
    event_id: str
    action_id: str
    status: str
    attempt: int = 0
    target_results: list[dict[str, Any]] = Field(default_factory=list)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    event_id: str | None = None


class ToolMetaItem(BaseModel):
    tool_name: str
    tool_category: str
    side_effect_level: str
    idempotency: bool
    async_mode: bool
    rollback_supported: bool


class ToolsResponse(BaseModel):
    items: list[ToolMetaItem] = Field(default_factory=list)


class KnowledgeResponse(PageMeta):
    items: list[dict[str, Any]] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total_events: int = 0
    open_events: int = 0
    closed_events: int = 0
    pending_approvals: int = 0
    pending_writebacks: int = 0
    external_unsynced_events: int = 0


# --------------------------------------------------------------------------- #
# Example builders for placeholder responses (schema-valid static data)
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


EXAMPLE_EVENT_ID = "evt-20260101-0a1b2c3d"
EXAMPLE_CLOSED_EVENT_ID = "evt-20260101-cl05ed00"


def example_source_reference() -> SourceReference:
    from app.models.enums import SourceObjectKind

    return SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="t1",
        connector_id="conn-mock-1",
        source_object_id="INC-1001",
        ingested_at=_now(),
    )


def example_security_event(event_id: str = EXAMPLE_EVENT_ID) -> SecurityEvent:
    return SecurityEvent(
        event_id=event_id,
        event_type=EventType.INSIDER_THREAT,
        title="Example insider threat event",
        description="Static placeholder event.",
        status=EventStatus.ANALYZING,
        severity=Severity.HIGH,
        risk_score=72,
        confidence=0.8,
        final_verdict=FinalVerdict.NONE,
        creation_source_ref=example_source_reference(),
        disposition_policy=DispositionPolicy.REQUIRED,
        occurred_at=_now(),
        created_at=_now(),
        updated_at=_now(),
    )


def example_event_list_item(event_id: str = EXAMPLE_EVENT_ID) -> EventListItem:
    return EventListItem(
        event_id=event_id,
        event_type=EventType.INSIDER_THREAT,
        title="Example insider threat event",
        status=EventStatus.ANALYZING,
        severity=Severity.HIGH,
        risk_score=72,
        final_verdict=FinalVerdict.NONE,
        writeback_required=True,
        writeback_readiness=WritebackReadiness.CAPABILITY_UNKNOWN,
        writeback_overall_status=None,
        pending_writeback_count=0,
        created_at=_now(),
        updated_at=_now(),
        occurred_at=_now(),
    )


def example_action() -> Action:
    from app.models.enums import ActionCategory, ActionLevel

    return Action(
        action_id="act-0a1b2c3d",
        event_id=EXAMPLE_EVENT_ID,
        plan_revision=1,
        action_fingerprint="fp-example",
        action_category=ActionCategory.SYSTEM,
        action_name="Generate investigation report",
        tool_name="generate_report",
        action_level=ActionLevel.L0,
        reason="Static placeholder action.",
    )


def example_disposition_command() -> DispositionCommand:
    return DispositionCommand(
        disposition_id="disp-0a1b2c3d",
        action_id="act-0a1b2c3d",
        closure_cycle=1,
        intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
        source_locator=SourceObjectLocator(
            source_product="mock_xdr",
            source_tenant_id="t1",
            connector_id="conn-mock-1",
            source_kind=example_source_reference().source_kind,
            source_object_id="INC-1001",
        ),
        operation_code="set_event_disposition",
        operation_params=SetEventDispositionParams(target_disposition=SourceDisposition.CONTAINED),
        operator_id="system",
        idempotency_key="idem-example",
        execution_owner=ExecutionOwner.XDR_MANAGED,
    )


def example_report(event_id: str = EXAMPLE_EVENT_ID) -> InvestigationReport:
    from app.models.ids import report_id_for_event

    return InvestigationReport(
        report_id=report_id_for_event(event_id),
        event_id=event_id,
        title="Investigation report",
        summary="Static placeholder report.",
        sections=[ReportSection(key="overview", title="Overview", content="…")],
        final_verdict=FinalVerdict.NONE,
        risk_score=72,
        severity=Severity.HIGH,
        generated_at=_now(),
    )
