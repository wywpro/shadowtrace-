"""ORM models for the 18 core tables (ISSUE-003).

Column names and semantics mirror the ISSUE-002 Pydantic models one-to-one. JSON
container fields use JSONB. Internal investigation, action execution and external
disposition writeback are audited in separate tables so the three concerns never
share a status column.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# timezone-aware timestamp used across every table.
_TS = DateTime(timezone=True)


class SecurityEvent(Base):
    __tablename__ = "security_event"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String, default="new", nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String, default="low", nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    final_verdict: Mapped[str] = mapped_column(String, default="none", nullable=False)

    entities: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    creation_source_ref: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source_reference_snapshots: Mapped[list[Any]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    current_primary_source_record_id: Mapped[str | None] = mapped_column(String, nullable=True)
    disposition_source_ref: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    disposition_policy: Mapped[str] = mapped_column(String, default="not_required", nullable=False)

    raw_alert_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    raw_alert_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    source_type: Mapped[str | None] = mapped_column(String, nullable=True)

    occurred_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)

    replan_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    degraded_flags: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    external_unsynced: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    event_context_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # optimistic lock; atomically incremented on every controlled mutable update.
    row_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class SourceObject(Base):
    """Stores the SourceReference full field set plus mutable current_* state.

    Writeback delivery is serialized per source object via ``next_outbox_sequence``,
    allocated under a row lock with ``UPDATE ... RETURNING``.
    """

    __tablename__ = "source_object"
    __table_args__ = (
        UniqueConstraint(
            "source_product",
            "source_tenant_id",
            "connector_id",
            "source_kind",
            "source_object_id",
            name="uq_source_object_identity",
        ),
    )

    source_record_id: Mapped[str] = mapped_column(String, primary_key=True)

    # identity five-tuple + adapter-native type (not part of identity)
    source_product: Mapped[str] = mapped_column(String, nullable=False)
    source_tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    connector_id: Mapped[str] = mapped_column(
        String, ForeignKey("source_connector.connector_id"), nullable=False, index=True
    )
    source_kind: Mapped[str] = mapped_column(String, nullable=False)
    source_object_id: Mapped[str] = mapped_column(String, nullable=False)
    source_object_type: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_source_object_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # immutable investigation snapshot fields
    source_status_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    source_disposition: Mapped[str] = mapped_column(String, default="unknown", nullable=False)
    source_concurrency_token: Mapped[str | None] = mapped_column(String, nullable=True)
    source_updated_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    schema_version: Mapped[str] = mapped_column(String, default="1", nullable=False)
    ingested_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    raw_payload_hash: Mapped[str | None] = mapped_column(String, nullable=True)

    normalized: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # mutable current state (never overwrites the snapshot)
    current_source_status_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    current_source_disposition: Mapped[str] = mapped_column(
        String, default="unknown", nullable=False
    )
    current_concurrency_token: Mapped[str | None] = mapped_column(String, nullable=True)
    current_source_updated_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    current_state_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    source_sync_state: Mapped[str | None] = mapped_column(String, nullable=True)

    next_outbox_sequence: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SourceConnector(Base):
    __tablename__ = "source_connector"

    connector_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_product: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    device_type: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="unknown", nullable=False)
    read_endpoint: Mapped[str | None] = mapped_column(String, nullable=True)
    disposition_endpoint: Mapped[str | None] = mapped_column(String, nullable=True)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # NULL = not explicitly provisioned (live must fail closed; mock/file set a value).
    disposition_policy_default: Mapped[str | None] = mapped_column(String, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    watermark: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    schema_version: Mapped[str] = mapped_column(String, default="1", nullable=False)
    # only credential references are stored; the secret material never lands here.
    read_credential_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    disposition_credential_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    connector_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SourceCheckpoint(Base):
    """Durable ingestion progress isolated by connector and source object kind."""

    __tablename__ = "source_checkpoint"

    connector_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("source_connector.connector_id", ondelete="CASCADE"),
        primary_key=True,
    )
    object_kind: Mapped[str] = mapped_column(String, primary_key=True)
    stream_scope: Mapped[str] = mapped_column(String, primary_key=True, default="")
    schema_version: Mapped[str] = mapped_column(String, nullable=False)
    cursor: Mapped[str | None] = mapped_column(String, nullable=True)
    watermark: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String, default="unknown", nullable=False)
    degraded_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    row_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SourceEventLink(Base):
    """Links a source object to an internal event with a role + promotion status."""

    __tablename__ = "source_event_link"
    __table_args__ = (
        UniqueConstraint("source_record_id", "event_id", name="uq_source_event_link_pair"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_record_id: Mapped[str] = mapped_column(
        String, ForeignKey("source_object.source_record_id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String, ForeignKey("security_event.event_id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String, default="primary", nullable=False)
    promotion_status: Mapped[str] = mapped_column(String, default="none", nullable=False)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Evidence(Base):
    __tablename__ = "evidence"

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String, ForeignKey("security_event.event_id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    evidence_type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    timestamp: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    related_entities: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    source_ref: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    raw_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    mitre_technique: Mapped[str | None] = mapped_column(String, nullable=True)
    is_conflicting: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)


class Action(Base):
    __tablename__ = "action"
    __table_args__ = (
        UniqueConstraint("action_fingerprint", name="uq_action_action_fingerprint"),
        Index("ix_action_idempotency_key", "idempotency_key"),
    )

    action_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String, ForeignKey("security_event.event_id"), nullable=False, index=True
    )
    plan_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    action_fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    action_category: Mapped[str] = mapped_column(String, nullable=False)
    action_name: Mapped[str] = mapped_column(String, nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    action_level: Mapped[str] = mapped_column(String, nullable=False)
    execution_phase: Mapped[str] = mapped_column(String, default="immediate", nullable=False)
    activation_condition: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_operation_template_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_terminal_dispositions: Mapped[list[Any]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    target_type: Mapped[str | None] = mapped_column(String, nullable=True)
    target: Mapped[str | None] = mapped_column(String, nullable=True)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    auto_execute: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    impact_assessment: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    playbook_id: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_name: Mapped[str | None] = mapped_column(String, nullable=True)
    execution_owner: Mapped[str | None] = mapped_column(String, nullable=True)
    execution_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String, nullable=True)
    writeback_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    writeback_applicable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    writeback_readiness: Mapped[str] = mapped_column(String, default="not_required", nullable=False)
    writeback_block_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    writeback_status: Mapped[str | None] = mapped_column(String, nullable=True)
    disposition_source_ref: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    superseded_by_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    effect_verification_status: Mapped[str | None] = mapped_column(String, nullable=True)
    rollback_status: Mapped[str | None] = mapped_column(String, nullable=True)
    source_action_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ActionExecutionJob(Base):
    __tablename__ = "action_execution_job"
    __table_args__ = (
        Index("ix_action_execution_job_status", "status"),
        Index("ix_action_execution_job_idempotency_key", "idempotency_key"),
    )

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String, ForeignKey("security_event.event_id"), nullable=False, index=True
    )
    action_id: Mapped[str] = mapped_column(
        String, ForeignKey("action.action_id"), nullable=False, index=True
    )
    provider_name: Mapped[str] = mapped_column(String, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False)
    provider_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="queued", nullable=False)
    claimed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    poll_after_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    provider_code: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Provider raw result retained here (internal only).
    raw_result: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)


class ActionTargetResult(Base):
    __tablename__ = "action_target_result"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String, ForeignKey("action_execution_job.job_id"), nullable=False, index=True
    )
    canonical_target: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    code: Mapped[str | None] = mapped_column(String, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # Provider raw result retained here (internal only).
    raw_result: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)


class DispositionOutbox(Base):
    """Reliable writeback outbox; the source of truth for writeback delivery.

    ``command_payload`` is immutable after creation. Only one active (non
    -superseded) EVENT_STATUS_UPDATE head is allowed per
    ``(event_id, closure_cycle, intent_kind, logical_slot)`` via a partial
    unique index over rows where ``superseded_by_disposition_id IS NULL``.
    This is deliberately event-scoped, NOT action-scoped: two different
    Actions racing to submit the terminal disposition for the same event/
    cycle/slot must collide on this index rather than silently coexist as
    two "active" heads (ISSUE-093 §4).
    """

    __tablename__ = "disposition_outbox"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_disposition_outbox_idempotency_key"),
        UniqueConstraint(
            "source_record_id", "source_sequence", name="uq_disposition_outbox_source_sequence"
        ),
        Index(
            "uq_disposition_outbox_event_status_active_head",
            "event_id",
            "closure_cycle",
            "intent_kind",
            "logical_slot",
            unique=True,
            postgresql_where=text(
                "superseded_by_disposition_id IS NULL AND intent_kind = 'event_status_update'"
            ),
        ),
        Index("ix_disposition_outbox_delivery_status", "delivery_status"),
        Index("ix_disposition_outbox_latest_writeback_status", "latest_writeback_status"),
        Index("ix_disposition_outbox_next_retry_at", "next_retry_at"),
        Index("ix_disposition_outbox_lease_expires_at", "lease_expires_at"),
        Index("ix_disposition_outbox_disposition_id", "disposition_id"),
    )

    outbox_id: Mapped[str] = mapped_column(String, primary_key=True)
    writeback_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    disposition_id: Mapped[str] = mapped_column(String, nullable=False)
    action_id: Mapped[str] = mapped_column(
        String, ForeignKey("action.action_id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String, ForeignKey("security_event.event_id"), nullable=False, index=True
    )
    closure_cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    source_record_id: Mapped[str] = mapped_column(
        String, ForeignKey("source_object.source_record_id"), nullable=False, index=True
    )
    source_locator_hash: Mapped[str] = mapped_column(String, nullable=False)
    source_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    intent_kind: Mapped[str] = mapped_column(String, nullable=False)
    logical_slot: Mapped[str] = mapped_column(String, nullable=False)
    supersedes_disposition_id: Mapped[str | None] = mapped_column(String, nullable=True)
    superseded_by_disposition_id: Mapped[str | None] = mapped_column(String, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False)
    # immutable after creation
    command_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    command_payload_sha256: Mapped[str] = mapped_column(String, nullable=False)
    delivery_status: Mapped[str] = mapped_column(String, default="ready", nullable=False)
    latest_writeback_status: Mapped[str | None] = mapped_column(String, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_retry_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String, nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    last_error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    delivered_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)


class DispositionReceipt(Base):
    """XDR writeback receipt; append-only keyed by (writeback_id, sequence)."""

    __tablename__ = "disposition_receipt"

    writeback_id: Mapped[str] = mapped_column(String, primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    disposition_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    action_id: Mapped[str] = mapped_column(
        String, ForeignKey("action.action_id"), nullable=False, index=True
    )
    source_record_id: Mapped[str] = mapped_column(
        String, ForeignKey("source_object.source_record_id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    confirmation_evidence: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_record_id: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_code: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    target_results: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    raw_result: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Report(Base):
    __tablename__ = "report"

    report_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String, ForeignKey("security_event.event_id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    sections: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    final_verdict: Mapped[str] = mapped_column(String, default="none", nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    severity: Mapped[str] = mapped_column(String, default="low", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    generated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    generated_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        _TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class AgentTrace(Base):
    __tablename__ = "agent_trace"

    trace_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    input_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    output_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)


class EventAuditLog(Base):
    __tablename__ = "event_audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    from_status: Mapped[str | None] = mapped_column(String, nullable=True)
    to_status: Mapped[str | None] = mapped_column(String, nullable=True)
    operator: Mapped[str | None] = mapped_column(String, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)


class ToolCallLog(Base):
    __tablename__ = "tool_call_log"

    call_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    action_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    tool_category: Mapped[str] = mapped_column(String, nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class LLMCallLog(Base):
    __tablename__ = "llm_call_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    prompt_key: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fallback_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)


class DataQualityError(Base):
    """Ingestion/normalization quality issues; event_id nullable (pre-event errors)."""

    __tablename__ = "data_quality_error"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    error_category: Mapped[str] = mapped_column(String, nullable=False)
    field_name: Mapped[str | None] = mapped_column(String, nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)


class EventContextJournal(Base):
    """Append-only versioned journal of EventContext field values."""

    __tablename__ = "event_context_journal"
    __table_args__ = (
        UniqueConstraint(
            "event_id", "field_name", "version", name="uq_event_context_journal_field_version"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    field_name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now(), nullable=False)


class EventContextFieldVersion(Base):
    """Sole allocation source for context field versions; PK (event_id, field_name)."""

    __tablename__ = "event_context_field_version"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    field_name: Mapped[str] = mapped_column(String, primary_key=True)
    current_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
