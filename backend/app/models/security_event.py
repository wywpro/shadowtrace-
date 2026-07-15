"""SecurityEvent: ShadowTrace's single internal investigation model (intro §4.3.1).

It is NOT the XDR incident and does not cover external alert/asset/ticket/plan
state. Source object IDs/statuses must never be written into the internal
``event_id``/``status``. Every successful mutable-field update bumps ``row_version``
by 1 in the same transaction (optimistic lock).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.disposition import SourceObjectLocator
from app.models.entities import EntitySet
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.source import SourceReference


class SecurityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType
    title: str
    description: str = ""
    status: EventStatus = EventStatus.NEW
    severity: Severity = Severity.LOW
    risk_score: int = Field(default=0, ge=0, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_verdict: FinalVerdict = FinalVerdict.NONE
    entities: EntitySet = Field(default_factory=EntitySet)

    # Immutable first snapshot; append-only snapshot list (existing elems unchanged).
    creation_source_ref: SourceReference
    source_reference_snapshots: list[SourceReference] = Field(default_factory=list)
    current_primary_source_record_id: str | None = None
    # Current disposition locator (nullable, single choice; frozen when action made).
    disposition_source_ref: SourceObjectLocator | None = None
    disposition_policy: DispositionPolicy = DispositionPolicy.NOT_REQUIRED

    raw_alert_ids: list[str] = Field(default_factory=list)
    raw_alert_snapshot: dict[str, Any] | None = None  # file fallback only
    source_type: str | None = None

    occurred_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    closed_at: datetime | None = None

    replan_count: int = 0
    degraded_flags: list[str] = Field(default_factory=list)
    escalated: bool = False
    external_unsynced: bool = False
    event_context_snapshot: dict[str, Any] | None = None
    row_version: int = Field(default=1, ge=1)


class EventListItem(BaseModel):
    """Redacted list-view projection of a :class:`SecurityEvent` (ISSUE-004).

    Lives alongside ``SecurityEvent`` (not in the API schemas module) so that
    ``EventContext.event`` (ISSUE-094 §2) can be typed as ``EventSummary``
    without the models layer depending on the API layer.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType
    title: str
    # ``status`` is always the local EventStatus (never an external status).
    status: EventStatus
    severity: Severity
    risk_score: int
    final_verdict: FinalVerdict
    writeback_required: bool
    writeback_readiness: WritebackReadiness
    # null when no writeback command exists; readiness distinguishes NOT_REQUIRED
    # from blocked.
    writeback_overall_status: WritebackStatus | None = None
    pending_writeback_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    occurred_at: datetime | None = None


class EventSummary(EventListItem):
    """The authoritative shape of ``EventContext.event`` (ISSUE-094 §2)."""

    disposition_policy: DispositionPolicy
    external_unsynced: bool = False
    escalated: bool = False
