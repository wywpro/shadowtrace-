"""EventContext field-set double-sided assertion (ISSUE-002 统一命名 13)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.context import EventContext
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    WritebackReadiness,
)
from app.models.security_event import EventSummary

# Canonical field set fixed by the ISSUE-002 spec (statement 13).
EXPECTED_CONTEXT_FIELDS = {
    "event",
    "source_snapshot",
    "source_sync_state",
    "triage_result",
    "false_positive_match",
    "evidence_output",
    "storyline",
    "graph_output",
    "rag_output",
    "risk_assessment",
    "execution_plan",
    "response_plan",
    "approval_records",
    "disposition_only_intent",
    "execution_substate",
    "execution_summary",
    "execution_jobs",
    "verification_result",
    "rollback_results",
    "impact_assessments",
    "report",
    "memory_output",
    "disposition_commands",
    "disposition_receipts",
    "writeback_summary",
    "state_history",
    "replan_count",
    "budget_usage",
    "guard_violations",
    "convergence_state",
    "quality_scores",
    "scratchpad",
    "degraded_flags",
    "triage_degraded",
    "graph_degraded",
    "storyline_degraded",
    "analysis_only_complete",
}


def test_event_context_field_set_matches_spec_both_directions() -> None:
    actual = set(EventContext.model_fields.keys())
    assert actual == EXPECTED_CONTEXT_FIELDS, {
        "missing": EXPECTED_CONTEXT_FIELDS - actual,
        "unexpected": actual - EXPECTED_CONTEXT_FIELDS,
    }


def _summary(event_id: str = "evt-1") -> EventSummary:
    return EventSummary(
        event_id=event_id,
        event_type=EventType.INSIDER_THREAT,
        title="t",
        status=EventStatus.NEW,
        severity=Severity.LOW,
        risk_score=0,
        final_verdict=FinalVerdict.NONE,
        writeback_required=False,
        writeback_readiness=WritebackReadiness.NOT_REQUIRED,
        disposition_policy=DispositionPolicy.NOT_REQUIRED,
    )


def test_event_context_event_field_is_event_summary_typed() -> None:
    """ISSUE-094 §2: ``event`` is EventSummary, never the full SecurityEvent."""
    annotation = EventContext.model_fields["event"].annotation
    assert annotation == (EventSummary | None)


def test_event_context_accepts_event_summary() -> None:
    ctx = EventContext(event=_summary())
    assert isinstance(ctx.event, EventSummary)
    assert ctx.event.event_id == "evt-1"


def test_event_context_none_event_ok() -> None:
    ctx = EventContext()
    assert ctx.event is None


def test_event_context_rejects_security_event_shaped_payload() -> None:
    """A raw SecurityEvent-shaped dict (missing writeback_* fields) must fail."""
    with pytest.raises(ValidationError):
        EventContext(
            event={
                "event_id": "evt-1",
                "event_type": "insider_threat",
                "title": "t",
                "creation_source_ref": {
                    "source_kind": "incident",
                    "source_product": "mock_xdr",
                    "source_tenant_id": "t1",
                    "connector_id": "conn-1",
                    "source_object_id": "INC-1",
                },
            }
        )
