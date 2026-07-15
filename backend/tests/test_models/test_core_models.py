"""Core model positive/negative tests (ISSUE-002 acceptance 1, 5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.action import Action
from app.models.context import EventContext
from app.models.entities import AccountEntity, EntitySet, IPEntity
from app.models.enums import (
    ActionCategory,
    ActionExecutionPhase,
    ActionLevel,
    DispositionPolicy,
    EventType,
    ExecutionOwner,
    SourceObjectKind,
    WritebackReadiness,
)
from app.models.evidence import Evidence, EvidenceSource
from app.models.security_event import EventSummary, SecurityEvent
from app.models.source import SourceReference


def _ref() -> SourceReference:
    return SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="t1",
        connector_id="conn-1",
        source_object_id="INC-1",
    )


def _event(**overrides: object) -> SecurityEvent:
    base = {
        "event_id": "evt-20260712-deadbeef",
        "event_type": EventType.INSIDER_THREAT,
        "title": "demo",
        "creation_source_ref": _ref(),
    }
    base.update(overrides)
    return SecurityEvent(**base)  # type: ignore[arg-type]


def test_security_event_minimal_ok() -> None:
    evt = _event()
    assert evt.row_version == 1
    assert evt.status.value == "new"


def test_security_event_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        _event(not_a_field=123)


def test_security_event_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValidationError):
        _event(risk_score=150)
    with pytest.raises(ValidationError):
        _event(confidence=2.0)


def test_reference_identity_five_tuple_excludes_type_and_token() -> None:
    ref = _ref()
    assert ref.identity == ("mock_xdr", "t1", "conn-1", "incident", "INC-1")


def test_entity_set_defaults_and_scope() -> None:
    es = EntitySet(
        accounts=[AccountEntity(entity_id="a1", username="svc")],
        ips=[IPEntity(entity_id="ip1", address="203.0.113.9", scope="external")],
    )
    assert es.accounts[0].entity_type == "account"
    assert es.ips[0].scope == "external"
    assert es.hosts == []


def test_evidence_confidence_bounds() -> None:
    ev = Evidence(
        evidence_id="evd-1",
        event_id="evt-1",
        source=EvidenceSource.IDENTITY,
        evidence_type="login",
        description="x",
        confidence=0.5,
    )
    assert ev.is_conflicting is False
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evd-2",
            event_id="evt-1",
            source=EvidenceSource.IDENTITY,
            evidence_type="login",
            description="x",
            confidence=1.5,
        )


def _action(**overrides: object) -> Action:
    base = {
        "action_id": "act-1",
        "event_id": "evt-1",
        "plan_revision": 1,
        "action_fingerprint": "fp",
        "action_category": ActionCategory.RESPONSE,
        "action_name": "block ip",
        "tool_name": "block_ip",
        "action_level": ActionLevel.L2,
        "execution_owner": ExecutionOwner.DIRECT_TOOL,
    }
    base.update(overrides)
    return Action(**base)  # type: ignore[arg-type]


def test_response_action_requires_single_owner() -> None:
    with pytest.raises(ValidationError):
        _action(execution_owner=None)


def test_system_action_forbids_owner_and_writeback() -> None:
    with pytest.raises(ValidationError):
        _action(
            action_category=ActionCategory.SYSTEM,
            tool_name="generate_report",
            execution_owner=ExecutionOwner.XDR_MANAGED,
        )
    ok = _action(
        action_category=ActionCategory.SYSTEM,
        tool_name="generate_report",
        execution_owner=None,
    )
    assert ok.writeback_required is False


def test_terminal_disposition_action_must_be_post_verify() -> None:
    # Wrong phase for the terminal tool -> invalid.
    with pytest.raises(ValidationError):
        _action(
            tool_name="update_source_event_disposition",
            execution_owner=ExecutionOwner.XDR_MANAGED,
            execution_phase=ActionExecutionPhase.IMMEDIATE,
        )
    ok = _action(
        tool_name="update_source_event_disposition",
        execution_owner=ExecutionOwner.XDR_MANAGED,
        execution_phase=ActionExecutionPhase.POST_VERIFY,
        activation_condition="after_effect_resolution",
    )
    assert ok.execution_phase is ActionExecutionPhase.POST_VERIFY


def test_non_terminal_action_cannot_be_post_verify() -> None:
    with pytest.raises(ValidationError):
        _action(execution_phase=ActionExecutionPhase.POST_VERIFY)


def test_required_writeback_stays_required_when_capability_blocked() -> None:
    # Business required must not be reverse-driven false; readiness carries block.
    act = _action(
        writeback_required=True,
        writeback_applicable=True,
        writeback_readiness=WritebackReadiness.CAPABILITY_UNKNOWN,
        writeback_block_reason="writeback_unsupported",
    )
    assert act.writeback_required is True
    assert act.writeback_readiness is WritebackReadiness.CAPABILITY_UNKNOWN


def test_writeback_not_required_forbids_applicable() -> None:
    """The literal impossible combo called out in ISSUE-093 §3."""
    with pytest.raises(ValidationError, match="writeback_applicable"):
        _action(
            writeback_required=False,
            writeback_applicable=True,
            writeback_readiness=WritebackReadiness.READY,
        )


def test_writeback_not_required_forbids_non_not_required_readiness() -> None:
    with pytest.raises(ValidationError, match="NOT_REQUIRED"):
        _action(
            writeback_required=False,
            writeback_readiness=WritebackReadiness.CAPABILITY_UNKNOWN,
        )


def test_writeback_not_required_forbids_status() -> None:
    from app.models.enums import WritebackStatus

    with pytest.raises(ValidationError, match="writeback_status"):
        _action(
            writeback_required=False,
            writeback_status=WritebackStatus.CONFIRMED,
        )


def test_writeback_required_not_applicable_requires_not_required_readiness() -> None:
    with pytest.raises(ValidationError, match="writeback_applicable=false"):
        _action(
            writeback_required=True,
            writeback_applicable=False,
            writeback_readiness=WritebackReadiness.READY,
        )
    # Valid: obligation exists at the event level but not on this action.
    ok = _action(
        writeback_required=True,
        writeback_applicable=False,
        writeback_readiness=WritebackReadiness.NOT_REQUIRED,
    )
    assert ok.writeback_applicable is False


def test_writeback_required_applicable_forbids_not_required_readiness() -> None:
    with pytest.raises(ValidationError, match="forbid"):
        _action(
            writeback_required=True,
            writeback_applicable=True,
            writeback_readiness=WritebackReadiness.NOT_REQUIRED,
        )


def test_writeback_blocked_readiness_forbids_status() -> None:
    from app.models.enums import WritebackStatus

    with pytest.raises(ValidationError, match="writeback_status must be null"):
        _action(
            writeback_required=True,
            writeback_applicable=True,
            writeback_readiness=WritebackReadiness.CAPABILITY_UNSUPPORTED,
            writeback_status=WritebackStatus.CONFIRMED,
        )


def test_writeback_ready_allows_status() -> None:
    from app.models.enums import WritebackStatus

    act = _action(
        writeback_required=True,
        writeback_applicable=True,
        writeback_readiness=WritebackReadiness.READY,
        writeback_status=WritebackStatus.CONFIRMED,
    )
    assert act.writeback_status is WritebackStatus.CONFIRMED
    # READY with no attempt yet (status=None) is also legal.
    act2 = _action(
        writeback_required=True,
        writeback_applicable=True,
        writeback_readiness=WritebackReadiness.READY,
    )
    assert act2.writeback_status is None


def _summary_from_event(event: SecurityEvent) -> EventSummary:
    """Minimal EventSummary projection matching ``event_summary_from_security_event``."""
    return EventSummary(
        event_id=event.event_id,
        event_type=event.event_type,
        title=event.title,
        status=event.status,
        severity=event.severity,
        risk_score=event.risk_score,
        final_verdict=event.final_verdict,
        writeback_required=event.disposition_policy is DispositionPolicy.REQUIRED,
        writeback_readiness=WritebackReadiness.NOT_REQUIRED,
        disposition_policy=event.disposition_policy,
    )


def test_event_context_defaults() -> None:
    # ISSUE-094 §2: EventContext.event is EventSummary, never the full SecurityEvent.
    ctx = EventContext(event=_summary_from_event(_event()))
    assert ctx.execution_substate.value == "none"
    assert ctx.disposition_only_intent is False
    assert ctx.disposition_commands == []
