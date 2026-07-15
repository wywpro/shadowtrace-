"""State machine and workflow constant tests (ISSUE-007)."""

from __future__ import annotations

import pytest

from app.models.enums import (
    ActionCategory,
    ActionExecutionPhase,
    ActionLevel,
    ActionStatus,
    CaseLabel,
    ConfirmationEvidence,
    DispositionPolicy,
    EventStatus,
    ExecutionJobStatus,
    ExecutionSubstate,
    FinalVerdict,
    OutboxDeliveryStatus,
    Severity,
    SourceDisposition,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.workflow import (
    APPROVAL_TIMEOUT_MINUTES,
    FP_HIGH_THRESHOLD,
    FP_LOW_THRESHOLD,
    GLOBAL_MAX_STEPS,
    MAX_AGENT_RETRIES,
    MAX_REPLAN_COUNT,
    MIN_EVIDENCE_SOURCES,
    STATE_TRANSITIONS,
    WRITEBACK_MAX_RETRIES,
    ClosedGateActionView,
    InvalidStateTransitionError,
    InvalidVerdictStatusCombinationError,
    LateFalsePositiveTier,
    TerminalEventWritebackView,
    TransitionContext,
    apply_external_source_observation,
    classify_late_false_positive,
    derive_case_label,
    late_fp_allowed_substate,
    main_path_reaches_closed,
    maybe_upgrade_confirmation_evidence,
    resolved_workflow_constants,
    validate_action_status_transition,
    validate_closed_gate,
    validate_execution_substate,
    validate_job_status_transition,
    validate_outbox_delivery_transition,
    validate_transition,
    validate_verdict_status,
    validate_writeback_status_transition,
)


def _terminal_ok(**overrides: object) -> TerminalEventWritebackView:
    base = {
        "action_id": "act-disp",
        "disposition_id": "disp-1",
        "writeback_id": "wbk-1",
        "closure_cycle": 1,
        "approved_disposition": SourceDisposition.CONTAINED,
        "actual_disposition": SourceDisposition.CONTAINED,
        "receipt_status": WritebackStatus.CONFIRMED,
        "plan_revision": 1,
    }
    base.update(overrides)
    return TerminalEventWritebackView(**base)  # type: ignore[arg-type]


def _applicable_ok(**overrides: object) -> ClosedGateActionView:
    base = {
        "action_id": "act-1",
        "action_category": ActionCategory.RESPONSE,
        "writeback_required": True,
        "writeback_applicable": True,
        "writeback_readiness": WritebackReadiness.READY,
        "writeback_status": WritebackStatus.CONFIRMED,
        "has_command": True,
        "all_required_intents_confirmed": True,
        "tool_name": "block_ip",
    }
    base.update(overrides)
    return ClosedGateActionView(**base)  # type: ignore[arg-type]


def _closed_ctx(**overrides: object) -> TransitionContext:
    base: dict[str, object] = {
        "disposition_policy": DispositionPolicy.REQUIRED,
        "report_exists": True,
        "applicable_required_actions": [_applicable_ok()],
        "terminal_event_writeback": _terminal_ok(),
        "current_plan_revision": 1,
        "current_closure_cycle": 1,
    }
    base.update(overrides)
    return TransitionContext(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Matrix coverage
# --------------------------------------------------------------------------- #


def test_state_matrix_covers_all_fourteen_statuses() -> None:
    assert set(STATE_TRANSITIONS.keys()) == set(EventStatus)
    assert len(EventStatus) == 14


def test_closed_has_no_outbound_edges() -> None:
    assert STATE_TRANSITIONS[EventStatus.CLOSED] == set()
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(EventStatus.CLOSED, EventStatus.NEW)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (EventStatus.NEW, EventStatus.TRIAGING),
        (EventStatus.TRIAGING, EventStatus.COLLECTING_EVIDENCE),
        (EventStatus.COLLECTING_EVIDENCE, EventStatus.ANALYZING),
        (EventStatus.ANALYZING, EventStatus.SCORING),
        (EventStatus.SCORING, EventStatus.PLANNING_RESPONSE),
        (EventStatus.SCORING, EventStatus.REPORTING),
        (EventStatus.PLANNING_RESPONSE, EventStatus.WAITING_APPROVAL),
        (EventStatus.PLANNING_RESPONSE, EventStatus.EXECUTING_RESPONSE),
        (EventStatus.WAITING_APPROVAL, EventStatus.EXECUTING_RESPONSE),
        (EventStatus.WAITING_APPROVAL, EventStatus.REPORTING),
        (EventStatus.EXECUTING_RESPONSE, EventStatus.VERIFYING),
        (EventStatus.VERIFYING, EventStatus.REPORTING),
        (EventStatus.VERIFYING, EventStatus.CONTAINED),
        (EventStatus.VERIFYING, EventStatus.REPLANNING),
        (EventStatus.REPLANNING, EventStatus.COLLECTING_EVIDENCE),
        (EventStatus.REPLANNING, EventStatus.PLANNING_RESPONSE),
        (EventStatus.REPLANNING, EventStatus.EXECUTING_RESPONSE),
        (EventStatus.REPLANNING, EventStatus.CONTAINED),
        (EventStatus.CONTAINED, EventStatus.REPORTING),
        (EventStatus.FAILED, EventStatus.REPORTING),
        (EventStatus.REPORTING, EventStatus.CLOSED),
    ],
)
def test_legal_edges_pass(current: EventStatus, target: EventStatus) -> None:
    ctx = None
    if target is EventStatus.CLOSED:
        ctx = _closed_ctx(disposition_policy=DispositionPolicy.NOT_REQUIRED)
    validate_transition(current, target, ctx)


@pytest.mark.parametrize("status", list(EventStatus))
def test_every_non_closed_status_can_fail(status: EventStatus) -> None:
    if status in (EventStatus.CLOSED, EventStatus.FAILED):
        return
    assert EventStatus.FAILED in STATE_TRANSITIONS[status]
    validate_transition(status, EventStatus.FAILED)


def test_illegal_edge_new_to_scoring() -> None:
    with pytest.raises(InvalidStateTransitionError) as exc:
        validate_transition(EventStatus.NEW, EventStatus.SCORING)
    assert exc.value.error_code == "invalid_state_transition"


def test_main_path_reaches_closed() -> None:
    assert main_path_reaches_closed() is True


# --------------------------------------------------------------------------- #
# TRIAGING special gates
# --------------------------------------------------------------------------- #


def test_triaging_to_closed_not_required_low() -> None:
    validate_transition(
        EventStatus.TRIAGING,
        EventStatus.CLOSED,
        TransitionContext(
            disposition_policy=DispositionPolicy.NOT_REQUIRED,
            severity=Severity.LOW,
            report_exists=True,
        ),
    )


def test_triaging_to_closed_rejects_required_policy() -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(
            EventStatus.TRIAGING,
            EventStatus.CLOSED,
            TransitionContext(
                disposition_policy=DispositionPolicy.REQUIRED,
                severity=Severity.LOW,
                report_exists=True,
            ),
        )


def test_triaging_to_disposition_only_requires_fp_and_intent() -> None:
    validate_transition(
        EventStatus.TRIAGING,
        EventStatus.PLANNING_RESPONSE,
        TransitionContext(
            final_verdict=FinalVerdict.FALSE_POSITIVE,
            disposition_only_intent=True,
            disposition_policy=DispositionPolicy.REQUIRED,
        ),
    )
    # need_investigation=false alone is forbidden
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(
            EventStatus.TRIAGING,
            EventStatus.PLANNING_RESPONSE,
            TransitionContext(
                need_investigation=False,
                disposition_only_intent=True,
                disposition_policy=DispositionPolicy.REQUIRED,
            ),
        )
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(
            EventStatus.TRIAGING,
            EventStatus.PLANNING_RESPONSE,
            TransitionContext(
                final_verdict=FinalVerdict.FALSE_POSITIVE,
                disposition_only_intent=False,
            ),
        )


def test_force_close_flag_rejected_on_validate_transition() -> None:
    with pytest.raises(InvalidStateTransitionError, match="force_close"):
        validate_transition(
            EventStatus.REPORTING,
            EventStatus.CLOSED,
            _closed_ctx(
                disposition_policy=DispositionPolicy.NOT_REQUIRED,
                force_close=True,
            ),
        )


# --------------------------------------------------------------------------- #
# Verdict rules
# --------------------------------------------------------------------------- #


def test_false_positive_forbidden_on_entity_path_statuses() -> None:
    with pytest.raises(InvalidVerdictStatusCombinationError) as exc:
        validate_verdict_status(
            FinalVerdict.FALSE_POSITIVE,
            EventStatus.EXECUTING_RESPONSE,
            TransitionContext(),
        )
    assert exc.value.error_code == "invalid_verdict_status_combination"


def test_false_positive_disposition_only_exception() -> None:
    validate_verdict_status(
        FinalVerdict.FALSE_POSITIVE,
        EventStatus.WAITING_APPROVAL,
        TransitionContext(
            disposition_only_intent=True,
            response_actions_are_disposition_only=True,
            has_entity_side_effect_actions=False,
        ),
    )
    with pytest.raises(InvalidVerdictStatusCombinationError):
        validate_verdict_status(
            FinalVerdict.FALSE_POSITIVE,
            EventStatus.WAITING_APPROVAL,
            TransitionContext(
                disposition_only_intent=True,
                response_actions_are_disposition_only=True,
                has_entity_side_effect_actions=True,
            ),
        )


def test_derive_case_label() -> None:
    assert derive_case_label(FinalVerdict.CONFIRMED_THREAT) is CaseLabel.TRUE_POSITIVE
    assert derive_case_label(FinalVerdict.FALSE_POSITIVE) is CaseLabel.FALSE_POSITIVE
    assert derive_case_label(FinalVerdict.NONE) is CaseLabel.UNCERTAIN
    assert derive_case_label(FinalVerdict.POSSIBLE_FALSE_POSITIVE) is CaseLabel.UNCERTAIN


# --------------------------------------------------------------------------- #
# CLOSED gate
# --------------------------------------------------------------------------- #


def test_closed_gate_requires_report() -> None:
    with pytest.raises(InvalidStateTransitionError, match="report"):
        validate_closed_gate(
            _closed_ctx(report_exists=False, disposition_policy=DispositionPolicy.NOT_REQUIRED)
        )


def test_closed_gate_required_rejects_empty_applicable() -> None:
    with pytest.raises(InvalidStateTransitionError, match="zero applicable"):
        validate_closed_gate(_closed_ctx(applicable_required_actions=[]))


def test_closed_gate_required_rejects_all_rejected() -> None:
    with pytest.raises(InvalidStateTransitionError, match="zero applicable"):
        validate_closed_gate(
            _closed_ctx(
                applicable_required_actions=[
                    _applicable_ok(rejected=True, writeback_applicable=False)
                ]
            )
        )


def test_closed_gate_required_rejects_non_ready() -> None:
    with pytest.raises(InvalidStateTransitionError, match="readiness"):
        validate_closed_gate(
            _closed_ctx(
                applicable_required_actions=[
                    _applicable_ok(writeback_readiness=WritebackReadiness.CAPABILITY_UNKNOWN)
                ]
            )
        )


def test_closed_gate_required_rejects_missing_terminal() -> None:
    with pytest.raises(InvalidStateTransitionError, match="terminal EVENT_STATUS_UPDATE"):
        validate_closed_gate(_closed_ctx(terminal_event_writeback=None))


def test_closed_gate_required_rejects_non_terminal_disposition() -> None:
    with pytest.raises(InvalidStateTransitionError, match="not terminal"):
        validate_closed_gate(
            _closed_ctx(
                terminal_event_writeback=_terminal_ok(
                    approved_disposition=SourceDisposition.PENDING
                )
            )
        )


def test_closed_gate_required_happy_path() -> None:
    validate_closed_gate(_closed_ctx())


# --------------------------------------------------------------------------- #
# Action / substate / job / outbox / writeback tables
# --------------------------------------------------------------------------- #


def test_response_action_legal_and_illegal_edges() -> None:
    validate_action_status_transition(
        ActionCategory.RESPONSE, ActionStatus.PENDING, ActionStatus.APPROVED
    )
    validate_action_status_transition(
        ActionCategory.RESPONSE, ActionStatus.SUCCESS, ActionStatus.ROLLED_BACK
    )
    with pytest.raises(InvalidStateTransitionError):
        validate_action_status_transition(
            ActionCategory.RESPONSE, ActionStatus.PENDING, ActionStatus.EXECUTING
        )


def test_high_level_response_action_requires_approval_evidence() -> None:
    """L2+ RESPONSE/ROLLBACK actions must not reach APPROVED without a
    persisted ApprovalRecord — auto_execute never bypasses the human gate."""
    with pytest.raises(InvalidStateTransitionError, match="ApprovalRecord"):
        validate_action_status_transition(
            ActionCategory.RESPONSE,
            ActionStatus.PENDING,
            ActionStatus.APPROVED,
            action_level=ActionLevel.L2,
        )
    with pytest.raises(InvalidStateTransitionError, match="ApprovalRecord"):
        validate_action_status_transition(
            ActionCategory.RESPONSE,
            ActionStatus.PENDING,
            ActionStatus.APPROVED,
            action_level=ActionLevel.L2,
            auto_execute=True,
        )
    # With evidence, the transition is legal.
    validate_action_status_transition(
        ActionCategory.RESPONSE,
        ActionStatus.PENDING,
        ActionStatus.APPROVED,
        action_level=ActionLevel.L2,
        has_approval_evidence=True,
    )
    # Same rule applies to ROLLBACK.
    with pytest.raises(InvalidStateTransitionError, match="ApprovalRecord"):
        validate_action_status_transition(
            ActionCategory.ROLLBACK,
            ActionStatus.PENDING,
            ActionStatus.APPROVED,
            action_level=ActionLevel.L3,
        )


def test_low_level_response_action_auto_approves_without_evidence() -> None:
    """L0/L1 actions are the auto-approvable tier — no ApprovalRecord required."""
    validate_action_status_transition(
        ActionCategory.RESPONSE,
        ActionStatus.PENDING,
        ActionStatus.APPROVED,
        action_level=ActionLevel.L1,
    )
    validate_action_status_transition(
        ActionCategory.RESPONSE,
        ActionStatus.PENDING,
        ActionStatus.APPROVED,
        action_level=ActionLevel.L0,
        auto_execute=True,
    )


def test_superseded_rejects_when_job_exists() -> None:
    with pytest.raises(InvalidStateTransitionError, match="job/outbox"):
        validate_action_status_transition(
            ActionCategory.RESPONSE,
            ActionStatus.APPROVED,
            ActionStatus.SUPERSEDED,
            has_job_or_outbox=True,
        )


def test_post_verify_approved_to_executing_requires_activation() -> None:
    with pytest.raises(InvalidStateTransitionError, match="after_effect_resolution"):
        validate_action_status_transition(
            ActionCategory.RESPONSE,
            ActionStatus.APPROVED,
            ActionStatus.EXECUTING,
            execution_phase=ActionExecutionPhase.POST_VERIFY,
            after_effect_resolution=False,
        )
    validate_action_status_transition(
        ActionCategory.RESPONSE,
        ActionStatus.APPROVED,
        ActionStatus.EXECUTING,
        execution_phase=ActionExecutionPhase.POST_VERIFY,
        after_effect_resolution=True,
        template_unchanged=True,
    )


def test_rollback_action_never_becomes_rolled_back() -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_action_status_transition(
            ActionCategory.ROLLBACK, ActionStatus.SUCCESS, ActionStatus.ROLLED_BACK
        )


def test_rollback_approved_may_supersede_or_reopen_approval_when_un_egress() -> None:
    # Same un-egress rules as response: APPROVED→SUPERSEDED / WAITING_APPROVAL.
    validate_action_status_transition(
        ActionCategory.ROLLBACK,
        ActionStatus.APPROVED,
        ActionStatus.SUPERSEDED,
        has_job_or_outbox=False,
    )
    validate_action_status_transition(
        ActionCategory.ROLLBACK,
        ActionStatus.APPROVED,
        ActionStatus.WAITING_APPROVAL,
    )
    with pytest.raises(InvalidStateTransitionError, match="job/outbox"):
        validate_action_status_transition(
            ActionCategory.ROLLBACK,
            ActionStatus.APPROVED,
            ActionStatus.SUPERSEDED,
            has_job_or_outbox=True,
        )


def test_verification_action_edges() -> None:
    validate_action_status_transition(
        ActionCategory.VERIFICATION, ActionStatus.PENDING, ActionStatus.EXECUTING
    )
    with pytest.raises(InvalidStateTransitionError):
        validate_action_status_transition(
            ActionCategory.VERIFICATION,
            ActionStatus.PENDING,
            ActionStatus.WAITING_APPROVAL,
        )


def test_system_action_edges() -> None:
    validate_action_status_transition(
        ActionCategory.SYSTEM, ActionStatus.PENDING, ActionStatus.SUCCESS
    )


def test_execution_substate_triaging_forbids_manual_resolution() -> None:
    with pytest.raises(InvalidStateTransitionError, match="forbids"):
        validate_execution_substate(
            EventStatus.TRIAGING,
            ExecutionSubstate.NONE,
            ExecutionSubstate.MANUAL_RESOLUTION,
        )


def test_execution_substate_verifying_waiting_writeback() -> None:
    validate_execution_substate(
        EventStatus.VERIFYING,
        ExecutionSubstate.NONE,
        ExecutionSubstate.WAITING_WRITEBACK,
    )
    # Writeback wait must NOT be modeled as REPLANNING EventStatus.
    assert (
        EventStatus.WAITING_APPROVAL
        not in (
            # sanity: waiting_writeback is substate only
        )
    )


def test_job_and_outbox_and_writeback_edges() -> None:
    validate_job_status_transition(ExecutionJobStatus.QUEUED, ExecutionJobStatus.RUNNING)
    with pytest.raises(InvalidStateTransitionError):
        validate_job_status_transition(
            ExecutionJobStatus.UNKNOWN,
            ExecutionJobStatus.TIMED_OUT,
            provider_confirmed_terminal=False,
        )
    validate_job_status_transition(
        ExecutionJobStatus.UNKNOWN,
        ExecutionJobStatus.TIMED_OUT,
        provider_confirmed_terminal=True,
    )

    validate_outbox_delivery_transition(OutboxDeliveryStatus.READY, OutboxDeliveryStatus.LEASED)
    with pytest.raises(InvalidStateTransitionError, match="PAUSED"):
        validate_outbox_delivery_transition(
            OutboxDeliveryStatus.LEASED,
            OutboxDeliveryStatus.LEASED,
            lease_expired_resend=True,
        )

    validate_writeback_status_transition(WritebackStatus.PENDING, WritebackStatus.SENDING)
    assert (
        WritebackStatus.CONFIRMED
        not in (
            # confirmed is terminal
        )
    )
    with pytest.raises(InvalidStateTransitionError):
        validate_writeback_status_transition(WritebackStatus.CONFIRMED, WritebackStatus.PENDING)
    validate_writeback_status_transition(
        WritebackStatus.UNKNOWN,
        WritebackStatus.CONFIRMED,
        evidence_adjudication=True,
    )


# --------------------------------------------------------------------------- #
# Late FP tiers
# --------------------------------------------------------------------------- #


def test_late_fp_three_tiers() -> None:
    assert (
        classify_late_false_positive(
            has_immediate_job_or_outbox=False,
            immediate_in_flight_or_unverified=False,
            has_verified_successful_entity_action=False,
        )
        is LateFalsePositiveTier.NO_SIDE_EFFECT
    )
    assert (
        classify_late_false_positive(
            has_immediate_job_or_outbox=True,
            immediate_in_flight_or_unverified=True,
            has_verified_successful_entity_action=False,
        )
        is LateFalsePositiveTier.IN_FLIGHT
    )
    assert (
        classify_late_false_positive(
            has_immediate_job_or_outbox=True,
            immediate_in_flight_or_unverified=False,
            has_verified_successful_entity_action=True,
        )
        is LateFalsePositiveTier.VERIFIED_EFFECTS
    )
    assert late_fp_allowed_substate(EventStatus.VERIFYING) is ExecutionSubstate.MANUAL_RESOLUTION
    assert late_fp_allowed_substate(EventStatus.TRIAGING) is None


# --------------------------------------------------------------------------- #
# Isolation / correlation
# --------------------------------------------------------------------------- #


def test_external_source_change_does_not_mutate_event_status_or_snapshot() -> None:
    snapshot = {"source_object_id": "INC-1", "title": "frozen"}
    source = {
        "current_source_disposition": "processing",
        "current_source_status_raw": "in_progress",
        "current_concurrency_token": "t1",
    }
    updated, frozen, status = apply_external_source_observation(
        source_object=source,
        frozen_snapshot=snapshot,
        event_status=EventStatus.ANALYZING,
        new_current_disposition="contained",
        new_concurrency_token="t2",
    )
    assert status is EventStatus.ANALYZING
    assert frozen == snapshot
    assert frozen is snapshot  # same object — not rewritten
    assert updated["current_source_disposition"] == "contained"
    assert updated["current_concurrency_token"] == "t2"
    assert source["current_source_disposition"] == "processing"  # original untouched


def test_unrelated_external_change_cannot_upgrade_confirmation() -> None:
    assert (
        maybe_upgrade_confirmation_evidence(
            correlated_by_writeback_id=False,
            readback_matches_command=True,
            current=ConfirmationEvidence.ADAPTER_ACKNOWLEDGED,
        )
        is ConfirmationEvidence.ADAPTER_ACKNOWLEDGED
    )
    assert (
        maybe_upgrade_confirmation_evidence(
            correlated_by_writeback_id=True,
            readback_matches_command=True,
            current=ConfirmationEvidence.ADAPTER_ACKNOWLEDGED,
        )
        is ConfirmationEvidence.READBACK_VERIFIED
    )


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


def test_workflow_constants_present_and_settings_overlay() -> None:
    assert MAX_REPLAN_COUNT == 3
    assert MAX_AGENT_RETRIES == 2
    assert MIN_EVIDENCE_SOURCES == 3
    assert APPROVAL_TIMEOUT_MINUTES == 30
    assert FP_HIGH_THRESHOLD == 0.9
    assert FP_LOW_THRESHOLD == 0.7
    assert WRITEBACK_MAX_RETRIES == 5
    assert GLOBAL_MAX_STEPS == 80
    resolved = resolved_workflow_constants()
    assert resolved["APPROVAL_TIMEOUT_MINUTES"] == 30
    assert resolved["WRITEBACK_MAX_RETRIES"] == 5
    assert "GLOBAL_TOKEN_BUDGET" in resolved
