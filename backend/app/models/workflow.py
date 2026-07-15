"""Internal state machine, sub-state CAS tables, and workflow constants (ISSUE-007).

External XDR / source dispositions must never overwrite ``EventStatus``.
``DispositionReceipt`` only records writeback facts; ``source_object.current_*``
is updated solely by SourceAdapter readback (or an identically-normalized
authoritative resource representation). The two state families are related but
not 1:1 mapped.

Force-close that bypasses the CLOSED writeback gate is **not** available through
``validate_transition`` — only a future ``StateMachineService.force_close``
(admin) may set ``external_unsynced=true``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import (
    InvalidStateTransitionError,
    InvalidVerdictStatusCombinationError,
)
from app.models.enums import (
    TERMINAL_SOURCE_DISPOSITIONS,
    ActionCategory,
    ActionExecutionPhase,
    ActionLevel,
    ActionStatus,
    CaseLabel,
    ConfirmationEvidence,
    DispositionIntentKind,
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

# InvalidStateTransitionError / InvalidVerdictStatusCombinationError are defined
# in ``app.core.errors`` (ISSUE-008) and re-exported here for ISSUE-007 imports.

# --------------------------------------------------------------------------- #
# Workflow constants (intro §4.8 / §4.10 / §4.12)
# Defaults; overridable values are resolved via Settings when requested.
# --------------------------------------------------------------------------- #

MAX_REPLAN_COUNT = 3
MAX_AGENT_RETRIES = 2
MIN_EVIDENCE_SOURCES = 3
CONFIDENCE_THRESHOLD = 0.7
GLOBAL_EVIDENCE_TIMEOUT_S = 30.0
SINGLE_SOURCE_TIMEOUT_S = 10.0
APPROVAL_TIMEOUT_MINUTES = 30
FP_HIGH_THRESHOLD = 0.9
FP_LOW_THRESHOLD = 0.7
WRITEBACK_SUBMIT_TIMEOUT_S = 10
WRITEBACK_CONFIRM_TIMEOUT_S = 120
WRITEBACK_MAX_RETRIES = 5

# Levels a system may auto-approve without a human ApprovalRecord (ISSUE-093 §4).
# L2+ is human-in-the-loop by default; auto_execute never bypasses this for L2+.
AUTO_APPROVABLE_ACTION_LEVELS: frozenset[ActionLevel] = frozenset(
    {ActionLevel.L0, ActionLevel.L1}
)

# Budget (§4.10) — defaults mirrored in Settings.
GLOBAL_TOKEN_BUDGET = 1_000_000
EVENT_TOKEN_BUDGET = 100_000
EVENT_COST_BUDGET_USD = 5.0
PER_AGENT_TOKEN_CAP = 20_000
MODEL_PRICE_TABLE: dict[str, float] = {
    "mock-model": 0.0,
}

# Convergence (§4.12)
GLOBAL_MAX_STEPS = 80
MAX_OSCILLATION = 2
MAX_DUPLICATE_TOOL_CALLS = 3
MAX_TOTAL_LLM_CALLS = 30


def resolved_workflow_constants() -> dict[str, Any]:
    """Return constants with Settings overlays for env-overridable knobs."""
    from app.core.config import get_settings

    settings = get_settings()
    return {
        "MAX_REPLAN_COUNT": MAX_REPLAN_COUNT,
        "MAX_AGENT_RETRIES": MAX_AGENT_RETRIES,
        "MIN_EVIDENCE_SOURCES": MIN_EVIDENCE_SOURCES,
        "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "GLOBAL_EVIDENCE_TIMEOUT_S": GLOBAL_EVIDENCE_TIMEOUT_S,
        "SINGLE_SOURCE_TIMEOUT_S": SINGLE_SOURCE_TIMEOUT_S,
        "APPROVAL_TIMEOUT_MINUTES": settings.approval_timeout_minutes,
        "FP_HIGH_THRESHOLD": FP_HIGH_THRESHOLD,
        "FP_LOW_THRESHOLD": FP_LOW_THRESHOLD,
        "WRITEBACK_SUBMIT_TIMEOUT_S": WRITEBACK_SUBMIT_TIMEOUT_S,
        "WRITEBACK_CONFIRM_TIMEOUT_S": WRITEBACK_CONFIRM_TIMEOUT_S,
        "WRITEBACK_MAX_RETRIES": settings.writeback_max_retries,
        "GLOBAL_TOKEN_BUDGET": settings.global_token_budget,
        "EVENT_TOKEN_BUDGET": settings.event_token_budget,
        "EVENT_COST_BUDGET_USD": settings.event_cost_budget_usd,
        "PER_AGENT_TOKEN_CAP": settings.per_agent_token_cap,
        "MODEL_PRICE_TABLE": MODEL_PRICE_TABLE,
        "GLOBAL_MAX_STEPS": GLOBAL_MAX_STEPS,
        "MAX_OSCILLATION": MAX_OSCILLATION,
        "MAX_DUPLICATE_TOOL_CALLS": MAX_DUPLICATE_TOOL_CALLS,
        "MAX_TOTAL_LLM_CALLS": MAX_TOTAL_LLM_CALLS,
    }


# --------------------------------------------------------------------------- #
# EventStatus transition matrix
# --------------------------------------------------------------------------- #

_NON_TERMINAL: frozenset[EventStatus] = frozenset(
    s for s in EventStatus if s not in (EventStatus.CLOSED, EventStatus.FAILED)
)


def _with_failed(targets: set[EventStatus], *, allow_failed: bool = True) -> set[EventStatus]:
    out = set(targets)
    if allow_failed:
        out.add(EventStatus.FAILED)
    return out


STATE_TRANSITIONS: dict[EventStatus, set[EventStatus]] = {
    EventStatus.NEW: _with_failed({EventStatus.TRIAGING}),
    EventStatus.TRIAGING: _with_failed(
        {
            EventStatus.COLLECTING_EVIDENCE,
            EventStatus.CLOSED,  # not_required low/fp only (gated)
            EventStatus.PLANNING_RESPONSE,  # disposition-only fp only (gated)
        }
    ),
    EventStatus.COLLECTING_EVIDENCE: _with_failed({EventStatus.ANALYZING}),
    EventStatus.ANALYZING: _with_failed({EventStatus.SCORING}),
    EventStatus.SCORING: _with_failed({EventStatus.PLANNING_RESPONSE, EventStatus.REPORTING}),
    EventStatus.PLANNING_RESPONSE: _with_failed(
        {EventStatus.WAITING_APPROVAL, EventStatus.EXECUTING_RESPONSE}
    ),
    EventStatus.WAITING_APPROVAL: _with_failed(
        {EventStatus.EXECUTING_RESPONSE, EventStatus.REPORTING}
    ),
    EventStatus.EXECUTING_RESPONSE: _with_failed({EventStatus.VERIFYING}),
    EventStatus.VERIFYING: _with_failed(
        {
            EventStatus.REPORTING,
            EventStatus.CONTAINED,
            EventStatus.REPLANNING,
            # waiting_writeback is a VERIFYING substate — not REPLANNING
        }
    ),
    EventStatus.REPLANNING: _with_failed(
        {
            EventStatus.COLLECTING_EVIDENCE,
            EventStatus.PLANNING_RESPONSE,
            EventStatus.EXECUTING_RESPONSE,
            EventStatus.CONTAINED,
        }
    ),
    EventStatus.CONTAINED: _with_failed({EventStatus.REPORTING}),
    EventStatus.FAILED: {EventStatus.REPORTING},  # no self-loop to FAILED
    EventStatus.REPORTING: _with_failed({EventStatus.CLOSED}),
    EventStatus.CLOSED: set(),  # terminal — no outbound edges
}


# EventStatuses where false_positive is normally forbidden (entity-side-effect path).
VERDICT_STATUS_RULES: dict[FinalVerdict, frozenset[EventStatus]] = {
    FinalVerdict.FALSE_POSITIVE: frozenset(
        {
            EventStatus.PLANNING_RESPONSE,
            EventStatus.WAITING_APPROVAL,
            EventStatus.EXECUTING_RESPONSE,
            EventStatus.VERIFYING,
        }
    ),
}


# --------------------------------------------------------------------------- #
# Action / job / outbox / writeback / execution_substate tables
# --------------------------------------------------------------------------- #

ACTION_STATUS_TRANSITIONS_BY_CATEGORY: dict[
    ActionCategory, dict[ActionStatus, set[ActionStatus]]
] = {
    ActionCategory.RESPONSE: {
        ActionStatus.PENDING: {
            ActionStatus.WAITING_APPROVAL,
            ActionStatus.APPROVED,
            ActionStatus.REJECTED,
            ActionStatus.SUPERSEDED,
        },
        ActionStatus.WAITING_APPROVAL: {
            ActionStatus.APPROVED,
            ActionStatus.REJECTED,
            ActionStatus.SUPERSEDED,
        },
        ActionStatus.APPROVED: {
            ActionStatus.EXECUTING,
            ActionStatus.WAITING_APPROVAL,  # pre-exec hard-gate / template change
            ActionStatus.SUPERSEDED,
        },
        ActionStatus.EXECUTING: {
            ActionStatus.PARTIAL_SUCCESS,
            ActionStatus.SUCCESS,
            ActionStatus.FAILED,
            ActionStatus.UNKNOWN,
        },
        ActionStatus.UNKNOWN: {
            ActionStatus.PARTIAL_SUCCESS,
            ActionStatus.SUCCESS,
            ActionStatus.FAILED,
        },
        ActionStatus.SUCCESS: {ActionStatus.ROLLED_BACK},
        ActionStatus.PARTIAL_SUCCESS: {ActionStatus.ROLLED_BACK},
        ActionStatus.REJECTED: set(),
        ActionStatus.SUPERSEDED: set(),
        ActionStatus.FAILED: set(),
        ActionStatus.ROLLED_BACK: set(),
    },
    ActionCategory.VERIFICATION: {
        ActionStatus.PENDING: {ActionStatus.EXECUTING},
        ActionStatus.EXECUTING: {
            ActionStatus.SUCCESS,
            ActionStatus.FAILED,
            ActionStatus.UNKNOWN,
        },
        ActionStatus.UNKNOWN: {ActionStatus.SUCCESS, ActionStatus.FAILED},
        ActionStatus.SUCCESS: set(),
        ActionStatus.FAILED: set(),
        ActionStatus.WAITING_APPROVAL: set(),
        ActionStatus.APPROVED: set(),
        ActionStatus.REJECTED: set(),
        ActionStatus.SUPERSEDED: set(),
        ActionStatus.PARTIAL_SUCCESS: set(),
        ActionStatus.ROLLED_BACK: set(),
    },
    ActionCategory.ROLLBACK: {
        ActionStatus.PENDING: {
            ActionStatus.WAITING_APPROVAL,
            ActionStatus.APPROVED,
            ActionStatus.REJECTED,
            ActionStatus.SUPERSEDED,
        },
        ActionStatus.WAITING_APPROVAL: {
            ActionStatus.APPROVED,
            ActionStatus.REJECTED,
            ActionStatus.SUPERSEDED,
        },
        ActionStatus.APPROVED: {
            ActionStatus.EXECUTING,
            ActionStatus.WAITING_APPROVAL,  # pre-exec hard-gate / template change
            ActionStatus.SUPERSEDED,  # un-egress only (CAS rejects if job/outbox exists)
        },
        ActionStatus.EXECUTING: {
            ActionStatus.PARTIAL_SUCCESS,
            ActionStatus.SUCCESS,
            ActionStatus.FAILED,
            ActionStatus.UNKNOWN,
        },
        ActionStatus.UNKNOWN: {
            ActionStatus.PARTIAL_SUCCESS,
            ActionStatus.SUCCESS,
            ActionStatus.FAILED,
        },
        # Successful rollback Action stays SUCCESS — it flips source_action_id.
        ActionStatus.SUCCESS: set(),
        ActionStatus.PARTIAL_SUCCESS: set(),
        ActionStatus.REJECTED: set(),
        ActionStatus.SUPERSEDED: set(),
        ActionStatus.FAILED: set(),
        ActionStatus.ROLLED_BACK: set(),
    },
    ActionCategory.SYSTEM: {
        ActionStatus.PENDING: {ActionStatus.EXECUTING, ActionStatus.SUCCESS},
        ActionStatus.EXECUTING: {ActionStatus.SUCCESS, ActionStatus.FAILED},
        ActionStatus.SUCCESS: set(),
        ActionStatus.FAILED: set(),
        ActionStatus.WAITING_APPROVAL: set(),
        ActionStatus.APPROVED: set(),
        ActionStatus.REJECTED: set(),
        ActionStatus.SUPERSEDED: set(),
        ActionStatus.PARTIAL_SUCCESS: set(),
        ActionStatus.UNKNOWN: set(),
        ActionStatus.ROLLED_BACK: set(),
    },
}

# EventStatus values that may hold a non-NONE execution_substate.
EXECUTION_SUBSTATE_HOSTS: dict[EventStatus, frozenset[ExecutionSubstate]] = {
    EventStatus.WAITING_APPROVAL: frozenset(
        {
            ExecutionSubstate.NONE,
            ExecutionSubstate.WAITING_APPROVAL,
            ExecutionSubstate.MANUAL_RESOLUTION,
        }
    ),
    EventStatus.EXECUTING_RESPONSE: frozenset(
        {
            ExecutionSubstate.NONE,
            ExecutionSubstate.WAITING_EXECUTION,
            ExecutionSubstate.MANUAL_RESOLUTION,
        }
    ),
    EventStatus.VERIFYING: frozenset(
        {
            ExecutionSubstate.NONE,
            ExecutionSubstate.WAITING_WRITEBACK,
            ExecutionSubstate.MANUAL_RESOLUTION,
        }
    ),
}

# Stages that must keep execution_substate=NONE (use degraded_flags instead).
EXECUTION_SUBSTATE_FORBIDDEN_STATUSES: frozenset[EventStatus] = frozenset(
    {
        EventStatus.NEW,
        EventStatus.TRIAGING,
        EventStatus.COLLECTING_EVIDENCE,
        EventStatus.ANALYZING,
        EventStatus.SCORING,
        EventStatus.PLANNING_RESPONSE,
        EventStatus.REPLANNING,
        EventStatus.CONTAINED,
        EventStatus.FAILED,
        EventStatus.REPORTING,
        EventStatus.CLOSED,
    }
)

# Logical substate edges (CAS helpers enforce EventStatus binding separately).
EXECUTION_SUBSTATE_TRANSITIONS: dict[ExecutionSubstate, set[ExecutionSubstate]] = {
    ExecutionSubstate.NONE: {
        ExecutionSubstate.WAITING_APPROVAL,
        ExecutionSubstate.WAITING_EXECUTION,
        ExecutionSubstate.WAITING_WRITEBACK,
        ExecutionSubstate.MANUAL_RESOLUTION,
    },
    ExecutionSubstate.WAITING_APPROVAL: {
        ExecutionSubstate.NONE,
        ExecutionSubstate.MANUAL_RESOLUTION,
    },
    ExecutionSubstate.WAITING_EXECUTION: {
        ExecutionSubstate.NONE,
        ExecutionSubstate.MANUAL_RESOLUTION,
    },
    ExecutionSubstate.WAITING_WRITEBACK: {
        ExecutionSubstate.NONE,
        ExecutionSubstate.MANUAL_RESOLUTION,
    },
    ExecutionSubstate.MANUAL_RESOLUTION: {ExecutionSubstate.NONE},
}

JOB_STATUS_TRANSITIONS: dict[ExecutionJobStatus, set[ExecutionJobStatus]] = {
    ExecutionJobStatus.QUEUED: {ExecutionJobStatus.RUNNING, ExecutionJobStatus.CANCELLED},
    ExecutionJobStatus.RUNNING: {
        ExecutionJobStatus.PARTIAL_SUCCESS,
        ExecutionJobStatus.SUCCESS,
        ExecutionJobStatus.FAILED,
        ExecutionJobStatus.TIMED_OUT,
        ExecutionJobStatus.CANCELLED,
        ExecutionJobStatus.UNKNOWN,
    },
    ExecutionJobStatus.UNKNOWN: {
        ExecutionJobStatus.PARTIAL_SUCCESS,
        ExecutionJobStatus.SUCCESS,
        ExecutionJobStatus.FAILED,
        ExecutionJobStatus.TIMED_OUT,  # only after Provider confirms terminal
        ExecutionJobStatus.CANCELLED,
    },
    ExecutionJobStatus.PARTIAL_SUCCESS: set(),
    ExecutionJobStatus.SUCCESS: set(),
    ExecutionJobStatus.FAILED: set(),
    ExecutionJobStatus.TIMED_OUT: set(),
    ExecutionJobStatus.CANCELLED: set(),
}

OUTBOX_DELIVERY_TRANSITIONS: dict[OutboxDeliveryStatus, set[OutboxDeliveryStatus]] = {
    OutboxDeliveryStatus.READY: {OutboxDeliveryStatus.LEASED},
    OutboxDeliveryStatus.LEASED: {
        OutboxDeliveryStatus.DELIVERED,
        OutboxDeliveryStatus.WAITING_RETRY,
        OutboxDeliveryStatus.PAUSED,
        OutboxDeliveryStatus.DEAD_LETTER,
    },
    OutboxDeliveryStatus.WAITING_RETRY: {OutboxDeliveryStatus.LEASED},
    OutboxDeliveryStatus.PAUSED: {
        OutboxDeliveryStatus.READY,  # after status lookup / manual adjudication
        OutboxDeliveryStatus.DEAD_LETTER,
    },
    OutboxDeliveryStatus.DELIVERED: set(),
    OutboxDeliveryStatus.DEAD_LETTER: set(),
}

WRITEBACK_STATUS_TRANSITIONS: dict[WritebackStatus, set[WritebackStatus]] = {
    WritebackStatus.PENDING: {
        WritebackStatus.SENDING,
        WritebackStatus.FAILED,  # pre-send guard/CAS blocked, never egressed
        WritebackStatus.CONFLICT,
    },
    WritebackStatus.SENDING: {
        WritebackStatus.ACCEPTED,
        WritebackStatus.CONFIRMED,
        WritebackStatus.PARTIAL,
        WritebackStatus.FAILED,
        WritebackStatus.CONFLICT,
        WritebackStatus.UNKNOWN,
    },
    WritebackStatus.ACCEPTED: {
        WritebackStatus.CONFIRMED,
        WritebackStatus.PARTIAL,
        WritebackStatus.FAILED,
        WritebackStatus.CONFLICT,
        WritebackStatus.UNKNOWN,
    },
    WritebackStatus.UNKNOWN: {
        WritebackStatus.CONFIRMED,
        WritebackStatus.FAILED,
        WritebackStatus.PENDING,  # only when lookup proves never-accepted
    },
    WritebackStatus.PARTIAL: {
        WritebackStatus.CONFIRMED,
        WritebackStatus.FAILED,
        WritebackStatus.PENDING,  # only when Adapter allows safe retry
    },
    WritebackStatus.FAILED: {
        WritebackStatus.CONFIRMED,
        WritebackStatus.FAILED,
        WritebackStatus.PENDING,
    },
    WritebackStatus.CONFLICT: {
        WritebackStatus.CONFIRMED,
        WritebackStatus.FAILED,
        # CONFLICT→new superseding disposition (new idempotency_key), never mutate payload
    },
    WritebackStatus.CONFIRMED: set(),  # terminal
}


# --------------------------------------------------------------------------- #
# Transition context + CLOSED gate projection
# --------------------------------------------------------------------------- #


class ClosedGateActionView(BaseModel):
    """Minimal projection of an applicable required Action for the CLOSED gate."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    action_category: ActionCategory
    writeback_required: bool
    writeback_applicable: bool
    writeback_readiness: WritebackReadiness
    writeback_status: WritebackStatus | None = None
    has_command: bool = False
    all_required_intents_confirmed: bool = False
    execution_phase: ActionExecutionPhase = ActionExecutionPhase.IMMEDIATE
    tool_name: str | None = None
    approved_terminal_dispositions: list[SourceDisposition] = Field(default_factory=list)
    superseded: bool = False
    rejected: bool = False
    has_job_or_outbox: bool = False


class TerminalEventWritebackView(BaseModel):
    """The single EVENT_STATUS_UPDATE that must close a required cycle."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    disposition_id: str
    writeback_id: str
    closure_cycle: int
    intent_kind: DispositionIntentKind = DispositionIntentKind.EVENT_STATUS_UPDATE
    approved_disposition: SourceDisposition
    actual_disposition: SourceDisposition
    receipt_status: WritebackStatus
    plan_revision: int


class TransitionContext(BaseModel):
    """Caller-supplied business inputs for ``validate_transition``.

    ``disposition_only_intent`` and writeback/gate projections must come from
    server-persisted state (WorkflowRuntimeService / DB) — never from API/LLM
    self-report. ``force_close`` is rejected here; only force_close service path
    may bypass the writeback gate.
    """

    model_config = ConfigDict(extra="forbid")

    final_verdict: FinalVerdict | None = None
    need_investigation: bool | None = None
    disposition_only_intent: bool = False
    disposition_policy: DispositionPolicy | None = None
    severity: Severity | None = None
    # Triage / FP matcher recommendation (e.g. close_as_fp).
    recommendation: str | None = None
    # After disposition-only planning: every response Action must be the deferred tool.
    response_actions_are_disposition_only: bool | None = None
    has_entity_side_effect_actions: bool = False
    # CLOSED gate projection
    report_exists: bool = False
    force_close: bool = False
    applicable_required_actions: list[ClosedGateActionView] = Field(default_factory=list)
    terminal_event_writeback: TerminalEventWritebackView | None = None
    current_plan_revision: int | None = None
    current_closure_cycle: int | None = None


# --------------------------------------------------------------------------- #
# Late false-positive classification (P0 tiers; P1 rollback is not claimed)
# --------------------------------------------------------------------------- #


class LateFalsePositiveTier(StrEnum):
    NO_SIDE_EFFECT = "no_side_effect"
    IN_FLIGHT = "in_flight"
    VERIFIED_EFFECTS = "verified_effects"  # P1 path — must not pretend rollback exists


def classify_late_false_positive(
    *,
    has_immediate_job_or_outbox: bool,
    immediate_in_flight_or_unverified: bool,
    has_verified_successful_entity_action: bool,
) -> LateFalsePositiveTier:
    if has_verified_successful_entity_action:
        return LateFalsePositiveTier.VERIFIED_EFFECTS
    if has_immediate_job_or_outbox or immediate_in_flight_or_unverified:
        return LateFalsePositiveTier.IN_FLIGHT
    return LateFalsePositiveTier.NO_SIDE_EFFECT


def late_fp_allowed_substate(event_status: EventStatus) -> ExecutionSubstate | None:
    """IN_FLIGHT tier must land in manual_resolution on a legal host status."""
    if event_status in (
        EventStatus.WAITING_APPROVAL,
        EventStatus.EXECUTING_RESPONSE,
        EventStatus.VERIFYING,
    ):
        return ExecutionSubstate.MANUAL_RESOLUTION
    return None  # caller must transition to CONTAINED first


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #


def _edge_allowed(table: dict[Any, set[Any]], current: Any, target: Any) -> bool:
    return target in table.get(current, set())


def validate_transition(
    current: EventStatus,
    target: EventStatus,
    context: TransitionContext | None = None,
) -> None:
    """Validate an EventStatus edge (and CLOSED / TRIAGING special gates)."""
    ctx = context or TransitionContext()

    if current is EventStatus.CLOSED:
        raise InvalidStateTransitionError(
            "CLOSED is terminal and has no outbound edges",
            current=current,
            target=target,
        )

    if not _edge_allowed(STATE_TRANSITIONS, current, target):
        raise InvalidStateTransitionError(
            f"illegal transition {current.value} → {target.value}",
            current=current,
            target=target,
        )

    # force_close is never accepted on the normal transition path
    if ctx.force_close:
        raise InvalidStateTransitionError(
            "force_close is not allowed on validate_transition; "
            "use StateMachineService.force_close",
            current=current,
            target=target,
            details={"force_close": True},
        )

    if current is EventStatus.TRIAGING and target is EventStatus.CLOSED:
        _validate_triaging_to_closed(ctx)

    if current is EventStatus.TRIAGING and target is EventStatus.PLANNING_RESPONSE:
        _validate_triaging_to_disposition_only(ctx)

    if target is EventStatus.CLOSED:
        validate_closed_gate(ctx)

    # When entering a disposition-side-effect status with false_positive, check rules.
    if ctx.final_verdict is not None:
        validate_verdict_status(ctx.final_verdict, target, ctx)


def _is_fp_signal(ctx: TransitionContext) -> bool:
    if ctx.final_verdict is FinalVerdict.FALSE_POSITIVE:
        return True
    return ctx.recommendation == "close_as_fp"


def _validate_triaging_to_closed(ctx: TransitionContext) -> None:
    if ctx.disposition_policy is not DispositionPolicy.NOT_REQUIRED:
        raise InvalidStateTransitionError(
            "TRIAGING→CLOSED requires disposition_policy=not_required",
            current=EventStatus.TRIAGING,
            target=EventStatus.CLOSED,
            details={"disposition_policy": getattr(ctx.disposition_policy, "value", None)},
        )
    low = ctx.severity is Severity.LOW
    if not (low or _is_fp_signal(ctx)):
        raise InvalidStateTransitionError(
            "TRIAGING→CLOSED only for not_required low-severity or false-positive",
            current=EventStatus.TRIAGING,
            target=EventStatus.CLOSED,
            details={
                "severity": getattr(ctx.severity, "value", None),
                "final_verdict": getattr(ctx.final_verdict, "value", None),
                "recommendation": ctx.recommendation,
            },
        )


def _validate_triaging_to_disposition_only(ctx: TransitionContext) -> None:
    # Forbidden: need_investigation=false alone must not enter disposition-only.
    if not _is_fp_signal(ctx):
        raise InvalidStateTransitionError(
            "TRIAGING→PLANNING_RESPONSE requires false_positive / close_as_fp "
            "(disposition-only); low-severity required non-FP must investigate",
            current=EventStatus.TRIAGING,
            target=EventStatus.PLANNING_RESPONSE,
            details={
                "need_investigation": ctx.need_investigation,
                "final_verdict": getattr(ctx.final_verdict, "value", None),
                "recommendation": ctx.recommendation,
            },
        )
    if not ctx.disposition_only_intent:
        raise InvalidStateTransitionError(
            "TRIAGING→PLANNING_RESPONSE disposition-only requires "
            "server-persisted disposition_only_intent=true",
            current=EventStatus.TRIAGING,
            target=EventStatus.PLANNING_RESPONSE,
            details={"disposition_only_intent": False},
        )
    # need_investigation=false alone is insufficient even with intent unset above;
    # if somehow intent is true without FP signal we already rejected.


def validate_verdict_status(
    verdict: FinalVerdict,
    status: EventStatus,
    context: TransitionContext | None = None,
) -> None:
    """Reject illegal FinalVerdict × EventStatus combinations."""
    ctx = context or TransitionContext()
    forbidden = VERDICT_STATUS_RULES.get(verdict, frozenset())
    if status not in forbidden:
        return

    # Unique pre-generation exception: trusted disposition_only_intent.
    if (
        verdict is FinalVerdict.FALSE_POSITIVE
        and ctx.disposition_only_intent
        and status is EventStatus.PLANNING_RESPONSE
    ):
        return

    # After plan materialization: every response Action must be disposition-only.
    if (
        verdict is FinalVerdict.FALSE_POSITIVE
        and ctx.disposition_only_intent
        and status
        in (
            EventStatus.WAITING_APPROVAL,
            EventStatus.EXECUTING_RESPONSE,
            EventStatus.VERIFYING,
            EventStatus.PLANNING_RESPONSE,
        )
    ):
        if ctx.has_entity_side_effect_actions:
            raise InvalidVerdictStatusCombinationError(
                "false_positive disposition-only plan must not contain entity side effects",
                details={
                    "final_verdict": verdict.value,
                    "status": status.value,
                    "has_entity_side_effect_actions": True,
                },
            )
        if ctx.response_actions_are_disposition_only is False:
            raise InvalidVerdictStatusCombinationError(
                "false_positive disposition-only plan requires all response Actions "
                "to be update_source_event_disposition",
                details={
                    "final_verdict": verdict.value,
                    "status": status.value,
                },
            )
        if ctx.response_actions_are_disposition_only is True:
            return

    raise InvalidVerdictStatusCombinationError(
        f"final_verdict={verdict.value} is illegal with status={status.value}",
        details={"final_verdict": verdict.value, "status": status.value},
    )


def validate_closed_gate(ctx: TransitionContext) -> None:
    """Hard gate for every transition into CLOSED (required writeback semantics)."""
    if not ctx.report_exists:
        raise InvalidStateTransitionError(
            "CLOSED requires an investigation report",
            target=EventStatus.CLOSED,
            details={"report_exists": False},
        )

    policy = ctx.disposition_policy
    if policy is DispositionPolicy.NOT_REQUIRED:
        return
    if policy is None:
        # Unknown policy: fail closed when targeting CLOSED with no explicit not_required.
        raise InvalidStateTransitionError(
            "CLOSED gate requires disposition_policy",
            target=EventStatus.CLOSED,
        )

    # disposition_policy=required
    applicable = [
        a
        for a in ctx.applicable_required_actions
        if a.writeback_required
        and a.writeback_applicable
        and a.action_category in (ActionCategory.RESPONSE, ActionCategory.ROLLBACK)
        and not a.superseded
        and not a.rejected
    ]
    if not applicable:
        raise InvalidStateTransitionError(
            "required CLOSED gate: zero applicable writeback Actions "
            "(empty / all-rejected sets cannot pass)",
            target=EventStatus.CLOSED,
            details={"applicable_count": 0},
        )

    for action in applicable:
        if action.writeback_readiness is not WritebackReadiness.READY:
            raise InvalidStateTransitionError(
                "required CLOSED gate: applicable Action readiness is not READY",
                target=EventStatus.CLOSED,
                details={
                    "action_id": action.action_id,
                    "writeback_readiness": action.writeback_readiness.value,
                },
            )
        if not action.has_command:
            raise InvalidStateTransitionError(
                "required CLOSED gate: applicable Action has no disposition command",
                target=EventStatus.CLOSED,
                details={"action_id": action.action_id},
            )
        if not action.all_required_intents_confirmed:
            raise InvalidStateTransitionError(
                "required CLOSED gate: required intents are not all CONFIRMED",
                target=EventStatus.CLOSED,
                details={
                    "action_id": action.action_id,
                    "writeback_status": getattr(action.writeback_status, "value", None),
                },
            )
        if action.writeback_status is not WritebackStatus.CONFIRMED:
            raise InvalidStateTransitionError(
                "required CLOSED gate: writeback_status must be CONFIRMED",
                target=EventStatus.CLOSED,
                details={
                    "action_id": action.action_id,
                    "writeback_status": getattr(action.writeback_status, "value", None),
                },
            )

    terminal = ctx.terminal_event_writeback
    if terminal is None:
        raise InvalidStateTransitionError(
            "required CLOSED gate: missing terminal EVENT_STATUS_UPDATE",
            target=EventStatus.CLOSED,
        )
    if terminal.intent_kind is not DispositionIntentKind.EVENT_STATUS_UPDATE:
        raise InvalidStateTransitionError(
            "required CLOSED gate: terminal writeback must be EVENT_STATUS_UPDATE",
            target=EventStatus.CLOSED,
            details={"intent_kind": terminal.intent_kind.value},
        )
    if (
        ctx.current_plan_revision is not None
        and terminal.plan_revision != ctx.current_plan_revision
    ):
        raise InvalidStateTransitionError(
            "required CLOSED gate: terminal writeback must bind current plan_revision",
            target=EventStatus.CLOSED,
            details={
                "plan_revision": terminal.plan_revision,
                "current_plan_revision": ctx.current_plan_revision,
            },
        )
    if (
        ctx.current_closure_cycle is not None
        and terminal.closure_cycle != ctx.current_closure_cycle
    ):
        raise InvalidStateTransitionError(
            "required CLOSED gate: terminal writeback closure_cycle mismatch",
            target=EventStatus.CLOSED,
            details={
                "closure_cycle": terminal.closure_cycle,
                "current_closure_cycle": ctx.current_closure_cycle,
            },
        )
    if terminal.approved_disposition not in TERMINAL_SOURCE_DISPOSITIONS:
        raise InvalidStateTransitionError(
            "required CLOSED gate: approved disposition not terminal",
            target=EventStatus.CLOSED,
            details={"approved_disposition": terminal.approved_disposition.value},
        )
    if terminal.actual_disposition not in TERMINAL_SOURCE_DISPOSITIONS:
        raise InvalidStateTransitionError(
            "required CLOSED gate: actual disposition not terminal",
            target=EventStatus.CLOSED,
            details={"actual_disposition": terminal.actual_disposition.value},
        )
    if terminal.receipt_status is not WritebackStatus.CONFIRMED:
        raise InvalidStateTransitionError(
            "required CLOSED gate: terminal receipt must be CONFIRMED",
            target=EventStatus.CLOSED,
            details={"receipt_status": terminal.receipt_status.value},
        )


def validate_action_status_transition(
    category: ActionCategory,
    current: ActionStatus,
    target: ActionStatus,
    *,
    execution_phase: ActionExecutionPhase = ActionExecutionPhase.IMMEDIATE,
    after_effect_resolution: bool = False,
    template_unchanged: bool = True,
    has_job_or_outbox: bool = False,
    action_level: ActionLevel = ActionLevel.L0,
    auto_execute: bool = False,
    has_approval_evidence: bool = False,
) -> None:
    table = ACTION_STATUS_TRANSITIONS_BY_CATEGORY[category]
    if not _edge_allowed(table, current, target):
        raise InvalidStateTransitionError(
            f"illegal {category.value} ActionStatus {current.value} → {target.value}",
            current=current,
            target=target,
            details={"action_category": category.value},
        )

    if (
        target is ActionStatus.APPROVED
        and category in (ActionCategory.RESPONSE, ActionCategory.ROLLBACK)
        and action_level not in AUTO_APPROVABLE_ACTION_LEVELS
        and not has_approval_evidence
    ):
        # auto_execute never bypasses the human-in-the-loop gate for L2+; it is
        # only meaningful for the already-auto-approvable L0/L1 tier.
        raise InvalidStateTransitionError(
            f"{action_level.value} {category.value} action requires a persisted "
            "ApprovalRecord before →APPROVED (human-in-the-loop gate)",
            current=current,
            target=target,
            details={
                "action_category": category.value,
                "action_level": action_level.value,
                "auto_execute": auto_execute,
            },
        )

    if target is ActionStatus.SUPERSEDED:
        if has_job_or_outbox:
            raise InvalidStateTransitionError(
                "SUPERSEDED only allowed when no job/outbox has been created",
                current=current,
                target=target,
            )
        if current not in (
            ActionStatus.PENDING,
            ActionStatus.WAITING_APPROVAL,
            ActionStatus.APPROVED,
        ):
            raise InvalidStateTransitionError(
                "SUPERSEDED only from un-dispatched PENDING/WAITING_APPROVAL/APPROVED",
                current=current,
                target=target,
            )

    if (
        current is ActionStatus.APPROVED
        and target is ActionStatus.EXECUTING
        and execution_phase is ActionExecutionPhase.POST_VERIFY
    ):
        if not after_effect_resolution or not template_unchanged:
            raise InvalidStateTransitionError(
                "POST_VERIFY APPROVED→EXECUTING requires after_effect_resolution "
                "and unchanged approved template/source/operation",
                current=current,
                target=target,
                details={
                    "after_effect_resolution": after_effect_resolution,
                    "template_unchanged": template_unchanged,
                },
            )

    if target is ActionStatus.ROLLED_BACK and category is not ActionCategory.RESPONSE:
        raise InvalidStateTransitionError(
            "ROLLED_BACK is only valid on the original response Action",
            current=current,
            target=target,
            details={"action_category": category.value},
        )


def validate_execution_substate(
    event_status: EventStatus,
    current: ExecutionSubstate,
    target: ExecutionSubstate,
) -> None:
    if event_status in EXECUTION_SUBSTATE_FORBIDDEN_STATUSES:
        if target is not ExecutionSubstate.NONE:
            raise InvalidStateTransitionError(
                f"{event_status.value} forbids non-NONE execution_substate "
                "(use degraded_flags / human queue instead)",
                current=current,
                target=target,
                details={"event_status": event_status.value},
            )
        return

    allowed = EXECUTION_SUBSTATE_HOSTS.get(event_status)
    if allowed is None or target not in allowed:
        raise InvalidStateTransitionError(
            f"execution_substate {target.value} illegal under {event_status.value}",
            current=current,
            target=target,
            details={"event_status": event_status.value},
        )

    if current is not target and not _edge_allowed(EXECUTION_SUBSTATE_TRANSITIONS, current, target):
        raise InvalidStateTransitionError(
            f"illegal execution_substate {current.value} → {target.value}",
            current=current,
            target=target,
        )


def validate_job_status_transition(
    current: ExecutionJobStatus,
    target: ExecutionJobStatus,
    *,
    provider_confirmed_terminal: bool = False,
) -> None:
    if not _edge_allowed(JOB_STATUS_TRANSITIONS, current, target):
        raise InvalidStateTransitionError(
            f"illegal job status {current.value} → {target.value}",
            current=current,
            target=target,
        )
    if (
        current is ExecutionJobStatus.UNKNOWN
        and target in (ExecutionJobStatus.TIMED_OUT, ExecutionJobStatus.CANCELLED)
        and not provider_confirmed_terminal
    ):
        raise InvalidStateTransitionError(
            "UNKNOWN→TIMED_OUT/CANCELLED requires Provider-confirmed terminal state",
            current=current,
            target=target,
        )


def validate_outbox_delivery_transition(
    current: OutboxDeliveryStatus,
    target: OutboxDeliveryStatus,
    *,
    lease_expired_resend: bool = False,
) -> None:
    if lease_expired_resend and current is OutboxDeliveryStatus.LEASED:
        # Expired lease must PAUSE + lookup first — never direct re-send.
        if target is OutboxDeliveryStatus.LEASED:
            raise InvalidStateTransitionError(
                "expired outbox lease must transition to PAUSED and lookup; "
                "cannot re-LEASED / re-send directly",
                current=current,
                target=target,
            )
    if not _edge_allowed(OUTBOX_DELIVERY_TRANSITIONS, current, target):
        raise InvalidStateTransitionError(
            f"illegal outbox delivery {current.value} → {target.value}",
            current=current,
            target=target,
        )


def validate_writeback_status_transition(
    current: WritebackStatus,
    target: WritebackStatus,
    *,
    lookup_never_accepted: bool = False,
    adapter_allows_safe_retry: bool = False,
    evidence_adjudication: bool = False,
) -> None:
    if not _edge_allowed(WRITEBACK_STATUS_TRANSITIONS, current, target):
        raise InvalidStateTransitionError(
            f"illegal writeback status {current.value} → {target.value}",
            current=current,
            target=target,
        )
    if current is WritebackStatus.UNKNOWN and target is WritebackStatus.PENDING:
        if not lookup_never_accepted:
            raise InvalidStateTransitionError(
                "UNKNOWN→PENDING only when lookup proves never-accepted",
                current=current,
                target=target,
            )
    if (
        current in (WritebackStatus.FAILED, WritebackStatus.PARTIAL)
        and target is WritebackStatus.PENDING
        and not adapter_allows_safe_retry
    ):
        raise InvalidStateTransitionError(
            "FAILED/PARTIAL→PENDING only when Adapter allows safe full/idempotent retry",
            current=current,
            target=target,
        )
    if (
        current
        in (
            WritebackStatus.UNKNOWN,
            WritebackStatus.PARTIAL,
            WritebackStatus.FAILED,
            WritebackStatus.CONFLICT,
        )
        and target in (WritebackStatus.CONFIRMED, WritebackStatus.FAILED)
        and not evidence_adjudication
        and not (current is WritebackStatus.FAILED and target is WritebackStatus.FAILED)
    ):
        # Reaching CONFIRMED/FAILED from non-happy paths needs query or admin evidence.
        # (FAILED→PENDING is gated separately; FAILED→FAILED is a no-op keep.)
        if target is WritebackStatus.CONFIRMED or current is not WritebackStatus.FAILED:
            raise InvalidStateTransitionError(
                "non-happy-path → CONFIRMED/FAILED requires status query or admin adjudication",
                current=current,
                target=target,
            )


def derive_case_label(verdict: FinalVerdict) -> CaseLabel:
    if verdict is FinalVerdict.CONFIRMED_THREAT:
        return CaseLabel.TRUE_POSITIVE
    if verdict is FinalVerdict.FALSE_POSITIVE:
        return CaseLabel.FALSE_POSITIVE
    return CaseLabel.UNCERTAIN


def main_path_reaches_closed() -> bool:
    """BFS connectivity: NEW can reach CLOSED via the matrix (ignoring gates)."""
    seen: set[EventStatus] = set()
    stack = [EventStatus.NEW]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur is EventStatus.CLOSED:
            return True
        stack.extend(STATE_TRANSITIONS.get(cur, ()))
    return False


# --------------------------------------------------------------------------- #
# Source / writeback isolation helpers (acceptance step 10)
# --------------------------------------------------------------------------- #


def apply_external_source_observation(
    *,
    source_object: dict[str, Any],
    frozen_snapshot: dict[str, Any],
    event_status: EventStatus,
    new_current_disposition: str | None = None,
    new_current_status_raw: str | None = None,
    new_concurrency_token: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], EventStatus]:
    """Natural external change updates source current_* only — never EventStatus/snapshot."""
    updated = dict(source_object)
    if new_current_disposition is not None:
        updated["current_source_disposition"] = new_current_disposition
    if new_current_status_raw is not None:
        updated["current_source_status_raw"] = new_current_status_raw
    if new_concurrency_token is not None:
        updated["current_concurrency_token"] = new_concurrency_token
    updated["source_sync_state"] = "observed_external_change"
    # Frozen investigation snapshot must remain byte-identical for identity fields.
    return updated, frozen_snapshot, event_status


def maybe_upgrade_confirmation_evidence(
    *,
    correlated_by_writeback_id: bool,
    readback_matches_command: bool,
    current: ConfirmationEvidence | None,
) -> ConfirmationEvidence | None:
    """Only trusted writeback_id correlation + matching readback may strengthen evidence."""
    if not (correlated_by_writeback_id and readback_matches_command):
        return current  # unrelated external churn cannot impersonate our success
    return ConfirmationEvidence.READBACK_VERIFIED
