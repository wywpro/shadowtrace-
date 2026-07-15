"""Action and ImpactAssessment models (ISSUE-002 field spec).

Key invariants enforced here (intro §4.5 / §4.6):
- system / verification actions: execution_owner is null, writeback not required
  and not applicable.
- response / rollback actions (external side effects / disposition): exactly one
  execution_owner (XDR_MANAGED xor DIRECT_TOOL).
- ``update_source_event_disposition`` is the only POST_VERIFY action, with
  ``activation_condition=after_effect_resolution``; all other actions are IMMEDIATE.
Business ``writeback_required`` is a policy snapshot and must NOT be reverse-driven
by technical capability; readiness carries the blocking reason instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.disposition import SourceObjectLocator
from app.models.enums import (
    ActionCategory,
    ActionExecutionPhase,
    ActionLevel,
    ActionStatus,
    ExecutionOwner,
    Severity,
    SourceDisposition,
    WritebackReadiness,
    WritebackStatus,
)

TERMINAL_DISPOSITION_TOOL = "update_source_event_disposition"


class ImpactAssessment(BaseModel):
    """Estimated blast radius / reversibility of an action (detailed in ISSUE-079)."""

    model_config = ConfigDict(extra="forbid")

    impact_level: Severity = Severity.LOW
    reversible: bool = True
    affected_entity_count: int = 0
    affected_targets: list[str] = Field(default_factory=list)
    notes: str | None = None
    assessed_by: str | None = None


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    event_id: str
    plan_revision: int
    action_fingerprint: str
    action_category: ActionCategory
    action_name: str
    tool_name: str
    action_level: ActionLevel
    execution_phase: ActionExecutionPhase = ActionExecutionPhase.IMMEDIATE
    activation_condition: str | None = None
    approved_operation_template_hash: str | None = None
    approved_terminal_dispositions: list[SourceDisposition] = Field(default_factory=list)
    target_type: str | None = None
    target: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    auto_execute: bool = False
    reason: str | None = None
    impact_assessment: ImpactAssessment | None = None
    playbook_id: str | None = None
    provider_name: str | None = None
    execution_owner: ExecutionOwner | None = None
    execution_job_id: str | None = None
    tool_call_id: str | None = None
    idempotency_key: str | None = None
    writeback_required: bool = False
    writeback_applicable: bool = False
    writeback_readiness: WritebackReadiness = WritebackReadiness.NOT_REQUIRED
    writeback_block_reason: str | None = None
    writeback_status: WritebackStatus | None = None
    disposition_source_ref: SourceObjectLocator | None = None
    superseded_by_revision: int | None = None
    executed_at: datetime | None = None
    effect_verification_status: str | None = None
    rollback_status: ActionStatus | None = None
    source_action_id: str | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def _enforce_owner_and_phase(self) -> Action:
        # system / verification never own external submission or writeback.
        if self.action_category in (ActionCategory.SYSTEM, ActionCategory.VERIFICATION):
            if self.execution_owner is not None:
                raise ValueError(
                    f"{self.action_category.value} action must not set execution_owner"
                )
            if self.writeback_required or self.writeback_applicable:
                raise ValueError(
                    f"{self.action_category.value} action cannot require/apply writeback"
                )
        # response / rollback produce external effects: exactly one owner.
        if self.action_category in (ActionCategory.RESPONSE, ActionCategory.ROLLBACK):
            if self.execution_owner is None:
                raise ValueError(
                    f"{self.action_category.value} action must select exactly one "
                    "execution_owner (XDR_MANAGED xor DIRECT_TOOL)"
                )

        # Only the terminal disposition tool is POST_VERIFY; all others IMMEDIATE.
        if self.tool_name == TERMINAL_DISPOSITION_TOOL:
            if self.execution_phase is not ActionExecutionPhase.POST_VERIFY:
                raise ValueError(f"{TERMINAL_DISPOSITION_TOOL} must be POST_VERIFY")
            if self.activation_condition != "after_effect_resolution":
                raise ValueError(
                    f"{TERMINAL_DISPOSITION_TOOL} requires "
                    "activation_condition=after_effect_resolution"
                )
        elif self.execution_phase is not ActionExecutionPhase.IMMEDIATE:
            raise ValueError("only update_source_event_disposition may be POST_VERIFY")
        return self

    @model_validator(mode="after")
    def _enforce_writeback_consistency(self) -> Action:
        """Reject impossible writeback field combinations (ISSUE-093 §3).

        ``writeback_required`` is a policy snapshot and must never be reverse
        -driven by capability; these rules only police *internal* consistency
        between required / applicable / readiness / status on a single Action.
        """
        if not self.writeback_required:
            if self.writeback_applicable:
                raise ValueError(
                    "writeback_required=false forbids writeback_applicable=true"
                )
            if self.writeback_readiness is not WritebackReadiness.NOT_REQUIRED:
                raise ValueError(
                    "writeback_required=false requires writeback_readiness=NOT_REQUIRED"
                )
            if self.writeback_status is not None:
                raise ValueError("writeback_required=false requires writeback_status=null")
            return self

        # writeback_required=True from here on.
        if not self.writeback_applicable:
            # The obligation exists at the event level but does not land on
            # this specific action (e.g. it targets no writable source object).
            if self.writeback_readiness is not WritebackReadiness.NOT_REQUIRED:
                raise ValueError(
                    "writeback_applicable=false requires writeback_readiness=NOT_REQUIRED"
                )
            if self.writeback_status is not None:
                raise ValueError("writeback_applicable=false requires writeback_status=null")
            return self

        # writeback_required=True and writeback_applicable=True: readiness must
        # reflect a real (non-NOT_REQUIRED) state, and a status may only exist
        # once the writeback is actually READY to be (or has been) attempted.
        if self.writeback_readiness is WritebackReadiness.NOT_REQUIRED:
            raise ValueError(
                "writeback_required=true and writeback_applicable=true forbid "
                "writeback_readiness=NOT_REQUIRED"
            )
        if (
            self.writeback_readiness is not WritebackReadiness.READY
            and self.writeback_status is not None
        ):
            raise ValueError(
                f"writeback_readiness={self.writeback_readiness.value} blocks any "
                "writeback attempt; writeback_status must be null"
            )
        return self
