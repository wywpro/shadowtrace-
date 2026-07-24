"""Tiered approval engine for response actions (ISSUE-058)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.auth import Principal
from app.core.config import get_settings
from app.core.errors import (
    ApprovalDecisionConflictError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
)
from app.core.event_bus import EventBus
from app.db import models as orm
from app.db.orm.approval import ApprovalRecordORM
from app.models.action import TERMINAL_DISPOSITION_TOOL, Action
from app.models.agent_io import RiskAssessment
from app.models.approval import ApprovalDecision, ApprovalDecisionKind, ApprovalRecord
from app.models.enums import (
    ActionCategory,
    ActionExecutionPhase,
    ActionLevel,
    ActionStatus,
    EventStatus,
    Severity,
    WritebackReadiness,
)
from app.models.ids import new_approval_id
from app.models.tool_meta import CapabilityManifest
from app.models.workflow import (
    AUTO_APPROVABLE_ACTION_LEVELS,
    validate_action_status_transition,
)
from app.services.state_machine_service import StateMachineService

if TYPE_CHECKING:
    from app.services.context_service import EventContextStore

logger = logging.getLogger(__name__)

SYSTEM_TIMEOUT_OPERATOR = "system_timeout"
APPROVAL_ENGINE_OPERATOR = "ApprovalEngine"

L2_CONFIDENCE_THRESHOLD = 0.8
L3_CONFIDENCE_THRESHOLD = 0.85

ResumeHook = Callable[[str], Awaitable[None]]

_APPROVAL_TERMINAL = frozenset({ActionStatus.APPROVED, ActionStatus.REJECTED})


def evaluate_hard_gates(
    action: Action,
    *,
    manifest: CapabilityManifest,
) -> ApprovalDecision | None:
    """Return auto_reject/require_approval when hard gates fail; None if gates pass."""
    if action.tool_name != TERMINAL_DISPOSITION_TOOL:
        if action.tool_name not in manifest.allowed_operations:
            return ApprovalDecision(
                decision=ApprovalDecisionKind.AUTO_REJECT,
                rule_applied="capability_not_allowed",
                reason=f"tool {action.tool_name} not in provider manifest",
            )

    readiness = action.writeback_readiness
    if readiness is WritebackReadiness.PERMISSION_DENIED:
        return ApprovalDecision(
            decision=ApprovalDecisionKind.AUTO_REJECT,
            rule_applied="permission_denied",
            reason="disposition permission denied for target/source",
        )
    if readiness in {
        WritebackReadiness.CAPABILITY_UNSUPPORTED,
        WritebackReadiness.NOT_CONFIGURED,
    }:
        return ApprovalDecision(
            decision=ApprovalDecisionKind.AUTO_REJECT,
            rule_applied="capability_unsupported",
            reason=f"writeback readiness={readiness.value}",
        )
    if readiness in {
        WritebackReadiness.SOURCE_UNRESOLVED,
        WritebackReadiness.CAPABILITY_UNKNOWN,
        WritebackReadiness.CONNECTOR_UNAVAILABLE,
    }:
        return ApprovalDecision(
            decision=ApprovalDecisionKind.REQUIRE_APPROVAL,
            rule_applied="source_or_capability_gate",
            reason=f"writeback readiness={readiness.value} requires human review",
        )

    if action.action_level in AUTO_APPROVABLE_ACTION_LEVELS and action.execution_owner is not None:
        if not manifest.supports_idempotency or not manifest.supports_lookup_by_idempotency:
            return ApprovalDecision(
                decision=ApprovalDecisionKind.REQUIRE_APPROVAL,
                rule_applied="provider_idempotency_missing",
                reason="provider lacks idempotency/lookup; manual approval required",
            )

    return None


def evaluate_level_rules(
    action: Action,
    *,
    confidence: float,
    severity: Severity,
) -> ApprovalDecision:
    """Apply ActionLevel + confidence tier rules after hard gates pass."""
    level = action.action_level

    if level in AUTO_APPROVABLE_ACTION_LEVELS:
        return ApprovalDecision(
            decision=ApprovalDecisionKind.AUTO_APPROVE,
            rule_applied="level_l0_l1",
            reason=f"{level.value} auto-approvable after gates",
        )

    if level is ActionLevel.L2:
        if confidence >= L2_CONFIDENCE_THRESHOLD:
            return ApprovalDecision(
                decision=ApprovalDecisionKind.AUTO_APPROVE,
                rule_applied="level_l2_confidence",
                reason=f"L2 confidence {confidence:.2f} >= {L2_CONFIDENCE_THRESHOLD}",
            )
        return ApprovalDecision(
            decision=ApprovalDecisionKind.REQUIRE_APPROVAL,
            rule_applied="level_l2_confidence",
            reason=f"L2 confidence {confidence:.2f} below {L2_CONFIDENCE_THRESHOLD}",
        )

    if level is ActionLevel.L3:
        if severity in {Severity.HIGH, Severity.CRITICAL} and confidence >= L3_CONFIDENCE_THRESHOLD:
            return ApprovalDecision(
                decision=ApprovalDecisionKind.AUTO_APPROVE,
                rule_applied="level_l3_high_confidence",
                reason=(
                    f"L3 severity={severity.value} confidence {confidence:.2f} "
                    f">= {L3_CONFIDENCE_THRESHOLD}"
                ),
            )
        return ApprovalDecision(
            decision=ApprovalDecisionKind.REQUIRE_APPROVAL,
            rule_applied="level_l3_high_confidence",
            reason="L3 requires high/critical severity and confidence >= 0.85",
        )

    # L4/L5 always manual.
    return ApprovalDecision(
        decision=ApprovalDecisionKind.REQUIRE_APPROVAL,
        rule_applied="level_l4_l5_manual",
        reason=f"{level.value} requires human approval",
    )


def resolve_evaluate_confidence(
    risk_assessment: RiskAssessment | None,
    *,
    disposition_confidence: float | None = None,
) -> tuple[float, Severity]:
    if risk_assessment is not None:
        return float(risk_assessment.confidence), risk_assessment.severity
    conf = float(disposition_confidence or 0.0)
    return conf, Severity.LOW


class ApprovalEngine:
    """Evaluate, persist, and decide tiered approvals for response actions."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        event_bus: EventBus | None = None,
        state_machine: StateMachineService | None = None,
        context_store: EventContextStore | None = None,
        capability_manifest: CapabilityManifest | None = None,
        resume_investigation: ResumeHook | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._event_bus = event_bus
        self._state_machine = state_machine
        self._context_store = context_store
        if capability_manifest is None:
            from app.agents.response_agent import build_mock_capability_manifest

            self._manifest = build_mock_capability_manifest()
        else:
            self._manifest = capability_manifest
        self._resume = resume_investigation
        self._approval_required_published: set[str] = set()

    async def evaluate(
        self,
        action: Action,
        risk_assessment: RiskAssessment | None,
        approval_cycle: int,
        *,
        disposition_confidence: float | None = None,
    ) -> ApprovalDecision:
        existing = await self._load_record(action.action_id, approval_cycle)
        if existing is not None:
            return ApprovalDecision(
                decision=ApprovalDecisionKind(existing.decision),
                rule_applied=str((existing.detail or {}).get("rule_applied", "existing_record")),
                reason=str((existing.detail or {}).get("reason", "existing approval record")),
            )

        gate = evaluate_hard_gates(action, manifest=self._manifest)
        if gate is not None:
            decision = gate
        else:
            confidence, severity = resolve_evaluate_confidence(
                risk_assessment,
                disposition_confidence=disposition_confidence,
            )
            decision = evaluate_level_rules(action, confidence=confidence, severity=severity)

        await self._persist_evaluation(action, decision, approval_cycle)
        await self._apply_evaluation(action, decision)
        return decision

    async def approve(
        self,
        action_id: str,
        principal: Principal,
        comment: str | None,
        decision_id: str | None,
    ) -> None:
        await self._decide(
            action_id,
            principal=principal,
            comment=comment,
            decision_id=decision_id,
            target_status=ActionStatus.APPROVED,
        )

    async def reject(
        self,
        action_id: str,
        principal: Principal,
        comment: str | None,
        decision_id: str | None,
    ) -> None:
        await self._decide(
            action_id,
            principal=principal,
            comment=comment,
            decision_id=decision_id,
            target_status=ActionStatus.REJECTED,
        )

    async def handle_timeout(self, action_id: str, approval_cycle: int) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                row = await self._load_action_row(session, action_id)
                if row is None:
                    return
                if row.status != ActionStatus.WAITING_APPROVAL.value:
                    return
                record = await self._load_record_row(session, action_id, approval_cycle)
                if record is None or record.decided_at is not None:
                    return
                now = datetime.now(UTC)
                record.operator = SYSTEM_TIMEOUT_OPERATOR
                record.comment = "approval timeout"
                record.decision = ApprovalDecisionKind.AUTO_REJECT.value
                record.decided_at = now
                row.status = ActionStatus.REJECTED.value
                row.updated_at = now
                await session.flush()
                action = _action_from_orm(row)
        await self._publish_approval_updated(action, "rejected", SYSTEM_TIMEOUT_OPERATOR, None)
        await self._maybe_advance_plan(action.event_id, action.plan_revision)

    async def scan_timeouts(self) -> list[str]:
        now = datetime.now(UTC)
        touched_events: list[str] = []
        async with self._session_factory() as session:
            locked = await session.scalar(text("SELECT pg_try_advisory_lock(580058)"))
            if not locked:
                return []
            try:
                rows = (
                    await session.scalars(
                        select(ApprovalRecordORM).where(
                            ApprovalRecordORM.decided_at.is_(None),
                            ApprovalRecordORM.timeout_at.is_not(None),
                            ApprovalRecordORM.timeout_at <= now,
                            ApprovalRecordORM.decision
                            == ApprovalDecisionKind.REQUIRE_APPROVAL.value,
                        )
                    )
                ).all()
                for record in rows:
                    action = await self._load_action_row(session, record.action_id)
                    if action is None:
                        continue
                    if action.status != ActionStatus.WAITING_APPROVAL.value:
                        continue
                    record.operator = SYSTEM_TIMEOUT_OPERATOR
                    record.comment = "approval timeout"
                    record.decision = ApprovalDecisionKind.AUTO_REJECT.value
                    record.decided_at = now
                    action.status = ActionStatus.REJECTED.value
                    action.updated_at = now
                    touched_events.append(record.event_id)
                await session.commit()
            finally:
                await session.execute(text("SELECT pg_advisory_unlock(580058)"))

        for event_id in dict.fromkeys(touched_events):
            await self._maybe_advance_plan(event_id, await self._latest_revision(event_id))
        return touched_events

    async def require_manual_review(
        self,
        action_id: str,
        reason: str,
        approval_cycle: int,
    ) -> None:
        async with self._session_factory() as session:
            row = await self._load_action_row(session, action_id)
            if row is None:
                raise ResourceNotFoundError(
                    "action not found",
                    details={"action_id": action_id},
                )
            action = _action_from_orm(row)
            decision = ApprovalDecision(
                decision=ApprovalDecisionKind.REQUIRE_APPROVAL,
                rule_applied="manual_review",
                reason=reason,
            )
            async with session.begin():
                await self._upsert_record(session, action, decision, approval_cycle)
                await self._set_action_status(session, action, ActionStatus.WAITING_APPROVAL)
        await self._ensure_event_waiting_approval(action.event_id)
        await self._publish_approval_required(action, approval_cycle)

    async def is_plan_fully_decided(self, event_id: str, plan_revision: int) -> bool:
        actions = await self._load_plan_response_actions(event_id, plan_revision)
        if not actions:
            return True
        return all(action.status in _APPROVAL_TERMINAL for action in actions)

    async def _decide(
        self,
        action_id: str,
        *,
        principal: Principal,
        comment: str | None,
        decision_id: str | None,
        target_status: ActionStatus,
    ) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                row = await self._load_action_row(session, action_id, for_update=True)
                if row is None:
                    raise ResourceNotFoundError(
                        "action not found",
                        details={"action_id": action_id},
                    )
                action = _action_from_orm(row)

                if decision_id:
                    replay = await session.scalar(
                        select(ApprovalRecordORM).where(
                            ApprovalRecordORM.action_id == action_id,
                            ApprovalRecordORM.decision_id == decision_id,
                            ApprovalRecordORM.decided_at.is_not(None),
                        )
                    )
                    if replay is not None:
                        return

                record = await self._load_pending_record_row(session, action_id)

                if record is not None and record.decided_at is not None:
                    if decision_id and record.decision_id == decision_id:
                        return
                    raise ApprovalDecisionConflictError(
                        "approval already decided by another operator",
                        details={
                            "action_id": action_id,
                            "current_operator": record.operator,
                            "current_status": row.status,
                        },
                    )

                if action.status is not ActionStatus.WAITING_APPROVAL:
                    if action.status in _APPROVAL_TERMINAL:
                        prior_decision = await session.scalar(
                            select(ApprovalRecordORM.approval_id).where(
                                ApprovalRecordORM.action_id == action_id,
                                ApprovalRecordORM.decided_at.is_not(None),
                            )
                        )
                        if prior_decision is not None:
                            raise ApprovalDecisionConflictError(
                                "approval already decided by another operator",
                                details={
                                    "action_id": action_id,
                                    "current_status": action.status.value,
                                },
                            )
                    raise InvalidStateTransitionError(
                        "action is not waiting for approval",
                        current=action.status.value,
                        target=target_status.value,
                        details={"action_id": action_id},
                    )

                if record is None:
                    raise ResourceNotFoundError(
                        "approval record missing",
                        details={"action_id": action_id},
                    )

                if decision_id:
                    conflict = await session.scalar(
                        select(ApprovalRecordORM.approval_id).where(
                            ApprovalRecordORM.decision_id == decision_id,
                            ApprovalRecordORM.action_id != action_id,
                        )
                    )
                    if conflict is not None:
                        raise ApprovalDecisionConflictError(
                            "decision_id already used",
                            details={"decision_id": decision_id},
                        )

                now = datetime.now(UTC)
                record.operator = principal.subject
                record.comment = comment
                record.decided_at = now
                record.decision_id = decision_id
                validate_action_status_transition(
                    action.action_category,
                    action.status,
                    target_status,
                    execution_phase=action.execution_phase,
                    action_level=action.action_level,
                    has_approval_evidence=True,
                )
                row.status = target_status.value
                row.updated_at = now
                await session.flush()
                decided_action = _action_from_orm(row)

        outcome = "approved" if target_status is ActionStatus.APPROVED else "rejected"
        await self._publish_approval_updated(
            decided_action,
            outcome,
            principal.subject,
            comment,
        )
        await self._maybe_advance_plan(decided_action.event_id, decided_action.plan_revision)

    async def _persist_evaluation(
        self,
        action: Action,
        decision: ApprovalDecision,
        approval_cycle: int,
    ) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                await self._upsert_record(session, action, decision, approval_cycle)
                if decision.decision is ApprovalDecisionKind.AUTO_APPROVE:
                    await self._set_action_status(
                        session,
                        action,
                        ActionStatus.APPROVED,
                        has_approval_evidence=True,
                    )
                elif decision.decision is ApprovalDecisionKind.AUTO_REJECT:
                    await self._set_action_status(session, action, ActionStatus.REJECTED)
                else:
                    await self._set_action_status(session, action, ActionStatus.WAITING_APPROVAL)
        if decision.decision is ApprovalDecisionKind.REQUIRE_APPROVAL:
            await self._ensure_event_waiting_approval(action.event_id)
            await self._publish_approval_required(action, approval_cycle)

    async def _apply_evaluation(self, action: Action, decision: ApprovalDecision) -> None:
        if decision.decision in {
            ApprovalDecisionKind.AUTO_APPROVE,
            ApprovalDecisionKind.AUTO_REJECT,
        }:
            await self._maybe_advance_plan(action.event_id, action.plan_revision)

    async def _upsert_record(
        self,
        session: AsyncSession,
        action: Action,
        decision: ApprovalDecision,
        approval_cycle: int,
    ) -> ApprovalRecordORM:
        existing = await self._load_record_row(session, action.action_id, approval_cycle)
        timeout_at = None
        if decision.decision is ApprovalDecisionKind.REQUIRE_APPROVAL:
            minutes = get_settings().approval_timeout_minutes
            timeout_at = datetime.now(UTC) + timedelta(minutes=minutes)

        detail = {
            "rule_applied": decision.rule_applied,
            "reason": decision.reason,
            "impact_assessment": (
                action.impact_assessment.model_dump(mode="json")
                if action.impact_assessment is not None
                else None
            ),
        }
        now = datetime.now(UTC)
        if existing is not None:
            existing.decision = decision.decision.value
            existing.detail = detail
            existing.timeout_at = timeout_at
            row = existing
        else:
            row = ApprovalRecordORM(
                approval_id=new_approval_id(),
                action_id=action.action_id,
                event_id=action.event_id,
                plan_revision=action.plan_revision,
                approval_cycle=approval_cycle,
                required_level=action.action_level.value,
                decision=decision.decision.value,
                operator=(
                    APPROVAL_ENGINE_OPERATOR
                    if decision.decision is not ApprovalDecisionKind.REQUIRE_APPROVAL
                    else None
                ),
                decided_at=(
                    now if decision.decision is not ApprovalDecisionKind.REQUIRE_APPROVAL else None
                ),
                detail=detail,
                requested_at=now,
                timeout_at=timeout_at,
            )
            session.add(row)
        await session.flush()
        await self._sync_wm_records(session, action.event_id)
        return row

    async def _set_action_status(
        self,
        session: AsyncSession,
        action: Action,
        target: ActionStatus,
        *,
        has_approval_evidence: bool = False,
    ) -> None:
        validate_action_status_transition(
            action.action_category,
            action.status,
            target,
            execution_phase=action.execution_phase,
            action_level=action.action_level,
            has_approval_evidence=has_approval_evidence or target is ActionStatus.APPROVED,
        )
        await session.execute(
            update(orm.Action)
            .where(orm.Action.action_id == action.action_id)
            .values(status=target.value, updated_at=datetime.now(UTC))
        )

    async def _ensure_event_waiting_approval(self, event_id: str) -> None:
        if self._state_machine is None:
            return
        status = await self._event_status(event_id)
        if status is None:
            return
        if status in {EventStatus.WAITING_APPROVAL, EventStatus.EXECUTING_RESPONSE}:
            return
        if status is EventStatus.PLANNING_RESPONSE:
            await self._state_machine.transition(
                event_id,
                EventStatus.WAITING_APPROVAL,
                operator=APPROVAL_ENGINE_OPERATOR,
                reason="approval_required",
            )

    async def _maybe_advance_plan(self, event_id: str, plan_revision: int) -> None:
        if not await self.is_plan_fully_decided(event_id, plan_revision):
            return
        actions = await self._load_plan_response_actions(event_id, plan_revision)
        approved = [a for a in actions if a.status is ActionStatus.APPROVED]
        rejected = [a for a in actions if a.status is ActionStatus.REJECTED]
        deferred = [a for a in actions if a.tool_name == TERMINAL_DISPOSITION_TOOL]
        deferred_approved = any(a.status is ActionStatus.APPROVED for a in deferred)
        deferred_rejected = any(a.status is ActionStatus.REJECTED for a in deferred)
        required = any(a.writeback_required for a in actions)

        target: EventStatus | None = None
        if approved and (not required or deferred_approved):
            target = EventStatus.EXECUTING_RESPONSE
        elif rejected and not approved:
            target = EventStatus.REPORTING
        elif required and deferred_rejected:
            target = EventStatus.REPORTING
        elif approved:
            target = EventStatus.REPORTING

        if target is not None and self._state_machine is not None:
            status = await self._event_status(event_id)
            if status in {EventStatus.WAITING_APPROVAL, EventStatus.PLANNING_RESPONSE}:
                try:
                    await self._state_machine.transition(
                        event_id,
                        target,
                        operator=APPROVAL_ENGINE_OPERATOR,
                        reason="plan_fully_decided",
                    )
                except Exception:
                    logger.warning(
                        "plan advance transition failed event=%s target=%s",
                        event_id,
                        target.value,
                        exc_info=True,
                    )

        if self._resume is not None:
            try:
                await self._resume(event_id)
            except Exception:
                logger.warning("resume_investigation hook failed event=%s", event_id, exc_info=True)
        elif self._resume is None:
            logger.warning(
                "resume_investigation not injected; approval facts persisted event=%s",
                event_id,
            )

    async def _event_status(self, event_id: str) -> EventStatus | None:
        async with self._session_factory() as session:
            raw = await session.scalar(
                select(orm.SecurityEvent.status).where(orm.SecurityEvent.event_id == event_id)
            )
            return EventStatus(raw) if raw else None

    async def _publish_approval_required(self, action: Action, approval_cycle: int) -> None:
        key = f"{action.action_id}:{approval_cycle}"
        if key in self._approval_required_published:
            return
        if self._event_bus is None:
            self._approval_required_published.add(key)
            return
        minutes = get_settings().approval_timeout_minutes
        deadline = datetime.now(UTC) + timedelta(minutes=minutes)
        payload = {
            "action_id": action.action_id,
            "action_name": action.action_name,
            "summary": action.reason or action.action_name,
            "target_count": 1 if action.target else 0,
            "deadline": deadline.isoformat(),
        }
        if await self._event_bus.publish_event(action.event_id, "approval_required", payload):
            self._approval_required_published.add(key)

    async def _publish_approval_updated(
        self,
        action: Action,
        decision: str,
        operator: str,
        comment: str | None,
    ) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish_event(
            action.event_id,
            "approval_updated",
            {
                "action_id": action.action_id,
                "decision": decision,
                "approver": operator,
                "comment": comment or "",
            },
        )

    async def _sync_wm_records(self, session: AsyncSession, event_id: str) -> None:
        if self._context_store is None:
            return
        from app.services.context_service import append_context_journal_in_session

        rows = (
            await session.scalars(
                select(ApprovalRecordORM).where(ApprovalRecordORM.event_id == event_id)
            )
        ).all()
        payload = [_record_to_model(row).model_dump(mode="json") for row in rows]
        await append_context_journal_in_session(
            session,
            event_id,
            "approval_records",
            payload,
        )

    async def _load_record(self, action_id: str, approval_cycle: int) -> ApprovalRecordORM | None:
        async with self._session_factory() as session:
            return await self._load_record_row(session, action_id, approval_cycle)

    async def _load_pending_record_row(
        self,
        session: AsyncSession,
        action_id: str,
    ) -> ApprovalRecordORM | None:
        return cast(
            ApprovalRecordORM | None,
            await session.scalar(
                select(ApprovalRecordORM)
                .where(
                    ApprovalRecordORM.action_id == action_id,
                    ApprovalRecordORM.decided_at.is_(None),
                )
                .order_by(ApprovalRecordORM.approval_cycle.desc())
                .limit(1)
            ),
        )

    async def _load_record_row(
        self,
        session: AsyncSession,
        action_id: str,
        approval_cycle: int,
    ) -> ApprovalRecordORM | None:
        return cast(
            ApprovalRecordORM | None,
            await session.scalar(
                select(ApprovalRecordORM).where(
                    ApprovalRecordORM.action_id == action_id,
                    ApprovalRecordORM.approval_cycle == approval_cycle,
                )
            ),
        )

    async def _load_action_row(
        self,
        session: AsyncSession,
        action_id: str,
        *,
        for_update: bool = False,
    ) -> orm.Action | None:
        stmt = select(orm.Action).where(orm.Action.action_id == action_id)
        if for_update:
            stmt = stmt.with_for_update()
        return cast(orm.Action | None, await session.scalar(stmt))

    async def _load_plan_response_actions(
        self,
        event_id: str,
        plan_revision: int,
    ) -> list[Action]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(orm.Action).where(
                        orm.Action.event_id == event_id,
                        orm.Action.plan_revision == plan_revision,
                        orm.Action.action_category == ActionCategory.RESPONSE.value,
                        orm.Action.superseded_by_revision.is_(None),
                        orm.Action.status != ActionStatus.SUPERSEDED.value,
                    )
                )
            ).all()
            return [_action_from_orm(row) for row in rows]

    async def _latest_revision(self, event_id: str) -> int:
        async with self._session_factory() as session:
            value = await session.scalar(
                select(orm.Action.plan_revision)
                .where(orm.Action.event_id == event_id)
                .order_by(orm.Action.plan_revision.desc())
                .limit(1)
            )
            return int(value or 0)


def _action_from_orm(row: orm.Action) -> Action:
    from app.models.enums import ActionCategory, ActionLevel, ActionStatus

    payload = {
        "action_id": row.action_id,
        "event_id": row.event_id,
        "plan_revision": row.plan_revision,
        "action_fingerprint": row.action_fingerprint,
        "action_category": ActionCategory(row.action_category),
        "action_name": row.action_name,
        "tool_name": row.tool_name,
        "action_level": ActionLevel(row.action_level),
        "execution_phase": ActionExecutionPhase(row.execution_phase),
        "activation_condition": row.activation_condition,
        "approved_operation_template_hash": row.approved_operation_template_hash,
        "approved_terminal_dispositions": row.approved_terminal_dispositions or [],
        "target_type": row.target_type,
        "target": row.target,
        "parameters": row.parameters or {},
        "status": ActionStatus(row.status),
        "auto_execute": row.auto_execute,
        "reason": row.reason,
        "impact_assessment": row.impact_assessment,
        "playbook_id": row.playbook_id,
        "provider_name": row.provider_name,
        "execution_owner": row.execution_owner,
        "execution_job_id": row.execution_job_id,
        "tool_call_id": row.tool_call_id,
        "idempotency_key": row.idempotency_key,
        "writeback_required": row.writeback_required,
        "writeback_applicable": row.writeback_applicable,
        "writeback_readiness": WritebackReadiness(row.writeback_readiness),
        "writeback_block_reason": row.writeback_block_reason,
        "writeback_status": row.writeback_status,
        "disposition_source_ref": row.disposition_source_ref,
        "superseded_by_revision": row.superseded_by_revision,
        "executed_at": row.executed_at,
        "effect_verification_status": row.effect_verification_status,
        "rollback_status": ActionStatus(row.rollback_status) if row.rollback_status else None,
        "source_action_id": row.source_action_id,
        "updated_at": row.updated_at,
    }
    return Action.model_validate(payload)


def _record_to_model(row: ApprovalRecordORM) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=row.approval_id,
        action_id=row.action_id,
        event_id=row.event_id,
        plan_revision=row.plan_revision,
        approval_cycle=row.approval_cycle,
        decision_id=row.decision_id,
        required_level=ActionLevel(row.required_level),
        decision=ApprovalDecisionKind(row.decision),
        operator=row.operator,
        comment=row.comment,
        detail=row.detail or {},
        requested_at=row.requested_at,
        decided_at=row.decided_at,
        timeout_at=row.timeout_at,
    )
