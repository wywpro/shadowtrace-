"""Action execution and writeback dispatch orchestration (ISSUE-059)."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.errors import EventNotFoundError, InvalidStateTransitionError, ValidationError
from app.core.event_bus import EventBus
from app.db import models as orm
from app.models.action import TERMINAL_DISPOSITION_TOOL, Action
from app.models.disposition import SourceObjectLocator
from app.models.enums import (
    ActionCategory,
    ActionExecutionPhase,
    ActionStatus,
    EventStatus,
    ExecutionJobStatus,
    ExecutionOwner,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.execution import ActionExecutionJob, ExecutionActionView, ExecutionSummary
from app.models.ids import new_disposition_id, new_job_id
from app.models.workflow import validate_action_status_transition
from app.services.context_service import EventContextStore
from app.services.disposition_command_factory import (
    DispositionCommandFactory,
    entity_action_code_for,
)
from app.services.disposition_sync_service import DispositionSyncService
from app.services.state_machine_service import StateMachineService
from app.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

_EXECUTION_OPERATOR = "ActionExecutionService"


def _action_from_row(row: orm.Action) -> Action:
    return Action.model_validate(
        {
            "action_id": row.action_id,
            "event_id": row.event_id,
            "plan_revision": row.plan_revision,
            "action_fingerprint": row.action_fingerprint,
            "action_category": row.action_category,
            "action_name": row.action_name,
            "tool_name": row.tool_name,
            "action_level": row.action_level,
            "execution_phase": row.execution_phase,
            "activation_condition": row.activation_condition,
            "approved_operation_template_hash": row.approved_operation_template_hash,
            "approved_terminal_dispositions": row.approved_terminal_dispositions or [],
            "target_type": row.target_type,
            "target": row.target,
            "parameters": row.parameters or {},
            "status": row.status,
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
            "writeback_readiness": row.writeback_readiness,
            "writeback_block_reason": row.writeback_block_reason,
            "writeback_status": row.writeback_status,
            "disposition_source_ref": row.disposition_source_ref,
            "superseded_by_revision": row.superseded_by_revision,
            "executed_at": row.executed_at,
            "effect_verification_status": row.effect_verification_status,
            "rollback_status": row.rollback_status,
            "source_action_id": row.source_action_id,
            "updated_at": row.updated_at,
        }
    )


class DbExecutionJobStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_job(self, job_id: str) -> ActionExecutionJob | None:
        async with self._session_factory() as session:
            row = await session.get(orm.ActionExecutionJob, job_id)
            if row is None:
                return None
            return _job_from_row(row)

    async def cas_update_job(
        self,
        job_id: str,
        updated: ActionExecutionJob,
        *,
        expected_status: ExecutionJobStatus,
    ) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(orm.ActionExecutionJob, job_id, with_for_update=True)
                if row is None or ExecutionJobStatus(row.status) is not expected_status:
                    return False
                _apply_job_row(row, updated)
                return True


def _job_from_row(row: orm.ActionExecutionJob) -> ActionExecutionJob:
    return ActionExecutionJob(
        job_id=row.job_id,
        event_id=row.event_id,
        action_id=row.action_id,
        provider_name=row.provider_name,
        idempotency_key=row.idempotency_key,
        provider_job_id=row.provider_job_id,
        status=ExecutionJobStatus(row.status),
        claimed_by=row.claimed_by,
        lease_expires_at=row.lease_expires_at,
        poll_after_ms=row.poll_after_ms,
        attempt=row.attempt,
        provider_code=row.provider_code,
        provider_message=row.provider_message,
        raw_result=row.raw_result or {},
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _apply_job_row(row: orm.ActionExecutionJob, job: ActionExecutionJob) -> None:
    row.status = job.status.value
    row.provider_name = job.provider_name
    row.provider_job_id = job.provider_job_id
    row.provider_code = job.provider_code
    row.provider_message = job.provider_message
    row.raw_result = job.raw_result
    row.updated_at = datetime.now(UTC)
    row.started_at = job.started_at
    row.finished_at = job.finished_at


class ActionExecutionService:
    """Execute approved IMMEDIATE actions and maintain execution_summary."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        disposition_sync: DispositionSyncService,
        tool_executor: ToolExecutor,
        state_machine: StateMachineService,
        context_store: EventContextStore,
        command_factory: DispositionCommandFactory | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._sync = disposition_sync
        self._executor = tool_executor
        self._state_machine = state_machine
        self._context_store = context_store
        self._factory = command_factory or DispositionCommandFactory()
        self._bus = event_bus
        self._job_store = DbExecutionJobStore(session_factory)
        if self._executor.job_store is None:
            self._executor.job_store = self._job_store

    async def get_actions_by_event(self, event_id: str) -> list[Action]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(orm.Action)
                    .where(orm.Action.event_id == event_id)
                    .order_by(orm.Action.plan_revision.asc(), orm.Action.action_id.asc())
                )
            ).all()
        return [_action_from_row(row) for row in rows]

    async def execute_plan(
        self,
        event_id: str,
        *,
        plan_revision: int | None = None,
        operator: str = _EXECUTION_OPERATOR,
    ) -> ExecutionSummary:
        settings = get_settings()
        if settings.allow_live_side_effects:
            raise ValidationError(
                "live side effects are disabled in ISSUE-059 P0",
                details={"allow_live_side_effects": True},
            )
        revision = plan_revision or await self._current_revision(event_id)
        immediate = await self._load_claimable_actions(event_id, revision)
        if not immediate:
            await self._state_machine.transition(
                event_id,
                EventStatus.VERIFYING,
                operator=operator,
                reason="execute_plan:no_immediate_actions",
            )
            return await self._build_summary(event_id, revision)

        for action in immediate:
            await self.execute_action(action.action_id, operator=operator)
        await self._sync.process_ready_outboxes(limit=len(immediate) + 1)
        return await self._build_summary(event_id, revision)

    async def execute_action(
        self, action_id: str, *, operator: str = _EXECUTION_OPERATOR
    ) -> Action:
        claimed = await self._claim_action(action_id)
        if claimed.execution_owner is ExecutionOwner.XDR_MANAGED:
            await self._execute_xdr_managed(claimed, operator=operator)
        elif claimed.execution_owner is ExecutionOwner.DIRECT_TOOL:
            await self._execute_direct_tool(claimed, operator=operator)
        else:
            raise ValidationError(
                "action missing execution_owner",
                details={"action_id": action_id},
            )
        async with self._session_factory() as session:
            row = await session.get(orm.Action, action_id)
            assert row is not None
            return _action_from_row(row)

    async def resolve_unknown(
        self,
        action_id: str,
        resolution: str,
        *,
        principal: str,
        comment: str,
        evidence_ref: str | None = None,
    ) -> Action:
        mapping = {
            "mark_success": ActionStatus.SUCCESS,
            "manual_confirmed": ActionStatus.SUCCESS,
            "mark_failed": ActionStatus.FAILED,
            "success": ActionStatus.SUCCESS,
            "partial_success": ActionStatus.PARTIAL_SUCCESS,
            "failed": ActionStatus.FAILED,
        }
        if resolution not in mapping:
            raise ValidationError(
                "unsupported action resolution",
                details={"resolution": resolution},
            )
        target = mapping[resolution]
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(orm.Action, action_id, with_for_update=True)
                if row is None:
                    raise EventNotFoundError(
                        f"action not found: {action_id}",
                        details={"action_id": action_id},
                    )
                current = ActionStatus(row.status)
                if current is not ActionStatus.UNKNOWN:
                    raise InvalidStateTransitionError(
                        "resolve_unknown requires UNKNOWN action",
                        current=current,
                        target=target,
                    )
                validate_action_status_transition(
                    ActionCategory(row.action_category),
                    current,
                    target,
                )
                row.status = target.value
                row.executed_at = datetime.now(UTC)
                row.updated_at = datetime.now(UTC)
                session.add(
                    orm.EventAuditLog(
                        event_id=row.event_id,
                        from_status=current.value,
                        to_status=target.value,
                        operator=principal,
                        reason=f"resolve_unknown:{resolution}:{comment}",
                    )
                )
        async with self._session_factory() as session:
            row = await session.get(orm.Action, action_id)
            assert row is not None
            return _action_from_row(row)

    async def _execute_xdr_managed(self, action: Action, *, operator: str) -> None:
        locator, source_record_id = await self._resolve_source(action)
        command = self._factory.build_entity_action_submit(
            action,
            source_locator=locator,
            source_concurrency_token=await self._current_concurrency_token(source_record_id),
            operator_id=operator,
            disposition_id=new_disposition_id(),
            writeback_id="pending",
            closure_cycle=1,
            entity_action_code=entity_action_code_for(action),
        )
        async with self._session_factory() as session:
            async with session.begin():
                await self._sync.enqueue_command(
                    session,
                    command=command,
                    event_id=action.event_id,
                    source_record_id=source_record_id,
                )

    async def _execute_direct_tool(self, action: Action, *, operator: str) -> None:
        job_id = new_job_id()
        idempotency_key = action.idempotency_key or f"{action.action_id}:direct"
        async with self._session_factory() as session:
            async with session.begin():
                session.add(
                    orm.ActionExecutionJob(
                        job_id=job_id,
                        event_id=action.event_id,
                        action_id=action.action_id,
                        provider_name="mock_tool_provider",
                        idempotency_key=idempotency_key,
                        status=ExecutionJobStatus.QUEUED.value,
                    )
                )
                action_row = await session.get(orm.Action, action.action_id, with_for_update=True)
                assert action_row is not None
                action_row.execution_job_id = job_id
        await self._executor.call(
            action.tool_name,
            dict(action.parameters or {}),
            action.event_id,
            action_id=action.action_id,
            execution_job_id=job_id,
            idempotency_key=idempotency_key,
            execution_owner=ExecutionOwner.DIRECT_TOOL,
        )
        job = await self._finalize_mock_direct_tool_job(job_id)
        if job is None:
            return
        terminal = job.status
        async with self._session_factory() as session:
            async with session.begin():
                action_row = await session.get(orm.Action, action.action_id, with_for_update=True)
                assert action_row is not None
                current = ActionStatus(action_row.status)
                target = _map_job_to_action_status(terminal)
                if target is not current:
                    validate_action_status_transition(
                        ActionCategory(action_row.action_category),
                        current,
                        target,
                    )
                    action_row.status = target.value
                    action_row.executed_at = datetime.now(UTC)
        if action.writeback_applicable and terminal in {
            ExecutionJobStatus.SUCCESS,
            ExecutionJobStatus.PARTIAL_SUCCESS,
        }:
            locator, source_record_id = await self._resolve_source(action)
            command = self._factory.build_execution_result_record(
                action,
                job,
                source_locator=locator,
                source_concurrency_token=await self._current_concurrency_token(source_record_id),
                operator_id=operator,
                disposition_id=new_disposition_id(),
                closure_cycle=1,
            )
            async with self._session_factory() as session:
                async with session.begin():
                    await self._sync.enqueue_command(
                        session,
                        command=command,
                        event_id=action.event_id,
                        source_record_id=source_record_id,
                    )

    async def _finalize_mock_direct_tool_job(self, job_id: str) -> ActionExecutionJob | None:
        """Run mock async Provider jobs to terminal before writeback enqueue (ISSUE-059 P0)."""
        job = await self._job_store.get_job(job_id)
        if job is None:
            return None
        if job.status not in {ExecutionJobStatus.QUEUED, ExecutionJobStatus.RUNNING}:
            return job
        settings = get_settings()
        tool_mode = settings.tool_mode.strip().lower()
        if "mock" not in tool_mode or not settings.simulation_enabled:
            return job
        from app.providers.tools.mock_provider import get_mock_tool_provider

        completed = await get_mock_tool_provider().run_job(job_id)
        await self._job_store.cas_update_job(job_id, completed, expected_status=job.status)
        return completed

    async def _claim_action(self, action_id: str) -> Action:
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(orm.Action, action_id, with_for_update=True)
                if row is None:
                    raise EventNotFoundError(
                        f"action not found: {action_id}",
                        details={"action_id": action_id},
                    )
                action = _action_from_row(row)
                if action.status is not ActionStatus.APPROVED:
                    raise InvalidStateTransitionError(
                        "action is not claimable",
                        current=action.status,
                        target=ActionStatus.EXECUTING,
                    )
                if action.execution_phase is ActionExecutionPhase.POST_VERIFY:
                    raise ValidationError(
                        "POST_VERIFY actions must remain APPROVED until ISSUE-059A",
                        details={
                            "action_id": action_id,
                            "execution_phase": action.execution_phase.value,
                        },
                    )
                if action.tool_name == TERMINAL_DISPOSITION_TOOL:
                    raise ValidationError(
                        "POST_VERIFY deferred actions are not executed by execute_plan",
                        details={"action_id": action_id},
                    )
                if action.execution_phase is not ActionExecutionPhase.IMMEDIATE:
                    raise ValidationError(
                        "only IMMEDIATE actions are claimable in execute_plan",
                        details={"action_id": action_id},
                    )
                if action.superseded_by_revision is not None:
                    raise ValidationError(
                        "superseded action cannot be claimed",
                        details={"action_id": action_id},
                    )
                if action.action_category not in {
                    ActionCategory.RESPONSE,
                    ActionCategory.ROLLBACK,
                }:
                    raise ValidationError(
                        "only response/rollback actions are executable",
                        details={"action_id": action_id},
                    )
                if action.writeback_applicable and (
                    action.writeback_readiness is not WritebackReadiness.READY
                    or action.disposition_source_ref is None
                ):
                    raise ValidationError(
                        "action writeback readiness blocks execution",
                        details={
                            "action_id": action_id,
                            "writeback_readiness": action.writeback_readiness.value,
                        },
                    )
                await self._validate_claim_preconditions(session, action)
                validate_action_status_transition(
                    action.action_category,
                    ActionStatus.APPROVED,
                    ActionStatus.EXECUTING,
                )
                row.status = ActionStatus.EXECUTING.value
                row.updated_at = datetime.now(UTC)
                return _action_from_row(row)

    async def _load_claimable_actions(self, event_id: str, revision: int) -> list[Action]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(orm.Action).where(
                        orm.Action.event_id == event_id,
                        orm.Action.plan_revision == revision,
                        orm.Action.status == ActionStatus.APPROVED.value,
                        orm.Action.execution_phase == ActionExecutionPhase.IMMEDIATE.value,
                        orm.Action.superseded_by_revision.is_(None),
                        orm.Action.action_category.in_(
                            (
                                ActionCategory.RESPONSE.value,
                                ActionCategory.ROLLBACK.value,
                            )
                        ),
                    )
                )
            ).all()
        return [_action_from_row(row) for row in rows]

    async def _current_revision(self, event_id: str) -> int:
        async with self._session_factory() as session:
            value = await session.scalar(
                select(func.max(orm.Action.plan_revision)).where(orm.Action.event_id == event_id)
            )
        return int(value or 1)

    async def _validate_claim_preconditions(
        self,
        session: AsyncSession,
        action: Action,
    ) -> None:
        settings = get_settings()
        if settings.allow_live_side_effects:
            raise ValidationError(
                "live side effects are disabled in ISSUE-059 P0",
                details={"allow_live_side_effects": True, "action_id": action.action_id},
            )
        if action.execution_owner is ExecutionOwner.XDR_MANAGED:
            disposition_mode = settings.disposition_mode.strip().lower()
            if "mock" not in disposition_mode and not settings.allow_xdr_writeback:
                raise ValidationError(
                    "xdr writeback is not enabled for live disposition mode",
                    details={
                        "action_id": action.action_id,
                        "disposition_mode": settings.disposition_mode,
                    },
                )
        if action.disposition_source_ref is None:
            if action.writeback_applicable or action.execution_owner is ExecutionOwner.XDR_MANAGED:
                raise ValidationError(
                    "action missing disposition_source_ref at claim time",
                    details={"action_id": action.action_id},
                )
            return
        locator = SourceObjectLocator.model_validate(action.disposition_source_ref)
        source = await session.scalar(
            select(orm.SourceObject).where(
                orm.SourceObject.source_product == locator.source_product,
                orm.SourceObject.source_tenant_id == locator.source_tenant_id,
                orm.SourceObject.connector_id == locator.connector_id,
                orm.SourceObject.source_kind == locator.source_kind.value,
                orm.SourceObject.source_object_id == locator.source_object_id,
            )
        )
        if source is None:
            raise ValidationError(
                "source object not found at claim time",
                details={"action_id": action.action_id},
            )

    async def _resolve_source(self, action: Action) -> tuple[SourceObjectLocator, str]:
        if action.disposition_source_ref is None:
            raise ValidationError(
                "action missing disposition_source_ref",
                details={"action_id": action.action_id},
            )
        locator = SourceObjectLocator.model_validate(action.disposition_source_ref)
        async with self._session_factory() as session:
            row = await session.scalar(
                select(orm.SourceObject.source_record_id).where(
                    orm.SourceObject.source_product == locator.source_product,
                    orm.SourceObject.source_tenant_id == locator.source_tenant_id,
                    orm.SourceObject.connector_id == locator.connector_id,
                    orm.SourceObject.source_kind == locator.source_kind.value,
                    orm.SourceObject.source_object_id == locator.source_object_id,
                )
            )
        if row is None:
            raise ValidationError(
                "source object not found for action",
                details={"action_id": action.action_id},
            )
        return locator, str(row)

    async def _current_concurrency_token(self, source_record_id: str) -> str | None:
        async with self._session_factory() as session:
            row = await session.get(orm.SourceObject, source_record_id)
        return row.current_concurrency_token if row is not None else None

    async def _build_summary(self, event_id: str, plan_revision: int) -> ExecutionSummary:
        async with self._session_factory() as session:
            actions = (
                await session.scalars(
                    select(orm.Action).where(
                        orm.Action.event_id == event_id,
                        orm.Action.plan_revision == plan_revision,
                    )
                )
            ).all()
            jobs = (
                await session.scalars(
                    select(orm.ActionExecutionJob).where(
                        orm.ActionExecutionJob.event_id == event_id
                    )
                )
            ).all()
            outboxes = (
                await session.scalars(
                    select(orm.DispositionOutbox).where(orm.DispositionOutbox.event_id == event_id)
                )
            ).all()
            writeback_summary = await self._context_store.get(event_id, "writeback_summary")
        writeback_counts: Counter[str] = Counter()
        writeback_ids: list[str] = []
        for outbox in outboxes:
            writeback_ids.append(outbox.writeback_id)
            if outbox.latest_writeback_status:
                writeback_counts[outbox.latest_writeback_status] += 1
        counts = Counter(ActionStatus(row.status).value for row in actions)
        action_views = [
            ExecutionActionView(
                action_id=row.action_id,
                action_status=ActionStatus(row.status),
                execution_phase=ActionExecutionPhase(row.execution_phase),
                writeback_required=bool(row.writeback_required),
                writeback_applicable=bool(row.writeback_applicable),
                writeback_readiness=WritebackReadiness(row.writeback_readiness),
                writeback_status=(
                    WritebackStatus(row.writeback_status) if row.writeback_status else None
                ),
            )
            for row in actions
        ]
        summary = ExecutionSummary(
            event_id=event_id,
            plan_revision=plan_revision,
            action_counts=dict(counts),
            jobs=[_job_from_row(job) for job in jobs],
            actions=action_views,
            writeback_counts=dict(writeback_counts),
            writeback_ids=writeback_ids,
            writeback_summary=writeback_summary,
            updated_at=datetime.now(UTC),
        )
        await self._context_store.set(
            event_id,
            "execution_summary",
            summary.model_dump(mode="json"),
        )
        return summary


def _map_job_to_action_status(status: ExecutionJobStatus) -> ActionStatus:
    if status is ExecutionJobStatus.SUCCESS:
        return ActionStatus.SUCCESS
    if status is ExecutionJobStatus.PARTIAL_SUCCESS:
        return ActionStatus.PARTIAL_SUCCESS
    if status in {
        ExecutionJobStatus.FAILED,
        ExecutionJobStatus.TIMED_OUT,
        ExecutionJobStatus.CANCELLED,
    }:
        return ActionStatus.FAILED
    if status is ExecutionJobStatus.UNKNOWN:
        return ActionStatus.UNKNOWN
    return ActionStatus.EXECUTING


__all__ = ["ActionExecutionService", "DbExecutionJobStore"]
