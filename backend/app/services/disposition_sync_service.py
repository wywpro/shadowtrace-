"""Reliable disposition outbox delivery (ISSUE-059)."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.disposition.base import BaseDispositionAdapter
from app.adapters.registry import DispositionAdapterRegistry
from app.core.errors import (
    EventNotFoundError,
    GuardrailViolationError,
    ValidationError,
    WritebackConflictError,
)
from app.core.event_bus import EventBus
from app.core.guardrails import OutboundDispositionGuard
from app.db import models as orm
from app.models.disposition import DispositionCommand, DispositionOutboxRecord, DispositionReceipt
from app.models.enums import (
    ConfirmationEvidence,
    ExecutionSubstate,
    OutboxDeliveryStatus,
    WritebackStatus,
)
from app.models.ids import new_writeback_id
from app.models.workflow import (
    validate_outbox_delivery_transition,
    validate_writeback_status_transition,
)
from app.services.context_service import (
    EventContextStore,
    append_context_journal_in_session,
    append_list_context_journal_in_session,
)
from app.services.disposition_command_factory import DispositionCommandFactory

logger = logging.getLogger(__name__)

ResumeInvestigationHook = Callable[[str], Awaitable[None]]
_DEFAULT_LEASE_SECONDS = 30


class _NullResumeHook:
    async def __call__(self, event_id: str) -> None:
        return None


def _new_outbox_id() -> str:
    return f"obx-{secrets.token_hex(4)}"


def _payload_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DispositionSyncService:
    """Owns disposition_commands/receipts/writeback_summary WorkingMemory fields."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        context_store: EventContextStore,
        adapter_registry: DispositionAdapterRegistry,
        command_factory: DispositionCommandFactory | None = None,
        outbound_guard: OutboundDispositionGuard | None = None,
        event_bus: EventBus | None = None,
        resume_investigation: ResumeInvestigationHook | None = None,
        worker_id: str = "outbox-worker-1",
    ) -> None:
        self._session_factory = session_factory
        self._context_store = context_store
        self._adapters = adapter_registry
        self._factory = command_factory or DispositionCommandFactory()
        self._guard = outbound_guard or OutboundDispositionGuard()
        self._bus = event_bus
        self._resume = resume_investigation or _NullResumeHook()
        self._worker_id = worker_id

    async def enqueue_command(
        self,
        session: AsyncSession,
        *,
        command: DispositionCommand,
        event_id: str,
        source_record_id: str,
        logical_slot: str = "default",
        guard_context: dict[str, Any] | None = None,
    ) -> DispositionOutboxRecord:
        ctx = {
            "event_id": event_id,
            "source_locator": command.source_locator,
            "approved_action_ids": [command.action_id],
            **(guard_context or {}),
        }
        await self._guard.validate(command, ctx)

        source_row = await session.get(
            orm.SourceObject,
            source_record_id,
            with_for_update=True,
        )
        if source_row is None:
            raise ValidationError(
                "source_record not found for outbox enqueue",
                details={"source_record_id": source_record_id},
            )
        source_row.next_outbox_sequence = int(source_row.next_outbox_sequence or 0) + 1
        source_sequence = int(source_row.next_outbox_sequence)
        await session.flush()

        payload = command.model_dump(mode="json")
        outbox = orm.DispositionOutbox(
            outbox_id=_new_outbox_id(),
            writeback_id=new_writeback_id(),
            disposition_id=command.disposition_id,
            action_id=command.action_id,
            event_id=event_id,
            closure_cycle=command.closure_cycle,
            source_record_id=source_record_id,
            source_locator_hash=self._factory.locator_hash(command.source_locator),
            source_sequence=source_sequence,
            intent_kind=command.intent_kind.value,
            logical_slot=logical_slot,
            idempotency_key=command.idempotency_key,
            command_payload=payload,
            command_payload_sha256=_payload_sha256(payload),
            delivery_status=OutboxDeliveryStatus.READY.value,
        )
        session.add(outbox)
        await session.flush()
        await append_list_context_journal_in_session(
            session,
            event_id,
            "disposition_commands",
            payload,
        )
        return DispositionOutboxRecord.model_validate(
            {
                "outbox_id": outbox.outbox_id,
                "writeback_id": outbox.writeback_id,
                "disposition_id": outbox.disposition_id,
                "action_id": outbox.action_id,
                "event_id": outbox.event_id,
                "closure_cycle": outbox.closure_cycle,
                "source_record_id": outbox.source_record_id,
                "source_locator_hash": outbox.source_locator_hash,
                "source_sequence": outbox.source_sequence,
                "intent_kind": outbox.intent_kind,
                "logical_slot": outbox.logical_slot,
                "idempotency_key": outbox.idempotency_key,
                "command_payload": outbox.command_payload,
                "command_payload_sha256": outbox.command_payload_sha256,
                "delivery_status": outbox.delivery_status,
            }
        )

    async def retry_writeback(self, writeback_id: str, *, operator: str) -> WritebackStatus:
        async with self._session_factory() as session:
            async with session.begin():
                outbox = await session.scalar(
                    select(orm.DispositionOutbox)
                    .where(orm.DispositionOutbox.writeback_id == writeback_id)
                    .with_for_update()
                )
                if outbox is None:
                    raise EventNotFoundError(
                        f"writeback not found: {writeback_id}",
                        details={"writeback_id": writeback_id},
                    )
                latest = (
                    WritebackStatus(outbox.latest_writeback_status)
                    if outbox.latest_writeback_status
                    else WritebackStatus.PENDING
                )
                if latest is WritebackStatus.UNKNOWN:
                    adapter = self._resolve_adapter(outbox)
                    if not adapter.capabilities().supports_lookup_by_idempotency:
                        raise WritebackConflictError(
                            "UNKNOWN writeback must be verified before retry",
                            details={"writeback_id": writeback_id},
                        )
                validate_outbox_delivery_transition(
                    OutboxDeliveryStatus(outbox.delivery_status),
                    OutboxDeliveryStatus.READY,
                )
                outbox.delivery_status = OutboxDeliveryStatus.READY.value
                outbox.locked_by = None
                outbox.locked_at = None
                outbox.lease_expires_at = None
                outbox.next_retry_at = None
                outbox.updated_at = datetime.now(UTC)
                session.add(
                    orm.EventAuditLog(
                        event_id=outbox.event_id,
                        from_status=outbox.latest_writeback_status,
                        to_status=WritebackStatus.PENDING.value,
                        operator=operator,
                        reason="retry_writeback:re-enqueued",
                    )
                )
        return WritebackStatus.PENDING

    async def resolve_writeback(
        self,
        writeback_id: str,
        resolution: str,
        *,
        principal: str,
        comment: str,
        evidence_ref: str | None = None,
    ) -> WritebackStatus:
        if resolution not in {"manual_confirmed", "mark_failed", "abandon"}:
            raise ValidationError(
                "unsupported writeback resolution",
                details={"resolution": resolution},
            )
        if resolution == "manual_confirmed" and not evidence_ref:
            raise ValidationError(
                "manual_confirmed requires evidence_ref",
                details={"writeback_id": writeback_id},
            )
        target = (
            WritebackStatus.CONFIRMED
            if resolution == "manual_confirmed"
            else WritebackStatus.FAILED
        )
        async with self._session_factory() as session:
            async with session.begin():
                outbox = await session.scalar(
                    select(orm.DispositionOutbox)
                    .where(orm.DispositionOutbox.writeback_id == writeback_id)
                    .with_for_update()
                )
                if outbox is None:
                    raise EventNotFoundError(
                        f"writeback not found: {writeback_id}",
                        details={"writeback_id": writeback_id},
                    )
                current_status = WritebackStatus(
                    outbox.latest_writeback_status or WritebackStatus.UNKNOWN.value
                )
                validate_writeback_status_transition(
                    current_status,
                    target,
                    evidence_adjudication=True,
                )
                await self._append_receipt(
                    session,
                    outbox,
                    status=target,
                    confirmation_evidence=(
                        ConfirmationEvidence.MANUAL_CONFIRMED
                        if target is WritebackStatus.CONFIRMED
                        else None
                    ),
                    provider_message=comment,
                )
                outbox.latest_writeback_status = target.value
                outbox.delivery_status = OutboxDeliveryStatus.DELIVERED.value
                action = await session.get(orm.Action, outbox.action_id, with_for_update=True)
                if action is not None:
                    action.writeback_status = target.value
                event_id = outbox.event_id
        await self._sync_writeback_summary(event_id)
        await self._maybe_resume(event_id)
        if self._bus is not None:
            await self._bus.publish_event(
                event_id,
                "writeback_updated",
                {"writeback_id": writeback_id, "status": target.value},
            )
        return target

    async def get_writeback(
        self, writeback_id: str
    ) -> tuple[DispositionOutboxRecord, DispositionReceipt | None]:
        async with self._session_factory() as session:
            outbox = await session.scalar(
                select(orm.DispositionOutbox).where(
                    orm.DispositionOutbox.writeback_id == writeback_id
                )
            )
            if outbox is None:
                raise EventNotFoundError(
                    f"writeback not found: {writeback_id}",
                    details={"writeback_id": writeback_id},
                )
            receipt = await session.scalar(
                select(orm.DispositionReceipt)
                .where(orm.DispositionReceipt.writeback_id == writeback_id)
                .order_by(orm.DispositionReceipt.sequence.desc())
                .limit(1)
            )
        record = DispositionOutboxRecord.model_validate(
            {
                "outbox_id": outbox.outbox_id,
                "writeback_id": outbox.writeback_id,
                "disposition_id": outbox.disposition_id,
                "action_id": outbox.action_id,
                "event_id": outbox.event_id,
                "closure_cycle": outbox.closure_cycle,
                "source_record_id": outbox.source_record_id,
                "source_locator_hash": outbox.source_locator_hash,
                "source_sequence": outbox.source_sequence,
                "intent_kind": outbox.intent_kind,
                "logical_slot": outbox.logical_slot,
                "idempotency_key": outbox.idempotency_key,
                "command_payload": outbox.command_payload,
                "command_payload_sha256": outbox.command_payload_sha256,
                "delivery_status": outbox.delivery_status,
                "latest_writeback_status": outbox.latest_writeback_status,
            }
        )
        parsed_receipt = None
        if receipt is not None:
            parsed_receipt = DispositionReceipt.model_validate(
                {
                    "writeback_id": receipt.writeback_id,
                    "sequence": receipt.sequence,
                    "disposition_id": receipt.disposition_id,
                    "action_id": receipt.action_id,
                    "source_record_id": receipt.source_record_id,
                    "status": receipt.status,
                    "confirmation_evidence": receipt.confirmation_evidence,
                    "provider_record_id": receipt.provider_record_id,
                    "provider_job_id": receipt.provider_job_id,
                    "provider_code": receipt.provider_code,
                    "provider_message": receipt.provider_message,
                    "observed_at": receipt.observed_at,
                    "submitted_at": receipt.submitted_at,
                    "confirmed_at": receipt.confirmed_at,
                    "target_results": receipt.target_results or [],
                    "raw_result": receipt.raw_result or {},
                    "truncated": receipt.truncated,
                    "simulated": receipt.simulated,
                }
            )
        return record, parsed_receipt

    async def list_event_dispositions(
        self, event_id: str
    ) -> list[tuple[DispositionCommand, WritebackStatus | None]]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(orm.DispositionOutbox)
                    .where(orm.DispositionOutbox.event_id == event_id)
                    .order_by(orm.DispositionOutbox.created_at.asc())
                )
            ).all()
        items: list[tuple[DispositionCommand, WritebackStatus | None]] = []
        for row in rows:
            command = DispositionCommand.model_validate(row.command_payload)
            status = (
                WritebackStatus(row.latest_writeback_status)
                if row.latest_writeback_status
                else None
            )
            items.append((command, status))
        return items

    async def get_disposition(
        self, disposition_id: str
    ) -> tuple[DispositionCommand, WritebackStatus | None]:
        async with self._session_factory() as session:
            outbox = await session.scalar(
                select(orm.DispositionOutbox).where(
                    orm.DispositionOutbox.disposition_id == disposition_id
                )
            )
        if outbox is None:
            raise EventNotFoundError(
                f"disposition not found: {disposition_id}",
                details={"disposition_id": disposition_id},
            )
        command = DispositionCommand.model_validate(outbox.command_payload)
        status = (
            WritebackStatus(outbox.latest_writeback_status)
            if outbox.latest_writeback_status
            else None
        )
        return command, status

    async def process_ready_outboxes(self, *, limit: int = 10) -> int:
        return await OutboxWorker(self).run_once(limit=limit)

    async def _deliver_outbox(self, outbox_id: str) -> None:
        command: DispositionCommand
        receipt: DispositionReceipt
        event_id: str
        async with self._session_factory() as session:
            async with session.begin():
                outbox = await session.scalar(
                    select(orm.DispositionOutbox)
                    .where(orm.DispositionOutbox.outbox_id == outbox_id)
                    .with_for_update()
                )
                if outbox is None:
                    return
                if OutboxDeliveryStatus(outbox.delivery_status) not in {
                    OutboxDeliveryStatus.READY,
                    OutboxDeliveryStatus.LEASED,
                    OutboxDeliveryStatus.WAITING_RETRY,
                }:
                    return
                command = DispositionCommand.model_validate(outbox.command_payload)
                await self._guard.validate(
                    command,
                    {
                        "event_id": outbox.event_id,
                        "source_locator": command.source_locator,
                        "approved_action_ids": [command.action_id],
                    },
                )
                adapter = self._resolve_adapter(outbox)
                adapter.validate_command(command)
                receipt = await adapter.submit(command)
                await self._append_receipt(session, outbox, receipt=receipt)
                outbox.latest_writeback_status = receipt.status.value
                outbox.delivery_status = OutboxDeliveryStatus.DELIVERED.value
                outbox.delivered_at = datetime.now(UTC)
                action = await session.get(orm.Action, outbox.action_id, with_for_update=True)
                if action is not None:
                    action.writeback_status = receipt.status.value
                    await self._apply_action_terminal_from_receipt(session, action, receipt)
                event_id = outbox.event_id
                writeback_id = outbox.writeback_id
        await self._sync_writeback_summary(event_id)
        await self._maybe_resume(event_id)
        if self._bus is not None:
            await self._bus.publish_event(
                event_id,
                "disposition_submitted",
                {
                    "disposition_id": command.disposition_id,
                    "intent_kind": command.intent_kind.value,
                },
            )
            await self._bus.publish_event(
                event_id,
                "writeback_updated",
                {"writeback_id": writeback_id, "status": receipt.status.value},
            )

    async def _append_receipt(
        self,
        session: AsyncSession,
        outbox: orm.DispositionOutbox,
        *,
        receipt: DispositionReceipt | None = None,
        status: WritebackStatus | None = None,
        confirmation_evidence: ConfirmationEvidence | None = None,
        provider_message: str | None = None,
    ) -> DispositionReceipt:
        seq_row = await session.scalar(
            select(orm.DispositionReceipt.sequence)
            .where(orm.DispositionReceipt.writeback_id == outbox.writeback_id)
            .order_by(orm.DispositionReceipt.sequence.desc())
            .limit(1)
        )
        sequence = int(seq_row or 0) + 1
        if receipt is not None:
            parsed = receipt.model_copy(
                update={"sequence": sequence, "writeback_id": outbox.writeback_id}
            )
        else:
            assert status is not None
            now = datetime.now(UTC)
            parsed = DispositionReceipt(
                writeback_id=outbox.writeback_id,
                sequence=sequence,
                disposition_id=outbox.disposition_id,
                action_id=outbox.action_id,
                source_record_id=outbox.source_record_id,
                status=status,
                confirmation_evidence=confirmation_evidence,
                provider_message=provider_message,
                observed_at=now,
                submitted_at=now,
                confirmed_at=now if status is WritebackStatus.CONFIRMED else None,
            )
        session.add(
            orm.DispositionReceipt(
                writeback_id=parsed.writeback_id,
                sequence=sequence,
                disposition_id=parsed.disposition_id,
                action_id=parsed.action_id,
                source_record_id=parsed.source_record_id,
                status=parsed.status.value,
                confirmation_evidence=(
                    parsed.confirmation_evidence.value
                    if parsed.confirmation_evidence is not None
                    else None
                ),
                provider_record_id=parsed.provider_record_id,
                provider_job_id=parsed.provider_job_id,
                provider_code=parsed.provider_code,
                provider_message=parsed.provider_message,
                observed_at=parsed.observed_at,
                submitted_at=parsed.submitted_at,
                confirmed_at=parsed.confirmed_at,
                target_results=[item.model_dump(mode="json") for item in parsed.target_results],
                raw_result=parsed.raw_result,
                truncated=parsed.truncated,
                simulated=parsed.simulated,
            )
        )
        await append_list_context_journal_in_session(
            session,
            outbox.event_id,
            "disposition_receipts",
            parsed.model_dump(mode="json"),
        )
        return parsed

    async def _apply_action_terminal_from_receipt(
        self,
        session: AsyncSession,
        action: orm.Action,
        receipt: DispositionReceipt,
    ) -> None:
        from app.models.enums import ActionCategory, ActionStatus
        from app.models.workflow import validate_action_status_transition

        current = ActionStatus(action.status)
        if current is not ActionStatus.EXECUTING:
            return
        if receipt.status in {WritebackStatus.CONFIRMED, WritebackStatus.ACCEPTED}:
            target = ActionStatus.SUCCESS
        elif receipt.status is WritebackStatus.PARTIAL:
            target = ActionStatus.PARTIAL_SUCCESS
        elif receipt.status is WritebackStatus.UNKNOWN:
            target = ActionStatus.UNKNOWN
        else:
            target = ActionStatus.FAILED
        validate_action_status_transition(
            ActionCategory(action.action_category),
            current,
            target,
        )
        action.status = target.value
        action.executed_at = datetime.now(UTC)

    def _resolve_adapter(self, outbox: orm.DispositionOutbox) -> BaseDispositionAdapter:
        payload = outbox.command_payload or {}
        locator = payload.get("source_locator") or {}
        product = str(locator.get("source_product") or "mock_xdr")
        return self._adapters.get(product)

    async def _sync_writeback_summary(self, event_id: str) -> None:
        summary_payload: dict[str, Any] | None = None
        async with self._session_factory() as session:
            async with session.begin():
                se = await session.get(orm.SecurityEvent, event_id)
                if se is None:
                    return
                summary = await self._context_store._merge_writeback_summary(session, se)
                if summary is not None:
                    summary_payload = summary.model_dump(mode="json")
                    await append_context_journal_in_session(
                        session,
                        event_id,
                        "writeback_summary",
                        summary_payload,
                    )
        if summary_payload is not None:
            await self._context_store.set(event_id, "writeback_summary", summary_payload)

    async def _maybe_resume(self, event_id: str) -> None:
        async with self._session_factory() as session:
            substate_raw = await session.scalar(
                select(orm.EventContextJournal.value)
                .where(
                    orm.EventContextJournal.event_id == event_id,
                    orm.EventContextJournal.field_name == "execution_substate",
                )
                .order_by(orm.EventContextJournal.version.desc())
                .limit(1)
            )
        if isinstance(substate_raw, dict) and set(substate_raw) == {"_scalar"}:
            substate_raw = substate_raw["_scalar"]
        if substate_raw in {
            ExecutionSubstate.WAITING_WRITEBACK.value,
            ExecutionSubstate.WAITING_EXECUTION.value,
        }:
            await self._resume(event_id)


class OutboxWorker:
    def __init__(self, service: DispositionSyncService) -> None:
        self._service = service

    async def run_once(self, *, limit: int = 10) -> int:
        claimed = await self._claim_batch(limit=limit)
        for outbox_id in claimed:
            try:
                await self._service._deliver_outbox(outbox_id)
            except GuardrailViolationError:
                logger.warning("outbox delivery blocked by guard outbox=%s", outbox_id)
            except Exception:
                logger.exception("outbox delivery failed outbox=%s", outbox_id)
        return len(claimed)

    async def _claim_batch(self, *, limit: int) -> list[str]:
        now = datetime.now(UTC)
        claimed: list[str] = []
        async with self._service._session_factory() as session:
            async with session.begin():
                rows = (
                    await session.scalars(
                        select(orm.DispositionOutbox)
                        .where(
                            or_(
                                orm.DispositionOutbox.delivery_status.in_(
                                    (
                                        OutboxDeliveryStatus.READY.value,
                                        OutboxDeliveryStatus.WAITING_RETRY.value,
                                    )
                                ),
                                and_(
                                    orm.DispositionOutbox.delivery_status
                                    == OutboxDeliveryStatus.LEASED.value,
                                    orm.DispositionOutbox.lease_expires_at.is_not(None),
                                    orm.DispositionOutbox.lease_expires_at < now,
                                ),
                            )
                        )
                        .order_by(orm.DispositionOutbox.created_at.asc())
                        .limit(limit)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
                for row in rows:
                    current = OutboxDeliveryStatus(row.delivery_status)
                    if (
                        current is OutboxDeliveryStatus.LEASED
                        and row.lease_expires_at is not None
                        and row.lease_expires_at < now
                    ):
                        validate_outbox_delivery_transition(
                            OutboxDeliveryStatus.LEASED,
                            OutboxDeliveryStatus.WAITING_RETRY,
                        )
                        row.delivery_status = OutboxDeliveryStatus.WAITING_RETRY.value
                        row.locked_by = None
                        row.locked_at = None
                        row.lease_expires_at = None
                        current = OutboxDeliveryStatus.WAITING_RETRY
                    validate_outbox_delivery_transition(
                        current,
                        OutboxDeliveryStatus.LEASED,
                    )
                    row.delivery_status = OutboxDeliveryStatus.LEASED.value
                    row.locked_by = self._service._worker_id
                    row.locked_at = now
                    row.lease_expires_at = now + timedelta(seconds=_DEFAULT_LEASE_SECONDS)
                    claimed.append(row.outbox_id)
        return claimed


__all__ = ["DispositionSyncService", "OutboxWorker", "ResumeInvestigationHook"]
