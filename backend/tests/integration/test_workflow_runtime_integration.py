"""Database transaction tests for WorkflowRuntimeService (ISSUE-048)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import ValidationError
from app.db import models as orm
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    ExecutionSubstate,
    FinalVerdict,
    Severity,
    SourceObjectKind,
    WritebackReadiness,
)
from app.models.source import SourceReference
from app.orchestration.workflow_runtime import WorkflowRuntimeService
from app.services.context_service import append_context_journal_in_session
from app.services.event_service import EventService, IngestableSource

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("clean_state"),
]


async def _ready(_: str) -> WritebackReadiness:
    return WritebackReadiness.READY


def _reference(object_id: str) -> SourceReference:
    return SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="tenant-1",
        connector_id="conn-workflow-runtime",
        source_object_id=object_id,
        ingested_at=datetime.now(UTC),
    )


async def _seed_required_fp(
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
    object_id: str,
) -> str:
    event = await event_service.ingest_source_object(
        IngestableSource(
            reference=_reference(object_id),
            title="disposition-only fixture",
            event_type=EventType.OTHER,
            severity=Severity.MEDIUM,
            source_type="mock_xdr",
        )
    )
    async with session_factory() as session:
        async with session.begin():
            row = await session.get(orm.SecurityEvent, event.event_id, with_for_update=True)
            assert row is not None
            row.status = EventStatus.TRIAGING.value
            row.disposition_policy = DispositionPolicy.REQUIRED.value
            await append_context_journal_in_session(
                session,
                event.event_id,
                "false_positive_match",
                {"recommendation": "close_as_fp", "max_score": 0.88},
            )
    return event.event_id


@pytest.mark.asyncio
async def test_begin_disposition_only_is_atomic_and_idempotent(
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_required_fp(
        event_service,
        session_factory,
        "INC-workflow-runtime-fp",
    )
    runtime = WorkflowRuntimeService(
        session_factory,
        event_service=event_service,
        readiness_resolver=_ready,
    )

    await runtime.begin_disposition_only(event_id)
    await runtime.begin_disposition_only(event_id)

    async with session_factory() as session:
        row = await session.get(orm.SecurityEvent, event_id)
        assert row is not None
        assert row.final_verdict == FinalVerdict.FALSE_POSITIVE.value
        assert float(row.confidence) >= 0.88
        intent = await session.scalar(
            select(orm.EventContextJournal)
            .where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name == "disposition_only_intent",
            )
            .order_by(orm.EventContextJournal.version.desc())
            .limit(1)
        )
        assert intent is not None
        assert intent.value == {"_scalar": True}
        intent_count = await session.scalar(
            select(func.count()).where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name == "disposition_only_intent",
            )
        )
        assert intent_count == 1

    assert await runtime.read_disposition_only_intent(event_id) is True


@pytest.mark.asyncio
async def test_begin_disposition_only_rolls_back_all_fields_on_failure(
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_required_fp(
        event_service,
        session_factory,
        "INC-workflow-runtime-rollback",
    )

    class FailingEventService:
        async def apply_final_verdict_in_session(
            self,
            session: AsyncSession,
            event_id: str,
            verdict: FinalVerdict,
            *,
            operator: str | None = None,
        ) -> tuple[bool, Any, Any]:
            row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
            assert row is not None
            row.final_verdict = verdict.value
            row.confidence = 0.99
            await session.flush()
            raise RuntimeError("injected transaction failure")

        async def publish_final_verdict_mutation(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("must not publish after rollback")

        async def sync_event_summary_mutation(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("must not synchronize after rollback")

    runtime = WorkflowRuntimeService(
        session_factory,
        event_service=FailingEventService(),
        readiness_resolver=_ready,
    )
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        await runtime.begin_disposition_only(event_id)

    async with session_factory() as session:
        row = await session.get(orm.SecurityEvent, event_id)
        assert row is not None
        assert row.final_verdict == FinalVerdict.NONE.value
        assert float(row.confidence) < 0.99
        intent = await session.scalar(
            select(orm.EventContextJournal).where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name == "disposition_only_intent",
            )
        )
        assert intent is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "status", "readiness"),
    [
        (
            DispositionPolicy.NOT_REQUIRED,
            EventStatus.TRIAGING,
            WritebackReadiness.READY,
        ),
        (
            DispositionPolicy.REQUIRED,
            EventStatus.NEW,
            WritebackReadiness.READY,
        ),
        (
            DispositionPolicy.REQUIRED,
            EventStatus.TRIAGING,
            WritebackReadiness.CAPABILITY_UNSUPPORTED,
        ),
    ],
)
async def test_begin_disposition_only_rejects_untrusted_entry_conditions(
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
    policy: DispositionPolicy,
    status: EventStatus,
    readiness: WritebackReadiness,
) -> None:
    event_id = await _seed_required_fp(
        event_service,
        session_factory,
        f"INC-workflow-runtime-reject-{policy.value}-{status.value}-{readiness.value}",
    )
    async with session_factory() as session:
        async with session.begin():
            row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
            assert row is not None
            row.disposition_policy = policy.value
            row.status = status.value

    async def resolve(_: str) -> WritebackReadiness:
        return readiness

    runtime = WorkflowRuntimeService(
        session_factory,
        event_service=event_service,
        readiness_resolver=resolve,
    )
    with pytest.raises(ValidationError):
        await runtime.begin_disposition_only(event_id)

    async with session_factory() as session:
        row = await session.get(orm.SecurityEvent, event_id)
        assert row is not None
        assert row.final_verdict == FinalVerdict.NONE.value
        intent = await session.scalar(
            select(orm.EventContextJournal).where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name == "disposition_only_intent",
            )
        )
        assert intent is None


@pytest.mark.asyncio
async def test_execution_substate_rejects_forged_event_status(
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_required_fp(
        event_service,
        session_factory,
        "INC-workflow-runtime-forged-substate",
    )
    runtime = WorkflowRuntimeService(
        session_factory,
        event_service=event_service,
        readiness_resolver=_ready,
    )

    with pytest.raises(ValidationError, match="authoritative"):
        await runtime.set_execution_substate(
            event_id,
            ExecutionSubstate.WAITING_APPROVAL,
            event_status=EventStatus.WAITING_APPROVAL,
        )

    async with session_factory() as session:
        substate = await session.scalar(
            select(orm.EventContextJournal).where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name == "execution_substate",
            )
        )
        assert substate is None


@pytest.mark.asyncio
async def test_existing_fp_confidence_mutation_updates_version_and_audit(
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_required_fp(
        event_service,
        session_factory,
        "INC-workflow-runtime-confidence",
    )
    async with session_factory() as session:
        async with session.begin():
            row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
            assert row is not None
            row.final_verdict = FinalVerdict.FALSE_POSITIVE.value
            row.confidence = 0.2
            initial_version = row.row_version

    class TrackingEventService:
        def __init__(self) -> None:
            self.synced = 0

        async def apply_final_verdict_in_session(
            self,
            session: AsyncSession,
            event_id: str,
            verdict: FinalVerdict,
            *,
            operator: str | None = None,
        ) -> tuple[bool, Any, Any]:
            return await event_service.apply_final_verdict_in_session(
                session,
                event_id,
                verdict,
                operator=operator,
            )

        async def publish_final_verdict_mutation(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("confidence-only mutation must not publish verdict")

        async def sync_event_summary_mutation(
            self,
            event_id: str,
            *,
            result: Any,
            summary: Any,
        ) -> None:
            self.synced += 1
            await event_service.sync_event_summary_mutation(
                event_id,
                result=result,
                summary=summary,
            )

    tracking_service = TrackingEventService()
    runtime = WorkflowRuntimeService(
        session_factory,
        event_service=tracking_service,
        readiness_resolver=_ready,
    )
    await runtime.begin_disposition_only(event_id)

    async with session_factory() as session:
        row = await session.get(orm.SecurityEvent, event_id)
        assert row is not None
        assert float(row.confidence) == pytest.approx(0.88)
        assert row.row_version > initial_version
        audit = await session.scalar(
            select(orm.EventAuditLog).where(
                orm.EventAuditLog.event_id == event_id,
                orm.EventAuditLog.reason.like("disposition_only_confidence:%"),
            )
        )
        assert audit is not None
    assert tracking_service.synced == 1
