"""DispositionSyncService tests (ISSUE-059).

Requires Compose PostgreSQL (+ Redis for context). Run from ``backend/``:

    pytest tests/test_services/test_disposition_sync.py -v
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.adapters.mock_xdr import MockXDRDispositionAdapter
from app.adapters.registry import DispositionAdapterRegistry
from app.core.errors import WritebackConflictError
from app.core.guardrails import OutboundDispositionGuard
from app.data_generators.scenarios import build_scenario
from app.db import models as orm
from app.mock_xdr.api import create_app
from app.mock_xdr.state import MockXDRState
from app.models.action import Action
from app.models.disposition import SourceObjectLocator
from app.models.enums import (
    ActionCategory,
    ActionExecutionPhase,
    ActionLevel,
    ActionStatus,
    DispositionPolicy,
    EventStatus,
    EventType,
    ExecutionOwner,
    ExecutionSubstate,
    FinalVerdict,
    Severity,
    SourceObjectKind,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.ids import new_disposition_id
from app.models.source import SourceReference
from app.services.context_service import EventContextStore, append_context_journal_in_session
from app.services.disposition_command_factory import DispositionCommandFactory
from app.services.disposition_sync_service import DispositionSyncService

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return cfg


@pytest.fixture(scope="module")
def migrated() -> None:
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)

    async def _probe() -> None:
        try:
            async with engine.connect() as conn:
                await conn.execute(select(1))
        except Exception as exc:  # noqa: BLE001
            await engine.dispose()
            pytest.skip(f"PostgreSQL not reachable: {exc}")

    import asyncio

    asyncio.run(_probe())
    command.upgrade(_alembic_config(), "head")
    asyncio.run(engine.dispose())


@pytest_asyncio.fixture
async def session_factory(
    migrated: None,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            await conn.execute(select(1))
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def redis_client():
    from app.core.redis_client import RedisClient

    client = RedisClient(url=REDIS_URL)
    if not await client.ping():
        await client.aclose()
        pytest.skip("Redis not reachable; start Compose redis first")
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def store(
    session_factory: async_sessionmaker[AsyncSession],
    redis_client,
) -> EventContextStore:
    return EventContextStore(redis_client, session_factory)


@pytest_asyncio.fixture
async def mock_xdr_client() -> AsyncIterator[httpx.AsyncClient]:
    state = MockXDRState()
    state.load_scenario(build_scenario("insider_data_exfiltration", seed=42))
    app = create_app(state=state)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://mock-xdr",
        timeout=30.0,
    ) as client:
        yield client


@pytest_asyncio.fixture
async def cleanup(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    yield
    async with session_factory() as session:
        async with session.begin():
            for table in (
                orm.EventAuditLog,
                orm.EventContextJournal,
                orm.EventContextFieldVersion,
                orm.ActionTargetResult,
                orm.ActionExecutionJob,
                orm.DispositionReceipt,
                orm.DispositionOutbox,
                orm.Action,
                orm.Evidence,
                orm.Report,
                orm.SourceEventLink,
                orm.SourceObject,
                orm.SecurityEvent,
            ):
                await session.execute(delete(table))


def _sfx() -> str:
    return uuid.uuid4().hex[:8]


def _locator(*, object_id: str = "88442201") -> SourceObjectLocator:
    return SourceObjectLocator(
        source_product="mock_xdr",
        source_tenant_id="tenant-demo",
        connector_id="conn-disposition",
        source_kind=SourceObjectKind.INCIDENT,
        source_object_id=object_id,
    )


async def _seed_event_action_source(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
) -> tuple[str, str, str, SourceObjectLocator]:
    sfx = _sfx()
    event_id = f"evt-sync-{sfx}"
    action_id = f"act-{sfx}"
    connector_id = "conn-disposition"
    source_record_id = f"src-{sfx}"
    object_id = f"INC-{sfx}"
    locator = _locator(object_id=object_id)
    ref = SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="tenant-demo",
        connector_id=connector_id,
        source_object_id=object_id,
        ingested_at=datetime.now(UTC),
    )
    async with session_factory() as session:
        async with session.begin():
            existing = await session.get(orm.SourceConnector, connector_id)
            if existing is None:
                session.add(
                    orm.SourceConnector(
                        connector_id=connector_id,
                        source_product="mock_xdr",
                        display_name="Mock XDR",
                    )
                )
            session.add(
                orm.SourceObject(
                    source_record_id=source_record_id,
                    source_product="mock_xdr",
                    source_tenant_id="tenant-demo",
                    connector_id=connector_id,
                    source_kind=SourceObjectKind.INCIDENT.value,
                    source_object_id=object_id,
                    current_concurrency_token="tok-1",
                    next_outbox_sequence=0,
                )
            )
            session.add(
                orm.SecurityEvent(
                    event_id=event_id,
                    event_type=EventType.OTHER.value,
                    title="sync-test",
                    description="",
                    status=EventStatus.EXECUTING_RESPONSE.value,
                    severity=Severity.HIGH.value,
                    risk_score=80,
                    confidence=0.9,
                    final_verdict=FinalVerdict.NONE.value,
                    creation_source_ref=ref.model_dump(mode="json"),
                    source_reference_snapshots=[ref.model_dump(mode="json")],
                    disposition_policy=DispositionPolicy.REQUIRED.value,
                    disposition_source_ref=locator.model_dump(mode="json"),
                    occurred_at=datetime.now(UTC),
                )
            )
            session.add(
                orm.Action(
                    action_id=action_id,
                    event_id=event_id,
                    plan_revision=1,
                    action_fingerprint=f"fp-{sfx}",
                    action_category=ActionCategory.RESPONSE.value,
                    action_name="block ip",
                    tool_name="block_ip",
                    action_level=ActionLevel.L2.value,
                    execution_phase=ActionExecutionPhase.IMMEDIATE.value,
                    status=ActionStatus.EXECUTING.value,
                    execution_owner=ExecutionOwner.XDR_MANAGED.value,
                    target_type="ip",
                    target="203.0.113.88",
                    writeback_required=True,
                    writeback_applicable=True,
                    writeback_readiness=WritebackReadiness.READY.value,
                    disposition_source_ref=locator.model_dump(mode="json"),
                    idempotency_key=f"idem-{sfx}",
                )
            )
    async with session_factory() as session:
        row = await session.get(orm.SecurityEvent, event_id)
        assert row is not None
        from app.services.context_service import event_summary_from_security_event

        await store.init_context(event_id, event_summary_from_security_event(row))
    return event_id, action_id, source_record_id, locator


def _sync_service(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    client: httpx.AsyncClient,
    *,
    resume: AsyncMock | None = None,
) -> DispositionSyncService:
    registry = DispositionAdapterRegistry()
    adapter = MockXDRDispositionAdapter(
        client=client,
        read_token="mock-read-token",
        write_token="mock-write-token",
    )
    registry.register("mock_xdr", adapter)
    return DispositionSyncService(
        session_factory,
        context_store=store,
        adapter_registry=registry,
        outbound_guard=OutboundDispositionGuard(),
        resume_investigation=resume,
    )


@pytest.mark.asyncio
async def test_enqueue_and_deliver_outbox(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    mock_xdr_client: httpx.AsyncClient,
    cleanup: None,
) -> None:
    event_id, action_id, source_record_id, locator = await _seed_event_action_source(
        session_factory, store
    )
    sync = _sync_service(session_factory, store, mock_xdr_client)
    factory = DispositionCommandFactory()
    action = Action.model_validate(
        {
            "action_id": action_id,
            "event_id": event_id,
            "plan_revision": 1,
            "action_fingerprint": "fp-test",
            "action_category": ActionCategory.RESPONSE,
            "action_name": "block ip",
            "tool_name": "block_ip",
            "action_level": ActionLevel.L2,
            "execution_owner": ExecutionOwner.XDR_MANAGED,
            "status": ActionStatus.EXECUTING,
            "target": "203.0.113.88",
            "writeback_required": True,
            "writeback_applicable": True,
            "writeback_readiness": WritebackReadiness.READY,
            "disposition_source_ref": locator,
            "idempotency_key": f"idem-{_sfx()}",
        }
    )
    command = factory.build_entity_action_submit(
        action,
        source_locator=locator,
        source_concurrency_token="tok-1",
        operator_id="ActionExecutionService",
        disposition_id=new_disposition_id(),
        writeback_id="pending",
        closure_cycle=1,
        entity_action_code="block_ip",
    )
    async with session_factory() as session:
        async with session.begin():
            record = await sync.enqueue_command(
                session,
                command=command,
                event_id=event_id,
                source_record_id=source_record_id,
            )
    assert record.intent_kind.value == "entity_action_submit"
    delivered = await sync.process_ready_outboxes(limit=1)
    assert delivered == 1
    async with session_factory() as session:
        action_row = await session.get(orm.Action, action_id)
        assert action_row is not None
        assert action_row.status == ActionStatus.SUCCESS.value
        receipt = await session.scalar(
            select(orm.DispositionReceipt).where(
                orm.DispositionReceipt.writeback_id == record.writeback_id
            )
        )
        assert receipt is not None


@pytest.mark.asyncio
async def test_retry_unknown_writeback_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    mock_xdr_client: httpx.AsyncClient,
    cleanup: None,
) -> None:
    event_id, action_id, source_record_id, locator = await _seed_event_action_source(
        session_factory, store
    )
    sync = _sync_service(session_factory, store, mock_xdr_client)
    writeback_id = f"wbk-{_sfx()}"
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.DispositionOutbox(
                    outbox_id=f"obx-{_sfx()}",
                    writeback_id=writeback_id,
                    disposition_id=f"disp-{_sfx()}",
                    action_id=action_id,
                    event_id=event_id,
                    closure_cycle=1,
                    source_record_id=source_record_id,
                    source_locator_hash="hash",
                    source_sequence=1,
                    intent_kind="entity_action_submit",
                    logical_slot="default",
                    idempotency_key=f"idem-{_sfx()}",
                    command_payload={"source_locator": locator.model_dump(mode="json")},
                    command_payload_sha256="deadbeef",
                    delivery_status="delivered",
                    latest_writeback_status=WritebackStatus.UNKNOWN.value,
                )
            )
    with pytest.raises(WritebackConflictError):
        await sync.retry_writeback(writeback_id, operator="operator-1")


@pytest.mark.asyncio
async def test_resolve_writeback_manual_confirmed(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    mock_xdr_client: httpx.AsyncClient,
    cleanup: None,
) -> None:
    event_id, action_id, source_record_id, locator = await _seed_event_action_source(
        session_factory, store
    )
    writeback_id = f"wbk-{_sfx()}"
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.DispositionOutbox(
                    outbox_id=f"obx-{_sfx()}",
                    writeback_id=writeback_id,
                    disposition_id=f"disp-{_sfx()}",
                    action_id=action_id,
                    event_id=event_id,
                    closure_cycle=1,
                    source_record_id=source_record_id,
                    source_locator_hash="hash",
                    source_sequence=1,
                    intent_kind="entity_action_submit",
                    logical_slot="default",
                    idempotency_key=f"idem-{_sfx()}",
                    command_payload={"source_locator": locator.model_dump(mode="json")},
                    command_payload_sha256="deadbeef",
                    delivery_status="delivered",
                    latest_writeback_status=WritebackStatus.UNKNOWN.value,
                )
            )
    sync = _sync_service(session_factory, store, mock_xdr_client)
    status = await sync.resolve_writeback(
        writeback_id,
        "manual_confirmed",
        principal="admin-1",
        comment="ticket-123",
        evidence_ref="evidence://ticket-123",
    )
    assert status is WritebackStatus.CONFIRMED


@pytest.mark.asyncio
async def test_resume_hook_called_on_terminal_writeback(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    mock_xdr_client: httpx.AsyncClient,
    cleanup: None,
) -> None:
    event_id, action_id, source_record_id, locator = await _seed_event_action_source(
        session_factory, store
    )
    resume = AsyncMock()
    sync = _sync_service(session_factory, store, mock_xdr_client, resume=resume)
    async with session_factory() as session:
        async with session.begin():
            await append_context_journal_in_session(
                session,
                event_id,
                "execution_substate",
                ExecutionSubstate.WAITING_WRITEBACK.value,
            )
            writeback_id = f"wbk-{_sfx()}"
            session.add(
                orm.DispositionOutbox(
                    outbox_id=f"obx-{_sfx()}",
                    writeback_id=writeback_id,
                    disposition_id=f"disp-{_sfx()}",
                    action_id=action_id,
                    event_id=event_id,
                    closure_cycle=1,
                    source_record_id=source_record_id,
                    source_locator_hash="hash",
                    source_sequence=1,
                    intent_kind="entity_action_submit",
                    logical_slot="default",
                    idempotency_key=f"idem-{_sfx()}",
                    command_payload={"source_locator": locator.model_dump(mode="json")},
                    command_payload_sha256="deadbeef",
                    delivery_status="delivered",
                    latest_writeback_status=WritebackStatus.UNKNOWN.value,
                )
            )
    await sync.resolve_writeback(
        writeback_id,
        "manual_confirmed",
        principal="admin-1",
        comment="done",
        evidence_ref="evidence://ok",
    )
    resume.assert_awaited_once_with(event_id)
