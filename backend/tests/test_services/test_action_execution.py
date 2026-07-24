"""ActionExecutionService tests (ISSUE-059).

Requires Compose PostgreSQL (+ Redis for context). Run from ``backend/``:

    pytest tests/test_services/test_action_execution.py -v
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

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
from app.core.errors import InvalidStateTransitionError, ValidationError
from app.core.guardrails import OutboundDispositionGuard
from app.data_generators.scenarios import build_scenario
from app.db import models as orm
from app.mock_xdr.api import create_app
from app.mock_xdr.state import MockXDRState
from app.models.action import TERMINAL_DISPOSITION_TOOL, Action
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
    FinalVerdict,
    Severity,
    SourceObjectKind,
    WritebackReadiness,
)
from app.models.source import SourceReference
from app.services.action_execution_service import ActionExecutionService
from app.services.context_service import EventContextStore, event_summary_from_security_event
from app.services.degraded_flag_service import DegradedFlagService
from app.services.disposition_sync_service import DispositionSyncService
from app.services.event_audit_log_service import EventAuditLogService
from app.services.state_machine_service import StateMachineService
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from tests.test_services._mock_xdr_test_helpers import (
    SCENARIO_INCIDENT_ID,
    fetch_mock_concurrency_token,
)

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
async def disposition_sync(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    mock_xdr_client: httpx.AsyncClient,
) -> DispositionSyncService:
    registry = DispositionAdapterRegistry()
    adapter = MockXDRDispositionAdapter(
        client=mock_xdr_client,
        read_token="mock-read-token",
        write_token="mock-write-token",
    )
    registry.register("mock_xdr", adapter)
    return DispositionSyncService(
        session_factory,
        context_store=store,
        adapter_registry=registry,
        outbound_guard=OutboundDispositionGuard(),
    )


@pytest_asyncio.fixture
async def tool_executor() -> ToolExecutor:
    registry = ToolRegistry()
    await registry.auto_discover_for_mode(tool_mode="mock")
    return ToolExecutor(registry=registry)


@pytest_asyncio.fixture
async def state_machine(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    redis_client,
) -> StateMachineService:
    from app.core.event_bus import EventBus

    return StateMachineService(
        session_factory,
        store,
        event_bus=EventBus(redis_client),
        audit_log=EventAuditLogService(session_factory),
        degraded_flags=DegradedFlagService(store, session_factory),
    )


@pytest_asyncio.fixture
async def execution_service(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    disposition_sync: DispositionSyncService,
    tool_executor: ToolExecutor,
    state_machine: StateMachineService,
) -> ActionExecutionService:
    return ActionExecutionService(
        session_factory,
        disposition_sync=disposition_sync,
        tool_executor=tool_executor,
        state_machine=state_machine,
        context_store=store,
    )


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


def _ref(*, object_id: str) -> SourceReference:
    return SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="tenant-demo",
        connector_id="conn-disposition",
        source_object_id=object_id,
        ingested_at=datetime.now(UTC),
    )


def _action_model(**overrides: object) -> Action:
    locator = _locator()
    base = {
        "action_id": f"act-{_sfx()}",
        "event_id": "evt-placeholder",
        "plan_revision": 1,
        "action_fingerprint": f"fp-{_sfx()}",
        "action_category": ActionCategory.RESPONSE,
        "action_name": "block ip",
        "tool_name": "block_ip",
        "action_level": ActionLevel.L2,
        "execution_owner": ExecutionOwner.XDR_MANAGED,
        "status": ActionStatus.APPROVED,
        "target_type": "ip",
        "target": "203.0.113.88",
        "writeback_required": True,
        "writeback_applicable": True,
        "writeback_readiness": WritebackReadiness.READY,
        "disposition_source_ref": locator,
        "idempotency_key": f"idem-{_sfx()}",
    }
    base.update(overrides)
    return Action.model_validate(base)


async def _seed_connector_and_source(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    object_id: str = SCENARIO_INCIDENT_ID,
    mock_xdr_client: httpx.AsyncClient | None = None,
) -> str:
    sfx = _sfx()
    connector_id = "conn-disposition"
    source_record_id = f"src-{sfx}"
    concurrency_token = "tok-1"
    if mock_xdr_client is not None and object_id == SCENARIO_INCIDENT_ID:
        concurrency_token = await fetch_mock_concurrency_token(mock_xdr_client, object_id=object_id)
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
                    current_concurrency_token=concurrency_token,
                    next_outbox_sequence=0,
                )
            )
    return source_record_id


async def _create_event(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    *,
    status: EventStatus = EventStatus.EXECUTING_RESPONSE,
    object_id: str | None = None,
) -> str:
    sfx = _sfx()
    event_id = f"evt-exec-{sfx}"
    resolved_object_id = object_id or f"INC-{sfx}"
    ref = _ref(object_id=resolved_object_id)
    locator = _locator(object_id=resolved_object_id)
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SecurityEvent(
                    event_id=event_id,
                    event_type=EventType.OTHER.value,
                    title="execution-test",
                    description="",
                    status=status.value,
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
    async with session_factory() as session:
        row = await session.get(orm.SecurityEvent, event_id)
        assert row is not None
        await store.init_context(event_id, event_summary_from_security_event(row))
    return event_id


async def _insert_action(
    session_factory: async_sessionmaker[AsyncSession],
    event_id: str,
    action: Action,
) -> Action:
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.Action(
                    action_id=action.action_id,
                    event_id=event_id,
                    plan_revision=action.plan_revision,
                    action_fingerprint=action.action_fingerprint,
                    action_category=action.action_category.value,
                    action_name=action.action_name,
                    tool_name=action.tool_name,
                    action_level=action.action_level.value,
                    execution_phase=action.execution_phase.value,
                    activation_condition=action.activation_condition,
                    status=action.status.value,
                    execution_owner=(
                        action.execution_owner.value if action.execution_owner else None
                    ),
                    target_type=action.target_type,
                    target=action.target,
                    parameters=action.parameters or {},
                    writeback_required=action.writeback_required,
                    writeback_applicable=action.writeback_applicable,
                    writeback_readiness=action.writeback_readiness.value,
                    disposition_source_ref=(
                        action.disposition_source_ref.model_dump(mode="json")
                        if action.disposition_source_ref
                        else None
                    ),
                    idempotency_key=action.idempotency_key,
                    reason=action.reason,
                )
            )
    return action.model_copy(update={"event_id": event_id})


@pytest.mark.asyncio
async def test_empty_immediate_transitions_to_verifying(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    event_id = await _create_event(session_factory, store)
    await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            execution_phase=ActionExecutionPhase.POST_VERIFY,
            tool_name=TERMINAL_DISPOSITION_TOOL,
            activation_condition="after_effect_resolution",
            execution_owner=ExecutionOwner.XDR_MANAGED,
        ),
    )
    summary = await execution_service.execute_plan(event_id)
    async with session_factory() as session:
        event = await session.get(orm.SecurityEvent, event_id)
        assert event is not None
        assert event.status == EventStatus.VERIFYING.value
    assert summary.action_counts.get(ActionStatus.APPROVED.value, 0) == 1


@pytest.mark.asyncio
async def test_xdr_managed_execute_plan_submits_outbox(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    mock_xdr_client: httpx.AsyncClient,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    oid = SCENARIO_INCIDENT_ID
    await _seed_connector_and_source(
        session_factory, object_id=oid, mock_xdr_client=mock_xdr_client
    )
    event_id = await _create_event(session_factory, store, object_id=oid)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            execution_owner=ExecutionOwner.XDR_MANAGED,
            disposition_source_ref=_locator(object_id=oid),
        ),
    )
    summary = await execution_service.execute_plan(event_id)
    async with session_factory() as session:
        row = await session.get(orm.Action, action.action_id)
        assert row is not None
        assert row.status in {
            ActionStatus.SUCCESS.value,
            ActionStatus.EXECUTING.value,
        }
        outboxes = (
            await session.scalars(
                select(orm.DispositionOutbox).where(orm.DispositionOutbox.event_id == event_id)
            )
        ).all()
        assert len(outboxes) == 1
        assert outboxes[0].intent_kind == "entity_action_submit"
    assert summary.jobs == []


@pytest.mark.asyncio
async def test_post_verify_action_rejected_by_execute_action(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            execution_phase=ActionExecutionPhase.POST_VERIFY,
            tool_name=TERMINAL_DISPOSITION_TOOL,
            activation_condition="after_effect_resolution",
        ),
    )
    with pytest.raises(ValidationError, match="POST_VERIFY"):
        await execution_service.execute_action(action.action_id)


@pytest.mark.asyncio
async def test_resolve_unknown_action(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, status=ActionStatus.UNKNOWN),
    )
    resolved = await execution_service.resolve_unknown(
        action.action_id,
        "mark_success",
        principal="admin-1",
        comment="verified offline",
    )
    assert resolved.status is ActionStatus.SUCCESS


@pytest.mark.asyncio
async def test_resolve_unknown_requires_unknown_status(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, status=ActionStatus.APPROVED),
    )
    with pytest.raises(InvalidStateTransitionError):
        await execution_service.resolve_unknown(
            action.action_id,
            "mark_failed",
            principal="admin-1",
            comment="n/a",
        )


@pytest.mark.asyncio
async def test_resolve_unknown_partial_success(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, status=ActionStatus.UNKNOWN),
    )
    resolved = await execution_service.resolve_unknown(
        action.action_id,
        "partial_success",
        principal="admin-1",
        comment="partially effective",
    )
    assert resolved.status is ActionStatus.PARTIAL_SUCCESS


@pytest.mark.asyncio
async def test_claim_rejected_when_writeback_not_ready(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    oid = f"INC-{_sfx()}"
    await _seed_connector_and_source(session_factory, object_id=oid)
    event_id = await _create_event(session_factory, store, object_id=oid)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            execution_owner=ExecutionOwner.XDR_MANAGED,
            disposition_source_ref=_locator(object_id=oid),
            writeback_readiness=WritebackReadiness.CONNECTOR_UNAVAILABLE,
        ),
    )
    with pytest.raises(ValidationError, match="writeback readiness blocks"):
        await execution_service.execute_action(action.action_id)


@pytest.mark.asyncio
async def test_direct_tool_enqueue_execution_result_record(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    mock_xdr_client: httpx.AsyncClient,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    oid = SCENARIO_INCIDENT_ID
    await _seed_connector_and_source(
        session_factory, object_id=oid, mock_xdr_client=mock_xdr_client
    )
    event_id = await _create_event(session_factory, store, object_id=oid)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            execution_owner=ExecutionOwner.DIRECT_TOOL,
            disposition_source_ref=_locator(object_id=oid),
            target="203.0.113.88",
            parameters={"target_type": "ip", "target": "203.0.113.88"},
        ),
    )
    summary = await execution_service.execute_plan(event_id)
    async with session_factory() as session:
        jobs = (
            await session.scalars(
                select(orm.ActionExecutionJob).where(
                    orm.ActionExecutionJob.action_id == action.action_id
                )
            )
        ).all()
        outboxes = (
            await session.scalars(
                select(orm.DispositionOutbox).where(orm.DispositionOutbox.event_id == event_id)
            )
        ).all()
    assert jobs
    assert any(o.intent_kind == "execution_result_record" for o in outboxes)
    assert summary.writeback_ids


@pytest.mark.asyncio
async def test_concurrent_claim_single_winner(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    execution_service: ActionExecutionService,
    cleanup: None,
) -> None:
    oid = f"INC-{_sfx()}"
    await _seed_connector_and_source(session_factory, object_id=oid)
    event_id = await _create_event(session_factory, store, object_id=oid)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            execution_owner=ExecutionOwner.XDR_MANAGED,
            disposition_source_ref=_locator(object_id=oid),
        ),
    )
    results = await asyncio.gather(
        execution_service.execute_action(action.action_id),
        execution_service.execute_action(action.action_id),
        return_exceptions=True,
    )
    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
