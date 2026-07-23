"""ApprovalEngine tests (ISSUE-058).

Requires Compose PostgreSQL (+ Redis for bus/state tests). Run from ``backend/``:

    pytest tests/test_services/test_approval_engine.py -v
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.agents.response_agent import build_mock_capability_manifest
from app.core.auth import Principal
from app.core.errors import ApprovalDecisionConflictError, InvalidStateTransitionError
from app.core.event_bus import EventBus
from app.core.redis_client import RedisClient
from app.db import models as orm
from app.db.orm.approval import ApprovalRecordORM
from app.models.action import TERMINAL_DISPOSITION_TOOL, Action
from app.models.agent_io import RiskAssessment, ScoringMode
from app.models.approval import ApprovalDecisionKind
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
)
from app.models.source import SourceReference
from app.services.approval_engine import (
    SYSTEM_TIMEOUT_OPERATOR,
    ApprovalEngine,
    evaluate_hard_gates,
    evaluate_level_rules,
)
from app.services.context_service import EventContextStore, event_summary_from_security_event
from app.services.degraded_flag_service import DegradedFlagService
from app.services.event_audit_log_service import EventAuditLogService
from app.services.state_machine_service import StateMachineService

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class FakeEventBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, dict]] = []

    async def publish_event(
        self,
        event_id: str,
        message_type: str,
        payload: dict | None = None,
    ) -> bool:
        self.published.append((event_id, message_type, dict(payload or {})))
        return True


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return cfg


@pytest.fixture(scope="module")
def migrated() -> None:
    command.upgrade(_alembic_config(), "head")


@pytest_asyncio.fixture
async def session_factory(
    migrated: None,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[RedisClient]:
    client = RedisClient(url=REDIS_URL)
    if not await client.ping():
        await client.aclose()
        pytest.skip("Redis not reachable; start Compose redis first")
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def store(
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> EventContextStore:
    return EventContextStore(redis_client, session_factory)


@pytest_asyncio.fixture
async def fake_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest_asyncio.fixture
async def state_machine(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    redis_client: RedisClient,
) -> StateMachineService:
    bus = EventBus(redis_client)
    audit = EventAuditLogService(session_factory)
    degraded = DegradedFlagService(store, session_factory)
    return StateMachineService(
        session_factory,
        store,
        event_bus=bus,
        audit_log=audit,
        degraded_flags=degraded,
    )


@pytest_asyncio.fixture
async def engine(
    session_factory: async_sessionmaker[AsyncSession],
    fake_bus: FakeEventBus,
    store: EventContextStore,
    state_machine: StateMachineService,
    cleanup: None,
) -> ApprovalEngine:
    return ApprovalEngine(
        session_factory,
        event_bus=fake_bus,  # type: ignore[arg-type]
        state_machine=state_machine,
        context_store=store,
        capability_manifest=build_mock_capability_manifest(),
    )


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_db(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    async with session_factory() as session:
        async with session.begin():
            for table in (
                ApprovalRecordORM,
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
    yield


@pytest_asyncio.fixture
async def cleanup(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    yield
    async with session_factory() as session:
        async with session.begin():
            for table in (
                ApprovalRecordORM,
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


def _ref(*, kind: SourceObjectKind, object_id: str) -> SourceReference:
    return SourceReference(
        source_kind=kind,
        source_product="mock_xdr",
        source_tenant_id="tenant-1",
        connector_id="conn-mock",
        source_object_id=object_id,
        ingested_at=datetime.now(UTC),
    )


def _risk(*, confidence: float = 0.9, severity: Severity = Severity.HIGH) -> RiskAssessment:
    return RiskAssessment(
        risk_score=80,
        severity=severity,
        confidence=confidence,
        scoring_mode=ScoringMode.RULE_ONLY,
    )


def _action_model(**overrides: object) -> Action:
    base = {
        "action_id": f"act-{_sfx()}",
        "event_id": "evt-placeholder",
        "plan_revision": 1,
        "action_fingerprint": f"fp-{_sfx()}",
        "action_category": ActionCategory.RESPONSE,
        "action_name": "block ip",
        "tool_name": "block_ip",
        "action_level": ActionLevel.L4,
        "execution_owner": ExecutionOwner.DIRECT_TOOL,
        "status": ActionStatus.PENDING,
    }
    base.update(overrides)
    return Action.model_validate(base)


async def _create_event(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    *,
    status: EventStatus = EventStatus.PLANNING_RESPONSE,
    disposition_policy: DispositionPolicy = DispositionPolicy.NOT_REQUIRED,
) -> str:
    sfx = _sfx()
    event_id = f"evt-20260723-{sfx}"
    now = datetime.now(UTC)
    ref = _ref(kind=SourceObjectKind.INCIDENT, object_id=f"INC-{sfx}")
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SecurityEvent(
                    event_id=event_id,
                    event_type=EventType.OTHER.value,
                    title="approval-test",
                    description="",
                    status=status.value,
                    severity=Severity.LOW.value,
                    risk_score=10,
                    confidence=0.5,
                    final_verdict=FinalVerdict.NONE.value,
                    creation_source_ref=ref.model_dump(mode="json"),
                    source_reference_snapshots=[ref.model_dump(mode="json")],
                    disposition_policy=disposition_policy.value,
                    occurred_at=now,
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
                    writeback_required=action.writeback_required,
                    writeback_applicable=action.writeback_applicable,
                    writeback_readiness=action.writeback_readiness.value,
                    reason=action.reason,
                )
            )
    return action.model_copy(update={"event_id": event_id})


# --------------------------------------------------------------------------- #
# Pure rule tests
# --------------------------------------------------------------------------- #


def test_evaluate_level_rules_l0_auto_approve() -> None:
    action = _action_model(action_level=ActionLevel.L0)
    decision = evaluate_level_rules(action, confidence=0.1, severity=Severity.LOW)
    assert decision.decision is ApprovalDecisionKind.AUTO_APPROVE
    assert decision.rule_applied == "level_l0_l1"


def test_evaluate_level_rules_l2_below_threshold() -> None:
    action = _action_model(action_level=ActionLevel.L2)
    decision = evaluate_level_rules(action, confidence=0.5, severity=Severity.MEDIUM)
    assert decision.decision is ApprovalDecisionKind.REQUIRE_APPROVAL


def test_evaluate_level_rules_l3_high_confidence_auto() -> None:
    action = _action_model(action_level=ActionLevel.L3)
    decision = evaluate_level_rules(action, confidence=0.9, severity=Severity.HIGH)
    assert decision.decision is ApprovalDecisionKind.AUTO_APPROVE


def test_evaluate_level_rules_l4_requires_manual() -> None:
    action = _action_model(action_level=ActionLevel.L4)
    decision = evaluate_level_rules(action, confidence=1.0, severity=Severity.CRITICAL)
    assert decision.decision is ApprovalDecisionKind.REQUIRE_APPROVAL
    assert decision.rule_applied == "level_l4_l5_manual"


def test_evaluate_hard_gates_rejects_unknown_tool() -> None:
    manifest = build_mock_capability_manifest(disabled_tools=frozenset({"block_ip"}))
    action = _action_model(tool_name="block_ip")
    gate = evaluate_hard_gates(action, manifest=manifest)
    assert gate is not None
    assert gate.decision is ApprovalDecisionKind.AUTO_REJECT


# --------------------------------------------------------------------------- #
# Integration tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_l0_auto_approve_without_approval_required(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    fake_bus: FakeEventBus,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L0),
    )
    decision = await engine.evaluate(action, _risk(), approval_cycle=0)
    assert decision.decision is ApprovalDecisionKind.AUTO_APPROVE

    async with session_factory() as session:
        row = await session.get(orm.Action, action.action_id)
        assert row is not None
        assert row.status == ActionStatus.APPROVED.value

    assert not any(msg_type == "approval_required" for _, msg_type, _ in fake_bus.published)


@pytest.mark.asyncio
async def test_l4_waiting_approval_publishes_once(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    fake_bus: FakeEventBus,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    decision = await engine.evaluate(action, _risk(), approval_cycle=0)
    assert decision.decision is ApprovalDecisionKind.REQUIRE_APPROVAL

    async with session_factory() as session:
        row = await session.get(orm.Action, action.action_id)
        assert row is not None
        assert row.status == ActionStatus.WAITING_APPROVAL.value
        event = await session.get(orm.SecurityEvent, event_id)
        assert event is not None
        assert event.status == EventStatus.WAITING_APPROVAL.value

    required = [p for p in fake_bus.published if p[1] == "approval_required"]
    assert len(required) == 1
    assert required[0][2]["action_id"] == action.action_id


@pytest.mark.asyncio
async def test_evaluate_replay_does_not_duplicate_notification(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    fake_bus: FakeEventBus,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    await engine.evaluate(action, _risk(), approval_cycle=0)
    await engine.evaluate(action, _risk(), approval_cycle=0)
    required = [p for p in fake_bus.published if p[1] == "approval_required"]
    assert len(required) == 1


@pytest.mark.asyncio
async def test_approve_and_reject_flow(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    fake_bus: FakeEventBus,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    await engine.evaluate(action, _risk(), approval_cycle=0)
    principal = Principal(subject="approver-1", roles=["approver"])
    await engine.approve(action.action_id, principal, "ok", "dec-1")
    async with session_factory() as session:
        row = await session.get(orm.Action, action.action_id)
        assert row is not None
        assert row.status == ActionStatus.APPROVED.value

    updated = [p for p in fake_bus.published if p[1] == "approval_updated"]
    assert updated
    assert updated[-1][2]["decision"] == "approved"


@pytest.mark.asyncio
async def test_approve_non_waiting_returns_400(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            action_level=ActionLevel.L0,
            status=ActionStatus.APPROVED,
        ),
    )
    principal = Principal(subject="approver-1", roles=["approver"])
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        await engine.approve(action.action_id, principal, None, None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_decision_id_replay_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    await engine.evaluate(action, _risk(), approval_cycle=0)
    principal = Principal(subject="approver-1", roles=["approver"])
    await engine.approve(action.action_id, principal, "ok", "dec-replay")
    await engine.approve(action.action_id, principal, "ok", "dec-replay")


@pytest.mark.asyncio
async def test_concurrent_decision_conflict(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    await engine.evaluate(action, _risk(), approval_cycle=0)
    p1 = Principal(subject="approver-a", roles=["approver"])
    p2 = Principal(subject="approver-b", roles=["approver"])
    await engine.approve(action.action_id, p1, "first", "dec-a")
    with pytest.raises(ApprovalDecisionConflictError):
        await engine.reject(action.action_id, p2, "second", "dec-b")


@pytest.mark.asyncio
async def test_timeout_rejects_with_system_timeout(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    await engine.evaluate(action, _risk(), approval_cycle=0)
    async with session_factory() as session:
        async with session.begin():
            record = await session.scalar(
                select(ApprovalRecordORM).where(ApprovalRecordORM.action_id == action.action_id)
            )
            assert record is not None
            record.timeout_at = datetime.now(UTC) - timedelta(minutes=1)
    await engine.handle_timeout(action.action_id, approval_cycle=0)
    async with session_factory() as session:
        row = await session.get(orm.Action, action.action_id)
        assert row is not None
        assert row.status == ActionStatus.REJECTED.value
        record = await session.scalar(
            select(ApprovalRecordORM).where(ApprovalRecordORM.action_id == action.action_id)
        )
        assert record is not None
        assert record.operator == SYSTEM_TIMEOUT_OPERATOR
        assert record.decision == ApprovalDecisionKind.AUTO_REJECT.value


@pytest.mark.asyncio
async def test_scan_timeouts_rejects_expired_waiting_approval(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    await engine.evaluate(action, _risk(), approval_cycle=0)
    async with session_factory() as session:
        async with session.begin():
            record = await session.scalar(
                select(ApprovalRecordORM).where(ApprovalRecordORM.action_id == action.action_id)
            )
            assert record is not None
            record.timeout_at = datetime.now(UTC) - timedelta(minutes=1)

    touched = await engine.scan_timeouts()
    assert event_id in touched

    async with session_factory() as session:
        row = await session.get(orm.Action, action.action_id)
        assert row is not None
        assert row.status == ActionStatus.REJECTED.value
        record = await session.scalar(
            select(ApprovalRecordORM).where(ApprovalRecordORM.action_id == action.action_id)
        )
        assert record is not None
        assert record.operator == SYSTEM_TIMEOUT_OPERATOR
        assert record.decision == ApprovalDecisionKind.AUTO_REJECT.value


@pytest.mark.asyncio
async def test_l0_without_idempotency_requires_manual(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    fake_bus: FakeEventBus,
    cleanup: None,
) -> None:
    manifest = build_mock_capability_manifest().model_copy(
        update={"supports_idempotency": False, "supports_lookup_by_idempotency": False}
    )
    engine = ApprovalEngine(
        session_factory,
        event_bus=fake_bus,  # type: ignore[arg-type]
        context_store=store,
        capability_manifest=manifest,
    )
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L0),
    )
    decision = await engine.evaluate(action, _risk(), approval_cycle=0)
    assert decision.decision is ApprovalDecisionKind.REQUIRE_APPROVAL

    async with session_factory() as session:
        row = await session.get(orm.Action, action.action_id)
        assert row is not None
        assert row.status == ActionStatus.WAITING_APPROVAL.value


@pytest.mark.asyncio
async def test_mixed_l0_l4_stays_waiting_until_l4_decided(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    fake_bus: FakeEventBus,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    l0 = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L0, action_name="auto"),
    )
    l4 = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4, action_name="manual"),
    )
    await engine.evaluate(l0, _risk(), approval_cycle=0)
    await engine.evaluate(l4, _risk(), approval_cycle=0)

    async with session_factory() as session:
        event = await session.get(orm.SecurityEvent, event_id)
        assert event is not None
        assert event.status == EventStatus.WAITING_APPROVAL.value
        l0_row = await session.get(orm.Action, l0.action_id)
        l4_row = await session.get(orm.Action, l4.action_id)
        assert l0_row is not None and l0_row.status == ActionStatus.APPROVED.value
        assert l4_row is not None and l4_row.status == ActionStatus.WAITING_APPROVAL.value


@pytest.mark.asyncio
async def test_plan_fully_decided_advances_to_executing(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    state_machine: StateMachineService,
    fake_bus: FakeEventBus,
    cleanup: None,
) -> None:
    engine = ApprovalEngine(
        session_factory,
        event_bus=fake_bus,  # type: ignore[arg-type]
        state_machine=state_machine,
        capability_manifest=build_mock_capability_manifest(),
    )
    event_id = await _create_event(session_factory, store)
    a1 = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L0),
    )
    await engine.evaluate(a1, _risk(), approval_cycle=0)
    async with session_factory() as session:
        event = await session.get(orm.SecurityEvent, event_id)
        assert event is not None
        assert event.status == EventStatus.EXECUTING_RESPONSE.value


@pytest.mark.asyncio
async def test_multiple_l4_single_event_waiting_approval(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    fake_bus: FakeEventBus,
    engine: ApprovalEngine,
) -> None:
    event_id = await _create_event(session_factory, store)
    a1 = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4, action_name="a1"),
    )
    a2 = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4, action_name="a2"),
    )
    await engine.evaluate(a1, _risk(), approval_cycle=0)
    await engine.evaluate(a2, _risk(), approval_cycle=0)

    async with session_factory() as session:
        event = await session.get(orm.SecurityEvent, event_id)
        assert event is not None
        assert event.status == EventStatus.WAITING_APPROVAL.value

    required = [p for p in fake_bus.published if p[1] == "approval_required"]
    assert len(required) == 2
    assert {p[2]["action_id"] for p in required} == {a1.action_id, a2.action_id}


@pytest.mark.asyncio
async def test_required_deferred_rejected_blocks_executing(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    state_machine: StateMachineService,
    fake_bus: FakeEventBus,
    cleanup: None,
) -> None:
    resume = AsyncMock()
    engine = ApprovalEngine(
        session_factory,
        event_bus=fake_bus,  # type: ignore[arg-type]
        state_machine=state_machine,
        resume_investigation=resume,
        capability_manifest=build_mock_capability_manifest(),
    )
    event_id = await _create_event(
        session_factory,
        store,
        disposition_policy=DispositionPolicy.REQUIRED,
    )
    immediate = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L0),
    )
    deferred = await _insert_action(
        session_factory,
        event_id,
        _action_model(
            event_id=event_id,
            action_level=ActionLevel.L4,
            tool_name=TERMINAL_DISPOSITION_TOOL,
            execution_phase=ActionExecutionPhase.POST_VERIFY,
            execution_owner=ExecutionOwner.XDR_MANAGED,
            activation_condition="after_effect_resolution",
            writeback_required=True,
        ),
    )
    await engine.evaluate(immediate, _risk(), approval_cycle=0)
    await engine.evaluate(deferred, _risk(), approval_cycle=0)
    principal = Principal(subject="approver-1", roles=["approver"])
    await engine.reject(deferred.action_id, principal, "no writeback", "dec-def")

    async with session_factory() as session:
        event = await session.get(orm.SecurityEvent, event_id)
        assert event is not None
        assert event.status == EventStatus.REPORTING.value
    resume.assert_awaited()


@pytest.mark.asyncio
async def test_all_rejected_transitions_to_reporting(
    session_factory: async_sessionmaker[AsyncSession],
    store: EventContextStore,
    state_machine: StateMachineService,
    fake_bus: FakeEventBus,
    cleanup: None,
) -> None:
    engine = ApprovalEngine(
        session_factory,
        event_bus=fake_bus,  # type: ignore[arg-type]
        state_machine=state_machine,
        capability_manifest=build_mock_capability_manifest(),
    )
    event_id = await _create_event(session_factory, store)
    action = await _insert_action(
        session_factory,
        event_id,
        _action_model(event_id=event_id, action_level=ActionLevel.L4),
    )
    await engine.evaluate(action, _risk(), approval_cycle=0)
    principal = Principal(subject="approver-1", roles=["approver"])
    await engine.reject(action.action_id, principal, "declined", "dec-r")
    async with session_factory() as session:
        event = await session.get(orm.SecurityEvent, event_id)
        assert event is not None
        assert event.status == EventStatus.REPORTING.value
