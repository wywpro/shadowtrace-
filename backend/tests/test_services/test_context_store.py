"""EventContextStore tests against Compose Redis + PostgreSQL (ISSUE-013)."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.v1.schemas import EventSummary
from app.core.redis_client import RedisClient
from app.db import models as orm
from app.models.enums import (
    DispositionIntentKind,
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    WritebackReadiness,
    WritebackStatus,
)
from app.services.context_service import (
    CLOSED_TTL_SECONDS,
    EventContextStore,
    InitResult,
    SetResult,
    ctx_key,
)
from app.services.degraded_flag_service import DegradedFlagService

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
    command.upgrade(_alembic_config(), "head")


@pytest_asyncio.fixture
async def session_factory(migrated: None) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
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


def _sfx() -> str:
    return uuid.uuid4().hex[:8]


def _summary(event_id: str, **kwargs: Any) -> EventSummary:
    base = dict(
        event_id=event_id,
        event_type=EventType.INSIDER_THREAT,
        title="context-store-test",
        status=EventStatus.NEW,
        severity=Severity.LOW,
        risk_score=10,
        final_verdict=FinalVerdict.NONE,
        writeback_required=False,
        writeback_readiness=WritebackReadiness.NOT_REQUIRED,
        disposition_policy=DispositionPolicy.NOT_REQUIRED,
    )
    base.update(kwargs)
    return EventSummary(**base)  # type: ignore[arg-type]


async def _seed_event(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: str = "new",
    disposition_policy: str = "not_required",
    replan_count: int = 0,
    degraded_flags: list[Any] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> str:
    event_id = f"evt-20260712-{_sfx()}"
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SecurityEvent(
                    event_id=event_id,
                    event_type="insider_threat",
                    title="context-store-test",
                    status=status,
                    disposition_policy=disposition_policy,
                    replan_count=replan_count,
                    degraded_flags=degraded_flags or [],
                    event_context_snapshot=snapshot,
                    creation_source_ref={"source_object_id": f"INC-{_sfx()}"},
                )
            )
    return event_id


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_init_context_writes_version_journal_and_redis(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> None:
    event_id = await _seed_event(session_factory)
    result = await store.init_context(event_id, _summary(event_id))
    assert isinstance(result, InitResult)
    assert result.redis_ok is True
    assert result.version == 1
    assert result.initialized is True

    async with session_factory() as session:
        ver = await session.get(orm.EventContextFieldVersion, (event_id, "event"))
        assert ver is not None
        assert ver.current_version == 1
        journal = (
            await session.scalars(
                select(orm.EventContextJournal).where(
                    orm.EventContextJournal.event_id == event_id,
                    orm.EventContextJournal.field_name == "event",
                )
            )
        ).all()
        assert len(journal) == 1
        assert journal[0].version == 1

    raw = await redis_client.get_client().hget(ctx_key(event_id), "event")
    assert raw is not None
    loaded = RedisClient.loads(raw)
    assert loaded["event_id"] == event_id

    got = await store.get(event_id, "event")
    assert got["event_id"] == event_id
    full = await store.get_full_context(event_id)
    assert full.replan_count == 0


@pytest.mark.asyncio
async def test_init_context_is_atomic_and_idempotent(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    summary = _summary(event_id)
    first, second = await asyncio.gather(
        store.init_context(event_id, summary),
        store.init_context(event_id, summary),
    )
    assert {first.initialized, second.initialized} == {True, False}
    assert first.version == second.version == 1

    async with session_factory() as session:
        journals = (
            await session.scalars(
                select(orm.EventContextJournal).where(
                    orm.EventContextJournal.event_id == event_id,
                    orm.EventContextJournal.field_name == "event",
                )
            )
        ).all()
        version = await session.get(
            orm.EventContextFieldVersion,
            (event_id, "event"),
        )
    assert len(journals) == 1
    assert version is not None and version.current_version == 1


@pytest.mark.asyncio
async def test_set_and_compare_and_set_version_conflict(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))

    r1 = await store.set(event_id, "triage_result", {"verdict": "suspicious"})
    assert isinstance(r1, SetResult)
    assert r1.redis_ok is True
    assert r1.version == 1

    ok = await store.compare_and_set(
        event_id, "triage_result", expected_version=1, value={"verdict": "malicious"}
    )
    assert ok is True
    assert await store.get(event_id, "triage_result") == {"verdict": "malicious"}

    conflict = await store.compare_and_set(
        event_id, "triage_result", expected_version=1, value={"verdict": "stale"}
    )
    assert conflict is False
    assert await store.get(event_id, "triage_result") == {"verdict": "malicious"}


@pytest.mark.asyncio
async def test_compare_and_set_missing_field_returns_false_without_rows(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))

    ok = await store.compare_and_set(event_id, "storyline", expected_version=1, value={"steps": []})
    assert ok is False

    async with session_factory() as session:
        ver = await session.get(orm.EventContextFieldVersion, (event_id, "storyline"))
        assert ver is None
        journal = (
            await session.scalars(
                select(orm.EventContextJournal).where(
                    orm.EventContextJournal.event_id == event_id,
                    orm.EventContextJournal.field_name == "storyline",
                )
            )
        ).all()
        assert journal == []


@pytest.mark.asyncio
async def test_set_closed_ttl_about_24h(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    assert await store.set_closed_ttl(event_id) is True
    ttl = await redis_client.get_client().ttl(ctx_key(event_id))
    assert CLOSED_TTL_SECONDS - 30 <= ttl <= CLOSED_TTL_SECONDS


# --------------------------------------------------------------------------- #
# CLOSED snapshot / rebuild
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rebuild_closed_from_snapshot(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> None:
    event_id = await _seed_event(
        session_factory,
        status="closed",
        replan_count=2,
        degraded_flags=["redis_context_unavailable=true"],
        snapshot={
            "triage_result": {"from": "snapshot"},
            "replan_count": 99,
            "degraded_flags": ["stale"],
        },
    )
    # Ensure Redis miss so rebuild path is exercised.
    await redis_client.get_client().delete(ctx_key(event_id))

    ctx = await store.rebuild_context(event_id)
    assert ctx.triage_result == {"from": "snapshot"}
    # security_event mirrors always win
    assert ctx.replan_count == 2
    assert ctx.degraded_flags == ["redis_context_unavailable=true"]
    # ISSUE-094 §2: EventContext.event is always a validated EventSummary.
    assert isinstance(ctx.event, EventSummary)
    assert ctx.event.event_id == event_id
    assert ctx.event.status.value == "closed"


@pytest.mark.asyncio
async def test_rebuild_from_journal_when_snapshot_empty(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> None:
    event_id = await _seed_event(session_factory, status="analyzing", replan_count=1)
    await store.init_context(event_id, _summary(event_id, status=EventStatus.ANALYZING))
    await store.set(event_id, "risk_assessment", {"score": 88})
    await redis_client.get_client().delete(ctx_key(event_id))

    ctx = await store.rebuild_context(event_id)
    assert ctx.risk_assessment == {"score": 88}
    assert ctx.replan_count == 1


@pytest.mark.asyncio
async def test_refresh_closed_snapshot_from_journal_without_redis(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(
        session_factory,
        status="closed",
        replan_count=3,
        degraded_flags=["hook"],
    )
    await store.init_context(event_id, _summary(event_id, status=EventStatus.CLOSED))
    await store.set(event_id, "memory_output", {"cases": 2})

    with patch.object(store._redis, "ping", new_callable=AsyncMock, return_value=False):
        with patch("app.services.context_service.asyncio.sleep", new_callable=AsyncMock):
            ctx = await store.refresh_closed_snapshot(event_id)

    assert ctx.memory_output == {"cases": 2}
    assert ctx.replan_count == 3
    assert ctx.degraded_flags == ["hook"]

    async with session_factory() as session:
        se = await session.get(orm.SecurityEvent, event_id)
        assert se is not None
        assert se.event_context_snapshot is not None
        assert se.event_context_snapshot["memory_output"] == {"cases": 2}
        assert se.event_context_snapshot["replan_count"] == 3


@pytest.mark.asyncio
async def test_rebuild_merges_late_writeback_receipt(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    sfx = _sfx()
    event_id = await _seed_event(
        session_factory,
        status="closed",
        disposition_policy="required",
        snapshot={"triage_result": {"ok": True}},
    )
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SourceConnector(
                    connector_id=f"conn-{sfx}",
                    source_product="mock_xdr",
                    display_name="Mock",
                )
            )
            await session.flush()
            session.add(
                orm.SourceObject(
                    source_record_id=f"src-{sfx}",
                    source_product="mock_xdr",
                    source_tenant_id="t1",
                    connector_id=f"conn-{sfx}",
                    source_kind="incident",
                    source_object_id=f"INC-{sfx}",
                )
            )
            session.add(
                orm.Action(
                    action_id=f"act-{sfx}",
                    event_id=event_id,
                    plan_revision=1,
                    action_fingerprint=f"fp-{sfx}",
                    action_category="response",
                    action_name="update_source_event_disposition",
                    tool_name="",
                    action_level="l1",
                    execution_owner="xdr_managed",
                )
            )
            await session.flush()
            session.add(
                orm.DispositionOutbox(
                    outbox_id=f"obx-{sfx}",
                    writeback_id=f"wbk-{sfx}",
                    disposition_id=f"disp-{sfx}",
                    action_id=f"act-{sfx}",
                    event_id=event_id,
                    closure_cycle=1,
                    source_record_id=f"src-{sfx}",
                    source_locator_hash="h" * 64,
                    source_sequence=1,
                    intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE.value,
                    logical_slot="terminal",
                    idempotency_key=f"idem-{sfx}",
                    command_payload={"disposition": "contained"},
                    command_payload_sha256="a" * 64,
                    delivery_status="delivered",
                    latest_writeback_status=WritebackStatus.ACCEPTED.value,
                )
            )
            session.add(
                orm.DispositionReceipt(
                    writeback_id=f"wbk-{sfx}",
                    sequence=1,
                    disposition_id=f"disp-{sfx}",
                    action_id=f"act-{sfx}",
                    source_record_id=f"src-{sfx}",
                    status=WritebackStatus.CONFIRMED.value,
                    confirmation_evidence="readback_verified",
                )
            )

    ctx = await store.rebuild_context(event_id)
    assert ctx.writeback_summary is not None
    assert ctx.writeback_summary.terminal_event_confirmed is True
    assert ctx.writeback_summary.terminal_event_writeback_id == f"wbk-{sfx}"
    assert ctx.writeback_summary.writeback_counts.get(WritebackStatus.CONFIRMED) == 1


@pytest.mark.asyncio
async def test_writeback_summary_never_invents_ready_for_empty_required_actions(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """ISSUE-093 §3: REQUIRED policy with no Action/outbox yet must not report READY."""
    event_id = await _seed_event(session_factory, disposition_policy="required")

    ctx = await store.rebuild_context(event_id)
    assert ctx.writeback_summary is not None
    assert ctx.writeback_summary.aggregate_readiness is WritebackReadiness.CAPABILITY_UNKNOWN
    assert ctx.writeback_summary.required_action_count == 0
    assert ctx.writeback_summary.applicable_action_count == 0


@pytest.mark.asyncio
async def test_writeback_summary_readiness_aggregate_fails_closed_on_worst_action(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Aggregate readiness must surface the worst blocking reason, never READY,
    when at least one applicable/required Action is not ready."""
    sfx = _sfx()
    event_id = await _seed_event(session_factory, disposition_policy="required")
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.Action(
                    action_id=f"act-ready-{sfx}",
                    event_id=event_id,
                    plan_revision=1,
                    action_fingerprint=f"fp-ready-{sfx}",
                    action_category="response",
                    action_name="block ip",
                    tool_name="block_ip",
                    action_level="l2",
                    execution_owner="direct_tool",
                    writeback_required=True,
                    writeback_applicable=True,
                    writeback_readiness=WritebackReadiness.READY.value,
                )
            )
            session.add(
                orm.Action(
                    action_id=f"act-blocked-{sfx}",
                    event_id=event_id,
                    plan_revision=1,
                    action_fingerprint=f"fp-blocked-{sfx}",
                    action_category="response",
                    action_name="isolate host",
                    tool_name="isolate_host",
                    action_level="l2",
                    execution_owner="direct_tool",
                    writeback_required=True,
                    writeback_applicable=True,
                    writeback_readiness=WritebackReadiness.CAPABILITY_UNSUPPORTED.value,
                )
            )

    ctx = await store.rebuild_context(event_id)
    assert ctx.writeback_summary is not None
    assert (
        ctx.writeback_summary.aggregate_readiness is WritebackReadiness.CAPABILITY_UNSUPPORTED
    )
    assert ctx.writeback_summary.required_action_count == 2
    assert ctx.writeback_summary.applicable_action_count == 2
    assert f"act-blocked-{sfx}" in ctx.writeback_summary.blocked_action_ids
    assert f"act-ready-{sfx}" not in ctx.writeback_summary.blocked_action_ids


@pytest.mark.asyncio
async def test_writeback_summary_status_aggregate_priority_order(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """CONFLICT > UNKNOWN > PENDING > FAILED > PARTIAL > CONFIRMED (ISSUE-093 §3)."""
    sfx = _sfx()
    event_id = await _seed_event(session_factory, disposition_policy="required")
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SourceConnector(
                    connector_id=f"conn-{sfx}",
                    source_product="mock_xdr",
                    display_name="Mock",
                )
            )
            await session.flush()
            session.add(
                orm.SourceObject(
                    source_record_id=f"src-{sfx}",
                    source_product="mock_xdr",
                    source_tenant_id="t1",
                    connector_id=f"conn-{sfx}",
                    source_kind="incident",
                    source_object_id=f"INC-{sfx}",
                )
            )
            session.add(
                orm.Action(
                    action_id=f"act-{sfx}",
                    event_id=event_id,
                    plan_revision=1,
                    action_fingerprint=f"fp-{sfx}",
                    action_category="response",
                    action_name="block ip",
                    tool_name="block_ip",
                    action_level="l2",
                    execution_owner="direct_tool",
                )
            )
            await session.flush()
            for idx, status in enumerate(
                (WritebackStatus.CONFIRMED, WritebackStatus.PENDING), start=1
            ):
                session.add(
                    orm.DispositionOutbox(
                        outbox_id=f"obx-{sfx}-{idx}",
                        writeback_id=f"wbk-{sfx}-{idx}",
                        disposition_id=f"disp-{sfx}-{idx}",
                        action_id=f"act-{sfx}",
                        event_id=event_id,
                        closure_cycle=1,
                        source_record_id=f"src-{sfx}",
                        source_locator_hash="h" * 64,
                        source_sequence=idx,
                        intent_kind=DispositionIntentKind.ENTITY_ACTION_SUBMIT.value,
                        logical_slot=f"slot-{idx}",
                        idempotency_key=f"idem-{sfx}-{idx}",
                        command_payload={},
                        command_payload_sha256="a" * 64,
                        delivery_status="delivered",
                        latest_writeback_status=status.value,
                    )
                )

    ctx = await store.rebuild_context(event_id)
    assert ctx.writeback_summary is not None
    # PENDING outranks CONFIRMED — the cycle is not fully settled.
    assert ctx.writeback_summary.aggregate_status is WritebackStatus.PENDING


# --------------------------------------------------------------------------- #
# Redis degradation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_set_and_init_when_redis_down_persist_journal(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)

    with patch.object(store._redis, "ping", new_callable=AsyncMock, return_value=False):
        with patch("app.services.context_service.asyncio.sleep", new_callable=AsyncMock):
            init = await store.init_context(event_id, _summary(event_id))
            assert init.redis_ok is False
            assert init.version == 1
            sett = await store.set(event_id, "scratchpad", [{"note": "offline"}])
            assert sett.redis_ok is False
            assert sett.version == 1

    async with session_factory() as session:
        ver = await session.get(orm.EventContextFieldVersion, (event_id, "event"))
        assert ver is not None and ver.current_version == 1
        journals = (
            await session.scalars(
                select(orm.EventContextJournal).where(orm.EventContextJournal.event_id == event_id)
            )
        ).all()
        fields = {j.field_name for j in journals}
        assert "event" in fields
        assert "scratchpad" in fields


@pytest.mark.asyncio
async def test_degraded_memory_cache_refreshes_after_30s(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    await store.set(event_id, "graph_output", {"nodes": 1})

    with patch.object(store._redis, "ping", new_callable=AsyncMock, return_value=False):
        with patch("app.services.context_service.asyncio.sleep", new_callable=AsyncMock):
            first = await store.get(event_id, "graph_output")
            assert first == {"nodes": 1}

            # Simulate another worker writing a newer journal row while Redis is down.
            async with session_factory() as session:
                async with session.begin():
                    await session.execute(
                        text(
                            "INSERT INTO event_context_field_version "
                            "(event_id, field_name, current_version) "
                            "VALUES (:e, 'graph_output', 2) "
                            "ON CONFLICT (event_id, field_name) DO UPDATE "
                            "SET current_version = 2"
                        ),
                        {"e": event_id},
                    )
                    session.add(
                        orm.EventContextJournal(
                            event_id=event_id,
                            field_name="graph_output",
                            value={"nodes": 99},
                            version=2,
                        )
                    )

            # Within TTL: stale cache
            assert await store.get(event_id, "graph_output") == {"nodes": 1}

            # Expire degraded cache timestamp
            store._degraded_cache_ts[event_id] = time.monotonic() - 31.0
            refreshed = await store.get(event_id, "graph_output")
            assert refreshed == {"nodes": 99}


@pytest.mark.asyncio
async def test_redis_recovery_rebuilds_when_database_version_is_newer(
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    service = DegradedFlagService(store, session_factory)
    await service.set_flag(
        event_id,
        "disposition_writeback_blocked",
        "old",
        writer="EventService",
    )

    with patch.object(store._redis, "ping", new_callable=AsyncMock, return_value=False):
        with patch("app.services.context_service.asyncio.sleep", new_callable=AsyncMock):
            await service.set_flag(
                event_id,
                "disposition_writeback_blocked",
                "capability_unknown",
                writer="EventService",
            )

    recovered = await store.get(event_id, "degraded_flags")
    assert recovered == ["disposition_writeback_blocked=capability_unknown"]

    raw_version = await redis_client.get_client().hget(
        ctx_key(event_id),
        "degraded_flags__version",
    )
    assert RedisClient.loads(raw_version) == await store.get_field_version(
        event_id,
        "degraded_flags",
    )
    full = await store.get_full_context(event_id)
    assert full.degraded_flags == ["disposition_writeback_blocked=capability_unknown"]
