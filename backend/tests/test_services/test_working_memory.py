"""WorkingMemory + FIELD_OWNERSHIP tests (ISSUE-014)."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.v1.schemas import EventSummary
from app.core.errors import GuardrailViolationError
from app.core.redis_client import RedisClient
from app.db import models as orm
from app.models.context import EventContext
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    WritebackReadiness,
)
from app.services.context_service import EventContextStore
from app.services.degraded_flag_service import DegradedFlagService
from app.services.working_memory import (
    FIELD_OWNERSHIP,
    SCRATCHPAD_LIMIT,
    WRITER_ALIASES,
    WorkingMemory,
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


@pytest_asyncio.fixture
async def wm(
    store: EventContextStore,
    redis_client: RedisClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> WorkingMemory:
    memory = WorkingMemory(store, redis_client, wm_strict=True)
    degraded = DegradedFlagService(store, session_factory)
    memory.bind_degraded_flag_service(degraded)
    return memory


def _sfx() -> str:
    return uuid.uuid4().hex[:8]


def _summary(event_id: str) -> EventSummary:
    return EventSummary(
        event_id=event_id,
        event_type=EventType.INSIDER_THREAT,
        title="wm-test",
        status=EventStatus.NEW,
        severity=Severity.LOW,
        risk_score=10,
        final_verdict=FinalVerdict.NONE,
        writeback_required=False,
        writeback_readiness=WritebackReadiness.NOT_REQUIRED,
        disposition_policy=DispositionPolicy.NOT_REQUIRED,
    )


async def _seed_event(session_factory: async_sessionmaker[AsyncSession]) -> str:
    event_id = f"evt-20260713-{_sfx()}"
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SecurityEvent(
                    event_id=event_id,
                    event_type="insider_threat",
                    title="wm-test",
                    creation_source_ref={"source_object_id": f"INC-{_sfx()}"},
                )
            )
    return event_id


# --------------------------------------------------------------------------- #
# Ownership table
# --------------------------------------------------------------------------- #


def test_field_ownership_covers_event_context_both_directions() -> None:
    schema = set(EventContext.model_fields.keys())
    owned = set(FIELD_OWNERSHIP.keys())
    assert schema == owned, {
        "missing": sorted(schema - owned),
        "ghost": sorted(owned - schema),
    }
    assert "system" not in FIELD_OWNERSHIP.values()
    assert FIELD_OWNERSHIP["false_positive_match"] == "FalsePositiveMatcher"
    assert WRITER_ALIASES["RuleBasedFalsePositiveHook"] == "FalsePositiveMatcher"
    assert FIELD_OWNERSHIP["degraded_flags"] == "DegradedFlagService"
    assert FIELD_OWNERSHIP["scratchpad"] == "WorkingMemory"


# --------------------------------------------------------------------------- #
# Read / write
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_owner_write_success_and_access_log(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    triage = wm.for_writer("TriageAgent")
    risk = wm.for_writer("RiskAgent")

    await triage.write(
        event_id,
        "triage_result",
        {"severity": "high"},
    )
    value = await risk.read(event_id, "triage_result")
    assert value == {"severity": "high"}

    logs = await wm.get_access_log(event_id)
    write_logs = [e for e in logs if e.op == "write" and e.key == "triage_result"]
    read_logs = [e for e in logs if e.op == "read" and e.key == "triage_result"]
    assert write_logs and write_logs[-1].allowed is True
    assert write_logs[-1].agent_name == "TriageAgent"
    assert read_logs and read_logs[-1].allowed is True


@pytest.mark.asyncio
async def test_non_owner_write_rejected_and_logged(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    triage = wm.for_writer("TriageAgent")
    evidence = wm.for_writer("EvidenceAgent")
    await triage.write(event_id, "triage_result", {"ok": True})

    with pytest.raises(GuardrailViolationError) as exc_info:
        await evidence.write(
            event_id,
            "triage_result",
            {"ok": False},
        )
    assert exc_info.value.error_code == "working_memory_unauthorized_write"
    assert await store.get(event_id, "triage_result") == {"ok": True}

    logs = await wm.get_access_log(event_id)
    denied = [e for e in logs if e.op == "write" and e.allowed is False]
    assert denied
    assert denied[-1].agent_name == "EvidenceAgent"
    assert denied[-1].key == "triage_result"


@pytest.mark.asyncio
async def test_false_positive_hook_alias_allowed(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    hook = wm.for_writer("RuleBasedFalsePositiveHook")
    await hook.write(
        event_id,
        "false_positive_match",
        {"close_as_fp": True},
    )
    assert await store.get(event_id, "false_positive_match") == {"close_as_fp": True}


@pytest.mark.asyncio
async def test_wm_strict_false_still_rejects_non_owner(
    store: EventContextStore,
    redis_client: RedisClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    memory = WorkingMemory(store, redis_client, wm_strict=False)
    memory.bind_degraded_flag_service(DegradedFlagService(store, session_factory))
    evidence = memory.for_writer("EvidenceAgent")

    with pytest.raises(GuardrailViolationError):
        await evidence.write(event_id, "triage_result", {"via": "non-owner"})
    assert await store.get(event_id, "triage_result") is None


@pytest.mark.asyncio
async def test_plain_writer_name_cannot_spoof_capability(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))

    with pytest.raises(GuardrailViolationError):
        await wm.write(
            event_id,
            "triage_result",
            {"spoofed": True},
            writer="TriageAgent",  # type: ignore[arg-type]
        )
    assert await store.get(event_id, "triage_result") is None


@pytest.mark.asyncio
async def test_version_conflict_retries_then_succeeds(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    graph = wm.for_writer("GraphAgent")
    await graph.write(event_id, "graph_output", {"nodes": 1})

    calls = {"n": 0}
    real_cas = store.compare_and_set

    async def flaky_cas(*args: Any, **kwargs: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return await real_cas(*args, **kwargs)

    with patch.object(store, "compare_and_set", side_effect=flaky_cas):
        await graph.write(event_id, "graph_output", {"nodes": 2})

    assert calls["n"] >= 2
    assert await store.get(event_id, "graph_output") == {"nodes": 2}


@pytest.mark.asyncio
async def test_stale_redis_version_does_not_block_owner_write(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    graph = wm.for_writer("GraphAgent")
    await graph.write(event_id, "graph_output", {"n": 1})

    # Simulate a degraded (Redis-down) write that advanced the authoritative DB
    # version while the Redis {key}__version cache stayed behind.
    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                text(
                    "UPDATE event_context_field_version SET current_version = 5 "
                    "WHERE event_id = :e AND field_name = 'graph_output'"
                ),
                {"e": event_id},
            )

    # Owner write must still succeed: CAS ``expected`` comes from the DB, not the
    # stale Redis cache (else this would raise version_conflict).
    await graph.write(event_id, "graph_output", {"n": 2})
    assert await store.get(event_id, "graph_output") == {"n": 2}
    assert await store.get_field_version(event_id, "graph_output") == 6


# --------------------------------------------------------------------------- #
# Scratchpad
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scratchpad_append_and_fifo_roll(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    triage = wm.for_writer("TriageAgent")

    for i in range(SCRATCHPAD_LIMIT + 5):
        await triage.append_scratchpad(event_id, f"note-{i}")

    entries = await triage.read_scratchpad(event_id)
    assert len(entries) == SCRATCHPAD_LIMIT
    assert entries[0].note == "note-5"
    assert entries[-1].note == f"note-{SCRATCHPAD_LIMIT + 4}"

    mirrored = await store.get(event_id, "scratchpad")
    assert isinstance(mirrored, list)
    assert len(mirrored) == SCRATCHPAD_LIMIT

    raw = await redis_client.get_client().hget(f"shadowtrace:wm:{event_id}", "scratchpad")
    assert raw is not None
    assert len(RedisClient.loads(raw)) == SCRATCHPAD_LIMIT


@pytest.mark.asyncio
async def test_concurrent_scratchpad_appends_recompute_after_cas_conflict(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    triage = wm.for_writer("TriageAgent")
    evidence = wm.for_writer("EvidenceAgent")
    real_get_versioned = store.get_versioned_field
    reads_ready = asyncio.Event()
    reads = 0

    async def synchronized_get_versioned(
        target_event_id: str,
        key: str,
    ) -> tuple[Any, int]:
        nonlocal reads
        result = await real_get_versioned(target_event_id, key)
        if key == "scratchpad" and reads < 2:
            reads += 1
            if reads == 2:
                reads_ready.set()
            await reads_ready.wait()
        return result

    with patch.object(
        store,
        "get_versioned_field",
        side_effect=synchronized_get_versioned,
    ):
        await asyncio.gather(
            triage.append_scratchpad(event_id, "first"),
            evidence.append_scratchpad(event_id, "second"),
        )

    entries = await triage.read_scratchpad(event_id)
    assert {entry.note for entry in entries} == {"first", "second"}
    assert {entry.agent_name for entry in entries} == {"TriageAgent", "EvidenceAgent"}


@pytest.mark.asyncio
async def test_redis_unavailable_marks_degraded_flag_once(
    wm: WorkingMemory,
    store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_event(session_factory)
    await store.init_context(event_id, _summary(event_id))
    risk = wm.for_writer("RiskAgent")

    with patch.object(store._redis, "ping", new_callable=AsyncMock, return_value=False):
        with patch.object(wm._redis, "ping", new_callable=AsyncMock, return_value=False):
            with patch("app.services.context_service.asyncio.sleep", new_callable=AsyncMock):
                await risk.write(
                    event_id,
                    "risk_assessment",
                    {"score": 1},
                )
                await risk.write(
                    event_id,
                    "risk_assessment",
                    {"score": 2},
                )

    async with session_factory() as session:
        se = await session.get(orm.SecurityEvent, event_id)
        assert se is not None
        assert any(
            str(f).startswith("redis_context_unavailable=") for f in (se.degraded_flags or [])
        )

    flags = await store.get(event_id, "degraded_flags")
    assert "redis_context_unavailable=true" in flags
