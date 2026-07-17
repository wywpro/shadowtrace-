"""EventAuditLogService persistence and ordering tests (ISSUE-028)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db import models as orm
from app.services.event_audit_log_service import EventAuditLogService

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)


def _alembic_config() -> Config:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return config


@pytest.fixture(scope="module")
def migrated_database() -> None:
    command.upgrade(_alembic_config(), "head")


@pytest_asyncio.fixture
async def session_factory(
    migrated_database: None,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_logs(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    async with session_factory() as session:
        async with session.begin():
            await session.execute(delete(orm.EventAuditLog))
    yield
    async with session_factory() as session:
        async with session.begin():
            await session.execute(delete(orm.EventAuditLog))


@pytest.fixture
def service(
    session_factory: async_sessionmaker[AsyncSession],
) -> EventAuditLogService:
    return EventAuditLogService(session_factory)


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Basic persistence tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_log_transition_persists_and_returns_id(
    service: EventAuditLogService,
) -> None:
    event_id = _id("evt")
    log_id = await service.log_transition(
        event_id=event_id,
        from_status="new",
        to_status="triaging",
        operator="system",
        reason="ingestion triggered investigation",
    )

    assert log_id.isdigit()

    rows = await service.get_logs_by_event(event_id)
    assert len(rows) == 1
    assert rows[0].event_id == event_id
    assert rows[0].from_status == "new"
    assert rows[0].to_status == "triaging"
    assert rows[0].operator == "system"
    assert rows[0].reason == "ingestion triggered investigation"
    assert rows[0].created_at is not None


@pytest.mark.asyncio
async def test_log_transition_redacts_credential_text(
    service: EventAuditLogService,
) -> None:
    event_id = _id("evt")
    await service.log_transition(
        event_id=event_id,
        from_status="triaging",
        to_status="collecting_evidence",
        operator="Authorization: Bearer s3cr3t-t0k3n-abc123",
        reason="token=ghp_fake_github_pat_1234567890abcdef in reason text",
    )

    rows = await service.get_logs_by_event(event_id)
    assert len(rows) == 1

    operator_text = str(rows[0].operator)
    reason_text = str(rows[0].reason)
    combined = operator_text + reason_text
    assert "s3cr3t-t0k3n-abc123" not in combined
    assert "ghp_fake_github_pat_1234567890abcdef" not in combined


@pytest.mark.asyncio
async def test_logs_ordered_by_created_at_asc(
    service: EventAuditLogService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_id = _id("evt")
    base = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
    clock_iter = iter((
        base,
        base + timedelta(seconds=1),
        base + timedelta(seconds=2),
    ))

    # We need to patch _utc_now in the service module to control ordering.
    # Since log_transition creates its own session+transaction, we mock the
    # module-level _utc_now used in event_audit_log_service.
    import app.services.event_audit_log_service as svc_mod

    original_utc_now = svc_mod._utc_now
    try:
        svc_mod._utc_now = lambda: next(clock_iter)

        await service.log_transition(event_id, "new", "triaging", "system", "step 1")
        await service.log_transition(event_id, "triaging", "analyzing", "system", "step 2")
        await service.log_transition(event_id, "analyzing", "scoring", "system", "step 3")
    finally:
        svc_mod._utc_now = original_utc_now

    rows = await service.get_logs_by_event(event_id)
    assert len(rows) == 3
    assert [r.reason for r in rows] == ["step 1", "step 2", "step 3"]


@pytest.mark.asyncio
async def test_get_logs_by_event_empty(
    service: EventAuditLogService,
) -> None:
    rows = await service.get_logs_by_event("evt-nonexistent")
    assert rows == []


# --------------------------------------------------------------------------- #
# In-session variant
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_log_transition_in_session_shares_transaction(
    service: EventAuditLogService,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When logged inside an existing transaction, the row should be visible
    within that same session after flush."""
    event_id = _id("evt")
    now = datetime(2026, 7, 18, 8, 0, 0, tzinfo=UTC)
    import app.services.event_audit_log_service as svc_mod

    original = svc_mod._utc_now
    svc_mod._utc_now = lambda: now
    try:
        async with session_factory() as session:
            async with session.begin():
                log_id = await service.log_transition_in_session(
                    session,
                    event_id,
                    "scoring",
                    "planning_response",
                    "approval_engine",
                    "auto-escalated",
                )
                # Should be visible within this uncommitted transaction
                row = await session.get(orm.EventAuditLog, int(log_id))
                assert row is not None
                assert row.reason == "auto-escalated"
    finally:
        svc_mod._utc_now = original

    # After commit, it persists
    rows = await service.get_logs_by_event(event_id)
    assert len(rows) == 1
    assert rows[0].operator == "approval_engine"


@pytest.mark.asyncio
async def test_none_operator_and_reason_are_accepted(
    service: EventAuditLogService,
) -> None:
    event_id = _id("evt")
    await service.log_transition(
        event_id=event_id,
        from_status=None,
        to_status="failed",
        operator=None,
        reason=None,
    )
    rows = await service.get_logs_by_event(event_id)
    assert len(rows) == 1
    assert rows[0].from_status is None
    assert rows[0].operator is None
    assert rows[0].reason is None
