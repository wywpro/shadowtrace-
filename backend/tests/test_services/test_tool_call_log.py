"""ToolCallLogService persistence and redaction tests (ISSUE-023)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import orjson
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db import models as orm
from app.models.enums import ToolCategory
from app.models.tool_meta import ToolResultStatus
from app.services.tool_call_log_service import (
    MAX_AUDIT_FIELD_BYTES,
    ToolCallLogService,
)

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
async def clean_tool_logs(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    async with session_factory() as session:
        async with session.begin():
            await session.execute(delete(orm.ToolCallLog))
    yield
    async with session_factory() as session:
        async with session.begin():
            await session.execute(delete(orm.ToolCallLog))


@pytest.fixture
def service(
    session_factory: async_sessionmaker[AsyncSession],
) -> ToolCallLogService:
    return ToolCallLogService(session_factory)


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_two_phase_write_persists_redacted_replay_context(
    service: ToolCallLogService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_at = datetime(2026, 7, 17, 1, 0, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=1_250)
    clock = iter((started_at, completed_at))
    monkeypatch.setattr(
        "app.services.tool_call_log_service._utc_now",
        lambda: next(clock),
    )
    call_id = _id("call")
    event_id = _id("evt")
    credential_ref = "vault://credentials/mock-tool-read"
    secrets = {
        "password": "password-must-not-persist",
        "token": "token-must-not-persist",
        "note": "Authorization: Bearer bearer-must-not-persist",
        "error": "cookie=session-must-not-persist",
    }

    returned = await service.log_start(
        call_id,
        event_id,
        _id("act"),
        "block_ip",
        ToolCategory.RESPONSE,
        {
            **secrets,
            "credential_ref": credential_ref,
            "target": "203.0.113.23",
            "raw_payload": {
                "provider_record": "raw-provider-record",
                "secret": "nested-secret-must-not-persist",
            },
        },
    )
    await service.log_finish(
        call_id,
        ToolResultStatus.SUCCESS,
        {
            "status": "blocked",
            "raw_result": {
                "provider_response": "complete-provider-payload",
                "authorization": "raw-result-secret",
            },
        },
        "Authorization: Bearer finish-error-secret",
        2,
    )

    assert returned == call_id
    row = await service.get_log(call_id)
    assert row is not None
    assert row.status == ToolResultStatus.SUCCESS.value
    assert row.started_at == started_at
    assert row.completed_at == completed_at
    assert row.duration_ms == 1_250
    assert row.retry_count == 2
    assert row.parameters["credential_ref"] == credential_ref
    assert row.parameters["target"] == "203.0.113.23"
    assert row.parameters["password"] == "[REDACTED]"
    assert row.parameters["token"] == "[REDACTED]"
    assert row.parameters["raw_payload"]["reason"] == "raw_payload"
    assert len(row.parameters["raw_payload"]["sha256"]) == 64
    assert row.result["raw_result"]["reason"] == "raw_payload"
    assert row.error_detail is not None

    serialized = orjson.dumps(
        {
            "parameters": row.parameters,
            "result": row.result,
            "error_detail": row.error_detail,
        }
    ).decode()
    for secret in (
        *secrets.values(),
        "raw-provider-record",
        "nested-secret-must-not-persist",
        "complete-provider-payload",
        "raw-result-secret",
        "finish-error-secret",
    ):
        assert secret not in serialized


@pytest.mark.asyncio
async def test_queries_filter_and_order_by_started_at(
    service: ToolCallLogService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = datetime(2026, 7, 17, 2, 0, tzinfo=UTC)
    current = base
    monkeypatch.setattr(
        "app.services.tool_call_log_service._utc_now",
        lambda: current,
    )
    event_id = _id("evt")
    other_event_id = _id("evt")
    later_call = _id("call")
    other_call = _id("call")
    earlier_call = _id("call")

    current = base + timedelta(seconds=3)
    await service.log_start(
        later_call,
        event_id,
        None,
        "query_asset_info",
        ToolCategory.QUERY,
        {"ip": "203.0.113.24"},
    )
    current = base + timedelta(seconds=1)
    await service.log_start(
        other_call,
        other_event_id,
        None,
        "query_asset_info",
        ToolCategory.QUERY,
        {"ip": "203.0.113.25"},
    )
    current = base + timedelta(seconds=2)
    await service.log_start(
        earlier_call,
        event_id,
        None,
        "query_dns",
        ToolCategory.QUERY,
        {"domain": "audit.example"},
    )

    by_event = await service.get_logs_by_event(event_id)
    assert [row.call_id for row in by_event] == [earlier_call, later_call]
    by_tool = await service.get_logs_by_tool("query_asset_info")
    assert [row.call_id for row in by_tool] == [other_call, later_call]
    limited = await service.get_logs_by_tool("query_asset_info", limit=1)
    assert [row.call_id for row in limited] == [other_call]
    assert await service.get_log("call-does-not-exist") is None
    with pytest.raises(ValueError, match="limit must be positive"):
        await service.get_logs_by_tool("query_asset_info", limit=0)


@pytest.mark.asyncio
async def test_oversized_fields_are_replaced_by_hash_markers(
    service: ToolCallLogService,
) -> None:
    call_id = _id("call")
    oversized = "x" * (MAX_AUDIT_FIELD_BYTES + 1_024)

    await service.log_start(
        call_id,
        _id("evt"),
        None,
        "query_history_cases",
        ToolCategory.QUERY,
        {"blob": oversized},
    )
    await service.log_finish(
        call_id,
        ToolResultStatus.FAILED,
        {"blob": oversized},
        oversized,
        0,
    )

    row = await service.get_log(call_id)
    assert row is not None
    for projection in (row.parameters, row.result):
        assert projection["_truncated"] is True
        assert projection["original_size_bytes"] > MAX_AUDIT_FIELD_BYTES
        assert len(projection["sha256"]) == 64
        assert projection["top_level_keys"] == ["blob"]
        assert oversized not in str(projection)
    assert row.error_detail is not None
    assert row.error_detail.startswith("[TRUNCATED original_size_bytes=")
    assert oversized not in row.error_detail
