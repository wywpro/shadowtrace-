"""Migration + schema-behavior tests against a real PostgreSQL (ISSUE-003).

Requires the Compose PostgreSQL to be reachable via ``DATABASE_URL`` (async).
Run with e.g.::

    DATABASE_URL=postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace \\
        pytest tests/test_db/test_migrations.py -v
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db import models as m

BACKEND_DIR = Path(__file__).resolve().parents[2]

CORE_TABLES = {
    "security_event",
    "source_object",
    "source_event_link",
    "source_connector",
    "source_checkpoint",
    "evidence",
    "action",
    "action_execution_job",
    "action_target_result",
    "disposition_outbox",
    "disposition_receipt",
    "report",
    "agent_trace",
    "event_audit_log",
    "tool_call_log",
    "llm_call_log",
    "data_quality_error",
    "event_context_journal",
    "event_context_field_version",
}


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return cfg


@pytest.fixture(scope="module")
def migrated() -> None:
    """Ensure the schema is at head for the module (sync; runs its own loop)."""
    command.upgrade(_alembic_config(), "head")


@pytest_asyncio.fixture
async def session(migrated: None) -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess
    await engine.dispose()


def _sfx() -> str:
    return uuid.uuid4().hex[:8]


async def _seed_event(session: AsyncSession, sfx: str) -> str:
    event_id = f"evt-2026-{sfx}"
    session.add(
        m.SecurityEvent(
            event_id=event_id,
            event_type="insider_threat",
            title="test",
            creation_source_ref={"source_object_id": f"INC-{sfx}"},
        )
    )
    await session.flush()
    return event_id


async def _seed_connector_source(session: AsyncSession, sfx: str) -> tuple[str, str]:
    connector_id = f"conn-{sfx}"
    source_record_id = f"src-{sfx}"
    session.add(
        m.SourceConnector(connector_id=connector_id, source_product="mock_xdr", display_name="Mock")
    )
    await session.flush()
    session.add(
        m.SourceObject(
            source_record_id=source_record_id,
            source_product="mock_xdr",
            source_tenant_id="t1",
            connector_id=connector_id,
            source_kind="incident",
            source_object_id=f"INC-{sfx}",
        )
    )
    await session.flush()
    return connector_id, source_record_id


async def _seed_action(session: AsyncSession, event_id: str, sfx: str, fingerprint: str) -> str:
    action_id = f"act-{sfx}"
    session.add(
        m.Action(
            action_id=action_id,
            event_id=event_id,
            plan_revision=1,
            action_fingerprint=fingerprint,
            action_category="response",
            action_name="block ip",
            tool_name="block_ip",
            action_level="l2",
            execution_owner="direct_tool",
        )
    )
    await session.flush()
    return action_id


# --------------------------------------------------------------------------- #


async def test_all_core_tables_exist(session: AsyncSession) -> None:
    rows = await session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name <> 'alembic_version'"
        )
    )
    present = {r[0] for r in rows}
    assert CORE_TABLES <= present, {"missing": CORE_TABLES - present}
    # The original 18 core tables plus connector/kind checkpoints.
    assert present == CORE_TABLES, {"unexpected": present - CORE_TABLES}


async def test_llm_call_log_supports_prompt_status_and_failure_audit(
    session: AsyncSession,
) -> None:
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    row = m.LLMCallLog(
        event_id=event_id,
        agent_name="RiskAgent",
        prompt_key="risk_score",
        model_name="primary-model",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        latency_ms=25,
        fallback_level=0,
        status="llm_timeout",
    )
    session.add(row)
    await session.flush()

    stored = await session.get(m.LLMCallLog, row.id)
    assert stored is not None
    assert stored.prompt_key == "risk_score"
    assert stored.status == "llm_timeout"
    await session.rollback()


async def test_action_fingerprint_unique(session: AsyncSession) -> None:
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    fp = f"fp-{sfx}"
    await _seed_action(session, event_id, sfx, fp)
    session.add(
        m.Action(
            action_id=f"act-dup-{sfx}",
            event_id=event_id,
            plan_revision=1,
            action_fingerprint=fp,
            action_category="response",
            action_name="dup",
            tool_name="block_ip",
            action_level="l2",
            execution_owner="direct_tool",
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_action_event_foreign_key(session: AsyncSession) -> None:
    sfx = _sfx()
    session.add(
        m.Action(
            action_id=f"act-{sfx}",
            event_id=f"evt-missing-{sfx}",
            plan_revision=1,
            action_fingerprint=f"fp-{sfx}",
            action_category="response",
            action_name="orphan",
            tool_name="block_ip",
            action_level="l2",
            execution_owner="direct_tool",
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_outbox_idempotency_and_source_sequence_unique(session: AsyncSession) -> None:
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    _, source_record_id = await _seed_connector_source(session, sfx)
    action_id = await _seed_action(session, event_id, sfx, f"fp-{sfx}")

    def _outbox(oid: str, idem: str, seq: int, slot: str) -> m.DispositionOutbox:
        return m.DispositionOutbox(
            outbox_id=oid,
            writeback_id=f"wbk-{oid}",
            disposition_id=f"disp-{oid}",
            action_id=action_id,
            event_id=event_id,
            closure_cycle=1,
            source_record_id=source_record_id,
            source_locator_hash="hash",
            source_sequence=seq,
            intent_kind="entity_action_submit",
            logical_slot=slot,
            idempotency_key=idem,
            command_payload={"k": "v"},
            command_payload_sha256="sha",
        )

    session.add(_outbox(f"ob1-{sfx}", f"idem-{sfx}", 1, "slot-a"))
    await session.flush()

    # duplicate idempotency_key
    session.add(_outbox(f"ob2-{sfx}", f"idem-{sfx}", 2, "slot-b"))
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()

    # re-seed then duplicate (source_record_id, source_sequence)
    sfx2 = _sfx()
    event_id = await _seed_event(session, sfx2)
    _, source_record_id = await _seed_connector_source(session, sfx2)
    action_id = await _seed_action(session, event_id, sfx2, f"fp-{sfx2}")
    session.add(_outbox(f"ob1-{sfx2}", f"idemA-{sfx2}", 5, "slot-a"))
    await session.flush()
    session.add(_outbox(f"ob2-{sfx2}", f"idemB-{sfx2}", 5, "slot-b"))
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_event_status_update_single_active_head_and_superseding(
    session: AsyncSession,
) -> None:
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    _, source_record_id = await _seed_connector_source(session, sfx)
    action_id = await _seed_action(session, event_id, sfx, f"fp-{sfx}")

    def _head(oid: str, seq: int, superseded_by: str | None) -> m.DispositionOutbox:
        return m.DispositionOutbox(
            outbox_id=oid,
            writeback_id=f"wbk-{oid}",
            disposition_id=f"disp-{oid}",
            action_id=action_id,
            event_id=event_id,
            closure_cycle=1,
            source_record_id=source_record_id,
            source_locator_hash="hash",
            source_sequence=seq,
            intent_kind="event_status_update",
            logical_slot="terminal",
            supersedes_disposition_id=None,
            superseded_by_disposition_id=superseded_by,
            idempotency_key=f"idem-{oid}",
            command_payload={"op": "set_event_disposition"},
            command_payload_sha256="sha",
        )

    session.add(_head(f"h1-{sfx}", 1, None))
    await session.flush()
    # second active head for same lineage violates the partial unique index
    session.add(_head(f"h2-{sfx}", 2, None))
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()

    # legal superseding: mark old head superseded first, then insert new active head
    event_id = await _seed_event(session, sfx + "b")
    _, source_record_id = await _seed_connector_source(session, sfx + "b")
    action_id = await _seed_action(session, event_id, sfx + "b", f"fp-{sfx}b")
    old = _head(f"old-{sfx}", 1, None)
    session.add(old)
    await session.flush()
    old.superseded_by_disposition_id = f"disp-new-{sfx}"
    await session.flush()
    session.add(_head(f"new-{sfx}", 2, None))
    await session.flush()  # succeeds: only one active head remains
    await session.rollback()


async def test_event_status_update_active_head_is_event_scoped_not_action(
    session: AsyncSession,
) -> None:
    """ISSUE-093 §4: two *different* Actions on the same event/cycle/slot must
    collide on the active-head index — it is not enough that each Action has
    at most one active head of its own."""
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    _, source_record_id = await _seed_connector_source(session, sfx)
    action_a = await _seed_action(session, event_id, sfx, f"fp-a-{sfx}")
    action_b = await _seed_action(session, event_id, sfx + "b", f"fp-b-{sfx}")

    def _head(oid: str, action_id: str, seq: int) -> m.DispositionOutbox:
        return m.DispositionOutbox(
            outbox_id=oid,
            writeback_id=f"wbk-{oid}",
            disposition_id=f"disp-{oid}",
            action_id=action_id,
            event_id=event_id,
            closure_cycle=1,
            source_record_id=source_record_id,
            source_locator_hash="hash",
            source_sequence=seq,
            intent_kind="event_status_update",
            logical_slot="terminal",
            supersedes_disposition_id=None,
            superseded_by_disposition_id=None,
            idempotency_key=f"idem-{oid}",
            command_payload={"op": "set_event_disposition"},
            command_payload_sha256="sha",
        )

    session.add(_head(f"ha-{sfx}", action_a, 1))
    await session.flush()
    # A *different* action claiming an active head for the same
    # event/closure_cycle/slot must be rejected, even though action_b itself
    # has no other active head.
    session.add(_head(f"hb-{sfx}", action_b, 2))
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_receipt_writeback_sequence_pk(session: AsyncSession) -> None:
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    _, source_record_id = await _seed_connector_source(session, sfx)
    action_id = await _seed_action(session, event_id, sfx, f"fp-{sfx}")

    def _receipt(seq: int, status: str) -> m.DispositionReceipt:
        return m.DispositionReceipt(
            writeback_id=f"wbk-{sfx}",
            sequence=seq,
            disposition_id=f"disp-{sfx}",
            action_id=action_id,
            source_record_id=source_record_id,
            status=status,
        )

    session.add(_receipt(1, "sending"))
    await session.flush()
    session.add(_receipt(2, "confirmed"))  # different sequence is fine
    await session.flush()
    session.add(_receipt(1, "unknown"))  # duplicate (writeback_id, sequence)
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_row_version_cas(session: AsyncSession) -> None:
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    await session.commit()

    # optimistic update from version 1 -> 2 succeeds
    res = await session.execute(
        update(m.SecurityEvent)
        .where(m.SecurityEvent.event_id == event_id, m.SecurityEvent.row_version == 1)
        .values(status="triaging", row_version=2)
    )
    assert res.rowcount == 1

    # a stale writer still using version 1 matches no rows (CAS miss)
    res_stale = await session.execute(
        update(m.SecurityEvent)
        .where(m.SecurityEvent.event_id == event_id, m.SecurityEvent.row_version == 1)
        .values(status="analyzing", row_version=2)
    )
    assert res_stale.rowcount == 0
    await session.rollback()


async def test_source_checkpoint_identity_is_connector_and_kind(
    session: AsyncSession,
) -> None:
    sfx = _sfx()
    connector_id, _ = await _seed_connector_source(session, sfx)
    session.add(
        m.SourceCheckpoint(
            connector_id=connector_id,
            object_kind="incident",
            schema_version="1",
            watermark={"cursor": "incident"},
        )
    )
    session.add(
        m.SourceCheckpoint(
            connector_id=connector_id,
            object_kind="alert",
            schema_version="1",
            watermark={"cursor": "alert"},
        )
    )
    session.add(
        m.SourceCheckpoint(
            connector_id=connector_id,
            object_kind="incident",
            stream_scope="file:scenario-b",
            schema_version="1",
            watermark={"cursor": "scenario-b"},
        )
    )
    await session.flush()
    duplicate = m.SourceCheckpoint(
        connector_id=connector_id,
        object_kind="incident",
        schema_version="1",
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


def test_checkpoint_upgrade_does_not_backfill_legacy_global_watermark(
    migrated: None,
) -> None:
    cfg = _alembic_config()
    command.downgrade(cfg, "0003_outbox_active_head_evt")
    connector_id = f"legacy-{_sfx()}"

    async def _seed_legacy_connector() -> None:
        engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO source_connector "
                        "(connector_id, source_product, display_name, status, capabilities, "
                        "watermark, schema_version, connector_metadata) "
                        "VALUES (:connector_id, 'mock_xdr', 'Legacy', 'online', "
                        "'{}'::jsonb, '{\"cursor\":\"unsafe-global\"}'::jsonb, '1', "
                        "'{}'::jsonb)"
                    ),
                    {"connector_id": connector_id},
                )
        finally:
            await engine.dispose()

    asyncio.run(_seed_legacy_connector())
    command.upgrade(cfg, "head")

    async def _assert_no_backfill() -> None:
        engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                checkpoint_count = await conn.scalar(
                    text(
                        "SELECT count(*) FROM source_checkpoint WHERE connector_id = :connector_id"
                    ),
                    {"connector_id": connector_id},
                )
                watermark = await conn.scalar(
                    text(
                        "SELECT watermark FROM source_connector WHERE connector_id = :connector_id"
                    ),
                    {"connector_id": connector_id},
                )
                assert checkpoint_count == 0
                assert watermark == {"cursor": "unsafe-global"}
                await conn.execute(
                    text("DELETE FROM source_connector WHERE connector_id = :connector_id"),
                    {"connector_id": connector_id},
                )
        finally:
            await engine.dispose()

    asyncio.run(_assert_no_backfill())


async def test_transaction_rollback(session: AsyncSession) -> None:
    sfx = _sfx()
    event_id = await _seed_event(session, sfx)
    await session.rollback()
    found = await session.execute(
        select(func.count())
        .select_from(m.SecurityEvent)
        .where(m.SecurityEvent.event_id == event_id)
    )
    assert found.scalar_one() == 0


def test_downgrade_base_then_upgrade_head_roundtrip(migrated: None) -> None:
    # Sync test: Alembic runs its own event loop, so it must not be called from
    # inside a running (async test) loop. Prove a full rollback works, then
    # restore head for any following tests.
    cfg = _alembic_config()
    command.downgrade(cfg, "base")

    async def _remaining_core_tables() -> set[str]:
        engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                rows = await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name <> 'alembic_version'"
                    )
                )
                return {r[0] for r in rows} & CORE_TABLES
        finally:
            await engine.dispose()

    assert asyncio.run(_remaining_core_tables()) == set()
    command.upgrade(cfg, "head")
