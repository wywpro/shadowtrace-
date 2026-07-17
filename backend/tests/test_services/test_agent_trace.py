"""AgentTraceService persistence, projection, and BaseAgent integration (ISSUE-028)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.agents.base import BaseAgent
from app.db import models as orm
from app.models.agent_io import TriageAgentInput
from app.services.agent_trace_service import (
    MAX_AUDIT_FIELD_BYTES,
    AgentTraceService,
    TraceProjection,
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
async def clean_tables(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    async with session_factory() as session:
        async with session.begin():
            await session.execute(delete(orm.AgentTrace))
            await session.execute(delete(orm.EventAuditLog))
    yield
    async with session_factory() as session:
        async with session.begin():
            await session.execute(delete(orm.AgentTrace))
            await session.execute(delete(orm.EventAuditLog))


@pytest.fixture
def service(
    session_factory: async_sessionmaker[AsyncSession],
) -> AgentTraceService:
    return AgentTraceService(session_factory)


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# TraceProjection tests
# --------------------------------------------------------------------------- #


class _NestedModel(BaseModel):
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class _SampleOutput(BaseModel):
    event_id: str
    summary: str = ""
    evidence_list: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    nested: _NestedModel = Field(default_factory=_NestedModel)
    password: str = ""
    token: str = ""
    raw_data: dict[str, Any] = Field(default_factory=dict)


def test_projection_strips_raw_payload_keys() -> None:
    output = _SampleOutput(
        event_id="evt-20260717-a1b2c3d4",
        summary="test summary",
        nested=_NestedModel(
            raw_payload={"secret": "s3cret", "data": "important"},
            reasoning="the attacker used phishing",
        ),
        password="my-password",
        token="my-token",
        raw_data={"binary": b"\x00\x01"},
    )
    projected = TraceProjection.project(output)

    assert projected["event_id"] == "evt-20260717-a1b2c3d4"
    assert projected["password"] == "[REDACTED]"
    assert projected["token"] == "[REDACTED]"

    raw_data_block = projected["raw_data"]
    assert raw_data_block["_redacted"] is True
    assert raw_data_block["reason"] == "raw_block"
    assert len(raw_data_block["sha256"]) == 64

    nested = projected["nested"]
    nested_raw = nested["raw_payload"]
    assert nested_raw["_redacted"] is True
    assert nested_raw["reason"] == "raw_block"
    assert nested["reasoning"] == "the attacker used phishing"


def test_decision_basis_extracts_structured_summary() -> None:
    output = _SampleOutput(
        event_id="evt-20260717-a1b2c3d4",
        summary="critical data exfiltration detected",
        evidence_list=[
            {"evidence_id": "evd-aaaaaaaa"},
            {"evidence_id": "evd-bbbbbbbb"},
        ],
        confidence=0.95,
        nested=_NestedModel(
            reasoning="high confidence threat",
        ),
    )
    basis = TraceProjection.decision_basis(output)

    assert basis["input_summary"] == "evt-20260717-a1b2c3d4"
    assert basis["structured_conclusion"] == "critical data exfiltration detected"
    assert "evd-aaaaaaaa" in basis["evidence_refs"]
    assert "evd-bbbbbbbb" in basis["evidence_refs"]
    assert basis["confidence"] == 0.95


def test_oversized_field_is_truncated_to_hash_marker() -> None:
    oversized = "x" * (MAX_AUDIT_FIELD_BYTES + 2_048)
    projected = TraceProjection.project({"key": oversized})

    assert projected["_truncated"] is True
    assert projected["original_size_bytes"] > MAX_AUDIT_FIELD_BYTES
    assert len(projected["sha256"]) == 64
    assert "top_level_keys" in projected
    assert oversized not in str(projected)


# --------------------------------------------------------------------------- #
# AgentTraceService tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_log_trace_persists_and_is_queryable(
    service: AgentTraceService,
) -> None:
    event_id = _id("evt")
    started_at = datetime(2026, 7, 17, 10, 0, 0, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=1_500)

    trace_id = await service.log_trace(
        event_id=event_id,
        agent_name="triage_agent",
        input_data={"event_id": event_id, "summary": "test input"},
        output_data={"event_type": "data_exfiltration", "severity": "high"},
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        llm_model="mock-model",
        llm_tokens_used=150,
    )

    assert trace_id.startswith("trc-")
    assert len(trace_id) == 12  # "trc-" + 8 hex

    row = await service.get_trace(trace_id)
    assert row is not None
    assert row.event_id == event_id
    assert row.agent_name == "triage_agent"
    assert row.status == "completed"
    assert row.started_at == started_at
    assert row.completed_at == completed_at
    assert row.duration_ms == 1_500
    assert row.llm_model == "mock-model"
    assert row.llm_tokens_used == 150
    assert row.error_detail is None
    assert "_decision_basis" in row.output_data


@pytest.mark.asyncio
async def test_failed_trace_with_error_detail(
    service: AgentTraceService,
) -> None:
    event_id = _id("evt")
    started_at = datetime(2026, 7, 17, 11, 0, 0, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=300)

    trace_id = await service.log_trace(
        event_id=event_id,
        agent_name="evidence_agent",
        input_data={"event_id": event_id},
        output_data=None,
        status="failed",
        started_at=started_at,
        completed_at=completed_at,
        error_detail="Connection timed out: Authorization: Bearer secret-token-12345",
    )

    row = await service.get_trace(trace_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_detail is not None
    assert "secret-token-12345" not in row.error_detail
    original = "Connection timed out: Authorization: Bearer secret-token-12345"
    assert "[REDACTED]" in row.error_detail or row.error_detail != original


@pytest.mark.asyncio
async def test_traces_ordered_by_started_at_asc(
    service: AgentTraceService,
) -> None:
    event_id = _id("evt")
    base = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)
    trace_ids: list[str] = []

    for i, offset_s in enumerate((3, 1, 5)):
        started = base + timedelta(seconds=offset_s)
        trace_id = await service.log_trace(
            event_id=event_id,
            agent_name=f"agent_{i}",
            input_data={},
            output_data={},
            status="completed",
            started_at=started,
            completed_at=started + timedelta(seconds=1),
        )
        trace_ids.append(trace_id)

    rows = await service.get_traces_by_event(event_id)
    assert len(rows) == 3
    assert [r.trace_id for r in rows] == [trace_ids[1], trace_ids[0], trace_ids[2]]


@pytest.mark.asyncio
async def test_get_trace_returns_none_for_missing(
    service: AgentTraceService,
) -> None:
    result = await service.get_trace("trc-deadbeef")
    assert result is None


@pytest.mark.asyncio
async def test_get_traces_by_event_empty(
    service: AgentTraceService,
) -> None:
    rows = await service.get_traces_by_event("evt-no-such-event")
    assert rows == []


# --------------------------------------------------------------------------- #
# BaseAgent integration tests
# --------------------------------------------------------------------------- #


class _FakeSuccessOutput(BaseModel):
    verdict: str
    confidence: float


class _FakeSuccessAgent(BaseAgent[TriageAgentInput, _FakeSuccessOutput]):
    agent_name = "triage_agent"

    async def _run(self, input: TriageAgentInput) -> _FakeSuccessOutput:
        return _FakeSuccessOutput(verdict="confirmed_threat", confidence=0.92)


class _FakeFailingAgent(BaseAgent[TriageAgentInput, _FakeSuccessOutput]):
    agent_name = "triage_agent"

    async def _run(self, input: TriageAgentInput) -> _FakeSuccessOutput:
        raise RuntimeError("simulated agent crash")


class _FakeWrongNameAgent(BaseAgent[TriageAgentInput, _FakeSuccessOutput]):
    agent_name = "risk_agent"

    async def _run(self, input: TriageAgentInput) -> _FakeSuccessOutput:
        return _FakeSuccessOutput(verdict="ok", confidence=0.5)


@pytest.mark.asyncio
async def test_agent_success_writes_trace(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    trace_svc = AgentTraceService(session_factory)
    agent = _FakeSuccessAgent(trace_service=trace_svc)
    input = TriageAgentInput(event_id=_id("evt"))

    output = await agent.execute(input)

    assert output.verdict == "confirmed_threat"
    assert output.confidence == 0.92

    traces = await trace_svc.get_traces_by_event(input.event_id)
    assert len(traces) == 1
    assert traces[0].agent_name == "triage_agent"
    assert traces[0].status == "completed"
    assert traces[0].duration_ms is not None
    assert traces[0].duration_ms >= 0
    assert traces[0].started_at is not None
    assert traces[0].completed_at is not None
    assert "[REDACTED]" not in (traces[0].error_detail or "")


@pytest.mark.asyncio
async def test_agent_failure_writes_failed_trace(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    trace_svc = AgentTraceService(session_factory)
    agent = _FakeFailingAgent(trace_service=trace_svc)
    input = TriageAgentInput(event_id=_id("evt"))

    with pytest.raises(RuntimeError, match="simulated agent crash"):
        await agent.execute(input)

    traces = await trace_svc.get_traces_by_event(input.event_id)
    assert len(traces) == 1
    assert traces[0].agent_name == "triage_agent"
    assert traces[0].status == "failed"
    assert traces[0].error_detail == "simulated agent crash"


@pytest.mark.asyncio
async def test_agent_without_trace_service_does_not_crash(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    agent = _FakeSuccessAgent()  # No trace_service injected
    input = TriageAgentInput(event_id=_id("evt"))

    output = await agent.execute(input)
    assert output.verdict == "confirmed_threat"


@pytest.mark.asyncio
async def test_agent_wrong_input_type_raises_before_trace() -> None:
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        trace_svc = AgentTraceService(async_sessionmaker(bind=engine))
        agent = _FakeWrongNameAgent(trace_service=trace_svc)
        input = TriageAgentInput(event_id=_id("evt"))

        with pytest.raises(TypeError, match="requires RiskAgentInput"):
            await agent.execute(input)
    finally:
        await engine.dispose()
