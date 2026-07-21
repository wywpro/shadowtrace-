"""ToolExecutor tests (ISSUE-024)."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.errors import ToolExecutionError
from app.models.enums import (
    ActionCategory,
    DispositionIntentKind,
    ExecutionJobStatus,
    ExecutionOwner,
    ToolCategory,
)
from app.models.execution import ActionExecutionJob
from app.models.tool_meta import (
    ExecutionChannel,
    ProviderToolBinding,
    RoutingKind,
    SideEffectLevel,
    ToolMeta,
    ToolResult,
    ToolResultStatus,
    WrongExecutionChannelError,
)
from app.models.workflow import MAX_DUPLICATE_TOOL_CALLS
from app.providers.tools.mock_provider import MockToolProvider, bind_mock_tool_provider
from app.services.tool_call_log_service import ToolCallLogService
from app.tools.circuit_breaker import CircuitBreakerRegistry
from app.tools.executor import InMemoryExecutionJobStore, ToolExecutor, derive_call_nature
from app.tools.mock_state import MockEnvironmentState
from app.tools.registry import ToolRegistry, ToolValidationError
from app.tools.retry import RetryPolicy

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)


def _sfx() -> str:
    return uuid.uuid4().hex[:8]


def _query_meta(name: str, *, timeout_s: float = 5.0) -> ToolMeta:
    return ToolMeta(
        tool_name=name,
        tool_category=ToolCategory.QUERY,
        routing_kind=RoutingKind.TOOL_PROVIDER_ONLY,
        default_timeout_s=timeout_s,
        input_schema={
            "type": "object",
            "properties": {
                "delay_s": {"type": "number"},
                "fail_times": {"type": "integer"},
                "mode": {"type": "string"},
            },
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "call_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "provider_name": {"type": "string"},
                "status": {"type": "string"},
                "data": {"type": "object"},
            },
            "required": ["call_id", "tool_name", "provider_name", "status", "data"],
            "additionalProperties": True,
        },
    )


class _FakeToolState:
    def __init__(self) -> None:
        self.attempts: dict[str, int] = {}


def _make_registry(state: _FakeToolState) -> ToolRegistry:
    registry = ToolRegistry()

    async def ok_execute(params: dict[str, Any]) -> dict[str, Any]:
        delay = float(params.get("delay_s", 0))
        if delay:
            await asyncio.sleep(delay)
        return ToolResult(
            call_id=f"call-internal-{_sfx()}",
            tool_name="fake_ok",
            provider_name="fake",
            status=ToolResultStatus.SUCCESS,
            data={"ok": True, **params},
        ).model_dump(mode="json")

    async def flaky_execute(params: dict[str, Any]) -> dict[str, Any]:
        tool = "fake_flaky"
        state.attempts[tool] = state.attempts.get(tool, 0) + 1
        fail_times = int(params.get("fail_times", 1))
        if state.attempts[tool] <= fail_times:
            raise ToolExecutionError("transient provider fault")
        return ToolResult(
            call_id=f"call-internal-{_sfx()}",
            tool_name=tool,
            provider_name="fake",
            status=ToolResultStatus.SUCCESS,
            data={"attempt": state.attempts[tool]},
        ).model_dump(mode="json")

    async def slow_execute(params: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(float(params.get("delay_s", 0.2)))
        return ToolResult(
            call_id=f"call-internal-{_sfx()}",
            tool_name="fake_slow",
            provider_name="fake",
            status=ToolResultStatus.SUCCESS,
            data={"done": True},
        ).model_dump(mode="json")

    registry.register(_query_meta("fake_ok"), ok_execute)
    registry.register(_query_meta("fake_flaky"), flaky_execute)
    registry.register(_query_meta("fake_slow", timeout_s=0.05), slow_execute)
    return registry


class RecordingConvergenceGuard:
    def __init__(self) -> None:
        self.steps: list[tuple[str, str, dict[str, Any] | None]] = []

    async def record_step(
        self,
        event_id: str,
        *,
        tool_name: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.steps.append((event_id, tool_name, params))

    async def should_stop(self, event_id: str) -> bool:
        return False


class RecordingAuditService:
    def __init__(self) -> None:
        self.starts = 0
        self.finishes = 0
        self.rows: dict[str, dict[str, Any]] = {}

    async def log_start(
        self,
        call_id: str,
        event_id: str,
        action_id: str | None,
        tool_name: str,
        tool_category: str,
        parameters: dict[str, Any] | None,
    ) -> str:
        self.starts += 1
        self.rows[call_id] = {
            "event_id": event_id,
            "tool_name": tool_name,
            "parameters": parameters or {},
        }
        return call_id

    async def log_finish(
        self,
        call_id: str,
        status: str,
        result: dict[str, Any] | None,
        error_detail: str | None,
        retry_count: int,
    ) -> None:
        self.finishes += 1
        row = self.rows.setdefault(call_id, {})
        row.update(
            {
                "status": status,
                "result": result or {},
                "error_detail": error_detail,
                "retry_count": retry_count,
            }
        )


@pytest.fixture
def fake_state() -> _FakeToolState:
    return _FakeToolState()


@pytest.fixture
def registry(fake_state: _FakeToolState) -> ToolRegistry:
    return _make_registry(fake_state)


@pytest.fixture
def audit() -> RecordingAuditService:
    return RecordingAuditService()


@pytest.fixture
def executor(registry: ToolRegistry, audit: RecordingAuditService) -> ToolExecutor:
    return ToolExecutor(registry=registry, audit_service=audit)


@pytest.mark.asyncio
async def test_normal_call_returns_success_and_one_audit_record(
    executor: ToolExecutor,
    audit: RecordingAuditService,
) -> None:
    event_id = f"evt-{_sfx()}"
    result = await executor.call(
        "fake_ok",
        {"mode": "brief"},
        event_id,
        timeout=2.0,
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert audit.starts == 1
    assert audit.finishes == 1
    assert len(audit.rows) == 1


@pytest.mark.asyncio
async def test_query_timeout_returns_timeout(executor: ToolExecutor) -> None:
    result = await executor.call(
        "fake_slow",
        {"delay_s": 0.2},
        f"evt-{_sfx()}",
        timeout=0.05,
        retry_policy=RetryPolicy(max_retries=0),
    )

    assert result.status is ToolResultStatus.TIMEOUT


@pytest.mark.asyncio
async def test_retry_backoff_uses_exponential_delays(
    registry: ToolRegistry,
    audit: RecordingAuditService,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    executor = ToolExecutor(
        registry=registry,
        audit_service=audit,
        sleep=fake_sleep,
    )
    result = await executor.call(
        "fake_flaky",
        {"fail_times": 2},
        f"evt-{_sfx()}",
        retry_policy=RetryPolicy(max_retries=3),
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert sleeps == [2.0, 4.0]


@pytest.mark.asyncio
async def test_circuit_opens_after_five_failures_and_half_open_recovers(
    registry: ToolRegistry,
    audit: RecordingAuditService,
) -> None:
    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    breaker_registry = CircuitBreakerRegistry(
        failure_threshold=5,
        recovery_timeout_s=60.0,
        clock=monotonic,
    )
    executor = ToolExecutor(
        registry=registry,
        audit_service=audit,
        breaker_registry=breaker_registry,
    )
    event_id = f"evt-{_sfx()}"

    for _ in range(5):
        result = await executor.call(
            "fake_flaky",
            {"fail_times": 999},
            event_id,
            retry_policy=RetryPolicy(max_retries=0),
        )
        assert result.status is ToolResultStatus.FAILED

    starts_before_block = audit.starts
    finishes_before_block = audit.finishes
    blocked = await executor.call(
        "fake_flaky",
        {"fail_times": 999},
        event_id,
        retry_policy=RetryPolicy(max_retries=0),
    )
    assert blocked.status is ToolResultStatus.CIRCUIT_OPEN
    assert audit.starts == starts_before_block + 1
    assert audit.finishes == finishes_before_block + 1

    clock["now"] = 60.0
    recovered = await executor.call(
        "fake_flaky",
        {"fail_times": 0},
        event_id,
        retry_policy=RetryPolicy(max_retries=0),
    )
    assert recovered.status is ToolResultStatus.SUCCESS


@pytest.mark.asyncio
async def test_convergence_guard_records_each_dispatch_attempt(
    registry: ToolRegistry,
) -> None:
    guard = RecordingConvergenceGuard()
    executor = ToolExecutor(
        registry=registry,
        convergence_guard=guard,
        sleep=lambda _delay: asyncio.sleep(0),
    )
    await executor.call(
        "fake_flaky",
        {"fail_times": 2},
        f"evt-{_sfx()}",
        retry_policy=RetryPolicy(max_retries=3),
    )

    assert len(guard.steps) == 3
    assert guard.steps[0][2] == {"fail_times": 2}


@pytest.mark.asyncio
async def test_convergence_guard_same_tool_different_params_no_duplicate_stop(
    registry: ToolRegistry,
) -> None:
    from app.orchestration.convergence_guard import ConvergenceGuard

    guard = ConvergenceGuard()
    executor = ToolExecutor(registry=registry, convergence_guard=guard)
    event_id = f"evt-{_sfx()}"
    for i in range(MAX_DUPLICATE_TOOL_CALLS):
        result = await executor.call("fake_flaky", {"fail_times": i}, event_id)
        assert result.status is ToolResultStatus.SUCCESS
    decision = await guard.should_stop(event_id)
    assert decision.stop is False


@pytest.mark.asyncio
async def test_virtual_tool_is_rejected() -> None:
    registry = ToolRegistry()
    registry.auto_discover()
    executor = ToolExecutor(registry=registry)

    with pytest.raises(WrongExecutionChannelError):
        await executor.call(
            "update_source_event_disposition",
            {},
            f"evt-{_sfx()}",
        )


@pytest_asyncio.fixture
async def state() -> MockEnvironmentState:
    store = MockEnvironmentState.in_memory()
    await store.clear_all()
    return store


@pytest.mark.asyncio
async def test_side_effect_requires_precreated_job_and_reuses_job_id(
    state: MockEnvironmentState,
) -> None:
    await state.clear_all()
    registry = ToolRegistry()
    registry.auto_discover()
    provider = MockToolProvider(state)
    provider.register_bindings(registry)

    job_store = InMemoryExecutionJobStore()
    job_id = f"job-precreated-{_sfx()}"
    event_id = f"evt-{_sfx()}"
    action_id = f"act-{_sfx()}"
    idem = f"idem-{_sfx()}"
    await job_store.seed_job(
        ActionExecutionJob(
            job_id=job_id,
            event_id=event_id,
            action_id=action_id,
            provider_name="mock_tool_provider",
            idempotency_key=idem,
            status=ExecutionJobStatus.QUEUED,
        )
    )

    executor = ToolExecutor(
        registry=registry,
        job_store=job_store,
        provider_context=lambda: bind_mock_tool_provider(provider),
    )
    result = await executor.call(
        "block_ip",
        {
            "target_type": "ip",
            "target": "203.0.113.55",
            "parameters": {"reason": "test"},
        },
        event_id,
        action_id=action_id,
        execution_job_id=job_id,
        idempotency_key=idem,
        execution_owner=ExecutionOwner.DIRECT_TOOL,
        timeout=5.0,
        retry_policy=RetryPolicy(max_retries=0),
    )

    assert result.job_id == job_id
    assert len(await state.list_namespace("jobs")) == 1


@pytest.mark.asyncio
async def test_side_effect_missing_envelope_is_rejected(registry: ToolRegistry) -> None:
    response_meta = ToolMeta(
        tool_name="fixture_response",
        tool_category=ToolCategory.RESPONSE,
        action_category=ActionCategory.RESPONSE,
        routing_kind=RoutingKind.OWNER_ROUTED,
        supported_execution_owners=[ExecutionOwner.DIRECT_TOOL],
        required_disposition_intent_by_owner={
            ExecutionOwner.DIRECT_TOOL: DispositionIntentKind.EXECUTION_RESULT_RECORD,
        },
        side_effect_level=SideEffectLevel.MEDIUM,
        input_schema={
            "type": "object",
            "properties": {
                "target_type": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["target_type", "target"],
        },
        output_schema={"type": "object"},
    )

    async def execute(params: dict[str, Any]) -> dict[str, Any]:
        return ToolResult(
            call_id="call-x",
            tool_name="fixture_response",
            provider_name="fake",
            status=ToolResultStatus.SUCCESS,
            data=params,
        ).model_dump(mode="json")

    registry.register(response_meta, execute)
    registry.register_binding(
        ProviderToolBinding(
            tool_name="fixture_response",
            provider_name="mock_tool_provider",
            execution_owner=ExecutionOwner.DIRECT_TOOL,
            execution_channel=ExecutionChannel.TOOL_PROVIDER,
            capabilities=["entity_response"],
        )
    )
    executor = ToolExecutor(registry=registry)

    with pytest.raises(ToolValidationError, match="requires"):
        await executor.call(
            "fixture_response",
            {"target_type": "ip", "target": "1.1.1.1"},
            "evt-1",
        )


@pytest_asyncio.fixture
async def audit_session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    from app.db import models as orm

    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(orm.ToolCallLog.__table__.create, checkfirst=True)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tool_call_log_service_integration(
    registry: ToolRegistry,
    audit_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    audit = ToolCallLogService(audit_session_factory)
    executor = ToolExecutor(registry=registry, audit_service=audit)
    event_id = f"evt-{_sfx()}"

    result = await executor.call("fake_ok", {}, event_id)
    assert result.status is ToolResultStatus.SUCCESS

    logs = await audit.get_logs_by_event(event_id)
    assert len(logs) == 1
    assert logs[0].status == "success"
    assert logs[0].completed_at is not None


def test_derive_call_nature_from_registry_meta() -> None:
    query = _query_meta("q")
    assert derive_call_nature(query).value == "query"
