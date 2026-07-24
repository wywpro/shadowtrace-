"""Real PostgreSQL/Redis fixtures for integration and ISSUE-039 e2e_basic tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.adapters.mock_xdr import MockXDRSourceAdapter
from app.agents.evidence_agent import EVIDENCE_QUERY_ORDER, EvidenceAgent
from app.agents.rag_agent import RAGAgent
from app.agents.report_agent import ReportAgent
from app.agents.risk_agent import RiskAgent
from app.agents.triage_agent import TriageAgent
from app.core.config import Settings, get_settings
from app.core.guardrails import OutputGuard, WorkingMemoryGuardViolationWriter
from app.core.llm.base import InMemoryLLMCallAuditRecorder
from app.core.llm.mock_client import MockLLMClient
from app.core.redis_client import RedisClient
from app.data_generators.scenarios import build_scenario, write_scenario_artifacts
from app.db.base import Base
from app.ingestion.source_ingester import SourceIngester
from app.mock_xdr.api import create_app
from app.mock_xdr.state import MockXDRState
from app.models.tool_meta import ToolResult, ToolResultStatus
from app.services.agent_trace_service import AgentTraceService
from app.services.analysis_only_pipeline import AnalysisOnlyPipeline
from app.services.budget_service import BudgetService, WorkingMemoryBudgetUsageWriter
from app.services.context_service import EventContextStore
from app.services.degraded_flag_service import DegradedFlagService
from app.services.event_audit_log_service import EventAuditLogService
from app.services.event_service import EventService
from app.services.evidence_projection import EvidenceProjection, bind_evidence_projection
from app.services.state_machine_service import StateMachineService
from app.services.working_memory import WorkingMemory
from tests.test_tools.tool_system_fixtures import new_sfx

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
BUSINESS_TABLES = tuple(sorted(Base.metadata.tables))


def _alembic_config() -> Config:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return config


@pytest.fixture(scope="session")
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


@pytest_asyncio.fixture
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[RedisClient]:
    client = RedisClient(url=REDIS_URL)
    if not await client.ping():
        await client.aclose()
        pytest.fail("Redis is required for integration tests; run `make integration-test`")
    yield client
    await client.aclose()


async def _truncate_business_tables(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    quoted = ", ".join(f'"{table}"' for table in BUSINESS_TABLES)
    async with session_factory() as session:
        async with session.begin():
            await session.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))


async def _clear_shadowtrace_keys(redis_client: RedisClient) -> None:
    try:
        client = redis_client.get_client()
        keys = [key async for key in client.scan_iter(match="shadowtrace:*", count=500)]
        if keys:
            await client.delete(*keys)
    except RuntimeError:
        # TestClient may close the asyncio loop before fixture teardown runs.
        pass


@pytest_asyncio.fixture
async def clean_state(
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: RedisClient,
) -> AsyncIterator[None]:
    """Reset PG/Redis around a test.

    Not autouse: ``tool_system`` chains in this package are in-memory and must
    not pull Dockerized Postgres/Redis. Real ``@pytest.mark.integration``
    modules opt in via ``pytest.mark.usefixtures("clean_state")``.
    """
    await _truncate_business_tables(session_factory)
    await _clear_shadowtrace_keys(redis_client)
    yield
    await _clear_shadowtrace_keys(redis_client)
    await _truncate_business_tables(session_factory)


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    target = tmp_path / "mock-data"
    scenario = build_scenario("insider_data_exfiltration", seed=42)
    write_scenario_artifacts(scenario, target)
    return target


@pytest.fixture
def mock_xdr_state() -> MockXDRState:
    state = MockXDRState()
    state.load_scenario(build_scenario("insider_data_exfiltration", seed=42))
    return state


@pytest_asyncio.fixture
async def mock_xdr_client(
    mock_xdr_state: MockXDRState,
) -> AsyncIterator[httpx.AsyncClient]:
    transport = ASGITransport(app=create_app(state=mock_xdr_state))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://mock-xdr",
        timeout=30.0,
    ) as client:
        yield client


@pytest.fixture
def source_adapter(mock_xdr_client: httpx.AsyncClient) -> MockXDRSourceAdapter:
    return MockXDRSourceAdapter(
        base_url="http://mock-xdr",
        read_token="mock-read-token",
        write_token="mock-write-token",
        client=mock_xdr_client,
        max_retries=0,
    )


@pytest.fixture
def context_store(
    redis_client: RedisClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> EventContextStore:
    return EventContextStore(redis_client, session_factory)


@pytest.fixture
def degraded_flags(
    context_store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> DegradedFlagService:
    return DegradedFlagService(context_store, session_factory)


@pytest.fixture
def audit_log(
    session_factory: async_sessionmaker[AsyncSession],
) -> EventAuditLogService:
    return EventAuditLogService(session_factory)


@pytest.fixture
def state_machine_service(
    session_factory: async_sessionmaker[AsyncSession],
    context_store: EventContextStore,
    audit_log: EventAuditLogService,
    degraded_flags: DegradedFlagService,
) -> StateMachineService:
    return StateMachineService(
        session_factory,
        context_store,
        audit_log=audit_log,
        degraded_flags=degraded_flags,
    )


@pytest.fixture
def event_service(
    context_store: EventContextStore,
    session_factory: async_sessionmaker[AsyncSession],
    degraded_flags: DegradedFlagService,
    state_machine_service: StateMachineService,
) -> EventService:
    return EventService(
        session_factory,
        context_store,
        degraded_flags=degraded_flags,
        state_machine=state_machine_service,
    )


@pytest.fixture
def source_ingester(
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> SourceIngester:
    return SourceIngester(
        event_service,
        session_factory,
        source_mode="mock_xdr",
    )


@pytest.fixture
def e2e_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Mock/offline settings required by AnalysisOnlyPipeline."""
    monkeypatch.setenv("SOURCE_MODE", "mock_xdr")
    monkeypatch.setenv("DISPOSITION_MODE", "mock_xdr")
    monkeypatch.setenv("ALLOW_LIVE_SIDE_EFFECTS", "false")
    monkeypatch.setenv("ALLOW_XDR_WRITEBACK", "false")
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("BUDGET_ENABLED", "true")
    get_settings.cache_clear()
    settings = Settings(
        SOURCE_MODE="mock_xdr",
        DISPOSITION_MODE="mock_xdr",
        ALLOW_LIVE_SIDE_EFFECTS=False,
        ALLOW_XDR_WRITEBACK=False,
        LLM_MODE="mock",
        BUDGET_ENABLED=True,
    )
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def working_memory(
    context_store: EventContextStore,
    redis_client: RedisClient,
    degraded_flags: DegradedFlagService,
) -> WorkingMemory:
    return WorkingMemory(store=context_store, redis=redis_client, degraded_flags=degraded_flags)


@pytest.fixture
def mock_llm_client(budget_service: BudgetService) -> MockLLMClient:
    return MockLLMClient(
        audit_recorder=InMemoryLLMCallAuditRecorder(),
        budget_service=budget_service,
    )


@pytest.fixture
def agent_trace_service(
    session_factory: async_sessionmaker[AsyncSession],
) -> AgentTraceService:
    return AgentTraceService(session_factory)


@pytest.fixture
def budget_service(
    redis_client: RedisClient,
    working_memory: WorkingMemory,
    e2e_settings: Settings,
) -> BudgetService:
    writer = WorkingMemoryBudgetUsageWriter(working_memory)
    return BudgetService(redis=redis_client, usage_writer=writer, settings=e2e_settings)


@pytest.fixture
def output_guard(working_memory: WorkingMemory) -> OutputGuard:
    return OutputGuard(
        violation_writer=WorkingMemoryGuardViolationWriter(working_memory),
    )


class FlakyToolExecutor:
    """Force selected query tools to fail while delegating others."""

    def __init__(self, inner: Any, fail_tools: set[str]) -> None:
        self._inner = inner
        self._fail_tools = fail_tools

    async def call(
        self,
        tool_name: str,
        params: dict[str, Any],
        event_id: str,
        **kwargs: Any,
    ) -> ToolResult:
        if tool_name in self._fail_tools:
            return ToolResult(
                call_id=f"call-fail-{new_sfx()}",
                tool_name=tool_name,
                provider_name="test",
                status=ToolResultStatus.FAILED,
                error_detail=f"forced failure for {tool_name}",
                execution_time_ms=3,
            )
        return await self._inner.call(tool_name, params, event_id, **kwargs)


class FailingLLMClient:
    """Always-fail LLM stub for degradation scenarios."""

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("llm unavailable")


@pytest.fixture
def e2e_tool_executor(tool_executor: Any, budget_service: BudgetService) -> Any:
    tool_executor.budget_service = budget_service
    return tool_executor


@pytest.fixture
def build_analysis_pipeline(
    session_factory: async_sessionmaker[AsyncSession],
    event_service: EventService,
    state_machine_service: StateMachineService,
    context_store: EventContextStore,
    degraded_flags: DegradedFlagService,
    working_memory: WorkingMemory,
    mock_llm_client: MockLLMClient,
    budget_service: BudgetService,
    output_guard: OutputGuard,
    agent_trace_service: AgentTraceService,
    e2e_settings: Settings,
    e2e_tool_executor: Any,
) -> Callable[..., tuple[AnalysisOnlyPipeline, EvidenceProjection]]:
    """Factory for ISSUE-039 analysis-only pipelines."""

    def _build(
        *,
        llm_client: Any | None = None,
        fail_tools: set[str] | None = None,
        scenario_id: str | None = "insider_data_exfiltration",
        evidence_mode: str = "sequential",
    ) -> tuple[AnalysisOnlyPipeline, EvidenceProjection]:
        effective_llm = mock_llm_client if llm_client is None else llm_client
        effective_executor = e2e_tool_executor
        if fail_tools:
            effective_executor = FlakyToolExecutor(e2e_tool_executor, fail_tools)

        triage = TriageAgent(
            llm_client=effective_llm,
            working_memory=working_memory.for_writer("TriageAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=agent_trace_service,
        )
        evidence = EvidenceAgent(
            llm_client=effective_llm,
            tool_executor=effective_executor,
            working_memory=working_memory.for_writer("EvidenceAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=agent_trace_service,
            event_service=event_service,
            session_factory=session_factory,
            evidence_mode=evidence_mode,
        )
        rag = RAGAgent(
            working_memory=working_memory.for_writer("RAGAgent"),
            pipeline=None,
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=agent_trace_service,
        )
        risk = RiskAgent(
            llm_client=effective_llm,
            working_memory=working_memory.for_writer("RiskAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=agent_trace_service,
            event_service=event_service,
            scenario_id=scenario_id,
        )
        report = ReportAgent(
            llm_client=effective_llm,
            working_memory=working_memory.for_writer("ReportAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=agent_trace_service,
            event_service=event_service,
            scenario_id=scenario_id,
        )
        pipeline = AnalysisOnlyPipeline(
            event_service=event_service,
            state_machine=state_machine_service,
            triage_agent=triage,
            evidence_agent=evidence,
            rag_agent=rag,
            risk_agent=risk,
            report_agent=report,
            context_store=context_store,
            degraded_flags=degraded_flags,
            settings=e2e_settings,
        )
        projection = EvidenceProjection(session_factory)
        return pipeline, projection

    return _build


@pytest.fixture
def run_analysis_pipeline(
    build_analysis_pipeline: Callable[..., tuple[AnalysisOnlyPipeline, EvidenceProjection]],
) -> Callable[..., Any]:
    """Run the pipeline with the PG-backed evidence projection bound."""

    async def _run(
        event_id: str,
        *,
        llm_client: Any | None = None,
        fail_tools: set[str] | None = None,
        scenario_id: str | None = "insider_data_exfiltration",
    ) -> Any:
        pipeline, projection = build_analysis_pipeline(
            llm_client=llm_client,
            fail_tools=fail_tools,
            scenario_id=scenario_id,
        )
        with bind_evidence_projection(projection):
            return await pipeline.run(event_id)

    return _run


DEFAULT_PARTIAL_FAIL_TOOLS = frozenset(
    {
        "query_dns",
        "query_asset_info",
        "query_threat_intel",
    }
)
assert DEFAULT_PARTIAL_FAIL_TOOLS.issubset(set(EVIDENCE_QUERY_ORDER))
