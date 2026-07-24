"""FastAPI dependency injection for services (ISSUE-038 / ISSUE-058).

Lazily creates singleton service instances from settings. Tests override
via ``app.dependency_overrides``.

IMPORTANT: All service imports are lazy (inside function bodies) to avoid
circular imports with ``app.api.v1.schemas`` → ``app.services.context_service``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.redis_client import RedisClient

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Lazy singletons
# --------------------------------------------------------------------------- #

_session_factory: async_sessionmaker[AsyncSession] | None = None
_redis_client: RedisClient | None = None
_context_store: Any = None  # EventContextStore
_degraded_flags: Any = None  # DegradedFlagService
_audit_log: Any = None  # EventAuditLogService
_event_service: Any = None  # EventService
_state_machine: Any = None  # StateMachineService
_event_bus: Any = None  # EventBus
_pipeline: Any = None  # AnalysisOnlyPipeline
_approval_engine: Any = None  # ApprovalEngine
_disposition_sync: Any = None  # DispositionSyncService
_action_execution: Any = None  # ActionExecutionService
_adapter_registry: Any = None  # DispositionAdapterRegistry
_tool_executor: Any = None  # ToolExecutor


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        settings = get_settings()
        engine = create_async_engine(settings.database_url, poolclass=NullPool)
        _session_factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    return _session_factory


def _get_redis() -> RedisClient:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = RedisClient(url=settings.redis_url)
    return _redis_client


def _get_context_store() -> Any:
    global _context_store
    if _context_store is None:
        from app.services.context_service import EventContextStore

        _context_store = EventContextStore(_get_redis(), _get_session_factory())
    return _context_store


def _get_degraded_flags() -> Any:
    global _degraded_flags
    if _degraded_flags is None:
        from app.services.degraded_flag_service import DegradedFlagService

        _degraded_flags = DegradedFlagService(_get_context_store(), _get_session_factory())
    return _degraded_flags


def _get_audit_log() -> Any:
    global _audit_log
    if _audit_log is None:
        from app.services.event_audit_log_service import EventAuditLogService

        _audit_log = EventAuditLogService(_get_session_factory())
    return _audit_log


def _get_event_bus() -> Any:
    global _event_bus
    if _event_bus is None:
        from app.core.event_bus import EventBus

        _event_bus = EventBus(_get_redis())
    return _event_bus


async def get_event_service() -> Any:
    global _event_service
    if _event_service is None:
        from app.services.event_service import EventService

        state_machine = await get_state_machine()
        _event_service = EventService(
            _get_session_factory(),
            _get_context_store(),
            degraded_flags=_get_degraded_flags(),
            state_machine=state_machine,
            event_bus=_get_event_bus(),
        )
    return _event_service


async def get_state_machine() -> Any:
    global _state_machine
    if _state_machine is None:
        from app.services.state_machine_service import StateMachineService

        _state_machine = StateMachineService(
            _get_session_factory(),
            _get_context_store(),
            audit_log=_get_audit_log(),
            degraded_flags=_get_degraded_flags(),
        )
    return _state_machine


async def get_approval_engine() -> Any:
    """Return the tiered approval engine singleton (ISSUE-058)."""
    global _approval_engine
    if _approval_engine is None:
        from app.services.approval_engine import ApprovalEngine

        state_machine = await get_state_machine()
        _approval_engine = ApprovalEngine(
            _get_session_factory(),
            event_bus=_get_event_bus(),
            state_machine=state_machine,
            context_store=_get_context_store(),
        )
    return _approval_engine


ApprovalEngineDep = Annotated[Any, Depends(get_approval_engine)]


def _get_adapter_registry() -> Any:
    global _adapter_registry
    if _adapter_registry is None:
        from app.adapters.mock_xdr import MockXDRDispositionAdapter
        from app.adapters.registry import DispositionAdapterRegistry

        settings = get_settings()
        registry = DispositionAdapterRegistry()
        base_url = settings.disposition_base_url or "http://mock-xdr"
        adapter = MockXDRDispositionAdapter(
            base_url=base_url,
            read_token="mock-read-token",
            write_token="mock-write-token",
        )
        registry.register("mock_xdr", adapter)
        _adapter_registry = registry
    return _adapter_registry


def _get_tool_executor() -> Any:
    global _tool_executor
    if _tool_executor is None:
        from app.tools.executor import ToolExecutor
        from app.tools.registry import tool_registry

        _tool_executor = ToolExecutor(registry=tool_registry)
    return _tool_executor


async def get_disposition_sync() -> Any:
    global _disposition_sync
    if _disposition_sync is None:
        from app.core.guardrails import OutboundDispositionGuard
        from app.services.disposition_sync_service import DispositionSyncService

        _disposition_sync = DispositionSyncService(
            _get_session_factory(),
            context_store=_get_context_store(),
            adapter_registry=_get_adapter_registry(),
            outbound_guard=OutboundDispositionGuard(),
            event_bus=_get_event_bus(),
        )
    return _disposition_sync


async def get_action_execution() -> Any:
    global _action_execution
    if _action_execution is None:
        from app.services.action_execution_service import ActionExecutionService

        state_machine = await get_state_machine()
        _action_execution = ActionExecutionService(
            _get_session_factory(),
            disposition_sync=await get_disposition_sync(),
            tool_executor=_get_tool_executor(),
            state_machine=state_machine,
            context_store=_get_context_store(),
            event_bus=_get_event_bus(),
        )
    return _action_execution


DispositionSyncDep = Annotated[Any, Depends(get_disposition_sync)]
ActionExecutionDep = Annotated[Any, Depends(get_action_execution)]


async def _get_wm() -> Any:
    """Return a shared WorkingMemory instance."""
    from app.services.working_memory import WorkingMemory

    return WorkingMemory(
        store=_get_context_store(),
        redis=_get_redis(),
        degraded_flags=_get_degraded_flags(),
    )


async def get_pipeline() -> Any:
    """Return AnalysisOnlyPipeline (lazy import)."""
    global _pipeline
    if _pipeline is None:
        from app.agents.evidence_agent import EvidenceAgent
        from app.agents.rag_agent import RAGAgent
        from app.agents.report_agent import ReportAgent
        from app.agents.risk_agent import RiskAgent
        from app.agents.triage_agent import TriageAgent
        from app.core.guardrails import OutputGuard
        from app.core.llm.factory import get_llm_client
        from app.services.agent_trace_service import AgentTraceService
        from app.services.analysis_only_pipeline import AnalysisOnlyPipeline
        from app.services.budget_service import BudgetService
        from app.tools.executor import get_tool_executor

        settings = get_settings()
        event_service = await get_event_service()
        state_machine = await get_state_machine()
        wm = await _get_wm()
        session_factory = _get_session_factory()
        budget_service = BudgetService(redis=_get_redis(), settings=settings)
        output_guard = OutputGuard()
        trace_service = AgentTraceService(session_factory)
        llm_client = get_llm_client(settings=settings, budget_service=budget_service)
        tool_executor = get_tool_executor()
        tool_executor.budget_service = budget_service

        triage = TriageAgent(
            llm_client=llm_client,
            working_memory=wm.for_writer("TriageAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
        )
        evidence = EvidenceAgent(
            llm_client=llm_client,
            tool_executor=tool_executor,
            working_memory=wm.for_writer("EvidenceAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
            event_service=event_service,
            session_factory=session_factory,
        )
        rag = RAGAgent(
            working_memory=wm.for_writer("RAGAgent"),
            pipeline=None,
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
        )
        risk = RiskAgent(
            llm_client=llm_client,
            working_memory=wm.for_writer("RiskAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
            event_service=event_service,
            scenario_id="insider_data_exfiltration",
        )
        report = ReportAgent(
            llm_client=llm_client,
            working_memory=wm.for_writer("ReportAgent"),
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
            event_service=event_service,
            event_bus=_get_event_bus(),
            scenario_id="insider_data_exfiltration",
        )

        _pipeline = AnalysisOnlyPipeline(
            event_service=event_service,
            state_machine=state_machine,
            triage_agent=triage,
            evidence_agent=evidence,
            rag_agent=rag,
            risk_agent=risk,
            report_agent=report,
            context_store=_get_context_store(),
            degraded_flags=_get_degraded_flags(),
            settings=settings,
        )
    return _pipeline


def reset_deps() -> None:
    """Reset all lazy singletons (for tests)."""
    global _session_factory, _redis_client, _context_store, _degraded_flags
    global _audit_log, _event_service, _state_machine, _event_bus, _pipeline, _approval_engine
    global _disposition_sync, _action_execution, _adapter_registry, _tool_executor
    _session_factory = None
    _redis_client = None
    _context_store = None
    _degraded_flags = None
    _audit_log = None
    _event_service = None
    _state_machine = None
    _event_bus = None
    _pipeline = None
    _approval_engine = None
    _disposition_sync = None
    _action_execution = None
    _adapter_registry = None
    _tool_executor = None
