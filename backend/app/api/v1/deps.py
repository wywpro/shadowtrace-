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
        from app.services.analysis_only_pipeline import AnalysisOnlyPipeline
        from app.tools.executor import get_tool_executor

        event_service = await get_event_service()
        state_machine = await get_state_machine()
        wm = await _get_wm()

        triage = TriageAgent(
            llm_client=None,
            working_memory=wm.for_writer("TriageAgent"),
        )
        evidence = EvidenceAgent(
            llm_client=None,
            tool_executor=get_tool_executor(),
            working_memory=wm.for_writer("EvidenceAgent"),
        )
        rag = RAGAgent(
            working_memory=wm.for_writer("RAGAgent"),
            pipeline=None,
        )
        risk = RiskAgent(
            llm_client=None,
            working_memory=wm.for_writer("RiskAgent"),
            event_service=event_service,
        )
        report = ReportAgent(
            llm_client=None,
            working_memory=wm.for_writer("ReportAgent"),
            event_service=event_service,
            event_bus=_get_event_bus(),
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
        )
    return _pipeline


def reset_deps() -> None:
    """Reset all lazy singletons (for tests)."""
    global _session_factory, _redis_client, _context_store, _degraded_flags
    global _audit_log, _event_service, _state_machine, _event_bus, _pipeline, _approval_engine
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
    from app.tools import executor as tool_executor_module

    tool_executor_module.tool_executor = None
