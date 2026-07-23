"""ISSUE-039 analysis-only E2E integration tests (alert → report).

Four scenarios:
1. Golden path (mock_xdr insider scenario → REPORTING)
2. Low-risk not_required file fallback short-circuit → CLOSED
3. Data-source degradation (3 query tools fail → partial_done, still reports)
4. LLM degradation (regex triage, rule scoring, template report)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.mock_xdr import MockXDRSourceAdapter
from app.agents.report_agent import GENERATED_BY_TEMPLATE
from app.agents.report_section_builder import SECTION_KEYS
from app.agents.risk_agent import RiskAgent
from app.db import models as orm
from app.ingestion.file_ingester import FileIngester
from app.ingestion.source_ingester import SourceIngester
from app.models.agent_io import CollectionStatus, ScoringMode
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    SourceObjectKind,
)
from app.models.ids import report_id_for_event
from app.models.report import InvestigationReport
from app.services.agent_trace_service import AgentTraceService
from app.services.analysis_only_pipeline import AnalysisOnlyPipeline
from app.services.context_service import EventContextStore
from app.services.event_service import EventService
from app.services.evidence_projection import bind_evidence_projection
from tests.integration.conftest import (
    DEFAULT_PARTIAL_FAIL_TOOLS,
    FailingLLMClient,
)
from tests.integration.rag_scenario_fixtures import (
    FakeEventService,
    FakeWorkingMemory,
    main_evidence,
)

pytestmark = pytest.mark.e2e_basic

ALL_SOURCE_KINDS = [
    SourceObjectKind.INCIDENT,
    SourceObjectKind.ALERT,
    SourceObjectKind.ASSET,
    SourceObjectKind.LOG,
]

GOLDEN_STATUS_SEQUENCE = (
    EventStatus.NEW,
    EventStatus.TRIAGING,
    EventStatus.COLLECTING_EVIDENCE,
    EventStatus.ANALYZING,
    EventStatus.SCORING,
    EventStatus.REPORTING,
)

PIPELINE_AGENT_NAMES = frozenset(
    {
        "triage_agent",
        "evidence_agent",
        "rag_agent",
        "risk_agent",
        "report_agent",
    }
)

SHORT_CIRCUIT_AGENT_NAMES = frozenset({"triage_agent", "report_agent"})

GOLDEN_PATH_MAX_SECONDS = 60.0

SHORT_CIRCUIT_STATUS_SEQUENCE = (
    EventStatus.TRIAGING,
    EventStatus.CLOSED,
)


class _StubAgent:
    def __init__(self, result: Any) -> None:
        self.result = result

    async def execute(self, input: Any) -> Any:
        return self.result


class _FailingRAGAgent:
    async def execute(self, input: Any) -> Any:
        raise RuntimeError("rag unavailable")


async def _ingest_main_scenario(
    source_adapter: MockXDRSourceAdapter,
    source_ingester: SourceIngester,
    event_service: EventService,
) -> str:
    summary = await source_ingester.poll(source_adapter, ALL_SOURCE_KINDS, batch_size=10)
    assert summary.rejected == 0, summary.errors
    listed = await event_service.list_events(status=EventStatus.NEW)
    assert listed.total == 1
    event = listed.items[0]
    assert event.disposition_policy is DispositionPolicy.REQUIRED
    return event.event_id


async def _ingest_single_login_failure_file(
    tmp_path: Path,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> str:
    login_record = {
        "record_id": "id-e2e-login-fail-001",
        "channel": "identity",
        "logged_at": "2024-06-15T09:00:00+00:00",
        "is_key_event": True,
        "event_type": "login",
        "result": "failure",
        "account": "svc-backup-01",
        "src_ip": "10.20.30.50",
    }
    (tmp_path / "identity_logs.json").write_text(
        json.dumps([login_record]),
        encoding="utf-8",
    )
    source_ingester = SourceIngester(
        event_service,
        session_factory,
        source_mode="file",
    )
    file_ingester = FileIngester(
        source_ingester,
        event_service,
        source_mode="file",
    )
    summary = await file_ingester.ingest(tmp_path)
    assert summary.rejected == 0
    assert summary.accepted >= 1

    listed = await event_service.list_events(status=EventStatus.NEW)
    assert listed.total >= 1
    event = next(item for item in listed.items if item.source_type == "file")
    assert event.disposition_policy is DispositionPolicy.NOT_REQUIRED
    return event.event_id


async def _audit_status_sequence(
    session_factory: async_sessionmaker[AsyncSession],
    event_id: str,
) -> list[str]:
    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(orm.EventAuditLog)
                .where(
                    orm.EventAuditLog.event_id == event_id,
                    orm.EventAuditLog.to_status.is_not(None),
                )
                .order_by(orm.EventAuditLog.id)
            )
        ).all()
    return [row.to_status for row in rows if row.to_status is not None]


def _assert_ordered_status_subsequence(
    observed: list[str],
    expected: tuple[EventStatus, ...],
) -> None:
    """Assert *expected* appears in order within *observed* (extras allowed)."""
    index = 0
    for status in expected:
        target = status.value
        while index < len(observed) and observed[index] != target:
            index += 1
        assert index < len(observed), f"missing ordered status transition: {target}"
        index += 1


async def _assert_observability(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    context_store: EventContextStore,
    agent_trace_service: AgentTraceService,
    event_id: str,
    expect_guard_clean: bool,
    expected_agents: frozenset[str] | None = None,
) -> None:
    traces = await agent_trace_service.get_traces_by_event(event_id)
    assert traces, "agent_trace must record pipeline agents"
    traced_agents = {row.agent_name for row in traces}
    if expected_agents is not None:
        missing = expected_agents - traced_agents
        assert not missing, f"missing agent_trace rows for: {sorted(missing)}"

    async with session_factory() as session:
        audit_rows = (
            await session.scalars(
                select(orm.EventAuditLog).where(orm.EventAuditLog.event_id == event_id)
            )
        ).all()
    assert audit_rows, "event_audit_log must record lifecycle transitions"
    status_transitions = [row.to_status for row in audit_rows if row.to_status is not None]
    assert status_transitions, "event_audit_log must include status transitions"

    budget_usage = await context_store.get(event_id, "budget_usage")
    assert budget_usage, "budget_usage must be persisted after analysis"

    guard_violations = await context_store.get(event_id, "guard_violations")
    violations = guard_violations if isinstance(guard_violations, list) else []
    if expect_guard_clean:
        block_violations = [
            item
            for item in violations
            if isinstance(item, dict) and item.get("severity") == "block"
        ]
        assert block_violations == []


async def _assert_report_persisted(
    session_factory: async_sessionmaker[AsyncSession],
    context_store: EventContextStore,
    event_id: str,
    *,
    section_count: int = 15,
) -> None:
    async with session_factory() as session:
        row = await session.scalar(select(orm.Report).where(orm.Report.event_id == event_id))
    assert row is not None, "report table row must exist"
    assert len(row.sections) == section_count
    assert [section["key"] for section in row.sections] == list(SECTION_KEYS)
    assert row.title.strip(), "report title must contain key case information"

    ctx_report = await context_store.get(event_id, "report")
    assert ctx_report is not None, "EventContext.report must exist"
    assert len(ctx_report.get("sections") or []) == section_count
    assert str(ctx_report.get("title") or "").strip()


@pytest.mark.usefixtures("clean_state")
@pytest.mark.asyncio
async def test_golden_path_analysis_only_state_sequence(
    source_adapter: MockXDRSourceAdapter,
    source_ingester: SourceIngester,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
    context_store: EventContextStore,
    agent_trace_service: AgentTraceService,
    build_analysis_pipeline,
) -> None:
    """Scenario 1: NEW→…→REPORTING without CLOSED; confirmed high-risk threat."""
    event_id = await _ingest_main_scenario(source_adapter, source_ingester, event_service)
    event_before = await event_service.get_event(event_id)
    assert event_before is not None
    assert event_before.status is EventStatus.NEW

    pipeline, projection = build_analysis_pipeline()
    started = time.perf_counter()
    with bind_evidence_projection(projection):
        result = await pipeline.run(event_id)
    elapsed = time.perf_counter() - started
    assert elapsed < GOLDEN_PATH_MAX_SECONDS, f"golden path exceeded {GOLDEN_PATH_MAX_SECONDS}s"

    assert result.analysis_only_complete is True
    assert result.disposition_policy == "required"
    assert result.short_circuit is False
    assert result.risk_assessment is not None
    assert result.risk_assessment.risk_score >= 70
    assert result.final_verdict is FinalVerdict.CONFIRMED_THREAT
    assert result.report is not None

    event = await event_service.get_event(event_id)
    assert event is not None
    assert event.status is EventStatus.REPORTING
    assert event.final_verdict is FinalVerdict.CONFIRMED_THREAT
    assert event.risk_score >= 70

    observed = await _audit_status_sequence(session_factory, event_id)
    _assert_ordered_status_subsequence(observed, GOLDEN_STATUS_SEQUENCE)
    assert EventStatus.CLOSED.value not in observed

    triage_ctx = await context_store.get(event_id, "triage_result")
    evidence_ctx = await context_store.get(event_id, "evidence_output")
    risk_ctx = await context_store.get(event_id, "risk_assessment")
    assert triage_ctx and evidence_ctx and risk_ctx

    analysis_only_complete = await context_store.get(event_id, "analysis_only_complete")
    assert analysis_only_complete is True

    await _assert_report_persisted(session_factory, context_store, event_id)
    await _assert_observability(
        session_factory=session_factory,
        context_store=context_store,
        agent_trace_service=agent_trace_service,
        event_id=event_id,
        expect_guard_clean=True,
        expected_agents=PIPELINE_AGENT_NAMES,
    )


@pytest.mark.usefixtures("clean_state")
@pytest.mark.asyncio
async def test_low_risk_not_required_short_circuit_closed(
    tmp_path: Path,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
    context_store: EventContextStore,
    agent_trace_service: AgentTraceService,
    run_analysis_pipeline,
) -> None:
    """Scenario 2: file fallback single login failure → CLOSED without evidence."""
    event_id = await _ingest_single_login_failure_file(tmp_path, event_service, session_factory)
    result = await run_analysis_pipeline(event_id, scenario_id=None)

    assert result.short_circuit is True
    assert result.disposition_policy == "not_required"
    assert result.analysis_only_complete is True
    assert result.triage_result.severity is Severity.LOW
    assert result.triage_result.need_investigation is False
    assert result.triage_result.event_type is EventType.ACCOUNT_ANOMALY
    assert result.final_verdict is FinalVerdict.FALSE_POSITIVE
    assert result.evidence_output is not None
    assert result.evidence_output.collection_status is CollectionStatus.COMPLETED
    assert result.evidence_output.evidence_list == []

    event = await event_service.get_event(event_id)
    assert event is not None
    assert event.status is EventStatus.CLOSED

    observed = await _audit_status_sequence(session_factory, event_id)
    for status in SHORT_CIRCUIT_STATUS_SEQUENCE:
        assert status.value in observed
    assert EventStatus.COLLECTING_EVIDENCE.value not in observed

    await _assert_report_persisted(session_factory, context_store, event_id)
    await _assert_observability(
        session_factory=session_factory,
        context_store=context_store,
        agent_trace_service=agent_trace_service,
        event_id=event_id,
        expect_guard_clean=True,
        expected_agents=SHORT_CIRCUIT_AGENT_NAMES,
    )


@pytest.mark.usefixtures("clean_state")
@pytest.mark.asyncio
async def test_data_source_degradation_still_reports(
    source_adapter: MockXDRSourceAdapter,
    source_ingester: SourceIngester,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
    context_store: EventContextStore,
    agent_trace_service: AgentTraceService,
    run_analysis_pipeline,
) -> None:
    """Scenario 3: 3 query-tool failures → partial_done, report, stay REPORTING."""
    event_id = await _ingest_main_scenario(source_adapter, source_ingester, event_service)
    result = await run_analysis_pipeline(
        event_id,
        fail_tools=set(DEFAULT_PARTIAL_FAIL_TOOLS),
    )

    assert result.evidence_output is not None
    assert result.evidence_output.collection_status is CollectionStatus.PARTIAL_DONE
    assert 3 <= len(result.evidence_output.success_sources) <= 4
    assert result.report is not None
    assert result.analysis_only_complete is True

    event = await event_service.get_event(event_id)
    assert event is not None
    assert event.status is EventStatus.REPORTING

    await _assert_report_persisted(session_factory, context_store, event_id)
    await _assert_observability(
        session_factory=session_factory,
        context_store=context_store,
        agent_trace_service=agent_trace_service,
        event_id=event_id,
        expect_guard_clean=True,
        expected_agents=PIPELINE_AGENT_NAMES,
    )


@pytest.mark.usefixtures("clean_state")
@pytest.mark.asyncio
async def test_llm_degradation_fallback_without_bypassing_gates(
    source_adapter: MockXDRSourceAdapter,
    source_ingester: SourceIngester,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
    context_store: EventContextStore,
    agent_trace_service: AgentTraceService,
    build_analysis_pipeline,
) -> None:
    """Scenario 4: all LLM fail → regex/rules/template; required stays REPORTING."""
    event_id = await _ingest_main_scenario(source_adapter, source_ingester, event_service)
    pipeline, projection = build_analysis_pipeline(llm_client=FailingLLMClient())
    with bind_evidence_projection(projection):
        result = await pipeline.run(event_id)

    assert result.triage_result.degraded is True
    assert result.risk_assessment is not None
    assert result.risk_assessment.scoring_mode is ScoringMode.RULE_ONLY
    assert result.report is not None
    assert result.report.generated_by == GENERATED_BY_TEMPLATE
    assert result.analysis_only_complete is True
    assert result.disposition_policy == "required"

    event = await event_service.get_event(event_id)
    assert event is not None
    assert event.status is EventStatus.REPORTING
    assert event.disposition_policy is DispositionPolicy.REQUIRED
    assert EventStatus.CLOSED.value not in await _audit_status_sequence(session_factory, event_id)

    await _assert_report_persisted(session_factory, context_store, event_id)
    await _assert_observability(
        session_factory=session_factory,
        context_store=context_store,
        agent_trace_service=agent_trace_service,
        event_id=event_id,
        expect_guard_clean=True,
        expected_agents=PIPELINE_AGENT_NAMES,
    )


@pytest.mark.asyncio
async def test_basic_loop_survives_rag_failure() -> None:
    """ISSUE-047 acceptance #2 gate: RAG failure must not break scoring + report."""
    from uuid import uuid4

    from app.models.agent_io import TriageResult

    event_id = f"evt-e2e-rag-fail-{uuid4().hex[:8]}"
    triage = TriageResult(
        event_type=EventType.DATA_EXFILTRATION,
        severity=Severity.HIGH,
        need_investigation=True,
        reasoning="basic loop",
    )
    evidence = main_evidence(event_id)
    event_service = FakeEventService()
    risk_agent = RiskAgent(
        llm_client=FailingLLMClient(),
        working_memory=FakeWorkingMemory(),
        event_service=event_service,
    )
    report = InvestigationReport(
        report_id=report_id_for_event(event_id),
        event_id=event_id,
        title="basic loop report",
    )
    pipeline = AnalysisOnlyPipeline(
        triage_agent=_StubAgent(triage),
        evidence_agent=_StubAgent(evidence),
        rag_agent=_FailingRAGAgent(),
        risk_agent=risk_agent,
        report_agent=_StubAgent(report),
        event_service=event_service,
    )
    result = await pipeline.run(event_id)

    assert result.rag_degraded is True
    assert result.risk_assessment.risk_score >= 70
    assert result.final_verdict is FinalVerdict.CONFIRMED_THREAT
    assert event_service.final_verdict_by_event[event_id] is FinalVerdict.CONFIRMED_THREAT
    assert result.report is not None
    assert EventStatus.TRIAGING in event_service.transitions
    assert EventStatus.REPORTING in event_service.transitions
    assert EventStatus.CLOSED not in event_service.transitions
