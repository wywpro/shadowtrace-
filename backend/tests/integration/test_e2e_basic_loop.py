"""Minimal analysis-only loop coverage for ISSUE-039 / ISSUE-047 gate.

Full four-scenario ISSUE-039 coverage lands with ISSUE-038 API wiring. This module
currently implements the ISSUE-047 RAG-failure survival gate; remaining ISSUE-039
scenarios are tracked as explicit skips until ISSUE-038 delivers the HTTP pipeline.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.agents.risk_agent import RiskAgent
from app.models.agent_io import TriageResult
from app.models.enums import EventStatus, EventType, FinalVerdict, Severity
from app.models.ids import report_id_for_event
from app.models.report import InvestigationReport
from app.services.analysis_only_pipeline import AnalysisOnlyPipeline
from tests.integration.rag_scenario_fixtures import (
    FakeEventService,
    FakeWorkingMemory,
    main_evidence,
)

pytestmark = pytest.mark.e2e_basic


class _StubAgent:
    def __init__(self, result: Any) -> None:
        self.result = result

    async def execute(self, input: Any) -> Any:
        return self.result


class _FailingRAGAgent:
    async def execute(self, input: Any) -> Any:
        raise RuntimeError("rag unavailable")


class _FailingLLM:
    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("llm unavailable")


@pytest.mark.skip(reason="ISSUE-038 AnalysisOnlyPipeline + events API golden path")
@pytest.mark.asyncio
async def test_golden_path_analysis_only_state_sequence() -> None:
    """ISSUE-039 scenario 1: NEW→…→REPORTING without CLOSED."""


@pytest.mark.skip(
    reason="ISSUE-038 low-risk not_required short-circuit + local CLOSED + 15-section report"
)
@pytest.mark.asyncio
async def test_low_risk_not_required_short_circuit_closed() -> None:
    """ISSUE-039 scenario 2: disposition_policy=not_required file fixture → CLOSED."""


@pytest.mark.skip(reason="ISSUE-038 evidence tool failure partial_done path")
@pytest.mark.asyncio
async def test_data_source_degradation_still_reports() -> None:
    """ISSUE-039 scenario 3: partial collection still reaches REPORTING/report."""


@pytest.mark.skip(reason="ISSUE-038 LLM-all-fail regex/triage/report fallback path")
@pytest.mark.asyncio
async def test_llm_degradation_fallback_without_bypassing_gates() -> None:
    """ISSUE-039 scenario 4: degraded triage/scoring/report without disposition bypass."""


@pytest.mark.asyncio
async def test_basic_loop_survives_rag_failure() -> None:
    """ISSUE-047 acceptance #2 gate: RAG failure must not break scoring + report."""
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
        llm_client=_FailingLLM(),
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
