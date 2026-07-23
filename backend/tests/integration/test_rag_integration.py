"""RAG + risk scoring + verdict integration tests (ISSUE-047)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.agents.rag_agent import RAGAgent
from app.agents.risk_agent import RiskAgent
from app.agents.risk_scoring_engine import FACTOR_WEIGHTS, RiskScoringEngine
from app.agents.verdict_resolver import VerdictResolver
from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.models.agent_io import (
    AttackTechniqueMatch,
    CollectionStatus,
    EvidenceOutput,
    FpSimilarity,
    RAGOutput,
    RiskAgentInput,
    RiskAssessment,
    ScoringMode,
    TriageResult,
)
from app.models.context import EventContext
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    EvidenceSource,
    FinalVerdict,
    Severity,
    WritebackReadiness,
)
from app.models.ids import report_id_for_event
from app.models.report import InvestigationReport
from app.models.security_event import EventSummary
from app.models.workflow import FP_HIGH_THRESHOLD
from app.orchestration.workflow_graph import rag_node
from app.services.analysis_only_pipeline import AnalysisOnlyPipeline, assert_analysis_only_mode
from tests.integration.rag_scenario_fixtures import (
    FakeEventService,
    FakeWorkingMemory,
    main_evidence,
    main_triage,
    make_evidence_item,
)

pytestmark = pytest.mark.rag


class _FailingLLM:
    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("llm unavailable")


def _main_rag_output() -> RAGOutput:
    return RAGOutput(
        attack_techniques=[
            AttackTechniqueMatch(
                technique_id="T1567.002",
                technique_name="Exfiltration Over Web Service",
                tactics=["exfiltration"],
                match_confidence=0.92,
                citation_id="cit-rag-1",
            ),
            AttackTechniqueMatch(
                technique_id="T1041",
                technique_name="Exfiltration Over C2 Channel",
                tactics=["exfiltration"],
                match_confidence=0.88,
                citation_id="cit-rag-2",
            ),
            AttackTechniqueMatch(
                technique_id="T1486",
                technique_name="Data Encrypted for Impact",
                tactics=["impact"],
                match_confidence=0.8,
                citation_id="cit-rag-3",
            ),
        ],
    )


def _merged_rule_score(
    engine: RiskScoringEngine,
    *,
    triage: TriageResult,
    evidence: EvidenceOutput,
    rag: RAGOutput | None,
) -> float:
    scores = engine.score(
        triage_result=triage,
        evidence_output=evidence,
        rag_output=rag,
    )
    return sum(scores[name][0] * FACTOR_WEIGHTS[name] for name in FACTOR_WEIGHTS)


class _StubAgent:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    async def execute(self, input: Any) -> Any:
        self.calls.append(input)
        return self.result


class _FailingStubAgent:
    async def execute(self, input: Any) -> Any:
        raise RuntimeError("rag subsystem unavailable")


class _RecordingRiskAgent(RiskAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rag_inputs: list[RAGOutput | None] = []

    async def _run(self, input: RiskAgentInput) -> RiskAssessment:
        self.rag_inputs.append(input.rag_output)
        return await super()._run(input)


def _event_summary(event_id: str) -> EventSummary:
    return EventSummary(
        event_id=event_id,
        event_type=EventType.DATA_EXFILTRATION,
        title="rag node test",
        status=EventStatus.ANALYZING,
        severity=Severity.HIGH,
        risk_score=50,
        final_verdict=FinalVerdict.NONE,
        writeback_required=False,
        writeback_readiness=WritebackReadiness.NOT_REQUIRED,
        disposition_policy=DispositionPolicy.NOT_REQUIRED,
    )


def test_rag_boosts_rule_baseline_for_main_scenario() -> None:
    engine = RiskScoringEngine()
    event_id = f"evt-rag-baseline-{uuid4().hex[:8]}"
    triage = main_triage()
    evidence = main_evidence(event_id)
    baseline = _merged_rule_score(engine, triage=triage, evidence=evidence, rag=None)
    with_rag = _merged_rule_score(
        engine,
        triage=triage,
        evidence=evidence,
        rag=_main_rag_output(),
    )
    assert with_rag >= baseline
    assert with_rag >= 70.0


@pytest.mark.asyncio
async def test_main_scenario_with_rag_confirmed_threat() -> None:
    event_id = f"evt-rag-main-{uuid4().hex[:8]}"
    wm = FakeWorkingMemory()
    event_service = FakeEventService()
    agent = RiskAgent(
        llm_client=_FailingLLM(),
        working_memory=wm,
        event_service=event_service,
    )
    baseline_agent = RiskAgent(
        llm_client=_FailingLLM(),
        working_memory=FakeWorkingMemory(),
        event_service=FakeEventService(),
    )
    evidence = main_evidence(event_id)
    triage = main_triage()
    baseline = await baseline_agent.execute(
        RiskAgentInput(
            event_id=event_id,
            triage_result=triage,
            evidence_output=evidence,
        )
    )
    output = await agent.execute(
        RiskAgentInput(
            event_id=event_id,
            triage_result=triage,
            evidence_output=evidence,
            rag_output=_main_rag_output(),
        )
    )
    assert output.risk_score >= baseline.risk_score
    assert output.risk_score >= 70
    assert event_service.verdicts[-1] is FinalVerdict.CONFIRMED_THREAT


@pytest.mark.asyncio
async def test_fp_scenario_false_positive_via_rag_similarity() -> None:
    event_id = f"evt-rag-fp-{uuid4().hex[:8]}"
    wm = FakeWorkingMemory()
    event_service = FakeEventService()
    agent = RiskAgent(
        llm_client=_FailingLLM(),
        working_memory=wm,
        event_service=event_service,
    )
    weak = EvidenceOutput(
        evidence_list=[
            make_evidence_item(
                source=EvidenceSource.DNS,
                evidence_type="dns_query",
                confidence=0.35,
                event_id=event_id,
                description="benign lookup",
                raw={"query": "update.example.com"},
            )
        ],
        success_sources=["dns"],
        failed_sources=[],
        overall_confidence=0.35,
        collection_status=CollectionStatus.DEGRADED,
    )
    rag = RAGOutput(
        fp_similarity=FpSimilarity(max_score=FP_HIGH_THRESHOLD, matched_case_id="fp-case-ops")
    )
    output = await agent.execute(
        RiskAgentInput(
            event_id=event_id,
            triage_result=TriageResult(
                event_type=EventType.OTHER,
                severity=Severity.LOW,
                need_investigation=True,
            ),
            evidence_output=weak,
            rag_output=rag,
        )
    )
    assert output.risk_score < 40
    assert event_service.verdicts[-1] is FinalVerdict.FALSE_POSITIVE


def test_close_as_fp_beats_high_rag_fp_similarity() -> None:
    resolver = VerdictResolver()
    rag = RAGOutput(fp_similarity=FpSimilarity(max_score=0.99, matched_case_id="rag-fp"))
    verdict = resolver.resolve(
        RiskAssessment(
            risk_score=85,
            severity=Severity.HIGH,
            confidence=0.8,
            scoring_mode=ScoringMode.RULE_ONLY,
        ),
        false_positive_match={"recommendation": "close_as_fp", "max_score": 0.5},
        rag_output=rag,
    )
    assert verdict is FinalVerdict.FALSE_POSITIVE


@pytest.mark.asyncio
async def test_pipeline_wires_rag_between_evidence_and_risk() -> None:
    event_id = f"evt-rag-pipe-{uuid4().hex[:8]}"
    triage = main_triage()
    evidence = main_evidence(event_id)
    rag_output = _main_rag_output()
    event_service = FakeEventService()
    risk_agent = _RecordingRiskAgent(
        llm_client=_FailingLLM(),
        working_memory=FakeWorkingMemory(),
        event_service=event_service,
    )
    report = InvestigationReport(
        report_id=report_id_for_event(event_id),
        event_id=event_id,
        title="stub report",
    )
    pipeline = AnalysisOnlyPipeline(
        triage_agent=_StubAgent(triage),
        evidence_agent=_StubAgent(evidence),
        rag_agent=_StubAgent(rag_output),
        risk_agent=risk_agent,
        report_agent=_StubAgent(report),
        event_service=event_service,
    )
    result = await pipeline.run(event_id)
    assert result.rag_output == rag_output
    assert result.rag_degraded is False
    assert result.final_verdict is FinalVerdict.CONFIRMED_THREAT
    assert event_service.final_verdict_by_event[event_id] is FinalVerdict.CONFIRMED_THREAT
    assert risk_agent.rag_inputs == [rag_output]
    assert result.analysis_only_complete is True


@pytest.mark.asyncio
async def test_pipeline_rag_failure_degrades_without_blocking() -> None:
    event_id = f"evt-rag-fail-{uuid4().hex[:8]}"
    triage = main_triage()
    evidence = main_evidence(event_id)
    event_service = FakeEventService()
    risk_agent = _RecordingRiskAgent(
        llm_client=_FailingLLM(),
        working_memory=FakeWorkingMemory(),
        event_service=event_service,
    )
    report = InvestigationReport(
        report_id=report_id_for_event(event_id),
        event_id=event_id,
        title="stub report",
    )
    pipeline = AnalysisOnlyPipeline(
        triage_agent=_StubAgent(triage),
        evidence_agent=_StubAgent(evidence),
        rag_agent=_FailingStubAgent(),
        risk_agent=risk_agent,
        report_agent=_StubAgent(report),
        event_service=event_service,
    )
    result = await pipeline.run(event_id)
    assert result.rag_output is None
    assert result.rag_degraded is True
    assert result.risk_assessment.risk_score >= 70
    assert result.final_verdict is FinalVerdict.CONFIRMED_THREAT
    assert risk_agent.rag_inputs == [None]
    assert result.report is not None


@pytest.mark.asyncio
async def test_pipeline_real_rag_agent_writes_wm_and_passes_output() -> None:
    event_id = f"evt-rag-real-{uuid4().hex[:8]}"
    wm = FakeWorkingMemory()
    event_service = FakeEventService()
    risk_agent = _RecordingRiskAgent(
        llm_client=_FailingLLM(),
        working_memory=wm,
        event_service=event_service,
    )
    report = InvestigationReport(
        report_id=report_id_for_event(event_id),
        event_id=event_id,
        title="stub report",
    )
    pipeline = AnalysisOnlyPipeline(
        triage_agent=_StubAgent(main_triage()),
        evidence_agent=_StubAgent(main_evidence(event_id)),
        rag_agent=RAGAgent(working_memory=wm, pipeline=None),
        risk_agent=risk_agent,
        report_agent=_StubAgent(report),
        event_service=event_service,
    )
    result = await pipeline.run(event_id)

    stored = await wm.read(event_id, "rag_output")
    assert stored is not None
    assert stored["degraded"] is True
    assert result.rag_output is not None
    assert result.rag_degraded is True
    assert risk_agent.rag_inputs[0] is not None
    assert result.risk_assessment.risk_score >= 70
    assert result.final_verdict is FinalVerdict.CONFIRMED_THREAT


@pytest.mark.asyncio
async def test_pipeline_rag_fail_still_uses_false_positive_match() -> None:
    event_id = f"evt-rag-fp-pipe-{uuid4().hex[:8]}"
    wm = FakeWorkingMemory()
    wm.values[(event_id, "false_positive_match")] = {
        "recommendation": "close_as_fp",
        "max_score": 0.96,
    }
    event_service = FakeEventService()
    weak = EvidenceOutput(
        evidence_list=[
            make_evidence_item(
                source=EvidenceSource.DNS,
                evidence_type="dns_query",
                confidence=0.35,
                event_id=event_id,
                description="benign lookup",
                raw={"query": "update.example.com"},
            )
        ],
        success_sources=["dns"],
        failed_sources=[],
        overall_confidence=0.35,
        collection_status=CollectionStatus.DEGRADED,
    )
    risk_agent = RiskAgent(
        llm_client=_FailingLLM(),
        working_memory=wm,
        event_service=event_service,
    )
    pipeline = AnalysisOnlyPipeline(
        triage_agent=_StubAgent(
            TriageResult(
                event_type=EventType.OTHER,
                severity=Severity.LOW,
                need_investigation=True,
            )
        ),
        evidence_agent=_StubAgent(weak),
        rag_agent=_FailingStubAgent(),
        risk_agent=risk_agent,
        report_agent=_StubAgent(
            InvestigationReport(
                report_id=report_id_for_event(event_id),
                event_id=event_id,
                title="fp report",
            )
        ),
        event_service=event_service,
    )
    result = await pipeline.run(event_id)

    assert result.rag_output is None
    assert result.rag_degraded is True
    assert result.risk_assessment.risk_score < 40
    assert result.final_verdict is FinalVerdict.FALSE_POSITIVE
    assert event_service.final_verdict_by_event[event_id] is FinalVerdict.FALSE_POSITIVE


@pytest.mark.asyncio
async def test_rag_node_degrades_on_failure() -> None:
    event_id = f"evt-rag-node-{uuid4().hex[:8]}"
    context = EventContext(event=_event_summary(event_id))
    output = await rag_node(
        context,
        _FailingStubAgent(),  # type: ignore[arg-type]
        triage_result=main_triage(),
        evidence_output=main_evidence(event_id),
    )
    assert output is None


def test_analysis_only_pipeline_requires_mock_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_LIVE_SIDE_EFFECTS", "true")
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigurationError, match="ALLOW_LIVE_SIDE_EFFECTS"):
            assert_analysis_only_mode()
    finally:
        monkeypatch.delenv("ALLOW_LIVE_SIDE_EFFECTS", raising=False)
        get_settings.cache_clear()
