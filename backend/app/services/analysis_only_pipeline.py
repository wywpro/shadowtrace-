"""Sequential analysis-only pipeline (ISSUE-038 / ISSUE-047).

Runs Triage → Evidence → RAG → Risk → Report for mock/offline development.
RAGAgent sits after Evidence and before Risk; failures degrade to ``rag_output=None``
without blocking downstream scoring or reporting.

038 lifecycle features (NEW guard, short-circuit close, disposition policy,
``analysis_only_complete`` persistence) are preserved. 047 RAG wiring is reused by
``rag_node`` in ``app.orchestration.workflow_graph``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.agents.evidence_agent import EvidenceAgent
from app.agents.report_agent import ReportAgent
from app.agents.risk_agent import RiskAgent
from app.agents.triage_agent import TriageAgent
from app.core.config import Settings, get_settings
from app.core.errors import (
    ConfigurationError,
    InvalidStateTransitionError,
    ShadowTraceError,
)
from app.models.agent_io import (
    CollectionStatus,
    EvidenceAgentInput,
    EvidenceOutput,
    RAGAgentInput,
    RAGOutput,
    ReportAgentInput,
    RiskAgentInput,
    RiskAssessment,
    ScoringMode,
    TriageAgentInput,
    TriageResult,
)
from app.models.entities import EntitySet
from app.models.enums import DispositionPolicy, EventStatus, FinalVerdict
from app.models.report import InvestigationReport
from app.models.workflow import TransitionContext
from app.services.event_service import EventService, StateMachinePort

logger = logging.getLogger(__name__)

_PIPELINE_OPERATOR = "AnalysisOnlyPipeline"


async def _read_persisted_final_verdict(
    event_service: Any | None,
    event_id: str,
) -> FinalVerdict:
    """Read verdict persisted by RiskAgent via ``EventService.set_final_verdict``."""
    if event_service is None:
        return FinalVerdict.NONE
    get_event = getattr(event_service, "get_event", None)
    if get_event is None:
        return FinalVerdict.NONE
    try:
        event = await get_event(event_id)
    except Exception:
        logger.debug(
            "failed to read persisted final_verdict for event=%s",
            event_id,
            exc_info=True,
        )
        return FinalVerdict.NONE
    if event is None:
        return FinalVerdict.NONE
    verdict = getattr(event, "final_verdict", None)
    if isinstance(verdict, FinalVerdict):
        return verdict
    if isinstance(verdict, str):
        try:
            return FinalVerdict(verdict)
        except ValueError:
            return FinalVerdict.NONE
    return FinalVerdict.NONE


class _AgentProtocol(Protocol):
    async def execute(self, input: Any) -> Any: ...


@dataclass(frozen=True)
class AnalysisOnlyPipelineResult:
    """Outcome of a single analysis-only run."""

    event_id: str
    triage_result: TriageResult
    evidence_output: EvidenceOutput | None = None
    rag_output: RAGOutput | None = None
    rag_degraded: bool = False
    risk_assessment: RiskAssessment | None = None
    report: InvestigationReport | None = None
    final_verdict: FinalVerdict = FinalVerdict.NONE
    analysis_only_complete: bool = False
    status: EventStatus | None = None
    disposition_policy: str | None = None
    short_circuit: bool = False


def assert_analysis_only_mode(settings: Settings | None = None) -> None:
    """Fail closed unless mock/offline side effects are disabled."""
    cfg = settings or get_settings()
    if cfg.allow_live_side_effects or cfg.allow_xdr_writeback:
        raise ConfigurationError(
            "AnalysisOnlyPipeline requires ALLOW_LIVE_SIDE_EFFECTS=false "
            "and ALLOW_XDR_WRITEBACK=false",
            error_code="configuration_error",
            details={
                "allow_live_side_effects": cfg.allow_live_side_effects,
                "allow_xdr_writeback": cfg.allow_xdr_writeback,
            },
        )
    source = (cfg.source_mode or "").strip().lower()
    disposition = (cfg.disposition_mode or "").strip().lower()
    if "mock" not in source or "mock" not in disposition:
        raise ConfigurationError(
            "AnalysisOnlyPipeline requires SOURCE_MODE and DISPOSITION_MODE mock modes",
            error_code="configuration_error",
            details={"source_mode": cfg.source_mode, "disposition_mode": cfg.disposition_mode},
        )


async def run_rag_stage(
    rag_agent: _AgentProtocol,
    *,
    event_id: str,
    triage_result: TriageResult,
    evidence_output: EvidenceOutput,
) -> tuple[RAGOutput | None, bool]:
    """Invoke RAGAgent between evidence and risk; never raise to callers."""
    try:
        output = await rag_agent.execute(
            RAGAgentInput(
                event_id=event_id,
                triage_result=triage_result,
                evidence_output=evidence_output,
            )
        )
        if not isinstance(output, RAGOutput):
            logger.warning(
                "RAGAgent returned unexpected type %s for event=%s; degrading",
                type(output).__name__,
                event_id,
            )
            return None, True
        return output, bool(output.degraded)
    except Exception:
        logger.warning(
            "RAGAgent failed for event=%s; continuing without RAG enhancement",
            event_id,
            exc_info=True,
        )
        return None, True


class AnalysisOnlyPipeline:
    """Temporary sequential analysis pipeline (pre-SuperAgent, ISSUE-054).

    Only runs in dev/offline mode. High-risk required-disposition events stay at
    REPORTING; only ``disposition_policy=not_required`` events may reach CLOSED.
    """

    def __init__(
        self,
        *,
        triage_agent: TriageAgent | _AgentProtocol,
        evidence_agent: EvidenceAgent | _AgentProtocol,
        rag_agent: _AgentProtocol,
        risk_agent: RiskAgent | _AgentProtocol,
        report_agent: ReportAgent | _AgentProtocol,
        event_service: EventService | Any | None = None,
        state_machine: StateMachinePort | None = None,
        context_store: Any | None = None,
        degraded_flags: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._triage = triage_agent
        self._evidence = evidence_agent
        self._rag = rag_agent
        self._risk = risk_agent
        self._report = report_agent
        self._event_service = event_service
        self._state_machine = state_machine
        self._context_store = context_store
        self._degraded_flags = degraded_flags
        self._settings = settings

        # Back-compat aliases for ISSUE-047 unit tests.
        self.triage_agent = triage_agent
        self.evidence_agent = evidence_agent
        self.rag_agent = rag_agent
        self.risk_agent = risk_agent
        self.report_agent = report_agent
        self.event_service = event_service
        self.settings = settings

    async def run(
        self,
        event_id: str,
        *,
        raw_event_summary: str = "",
        hint_entities: Any | None = None,
    ) -> AnalysisOnlyPipelineResult:
        """Execute the analysis-only pipeline for *event_id*."""
        assert_analysis_only_mode(self._settings)

        event = None
        if self._event_service is not None and self._state_machine is not None:
            event = await self._event_service.get_event(event_id)
            if event is None:
                raise ShadowTraceError(
                    f"event {event_id} not found",
                    error_code="event_not_found",
                )
            if event.status is not EventStatus.NEW:
                raise InvalidStateTransitionError(
                    f"AnalysisOnlyPipeline requires event in NEW status, got {event.status.value}",
                    current=event.status,
                    target=EventStatus.TRIAGING,
                    details={"event_id": event_id},
                )
        elif self._event_service is not None and self._state_machine is None:
            # ISSUE-047 unit tests: event_service tracks verdicts only.
            pass

        await self._transition(
            event_id,
            EventStatus.TRIAGING,
            reason="analysis_pipeline:triage_start",
        )

        if event is not None and self._state_machine is not None and hasattr(event, "title"):
            triage_result = await self._run_triage(event_id, event)
        else:
            triage_input = TriageAgentInput(
                event_id=event_id,
                raw_event_summary=raw_event_summary,
                hint_entities=hint_entities if hint_entities is not None else EntitySet(),
            )
            triage_result = await self._triage.execute(triage_input)
            if not isinstance(triage_result, TriageResult):
                raise TypeError("TriageAgent must return TriageResult")

        logger.info(
            "AnalysisOnlyPipeline triage complete event=%s type=%s severity=%s need_inv=%s",
            event_id,
            triage_result.event_type.value,
            triage_result.severity.value,
            triage_result.need_investigation,
        )

        disposition_policy = DispositionPolicy.NOT_REQUIRED
        if event is not None and hasattr(event, "disposition_policy"):
            disposition_policy = event.disposition_policy
        if (
            not triage_result.need_investigation
            and disposition_policy == DispositionPolicy.NOT_REQUIRED
            and event is not None
            and self._state_machine is not None
            and hasattr(event, "title")
        ):
            return await self._short_circuit_close(event_id, event, triage_result)

        await self._transition(
            event_id,
            EventStatus.COLLECTING_EVIDENCE,
            context=TransitionContext(need_investigation=True),
            reason="analysis_pipeline:evidence_collect",
        )
        evidence_output = await self._run_evidence(event_id, triage_result)

        await self._transition(
            event_id,
            EventStatus.ANALYZING,
            reason="analysis_pipeline:evidence_analyze",
        )
        rag_output, rag_degraded = await run_rag_stage(
            self._rag,
            event_id=event_id,
            triage_result=triage_result,
            evidence_output=evidence_output,
        )

        await self._transition(
            event_id,
            EventStatus.SCORING,
            reason="analysis_pipeline:risk_score",
        )
        risk_assessment = await self._run_risk(
            event_id,
            triage_result,
            evidence_output,
            rag_output,
        )
        final_verdict = await _read_persisted_final_verdict(self._event_service, event_id)

        await self._transition(
            event_id,
            EventStatus.REPORTING,
            reason="analysis_pipeline:report_generate",
        )
        report = await self._run_report(event_id, evidence_output, risk_assessment)

        if self._state_machine is not None and self._event_service is not None:
            event = await self._event_service.get_event(event_id)
            if event is None:
                raise ShadowTraceError(
                    f"event {event_id} disappeared during pipeline execution",
                    error_code="event_not_found",
                )

            if event.disposition_policy == DispositionPolicy.REQUIRED:
                logger.info(
                    "AnalysisOnlyPipeline: event=%s requires disposition, staying at REPORTING",
                    event_id,
                )
                await self._persist_analysis_only_complete(event_id)
                return AnalysisOnlyPipelineResult(
                    event_id=event_id,
                    triage_result=triage_result,
                    evidence_output=evidence_output,
                    rag_output=rag_output,
                    rag_degraded=rag_degraded,
                    risk_assessment=risk_assessment,
                    report=report,
                    final_verdict=final_verdict,
                    analysis_only_complete=True,
                    status=EventStatus.REPORTING,
                    disposition_policy="required",
                )

            await self._transition(
                event_id,
                EventStatus.CLOSED,
                context=TransitionContext(
                    need_investigation=triage_result.need_investigation,
                ),
                reason="analysis_pipeline:complete_not_required",
            )
            await self._persist_analysis_only_complete(event_id)
            return AnalysisOnlyPipelineResult(
                event_id=event_id,
                triage_result=triage_result,
                evidence_output=evidence_output,
                rag_output=rag_output,
                rag_degraded=rag_degraded,
                risk_assessment=risk_assessment,
                report=report,
                final_verdict=final_verdict,
                analysis_only_complete=True,
                status=EventStatus.CLOSED,
                disposition_policy="not_required",
            )

        await self._persist_analysis_only_complete(event_id)
        return AnalysisOnlyPipelineResult(
            event_id=event_id,
            triage_result=triage_result,
            evidence_output=evidence_output,
            rag_output=rag_output,
            rag_degraded=rag_degraded,
            risk_assessment=risk_assessment,
            report=report,
            final_verdict=final_verdict,
            analysis_only_complete=True,
        )

    async def _transition(
        self,
        event_id: str,
        target: EventStatus,
        *,
        context: TransitionContext | None = None,
        reason: str | None = None,
    ) -> None:
        if self._state_machine is not None:
            await self._state_machine.transition(
                event_id,
                target,
                context=context,
                operator=_PIPELINE_OPERATOR,
                reason=reason or f"analysis_only:{target.value}",
            )
            return
        if self._event_service is None:
            return
        transition = getattr(self._event_service, "transition_status", None)
        if transition is None:
            return
        await transition(
            event_id,
            target,
            context=context,
            operator=_PIPELINE_OPERATOR,
            reason=reason or f"analysis_only:{target.value}",
        )

    async def _run_triage(self, event_id: str, event: Any) -> TriageResult:
        raw_summary = f"{event.title}. {event.description}"
        triage_input = TriageAgentInput(
            event_id=event_id,
            raw_event_summary=raw_summary,
            hint_entities=event.entities,
        )
        return await self._triage.execute(triage_input)

    async def _run_evidence(self, event_id: str, triage_result: TriageResult) -> EvidenceOutput:
        evidence_input = EvidenceAgentInput(
            event_id=event_id,
            triage_result=triage_result,
        )
        output = await self._evidence.execute(evidence_input)
        if not isinstance(output, EvidenceOutput):
            raise TypeError("EvidenceAgent must return EvidenceOutput")
        return output

    async def _run_risk(
        self,
        event_id: str,
        triage_result: TriageResult,
        evidence_output: EvidenceOutput,
        rag_output: RAGOutput | None,
    ) -> RiskAssessment:
        risk_input = RiskAgentInput(
            event_id=event_id,
            triage_result=triage_result,
            evidence_output=evidence_output,
            rag_output=rag_output,
        )
        output = await self._risk.execute(risk_input)
        if not isinstance(output, RiskAssessment):
            raise TypeError("RiskAgent must return RiskAssessment")
        return output

    async def _run_report(
        self,
        event_id: str,
        evidence_output: EvidenceOutput,
        risk_assessment: RiskAssessment,
    ) -> InvestigationReport | None:
        report_input = ReportAgentInput(
            event_id=event_id,
            evidence_output=evidence_output,
            risk_assessment=risk_assessment,
        )
        report = await self._report.execute(report_input)
        if report is not None and not isinstance(report, InvestigationReport):
            raise TypeError("ReportAgent must return InvestigationReport or None")
        return report

    async def _persist_analysis_only_complete(self, event_id: str) -> None:
        if self._context_store is not None:
            try:
                await self._context_store.set(event_id, "analysis_only_complete", True)
            except Exception:
                logger.warning(
                    "Failed to persist analysis_only_complete for event=%s",
                    event_id,
                    exc_info=True,
                )
                if self._degraded_flags is not None:
                    try:
                        await self._degraded_flags.set_flag(
                            event_id,
                            "redis_context_unavailable",
                            True,
                            writer=_PIPELINE_OPERATOR,
                        )
                    except Exception:
                        logger.error(
                            "Failed to set redis_context_unavailable degraded flag for event=%s",
                            event_id,
                            exc_info=True,
                        )

    async def _short_circuit_close(
        self,
        event_id: str,
        event: Any,
        triage_result: TriageResult,
    ) -> AnalysisOnlyPipelineResult:
        logger.info(
            "AnalysisOnlyPipeline: short-circuit close event=%s severity=%s",
            event_id,
            triage_result.severity.value,
        )

        placeholder_evidence = EvidenceOutput(
            evidence_list=[],
            conflicts=[],
            gaps=[],
            success_sources=[],
            failed_sources=[],
            overall_confidence=0.0,
            collection_status=CollectionStatus.COMPLETED,
        )
        placeholder_risk = RiskAssessment(
            risk_score=0,
            severity=triage_result.severity,
            confidence=0.9,
            risk_factors=[],
            possible_false_positive=True,
            scoring_mode=ScoringMode.RULE_ONLY,
        )

        report = await self._run_report(event_id, placeholder_evidence, placeholder_risk)

        ctx = TransitionContext(
            need_investigation=False,
            recommendation="close_as_fp",
        )
        await self._transition(
            event_id,
            EventStatus.CLOSED,
            context=ctx,
            reason="analysis_pipeline:short_circuit_closed",
        )

        await self._persist_analysis_only_complete(event_id)
        return AnalysisOnlyPipelineResult(
            event_id=event_id,
            triage_result=triage_result,
            evidence_output=placeholder_evidence,
            rag_output=None,
            rag_degraded=False,
            risk_assessment=placeholder_risk,
            report=report,
            final_verdict=FinalVerdict.FALSE_POSITIVE,
            analysis_only_complete=True,
            status=EventStatus.CLOSED,
            disposition_policy="not_required",
            short_circuit=True,
        )


__all__ = [
    "AnalysisOnlyPipeline",
    "AnalysisOnlyPipelineResult",
    "assert_analysis_only_mode",
    "run_rag_stage",
]
