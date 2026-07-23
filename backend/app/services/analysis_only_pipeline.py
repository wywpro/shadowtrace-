"""Sequential analysis-only pipeline (ISSUE-038 / ISSUE-047).

Runs Triage → Evidence → RAG → Risk → Report for mock/offline development.
RAGAgent sits after Evidence and before Risk; failures degrade to ``rag_output=None``
without blocking downstream scoring or reporting.

Production orchestration is superseded by LangGraph (ISSUE-048+); ``rag_node`` in
``app.orchestration.workflow_graph`` reuses ``run_rag_stage`` for the same ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError
from app.models.agent_io import (
    EvidenceAgentInput,
    EvidenceOutput,
    RAGAgentInput,
    RAGOutput,
    ReportAgentInput,
    RiskAgentInput,
    RiskAssessment,
    TriageAgentInput,
    TriageResult,
)
from app.models.entities import EntitySet
from app.models.enums import EventStatus, FinalVerdict
from app.models.report import InvestigationReport
from app.models.workflow import TransitionContext

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
    evidence_output: EvidenceOutput
    rag_output: RAGOutput | None
    rag_degraded: bool
    risk_assessment: RiskAssessment
    report: InvestigationReport | None
    final_verdict: FinalVerdict
    analysis_only_complete: bool


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
    """Temporary sequential analysis driver (mock XDR / offline only)."""

    def __init__(
        self,
        *,
        triage_agent: _AgentProtocol,
        evidence_agent: _AgentProtocol,
        rag_agent: _AgentProtocol,
        risk_agent: _AgentProtocol,
        report_agent: _AgentProtocol,
        event_service: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
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
        hint_entities: EntitySet | None = None,
    ) -> AnalysisOnlyPipelineResult:
        assert_analysis_only_mode(self.settings)

        await self._transition(event_id, EventStatus.TRIAGING)

        triage_result = await self.triage_agent.execute(
            TriageAgentInput(
                event_id=event_id,
                raw_event_summary=raw_event_summary,
                hint_entities=hint_entities or EntitySet(),
            )
        )
        if not isinstance(triage_result, TriageResult):
            raise TypeError("TriageAgent must return TriageResult")

        if not triage_result.need_investigation:
            # ISSUE-038 owns need_investigation=false short-circuit: 15-section
            # low-risk fast-close report + optional TRIAGING→CLOSED when
            # disposition_policy=not_required.
            raise NotImplementedError(
                "need_investigation=false short-circuit is owned by ISSUE-038; "
                "use disposition_policy=not_required fixtures there"
            )

        await self._transition(
            event_id,
            EventStatus.COLLECTING_EVIDENCE,
            context=TransitionContext(need_investigation=True),
        )
        evidence_output = await self.evidence_agent.execute(
            EvidenceAgentInput(event_id=event_id, triage_result=triage_result)
        )
        if not isinstance(evidence_output, EvidenceOutput):
            raise TypeError("EvidenceAgent must return EvidenceOutput")

        await self._transition(event_id, EventStatus.ANALYZING)
        rag_output, rag_degraded = await run_rag_stage(
            self.rag_agent,
            event_id=event_id,
            triage_result=triage_result,
            evidence_output=evidence_output,
        )

        await self._transition(event_id, EventStatus.SCORING)
        risk_assessment = await self.risk_agent.execute(
            RiskAgentInput(
                event_id=event_id,
                triage_result=triage_result,
                evidence_output=evidence_output,
                rag_output=rag_output,
            )
        )
        if not isinstance(risk_assessment, RiskAssessment):
            raise TypeError("RiskAgent must return RiskAssessment")

        final_verdict = await _read_persisted_final_verdict(self.event_service, event_id)

        await self._transition(event_id, EventStatus.REPORTING)
        report = await self.report_agent.execute(
            ReportAgentInput(
                event_id=event_id,
                evidence_output=evidence_output,
                risk_assessment=risk_assessment,
            )
        )
        if report is not None and not isinstance(report, InvestigationReport):
            raise TypeError("ReportAgent must return InvestigationReport or None")

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
    ) -> None:
        if self.event_service is None:
            return
        transition = getattr(self.event_service, "transition_status", None)
        if transition is None:
            return
        await transition(
            event_id,
            target,
            context=context,
            operator=_PIPELINE_OPERATOR,
            reason=f"analysis_only:{target.value}",
        )
