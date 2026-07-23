"""AnalysisOnlyPipeline — temporary sequential analysis-only pipeline (ISSUE-038).

Runs TriageAgent → EvidenceAgent → RiskAgent → ReportAgent sequentially,
advancing EventStatus via StateMachineService. Only allowed when
``ALLOW_LIVE_SIDE_EFFECTS=false`` and ``ALLOW_XDR_WRITEBACK=false``.

High-risk events that require disposition stay at REPORTING with
``analysis_only_complete=true``. Only ``disposition_policy=not_required``
low-severity / false-positive events may reach CLOSED through this pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.evidence_agent import EvidenceAgent
from app.agents.report_agent import ReportAgent
from app.agents.risk_agent import RiskAgent
from app.agents.triage_agent import TriageAgent
from app.core.config import get_settings
from app.core.errors import (
    InvalidStateTransitionError,
    ShadowTraceError,
    ValidationError,
)
from app.models.agent_io import (
    CollectionStatus,
    EvidenceAgentInput,
    EvidenceOutput,
    ReportAgentInput,
    RiskAgentInput,
    RiskAssessment,
    ScoringMode,
    TriageAgentInput,
    TriageResult,
)
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
)
from app.models.workflow import TransitionContext
from app.services.event_service import EventService, StateMachinePort

logger = logging.getLogger(__name__)

_PIPELINE_OPERATOR = "AnalysisOnlyPipeline"


class AnalysisOnlyPipeline:
    """Temporary sequential analysis pipeline (pre-SuperAgent, ISSUE-054).

    Only runs in dev/offline mode (ALLOW_LIVE_SIDE_EFFECTS=false,
    ALLOW_XDR_WRITEBACK=false). High-risk required-disposition events
    stay at REPORTING — they never auto-close.

    Requires event in NEW status; call exactly once per event.
    """

    def __init__(
        self,
        event_service: EventService,
        state_machine: StateMachinePort,
        *,
        triage_agent: TriageAgent,
        evidence_agent: EvidenceAgent,
        risk_agent: RiskAgent,
        report_agent: ReportAgent,
        context_store: Any | None = None,
        degraded_flags: Any | None = None,
    ) -> None:
        self._event_service = event_service
        self._state_machine = state_machine
        self._triage = triage_agent
        self._evidence = evidence_agent
        self._risk = risk_agent
        self._report = report_agent
        self._context_store = context_store
        self._degraded_flags = degraded_flags

    async def run(self, event_id: str) -> dict[str, Any]:
        """Execute the analysis-only pipeline for *event_id*.

        Returns a summary dict with final status and analysis_only flag.
        """
        settings = get_settings()
        if settings.allow_live_side_effects:
            raise ValidationError(
                "AnalysisOnlyPipeline requires ALLOW_LIVE_SIDE_EFFECTS=false",
                error_code="configuration_error",
                details={"allow_live_side_effects": True},
            )
        if settings.allow_xdr_writeback:
            raise ValidationError(
                "AnalysisOnlyPipeline requires ALLOW_XDR_WRITEBACK=false",
                error_code="configuration_error",
                details={"allow_xdr_writeback": True},
            )

        # Load the event.
        event = await self._event_service.get_event(event_id)
        if event is None:
            raise ShadowTraceError(
                f"event {event_id} not found",
                error_code="event_not_found",
            )

        current_status = event.status

        # Guard: only NEW events can start the pipeline.
        if current_status is not EventStatus.NEW:
            raise InvalidStateTransitionError(
                f"AnalysisOnlyPipeline requires event in NEW status, got {current_status.value}",
                current=current_status,
                target=EventStatus.TRIAGING,
                details={"event_id": event_id},
            )

        # ---- Step 1: Triage ----
        await self._state_machine.transition(
            event_id,
            EventStatus.TRIAGING,
            operator=_PIPELINE_OPERATOR,
            reason="analysis_pipeline:triage_start",
        )

        triage_result = await self._run_triage(event_id, event)
        logger.info(
            "AnalysisOnlyPipeline triage complete event=%s type=%s severity=%s need_inv=%s",
            event_id,
            triage_result.event_type.value,
            triage_result.severity.value,
            triage_result.need_investigation,
        )

        # ---- Short-circuit: not_required low/fp → quick close ----
        if (
            not triage_result.need_investigation
            and event.disposition_policy == DispositionPolicy.NOT_REQUIRED
        ):
            return await self._short_circuit_close(event_id, event, triage_result)

        # ---- Step 2: Evidence ----
        await self._state_machine.transition(
            event_id,
            EventStatus.COLLECTING_EVIDENCE,
            operator=_PIPELINE_OPERATOR,
            reason="analysis_pipeline:evidence_collect",
        )
        await self._state_machine.transition(
            event_id,
            EventStatus.ANALYZING,
            operator=_PIPELINE_OPERATOR,
            reason="analysis_pipeline:evidence_analyze",
        )
        evidence_output = await self._run_evidence(event_id, triage_result)

        # ---- Step 3: Risk Scoring ----
        await self._state_machine.transition(
            event_id,
            EventStatus.SCORING,
            operator=_PIPELINE_OPERATOR,
            reason="analysis_pipeline:risk_score",
        )
        risk_assessment = await self._run_risk(event_id, triage_result, evidence_output)

        # ---- Step 4: Report ----
        # Transition to REPORTING must happen before report generation so the
        # report_exists gate in StateMachineService can see it.
        await self._state_machine.transition(
            event_id,
            EventStatus.REPORTING,
            operator=_PIPELINE_OPERATOR,
            reason="analysis_pipeline:report_generate",
        )
        await self._run_report(event_id, triage_result, evidence_output, risk_assessment)

        event = await self._event_service.get_event(event_id)
        if event is None:
            raise ShadowTraceError(
                f"event {event_id} disappeared during pipeline execution",
                error_code="event_not_found",
            )

        if event.disposition_policy == DispositionPolicy.REQUIRED:
            # High-risk: stay at REPORTING, mark analysis_only_complete.
            logger.info(
                "AnalysisOnlyPipeline: event=%s requires disposition, staying at REPORTING",
                event_id,
            )
            await self._persist_analysis_only_complete(event_id)
            return {
                "event_id": event_id,
                "status": EventStatus.REPORTING.value,
                "analysis_only_complete": True,
                "disposition_policy": "required",
            }

        # not_required → CLOSED (report already exists from Step 4)
        await self._state_machine.transition(
            event_id,
            EventStatus.CLOSED,
            context=TransitionContext(
                need_investigation=triage_result.need_investigation,
            ),
            operator=_PIPELINE_OPERATOR,
            reason="analysis_pipeline:complete_not_required",
        )
        await self._persist_analysis_only_complete(event_id)
        return {
            "event_id": event_id,
            "status": EventStatus.CLOSED.value,
            "analysis_only_complete": True,
            "disposition_policy": "not_required",
        }

    # ------------------------------------------------------------------ #
    # Step runners
    # ------------------------------------------------------------------ #

    async def _run_triage(self, event_id: str, event: Any) -> TriageResult:
        """Run TriageAgent and return TriageResult."""
        raw_summary = f"{event.title}. {event.description}"
        triage_input = TriageAgentInput(
            event_id=event_id,
            raw_event_summary=raw_summary,
            hint_entities=event.entities,
        )
        return await self._triage.execute(triage_input)

    async def _run_evidence(self, event_id: str, triage_result: TriageResult) -> EvidenceOutput:
        """Run EvidenceAgent and return EvidenceOutput."""
        evidence_input = EvidenceAgentInput(
            event_id=event_id,
            triage_result=triage_result,
        )
        return await self._evidence.execute(evidence_input)

    async def _run_risk(
        self,
        event_id: str,
        triage_result: TriageResult,
        evidence_output: EvidenceOutput,
    ) -> RiskAssessment:
        """Run RiskAgent and return RiskAssessment."""
        risk_input = RiskAgentInput(
            event_id=event_id,
            triage_result=triage_result,
            evidence_output=evidence_output,
        )
        return await self._risk.execute(risk_input)

    async def _run_report(
        self,
        event_id: str,
        triage_result: TriageResult,
        evidence_output: EvidenceOutput,
        risk_assessment: RiskAssessment,
    ) -> None:
        """Generate and persist the investigation report."""
        report_input = ReportAgentInput(
            event_id=event_id,
            triage_result=triage_result,
            evidence_output=evidence_output,
            risk_assessment=risk_assessment,
        )
        await self._report.execute(report_input)

    # ------------------------------------------------------------------ #
    # Short-circuit close
    # ------------------------------------------------------------------ #

    async def _persist_analysis_only_complete(self, event_id: str) -> None:
        """Persist analysis_only_complete=true to EventContextStore.

        On Redis failure, sets the ``redis_context_unavailable`` degraded flag
        so downstream systems (ISSUE-039 integration tests, ISSUE-054 SuperAgent
        takeover) can detect the degraded state rather than silently missing the
        signal.
        """
        if self._context_store is not None:
            try:
                await self._context_store.set(event_id, "analysis_only_complete", True)
            except Exception:
                logger.warning(
                    "Failed to persist analysis_only_complete for event=%s",
                    event_id,
                    exc_info=True,
                )
                # Set degraded flag so downstream systems know the signal is
                # unreliable for this event.
                if self._degraded_flags is not None:
                    try:
                        await self._degraded_flags.set_flag(
                            event_id,
                            "redis_context_unavailable",
                            True,
                            writer="AnalysisOnlyPipeline",
                        )
                    except Exception:
                        logger.error(
                            "Failed to set redis_context_unavailable degraded flag "
                            "for event=%s",
                            event_id,
                            exc_info=True,
                        )

    async def _short_circuit_close(
        self,
        event_id: str,
        event: Any,
        triage_result: TriageResult,
    ) -> dict[str, Any]:
        """Generate a low-risk quick-close report and transition to CLOSED.

        Only for not_required + low-severity / false-positive events.
        Evidence, response, and verification sections use placeholder text;
        overview and recommendations explain the low-risk reason.
        """
        logger.info(
            "AnalysisOnlyPipeline: short-circuit close event=%s severity=%s",
            event_id,
            triage_result.severity.value,
        )

        # Generate quick-close report first (satisfies CLOSED gate's report_exists),
        # then transition directly TRIAGING→CLOSED.
        # TRIAGING→REPORTING is an illegal edge in STATE_TRANSITIONS; going
        # directly to CLOSED matches close_event's TRIAGING path (events.py:548-565)
        # and ISSUE-038's "TRIAGING + not_required → quick close" spec.

        # Build placeholder evidence/risk for the quick-close report.
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

        await self._run_report(event_id, triage_result, placeholder_evidence, placeholder_risk)

        # Transition directly TRIAGING→CLOSED (valid edge; gated by
        # disposition_policy=not_required + severity=LOW / close_as_fp).
        ctx = TransitionContext(
            need_investigation=False,
            recommendation="close_as_fp",
        )
        await self._state_machine.transition(
            event_id,
            EventStatus.CLOSED,
            context=ctx,
            operator=_PIPELINE_OPERATOR,
            reason="analysis_pipeline:short_circuit_closed",
        )

        await self._persist_analysis_only_complete(event_id)
        return {
            "event_id": event_id,
            "status": EventStatus.CLOSED.value,
            "analysis_only_complete": True,
            "disposition_policy": "not_required",
            "short_circuit": True,
        }
