"""LangGraph investigation workflow skeleton (ISSUE-048/ISSUE-049)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.planner_agent import PlannerAgent
from app.agents.rag_agent import RAGAgent
from app.core.errors import InvalidStateTransitionError
from app.models.agent_io import (
    CollectionStatus,
    EvidenceAgentInput,
    EvidenceOutput,
    ExecutionPlan,
    PlannerAgentInput,
    RAGOutput,
    ReportAgentInput,
    RiskAgentInput,
    RiskAssessment,
    ScoringMode,
    TriageAgentInput,
    TriageResult,
)
from app.models.context import EventContext
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    ExecutionSubstate,
    FinalVerdict,
    Severity,
    WritebackReadiness,
)
from app.models.security_event import EventSummary
from app.models.workflow import TransitionContext
from app.orchestration.graph_state import InvestigationState
from app.services.analysis_only_pipeline import run_rag_stage
from app.services.context_service import EventContextStore
from app.services.degraded_flag_service import DegradedFlagService
from app.services.state_machine_service import StateMachineService

logger = logging.getLogger(__name__)

CompiledInvestigationGraph = CompiledStateGraph[
    InvestigationState, None, InvestigationState, InvestigationState
]

_GRAPH_OPERATOR = "InvestigationGraph"

NODE_TRIAGE = "triage_node"
NODE_BEGIN_DISPOSITION_ONLY = "begin_disposition_only_node"
NODE_MANUAL_HOLD = "manual_hold_node"
NODE_CLOSE = "close_node"
NODE_PLANNER = "planner_node"
NODE_EVIDENCE = "evidence_node"
NODE_RAG = "rag_node"
NODE_RISK = "risk_node"
NODE_RESPONSE = "response_node"
NODE_APPROVAL = "approval_node"
NODE_APPROVAL_WAIT = "approval_wait_node"
NODE_EXECUTE = "execute_node"
NODE_VERIFY = "verify_node"
NODE_REPLAN = "replan_node"
NODE_REPORT = "report_node"
NODE_HALT = "halt_node"

P0_NODE_SEQUENCE = (
    NODE_TRIAGE,
    NODE_PLANNER,
    NODE_EVIDENCE,
    NODE_RISK,
    NODE_RESPONSE,
    NODE_APPROVAL,
    NODE_EXECUTE,
    NODE_VERIFY,
    NODE_REPORT,
    NODE_CLOSE,
)

ROUTE_CLOSE = "close"
ROUTE_DISPOSITION_ONLY = "disposition_only"
ROUTE_MANUAL_HOLD = "manual_hold"
ROUTE_INVESTIGATE = "investigate"
ROUTE_RESPONSE = "response"
ROUTE_EVIDENCE = "evidence"
ROUTE_EXECUTE = "execute"
ROUTE_WAIT = "wait"
ROUTE_REPORT = "report"
ROUTE_REPLAN = "replan"
ROUTE_MANUAL = "manual"
ROUTE_WRITEBACK = "writeback"
ROUTE_HALT = "halt"


class _AgentLike(Protocol):
    async def execute(self, input: Any) -> Any: ...


class _WorkflowRuntimeLike(Protocol):
    async def get_event_status_update_readiness(
        self,
        event_id: str,
    ) -> WritebackReadiness: ...

    async def begin_disposition_only(self, event_id: str) -> None: ...

    async def read_disposition_only_intent(self, event_id: str) -> bool: ...

    async def set_execution_substate(
        self,
        event_id: str,
        substate: ExecutionSubstate,
        *,
        event_status: EventStatus,
    ) -> None: ...

    async def assert_disposition_only_transition_allowed(
        self,
        event_id: str,
        *,
        current: EventStatus,
        target: EventStatus,
    ) -> None: ...


class _EventServiceLike(Protocol):
    async def set_final_verdict(
        self,
        event_id: str,
        verdict: FinalVerdict,
        *,
        operator: str | None = None,
    ) -> Any: ...


def _is_close_as_fp(state: InvestigationState) -> bool:
    fp = state.get("false_positive_match") or {}
    return fp.get("recommendation") == "close_as_fp"


def route_after_triage(state: InvestigationState) -> str:
    """Mirror the locked TRIAGING gates without broadening them."""
    policy = DispositionPolicy(
        state.get("disposition_policy", DispositionPolicy.NOT_REQUIRED.value)
    )
    is_fp = _is_close_as_fp(state)
    if policy is DispositionPolicy.NOT_REQUIRED and (
        state.get("need_investigation") is False or is_fp
    ):
        return ROUTE_CLOSE
    if policy is DispositionPolicy.REQUIRED and is_fp:
        readiness = WritebackReadiness(
            state.get(
                "event_status_update_readiness",
                WritebackReadiness.CAPABILITY_UNKNOWN.value,
            )
        )
        return (
            ROUTE_DISPOSITION_ONLY if readiness is WritebackReadiness.READY else ROUTE_MANUAL_HOLD
        )
    return ROUTE_INVESTIGATE


def route_after_planner(state: InvestigationState) -> str:
    """Use only the trusted intent hydrated by planner_graph_node."""
    return ROUTE_RESPONSE if state.get("disposition_only_intent") else ROUTE_EVIDENCE


def route_after_risk(state: InvestigationState) -> str:
    return ROUTE_RESPONSE


def route_after_approval(state: InvestigationState) -> str:
    if state.get("execution_substate") == ExecutionSubstate.WAITING_APPROVAL.value:
        return ROUTE_WAIT
    return ROUTE_EXECUTE


def route_after_verify(state: InvestigationState) -> str:
    if state.get("verify_need_manual_resolution"):
        return ROUTE_MANUAL
    if state.get("verify_need_writeback_recovery"):
        return ROUTE_WRITEBACK
    if state.get("verify_need_action_replan"):
        return ROUTE_REPLAN
    if (
        state.get("disposition_only_intent")
        or state.get("disposition_policy") == DispositionPolicy.REQUIRED.value
    ):
        return ROUTE_HALT
    return ROUTE_REPORT


def _route_after_response(state: InvestigationState) -> str:
    return ROUTE_HALT if state.get("halted") else ROUTE_EXECUTE


def _trace(node_name: str) -> InvestigationState:
    return cast(InvestigationState, {"node_trace": [node_name]})


def _patch_state(*parts: Mapping[str, Any]) -> InvestigationState:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return cast(InvestigationState, merged)


async def _mark_graph_failed(
    services: dict[str, Any],
    state: InvestigationState,
    error: Exception,
) -> None:
    state_machine = cast(StateMachineService, services["state_machine"])
    reason = f"investigation_graph:error:{type(error).__name__}:{error}"[:500]
    try:
        await state_machine.transition(
            state["event_id"],
            EventStatus.FAILED,
            operator=_GRAPH_OPERATOR,
            reason=reason,
        )
    except Exception:
        logger.exception("failed to mark event=%s FAILED", state.get("event_id"))


def _wrap_node(
    services: dict[str, Any],
    fn: Callable[[InvestigationState], Coroutine[Any, Any, InvestigationState]],
) -> Callable[[InvestigationState], Coroutine[Any, Any, InvestigationState]]:
    async def wrapped(state: InvestigationState) -> InvestigationState:
        try:
            return await fn(state)
        except Exception as exc:
            await _mark_graph_failed(services, state, exc)
            raise

    return wrapped


async def invoke_investigation_graph(
    graph: CompiledInvestigationGraph,
    state: InvestigationState | None,
    config: RunnableConfig,
) -> InvestigationState:
    """Invoke or resume a graph with its configured checkpoint saver."""
    result = await graph.ainvoke(state, config)
    return cast(InvestigationState, result)


def _event_context_from_state(state: InvestigationState) -> EventContext:
    policy = DispositionPolicy(
        state.get("disposition_policy", DispositionPolicy.NOT_REQUIRED.value)
    )
    summary = EventSummary(
        event_id=state["event_id"],
        event_type=EventType.OTHER,
        title="investigation",
        status=EventStatus(state.get("event_status", EventStatus.TRIAGING.value)),
        severity=Severity(state.get("severity", Severity.MEDIUM.value)),
        risk_score=0,
        final_verdict=FinalVerdict(state.get("final_verdict") or FinalVerdict.NONE.value),
        writeback_required=policy is DispositionPolicy.REQUIRED,
        writeback_readiness=WritebackReadiness(
            state.get(
                "event_status_update_readiness",
                WritebackReadiness.NOT_REQUIRED.value,
            )
        ),
        disposition_policy=policy,
    )
    return EventContext(
        event=summary,
        triage_result=state.get("triage_result"),
        false_positive_match=state.get("false_positive_match"),
        source_snapshot=state.get("source_snapshot"),
        disposition_only_intent=bool(state.get("disposition_only_intent")),
        execution_substate=ExecutionSubstate(
            state.get("execution_substate", ExecutionSubstate.NONE.value)
        ),
        execution_plan=state.get("execution_plan"),
    )


async def _transition_status(
    services: dict[str, Any],
    state: InvestigationState,
    target: EventStatus,
    *,
    context: TransitionContext | None = None,
    reason: str,
) -> InvestigationState:
    state_machine = cast(StateMachineService, services["state_machine"])
    await state_machine.transition(
        state["event_id"],
        target,
        context=context,
        operator=_GRAPH_OPERATOR,
        reason=reason,
    )
    return cast(InvestigationState, {"event_status": target.value})


async def _hydrate_context(
    services: dict[str, Any],
    event_id: str,
    target: dict[str, Any],
) -> None:
    store = cast(EventContextStore, services["context_store"])
    context = await store.get_full_context(event_id)
    if context.false_positive_match is not None:
        target["false_positive_match"] = context.false_positive_match
    if context.source_snapshot is not None:
        target["source_snapshot"] = context.source_snapshot
    if context.event is not None:
        target["disposition_policy"] = context.event.disposition_policy.value
        target["severity"] = context.event.severity.value
        target["event_status"] = context.event.status.value
        target["final_verdict"] = context.event.final_verdict.value


def build_investigation_graph(
    agents: dict[str, Any],
    services: dict[str, Any],
    *,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledInvestigationGraph:
    """Build the investigation graph exclusively from injected dependencies."""
    required_services = (
        "state_machine",
        "event_service",
        "workflow_runtime",
        "degraded_flags",
        "context_store",
    )
    missing_services = [name for name in required_services if services.get(name) is None]
    if missing_services:
        raise ValueError(f"missing required workflow services: {', '.join(missing_services)}")

    triage_agent = cast(_AgentLike, agents["triage_agent"])
    planner_agent = cast(PlannerAgent, agents["planner_agent"])
    evidence_agent = cast(_AgentLike, agents["evidence_agent"])
    risk_agent = cast(_AgentLike, agents["risk_agent"])
    report_agent = cast(_AgentLike, agents["report_agent"])
    rag_agent = cast(RAGAgent | None, agents.get("rag_agent"))
    runtime = cast(_WorkflowRuntimeLike, services["workflow_runtime"])
    event_service = cast(_EventServiceLike, services["event_service"])
    degraded_flags = cast(DegradedFlagService, services["degraded_flags"])

    async def triage_graph_node(state: InvestigationState) -> InvestigationState:
        result = await triage_agent.execute(
            TriageAgentInput(event_id=state["event_id"], raw_event_summary="")
        )
        if not isinstance(result, TriageResult):
            raise TypeError("triage_agent must return TriageResult")
        update: dict[str, Any] = {
            "triage_result": result.model_dump(mode="json"),
            "need_investigation": result.need_investigation,
            "severity": result.severity.value,
        }
        await _hydrate_context(services, state["event_id"], update)
        update["event_status_update_readiness"] = (
            await runtime.get_event_status_update_readiness(state["event_id"])
        ).value
        return _patch_state(_trace(NODE_TRIAGE), update)

    async def begin_disposition_only_node(
        state: InvestigationState,
    ) -> InvestigationState:
        await runtime.begin_disposition_only(state["event_id"])
        current = EventStatus(state.get("event_status", EventStatus.TRIAGING.value))
        await runtime.assert_disposition_only_transition_allowed(
            state["event_id"],
            current=current,
            target=EventStatus.PLANNING_RESPONSE,
        )
        status = await _transition_status(
            services,
            state,
            EventStatus.PLANNING_RESPONSE,
            context=TransitionContext(
                final_verdict=FinalVerdict.FALSE_POSITIVE,
                disposition_only_intent=True,
                disposition_policy=DispositionPolicy.REQUIRED,
                recommendation="close_as_fp",
            ),
            reason="disposition_only:begin",
        )
        return _patch_state(
            _trace(NODE_BEGIN_DISPOSITION_ONLY),
            status,
            {
                "disposition_only_intent": True,
                "final_verdict": FinalVerdict.FALSE_POSITIVE.value,
            },
        )

    async def manual_hold_node(state: InvestigationState) -> InvestigationState:
        readiness = state.get(
            "event_status_update_readiness",
            WritebackReadiness.CAPABILITY_UNKNOWN.value,
        )
        flags = list(state.get("degraded_flags") or [])
        entry = f"disposition_writeback_blocked={readiness}"
        if entry not in flags:
            flags.append(entry)
        flags = await degraded_flags.set_flag(
            state["event_id"],
            "disposition_writeback_blocked",
            readiness,
            writer="DegradedFlagService",
        )
        return _patch_state(
            _trace(NODE_MANUAL_HOLD),
            {
                "degraded_flags": flags,
                "execution_substate": ExecutionSubstate.NONE.value,
                "halted": True,
            },
        )

    async def close_node(state: InvestigationState) -> InvestigationState:
        triage = TriageResult.model_validate(state["triage_result"])
        final_verdict = state.get("final_verdict")
        short_circuit = state.get("risk_assessment") is None
        if short_circuit and not final_verdict:
            await event_service.set_final_verdict(
                state["event_id"],
                FinalVerdict.FALSE_POSITIVE,
                operator=_GRAPH_OPERATOR,
            )
            final_verdict = FinalVerdict.FALSE_POSITIVE.value

        report_generated = bool(state.get("report_generated"))
        if not report_generated:
            evidence = EvidenceOutput(
                evidence_list=[],
                conflicts=[],
                gaps=[],
                success_sources=[],
                failed_sources=[],
                overall_confidence=0.0,
                collection_status=CollectionStatus.COMPLETED,
            )
            assessment = RiskAssessment(
                risk_score=0,
                severity=triage.severity,
                confidence=0.9,
                risk_factors=[],
                possible_false_positive=True,
                scoring_mode=ScoringMode.RULE_ONLY,
            )
            report = await report_agent.execute(
                ReportAgentInput(
                    event_id=state["event_id"],
                    evidence_output=evidence,
                    risk_assessment=assessment,
                )
            )
            report_generated = report is not None

        status = await _transition_status(
            services,
            state,
            EventStatus.CLOSED,
            context=TransitionContext(
                need_investigation=triage.need_investigation,
                disposition_policy=DispositionPolicy(
                    state.get(
                        "disposition_policy",
                        DispositionPolicy.NOT_REQUIRED.value,
                    )
                ),
                severity=triage.severity,
                recommendation=((state.get("false_positive_match") or {}).get("recommendation")),
                final_verdict=FinalVerdict(final_verdict) if final_verdict else None,
                report_exists=report_generated,
            ),
            reason="investigation:close",
        )
        return _patch_state(
            _trace(NODE_CLOSE),
            status,
            {
                "final_verdict": final_verdict,
                "report_generated": report_generated,
                "halted": False,
            },
        )

    async def planner_graph_node(state: InvestigationState) -> InvestigationState:
        persisted = await runtime.read_disposition_only_intent(state["event_id"])
        if state.get("disposition_only_intent") and not persisted:
            raise InvalidStateTransitionError(
                "forged disposition_only_intent without server persistence",
                current=state.get("event_status", EventStatus.TRIAGING.value),
                target=EventStatus.PLANNING_RESPONSE.value,
                details={"event_id": state["event_id"]},
            )
        context = _event_context_from_state(
            _patch_state(state, {"disposition_only_intent": persisted})
        )
        plan = await planner_node(
            context,
            planner_agent,
            disposition_only=persisted,
        )
        return _patch_state(
            _trace(NODE_PLANNER),
            {
                "execution_plan": plan.model_dump(mode="json"),
                "disposition_only_intent": persisted,
            },
        )

    async def evidence_node(state: InvestigationState) -> InvestigationState:
        triage = TriageResult.model_validate(state["triage_result"])
        status = await _transition_status(
            services,
            state,
            EventStatus.COLLECTING_EVIDENCE,
            context=TransitionContext(need_investigation=True),
            reason="investigation:evidence",
        )
        result = await evidence_agent.execute(
            EvidenceAgentInput(event_id=state["event_id"], triage_result=triage)
        )
        if not isinstance(result, EvidenceOutput):
            raise TypeError("evidence_agent must return EvidenceOutput")
        await _transition_status(
            services,
            _patch_state(state, status),
            EventStatus.ANALYZING,
            reason="investigation:analyze",
        )
        return _patch_state(
            _trace(NODE_EVIDENCE),
            {
                "event_status": EventStatus.ANALYZING.value,
                "evidence_output": result.model_dump(mode="json"),
            },
        )

    async def rag_graph_node(state: InvestigationState) -> InvestigationState:
        if rag_agent is None:
            return _trace(NODE_RAG)
        output = await rag_node(
            _event_context_from_state(state),
            rag_agent,
            triage_result=TriageResult.model_validate(state["triage_result"]),
            evidence_output=EvidenceOutput.model_validate(state["evidence_output"]),
        )
        update: dict[str, Any] = {}
        if output is not None:
            update["rag_output"] = output.model_dump(mode="json")
        return _patch_state(_trace(NODE_RAG), update)

    async def risk_node(state: InvestigationState) -> InvestigationState:
        rag_output = (
            RAGOutput.model_validate(state["rag_output"])
            if state.get("rag_output") is not None
            else None
        )
        await _transition_status(
            services,
            state,
            EventStatus.SCORING,
            reason="investigation:score",
        )
        result = await risk_agent.execute(
            RiskAgentInput(
                event_id=state["event_id"],
                triage_result=TriageResult.model_validate(state["triage_result"]),
                evidence_output=EvidenceOutput.model_validate(state["evidence_output"]),
                rag_output=rag_output,
            )
        )
        if not isinstance(result, RiskAssessment):
            raise TypeError("risk_agent must return RiskAssessment")
        await _transition_status(
            services,
            state,
            EventStatus.PLANNING_RESPONSE,
            reason="investigation:plan_response",
        )
        update: dict[str, Any] = {
            "event_status": EventStatus.PLANNING_RESPONSE.value,
            "risk_assessment": result.model_dump(mode="json"),
            "severity": result.severity.value,
        }
        await _hydrate_context(services, state["event_id"], update)
        update["event_status"] = EventStatus.PLANNING_RESPONSE.value
        return _patch_state(
            _trace(NODE_RISK),
            update,
        )

    async def response_node(state: InvestigationState) -> InvestigationState:
        if state.get("disposition_only_intent"):
            return _patch_state(
                _trace(NODE_RESPONSE),
                {"halted": True},
            )
        verdict_raw = state.get("final_verdict")
        status = await _transition_status(
            services,
            state,
            EventStatus.WAITING_APPROVAL,
            context=TransitionContext(
                disposition_only_intent=bool(state.get("disposition_only_intent")),
                final_verdict=FinalVerdict(verdict_raw) if verdict_raw else None,
            ),
            reason="investigation:response_stub",
        )
        return _patch_state(_trace(NODE_RESPONSE), status)

    async def approval_node(state: InvestigationState) -> InvestigationState:
        if state.get("needs_approval_wait"):
            await runtime.set_execution_substate(
                state["event_id"],
                ExecutionSubstate.WAITING_APPROVAL,
                event_status=EventStatus.WAITING_APPROVAL,
            )
            return _patch_state(
                _trace(NODE_APPROVAL),
                {"execution_substate": ExecutionSubstate.WAITING_APPROVAL.value},
            )
        await runtime.set_execution_substate(
            state["event_id"],
            ExecutionSubstate.NONE,
            event_status=EventStatus.WAITING_APPROVAL,
        )
        status = await _transition_status(
            services,
            state,
            EventStatus.EXECUTING_RESPONSE,
            reason="investigation:approval_stub",
        )
        return _patch_state(
            _trace(NODE_APPROVAL),
            status,
            {"execution_substate": ExecutionSubstate.NONE.value},
        )

    async def approval_wait_node(state: InvestigationState) -> InvestigationState:
        return _patch_state(
            _trace(NODE_APPROVAL_WAIT),
            {"halted": True},
        )

    async def execute_node(state: InvestigationState) -> InvestigationState:
        status = await _transition_status(
            services,
            state,
            EventStatus.VERIFYING,
            reason="investigation:execute_stub",
        )
        return _patch_state(_trace(NODE_EXECUTE), status)

    async def verify_node(state: InvestigationState) -> InvestigationState:
        if (
            state.get("disposition_only_intent")
            or state.get("disposition_policy") == DispositionPolicy.REQUIRED.value
        ):
            return _trace(NODE_VERIFY)
        status = await _transition_status(
            services,
            state,
            EventStatus.REPORTING,
            reason="investigation:verify_stub",
        )
        return _patch_state(_trace(NODE_VERIFY), status)

    async def replan_node(state: InvestigationState) -> InvestigationState:
        status = await _transition_status(
            services,
            state,
            EventStatus.REPLANNING,
            reason="investigation:replan_stub",
        )
        return _patch_state(_trace(NODE_REPLAN), status)

    async def report_node(state: InvestigationState) -> InvestigationState:
        report = await report_agent.execute(
            ReportAgentInput(
                event_id=state["event_id"],
                evidence_output=EvidenceOutput.model_validate(state["evidence_output"]),
                risk_assessment=RiskAssessment.model_validate(state["risk_assessment"]),
            )
        )
        return _patch_state(
            _trace(NODE_REPORT),
            {
                "event_status": EventStatus.REPORTING.value,
                "report_generated": report is not None,
            },
        )

    async def halt_node(state: InvestigationState) -> InvestigationState:
        return _patch_state(_trace(NODE_HALT), {"halted": True})

    graph: StateGraph[InvestigationState] = StateGraph(InvestigationState)

    def register(
        name: str,
        node: Callable[
            [InvestigationState],
            Coroutine[Any, Any, InvestigationState],
        ],
    ) -> None:
        graph.add_node(name, cast(Any, _wrap_node(services, node)))

    register(NODE_TRIAGE, triage_graph_node)
    register(NODE_BEGIN_DISPOSITION_ONLY, begin_disposition_only_node)
    register(NODE_MANUAL_HOLD, manual_hold_node)
    register(NODE_CLOSE, close_node)
    register(NODE_PLANNER, planner_graph_node)
    register(NODE_EVIDENCE, evidence_node)
    register(NODE_RISK, risk_node)
    register(NODE_RESPONSE, response_node)
    register(NODE_APPROVAL, approval_node)
    register(NODE_APPROVAL_WAIT, approval_wait_node)
    register(NODE_EXECUTE, execute_node)
    register(NODE_VERIFY, verify_node)
    register(NODE_REPLAN, replan_node)
    register(NODE_REPORT, report_node)
    register(NODE_HALT, halt_node)
    if rag_agent is not None:
        register(NODE_RAG, rag_graph_node)

    graph.add_edge(START, NODE_TRIAGE)
    graph.add_conditional_edges(
        NODE_TRIAGE,
        route_after_triage,
        {
            ROUTE_CLOSE: NODE_CLOSE,
            ROUTE_DISPOSITION_ONLY: NODE_BEGIN_DISPOSITION_ONLY,
            ROUTE_MANUAL_HOLD: NODE_MANUAL_HOLD,
            ROUTE_INVESTIGATE: NODE_PLANNER,
        },
    )
    graph.add_edge(NODE_BEGIN_DISPOSITION_ONLY, NODE_PLANNER)
    graph.add_edge(NODE_MANUAL_HOLD, END)
    graph.add_conditional_edges(
        NODE_PLANNER,
        route_after_planner,
        {
            ROUTE_RESPONSE: NODE_RESPONSE,
            ROUTE_EVIDENCE: NODE_EVIDENCE,
        },
    )
    if rag_agent is not None:
        graph.add_edge(NODE_EVIDENCE, NODE_RAG)
        graph.add_edge(NODE_RAG, NODE_RISK)
    else:
        graph.add_edge(NODE_EVIDENCE, NODE_RISK)
    graph.add_conditional_edges(
        NODE_RISK,
        route_after_risk,
        {ROUTE_RESPONSE: NODE_RESPONSE},
    )
    graph.add_conditional_edges(
        NODE_RESPONSE,
        _route_after_response,
        {
            ROUTE_HALT: NODE_HALT,
            ROUTE_EXECUTE: NODE_APPROVAL,
        },
    )
    graph.add_conditional_edges(
        NODE_APPROVAL,
        route_after_approval,
        {
            ROUTE_EXECUTE: NODE_EXECUTE,
            ROUTE_WAIT: NODE_APPROVAL_WAIT,
        },
    )
    graph.add_edge(NODE_APPROVAL_WAIT, END)
    graph.add_edge(NODE_EXECUTE, NODE_VERIFY)
    graph.add_conditional_edges(
        NODE_VERIFY,
        route_after_verify,
        {
            ROUTE_REPORT: NODE_REPORT,
            ROUTE_REPLAN: NODE_REPLAN,
            ROUTE_MANUAL: NODE_MANUAL_HOLD,
            # P0 placeholder: writeback recovery shares manual_hold until ISSUE-062.
            ROUTE_WRITEBACK: NODE_MANUAL_HOLD,
            ROUTE_HALT: NODE_HALT,
        },
    )
    graph.add_edge(NODE_REPLAN, NODE_PLANNER)
    graph.add_edge(NODE_REPORT, NODE_CLOSE)
    graph.add_edge(NODE_CLOSE, END)
    graph.add_edge(NODE_HALT, END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )


def _synthesized_fallback_triage(
    event_context: EventContext,
    *,
    reasoning: str,
) -> TriageResult:
    return TriageResult(
        event_type=EventType.OTHER,
        severity=event_context.event.severity if event_context.event else Severity.MEDIUM,
        need_investigation=True,
        reasoning=reasoning,
        degraded=True,
    )


async def planner_node(
    event_context: EventContext,
    planner: PlannerAgent,
    *,
    disposition_only: bool = False,
) -> ExecutionPlan:
    """Generate or revise an investigation plan for the given event context.

    This is the canonical entry point for the ``planner_node`` in the
    LangGraph investigation workflow (ISSUE-048 / ISSUE-054).

    Args:
        event_context: The current ``EventContext``, which must carry at least
            a valid ``event_id`` and, for normal paths, a ``triage_result``.
        planner: A configured ``PlannerAgent`` instance (LLM client + working
            memory already injected).
        disposition_only: When ``True``, produce the deterministic single-step
            disposition-only plan instead of a full investigation plan.

    Returns:
        The generated ``ExecutionPlan`` (already persisted to
        ``EventContext.execution_plan`` via working memory).
    """
    event_id = event_context.event.event_id if event_context.event else "unknown"

    if disposition_only:
        logger.info(
            "planner_node: generating disposition-only plan for event=%s",
            event_id,
        )
        return await planner.plan_disposition_only(event_context)

    triage_data = event_context.triage_result
    triage_result: TriageResult | None = None
    if triage_data is not None:
        try:
            triage_result = TriageResult.model_validate(triage_data)
        except Exception:
            logger.warning(
                "planner_node: corrupt triage_result in EventContext for event=%s, "
                "falling back to DEFAULT_PLANS (EventType.OTHER)",
                event_id,
                exc_info=True,
            )
            triage_result = _synthesized_fallback_triage(
                event_context,
                reasoning="triage data corrupt — using conservative rule-based plan",
            )
    else:
        logger.warning(
            "planner_node: missing triage_result for event=%s, "
            "using conservative DEFAULT_PLANS path",
            event_id,
        )
        triage_result = _synthesized_fallback_triage(
            event_context,
            reasoning="triage unavailable — using conservative rule-based plan",
        )

    if event_context.replan_count > 0:
        existing_plan_data = event_context.execution_plan
        if existing_plan_data is not None:
            try:
                previous_plan = ExecutionPlan.model_validate(existing_plan_data)
                logger.info(
                    "planner_node: revising plan for event=%s replan_count=%d",
                    event_id,
                    event_context.replan_count,
                )
                return await planner.revise(
                    event_context,
                    failure_reason=(f"replan triggered (count={event_context.replan_count})"),
                    previous_plan=previous_plan,
                )
            except Exception:
                logger.warning(
                    "planner_node: failed to parse existing plan for revision, "
                    "falling back to fresh plan for event=%s",
                    event_id,
                    exc_info=True,
                )

    input = PlannerAgentInput(
        event_id=event_id,
        triage_result=triage_result,
    )
    return await planner.execute(input)


async def rag_node(
    event_context: EventContext,
    rag_agent: RAGAgent,
    *,
    triage_result: TriageResult,
    evidence_output: EvidenceOutput,
) -> RAGOutput | None:
    """LangGraph node: RAG retrieval after evidence, before risk (ISSUE-047).

    Failures degrade to ``None`` so RiskAgent can continue without enhancement.
    """
    event_id = event_context.event.event_id if event_context.event else "unknown"
    output, _degraded = await run_rag_stage(
        rag_agent,
        event_id=event_id,
        triage_result=triage_result,
        evidence_output=evidence_output,
    )
    return output


__all__ = [
    "NODE_APPROVAL",
    "NODE_APPROVAL_WAIT",
    "NODE_BEGIN_DISPOSITION_ONLY",
    "NODE_CLOSE",
    "NODE_EVIDENCE",
    "NODE_EXECUTE",
    "NODE_HALT",
    "NODE_MANUAL_HOLD",
    "NODE_PLANNER",
    "NODE_RAG",
    "NODE_REPORT",
    "NODE_RESPONSE",
    "NODE_RISK",
    "NODE_TRIAGE",
    "NODE_VERIFY",
    "P0_NODE_SEQUENCE",
    "build_investigation_graph",
    "invoke_investigation_graph",
    "planner_node",
    "rag_node",
    "route_after_approval",
    "route_after_planner",
    "route_after_risk",
    "route_after_triage",
    "route_after_verify",
]
