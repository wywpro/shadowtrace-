"""ISSUE-048 StateGraph unit and recovery tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents.planner_agent import PlannerAgent
from app.core.errors import InvalidStateTransitionError
from app.models.agent_io import (
    CollectionStatus,
    EvidenceOutput,
    RiskAssessment,
    ScoringMode,
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
from app.models.workflow import TransitionContext, validate_transition
from app.orchestration.checkpointer import (
    CHECKPOINT_TTL_SECONDS,
    RedisCheckpointer,
    checkpoint_key_for_event,
)
from app.orchestration.graph_state import InvestigationState
from app.orchestration.workflow_graph import (
    NODE_BEGIN_DISPOSITION_ONLY,
    NODE_CLOSE,
    NODE_HALT,
    NODE_MANUAL_HOLD,
    NODE_RAG,
    NODE_RESPONSE,
    NODE_RISK,
    P0_NODE_SEQUENCE,
    ROUTE_CLOSE,
    ROUTE_DISPOSITION_ONLY,
    ROUTE_EVIDENCE,
    ROUTE_EXECUTE,
    ROUTE_HALT,
    ROUTE_INVESTIGATE,
    ROUTE_MANUAL,
    ROUTE_MANUAL_HOLD,
    ROUTE_REPLAN,
    ROUTE_REPORT,
    ROUTE_RESPONSE,
    ROUTE_WAIT,
    ROUTE_WRITEBACK,
    build_investigation_graph,
    route_after_approval,
    route_after_planner,
    route_after_risk,
    route_after_triage,
    route_after_verify,
)


def _base_state(**overrides: Any) -> InvestigationState:
    state: InvestigationState = {
        "event_id": "evt-graph-001",
        "event_status": EventStatus.TRIAGING.value,
        "disposition_policy": DispositionPolicy.NOT_REQUIRED.value,
        "severity": Severity.HIGH.value,
        "final_verdict": None,
        "confidence": 0.0,
        "need_investigation": True,
        "execution_substate": ExecutionSubstate.NONE.value,
        "event_status_update_readiness": WritebackReadiness.NOT_REQUIRED.value,
        "degraded_flags": [],
        "node_trace": [],
        "halted": False,
        "disposition_only_intent": False,
        "report_generated": False,
        "needs_approval_wait": False,
    }
    state.update(overrides)
    return state


class StubAgent:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    async def execute(self, input: Any) -> Any:
        self.calls.append(input)
        return self.result


@dataclass
class FakeStateMachine:
    status: EventStatus = EventStatus.TRIAGING
    transitions: list[tuple[str, EventStatus, str | None]] = field(default_factory=list)
    statuses: dict[str, EventStatus] = field(default_factory=dict)

    async def transition(
        self,
        event_id: str,
        target: EventStatus,
        *,
        context: Any = None,
        operator: str | None = None,
        reason: str | None = None,
    ) -> Any:
        current = self.statuses.get(event_id, EventStatus.TRIAGING)
        validate_transition(current, target, context or TransitionContext())
        self.transitions.append((event_id, target, reason))
        self.statuses[event_id] = target
        self.status = target
        return SimpleNamespace(event_id=event_id, status=target)


class FakeEventService:
    def __init__(self) -> None:
        self.verdicts: list[FinalVerdict] = []

    async def set_final_verdict(
        self,
        event_id: str,
        verdict: FinalVerdict,
        *,
        operator: str | None = None,
    ) -> Any:
        self.verdicts.append(verdict)
        return SimpleNamespace(event_id=event_id, final_verdict=verdict)


class FakeRuntime:
    def __init__(
        self,
        readiness: WritebackReadiness = WritebackReadiness.NOT_REQUIRED,
    ) -> None:
        self.intent = False
        self.readiness = readiness
        self.begun: list[str] = []
        self.substates: list[ExecutionSubstate] = []

    async def get_event_status_update_readiness(
        self,
        event_id: str,
    ) -> WritebackReadiness:
        return self.readiness

    async def begin_disposition_only(self, event_id: str) -> None:
        self.begun.append(event_id)
        self.intent = True

    async def read_disposition_only_intent(self, event_id: str) -> bool:
        return self.intent

    async def set_execution_substate(
        self,
        event_id: str,
        substate: ExecutionSubstate,
        *,
        event_status: EventStatus,
    ) -> None:
        self.substates.append(substate)

    async def assert_disposition_only_transition_allowed(
        self,
        event_id: str,
        *,
        current: EventStatus,
        target: EventStatus,
    ) -> None:
        assert self.intent is True


class FakeDegradedFlags:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, str]] = []

    async def set_flag(
        self,
        event_id: str,
        flag_name: str,
        value: Any,
        writer: str,
    ) -> list[str]:
        self.calls.append((event_id, flag_name, value, writer))
        return [f"{flag_name}={value}"]


class FakeContextStore:
    async def get_full_context(self, event_id: str) -> EventContext:
        return EventContext()


class FakeRedisStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes, *, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)


class FakeRedisClient:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.store = FakeRedisStore()

    async def ping(self) -> bool:
        return self.available

    def get_client(self) -> FakeRedisStore:
        return self.store


def _agents(*, triage: TriageResult | None = None) -> dict[str, Any]:
    triage_result = triage or TriageResult(
        event_type=EventType.DATA_EXFILTRATION,
        severity=Severity.HIGH,
        need_investigation=True,
        reasoning="investigate",
    )
    return {
        "triage_agent": StubAgent(triage_result),
        "planner_agent": PlannerAgent(),
        "evidence_agent": StubAgent(EvidenceOutput(collection_status=CollectionStatus.COMPLETED)),
        "risk_agent": StubAgent(
            RiskAssessment(
                risk_score=80,
                severity=Severity.HIGH,
                confidence=0.9,
                scoring_mode=ScoringMode.RULE_ONLY,
            )
        ),
        "report_agent": StubAgent(SimpleNamespace(report_id="rpt-stub")),
    }


def _services(
    state_machine: FakeStateMachine | None = None,
    *,
    runtime: FakeRuntime | None = None,
) -> dict[str, Any]:
    return {
        "state_machine": state_machine or FakeStateMachine(),
        "event_service": FakeEventService(),
        "workflow_runtime": runtime or FakeRuntime(),
        "degraded_flags": FakeDegradedFlags(),
        "context_store": FakeContextStore(),
    }


class TestRouteAfterTriage:
    def test_not_required_no_investigation_closes(self) -> None:
        assert route_after_triage(_base_state(need_investigation=False)) == ROUTE_CLOSE

    def test_not_required_fp_closes(self) -> None:
        state = _base_state(
            false_positive_match={"recommendation": "close_as_fp", "max_score": 0.95}
        )
        assert route_after_triage(state) == ROUTE_CLOSE

    def test_required_fp_ready_uses_disposition_only(self) -> None:
        state = _base_state(
            disposition_policy=DispositionPolicy.REQUIRED.value,
            false_positive_match={"recommendation": "close_as_fp", "max_score": 0.95},
            event_status_update_readiness=WritebackReadiness.READY.value,
        )
        assert route_after_triage(state) == ROUTE_DISPOSITION_ONLY

    def test_required_fp_not_ready_holds(self) -> None:
        state = _base_state(
            disposition_policy=DispositionPolicy.REQUIRED.value,
            false_positive_match={"recommendation": "close_as_fp"},
            event_status_update_readiness=WritebackReadiness.CAPABILITY_UNKNOWN.value,
        )
        assert route_after_triage(state) == ROUTE_MANUAL_HOLD

    @pytest.mark.parametrize(
        "state",
        [
            _base_state(need_investigation=True),
            _base_state(
                disposition_policy=DispositionPolicy.REQUIRED.value,
                need_investigation=False,
            ),
        ],
    )
    def test_other_paths_investigate(self, state: InvestigationState) -> None:
        assert route_after_triage(state) == ROUTE_INVESTIGATE


def test_remaining_route_truth_tables() -> None:
    assert route_after_planner(_base_state(disposition_only_intent=True)) == ROUTE_RESPONSE
    assert route_after_planner(_base_state()) == ROUTE_EVIDENCE
    assert route_after_risk(_base_state()) == ROUTE_RESPONSE
    assert (
        route_after_approval(
            _base_state(execution_substate=ExecutionSubstate.WAITING_APPROVAL.value)
        )
        == ROUTE_WAIT
    )
    assert route_after_approval(_base_state()) == ROUTE_EXECUTE
    assert route_after_verify(_base_state(verify_need_manual_resolution=True)) == ROUTE_MANUAL
    assert route_after_verify(_base_state(verify_need_writeback_recovery=True)) == ROUTE_WRITEBACK
    assert route_after_verify(_base_state(verify_need_action_replan=True)) == ROUTE_REPLAN
    assert route_after_verify(_base_state(disposition_only_intent=True)) == ROUTE_HALT
    assert (
        route_after_verify(_base_state(disposition_policy=DispositionPolicy.REQUIRED.value))
        == ROUTE_HALT
    )
    assert route_after_verify(_base_state()) == ROUTE_REPORT


@pytest.mark.asyncio
async def test_graph_compiles_and_golden_path_order() -> None:
    """not_required full investigation path runs P0 sequence and closes."""
    machine = FakeStateMachine()
    services = _services(machine)
    graph = build_investigation_graph(_agents(), services)
    assert NODE_RAG not in graph.get_graph().nodes

    final = await graph.ainvoke(
        _base_state(),
        {"configurable": {"thread_id": "evt-graph-001"}},
    )
    trace = final["node_trace"]
    assert tuple(trace) == P0_NODE_SEQUENCE
    assert machine.status is EventStatus.CLOSED
    assert services["event_service"].verdicts == []


@pytest.mark.asyncio
async def test_optional_rag_is_between_evidence_and_risk() -> None:
    agents = _agents()
    agents["rag_agent"] = StubAgent(None)
    graph = build_investigation_graph(agents, _services())
    graph_view = graph.get_graph()
    assert NODE_RAG in graph_view.nodes
    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert ("evidence_node", NODE_RAG) in edges
    assert (NODE_RAG, NODE_RISK) in edges


@pytest.mark.asyncio
async def test_not_required_short_circuit_generates_report_and_closes() -> None:
    triage = TriageResult(
        event_type=EventType.OTHER,
        severity=Severity.LOW,
        need_investigation=False,
        reasoning="no investigation",
    )
    agents = _agents(triage=triage)
    services = _services()
    final = await build_investigation_graph(agents, services).ainvoke(
        _base_state(need_investigation=False, severity=Severity.LOW.value),
        {"configurable": {"thread_id": "evt-short"}},
    )
    assert final["node_trace"] == ["triage_node", NODE_CLOSE]
    assert final["report_generated"] is True
    assert services["event_service"].verdicts == [FinalVerdict.FALSE_POSITIVE]


@pytest.mark.asyncio
async def test_required_threat_never_enters_disposition_only() -> None:
    runtime = FakeRuntime(WritebackReadiness.READY)
    final = await build_investigation_graph(
        _agents(),
        _services(runtime=runtime),
    ).ainvoke(
        _base_state(
            disposition_policy=DispositionPolicy.REQUIRED.value,
            event_status_update_readiness=WritebackReadiness.READY.value,
        ),
        {"configurable": {"thread_id": "evt-threat"}},
    )
    assert NODE_BEGIN_DISPOSITION_ONLY not in final["node_trace"]
    assert NODE_CLOSE not in final["node_trace"]
    assert NODE_HALT in final["node_trace"]
    assert final["event_status"] == EventStatus.VERIFYING.value
    assert runtime.begun == []


@pytest.mark.asyncio
async def test_required_golden_path_order_halts_at_verify() -> None:
    """P0 main-chain order through verify, then HALT before report/close."""
    final = await build_investigation_graph(
        _agents(),
        _services(runtime=FakeRuntime(WritebackReadiness.READY)),
    ).ainvoke(
        _base_state(
            disposition_policy=DispositionPolicy.REQUIRED.value,
            event_status_update_readiness=WritebackReadiness.READY.value,
        ),
        {"configurable": {"thread_id": "evt-required-golden"}},
    )
    expected_trace = (*P0_NODE_SEQUENCE[:8], NODE_HALT)
    assert tuple(final["node_trace"]) == expected_trace
    assert NODE_CLOSE not in final["node_trace"]
    assert final["event_status"] == EventStatus.VERIFYING.value
    assert final["halted"] is True


@pytest.mark.asyncio
async def test_required_fp_ready_is_deterministic_and_halts_before_close() -> None:
    first_runtime = FakeRuntime(WritebackReadiness.READY)
    second_runtime = FakeRuntime(WritebackReadiness.READY)
    first_graph = build_investigation_graph(
        _agents(),
        _services(runtime=first_runtime),
    )
    second_graph = build_investigation_graph(
        _agents(),
        _services(runtime=second_runtime),
    )
    initial = _base_state(
        disposition_policy=DispositionPolicy.REQUIRED.value,
        false_positive_match={"recommendation": "close_as_fp", "max_score": 0.92},
        event_status_update_readiness=WritebackReadiness.READY.value,
    )
    first = await first_graph.ainvoke(
        initial,
        {"configurable": {"thread_id": "evt-fp-a"}},
    )
    second = await second_graph.ainvoke(
        initial,
        {"configurable": {"thread_id": "evt-fp-b"}},
    )

    assert first["execution_plan"] == second["execution_plan"]
    assert first["execution_plan"]["revision"] == 0
    assert len(first["execution_plan"]["steps"]) == 1
    assert first["execution_plan"]["steps"][0]["assigned_agent"] == "response_agent"
    assert NODE_RESPONSE in first["node_trace"]
    assert NODE_HALT in first["node_trace"]
    assert NODE_CLOSE not in first["node_trace"]
    assert first["event_status"] == EventStatus.PLANNING_RESPONSE.value
    assert first["node_trace"][-2:] == [NODE_RESPONSE, NODE_HALT]
    assert first["halted"] is True


@pytest.mark.asyncio
async def test_blocked_fp_sets_degraded_flag_without_illegal_substate() -> None:
    degraded = FakeDegradedFlags()
    services = _services(runtime=FakeRuntime(WritebackReadiness.CAPABILITY_UNSUPPORTED))
    services["degraded_flags"] = degraded
    final = await build_investigation_graph(_agents(), services).ainvoke(
        _base_state(
            disposition_policy=DispositionPolicy.REQUIRED.value,
            false_positive_match={"recommendation": "close_as_fp"},
            event_status_update_readiness=WritebackReadiness.CAPABILITY_UNSUPPORTED.value,
        ),
        {"configurable": {"thread_id": "evt-blocked"}},
    )
    assert NODE_MANUAL_HOLD in final["node_trace"]
    assert final["execution_substate"] == ExecutionSubstate.NONE.value
    assert final["degraded_flags"] == ["disposition_writeback_blocked=capability_unsupported"]
    assert degraded.calls[0][-1] == "DegradedFlagService"


@pytest.mark.asyncio
async def test_forged_disposition_only_intent_is_rejected() -> None:
    runtime = FakeRuntime()
    graph = build_investigation_graph(_agents(), _services(runtime=runtime))
    with pytest.raises(InvalidStateTransitionError):
        await graph.ainvoke(
            _base_state(disposition_only_intent=True),
            {"configurable": {"thread_id": "evt-forged"}},
        )


@pytest.mark.asyncio
async def test_graph_error_marks_event_failed_and_keeps_reason() -> None:
    class FailingAgent(StubAgent):
        async def execute(self, input: Any) -> Any:
            raise RuntimeError("triage boom")

    agents = _agents()
    agents["triage_agent"] = FailingAgent(None)
    machine = FakeStateMachine()
    graph = build_investigation_graph(agents, _services(machine))
    with pytest.raises(RuntimeError, match="triage boom"):
        await graph.ainvoke(
            _base_state(),
            {"configurable": {"thread_id": "evt-failed"}},
        )
    assert machine.status is EventStatus.FAILED
    assert "triage boom" in (machine.transitions[-1][2] or "")


@pytest.mark.asyncio
async def test_checkpoint_persists_with_ttl_and_resumes_in_new_saver() -> None:
    redis = FakeRedisClient()
    first_saver = await RedisCheckpointer.create(redis)  # type: ignore[arg-type]
    first_graph = build_investigation_graph(
        _agents(),
        _services(),
        checkpointer=first_saver,
        interrupt_before=[NODE_RISK],
    )
    config = {"configurable": {"thread_id": "evt-resume"}}
    await first_graph.ainvoke(_base_state(event_id="evt-resume"), config)

    key = checkpoint_key_for_event("evt-resume")
    assert key in redis.store.values
    assert redis.store.ttls[key] == CHECKPOINT_TTL_SECONDS
    assert first_saver.recoverable is True

    second_saver = await RedisCheckpointer.create(redis)  # type: ignore[arg-type]
    second_graph = build_investigation_graph(
        _agents(),
        _services(
            FakeStateMachine(
                status=EventStatus.ANALYZING,
                statuses={"evt-resume": EventStatus.ANALYZING},
            )
        ),
        checkpointer=second_saver,
    )
    final = await second_graph.ainvoke(None, config)
    assert NODE_CLOSE in final["node_trace"]
    assert final["node_trace"].count(NODE_RISK) == 1


@pytest.mark.asyncio
async def test_redis_unavailable_uses_nonrecoverable_memory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedisClient(available=False)
    warnings: list[str] = []

    def _capture_warning(message: str, *args: object, **kwargs: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(
        "app.orchestration.checkpointer.logger.warning",
        _capture_warning,
    )
    saver = await RedisCheckpointer.create(redis)  # type: ignore[arg-type]
    assert saver.memory_fallback is True
    assert saver.recoverable is False
    assert any("process restart cannot recover" in message for message in warnings)


@pytest.mark.asyncio
async def test_sync_checkpoint_api_explicitly_downgrades_recoverability() -> None:
    saver = await RedisCheckpointer.create(FakeRedisClient())  # type: ignore[arg-type]
    assert saver.recoverable is True

    assert saver.get_tuple({"configurable": {"thread_id": "evt-sync"}}) is None
    assert saver.recoverable is False


@pytest.mark.parametrize(
    "service_name",
    ["state_machine", "degraded_flags"],
)
def test_required_workflow_services_reject_none(service_name: str) -> None:
    services = _services()
    services[service_name] = None

    with pytest.raises(ValueError, match=service_name):
        build_investigation_graph(_agents(), services)
