"""PlannerAgent tests (ISSUE-049)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.planner_agent import (
    PlannerAgent,
    _generate_disposition_only_plan_id,
    _generate_plan_id,
    _validate_execution_plan,
)
from app.core.llm.base import InMemoryLLMCallAuditRecorder
from app.core.llm.mock_client import MockLLMClient
from app.models.agent_io import ExecutionPlan, PlanBudget, PlannerAgentInput, PlanStep, TriageResult
from app.models.enums import EventType, Severity

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_mock_llm(tmp_path: Path) -> MockLLMClient:
    """Create a MockLLMClient with the required audit_recorder."""
    return MockLLMClient(
        golden_root=tmp_path,
        audit_recorder=InMemoryLLMCallAuditRecorder(),
    )


def _make_triage(
    event_type: EventType = EventType.DATA_EXFILTRATION,
) -> TriageResult:
    return TriageResult(
        event_type=event_type,
        severity=Severity.HIGH,
        need_investigation=True,
        reasoning="Data exfiltration detected from file server",
    )


def _make_previous_plan(event_id: str) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="pln-prev0001",
        event_id=event_id,
        steps=[
            PlanStep(
                step_order=1,
                step_goal="Collect threat intel",
                assigned_agent="evidence_agent",
                required_tools=["query_threat_intel"],
                success_criteria="at least 1 hit",
            ),
            PlanStep(
                step_order=2,
                step_goal="Risk assessment",
                assigned_agent="risk_agent",
                required_tools=[],
                success_criteria="score computed",
            ),
        ],
        budget=PlanBudget(),
        revision=0,
    )


def _write_golden(
    tmp_path: Path,
    prompt_key: str,
    plan_data: dict,
) -> None:
    """Write a golden response JSON for a given prompt_key."""
    golden_dir = tmp_path / prompt_key
    golden_dir.mkdir(parents=True, exist_ok=True)
    (golden_dir / "default.json").write_text(json.dumps(plan_data), encoding="utf-8")


# --------------------------------------------------------------------------- #
# plan_id generation
# --------------------------------------------------------------------------- #


def test_generate_plan_id_is_stable() -> None:
    a = _generate_plan_id("evt-test", 0)
    b = _generate_plan_id("evt-test", 0)
    assert a == b
    assert a.startswith("pln-")
    assert len(a) == 12  # "pln-" + 8 hex


def test_generate_plan_id_differs_by_revision() -> None:
    a = _generate_plan_id("evt-test", 0)
    b = _generate_plan_id("evt-test", 1)
    assert a != b


def test_disposition_only_plan_id_stable() -> None:
    a = _generate_disposition_only_plan_id("evt-disp")
    b = _generate_disposition_only_plan_id("evt-disp")
    assert a == b
    assert a.startswith("pln-")


# --------------------------------------------------------------------------- #
# plan_disposition_only
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_plan_disposition_only_single_step() -> None:
    agent = PlannerAgent()
    ctx = MagicMock()
    ctx.event.event_id = "evt-disp-01"
    plan = await agent.plan_disposition_only(ctx)
    assert plan.revision == 0
    assert plan.degraded is False
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.assigned_agent == "response_agent"
    assert step.step_order == 1
    assert "update_source_event_disposition" in step.required_tools


@pytest.mark.asyncio
async def test_plan_disposition_only_stable() -> None:
    agent = PlannerAgent()
    ctx = MagicMock()
    ctx.event.event_id = "evt-disp-stable"
    plan1 = await agent.plan_disposition_only(ctx)
    plan2 = await agent.plan_disposition_only(ctx)
    assert plan1.plan_id == plan2.plan_id
    assert plan1.steps == plan2.steps


# --------------------------------------------------------------------------- #
# _run — disposition-only path (no triage_result in input)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_disposition_only_when_no_triage() -> None:
    agent = PlannerAgent()
    inp = PlannerAgentInput(event_id="evt-no-triage")
    plan = await agent._run(inp)
    assert plan.revision == 0
    assert len(plan.steps) == 1
    assert plan.steps[0].assigned_agent == "response_agent"


# --------------------------------------------------------------------------- #
# _run — normal plan with LLM (MockLLMClient)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_plan_with_mock_llm(tmp_path: Path) -> None:
    """Main scenario: LLM produces a plan with >=4 steps and valid agents/tools."""
    _write_golden(
        tmp_path,
        "plan_generate",
        {
            "content": {
                "plan_id": "pln-deadbeef",
                "event_id": "evt-test-001",
                "steps": [
                    {
                        "step_order": 1,
                        "step_goal": "Query threat intel",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_threat_intel", "query_dns"],
                        "success_criteria": "IOC data collected",
                    },
                    {
                        "step_order": 2,
                        "step_goal": "Collect process evidence",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_process_tree"],
                        "success_criteria": "process tree obtained",
                    },
                    {
                        "step_order": 3,
                        "step_goal": "Network evidence",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_network_connections"],
                        "success_criteria": "connections logged",
                    },
                    {
                        "step_order": 4,
                        "step_goal": "Risk scoring",
                        "assigned_agent": "risk_agent",
                        "required_tools": [],
                        "success_criteria": "score computed",
                    },
                    {
                        "step_order": 5,
                        "step_goal": "Response plan",
                        "assigned_agent": "response_agent",
                        "required_tools": [],
                        "success_criteria": "actions generated",
                    },
                ],
                "budget": {"max_tool_calls": 30, "max_llm_calls": 20, "max_duration_s": 300},
                "revision": 0,
                "revise_reason": None,
                "degraded": False,
            },
            "model_name": "mock-model",
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
        },
    )

    llm = _make_mock_llm(tmp_path)
    agent = PlannerAgent(llm_client=llm)
    inp = PlannerAgentInput(
        event_id="evt-test-001",
        triage_result=_make_triage(),
    )
    plan = await agent._run(inp)

    # Acceptance criteria 1: >= 4 steps, all agents/tools valid
    assert len(plan.steps) >= 4
    for step in plan.steps:
        assert step.assigned_agent in {
            "evidence_agent",
            "risk_agent",
            "response_agent",
            "graph_agent",
            "rag_agent",
            "verify_agent",
            "report_agent",
            "triage_agent",
            "planner_agent",
            "memory_agent",
            "tool_agent",
            "super_agent",
        }
    assert plan.revision == 0
    assert plan.degraded is False


# --------------------------------------------------------------------------- #
# _run — revise with LLM
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_revise_with_mock_llm(tmp_path: Path) -> None:
    """Revise scenario: revision increments, revise_reason present, steps differ."""
    _write_golden(
        tmp_path,
        "plan_revise",
        {
            "content": {
                "plan_id": "pln-prev0001",
                "event_id": "evt-test-revise",
                "steps": [
                    {
                        "step_order": 1,
                        "step_goal": "[revised] Query threat intel expanded",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_threat_intel", "query_dns", "query_whois"],
                        "success_criteria": "expanded IOC data",
                    },
                    {
                        "step_order": 2,
                        "step_goal": "[revised] Process + network evidence",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_process_tree", "query_network_connections"],
                        "success_criteria": "full endpoint data",
                    },
                    {
                        "step_order": 3,
                        "step_goal": "Risk scoring",
                        "assigned_agent": "risk_agent",
                        "required_tools": [],
                        "success_criteria": "score computed",
                    },
                    {
                        "step_order": 4,
                        "step_goal": "Response plan",
                        "assigned_agent": "response_agent",
                        "required_tools": [],
                        "success_criteria": "actions generated",
                    },
                ],
                "budget": {"max_tool_calls": 30, "max_llm_calls": 20, "max_duration_s": 300},
                "revision": 1,
                "revise_reason": "Insufficient threat intel in previous round",
                "degraded": False,
            },
            "model_name": "mock-model",
            "prompt_tokens": 150,
            "completion_tokens": 250,
            "total_tokens": 400,
        },
    )

    llm = _make_mock_llm(tmp_path)
    agent = PlannerAgent(llm_client=llm)
    previous = _make_previous_plan("evt-test-revise")
    inp = PlannerAgentInput(
        event_id="evt-test-revise",
        triage_result=_make_triage(),
        previous_plan=previous,
        revise_reason="Insufficient threat intel in previous round",
    )
    plan = await agent._run(inp)

    # Acceptance criteria 2: revision=1, revise_reason present, steps differ
    assert plan.revision == 1
    assert plan.revise_reason is not None
    assert plan.revise_reason == "Insufficient threat intel in previous round"
    # Steps should NOT be identical to previous plan
    prev_goals = {s.step_goal for s in previous.steps}
    new_goals = {s.step_goal for s in plan.steps}
    assert prev_goals != new_goals, "revised plan should have different steps"


# --------------------------------------------------------------------------- #
# LLM failure → DEFAULT_PLANS fallback (acceptance criterion 3)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_default_plans(tmp_path: Path) -> None:
    """When the LLM fails, use DEFAULT_PLANS with degraded=True."""
    # Golden root with NO plan_generate directory → LLMProviderError
    llm = _make_mock_llm(tmp_path)
    agent = PlannerAgent(llm_client=llm)
    inp = PlannerAgentInput(
        event_id="evt-test-fallback",
        triage_result=_make_triage(EventType.DATA_EXFILTRATION),
    )
    plan = await agent._run(inp)

    # Acceptance criterion 3: degraded=True and uses DEFAULT_PLANS
    assert plan.degraded is True
    assert len(plan.steps) >= 4  # DATA_EXFILTRATION default has 7 steps
    assert plan.revision == 0


@pytest.mark.asyncio
async def test_llm_none_uses_default_plans() -> None:
    """When no LLM client is set, use DEFAULT_PLANS directly."""
    agent = PlannerAgent(llm_client=None)
    inp = PlannerAgentInput(
        event_id="evt-test-no-llm",
        triage_result=_make_triage(EventType.ACCOUNT_ANOMALY),
    )
    plan = await agent._run(inp)
    assert plan.degraded is True
    assert len(plan.steps) >= 3


# --------------------------------------------------------------------------- #
# Invalid tool stripping
# --------------------------------------------------------------------------- #


def test_validate_plan_strips_invalid_tools() -> None:
    """Tools not in the canonical list for an agent are stripped with warning."""
    plan = ExecutionPlan(
        plan_id="pln-test0001",
        event_id="evt-test-strip",
        steps=[
            PlanStep(
                step_order=1,
                step_goal="Test step",
                assigned_agent="evidence_agent",
                required_tools=["query_threat_intel", "invalid_tool_xyz", "query_dns"],
                success_criteria="ok",
            ),
        ],
        budget=PlanBudget(),
        revision=0,
    )
    clean = _validate_execution_plan(plan)
    step = clean.steps[0]
    assert "query_threat_intel" in step.required_tools
    assert "query_dns" in step.required_tools
    assert "invalid_tool_xyz" not in step.required_tools


# --------------------------------------------------------------------------- #
# All 8 EventType default plans exist
# --------------------------------------------------------------------------- #


def test_all_event_types_have_default_plans() -> None:
    """Every EventType must have a corresponding DEFAULT_PLANS entry."""
    from app.agents.rules.default_plans import get_default_plan

    for ev_type in EventType:
        plan = get_default_plan("evt-test-alltypes", ev_type, f"pln-{ev_type.value[:8]}")
        assert isinstance(plan, ExecutionPlan)
        assert plan.degraded is True
        assert len(plan.steps) >= 2
        # Every step must be valid
        for step in plan.steps:
            assert step.assigned_agent in {
                "evidence_agent",
                "risk_agent",
                "response_agent",
                "graph_agent",
                "rag_agent",
            }
            assert step.step_order >= 1


# --------------------------------------------------------------------------- #
# BaseAgent.execute integration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_normal_plan(tmp_path: Path) -> None:
    """Full BaseAgent.execute path with MockLLM."""
    _write_golden(
        tmp_path,
        "plan_generate",
        {
            "content": {
                "plan_id": "pln-exec0001",
                "event_id": "evt-exec-01",
                "steps": [
                    {
                        "step_order": 1,
                        "step_goal": "Evidence collection",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_threat_intel"],
                        "success_criteria": "data collected",
                    },
                    {
                        "step_order": 2,
                        "step_goal": "Risk scoring",
                        "assigned_agent": "risk_agent",
                        "required_tools": [],
                        "success_criteria": "score done",
                    },
                    {
                        "step_order": 3,
                        "step_goal": "Response generation",
                        "assigned_agent": "response_agent",
                        "required_tools": [],
                        "success_criteria": "actions ready",
                    },
                    {
                        "step_order": 4,
                        "step_goal": "Report draft",
                        "assigned_agent": "report_agent",
                        "required_tools": [],
                        "success_criteria": "report ready",
                    },
                ],
                "budget": {"max_tool_calls": 30, "max_llm_calls": 20, "max_duration_s": 300},
                "revision": 0,
                "revise_reason": None,
                "degraded": False,
            },
            "model_name": "mock-model",
            "prompt_tokens": 50,
            "completion_tokens": 150,
            "total_tokens": 200,
        },
    )

    llm = _make_mock_llm(tmp_path)
    agent = PlannerAgent(llm_client=llm)
    inp = PlannerAgentInput(
        event_id="evt-exec-01",
        triage_result=_make_triage(),
    )
    plan = await agent.execute(inp)
    assert len(plan.steps) >= 4
    assert plan.plan_id.startswith("pln-")


@pytest.mark.asyncio
async def test_execute_revise(tmp_path: Path) -> None:
    """Full BaseAgent.execute path for revision."""
    _write_golden(
        tmp_path,
        "plan_revise",
        {
            "content": {
                "plan_id": "pln-prev0001",
                "event_id": "evt-exec-revise",
                "steps": [
                    {
                        "step_order": 1,
                        "step_goal": "[revised] Expanded evidence",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_threat_intel", "query_dns"],
                        "success_criteria": "expanded data",
                    },
                    {
                        "step_order": 2,
                        "step_goal": "Risk scoring revised",
                        "assigned_agent": "risk_agent",
                        "required_tools": [],
                        "success_criteria": "updated score",
                    },
                    {
                        "step_order": 3,
                        "step_goal": "Response generation",
                        "assigned_agent": "response_agent",
                        "required_tools": [],
                        "success_criteria": "new actions",
                    },
                    {
                        "step_order": 4,
                        "step_goal": "Final report",
                        "assigned_agent": "report_agent",
                        "required_tools": [],
                        "success_criteria": "final report",
                    },
                ],
                "budget": {"max_tool_calls": 30, "max_llm_calls": 20, "max_duration_s": 300},
                "revision": 1,
                "revise_reason": "First round insufficient",
                "degraded": False,
            },
            "model_name": "mock-model",
            "prompt_tokens": 60,
            "completion_tokens": 160,
            "total_tokens": 220,
        },
    )

    llm = _make_mock_llm(tmp_path)
    agent = PlannerAgent(llm_client=llm)
    previous = _make_previous_plan("evt-exec-revise")
    inp = PlannerAgentInput(
        event_id="evt-exec-revise",
        triage_result=_make_triage(),
        previous_plan=previous,
        revise_reason="First round insufficient",
    )
    plan = await agent.execute(inp)
    assert plan.revision == 1
    assert plan.revise_reason == "First round insufficient"
    assert plan.degraded is False  # LLM succeeded, no persist failure


# --------------------------------------------------------------------------- #
# Idempotency — replay does not call LLM twice
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_plan_idempotent_via_working_memory(tmp_path: Path) -> None:
    """When a plan already exists in working memory, reuse it (no LLM call)."""
    from app.services.working_memory import BoundWorkingMemory

    # Create a MockLLM that would fail if called (no golden for plan_generate)
    llm = _make_mock_llm(tmp_path)

    # Mock working memory that returns an existing plan
    wm = MagicMock(spec=BoundWorkingMemory)
    existing_plan = ExecutionPlan(
        plan_id="pln-cached001",
        event_id="evt-idempotent",
        steps=[
            PlanStep(
                step_order=1,
                step_goal="cached step",
                assigned_agent="evidence_agent",
                required_tools=["query_threat_intel"],
                success_criteria="ok",
            ),
        ],
        budget=PlanBudget(),
        revision=0,
    )
    wm.read = AsyncMock(return_value=existing_plan.model_dump(mode="json"))

    agent = PlannerAgent(llm_client=llm, working_memory=wm)
    inp = PlannerAgentInput(
        event_id="evt-idempotent",
        triage_result=_make_triage(),
    )
    plan = await agent._run(inp)

    # Should return the cached plan
    assert plan.plan_id == "pln-cached001"
    assert plan.revision == 0
    # LLM should NOT have been called (would fail without golden)


# --------------------------------------------------------------------------- #
# Blocker fix: plan() public method with missing triage
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_plan_public_api_without_triage() -> None:
    """plan() should fall back to disposition-only when triage is missing."""
    from app.services.working_memory import BoundWorkingMemory

    wm = MagicMock(spec=BoundWorkingMemory)
    # working_memory returns None for triage_result
    wm.read = AsyncMock(return_value=None)

    agent = PlannerAgent(working_memory=wm)
    ctx = MagicMock()
    ctx.event.event_id = "evt-plan-no-triage"

    plan = await agent.plan(ctx)
    # Must not crash with AttributeError; should return a valid plan
    assert plan.event_id == "evt-plan-no-triage"
    assert len(plan.steps) == 1
    assert plan.steps[0].assigned_agent == "response_agent"


# --------------------------------------------------------------------------- #
# Should-Fix #2: persist failure marks plan degraded
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_persist_failure_marks_degraded(tmp_path: Path) -> None:
    """When working_memory.write fails, the returned plan must be degraded."""
    from app.services.working_memory import BoundWorkingMemory

    _write_golden(
        tmp_path,
        "plan_generate",
        {
            "content": {
                "plan_id": "pln-pfail001",
                "event_id": "evt-persist-fail",
                "steps": [
                    {
                        "step_order": 1,
                        "step_goal": "Evidence",
                        "assigned_agent": "evidence_agent",
                        "required_tools": ["query_threat_intel"],
                        "success_criteria": "ok",
                    },
                    {
                        "step_order": 2,
                        "step_goal": "Risk",
                        "assigned_agent": "risk_agent",
                        "required_tools": [],
                        "success_criteria": "ok",
                    },
                    {
                        "step_order": 3,
                        "step_goal": "Response",
                        "assigned_agent": "response_agent",
                        "required_tools": [],
                        "success_criteria": "ok",
                    },
                    {
                        "step_order": 4,
                        "step_goal": "Report",
                        "assigned_agent": "report_agent",
                        "required_tools": [],
                        "success_criteria": "ok",
                    },
                ],
                "budget": {"max_tool_calls": 30, "max_llm_calls": 20, "max_duration_s": 300},
                "revision": 0,
                "revise_reason": None,
                "degraded": False,
            },
            "model_name": "mock-model",
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
        },
    )

    llm = _make_mock_llm(tmp_path)
    wm = MagicMock(spec=BoundWorkingMemory)
    # working_memory.write raises an exception
    wm.write = AsyncMock(side_effect=RuntimeError("write failed"))
    wm.read = AsyncMock(return_value=None)

    agent = PlannerAgent(llm_client=llm, working_memory=wm)
    inp = PlannerAgentInput(
        event_id="evt-persist-fail",
        triage_result=_make_triage(),
    )
    plan = await agent._run(inp)

    # LLM succeeded but persist failed → plan returned with degraded=True
    assert plan.plan_id == "pln-pfail001"
    assert plan.degraded is True
    assert len(plan.steps) >= 4


# --------------------------------------------------------------------------- #
# Nit: invalid agent step is dropped (not replaced)
# --------------------------------------------------------------------------- #


def test_validate_execution_plan_drops_invalid_agent_steps() -> None:
    """Steps with invalid assigned_agent are dropped, not replaced."""
    from app.agents.planner_agent import _validate_execution_plan

    # Use model_construct to bypass Pydantic Literal validation
    valid_step = PlanStep(
        step_order=1,
        step_goal="Valid step",
        assigned_agent="evidence_agent",
        required_tools=["query_threat_intel"],
        success_criteria="ok",
    )
    invalid_step = PlanStep.model_construct(
        step_order=2,
        step_goal="Invalid agent step",
        assigned_agent="invalid_agent_xxx",  # type: ignore[arg-type]
        required_tools=[],
        success_criteria="n/a",
    )
    another_valid = PlanStep(
        step_order=3,
        step_goal="Another valid step",
        assigned_agent="risk_agent",
        required_tools=[],
        success_criteria="ok",
    )

    plan = ExecutionPlan(
        plan_id="pln-drop0001",
        event_id="evt-drop-agent",
        steps=[valid_step, invalid_step, another_valid],
        budget=PlanBudget(),
        revision=0,
    )
    clean = _validate_execution_plan(plan)
    # Invalid step should be removed, only 2 valid steps remain
    assert len(clean.steps) == 2
    # step_order should be re-numbered contiguously
    assert clean.steps[0].step_order == 1
    assert clean.steps[0].assigned_agent == "evidence_agent"
    assert clean.steps[1].step_order == 2
    assert clean.steps[1].assigned_agent == "risk_agent"
