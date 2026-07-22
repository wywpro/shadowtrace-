"""planner_node tests (ISSUE-049)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.planner_agent import PlannerAgent
from app.core.llm.base import InMemoryLLMCallAuditRecorder
from app.core.llm.mock_client import MockLLMClient
from app.models.agent_io import ExecutionPlan, PlanBudget, PlanStep, TriageResult
from app.models.context import EventContext
from app.models.enums import EventType, Severity
from app.models.security_event import EventSummary
from app.orchestration.workflow_graph import planner_node


def _make_triage(event_type: EventType = EventType.DATA_EXFILTRATION) -> TriageResult:
    return TriageResult(
        event_type=event_type,
        severity=Severity.HIGH,
        need_investigation=True,
        reasoning="Test triage",
    )


def _make_event_summary(event_id: str) -> EventSummary:
    from app.models.enums import (
        DispositionPolicy,
    )
    from app.models.enums import (
        EventStatus as ES,
    )
    from app.models.enums import (
        FinalVerdict as FV,
    )
    from app.models.enums import (
        WritebackReadiness as WR,
    )

    return EventSummary(
        event_id=event_id,
        event_type=EventType.DATA_EXFILTRATION,
        title="Test event",
        status=ES.TRIAGING,
        severity=Severity.HIGH,
        risk_score=50,
        final_verdict=FV.NONE,
        writeback_required=False,
        writeback_readiness=WR.NOT_REQUIRED,
        disposition_policy=DispositionPolicy.NOT_REQUIRED,
    )


def _make_context(event_id: str, triage: TriageResult | None = None) -> EventContext:
    return EventContext(
        event=_make_event_summary(event_id),
        triage_result=triage.model_dump(mode="json") if triage else None,
    )


def _make_mock_llm(tmp_path: Path) -> MockLLMClient:
    return MockLLMClient(
        golden_root=tmp_path,
        audit_recorder=InMemoryLLMCallAuditRecorder(),
    )


# --------------------------------------------------------------------------- #
# planner_node — normal path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_planner_node_normal_path(tmp_path: Path) -> None:
    """Normal path: triage_result present -> generates plan via LLM."""
    golden_dir = tmp_path / "plan_generate"
    golden_dir.mkdir()
    golden = {
        "content": {
            "plan_id": "pln-node0001",
            "event_id": "evt-node-01",
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
    }
    (golden_dir / "default.json").write_text(json.dumps(golden), encoding="utf-8")

    llm = _make_mock_llm(tmp_path)
    planner = PlannerAgent(llm_client=llm)
    ctx = _make_context("evt-node-01", _make_triage())

    plan = await planner_node(ctx, planner)
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) >= 4
    assert plan.revision == 0


# --------------------------------------------------------------------------- #
# planner_node — disposition_only
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_planner_node_disposition_only() -> None:
    """disposition_only=True -> deterministic single-step plan."""
    planner = PlannerAgent()
    ctx = _make_context("evt-disp-node", _make_triage())

    plan = await planner_node(ctx, planner, disposition_only=True)
    assert len(plan.steps) == 1
    assert plan.steps[0].assigned_agent == "response_agent"
    assert plan.revision == 0
    assert plan.degraded is False


# --------------------------------------------------------------------------- #
# planner_node — missing triage falls back
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_planner_node_no_triage_uses_default_plan() -> None:
    """No triage_result in input -> conservative DEFAULT_PLANS path."""
    planner = PlannerAgent()
    ctx = EventContext(
        event=_make_event_summary("evt-no-triage-node"),
        triage_result=None,
    )

    plan = await planner_node(ctx, planner)
    assert len(plan.steps) >= 4
    assert plan.degraded is True
    assert plan.steps[0].assigned_agent == "evidence_agent"


# --------------------------------------------------------------------------- #
# planner_node — replan
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_planner_node_replan_triggers_revise(tmp_path: Path) -> None:
    """replan_count > 0 triggers revision via PlanAgent.revise."""
    golden_dir = tmp_path / "plan_revise"
    golden_dir.mkdir()
    golden = {
        "content": {
            "plan_id": "pln-exist001",
            "event_id": "evt-replan-node",
            "steps": [
                {
                    "step_order": 1,
                    "step_goal": "[revised] New approach",
                    "assigned_agent": "evidence_agent",
                    "required_tools": ["query_threat_intel", "query_dns"],
                    "success_criteria": "better data",
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
            "revision": 1,
            "revise_reason": "replan triggered (count=1)",
            "degraded": False,
        },
        "model_name": "mock-model",
        "prompt_tokens": 60,
        "completion_tokens": 120,
        "total_tokens": 180,
    }
    (golden_dir / "default.json").write_text(json.dumps(golden), encoding="utf-8")

    llm = _make_mock_llm(tmp_path)
    planner = PlannerAgent(llm_client=llm)

    previous_plan = ExecutionPlan(
        plan_id="pln-exist001",
        event_id="evt-replan-node",
        steps=[
            PlanStep(
                step_order=1,
                step_goal="Original step",
                assigned_agent="evidence_agent",
                required_tools=["query_threat_intel"],
                success_criteria="ok",
            ),
        ],
        budget=PlanBudget(),
        revision=0,
    )

    ctx = EventContext(
        event=_make_event_summary("evt-replan-node"),
        triage_result=_make_triage().model_dump(mode="json"),
        execution_plan=previous_plan.model_dump(mode="json"),
        replan_count=1,
    )

    plan = await planner_node(ctx, planner)
    assert plan.revision == 1
    assert plan.revise_reason is not None


# --------------------------------------------------------------------------- #
# Should-Fix #5: corrupt triage falls back to normal plan (NOT disposition-only)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_planner_node_corrupt_triage_uses_default_plan(tmp_path: Path) -> None:
    """Corrupt triage → synthesised fallback triage → normal plan path.

    Must NOT silently switch to disposition-only, which would skip
    evidence/risk/response and risk incorrect event closure.
    """
    # No golden → LLM fails → falls back to DEFAULT_PLANS (degraded=True)
    llm = _make_mock_llm(tmp_path)
    planner = PlannerAgent(llm_client=llm)

    ctx = EventContext(
        event=_make_event_summary("evt-corrupt-triage"),
        triage_result={"invalid": "garbage", "not_even": 123},
    )

    plan = await planner_node(ctx, planner)
    # Must be a full plan from DEFAULT_PLANS (via _plan_impl), not disposition-only
    assert len(plan.steps) >= 4
    assert plan.degraded is True
    # Must not be a single-step disposition-only plan
    assert not (len(plan.steps) == 1 and plan.steps[0].assigned_agent == "response_agent")
