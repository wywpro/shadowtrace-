"""Workflow graph nodes (ISSUE-048/ISSUE-049).

This module provides the ``planner_node`` function that replaces the ISSUE-048
placeholder. It wraps ``PlannerAgent`` and is designed to be used as a
LangGraph node once the full StateGraph is assembled in ISSUE-054.
"""

from __future__ import annotations

import logging

from app.agents.planner_agent import PlannerAgent
from app.models.agent_io import ExecutionPlan, PlannerAgentInput, TriageResult
from app.models.context import EventContext
from app.models.enums import EventType, Severity

logger = logging.getLogger(__name__)


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


__all__ = ["planner_node"]
