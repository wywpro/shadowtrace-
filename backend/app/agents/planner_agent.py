"""PlannerAgent — generates structured investigation plans (ISSUE-049).

Reads ``triage_result`` from EventContext (or input), produces an
``ExecutionPlan`` via LLM (JSON mode) or rule-based defaults, and writes it
back to EventContext under ``execution_plan``.

Supports three modes dispatched from ``_run``:

* **plan** — LLM-driven plan generation (prompt_key ``plan_generate``).
* **plan_disposition_only** — deterministic single-step plan for disposition-only
  workflows (no LLM).
* **revise** — LLM-driven plan revision (prompt_key ``plan_revise``), incrementing
  ``revision`` and recording ``revise_reason``.

When the LLM is unavailable the agent falls back to ``DEFAULT_PLANS`` keyed by
``EventType`` (降级策略).
"""

from __future__ import annotations

import hashlib
import logging

from app.agents.base import BaseAgent
from app.agents.prompts.planner_prompt import (
    build_plan_generate_messages,
    build_plan_revise_messages,
)
from app.agents.rules.default_plans import get_default_plan
from app.core.llm.base import LLMMessage
from app.models.agent_io import (
    AGENT_INPUT_BY_NAME as _AGENT_INPUT_BY_NAME,
)
from app.models.agent_io import (
    ExecutionPlan,
    PlanBudget,
    PlannerAgentInput,
    PlanStep,
    TriageResult,
)
from app.models.context import EventContext
from app.models.enums import EventType

logger = logging.getLogger(__name__)

# Valid agent names for assigned_agent validation
_VALID_AGENT_NAMES: frozenset[str] = frozenset(_AGENT_INPUT_BY_NAME.keys())

# Valid tool names per agent (canonical list from ISSUE-049 / planner_prompt).
# Agents NOT listed here (risk_agent, response_agent, graph_agent, etc.) have
# no PlannerAgent-level tool restrictions — their tools are validated by the
# respective agent implementations.  ``update_source_event_disposition`` is a
# disposition-only deferred Action owned by ``EventDispositionService``, never
# routed through ToolProvider; it is intentionally absent from this registry.
_VALID_TOOLS: dict[str, frozenset[str]] = {
    "evidence_agent": frozenset(
        {
            "query_threat_intel",
            "query_dns",
            "query_whois",
            "query_passive_dns",
            "query_process_tree",
            "query_network_connections",
            "query_file_events",
            "query_login_history",
            "query_account_activity",
            "query_data_access_logs",
            "query_dlp_events",
            "query_lateral_movement",
            "query_privilege_changes",
        }
    ),
    "rag_agent": frozenset(
        {
            "search_kb",
            "match_techniques",
        }
    ),
}


def _generate_plan_id(event_id: str, revision: int) -> str:
    """Generate a deterministic ``pln-{8 hex}`` id from event_id + revision."""
    digest = hashlib.sha256(f"{event_id}|{revision}".encode()).hexdigest()[:8]
    return f"pln-{digest}"


def _generate_disposition_only_plan_id(event_id: str) -> str:
    """Stable plan_id for disposition-only plans per ISSUE-049 spec."""
    raw = f"{event_id}|disposition_only|0"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"pln-{digest}"


def _build_disposition_only_plan(event_id: str) -> ExecutionPlan:
    """Deterministic single-step plan for disposition-only workflows.

    Centralised here so ``plan_disposition_only`` and the ``_run``
    disposition-only branch share one definition (Should-Fix #1).
    """
    plan_id = _generate_disposition_only_plan_id(event_id)
    step = PlanStep(
        step_order=1,
        step_goal="执行处置同步：update_source_event_disposition",
        assigned_agent="response_agent",
        required_tools=["update_source_event_disposition"],
        success_criteria="处置状态已同步至外部系统",
    )
    return ExecutionPlan(
        plan_id=plan_id,
        event_id=event_id,
        steps=[step],
        budget=PlanBudget(),
        revision=0,
        degraded=False,
    )


def _validate_plan_step(step: PlanStep) -> PlanStep | None:
    """Validate and sanitize a single PlanStep.

    - assigned_agent must be a valid AgentName → otherwise the step is
      **dropped** (returns ``None``).
    - required_tools must be valid for that agent → invalid tools stripped
      (logged).
    """
    if step.assigned_agent not in _VALID_AGENT_NAMES:
        logger.warning(
            "PlannerAgent: dropping step with invalid assigned_agent=%r",
            step.assigned_agent,
        )
        return None

    valid_tools = _VALID_TOOLS.get(step.assigned_agent)
    if valid_tools is None:
        # Agent has no PlannerAgent-level tool restrictions
        # (risk_agent, response_agent, graph_agent, …).
        return step

    clean_tools = [t for t in step.required_tools if t in valid_tools]
    invalid = set(step.required_tools) - set(clean_tools)
    if invalid:
        logger.warning(
            "PlannerAgent: stripping invalid tools %s from step %d (agent=%s)",
            sorted(invalid),
            step.step_order,
            step.assigned_agent,
        )
    # Re-numbering is not done here — caller is responsible for that.
    return PlanStep(
        step_order=step.step_order,
        step_goal=step.step_goal,
        assigned_agent=step.assigned_agent,
        required_tools=clean_tools,
        success_criteria=step.success_criteria,
    )


def _validate_execution_plan(plan: ExecutionPlan) -> ExecutionPlan:
    """Validate every step, dropping invalid ones and re-numbering survivors."""
    raw_steps = (_validate_plan_step(s) for s in plan.steps)
    clean_steps = [s for s in raw_steps if s is not None]
    # Re-number so step_order stays contiguous
    for idx, s in enumerate(clean_steps, start=1):
        if s.step_order != idx:
            clean_steps[idx - 1] = PlanStep(
                step_order=idx,
                step_goal=s.step_goal,
                assigned_agent=s.assigned_agent,
                required_tools=s.required_tools,
                success_criteria=s.success_criteria,
            )
    return ExecutionPlan(
        plan_id=plan.plan_id,
        event_id=plan.event_id,
        steps=clean_steps,
        budget=plan.budget,
        revision=plan.revision,
        revise_reason=plan.revise_reason,
        degraded=plan.degraded,
    )


class PlannerAgent(BaseAgent[PlannerAgentInput, ExecutionPlan]):
    """Generates structured investigation plans from triage results."""

    agent_name: str = "planner_agent"

    # ------------------------------------------------------------------ #
    # Public API (called by orchestration, not BaseAgent.execute)
    # ------------------------------------------------------------------ #

    async def plan(self, event_context: EventContext) -> ExecutionPlan:
        """Generate an investigation plan via LLM (or rule fallback).

        When ``triage_result`` is unavailable in working memory this falls
        back to a disposition-only plan instead of crashing (see Blocker fix).
        """
        ec_event_id = event_context.event.event_id if event_context.event else "unknown"
        triage = await self._read_triage_result(event_context)
        if triage is None:
            logger.warning(
                "PlannerAgent.plan: triage_result unavailable for event=%s, "
                "falling back to disposition-only plan",
                ec_event_id,
            )
            plan = _build_disposition_only_plan(ec_event_id)
            await self._persist_plan(ec_event_id, plan)
            return plan
        return await self._plan_impl(ec_event_id, triage)

    async def plan_disposition_only(self, event_context: EventContext) -> ExecutionPlan:
        """Generate a deterministic single-step plan for disposition-only."""
        event_id = event_context.event.event_id if event_context.event else "unknown"
        plan = _build_disposition_only_plan(event_id)
        await self._persist_plan(event_id, plan)
        return plan

    async def revise(
        self,
        event_context: EventContext,
        failure_reason: str,
        previous_plan: ExecutionPlan,
    ) -> ExecutionPlan:
        """Revise an existing plan based on a failure reason."""
        event_id = event_context.event.event_id if event_context.event else "unknown"
        return await self._revise_impl(event_id, failure_reason, previous_plan)

    # ------------------------------------------------------------------ #
    # BaseAgent._run — dispatcher
    # ------------------------------------------------------------------ #

    async def _run(self, input: PlannerAgentInput) -> ExecutionPlan:
        event_id = input.event_id

        # Revise path: previous_plan + revise_reason both present
        if input.previous_plan is not None and input.revise_reason is not None:
            return await self._revise_impl(
                event_id,
                input.revise_reason,
                input.previous_plan,
            )

        # Disposition-only path: no triage_result
        if input.triage_result is None:
            plan = _build_disposition_only_plan(event_id)
            await self._persist_plan(event_id, plan)
            return plan

        # Normal plan path
        return await self._plan_impl(event_id, input.triage_result)

    # ------------------------------------------------------------------ #
    # Internal implementation
    # ------------------------------------------------------------------ #

    async def _plan_impl(
        self,
        event_id: str,
        triage_result: TriageResult,
    ) -> ExecutionPlan:
        """Core plan generation: try LLM, fall back to DEFAULT_PLANS."""

        # Idempotency check: replay should not re-call LLM
        existing = await self._read_existing_plan(event_id, revision=0)
        if existing is not None:
            logger.info(
                "PlannerAgent: reusing existing plan %s for event=%s",
                existing.plan_id,
                event_id,
            )
            return existing

        # Try LLM
        if self.llm_client is not None:
            try:
                plan = await self._llm_plan(event_id, triage_result)
                if self.working_memory is not None:
                    persisted = await self._persist_plan(event_id, plan)
                    if not persisted:
                        plan = ExecutionPlan(
                            plan_id=plan.plan_id,
                            event_id=plan.event_id,
                            steps=plan.steps,
                            budget=plan.budget,
                            revision=plan.revision,
                            revise_reason=plan.revise_reason,
                            degraded=True,
                        )
                return plan
            except Exception:
                logger.warning(
                    "PlannerAgent: LLM plan generation failed for event=%s, "
                    "falling back to DEFAULT_PLANS",
                    event_id,
                    exc_info=True,
                )

        # Rule fallback
        plan = get_default_plan(event_id, triage_result.event_type, _generate_plan_id(event_id, 0))
        logger.info(
            "PlannerAgent: using DEFAULT_PLANS for event=%s type=%s",
            event_id,
            triage_result.event_type.value,
        )
        await self._persist_plan(event_id, plan)
        return plan

    async def _revise_impl(
        self,
        event_id: str,
        failure_reason: str,
        previous_plan: ExecutionPlan,
    ) -> ExecutionPlan:
        """Core plan revision: try LLM, fall back to rule-based revision."""
        new_revision = previous_plan.revision + 1

        # Idempotency check
        existing = await self._read_existing_plan(event_id, revision=new_revision)
        if existing is not None:
            logger.info(
                "PlannerAgent: reusing existing revised plan %s rev=%d for event=%s",
                existing.plan_id,
                new_revision,
                event_id,
            )
            return existing

        # Try LLM revision
        if self.llm_client is not None:
            try:
                plan = await self._llm_revise(
                    event_id,
                    failure_reason,
                    previous_plan,
                )
                if self.working_memory is not None:
                    persisted = await self._persist_plan(event_id, plan)
                    if not persisted:
                        plan = ExecutionPlan(
                            plan_id=plan.plan_id,
                            event_id=plan.event_id,
                            steps=plan.steps,
                            budget=plan.budget,
                            revision=plan.revision,
                            revise_reason=plan.revise_reason,
                            degraded=True,
                        )
                return plan
            except Exception:
                logger.warning(
                    "PlannerAgent: LLM plan revision failed for event=%s, "
                    "falling back to rule-based revision",
                    event_id,
                    exc_info=True,
                )

        # Rule-based revision: rebuild from DEFAULT_PLANS with incremented revision
        triage = await self._read_triage_from_memory(event_id)
        event_type = triage.event_type if triage else EventType.OTHER
        plan = get_default_plan(
            event_id,
            event_type,
            _generate_plan_id(event_id, new_revision),
        )
        plan = ExecutionPlan(
            plan_id=plan.plan_id,
            event_id=plan.event_id,
            steps=plan.steps,
            budget=plan.budget,
            revision=new_revision,
            revise_reason=failure_reason,
            degraded=True,
        )
        await self._persist_plan(event_id, plan)
        return plan

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #

    async def _llm_plan(
        self,
        event_id: str,
        triage_result: TriageResult,
    ) -> ExecutionPlan:
        """Call LLM in JSON mode to generate an ExecutionPlan."""
        msgs_raw = build_plan_generate_messages(event_id, triage_result)
        messages = [
            LLMMessage(role=m["role"], content=m["content"])  # type: ignore[arg-type]
            for m in msgs_raw
        ]

        response = await self.llm_client.chat(  # type: ignore[union-attr]
            messages,
            event_id=event_id,
            agent_name=self.agent_name,
            prompt_key="plan_generate",
            json_mode=True,
            response_model=ExecutionPlan,
        )

        if response.parsed is not None and isinstance(response.parsed, ExecutionPlan):
            plan = response.parsed
        else:
            raise ValueError("LLM did not return a valid ExecutionPlan")

        # Enforce plan_id format and event_id
        plan = ExecutionPlan(
            plan_id=(
                plan.plan_id if plan.plan_id.startswith("pln-") else _generate_plan_id(event_id, 0)
            ),
            event_id=event_id,
            steps=plan.steps,
            budget=plan.budget if isinstance(plan.budget, PlanBudget) else PlanBudget(),
            revision=0,
            revise_reason=None,
            degraded=bool(response.degraded_reason),
        )

        plan = _validate_execution_plan(plan)
        return plan

    async def _llm_revise(
        self,
        event_id: str,
        failure_reason: str,
        previous_plan: ExecutionPlan,
    ) -> ExecutionPlan:
        """Call LLM to revise an existing plan.

        The LLM is instructed to return the **same** ``plan_id`` as the
        previous plan, but we do not trust it: ``plan_id`` is **forced** to
        ``previous_plan.plan_id`` regardless of what the model returns
        (Should-Fix #4).
        """
        msgs_raw = build_plan_revise_messages(event_id, failure_reason, previous_plan)
        messages = [
            LLMMessage(role=m["role"], content=m["content"])  # type: ignore[arg-type]
            for m in msgs_raw
        ]

        response = await self.llm_client.chat(  # type: ignore[union-attr]
            messages,
            event_id=event_id,
            agent_name=self.agent_name,
            prompt_key="plan_revise",
            json_mode=True,
            response_model=ExecutionPlan,
        )

        if response.parsed is not None and isinstance(response.parsed, ExecutionPlan):
            plan = response.parsed
        else:
            raise ValueError("LLM did not return a valid ExecutionPlan")

        new_revision = previous_plan.revision + 1
        plan = ExecutionPlan(
            plan_id=previous_plan.plan_id,
            event_id=event_id,
            steps=plan.steps,
            budget=plan.budget if isinstance(plan.budget, PlanBudget) else previous_plan.budget,
            revision=new_revision,
            revise_reason=failure_reason,
            degraded=bool(response.degraded_reason),
        )

        plan = _validate_execution_plan(plan)
        return plan

    # ------------------------------------------------------------------ #
    # Working memory helpers
    # ------------------------------------------------------------------ #

    async def _read_triage_result(self, event_context: EventContext) -> TriageResult | None:
        """Read triage_result from working memory if available."""
        if self.working_memory is None:
            return None
        try:
            data = await self.working_memory.read(
                event_context.event.event_id if event_context.event else "",
                "triage_result",
            )
            if data is not None:
                return TriageResult.model_validate(data)
        except Exception:
            logger.debug(
                "PlannerAgent: failed to read triage_result from working memory",
                exc_info=True,
            )
        return None

    async def _read_triage_from_memory(self, event_id: str) -> TriageResult | None:
        """Read triage_result directly from working memory."""
        if self.working_memory is None:
            return None
        try:
            data = await self.working_memory.read(event_id, "triage_result")
            if data is not None:
                return TriageResult.model_validate(data)
        except Exception:
            logger.debug("PlannerAgent: failed to read triage_result", exc_info=True)
        return None

    async def _read_existing_plan(
        self,
        event_id: str,
        revision: int,
    ) -> ExecutionPlan | None:
        """Check if a plan already exists for idempotent replay."""
        if self.working_memory is None:
            return None
        try:
            data = await self.working_memory.read(event_id, "execution_plan")
            if data is not None and isinstance(data, dict):
                plan = ExecutionPlan.model_validate(data)
                if plan.revision == revision:
                    return plan
        except Exception:
            logger.debug("PlannerAgent: failed to read execution_plan", exc_info=True)
        return None

    async def _persist_plan(self, event_id: str, plan: ExecutionPlan) -> bool:
        """Write the plan to EventContext.execution_plan via working memory.

        Returns ``True`` on success, ``False`` when persistence was skipped
        or failed.  Callers should treat a ``False`` return as a signal to
        mark the plan ``degraded=True`` (Should-Fix #2).
        """
        if self.working_memory is None:
            logger.warning(
                "PlannerAgent: no working_memory bound, plan not persisted for event=%s",
                event_id,
            )
            return False
        try:
            await self.working_memory.write(
                event_id,
                "execution_plan",
                plan.model_dump(mode="json"),
            )
            return True
        except Exception:
            logger.warning(
                "PlannerAgent: failed to persist plan for event=%s",
                event_id,
                exc_info=True,
            )
            return False


__all__ = ["PlannerAgent"]
