"""Cross-loop convergence guard — step counting, oscillation detection, and
duplicate-tool-call detection (ISSUE-052).

Per the spec (§4.12 / ISSUE-052), every orchestration path must call
``record_step`` before each substantive action and ``should_stop`` afterwards.
When a stop condition fires and a ``BoundWorkingMemory`` is configured, the
guard writes ``convergence_state`` into ``EventContext`` so the orchestrator
can escalate to human review.

Implements ``ConvergenceGuardPort`` from ``app.tools.executor`` so it can be
injected directly into ``ToolExecutor``.

Degradation strategy:
* Storage-unavailable → in-process dict (still guarantees single-process convergence).
* Guard internal exception → log warning, return ``stop=False``; outer limits
  (``MAX_REPLAN_COUNT``, ReAct ``max_rounds``) serve as the fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import ShadowTraceError
from app.models.workflow import (
    GLOBAL_MAX_STEPS,
    MAX_DUPLICATE_TOOL_CALLS,
    MAX_OSCILLATION,
    MAX_TOTAL_LLM_CALLS,
)
from app.services.working_memory import BoundWorkingMemory

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Step type — the five call-site labels ISSUE-052 defines
# --------------------------------------------------------------------------- #

_VALID_STEP_TYPES: frozenset[str] = frozenset(
    {"react_round", "replan", "agent_retry", "tool_call", "llm_call"}
)

# Sliding-window size for recent_actions: 2 * MAX_OSCILLATION + 2 buffer entries.
_RECENT_ACTIONS_MAXLEN = 2 * MAX_OSCILLATION + 2


class StopReason(StrEnum):
    """Reasons the guard may order a stop (ISSUE-052)."""

    GLOBAL_MAX_STEPS = "global_max_steps"
    OSCILLATION = "oscillation"
    DUPLICATE_TOOL_CALLS = "duplicate_tool_calls"
    MAX_LLM_CALLS = "max_llm_calls"
    NONE = "none"


# --------------------------------------------------------------------------- #
# Pydantic payloads
# --------------------------------------------------------------------------- #


class ConvergenceState(BaseModel):
    """Per-event accumulation of convergence counters (ISSUE-052 §4).

    Written into ``EventContext.convergence_state`` on stop so downstream
    services and the UI can surface the reason.
    """

    total_steps: int = 0
    react_rounds: int = 0
    replan_count: int = 0
    llm_calls: int = 0
    tool_call_signatures: dict[str, int] = Field(default_factory=dict)
    recent_actions: list[str] = Field(default_factory=list)


class StopDecision(BaseModel):
    """Immutable stop-or-continue decision returned by ``should_stop``.

    Truthiness delegates to the ``stop`` field so a ``StopDecision`` can be
    used directly in ``if await guard.should_stop(eid):`` as required by
    ``ConvergenceGuardPort``.
    """

    stop: bool
    reason: StopReason
    detail: str = ""

    def __bool__(self) -> bool:
        return self.stop


# --------------------------------------------------------------------------- #
# ConvergenceGuard
# --------------------------------------------------------------------------- #


class ConvergenceGuard:
    """Cross-loop convergence gate for a single event's orchestration lifetime.

    Implements ``ConvergenceGuardPort`` so it can be injected directly into
    ``ToolExecutor`` without any adapter layer.

    Usage across the codebase (per ISSUE-052 §4):

    * ``BaseLLMClient``: ``await guard.record_step(event_id, "llm_call")``
      before every network attempt (including retries).
    * ``ToolExecutor``: calls ``await guard.record_step(event_id, tool_name=t)``
      and ``await guard.should_stop(event_id)`` before every dispatch.
    * ReAct engine (ISSUE-053): ``await guard.record_step(event_id, "react_round")``
      at the top of each round.
    * SuperAgent (ISSUE-054): ``await guard.record_step(event_id, "agent_retry")``
      per agent step.
    * PlannerAgent replan (ISSUE-062): ``await guard.record_step(event_id, "replan")``.

    Every call site follows the pattern::

        await guard.record_step(event_id, step_type, signature=...)
        decision = await guard.should_stop(event_id)
        if decision:  # bool delegates to decision.stop
            # convergence_state is persisted when working_memory is configured
            # escalate to human / abort further automation
    """

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def __init__(self, *, working_memory: BoundWorkingMemory | None = None) -> None:
        # In-process store remains the primary accumulator; working_memory
        # persists convergence_state on stop (Redis-backed EventContext).
        self._working_memory = working_memory
        self._states: dict[str, ConvergenceState] = {}

    # ------------------------------------------------------------------ #
    # record_step
    # ------------------------------------------------------------------ #

    async def record_step(
        self,
        event_id: str,
        step_type: str = "tool_call",
        *,
        tool_name: str | None = None,
        params: dict[str, Any] | None = None,
        signature: str | None = None,
    ) -> None:
        """Accumulate one step for *event_id*.

        Two calling conventions are supported:

        1. **Protocol** (``ConvergenceGuardPort`` from ``app.tools.executor``):
           ``await guard.record_step(event_id, tool_name="block_ip")``
           — ``step_type`` defaults to ``"tool_call"`` and ``signature`` is
           derived from *tool_name*.

        2. **Rich** (other orchestration callers):
           ``await guard.record_step(event_id, "react_round")``
           ``await guard.record_step(event_id, "llm_call")``
           ``await guard.record_step(event_id, "tool_call", signature=sig)``

        Args:
            event_id: The investigation event identifier.
            step_type: One of ``react_round``, ``replan``, ``agent_retry``,
                       ``tool_call``, ``llm_call``.  Defaults to ``"tool_call"``
                       for Protocol compatibility.
            tool_name: Keyword-only shorthand used by ``ToolExecutor``.
            params: Optional tool parameters; combined with *tool_name* via
                    ``make_tool_call_signature`` when *signature* is omitted.
            signature: Explicit stable fingerprint of ``(tool_name, params)``.
                       Takes precedence over *tool_name* / *params*.
        """
        if signature is not None:
            effective_sig: str | None = signature
        elif tool_name is not None and params is not None:
            effective_sig = make_tool_call_signature(tool_name, params)
        else:
            effective_sig = tool_name

        is_known = step_type in _VALID_STEP_TYPES
        if not is_known:
            logger.warning(
                "ConvergenceGuard.record_step: unknown step_type=%s (event=%s) — "
                "still counting toward total_steps for global limit safety",
                step_type,
                event_id,
            )

        try:
            state = self._get_or_create_state(event_id)
            state.total_steps += 1

            # Build the action label used for oscillation detection.
            label = f"{step_type}:{effective_sig}" if effective_sig else step_type

            if is_known:
                if step_type == "react_round":
                    state.react_rounds += 1
                elif step_type == "replan":
                    state.replan_count += 1
                elif step_type == "tool_call":
                    if effective_sig:
                        state.tool_call_signatures[effective_sig] = (
                            state.tool_call_signatures.get(effective_sig, 0) + 1
                        )
                elif step_type == "llm_call":
                    state.llm_calls += 1
                # agent_retry has no dedicated counter
                # — tracked via total_steps + recent_actions

            state.recent_actions.append(label)

            # Trim the sliding window so it never grows unbounded.
            if len(state.recent_actions) > _RECENT_ACTIONS_MAXLEN:
                state.recent_actions = state.recent_actions[-_RECENT_ACTIONS_MAXLEN:]

        except Exception:
            logger.exception(
                "ConvergenceGuard.record_step failed for event=%s step_type=%s",
                event_id,
                step_type,
            )
            # Degradation: exception inside the guard → no-op.  Outer limits
            # (MAX_REPLAN_COUNT, max_rounds) still protect the loop.

    # ------------------------------------------------------------------ #
    # should_stop
    # ------------------------------------------------------------------ #

    async def should_stop(self, event_id: str) -> StopDecision:
        """Check all stop conditions in priority order.

        Priority: global_max_steps → max_llm_calls → duplicate_tool_calls →
        oscillation.  The first match wins.

        Returns a ``StopDecision`` whose ``__bool__`` delegates to ``.stop``,
        so ``if await guard.should_stop(eid):`` works as expected by
        ``ConvergenceGuardPort``.
        """
        state = self._states.get(event_id)
        if state is None:
            return StopDecision(stop=False, reason=StopReason.NONE)

        try:
            # 1. Global step cap
            if state.total_steps >= GLOBAL_MAX_STEPS:
                return await self._stop_decision(
                    event_id,
                    state,
                    StopReason.GLOBAL_MAX_STEPS,
                    f"total_steps={state.total_steps} >= GLOBAL_MAX_STEPS={GLOBAL_MAX_STEPS}",
                )

            # 2. LLM call cap
            if state.llm_calls >= MAX_TOTAL_LLM_CALLS:
                return await self._stop_decision(
                    event_id,
                    state,
                    StopReason.MAX_LLM_CALLS,
                    (f"llm_calls={state.llm_calls} >= MAX_TOTAL_LLM_CALLS={MAX_TOTAL_LLM_CALLS}"),
                )

            # 3. Duplicate tool calls
            for sig, count in state.tool_call_signatures.items():
                if count > MAX_DUPLICATE_TOOL_CALLS:
                    return await self._stop_decision(
                        event_id,
                        state,
                        StopReason.DUPLICATE_TOOL_CALLS,
                        (
                            f"tool_call signature '{sig}' called {count} times "
                            f"(limit={MAX_DUPLICATE_TOOL_CALLS})"
                        ),
                    )

            # 4. Oscillation (A, B, A, B pattern)
            osc = self._detect_oscillation(state.recent_actions)
            if osc:
                return await self._stop_decision(
                    event_id,
                    state,
                    StopReason.OSCILLATION,
                    f"A/B oscillation detected: '{osc[0]}' ↔ '{osc[1]}'",
                )

            return StopDecision(stop=False, reason=StopReason.NONE)

        except Exception:
            logger.exception("ConvergenceGuard.should_stop failed for event=%s", event_id)
            # Degradation: guard internal error → don't force stop; let outer
            # limits (MAX_REPLAN_COUNT, ReAct max_rounds) catch runaway loops.
            return StopDecision(stop=False, reason=StopReason.NONE)

    # ------------------------------------------------------------------ #
    # get_state / reset
    # ------------------------------------------------------------------ #

    def get_state(self, event_id: str) -> ConvergenceState:
        """Return a copy of the current convergence state for *event_id*.

        Returns a fresh ``ConvergenceState`` when no steps have been recorded.
        """
        existing = self._states.get(event_id)
        if existing is None:
            return ConvergenceState()
        # Return a copy so callers can't mutate the internal accumulator.
        return existing.model_copy(deep=True)

    def reset(self, event_id: str) -> None:
        """Clear all convergence counters for *event_id*.

        **Caller responsibility:** the orchestrator (SuperAgent /
        WorkflowRuntimeService) MUST call this when an event reaches a
        terminal status (CLOSED or FAILED).  Without this the in-process
        ``_states`` dict accumulates every event forever and will leak
        memory in long-running processes.
        """
        self._states.pop(event_id, None)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_or_create_state(self, event_id: str) -> ConvergenceState:
        if event_id not in self._states:
            self._states[event_id] = ConvergenceState()
        return self._states[event_id]

    async def _stop_decision(
        self,
        event_id: str,
        state: ConvergenceState,
        reason: StopReason,
        detail: str,
    ) -> StopDecision:
        decision = StopDecision(stop=True, reason=reason, detail=detail)
        await self._persist_convergence_state(event_id, state, decision)
        return decision

    async def _persist_convergence_state(
        self,
        event_id: str,
        state: ConvergenceState,
        decision: StopDecision,
    ) -> None:
        wm = self._working_memory
        if wm is None:
            return
        payload = {
            **state.model_dump(),
            "stop_reason": decision.reason.value,
            "stop_detail": decision.detail,
        }
        try:
            await wm.write(event_id, "convergence_state", payload)
        except ShadowTraceError:
            logger.exception(
                "Failed to persist convergence_state for event=%s reason=%s",
                event_id,
                decision.reason.value,
            )

    @staticmethod
    def _detect_oscillation(actions: list[str]) -> tuple[str, str] | None:
        """Return ``(a, b)`` when the tail of *actions* shows an A,B,A,B
        oscillation, or ``None`` otherwise.

        An oscillation is defined as two distinct action labels alternating
        at least ``MAX_OSCILLATION`` times (i.e. A,B,A,B when
        ``MAX_OSCILLATION == 2``).
        """
        window = 2 * MAX_OSCILLATION  # = 4
        if len(actions) < window:
            return None

        recent = actions[-window:]
        a, b = recent[0], recent[1]
        if a == b:
            return None

        for i in range(window):
            expected = a if i % 2 == 0 else b
            if recent[i] != expected:
                return None

        return a, b


# --------------------------------------------------------------------------- #
# Convenience helpers for call-site signature construction
# --------------------------------------------------------------------------- #


def make_tool_call_signature(tool_name: str, params: dict[str, Any]) -> str:
    """Build a stable, compact fingerprint for tool-call dedup detection.

    The returned string is suitable as the *signature* argument to
    ``ConvergenceGuard.record_step`` for ``step_type="tool_call"``.
    """
    # Deterministic JSON (sorted keys) → short hash so signatures don't leak
    # parameter values into logs / plain-text state.
    try:
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        # Fallback for non-JSON-serializable values.  Sort keys to avoid
        # non-determinism from dict iteration order; use repr on values
        # rather than the whole items() list to avoid memory addresses.
        canonical = json.dumps(
            {str(k): repr(v) for k, v in sorted(params.items())},
            sort_keys=True,
            ensure_ascii=True,
        )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{tool_name}:{digest}"


# --------------------------------------------------------------------------- #
# Re-export the subset that callers outside this package need.
# --------------------------------------------------------------------------- #

__all__ = [
    "ConvergenceGuard",
    "ConvergenceState",
    "StopDecision",
    "StopReason",
    "make_tool_call_signature",
]
