"""ConvergenceGuard tests (ISSUE-052).

Coverage:
* Global step limit exceeded
* LLM call limit exceeded
* Duplicate tool call detection
* A/B oscillation detection
* Normal flow (3 rounds) — no false-positive stop
* State retrieval, reset, and edge cases
* make_tool_call_signature helper
* ConvergenceGuardPort Protocol compatibility
* StopDecision __bool__ truthiness
* Guard degradation — exception paths
* Reset lifecycle — memory leak prevention
"""

from __future__ import annotations

import json

import pytest

from app.models.workflow import (
    GLOBAL_MAX_STEPS,
    MAX_DUPLICATE_TOOL_CALLS,
    MAX_OSCILLATION,
    MAX_TOTAL_LLM_CALLS,
)
from app.orchestration.convergence_guard import (
    ConvergenceGuard,
    ConvergenceState,
    StopDecision,
    StopReason,
    make_tool_call_signature,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def guard() -> ConvergenceGuard:
    """Fresh guard instance for every test."""
    return ConvergenceGuard()


@pytest.fixture
def event_id() -> str:
    return "evt-test-052-0001"


# --------------------------------------------------------------------------- #
# StopDecision model
# --------------------------------------------------------------------------- #


class TestStopDecision:
    def test_defaults(self) -> None:
        sd = StopDecision(stop=False, reason=StopReason.NONE)
        assert sd.stop is False
        assert sd.reason is StopReason.NONE
        assert sd.detail == ""

    def test_stop_with_detail(self) -> None:
        sd = StopDecision(
            stop=True,
            reason=StopReason.GLOBAL_MAX_STEPS,
            detail="limit reached",
        )
        assert sd.stop is True
        assert sd.detail == "limit reached"

    def test_bool_delegates_to_stop(self) -> None:
        """__bool__ is used by ConvergenceGuardPort — truthiness == .stop."""
        assert bool(StopDecision(stop=True, reason=StopReason.NONE)) is True
        assert bool(StopDecision(stop=False, reason=StopReason.OSCILLATION)) is False


# --------------------------------------------------------------------------- #
# ConvergenceState model
# --------------------------------------------------------------------------- #


class TestConvergenceState:
    def test_defaults(self) -> None:
        cs = ConvergenceState()
        assert cs.total_steps == 0
        assert cs.react_rounds == 0
        assert cs.replan_count == 0
        assert cs.llm_calls == 0
        assert cs.tool_call_signatures == {}
        assert cs.recent_actions == []

    def test_model_copy_is_deep(self) -> None:
        cs = ConvergenceState(
            total_steps=5,
            tool_call_signatures={"a": 1},
            recent_actions=["x", "y"],
        )
        copy = cs.model_copy(deep=True)
        copy.tool_call_signatures["a"] = 99
        copy.recent_actions.append("z")
        assert cs.tool_call_signatures["a"] == 1
        assert cs.recent_actions == ["x", "y"]


# --------------------------------------------------------------------------- #
# record_step — basic counters
# --------------------------------------------------------------------------- #


class TestRecordStepCounters:
    async def test_total_steps_increments(self, guard: ConvergenceGuard, event_id: str) -> None:
        for _i in range(5):
            await guard.record_step(event_id, "llm_call")
        state = guard.get_state(event_id)
        assert state.total_steps == 5

    async def test_react_round_counter(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "react_round")
        await guard.record_step(event_id, "react_round")
        await guard.record_step(event_id, "react_round")
        state = guard.get_state(event_id)
        assert state.react_rounds == 3
        assert state.total_steps == 3

    async def test_replan_counter(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "replan")
        await guard.record_step(event_id, "replan")
        state = guard.get_state(event_id)
        assert state.replan_count == 2
        assert state.total_steps == 2

    async def test_llm_call_counter(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "llm_call")
        await guard.record_step(event_id, "llm_call")
        await guard.record_step(event_id, "llm_call")
        await guard.record_step(event_id, "llm_call")
        state = guard.get_state(event_id)
        assert state.llm_calls == 4
        assert state.total_steps == 4

    async def test_agent_retry(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "agent_retry", signature="EvidenceAgent")
        state = guard.get_state(event_id)
        assert state.total_steps == 1

    async def test_tool_call_signature_counted(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        sig = make_tool_call_signature("query_ip_reputation", {"ip": "10.0.0.1"})
        for _ in range(4):
            await guard.record_step(event_id, "tool_call", signature=sig)
        state = guard.get_state(event_id)
        assert state.total_steps == 4
        assert state.tool_call_signatures[sig] == 4

    async def test_different_tool_signatures_separate(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        sig_a = make_tool_call_signature("query_ip", {"ip": "1.1.1.1"})
        sig_b = make_tool_call_signature("query_domain", {"domain": "evil.com"})
        await guard.record_step(event_id, "tool_call", signature=sig_a)
        await guard.record_step(event_id, "tool_call", signature=sig_a)
        await guard.record_step(event_id, "tool_call", signature=sig_b)
        state = guard.get_state(event_id)
        assert state.tool_call_signatures[sig_a] == 2
        assert state.tool_call_signatures[sig_b] == 1

    # --- Protocol calling convention ---

    async def test_protocol_convention_tool_name(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        """ToolExecutor calls record_step(event_id, tool_name="block_ip")."""
        await guard.record_step(event_id, tool_name="block_ip")
        state = guard.get_state(event_id)
        assert state.total_steps == 1
        assert "block_ip" in state.tool_call_signatures
        assert state.tool_call_signatures["block_ip"] == 1

    async def test_protocol_convention_should_stop_bool(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        """ConvergenceGuardPort expects async should_stop → bool-like."""
        decision = await guard.should_stop(event_id)
        assert isinstance(decision, StopDecision)
        # __bool__ delegates → used in 'if await guard.should_stop(eid):'
        assert bool(decision) is False
        assert decision is not None  # model instances are never None
        # Fill to trigger stop
        sig = make_tool_call_signature("t", {"p": 1})
        for _ in range(GLOBAL_MAX_STEPS):
            await guard.record_step(event_id, "tool_call", signature=sig)
        decision2 = await guard.should_stop(event_id)
        assert decision2.stop is True
        assert bool(decision2) is True


# --------------------------------------------------------------------------- #
# record_step — recent_actions sliding window
# --------------------------------------------------------------------------- #


class TestRecentActionsWindow:
    async def test_actions_appended(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "react_round")
        await guard.record_step(event_id, "llm_call")
        await guard.record_step(event_id, "tool_call", signature="t1")
        state = guard.get_state(event_id)
        assert len(state.recent_actions) == 3
        assert state.recent_actions[0] == "react_round"
        assert state.recent_actions[2] == "tool_call:t1"

    async def test_window_trimmed(self, guard: ConvergenceGuard, event_id: str) -> None:
        for _i in range(10):
            await guard.record_step(event_id, "llm_call")
        state = guard.get_state(event_id)
        max_expected = 2 * MAX_OSCILLATION + 2
        assert len(state.recent_actions) == max_expected
        assert state.total_steps == 10


# --------------------------------------------------------------------------- #
# record_step — edge cases
# --------------------------------------------------------------------------- #


class TestRecordStepEdgeCases:
    async def test_unknown_step_type_still_counts_total_steps(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        await guard.record_step(event_id, "nonexistent_type")
        state = guard.get_state(event_id)
        assert state.total_steps == 1
        assert state.react_rounds == 0
        assert state.replan_count == 0
        assert state.llm_calls == 0
        assert state.tool_call_signatures == {}

    async def test_tool_call_without_signature(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        await guard.record_step(event_id, "tool_call")
        state = guard.get_state(event_id)
        assert state.total_steps == 1
        assert state.tool_call_signatures == {}
        assert state.recent_actions == ["tool_call"]

    async def test_no_signature_label(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "replan")
        state = guard.get_state(event_id)
        assert state.recent_actions == ["replan"]


# --------------------------------------------------------------------------- #
# should_stop — global_max_steps
# --------------------------------------------------------------------------- #


class TestShouldStopGlobalMaxSteps:
    async def test_stops_at_limit(self, guard: ConvergenceGuard, event_id: str) -> None:
        for _ in range(GLOBAL_MAX_STEPS):
            await guard.record_step(event_id, "llm_call")
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.GLOBAL_MAX_STEPS
        assert str(GLOBAL_MAX_STEPS) in decision.detail

    async def test_stops_just_below_limit(self, guard: ConvergenceGuard, event_id: str) -> None:
        for _ in range(GLOBAL_MAX_STEPS - 1):
            await guard.record_step(event_id, "replan")
        decision = await guard.should_stop(event_id)
        assert decision.stop is False


# --------------------------------------------------------------------------- #
# should_stop — max_llm_calls
# --------------------------------------------------------------------------- #


class TestShouldStopMaxLLMCalls:
    async def test_stops_at_llm_limit(self, guard: ConvergenceGuard, event_id: str) -> None:
        for _ in range(MAX_TOTAL_LLM_CALLS):
            await guard.record_step(event_id, "llm_call")
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.MAX_LLM_CALLS
        assert str(MAX_TOTAL_LLM_CALLS) in decision.detail

    async def test_no_stop_below_llm_limit(self, guard: ConvergenceGuard, event_id: str) -> None:
        for _ in range(MAX_TOTAL_LLM_CALLS - 1):
            await guard.record_step(event_id, "llm_call")
        decision = await guard.should_stop(event_id)
        assert decision.stop is False

    async def test_non_llm_steps_dont_count_toward_llm_limit(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        for _ in range(MAX_TOTAL_LLM_CALLS + 5):
            await guard.record_step(event_id, "replan")
        decision = await guard.should_stop(event_id)
        if decision.stop:
            assert decision.reason != StopReason.MAX_LLM_CALLS


# --------------------------------------------------------------------------- #
# should_stop — duplicate_tool_calls
# --------------------------------------------------------------------------- #


class TestShouldStopDuplicateToolCalls:
    async def test_stops_on_duplicate(self, guard: ConvergenceGuard, event_id: str) -> None:
        sig = make_tool_call_signature("block_ip", {"ip": "10.0.0.99"})
        for _ in range(MAX_DUPLICATE_TOOL_CALLS + 1):
            await guard.record_step(event_id, "tool_call", signature=sig)
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.DUPLICATE_TOOL_CALLS
        assert "block_ip" in decision.detail

    async def test_no_stop_at_limit(self, guard: ConvergenceGuard, event_id: str) -> None:
        sig = make_tool_call_signature("block_ip", {"ip": "10.0.0.99"})
        for _ in range(MAX_DUPLICATE_TOOL_CALLS):
            await guard.record_step(event_id, "tool_call", signature=sig)
        decision = await guard.should_stop(event_id)
        assert decision.stop is False

    async def test_different_params_are_different_signatures(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        sig_a = make_tool_call_signature("block_ip", {"ip": "10.0.0.1"})
        sig_b = make_tool_call_signature("block_ip", {"ip": "10.0.0.2"})
        sig_c = make_tool_call_signature("block_ip", {"ip": "10.0.0.3"})
        for _ in range(MAX_DUPLICATE_TOOL_CALLS):
            await guard.record_step(event_id, "tool_call", signature=sig_a)
            await guard.record_step(event_id, "tool_call", signature=sig_b)
            await guard.record_step(event_id, "tool_call", signature=sig_c)
        decision = await guard.should_stop(event_id)
        assert decision.stop is False

    async def test_different_tool_names_separate_signatures(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        sig_a = make_tool_call_signature("query_ip", {"ip": "1.1.1.1"})
        sig_b = make_tool_call_signature("query_domain", {"domain": "evil.com"})
        sig_c = make_tool_call_signature("query_hash", {"hash": "abc123"})
        for _ in range(MAX_DUPLICATE_TOOL_CALLS):
            await guard.record_step(event_id, "tool_call", signature=sig_a)
            await guard.record_step(event_id, "tool_call", signature=sig_b)
            await guard.record_step(event_id, "tool_call", signature=sig_c)
        decision = await guard.should_stop(event_id)
        assert decision.stop is False


# --------------------------------------------------------------------------- #
# should_stop — oscillation
# --------------------------------------------------------------------------- #


class TestShouldStopOscillation:
    async def test_ab_oscillation_detected(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "tool_call", signature="block_ip:10.0.0.1")
        await guard.record_step(event_id, "tool_call", signature="unblock_ip:10.0.0.1")
        await guard.record_step(event_id, "tool_call", signature="block_ip:10.0.0.1")
        await guard.record_step(event_id, "tool_call", signature="unblock_ip:10.0.0.1")
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.OSCILLATION
        assert "block_ip:10.0.0.1" in decision.detail
        assert "unblock_ip:10.0.0.1" in decision.detail
        assert "↔" in decision.detail

    async def test_three_alternations_no_oscillation(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        await guard.record_step(event_id, "tool_call", signature="a")
        await guard.record_step(event_id, "tool_call", signature="b")
        await guard.record_step(event_id, "tool_call", signature="a")
        decision = await guard.should_stop(event_id)
        assert decision.stop is False

    async def test_same_action_no_oscillation(self, guard: ConvergenceGuard, event_id: str) -> None:
        for _ in range(4):
            await guard.record_step(event_id, "tool_call", signature="same")
        decision = await guard.should_stop(event_id)
        if decision.stop:
            assert decision.reason != StopReason.OSCILLATION

    async def test_oscillation_must_be_consecutive_tail(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        await guard.record_step(event_id, "tool_call", signature="a")
        await guard.record_step(event_id, "tool_call", signature="b")
        await guard.record_step(event_id, "tool_call", signature="a")
        await guard.record_step(event_id, "tool_call", signature="b")
        await guard.record_step(event_id, "tool_call", signature="c")
        decision = await guard.should_stop(event_id)
        assert decision.stop is False

    async def test_oscillation_with_react_round_and_replan(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        await guard.record_step(event_id, "react_round")
        await guard.record_step(event_id, "replan")
        await guard.record_step(event_id, "react_round")
        await guard.record_step(event_id, "replan")
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.OSCILLATION


# --------------------------------------------------------------------------- #
# should_stop — priority order
# --------------------------------------------------------------------------- #


class TestShouldStopPriority:
    async def test_global_max_steps_takes_priority(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        sig = make_tool_call_signature("t", {"p": 1})
        for _ in range(GLOBAL_MAX_STEPS):
            await guard.record_step(event_id, "tool_call", signature=sig)
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.GLOBAL_MAX_STEPS

    async def test_llm_priority_over_duplicate(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        for _ in range(MAX_TOTAL_LLM_CALLS):
            await guard.record_step(event_id, "llm_call")
        sig = make_tool_call_signature("t", {"p": 1})
        for _ in range(MAX_DUPLICATE_TOOL_CALLS + 5):
            await guard.record_step(event_id, "tool_call", signature=sig)
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.MAX_LLM_CALLS


# --------------------------------------------------------------------------- #
# Normal flow — acceptance test
# --------------------------------------------------------------------------- #


class TestNormalFlowNoStop:
    async def test_three_round_convergence_no_stop(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        """Simulate a normal 3-round investigation that converges cleanly."""
        for round_idx in range(3):
            await guard.record_step(event_id, "react_round")
            tools_this_round = 2 if round_idx < 2 else 1
            for t in range(tools_this_round):
                sig = make_tool_call_signature(f"query_r{round_idx}_t{t}", {"round": round_idx})
                await guard.record_step(event_id, "tool_call", signature=sig)
            await guard.record_step(event_id, "llm_call")

        decision = await guard.should_stop(event_id)
        assert decision.stop is False, (
            f"Expected no stop for normal 3-round flow, got reason={decision.reason}"
        )
        state = guard.get_state(event_id)
        assert state.react_rounds == 3
        assert state.llm_calls == 3
        assert state.total_steps == 11

    async def test_many_unique_tools_no_duplicate_stop(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        """Call many different tools — none exceed duplicate limit."""
        for i in range(20):
            sig = make_tool_call_signature(f"tool_{i}", {"idx": i})
            await guard.record_step(event_id, "tool_call", signature=sig)
        decision = await guard.should_stop(event_id)
        assert decision.stop is False


# --------------------------------------------------------------------------- #
# get_state
# --------------------------------------------------------------------------- #


class TestGetState:
    def test_returns_default_for_unknown_event(self, guard: ConvergenceGuard) -> None:
        state = guard.get_state("evt-nonexistent")
        assert isinstance(state, ConvergenceState)
        assert state.total_steps == 0

    async def test_returns_copy_not_reference(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "llm_call")
        state = guard.get_state(event_id)
        state.total_steps = 999
        internal = guard.get_state(event_id)
        assert internal.total_steps == 1

    async def test_full_state_snapshot(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "react_round")
        await guard.record_step(event_id, "replan")
        await guard.record_step(event_id, "llm_call")
        sig = make_tool_call_signature("query", {"k": "v"})
        await guard.record_step(event_id, "tool_call", signature=sig)
        state = guard.get_state(event_id)
        assert state.total_steps == 4
        assert state.react_rounds == 1
        assert state.replan_count == 1
        assert state.llm_calls == 1
        assert sig in state.tool_call_signatures
        assert len(state.recent_actions) == 4


# --------------------------------------------------------------------------- #
# reset
# --------------------------------------------------------------------------- #


class TestReset:
    async def test_reset_clears_state(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "llm_call")
        await guard.record_step(event_id, "llm_call")
        assert guard.get_state(event_id).total_steps == 2
        guard.reset(event_id)
        assert guard.get_state(event_id).total_steps == 0

    def test_reset_unknown_event_noop(self, guard: ConvergenceGuard) -> None:
        guard.reset("evt-nonexistent")

    async def test_reset_then_reuse(self, guard: ConvergenceGuard, event_id: str) -> None:
        await guard.record_step(event_id, "llm_call")
        guard.reset(event_id)
        await guard.record_step(event_id, "react_round")
        state = guard.get_state(event_id)
        assert state.total_steps == 1
        assert state.react_rounds == 1
        assert state.llm_calls == 0


# --------------------------------------------------------------------------- #
# should_stop — edge cases
# --------------------------------------------------------------------------- #


class TestShouldStopEdgeCases:
    async def test_unknown_event_returns_none(self, guard: ConvergenceGuard) -> None:
        decision = await guard.should_stop("evt-nonexistent")
        assert decision.stop is False
        assert decision.reason == StopReason.NONE
        assert bool(decision) is False

    async def test_empty_state_no_stop(self, guard: ConvergenceGuard, event_id: str) -> None:
        decision = await guard.should_stop(event_id)
        assert decision.stop is False
        assert decision.reason == StopReason.NONE

    async def test_convergence_state_serializable(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        await guard.record_step(event_id, "react_round")
        await guard.record_step(event_id, "llm_call")
        await guard.record_step(event_id, "tool_call", signature="t:sig1")
        state = guard.get_state(event_id)
        dumped = state.model_dump()
        assert dumped["total_steps"] == 3
        assert dumped["react_rounds"] == 1
        assert "t:sig1" in dumped["tool_call_signatures"]
        json.dumps(dumped)  # should not raise


# --------------------------------------------------------------------------- #
# Multiple events — isolation
# --------------------------------------------------------------------------- #


class TestMultipleEventsIsolation:
    async def test_independent_counters(self, guard: ConvergenceGuard) -> None:
        e1, e2 = "evt-001", "evt-002"
        await guard.record_step(e1, "llm_call")
        await guard.record_step(e1, "llm_call")
        await guard.record_step(e2, "react_round")
        s1 = guard.get_state(e1)
        s2 = guard.get_state(e2)
        assert s1.total_steps == 2
        assert s1.llm_calls == 2
        assert s1.react_rounds == 0
        assert s2.total_steps == 1
        assert s2.react_rounds == 1
        assert s2.llm_calls == 0

    async def test_reset_does_not_affect_other_event(self, guard: ConvergenceGuard) -> None:
        e1, e2 = "evt-001", "evt-002"
        await guard.record_step(e1, "llm_call")
        await guard.record_step(e2, "llm_call")
        await guard.record_step(e2, "llm_call")
        guard.reset(e1)
        assert guard.get_state(e1).total_steps == 0
        assert guard.get_state(e2).total_steps == 2


# --------------------------------------------------------------------------- #
# make_tool_call_signature
# --------------------------------------------------------------------------- #


class TestMakeToolCallSignature:
    def test_same_params_same_signature(self) -> None:
        s1 = make_tool_call_signature("query_ip", {"ip": "1.2.3.4"})
        s2 = make_tool_call_signature("query_ip", {"ip": "1.2.3.4"})
        assert s1 == s2

    def test_different_params_different_signature(self) -> None:
        s1 = make_tool_call_signature("query_ip", {"ip": "1.2.3.4"})
        s2 = make_tool_call_signature("query_ip", {"ip": "5.6.7.8"})
        assert s1 != s2

    def test_different_tool_different_signature(self) -> None:
        s1 = make_tool_call_signature("query_ip", {"ip": "1.2.3.4"})
        s2 = make_tool_call_signature("query_domain", {"ip": "1.2.3.4"})
        assert s1 != s2

    def test_param_order_independent(self) -> None:
        s1 = make_tool_call_signature("t", {"a": 1, "b": 2})
        s2 = make_tool_call_signature("t", {"b": 2, "a": 1})
        assert s1 == s2

    def test_contains_tool_name(self) -> None:
        sig = make_tool_call_signature("block_ip", {"ip": "10.0.0.1"})
        assert sig.startswith("block_ip:")

    def test_short_digest_length(self) -> None:
        sig = make_tool_call_signature("t", {"p": 1})
        assert len(sig) == len("t:") + 16

    def test_non_json_serializable_params_fallback(self) -> None:
        """Non-serializable params use stable fallback, don't crash."""
        from datetime import datetime

        sig = make_tool_call_signature("t", {"ts": datetime(2026, 7, 19)})
        assert sig.startswith("t:")
        s2 = make_tool_call_signature("t", {"ts": datetime(2026, 7, 19)})
        assert sig == s2


# --------------------------------------------------------------------------- #
# Guard degradation — exception paths
# --------------------------------------------------------------------------- #


class TestGuardDegradation:
    async def test_record_step_exception_does_not_crash(
        self, guard: ConvergenceGuard, event_id: str, monkeypatch
    ) -> None:
        """Guard internal error in record_step must not propagate."""

        def _boom(_self, eid):
            raise RuntimeError("simulated storage failure")

        monkeypatch.setattr(guard, "_get_or_create_state", _boom)
        # Should not raise
        await guard.record_step(event_id, "llm_call")
        decision = await guard.should_stop(event_id)
        assert decision.stop is False

    async def test_should_stop_exception_returns_no_stop(
        self, guard: ConvergenceGuard, event_id: str, monkeypatch
    ) -> None:
        """Guard internal error in should_stop must return stop=False."""
        await guard.record_step(event_id, "llm_call")

        def _boom(actions):
            raise RuntimeError("simulated oscillation check failure")

        monkeypatch.setattr(guard, "_detect_oscillation", _boom)
        decision = await guard.should_stop(event_id)
        assert decision.stop is False
        assert decision.reason == StopReason.NONE


# --------------------------------------------------------------------------- #
# Reset lifecycle — memory leak prevention
# --------------------------------------------------------------------------- #


class TestResetLifecycle:
    async def test_reset_removes_internal_dict_entry(self, guard: ConvergenceGuard) -> None:
        eid = "evt-lifecycle-001"
        await guard.record_step(eid, "llm_call")
        await guard.record_step(eid, "react_round")
        assert eid in guard._states
        guard.reset(eid)
        assert eid not in guard._states

    async def test_full_lifecycle_record_stop_reset(self, guard: ConvergenceGuard) -> None:
        eid = "evt-lifecycle-002"
        sig = make_tool_call_signature("t", {"p": 1})
        for _ in range(GLOBAL_MAX_STEPS):
            await guard.record_step(eid, "tool_call", signature=sig)
        decision = await guard.should_stop(eid)
        assert decision.stop is True
        guard.reset(eid)
        assert guard.get_state(eid).total_steps == 0
        assert eid not in guard._states


# --------------------------------------------------------------------------- #
# Working memory persistence on stop
# --------------------------------------------------------------------------- #


class _MockBoundWorkingMemory:
    def __init__(self, writer_name: str = "ConvergenceGuard") -> None:
        self.writer_name = writer_name
        self._store: dict[str, object] = {}

    async def read(self, event_id: str, key: str) -> object:
        return self._store.get(key)

    async def write(self, event_id: str, key: str, value: object) -> None:
        self._store[key] = value


class TestWorkingMemoryPersistence:
    async def test_stop_writes_convergence_state(self, event_id: str) -> None:
        wm = _MockBoundWorkingMemory(writer_name="ConvergenceGuard")
        guard = ConvergenceGuard(working_memory=wm)
        for _ in range(GLOBAL_MAX_STEPS):
            await guard.record_step(event_id, "llm_call")
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        stored = await wm.read(event_id, "convergence_state")
        assert isinstance(stored, dict)
        assert stored["stop_reason"] == StopReason.GLOBAL_MAX_STEPS.value
        assert stored["total_steps"] == GLOBAL_MAX_STEPS
        assert "stop_detail" in stored

    async def test_no_working_memory_still_stops(self, event_id: str) -> None:
        guard = ConvergenceGuard()
        for _ in range(GLOBAL_MAX_STEPS):
            await guard.record_step(event_id, "llm_call")
        decision = await guard.should_stop(event_id)
        assert decision.stop is True


# --------------------------------------------------------------------------- #
# tool_name + params fingerprint (Protocol path)
# --------------------------------------------------------------------------- #


class TestToolNameParamsSignature:
    async def test_protocol_params_build_full_signature(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        await guard.record_step(
            event_id,
            tool_name="query_ip",
            params={"ip": "1.1.1.1"},
        )
        state = guard.get_state(event_id)
        expected = make_tool_call_signature("query_ip", {"ip": "1.1.1.1"})
        assert state.tool_call_signatures[expected] == 1

    async def test_same_tool_different_params_no_duplicate_stop(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        for i in range(MAX_DUPLICATE_TOOL_CALLS):
            await guard.record_step(
                event_id,
                tool_name="query_ip",
                params={"ip": f"10.0.0.{i}"},
            )
        decision = await guard.should_stop(event_id)
        assert decision.stop is False

    async def test_same_tool_same_params_duplicate_stop(
        self, guard: ConvergenceGuard, event_id: str
    ) -> None:
        params = {"ip": "10.0.0.99"}
        for _ in range(MAX_DUPLICATE_TOOL_CALLS + 1):
            await guard.record_step(event_id, tool_name="query_ip", params=params)
        decision = await guard.should_stop(event_id)
        assert decision.stop is True
        assert decision.reason == StopReason.DUPLICATE_TOOL_CALLS


# --------------------------------------------------------------------------- #
# Protocol compliance — structural subtyping
# --------------------------------------------------------------------------- #


class TestProtocolCompliance:
    def test_matches_convergence_guard_port(self) -> None:
        """Our ConvergenceGuard structurally satisfies ConvergenceGuardPort."""
        from app.tools.executor import ConvergenceGuardPort

        guard = ConvergenceGuard()
        # The Protocol is @runtime_checkable — isinstance must pass.
        assert isinstance(guard, ConvergenceGuardPort), (
            "ConvergenceGuard must satisfy ConvergenceGuardPort from app.tools.executor"
        )

    def test_noop_guard_matches_port(self) -> None:
        """Baseline: the existing NoopConvergenceGuard also matches."""
        from app.tools.executor import (
            ConvergenceGuardPort,
            NoopConvergenceGuard,
        )

        assert isinstance(NoopConvergenceGuard(), ConvergenceGuardPort)

    def test_stop_decision_bool_for_protocol(self) -> None:
        """Protocol expects should_stop → bool.  __bool__ bridges this."""
        sd = StopDecision(stop=True, reason=StopReason.NONE)
        assert sd is not None  # model instance, not None
        assert bool(sd) is True  # used in 'if await guard.should_stop(eid):'
