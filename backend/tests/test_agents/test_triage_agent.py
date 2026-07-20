"""Tests for TriageAgent, RuleBasedFalsePositiveHook, and helper functions (ISSUE-032).

Mock interfaces MUST match the real BoundWorkingMemory signature:
    write(self, event_id: str, key: str, value: Any) -> None
    read(self, event_id: str, key: str) -> Any
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pydantic
import pytest

from app.agents.triage_agent import (
    SEVERITY_RULES,
    RuleBasedFalsePositiveHook,
    TriageAgent,
    _apply_severity_rules,
    _extract_iocs,
    _map_event_type,
    _merge_hint_entities,
)
from app.core.errors import (
    DependencyUnavailableError,
    GuardrailViolationError,
    LLMError,
)
from app.models.agent_io import TriageAgentInput, TriageResult
from app.models.entities import (
    AccountEntity,
    EntitySet,
    HostEntity,
    IPEntity,
)
from app.models.enums import EventType, Severity
from app.services.working_memory import FIELD_OWNERSHIP

# --------------------------------------------------------------------------- #
# Mock-working-memory fixtures — signatures MATCH BoundWorkingMemory exactly
# --------------------------------------------------------------------------- #


class _MockBoundWorkingMemory:
    """Minimal mock matching BoundWorkingMemory interface exactly.

    write(self, event_id, key, value) — three positional args, NO ``writer`` keyword.

    Also exposes ``_memory`` and ``for_writer`` so ``TriageAgent.__init__`` can
    mint a separate FP hook memory (mirroring ``WorkingMemory.for_writer``).
    """

    def __init__(self, writer_name: str = "TriageAgent") -> None:
        self.writer_name = writer_name
        self._store: dict[str, object] = {}
        # ``_memory`` is a back-reference to self so that
        # ``working_memory._memory.for_writer(...)`` works in TriageAgent.__init__.
        self._memory = self

    def for_writer(self, writer: str) -> _MockBoundWorkingMemory:
        """Mint a new mock bound to *writer* (mirrors WorkingMemory.for_writer)."""
        from app.services.working_memory import normalize_writer
        return _MockBoundWorkingMemory(writer_name=normalize_writer(writer))

    async def read(self, event_id: str, key: str) -> object:
        return self._store.get(key)

    async def write(self, event_id: str, key: str, value: object) -> None:
        self._store[key] = value

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        pass

    async def read_scratchpad(self, event_id: str) -> list:
        return []


class _GuardrailMockBoundWorkingMemory:
    """Mock that enforces FIELD_OWNERSHIP like the real WorkingMemory.

    write(self, event_id, key, value) — raises GuardrailViolationError
    when writer_name != FIELD_OWNERSHIP[key].

    Also exposes ``_memory`` and ``for_writer`` so that TriageAgent.__init__
    can mint a separate FP hook memory (mirroring ``WorkingMemory.for_writer``).
    """

    def __init__(self, writer_name: str = "TriageAgent") -> None:
        self.writer_name = writer_name
        self._store: dict[str, object] = {}
        # ``_memory`` is a back-reference to self so that
        # ``working_memory._memory.for_writer(...)`` works in TriageAgent.__init__.
        self._memory = self

    def for_writer(self, writer: str) -> _GuardrailMockBoundWorkingMemory:
        """Mint a new guardrail-enforcing mock bound to *writer*."""
        from app.services.working_memory import normalize_writer
        return _GuardrailMockBoundWorkingMemory(writer_name=normalize_writer(writer))

    async def read(self, event_id: str, key: str) -> object:
        return self._store.get(key)

    async def write(self, event_id: str, key: str, value: object) -> None:
        owner = FIELD_OWNERSHIP.get(key)
        if owner and self.writer_name != owner:
            raise GuardrailViolationError(
                f"writer {self.writer_name!r} is not owner of {key!r} (owner={owner!r})",
                error_code="working_memory_unauthorized_write",
                details={
                    "event_id": event_id,
                    "key": key,
                    "writer": self.writer_name,
                    "owner": owner,
                },
            )
        self._store[key] = value

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        pass

    async def read_scratchpad(self, event_id: str) -> list:
        return []


class _MockLLMClient:
    """Configurable mock LLM client whose .chat() returns a set response."""

    def __init__(self, response: object = None, raise_error: Exception | None = None) -> None:
        self._response = response
        self._raise_error = raise_error
        self.chat_calls: list[dict] = []

    async def chat(self, messages, *, event_id, agent_name, prompt_key,
                   scenario_id=None, temperature=0.3, max_tokens=4096,
                   json_mode=False, response_model=None, timeout=None):
        self.chat_calls.append({
            "messages": messages,
            "event_id": event_id,
            "agent_name": agent_name,
            "prompt_key": prompt_key,
        })
        if self._raise_error:
            raise self._raise_error
        return self._response


# --------------------------------------------------------------------------- #
# Helper to build sample inputs
# --------------------------------------------------------------------------- #


def _make_input(
    event_id: str = "evt-001",
    raw_event_summary: str = "User admin logged in from 192.168.1.1",
    hint_entities: EntitySet | None = None,
) -> TriageAgentInput:
    return TriageAgentInput(
        event_id=event_id,
        raw_event_summary=raw_event_summary,
        hint_entities=hint_entities or EntitySet(),
    )


# --------------------------------------------------------------------------- #
# Tests: _apply_severity_rules
# --------------------------------------------------------------------------- #


class TestApplySeverityRules:
    def test_data_exfiltration_with_external_ip_is_high(self):
        """ISSUE-032: data_exfiltration + external IP → HIGH."""
        severity, need = _apply_severity_rules(
            EventType.DATA_EXFILTRATION,
            alert_text="Data exfiltration to external IP 45.153.12.88",
        )
        assert severity == Severity.HIGH
        assert need is True

    def test_data_exfiltration_without_external_ip_is_medium(self):
        """ISSUE-032: data_exfiltration without external IP → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.DATA_EXFILTRATION,
            alert_text="Data exfiltration to internal server 10.0.0.5",
        )
        assert severity == Severity.MEDIUM
        assert need is True

    def test_data_exfiltration_no_alert_text_is_medium(self):
        """No alert text → cannot verify external IP → MEDIUM."""
        severity, need = _apply_severity_rules(EventType.DATA_EXFILTRATION)
        assert severity == Severity.MEDIUM
        assert need is True

    def test_account_anomaly_is_low(self):
        severity, need = _apply_severity_rules(EventType.ACCOUNT_ANOMALY)
        assert severity == Severity.LOW
        assert need is False

    def test_unlisted_event_type_is_medium(self):
        severity, need = _apply_severity_rules(EventType.OTHER)
        assert severity == Severity.MEDIUM
        assert need is True

    def test_data_exfiltration_with_lateral_is_critical(self):
        severity, need = _apply_severity_rules(
            EventType.DATA_EXFILTRATION, alert_text="lateral movement detected",
        )
        assert severity == Severity.CRITICAL
        assert need is True

    def test_collateral_does_not_trigger_critical(self):
        """Word 'collateral' should NOT match the \blateral\b boundary check."""
        severity, need = _apply_severity_rules(
            EventType.DATA_EXFILTRATION,
            alert_text="collateral damage from data exfiltration to 45.153.12.88",
        )
        assert severity == Severity.HIGH  # external IP → HIGH, not CRITICAL

    def test_bilateral_does_not_trigger_critical(self):
        """Word 'bilateral' should NOT match the \blateral\b boundary check."""
        severity, need = _apply_severity_rules(
            EventType.DATA_EXFILTRATION,
            alert_text="bilateral data transfer to 8.8.8.8",
        )
        assert severity == Severity.HIGH  # external IP → HIGH, not CRITICAL


# --------------------------------------------------------------------------- #
# Tests: _extract_iocs
# --------------------------------------------------------------------------- #


class TestExtractIOCs:
    def test_external_ip_included(self):
        iocs = _extract_iocs("Connection from 45.153.12.88 detected")
        assert "45.153.12.88" in iocs

    def test_internal_ip_excluded(self):
        iocs = _extract_iocs("Login from 10.50.1.10")
        assert "10.50.1.10" not in iocs

    def test_loopback_ip_excluded(self):
        iocs = _extract_iocs("Connection from 127.0.0.1")
        assert "127.0.0.1" not in iocs

    def test_domain_included(self):
        iocs = _extract_iocs("Request to evil.example.com")
        assert "evil.example.com" in iocs

    def test_entity_ip_included_when_external(self):
        entities = EntitySet(ips=[IPEntity(entity_id="ip-1", address="8.8.8.8")])
        iocs = _extract_iocs("some text", entities)
        assert "8.8.8.8" in iocs

    def test_entity_ip_excluded_when_internal(self):
        entities = EntitySet(ips=[IPEntity(entity_id="ip-1", address="192.168.1.1")])
        iocs = _extract_iocs("some text", entities)
        assert "192.168.1.1" not in iocs


# --------------------------------------------------------------------------- #
# Tests: _map_event_type
# --------------------------------------------------------------------------- #


class TestMapEventType:
    def test_valid_raw_type_mapped(self):
        assert _map_event_type("data_exfiltration") == EventType.DATA_EXFILTRATION

    def test_unknown_raw_type_fallback_keyword(self):
        assert _map_event_type(None, "failed to login from 10.0.0.1") == EventType.ACCOUNT_ANOMALY

    def test_no_match_returns_other(self):
        assert _map_event_type(None, "some random text") == EventType.OTHER


# --------------------------------------------------------------------------- #
# Tests: _merge_hint_entities (idempotency + non-mutation)
# --------------------------------------------------------------------------- #


class TestMergeHintEntities:
    def test_merge_preserves_existing(self):
        llm = EntitySet(accounts=[AccountEntity(entity_id="acct-1", username="alice")])
        hint = EntitySet(hosts=[HostEntity(entity_id="host-1", hostname="PC-01")])
        merged = _merge_hint_entities(llm, hint)
        assert len(merged.accounts) == 1
        assert merged.accounts[0].username == "alice"
        assert len(merged.hosts) == 1
        assert merged.hosts[0].hostname == "PC-01"

    def test_merge_skips_duplicate_entity_id(self):
        llm = EntitySet(accounts=[AccountEntity(entity_id="acct-1", username="alice")])
        hint = EntitySet(accounts=[AccountEntity(entity_id="acct-1", username="alice_dup")])
        merged = _merge_hint_entities(llm, hint)
        assert len(merged.accounts) == 1  # duplicate skipped

    def test_merge_does_not_mutate_inputs(self):
        llm = EntitySet(accounts=[AccountEntity(entity_id="acct-1", username="alice")])
        hint = EntitySet()
        original_len = len(llm.accounts)
        _merge_hint_entities(llm, hint)
        assert len(llm.accounts) == original_len  # not mutated

    def test_merge_is_idempotent(self):
        llm = EntitySet(accounts=[AccountEntity(entity_id="acct-1", username="alice")])
        hint = EntitySet(hosts=[HostEntity(entity_id="host-1", hostname="PC-01")])
        first = _merge_hint_entities(llm, hint)
        second = _merge_hint_entities(first, hint)
        assert len(second.accounts) == len(first.accounts)
        assert len(second.hosts) == len(first.hosts)


# --------------------------------------------------------------------------- #
# Tests: RuleBasedFalsePositiveHook — FIELD_OWNERSHIP compliance
# --------------------------------------------------------------------------- #


class TestFPHook:
    @pytest.mark.asyncio
    async def test_write_with_correct_writer_identity_succeeds(self):
        """Hook with writer='FalsePositiveMatcher' can write false_positive_match."""
        fp_memory = _MockBoundWorkingMemory(writer_name="FalsePositiveMatcher")
        agent_memory = _MockBoundWorkingMemory(writer_name="TriageAgent")
        # Pre-populate agent memory with source_snapshot.
        await agent_memory.write("evt-001", "source_snapshot", {
            "scenario": "account_anomaly_fp",
        })

        agent = MagicMock()
        agent.working_memory = agent_memory

        hook = RuleBasedFalsePositiveHook(working_memory=fp_memory)
        await hook(agent, _make_input("evt-001"))

        # The hook's memory should now contain the FP match.
        result = await fp_memory.read("evt-001", "false_positive_match")
        assert result is not None
        assert isinstance(result, dict)
        assert result["matched_rule"] == "ops_change_window_bulk_login"

    @pytest.mark.asyncio
    async def test_write_raises_on_wrong_identity(self):
        """Hook through wrong writer's memory → GuardrailViolationError.

        The agent's memory (used for reading source_snapshot) is a plain mock.
        The hook's memory is a guardrail-enforcing mock bound to "TriageAgent" —
        writing "false_positive_match" via that identity must fail because
        FIELD_OWNERSHIP["false_positive_match"] = "FalsePositiveMatcher".
        """
        # Agent memory: plain mock (no guardrail) so reading source_snapshot works.
        agent_memory = _MockBoundWorkingMemory(writer_name="TriageAgent")
        await agent_memory.write("evt-001", "source_snapshot", {
            "scenario": "account_anomaly_fp",
        })

        # Hook memory: guardrail-enforcing mock bound to WRONG identity.
        hook_memory = _GuardrailMockBoundWorkingMemory(writer_name="TriageAgent")

        agent = MagicMock()
        agent.working_memory = agent_memory

        # Hook receives a memory bound to the WRONG writer — TriageAgent cannot
        # write false_positive_match (owner is FalsePositiveMatcher).
        hook = RuleBasedFalsePositiveHook(working_memory=hook_memory)
        with pytest.raises(GuardrailViolationError) as exc_info:
            await hook(agent, _make_input("evt-001"))
        assert "unauthorized_write" in str(exc_info.value.error_code)

    @pytest.mark.asyncio
    async def test_hook_noop_when_no_memory(self):
        """Hook without memory is a no-op."""
        agent = MagicMock()
        agent.working_memory = _MockBoundWorkingMemory()
        hook = RuleBasedFalsePositiveHook(working_memory=None)
        # Should not raise.
        await hook(agent, _make_input("evt-001"))

    @pytest.mark.asyncio
    async def test_hook_noop_when_no_agent_memory(self):
        """Hook without agent.working_memory is a no-op."""
        agent = MagicMock()
        agent.working_memory = None
        hook = RuleBasedFalsePositiveHook(
            working_memory=_MockBoundWorkingMemory(writer_name="FalsePositiveMatcher"),
        )
        # Should not raise.
        await hook(agent, _make_input("evt-001"))

    @pytest.mark.asyncio
    async def test_hook_noop_when_no_snapshot(self):
        """Hook is no-op when source_snapshot is not a dict."""
        fp_memory = _MockBoundWorkingMemory(writer_name="FalsePositiveMatcher")
        agent_memory = _MockBoundWorkingMemory()
        # source_snapshot not set → returns None.

        agent = MagicMock()
        agent.working_memory = agent_memory

        hook = RuleBasedFalsePositiveHook(working_memory=fp_memory)
        await hook(agent, _make_input("evt-001"))
        # No write should have occurred.
        result = await fp_memory.read("evt-001", "false_positive_match")
        assert result is None

    @pytest.mark.asyncio
    async def test_hook_matches_by_signature_field(self):
        """Hook matches on signature (not scenario) field."""
        fp_memory = _MockBoundWorkingMemory(writer_name="FalsePositiveMatcher")
        agent_memory = _MockBoundWorkingMemory(writer_name="TriageAgent")
        await agent_memory.write("evt-001", "source_snapshot", {
            "signature": "ops_change_window_bulk_login",
        })

        agent = MagicMock()
        agent.working_memory = agent_memory

        hook = RuleBasedFalsePositiveHook(working_memory=fp_memory)
        await hook(agent, _make_input("evt-001"))

        result = await fp_memory.read("evt-001", "false_positive_match")
        assert result is not None
        assert result["signature"] == "ops_change_window_bulk_login"


# --------------------------------------------------------------------------- #
# Tests: TriageAgent — main scenarios
# --------------------------------------------------------------------------- #


class TestTriageAgentBasic:
    @pytest.mark.asyncio
    async def test_no_llm_client_uses_regex_fallback(self):
        """Agent without llm_client → degraded regex extraction."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)
        # No llm_client — should not be provided.
        assert agent.llm_client is None

        input_ = _make_input(
            raw_event_summary=(
                "User zhangsan on host PC-FIN-023 executed powershell.exe and "
                "uploaded data to 203.0.113.88 (evil.example.com)"
            ),
        )
        result = await agent._run(input_)
        assert isinstance(result, TriageResult)
        assert result.degraded is True
        assert result.event_type == EventType.DATA_EXFILTRATION  # 'upload' keyword
        assert len(result.entities.ips) >= 1
        # External IP in IoC list.
        assert "203.0.113.88" in result.ioc_list

    @pytest.mark.asyncio
    async def test_single_login_failure_is_low(self):
        """Single login failure → account_anomaly → low severity."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(
            raw_event_summary="User svc-backup failed to login 1 time from 10.50.1.10",
        )
        result = await agent._run(input_)
        assert result.severity == Severity.LOW
        assert result.need_investigation is False

    @pytest.mark.asyncio
    async def test_writes_triage_result_to_event_context(self):
        """TriageResult is persisted via working_memory.write."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(raw_event_summary="Host compromise detected")
        result = await agent._run(input_)

        stored = await wm.read(input_.event_id, "triage_result")
        assert stored is not None
        assert stored["event_type"] == result.event_type.value

    @pytest.mark.asyncio
    async def test_hint_entities_are_merged(self):
        """Hint entities from input are merged into extracted entities."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        hint = EntitySet(
            accounts=[AccountEntity(entity_id="hint-acct-1", username="pre_known_user")],
        )
        input_ = _make_input(
            raw_event_summary="Suspicious activity detected",
            hint_entities=hint,
        )
        result = await agent._run(input_)
        assert any(e.entity_id == "hint-acct-1" for e in result.entities.accounts)


# --------------------------------------------------------------------------- #
# Tests: TriageAgent — LLM path
# --------------------------------------------------------------------------- #


class TestTriageAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_response_parsed_correctly(self):
        """LLM returns valid TriageLLMResponse → entities extracted."""
        from app.agents.prompts.triage_prompt import TriageLLMResponse
        from app.core.llm.base import LLMResponse

        llm_entities = EntitySet(
            accounts=[AccountEntity(entity_id="acct-1", username="testuser")],
            ips=[IPEntity(entity_id="ip-1", address="8.8.8.8")],
        )
        llm_response = LLMResponse(
            content="",
            parsed=TriageLLMResponse(
                event_type=EventType.MALICIOUS_PROCESS,
                entities=llm_entities,
                reasoning="Test reasoning",
            ),
            model_name="mock",
        )
        llm_client = _MockLLMClient(response=llm_response)

        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(llm_client=llm_client, working_memory=wm)

        input_ = _make_input(raw_event_summary="powershell.exe executed")
        result = await agent._run(input_)
        assert result.event_type == EventType.MALICIOUS_PROCESS
        assert len(result.entities.accounts) == 1
        assert result.entities.accounts[0].username == "testuser"
        assert "Test reasoning" in result.reasoning
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_llm_chat_raises_llm_error_triggers_fallback(self):
        """LLM chat() raises LLMError → degraded regex fallback."""
        llm_client = _MockLLMClient(raise_error=LLMError("LLM unavailable"))

        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(llm_client=llm_client, working_memory=wm)

        input_ = _make_input(
            raw_event_summary="User admin connected to 203.0.113.88",
        )
        result = await agent._run(input_)
        assert result.degraded is True
        assert "203.0.113.88" in result.ioc_list  # regex extracted IP

    @pytest.mark.asyncio
    async def test_empty_source_snapshot_no_crash(self):
        """Agent handles missing source_snapshot gracefully."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)
        # source_snapshot is not written → read returns None.

        input_ = _make_input(raw_event_summary="Some alert")
        result = await agent._run(input_)
        assert result.event_type == EventType.OTHER  # no keywords matched

    @pytest.mark.asyncio
    async def test_empty_alert_with_llm_client_returns_empty_entities(self):
        """Empty alert + LLM client present → empty EntitySet, no crash.

        When ``llm_client`` is configured (not None) but the alert text is
        empty, ``build_triage_messages("")`` would raise ``ValueError`` which
        is not a ``ShadowTraceError``.  The agent must short-circuit before
        the LLM call and return an empty ``EntitySet`` with ``degraded=False``
        (LLM did not fail — there is just nothing to extract).
        """
        from app.agents.prompts.triage_prompt import TriageLLMResponse
        from app.core.llm.base import LLMResponse

        llm_response = LLMResponse(
            content="",
            parsed=TriageLLMResponse(
                event_type=EventType.OTHER,
                entities=EntitySet(),
                reasoning="",
            ),
            model_name="mock",
        )
        llm_client = _MockLLMClient(response=llm_response)

        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(llm_client=llm_client, working_memory=wm)

        input_ = _make_input(raw_event_summary="")
        result = await agent._run(input_)

        # Must not crash; should return empty EntitySet, degraded=False.
        assert isinstance(result, TriageResult)
        assert result.entities == EntitySet()
        assert result.degraded is False
        # LLM should NOT have been called (empty input short-circuit).
        assert len(llm_client.chat_calls) == 0


# --------------------------------------------------------------------------- #
# Tests: TriageAgent — degraded scenarios
# --------------------------------------------------------------------------- #


class TestTriageAgentDegraded:
    @pytest.mark.asyncio
    async def test_no_working_memory_no_crash(self):
        """Agent without working_memory still produces a result."""
        agent = TriageAgent()  # no working_memory at all
        input_ = _make_input(raw_event_summary="Test alert")
        result = await agent._run(input_)
        assert isinstance(result, TriageResult)
        assert result.event_type is not None

    @pytest.mark.asyncio
    async def test_regex_fallback_extracts_accounts(self):
        """Regex fallback extracts account/usernames from alert text."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(
            raw_event_summary="Account jdoe failed authentication on host WEB-SERVER-01",
        )
        result = await agent._run(input_)
        # Should extract at least hostname "WEB-SERVER-01"
        assert len(result.entities.hosts) >= 1
        assert any("WEB" in (h.hostname or "") for h in result.entities.hosts)

    @pytest.mark.asyncio
    async def test_triage_result_has_agent_trace(self):
        """TriageResult is complete with all required fields."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(raw_event_summary="User admin logged in from 192.168.1.1")
        result = await agent._run(input_)
        assert result.event_type is not None
        assert result.severity is not None
        assert isinstance(result.need_investigation, bool)
        assert isinstance(result.entities, EntitySet)
        assert isinstance(result.ioc_list, list)
        assert isinstance(result.degraded, bool)


# --------------------------------------------------------------------------- #
# Tests: SEVERITY_RULES structure
# --------------------------------------------------------------------------- #


class TestSeverityRules:
    def test_rules_have_required_keys(self):
        """SEVERITY_RULES must have high, critical, low keys."""
        assert "high" in SEVERITY_RULES
        assert "critical" in SEVERITY_RULES
        assert "low" in SEVERITY_RULES

    def test_critical_severity_can_be_produced(self):
        """SEVERITY_RULES critical path is exercised."""
        severity, need = _apply_severity_rules(
            EventType.DATA_EXFILTRATION,
            alert_text="lateral movement from 10.0.0.1 detected",
        )
        assert severity == Severity.CRITICAL
        assert need is True

    def test_severity_rules_are_tuples(self):
        """Each rule is a list of (key, value) tuples."""
        for _level, rules in SEVERITY_RULES.items():
            assert isinstance(rules, list)
            for rule in rules:
                assert isinstance(rule, tuple)
                assert len(rule) == 2
                assert isinstance(rule[0], str)
                assert isinstance(rule[1], str)


# --------------------------------------------------------------------------- #
# Tests: TriageAgent — pre_triage_hooks alias
# --------------------------------------------------------------------------- #


class TestTriageAgentHooks:
    def test_pre_triage_hooks_is_alias_of_pre_hooks(self):
        agent = TriageAgent()
        assert agent.pre_triage_hooks is agent.pre_hooks
        assert agent.post_triage_hooks is agent.post_hooks

    def test_fp_hook_installed_when_working_memory_provided(self):
        """When working_memory is provided, FP hook is auto-installed."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        # Give the mock a _memory attribute so for_writer can be called.
        wm._memory = MagicMock()
        wm._memory.for_writer.return_value = _MockBoundWorkingMemory(
            writer_name="FalsePositiveMatcher",
        )

        agent = TriageAgent(working_memory=wm)
        assert len(agent.pre_triage_hooks) == 1
        assert isinstance(agent.pre_triage_hooks[0], RuleBasedFalsePositiveHook)


# --------------------------------------------------------------------------- #
# Tests: TriageAgentInput / TriageResult contract
# --------------------------------------------------------------------------- #


class TestTriageAgentContract:
    def test_agent_name_is_triage_agent(self):
        assert TriageAgent.agent_name == "triage_agent"

    def test_agent_name_in_io_mapping(self):
        from app.models.agent_io import AGENT_INPUT_BY_NAME
        assert AGENT_INPUT_BY_NAME.get("triage_agent") is TriageAgentInput

    def test_triage_result_extra_forbid(self):
        """TriageResult rejects extra fields."""
        with pytest.raises(pydantic.ValidationError):
            TriageResult.model_validate({
                "event_type": "other",
                "severity": "low",
                "need_investigation": False,
                "unknown_field": "should_reject",
            })


# --------------------------------------------------------------------------- #
# Tests: Golden response compatibility
# --------------------------------------------------------------------------- #

class TestGoldenResponse:
    def test_golden_response_parses_as_triage_llm_response(self):
        """The golden default.json must validate as TriageLLMResponse."""
        import json

        from app.agents.prompts.triage_prompt import TriageLLMResponse
        from app.core.llm.base import default_golden_root

        golden_path = default_golden_root() / "triage_extract" / "default.json"
        assert golden_path.is_file(), f"Golden file missing: {golden_path}"

        payload = json.loads(golden_path.read_text("utf-8"))
        content = payload.get("content", payload)
        assert isinstance(content, dict), "Golden content must be a dict"

        # Should parse without error.
        parsed = TriageLLMResponse.model_validate(content)
        assert parsed.event_type == EventType.OTHER
        assert isinstance(parsed.entities, EntitySet)
        assert isinstance(parsed.reasoning, str)


# --------------------------------------------------------------------------- #
# Mock WM that raises on write (for transient-failure tests)
# --------------------------------------------------------------------------- #


class _FailingWriteMockWM:
    """Mock WM that raises on write for a specific key."""

    def __init__(
        self,
        writer_name: str = "TriageAgent",
        *,
        fail_key: str | None = None,
        fail_error: Exception = DependencyUnavailableError("wm unavailable"),
    ) -> None:
        self.writer_name = writer_name
        self._store: dict[str, object] = {}
        self._fail_key = fail_key
        self._fail_error = fail_error
        self._memory = self

    def for_writer(self, writer: str) -> _FailingWriteMockWM:
        from app.services.working_memory import normalize_writer
        return _FailingWriteMockWM(
            writer_name=normalize_writer(writer),
            fail_key=self._fail_key,
            fail_error=self._fail_error,
        )

    async def read(self, event_id: str, key: str) -> object:
        return self._store.get(key)

    async def write(self, event_id: str, key: str, value: object) -> None:
        if self._fail_key is not None and key == self._fail_key:
            raise self._fail_error
        self._store[key] = value

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        pass

    async def read_scratchpad(self, event_id: str) -> list:
        return []


# --------------------------------------------------------------------------- #
# Tests: Should-Fix #1 — transient write failure marks degraded
# --------------------------------------------------------------------------- #


class TestWriteTriageResultTransientFailure:
    @pytest.mark.asyncio
    async def test_transient_write_failure_marks_degraded(self):
        """When wm.write raises DependencyUnavailableError, result.degraded=True."""
        wm = _FailingWriteMockWM(
            writer_name="TriageAgent",
            fail_key="triage_result",
            fail_error=DependencyUnavailableError("Redis down"),
        )
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(
            raw_event_summary="Host compromise detected on server-01",
        )
        result = await agent._run(input_)
        assert result.degraded is True
        assert "triage_result persistence failed" in result.reasoning
        assert "working memory unavailable" in result.reasoning

    @pytest.mark.asyncio
    async def test_retryable_shadowtrace_error_marks_degraded(self):
        """When wm.write raises a retryable ShadowTraceError, result.degraded=True."""
        from app.core.errors import ShadowTraceError

        wm = _FailingWriteMockWM(
            writer_name="TriageAgent",
            fail_key="triage_result",
            fail_error=ShadowTraceError(
                "DB timeout",
                error_code="db_timeout",
                retryable=True,
            ),
        )
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(raw_event_summary="Test alert")
        result = await agent._run(input_)
        assert result.degraded is True
        assert "triage_result persistence failed: db_timeout" in result.reasoning

    @pytest.mark.asyncio
    async def test_non_retryable_shadowtrace_error_raises(self):
        """Non-retryable ShadowTraceError propagates, not swallowed."""
        from app.core.errors import ShadowTraceError

        wm = _FailingWriteMockWM(
            writer_name="TriageAgent",
            fail_key="triage_result",
            fail_error=ShadowTraceError(
                "Schema mismatch",
                error_code="schema_error",
                retryable=False,
            ),
        )
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(raw_event_summary="Test alert")
        with pytest.raises(ShadowTraceError) as exc_info:
            await agent._run(input_)
        assert exc_info.value.error_code == "schema_error"


# --------------------------------------------------------------------------- #
# Tests: Should-Fix #3 — account_anomaly keyword-based upgrade
# --------------------------------------------------------------------------- #


class TestAccountAnomalyUpgrade:
    def test_single_login_failure_is_low(self):
        """Plain single login failure → LOW (unchanged behavior)."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="User svc-backup failed to login 1 time",
        )
        assert severity == Severity.LOW
        assert need is False

    def test_bulk_account_anomaly_is_medium(self):
        """Bulk account creation → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="Bulk account creation detected: 50 new users in 5 minutes",
        )
        assert severity == Severity.MEDIUM
        assert need is True

    def test_mass_account_anomaly_is_medium(self):
        """Mass login failures → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="Mass login failures from multiple IPs detected",
        )
        assert severity == Severity.MEDIUM
        assert need is True

    def test_privilege_escalation_is_medium(self):
        """Privilege escalation → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="Privilege escalation detected: user granted admin role",
        )
        assert severity == Severity.MEDIUM
        assert need is True

    def test_brute_force_is_medium(self):
        """Brute force attack → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="Brute force attack on SSH port detected",
        )
        assert severity == Severity.MEDIUM
        assert need is True

    def test_password_spray_is_medium(self):
        """Password spray → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="Password spray attack targeting O365 accounts",
        )
        assert severity == Severity.MEDIUM
        assert need is True

    def test_chinese_geo_anomaly_is_medium(self):
        """Chinese geo-anomaly description → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="检测到账号地域异常登录行为",
        )
        assert severity == Severity.MEDIUM
        assert need is True

    def test_impossible_travel_is_medium(self):
        """Impossible travel → MEDIUM."""
        severity, need = _apply_severity_rules(
            EventType.ACCOUNT_ANOMALY,
            alert_text="Impossible travel: login from Beijing then New York in 10 minutes",
        )
        assert severity == Severity.MEDIUM
        assert need is True


# --------------------------------------------------------------------------- #
# Tests: Should-Fix #4 — agent_trace recording
# --------------------------------------------------------------------------- #


class TestTriageAgentTrace:
    @pytest.mark.asyncio
    async def test_triage_agent_records_agent_trace(self):
        """TriageAgent.execute() calls trace_service.log_trace."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        trace_service = MagicMock()
        trace_service.log_trace = MagicMock()

        agent = TriageAgent(working_memory=wm, trace_service=trace_service)

        input_ = _make_input(raw_event_summary="Host compromise detected")
        await agent.execute(input_)

        # trace_service.log_trace must have been called once.
        trace_service.log_trace.assert_called_once()
        call_kwargs = trace_service.log_trace.call_args.kwargs
        assert call_kwargs["event_id"] == input_.event_id
        assert call_kwargs["agent_name"] == "triage_agent"
        assert call_kwargs["status"] == "completed"

    @pytest.mark.asyncio
    async def test_no_trace_service_no_crash(self):
        """Agent without trace_service still executes without error."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)  # no trace_service

        input_ = _make_input(raw_event_summary="Test alert")
        result = await agent.execute(input_)
        assert isinstance(result, TriageResult)


# --------------------------------------------------------------------------- #
# Tests: Boundary / edge cases (from review recommendations)
# --------------------------------------------------------------------------- #


class TestTriageAgentBoundaries:
    @pytest.mark.asyncio
    async def test_empty_alert_returns_other_event_type(self):
        """Empty alert string → OTHER event type, no crash."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(raw_event_summary="")
        result = await agent._run(input_)
        assert result.event_type == EventType.OTHER
        assert isinstance(result.entities, EntitySet)

    @pytest.mark.asyncio
    async def test_very_long_alert_does_not_crash(self):
        """Extremely long alert text (>10000 chars) does not crash the agent."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        long_text = "Event: " + "suspicious activity " * 2000  # ~24000 chars
        input_ = _make_input(raw_event_summary=long_text)
        result = await agent._run(input_)
        assert isinstance(result, TriageResult)
        assert result.event_type is not None

    @pytest.mark.asyncio
    async def test_chinese_alert_entity_extraction(self):
        """All-Chinese alert with no English keywords → extracts via regex."""
        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(working_memory=wm)

        # NOTE: IP regex uses \b which requires an ASCII word boundary;
        # Chinese characters are \w in Python 3 Unicode mode, so the IP
        # must be whitespace-separated from adjacent CJK text.
        input_ = _make_input(
            raw_event_summary=(
                "用户张三从主机 PC-FIN-023 执行了 powershell.exe "
                "并上传数据到 203.0.113.88 (evil.example.com)"
            ),
        )
        result = await agent._run(input_)
        assert isinstance(result, TriageResult)
        # Regex should still extract the external IP and domain.
        assert "203.0.113.88" in result.ioc_list
        assert "evil.example.com" in result.ioc_list

    def test_data_exfiltration_without_external_ip_is_medium_severity(self):
        """Data exfiltration WITHOUT external IP → MEDIUM (per ISSUE-032 spec)."""
        severity, need = _apply_severity_rules(
            EventType.DATA_EXFILTRATION,
            alert_text="Data exfiltration to internal server",
        )
        assert severity == Severity.MEDIUM
        assert need is True


# --------------------------------------------------------------------------- #
# Tests: Should-Fix #2 — _read_source_snapshot GuardrailViolationError
# --------------------------------------------------------------------------- #


class TestReadSourceSnapshotGuardrail:
    @pytest.mark.asyncio
    async def test_read_source_snapshot_guardrail_raises(self):
        """GuardrailViolationError from wm.read must propagate, not be swallowed."""
        from app.core.errors import GuardrailViolationError

        class _GuardrailFailingWM:
            """WM whose read() always raises GuardrailViolationError."""

            def __init__(self) -> None:
                self.writer_name = "TriageAgent"
                self._memory = self

            def for_writer(self, writer: str) -> _GuardrailFailingWM:
                return _GuardrailFailingWM()

            async def read(self, event_id: str, key: str) -> object:
                raise GuardrailViolationError(
                    "FIELD_OWNERSHIP: source_snapshot missing TriageAgent",
                    error_code="working_memory_unauthorized_read",
                )

            async def write(self, event_id: str, key: str, value: object) -> None:
                pass

        wm = _GuardrailFailingWM()
        agent = TriageAgent(working_memory=wm)

        with pytest.raises(GuardrailViolationError) as exc_info:
            await agent._read_source_snapshot("evt-001")
        assert "unauthorized_read" in str(exc_info.value.error_code)


# --------------------------------------------------------------------------- #
# Tests: Should-Fix #3 — LLM fallback model not degraded
# --------------------------------------------------------------------------- #


class TestLLMFallbackModel:
    @pytest.mark.asyncio
    async def test_llm_fallback_model_not_degraded(self):
        """LLM fallback model success → degraded=False (not regex fallback)."""
        from app.agents.prompts.triage_prompt import TriageLLMResponse
        from app.core.llm.base import LLMResponse

        llm_entities = EntitySet(
            accounts=[AccountEntity(entity_id="acct-1", username="fallback_user")],
        )
        # Simulate a response from a fallback model (fallback_level=1).
        llm_response = LLMResponse(
            content="",
            parsed=TriageLLMResponse(
                event_type=EventType.ACCOUNT_ANOMALY,
                entities=llm_entities,
                reasoning="Fallback model reasoning",
            ),
            model_name="fallback-model",
            fallback_level=1,  # primary unavailable, fallback succeeded
        )
        llm_client = _MockLLMClient(response=llm_response)

        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(llm_client=llm_client, working_memory=wm)

        input_ = _make_input(raw_event_summary="User svc-backup failed to login")
        result = await agent._run(input_)
        # Fallback model succeeded — NOT degraded (only regex fallback is degraded).
        assert result.degraded is False
        assert "Fallback model reasoning" in result.reasoning


# --------------------------------------------------------------------------- #
# Tests: Should-Fix #4 — write failure records degraded flag
# --------------------------------------------------------------------------- #


class TestWriteFailureDegradedFlag:
    @pytest.mark.asyncio
    async def test_write_failure_records_degraded_flag(self):
        """When triage_result write fails, a triage_degraded flag is persisted."""
        wm = _FailingWriteMockWM(
            writer_name="TriageAgent",
            fail_key="triage_result",
            fail_error=DependencyUnavailableError("Redis down"),
        )
        agent = TriageAgent(working_memory=wm)

        input_ = _make_input(raw_event_summary="Test alert")
        result = await agent._run(input_)
        assert result.degraded is True

        # The triage_degraded flag should have been written (best-effort).
        degraded_flag = await wm.read(input_.event_id, "triage_degraded")
        assert degraded_flag is not None
        assert degraded_flag["degraded"] is True
        assert "triage_result persistence failed" in degraded_flag["reason"]


# --------------------------------------------------------------------------- #
# Tests: Nit #4 — build_triage_messages input validation
# --------------------------------------------------------------------------- #


class TestBuildTriageMessages:
    def test_build_triage_messages_empty_alert_raises(self):
        """Empty alert_text must raise ValueError."""
        from app.agents.prompts.triage_prompt import build_triage_messages

        with pytest.raises(ValueError, match="non-empty string"):
            build_triage_messages("")

    def test_build_triage_messages_none_alert_raises(self):
        """None alert_text must raise ValueError."""
        from app.agents.prompts.triage_prompt import build_triage_messages

        with pytest.raises(ValueError, match="non-empty string"):
            build_triage_messages(None)  # type: ignore[arg-type]

    def test_build_triage_messages_valid_input(self):
        """Valid input returns two messages (system + user)."""
        from app.agents.prompts.triage_prompt import build_triage_messages

        messages = build_triage_messages("Test alert text")
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "Test alert text" in messages[1].content


# --------------------------------------------------------------------------- #
# Tests: Nit #8 — Chinese alert account extraction
# --------------------------------------------------------------------------- #


class TestChineseAlertAccountExtraction:
    def test_chinese_account_extraction(self):
        """Chinese keyword '账号' triggers account extraction."""
        from app.agents.rules.entity_extraction_rules import extract_entities_regex

        result = extract_entities_regex("账号 zhangsan 从主机登录")
        assert "zhangsan" in result.accounts

    def test_chinese_user_extraction(self):
        """Chinese keyword '用户' triggers account extraction."""
        from app.agents.rules.entity_extraction_rules import extract_entities_regex

        result = extract_entities_regex("用户 lisi 执行了敏感操作")
        assert "lisi" in result.accounts

    def test_chinese_username_extraction(self):
        """Chinese keyword '用户名' triggers account extraction."""
        from app.agents.rules.entity_extraction_rules import extract_entities_regex

        result = extract_entities_regex("用户名 wangwu 登录失败")
        assert "wangwu" in result.accounts


# --------------------------------------------------------------------------- #
# Tests: _map_event_type coverage
# --------------------------------------------------------------------------- #


class TestMapEventTypeCoverage:
    def test_map_event_type_all_eight_exact(self):
        """All 8 EventType enum values are reachable via exact match."""
        for event_type in EventType:
            result = _map_event_type(event_type.value)
            assert result == event_type

    def test_map_event_type_fallback_priority_data_exfiltration(self):
        """'exfil'/'upload' keywords take priority (checked first)."""
        # 'upload' keyword triggers DATA_EXFILTRATION even with other keywords.
        result = _map_event_type(None, "upload and process executed with escalation")
        assert result == EventType.DATA_EXFILTRATION

    def test_map_event_type_fallback_priority_login(self):
        """Login failure keywords fire before 'process' keyword."""
        result = _map_event_type(None, "failed to login process alert")
        assert result == EventType.ACCOUNT_ANOMALY

    def test_map_event_type_escalation_matches(self):
        """Full word 'escalation' matches insider_threat (not partial 'escalat')."""
        result = _map_event_type(None, "privilege escalation detected")
        assert result == EventType.INSIDER_THREAT

    def test_map_event_type_de_escalation_does_not_match(self):
        """'de-escalation' does NOT match \bescalation\b (word-boundary check)."""
        result = _map_event_type(None, "de-escalation procedure completed")
        assert result == EventType.OTHER


# --------------------------------------------------------------------------- #
# Tests: LLM response edge cases
# --------------------------------------------------------------------------- #


class TestLLMResponseEdgeCases:
    @pytest.mark.asyncio
    async def test_llm_parsed_wrong_type_triggers_fallback(self):
        """LLM response.parsed is a valid Pydantic model but wrong type → regex fallback."""
        from pydantic import BaseModel

        from app.core.llm.base import LLMResponse

        class _WrongModel(BaseModel):
            some_field: str = "unexpected"

        llm_response = LLMResponse(
            content="",
            parsed=_WrongModel(some_field="unexpected"),
            model_name="mock",
        )
        llm_client = _MockLLMClient(response=llm_response)

        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(llm_client=llm_client, working_memory=wm)

        input_ = _make_input(
            raw_event_summary="User admin connected to 203.0.113.88",
        )
        result = await agent._run(input_)
        # Should have fallen back to regex.
        assert result.degraded is True
        assert "203.0.113.88" in result.ioc_list

    @pytest.mark.asyncio
    async def test_llm_response_parsed_none_triggers_fallback(self):
        """LLM response.parsed is None → regex fallback."""
        from app.core.llm.base import LLMResponse

        llm_response = LLMResponse(
            content='{"event_type":"data_exfiltration"}',
            parsed=None,  # JSON parse failed
            model_name="mock",
        )
        llm_client = _MockLLMClient(response=llm_response)

        wm = _MockBoundWorkingMemory(writer_name="TriageAgent")
        agent = TriageAgent(llm_client=llm_client, working_memory=wm)

        input_ = _make_input(
            raw_event_summary="Connection from 45.153.12.88 to evil.example.com",
        )
        result = await agent._run(input_)
        assert result.degraded is True
        assert "45.153.12.88" in result.ioc_list


# --------------------------------------------------------------------------- #
# Tests: ReDoS resistance
# --------------------------------------------------------------------------- #


class TestReDoSResistance:
    def test_regex_no_redos_on_long_input(self):
        """Extremely long alert text with edge-case patterns does not hang."""
        import time

        from app.agents.rules.entity_extraction_rules import extract_entities_regex

        # Simulate a very long input with repetitive near-match patterns.
        long_text = "a." * 5000 + " " + "b-" * 5000 + " final.exe"
        start = time.monotonic()
        result = extract_entities_regex(long_text)
        elapsed = time.monotonic() - start
        # Should complete in well under 1 second (catastrophic backtracking → >10s).
        assert elapsed < 1.0, f"Regex extraction took {elapsed:.1f}s — possible ReDoS"
        assert "final.exe" in result.processes
