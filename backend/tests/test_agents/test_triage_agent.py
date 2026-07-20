"""Tests for TriageAgent, RuleBasedFalsePositiveHook, and helper functions (ISSUE-032).

Mock interfaces MUST match the real BoundWorkingMemory signature:
    write(self, event_id: str, key: str, value: Any) -> None
    read(self, event_id: str, key: str) -> Any
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.triage_agent import (
    RuleBasedFalsePositiveHook,
    SEVERITY_RULES,
    TriageAgent,
    _apply_severity_rules,
    _extract_iocs,
    _map_event_type,
    _merge_hint_entities,
)
from app.core.errors import GuardrailViolationError, LLMError
from app.models.agent_io import TriageAgentInput, TriageResult
from app.models.entities import (
    AccountEntity,
    DomainEntity,
    EntitySet,
    HostEntity,
    IPEntity,
    ProcessEntity,
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

    def for_writer(self, writer: str) -> "_MockBoundWorkingMemory":
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
    """

    def __init__(self, writer_name: str = "TriageAgent") -> None:
        self.writer_name = writer_name
        self._store: dict[str, object] = {}

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
                   json_mode=False, response_model=None):
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
    def test_data_exfiltration_is_high(self):
        severity, need = _apply_severity_rules(EventType.DATA_EXFILTRATION)
        assert severity == Severity.HIGH
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
        for level, rules in SEVERITY_RULES.items():
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
        with pytest.raises(Exception):
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
        from pathlib import Path
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
