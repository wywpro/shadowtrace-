from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.redis_client import RedisClient
from app.models.enums import (
    ExecutionJobStatus,
    ExecutionOwner,
    TargetExecutionStatus,
    ToolCategory,
)
from app.models.execution import ActionExecutionJob
from app.models.tool_meta import ExecutionChannel, ToolResult, ToolResultStatus
from app.providers.tools.mock_provider import (
    MockToolProvider,
    MockToolProviderConfig,
    ToolExecutionContext,
    bind_mock_tool_provider,
    bind_tool_execution_context,
)
from app.tools.mock_state import (
    MOCK_TOOL_STATE_KEY,
    MockEnvironmentState,
    MockObservationRecord,
)
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.specs import ROLLBACK_SOURCE_MAP, ROLLBACK_TOOL_METAS
from app.tools.verify._common import MockVerificationRuntime

ROLLBACK_NAMES = frozenset(meta.tool_name for meta in ROLLBACK_TOOL_METAS)


@pytest.fixture
async def state() -> MockEnvironmentState:
    store = MockEnvironmentState.in_memory()
    await store.clear_all()
    return store


def _context(suffix: str, *, idempotency_key: str | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(
        event_id=f"evt-20260714-{suffix:0>8}",
        action_id=f"act-{suffix:0>8}",
        idempotency_key=idempotency_key or f"rollback-idem-{suffix}",
    )


def _target(target_type: str, target: str, **parameters: Any) -> dict[str, Any]:
    return {
        "target_type": target_type,
        "target": target,
        "parameters": parameters,
    }


async def _run_async_tool(
    provider: MockToolProvider,
    tool_name: str,
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> ActionExecutionJob:
    queued = ActionExecutionJob.model_validate(
        await provider.execute(tool_name, params, context=context)
    )
    assert queued.status is ExecutionJobStatus.QUEUED
    return await provider.run_job(queued.job_id)


def _assert_rollback_result(
    completed: ActionExecutionJob,
    *,
    rolled_back: bool,
    warning: str | None,
) -> None:
    assert completed.status is ExecutionJobStatus.SUCCESS
    assert completed.raw_result["rolled_back"] is rolled_back
    assert completed.raw_result["warning"] == warning
    target = completed.target_results[0]
    assert target.status is TargetExecutionStatus.SUCCESS
    assert target.raw_result["rolled_back"] is rolled_back
    assert target.raw_result["warning"] == warning
    if rolled_back:
        assert datetime.fromisoformat(target.raw_result["rolled_back_at"]).tzinfo is not None
    else:
        assert target.raw_result["rolled_back_at"] is None


def test_registry_and_manifest_publish_all_baseline_rollback_tools(
    state: MockEnvironmentState,
) -> None:
    registry = ToolRegistry()
    discovered = set(registry.auto_discover())
    provider = MockToolProvider(state)
    provider.register_bindings(registry)
    manifest = provider.capability_manifest()

    assert ROLLBACK_NAMES.issubset(discovered)
    assert ROLLBACK_NAMES.issubset(set(manifest.allowed_operations))
    registered_names = {
        entry.tool_meta.tool_name for entry in registry.list_registered_tools(ToolCategory.ROLLBACK)
    }
    assert ROLLBACK_NAMES.issubset(registered_names)
    for tool_name in ROLLBACK_NAMES:
        assert registry.get_tool(tool_name).tool_impl is not None
        direct = registry.resolve_binding(tool_name, ExecutionOwner.DIRECT_TOOL, [])
        managed = registry.resolve_binding(tool_name, ExecutionOwner.XDR_MANAGED, [])
        assert direct.provider_name == "mock_tool_provider"
        assert direct.execution_channel is ExecutionChannel.TOOL_PROVIDER
        assert managed.provider_name == "mock_xdr"
        assert managed.execution_channel is ExecutionChannel.DISPOSITION_ADAPTER
        assert ROLLBACK_SOURCE_MAP[tool_name]


def test_process_registry_registers_rollback_implementations_and_bindings() -> None:
    for tool_name in ROLLBACK_NAMES:
        assert tool_registry.get_tool(tool_name).tool_impl is not None
        binding = tool_registry.resolve_binding(
            tool_name,
            ExecutionOwner.DIRECT_TOOL,
            [],
        )
        assert binding.provider_name == "mock_tool_provider"


@pytest.mark.asyncio
async def test_provider_rejects_invented_rollback_for_non_reversible_action(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)

    result = ToolResult.model_validate(
        await provider.execute(
            "unblock_process",
            _target("process", "sha256:not-reversible"),
            context=_context("manual-escalation"),
        )
    )

    assert result.status is ToolResultStatus.UNSUPPORTED
    assert result.provider_code == "capability_missing"
    assert result.data["manual_escalation_required"] is True
    assert result.data["reason"] == {
        "code": "rollback_mapping_missing",
        "tool_name": "unblock_process",
        "provider_name": "mock_tool_provider",
    }
    assert "unblock_process" not in provider.capability_manifest().allowed_operations
    assert await state.list_namespace("jobs") == {}


@pytest.mark.asyncio
async def test_provider_unsupported_rollback_requires_structured_manual_escalation(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(disabled_tools={"unblock_ip"}),
    )

    result = ToolResult.model_validate(
        await provider.execute(
            "unblock_ip",
            _target("ip", "203.0.113.250"),
            context=_context("unsupported-rollback"),
        )
    )

    assert result.status is ToolResultStatus.UNSUPPORTED
    assert result.data["manual_escalation_required"] is True
    assert result.data["reason"] == {
        "code": "provider_rollback_unsupported",
        "tool_name": "unblock_ip",
        "provider_name": "mock_tool_provider",
        "source_tool_name": "block_ip",
    }
    assert "warning" not in result.data
    assert await state.list_namespace("jobs") == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_tool", "rollback_tool", "target_type", "target", "namespace"),
    [
        ("block_ip", "unblock_ip", "ip", "203.0.113.22", "blocked_ips"),
        (
            "block_domain",
            "unblock_domain",
            "domain",
            "malicious.example",
            "blocked_domains",
        ),
        (
            "isolate_host",
            "cancel_host_isolation",
            "host",
            "host-22",
            "isolated_hosts",
        ),
        (
            "quarantine_file",
            "restore_file",
            "file",
            "sha256:issue022",
            "quarantined_files",
        ),
        (
            "disable_account",
            "restore_account",
            "account",
            "analyst@example.test",
            "accounts",
        ),
    ],
)
async def test_each_entity_rollback_is_async_and_preserves_history(
    state: MockEnvironmentState,
    response_tool: str,
    rollback_tool: str,
    target_type: str,
    target: str,
    namespace: str,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    response = await _run_async_tool(
        provider,
        response_tool,
        _target(target_type, target),
        _context(f"response-{rollback_tool}"),
    )
    original = await state.get_state(namespace, target)
    assert response.status is ExecutionJobStatus.SUCCESS
    assert isinstance(original, dict)

    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            rollback_tool,
            _target(target_type, target),
            context=_context(f"rollback-{rollback_tool}"),
        )
    )
    assert queued.status is ExecutionJobStatus.QUEUED
    assert await state.get_state(namespace, target) == original

    completed = await provider.run_job(queued.job_id)
    _assert_rollback_result(completed, rolled_back=True, warning=None)
    assert await state.get_state(namespace, target) is None
    history = await state.list_namespace("rollback_history")
    assert len(history) == 1
    audit = next(iter(history.values()))
    assert audit["rollback_tool_name"] == rollback_tool
    assert audit["source_tool_name"] == response_tool
    assert audit["original_record"] == original
    assert audit["job_id"] == queued.job_id


@pytest.mark.asyncio
async def test_block_verify_true_unblock_verify_false_full_chain(
    state: MockEnvironmentState,
) -> None:
    target = "198.51.100.22"
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    verifier = MockVerificationRuntime(state, wait_timeout_ms=20, poll_interval_ms=1)

    blocked = await _run_async_tool(
        provider,
        "block_ip",
        _target("ip", target),
        _context("block-chain"),
    )
    before = ToolResult.model_validate(
        await verifier.execute(
            "check_ip_block_status",
            _target("ip", target, job_id=blocked.job_id),
        )
    )
    assert before.data["is_verified"] is True

    unblocked = await _run_async_tool(
        provider,
        "unblock_ip",
        _target("ip", target),
        _context("unblock-chain"),
    )
    after = ToolResult.model_validate(
        await verifier.execute("check_ip_block_status", _target("ip", target))
    )
    traffic = ToolResult.model_validate(
        await verifier.execute("check_traffic_drop", _target("ip", target))
    )

    _assert_rollback_result(unblocked, rolled_back=True, warning=None)
    assert after.data["is_verified"] is False
    assert after.data["detail"] == "observed_status:allowed"
    assert traffic.data["is_verified"] is False


@pytest.mark.asyncio
async def test_missing_target_is_a_successful_business_result_without_history(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)

    completed = await _run_async_tool(
        provider,
        "unblock_ip",
        _target("ip", "192.0.2.222"),
        _context("missing"),
    )

    _assert_rollback_result(
        completed,
        rolled_back=False,
        warning="target_not_found",
    )
    assert completed.target_results[0].code == "target_not_found"
    assert await state.list_namespace("rollback_history") == {}


@pytest.mark.asyncio
async def test_same_idempotency_key_replays_one_job_and_one_history_record(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    target = "203.0.113.122"
    params = _target("ip", target)
    await _run_async_tool(provider, "block_ip", params, _context("idem-source"))
    rollback_context = _context("idem-rollback", idempotency_key="one-rollback")

    first = await _run_async_tool(provider, "unblock_ip", params, rollback_context)
    replay = ActionExecutionJob.model_validate(
        await provider.execute("unblock_ip", params, context=rollback_context)
    )

    assert replay == first
    _assert_rollback_result(replay, rolled_back=True, warning=None)
    assert len(await state.list_namespace("rollback_history")) == 1


@pytest.mark.asyncio
async def test_concurrent_distinct_rollbacks_have_one_effect_and_one_audit_record(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    target = "203.0.113.223"
    params = _target("ip", target)
    await _run_async_tool(provider, "block_ip", params, _context("race-source"))
    queued = [
        ActionExecutionJob.model_validate(
            await provider.execute(
                "unblock_ip",
                params,
                context=_context(f"race-{index}"),
            )
        )
        for index in range(2)
    ]

    completed = await asyncio.gather(*(provider.run_job(job.job_id) for job in queued))

    assert all(job.status is ExecutionJobStatus.SUCCESS for job in completed)
    assert sorted(job.raw_result["rolled_back"] for job in completed) == [False, True]
    assert len(await state.list_namespace("rollback_history")) == 1


@pytest.mark.asyncio
async def test_queued_rollback_cannot_delete_a_newer_reapplied_effect(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    target = "203.0.113.224"
    params = _target("ip", target)
    original = await _run_async_tool(
        provider,
        "block_ip",
        params,
        _context("stale-original"),
    )
    stale_rollback = ActionExecutionJob.model_validate(
        await provider.execute(
            "unblock_ip",
            params,
            context=_context("stale-queued"),
        )
    )
    winning_rollback = await _run_async_tool(
        provider,
        "unblock_ip",
        params,
        _context("stale-winning"),
    )
    replacement = await _run_async_tool(
        provider,
        "block_ip",
        params,
        _context("stale-replacement"),
    )

    stale_result = await provider.run_job(stale_rollback.job_id)
    current = await state.get_state("blocked_ips", target)

    assert original.status is ExecutionJobStatus.SUCCESS
    _assert_rollback_result(winning_rollback, rolled_back=True, warning=None)
    assert replacement.status is ExecutionJobStatus.SUCCESS
    assert stale_result.status is ExecutionJobStatus.FAILED
    assert stale_result.target_results[0].code == "stale_rollback_target"
    assert stale_result.raw_result["rolled_back"] is False
    assert isinstance(current, dict)
    assert current["job_id"] == replacement.job_id
    assert len(await state.list_namespace("rollback_history")) == 1


@pytest.mark.asyncio
async def test_rollback_observation_is_idempotent_during_recovery(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    target = "203.0.113.225"
    params = _target("ip", target)
    await _run_async_tool(provider, "block_ip", params, _context("observation-source"))
    rollback = await _run_async_tool(
        provider,
        "unblock_ip",
        params,
        _context("observation-rollback"),
    )
    record = await state.get_observation(
        "ip_blocks",
        target,
        job_id=rollback.job_id,
    )
    assert record is not None
    before = await state.list_observations()

    await state.set_observation(record.model_copy(update={"status": "blocked"}))

    after = await state.list_observations()
    assert after == before


@pytest.mark.asyncio
async def test_rollback_observation_idempotency_survives_projection_eviction(
    state: MockEnvironmentState,
) -> None:
    target = "203.0.113.227"
    original = MockObservationRecord(
        surface="ip_blocks",
        target=target,
        status="allowed",
        observed_at=datetime.now(UTC),
        available_at=datetime.now(UTC),
        observed_version=1,
        action_id="act-old-rollback",
        job_id="job-old-rollback",
        provider="mock_tool_provider",
        connector="mock-tool-connector",
    )
    await state.set_observation(original)
    for index in range(40):
        await state.set_observation(
            MockObservationRecord(
                surface="ip_blocks",
                target=target,
                status="blocked",
                observed_at=datetime.now(UTC),
                available_at=datetime.now(UTC),
                observed_version=index + 2,
                action_id=f"act-new-{index}",
                job_id=f"job-new-{index}",
                provider="mock_tool_provider",
                connector="mock-tool-connector",
            )
        )
    before = await state.get_observation("ip_blocks", target)
    assert before is not None
    assert before.job_id == "job-new-39"

    await state.set_observation(original)

    after = await state.get_observation("ip_blocks", target)
    assert after is not None
    assert after.job_id == before.job_id
    assert after.projection_generation == before.projection_generation


@pytest.mark.asyncio
async def test_injected_missing_rollback_target_uses_business_result_contract(
    state: MockEnvironmentState,
) -> None:
    target = "203.0.113.226"
    params = _target("ip", target)
    await _run_async_tool(
        MockToolProvider(state),
        "block_ip",
        params,
        _context("injected-missing-source"),
    )
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(missing_targets={target}),
    )

    completed = await _run_async_tool(
        provider,
        "unblock_ip",
        params,
        _context("injected-missing-rollback"),
    )

    _assert_rollback_result(
        completed,
        rolled_back=False,
        warning="target_not_found",
    )
    assert await state.get_state("blocked_ips", target) is not None
    assert await state.list_namespace("rollback_history") == {}


@pytest.mark.asyncio
async def test_restore_account_does_not_undo_an_unrelated_password_reset(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    target = "owner@example.test"
    await _run_async_tool(
        provider,
        "reset_password",
        _target("account", target),
        _context("password-reset"),
    )

    completed = await _run_async_tool(
        provider,
        "restore_account",
        _target("account", target),
        _context("wrong-source"),
    )

    _assert_rollback_result(
        completed,
        rolled_back=False,
        warning="target_not_found",
    )
    account = await state.get_state("accounts", target)
    assert isinstance(account, dict)
    assert account["status"] == "password_reset"
    assert await state.list_namespace("rollback_history") == {}


@pytest.mark.asyncio
async def test_close_false_positive_ticket_closes_in_place_and_audits_original(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    created = ToolResult.model_validate(
        await provider.execute(
            "create_ticket",
            {"title": "False positive candidate"},
            context=_context("ticket-source"),
        )
    )
    ticket_id = created.data["artifact_ids"][0]
    original = await state.get_state("tickets", ticket_id)
    assert isinstance(original, dict)

    completed = await _run_async_tool(
        provider,
        "close_false_positive_ticket",
        {"ticket_id": ticket_id, "reason": "confirmed false positive"},
        _context("ticket-rollback"),
    )

    _assert_rollback_result(completed, rolled_back=True, warning=None)
    closed = await state.get_state("tickets", ticket_id)
    assert isinstance(closed, dict)
    assert closed["status"] == "closed"
    assert closed["version"] == original["version"] + 1
    history = await state.list_namespace("rollback_history")
    assert len(history) == 1
    assert next(iter(history.values()))["original_record"] == original

    repeated = await _run_async_tool(
        provider,
        "close_false_positive_ticket",
        {"ticket_id": ticket_id},
        _context("ticket-repeat"),
    )
    _assert_rollback_result(
        repeated,
        rolled_back=False,
        warning="target_not_found",
    )
    assert len(await state.list_namespace("rollback_history")) == 1


@pytest.mark.asyncio
async def test_discovered_wrapper_returns_queued_job_without_applying_early(
    state: MockEnvironmentState,
) -> None:
    registry = ToolRegistry()
    registry.auto_discover()
    provider = MockToolProvider(state)
    target = "198.51.100.122"
    await _run_async_tool(
        provider,
        "block_ip",
        _target("ip", target),
        _context("wrapper-source"),
    )
    context = _context("wrapper-rollback")

    with bind_mock_tool_provider(provider), bind_tool_execution_context(context):
        raw = await registry.get_tool("unblock_ip").execute(_target("ip", target))
    registry.validate_output("unblock_ip", raw)
    queued = ActionExecutionJob.model_validate(raw)

    assert queued.status is ExecutionJobStatus.QUEUED
    assert await state.get_state("blocked_ips", target) is not None


class _RecordingRedis:
    def __init__(self) -> None:
        self.eval_args: tuple[Any, ...] | None = None

    async def eval(self, *args: Any) -> list[Any]:
        self.eval_args = args
        metadata = RedisClient.loads(args[-1])
        metadata["original_record"] = {"status": "blocked", "version": 1}
        return [RedisClient.dumps(metadata), False, 1, b"rolled_back"]


class _RecordingRedisClient:
    def __init__(self) -> None:
        self.client = _RecordingRedis()

    def get_client(self) -> _RecordingRedis:
        return self.client


@pytest.mark.asyncio
async def test_redis_api_path_uses_atomic_script_and_rollback_history_namespace() -> None:
    redis_client = _RecordingRedisClient()
    state = MockEnvironmentState(redis_client=redis_client)  # type: ignore[arg-type]

    history, resulting_state, applied, code = await state.apply_rollback(
        job_id="job-redis-rollback",
        rollback_tool_name="unblock_ip",
        source_tool_name="block_ip",
        namespace="blocked_ips",
        key="203.0.113.99",
        expected_status="blocked",
        expected_source_version=1,
        expected_source_job_id="job-original-effect",
        expect_absent=False,
        replacement_status=None,
        rolled_back_at=datetime.now(UTC),
        rolled_back_by="tester",
        provider="mock_tool_provider",
        connector="mock-tool-connector",
        action_id="act-redis-rollback",
    )

    assert redis_client.client.eval_args is not None
    args = redis_client.client.eval_args
    assert args[2] == MOCK_TOOL_STATE_KEY
    assert str(args[3]).startswith("rollback_effects:")
    assert args[4] == "blocked_ips:203.0.113.99"
    assert str(args[5]).startswith("rollback_history:")
    assert args[7] == "1"
    assert args[8] == "job-original-effect"
    assert args[9] == "0"
    assert history is not None
    assert resulting_state is None
    assert applied is True
    assert code == "rolled_back"
