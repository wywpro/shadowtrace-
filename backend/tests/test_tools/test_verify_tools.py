from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from app.core.redis_client import RedisClient
from app.models.enums import (
    ExecutionJobStatus,
    SourceObjectKind,
    ToolCategory,
)
from app.models.execution import ActionExecutionJob
from app.models.source import SourceReference
from app.models.tool_meta import ToolResult, ToolResultStatus
from app.providers.tools.mock_provider import (
    MockToolProvider,
    MockToolProviderConfig,
    ToolExecutionContext,
    bind_mock_tool_provider,
)
from app.tools.mock_state import (
    MOCK_OBSERVATION_PROJECTION_KEY,
    MOCK_VERIFY_OVERRIDE_KEY,
    MockEnvironmentState,
    MockObservationRecord,
)
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.verify._common import (
    VERIFICATION_SPECS,
    MockVerificationRuntime,
    bind_mock_verification_runtime,
)

VERIFY_NAMES = frozenset(VERIFICATION_SPECS)


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, bytes | str]] = {}

    async def hset(self, key: str, field: str, value: bytes | str) -> None:
        self.hashes.setdefault(key, {})[field] = value

    async def hget(self, key: str, field: str) -> bytes | str | None:
        return self.hashes.get(key, {}).get(field)

    async def hdel(self, key: str, field: str) -> None:
        self.hashes.get(key, {}).pop(field, None)

    async def hgetall(self, key: str) -> dict[str, bytes | str]:
        return dict(self.hashes.get(key, {}))

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self.hashes.pop(key, None)

    async def eval(
        self,
        _script: str,
        _num_keys: int,
        key: str,
        field: str,
        encoded_record: bytes,
        max_records: str,
    ) -> int:
        existing = self.hashes.get(key, {}).get(field)
        decoded = RedisClient.loads(existing) if existing is not None else []
        records = [decoded] if isinstance(decoded, dict) else list(decoded)
        incoming = RedisClient.loads(encoded_record)
        incoming["projection_generation"] = (
            max(
                (
                    int(item.get("projection_generation", 0))
                    for item in records
                    if isinstance(item, dict)
                ),
                default=0,
            )
            + 1
        )
        records.append(incoming)
        records = records[-int(max_records) :]
        self.hashes.setdefault(key, {})[field] = RedisClient.dumps(records)
        return len(records)


class _FakeRedisClient:
    def __init__(self) -> None:
        self.client = _FakeRedis()

    def get_client(self) -> _FakeRedis:
        return self.client


@pytest.fixture
async def state() -> MockEnvironmentState:
    store = MockEnvironmentState.in_memory()
    await store.clear_all()
    return store


def _context(suffix: str = "1") -> ToolExecutionContext:
    return ToolExecutionContext(
        event_id=f"evt-20260714-{suffix:0>8}",
        action_id=f"act-{suffix:0>8}",
        idempotency_key=f"verify-idem-{suffix}",
    )


def _target(target_type: str, target: str, **parameters: Any) -> dict[str, Any]:
    return {
        "target_type": target_type,
        "target": target,
        "parameters": parameters,
    }


async def _run_action(
    provider: MockToolProvider,
    tool_name: str,
    target_type: str,
    target: str,
    context: ToolExecutionContext,
) -> ActionExecutionJob:
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            tool_name,
            _target(target_type, target),
            context=context,
        )
    )
    return await provider.run_job(queued.job_id)


async def _run_verify(
    registry: ToolRegistry,
    runtime: MockVerificationRuntime,
    tool_name: str,
    params: dict[str, Any],
) -> ToolResult:
    registry.validate_input(tool_name, params)
    with bind_mock_verification_runtime(runtime):
        raw = await registry.get_tool(tool_name).execute(params)
    registry.validate_output(tool_name, raw)
    return ToolResult.model_validate(raw)


def test_registry_discovers_all_baseline_verification_tools_by_target_and_capability() -> None:
    registry = ToolRegistry()
    discovered = set(registry.auto_discover())

    assert VERIFY_NAMES.issubset(discovered)
    metas = {meta.tool_name: meta for meta in registry.list_tools(ToolCategory.VERIFICATION)}
    assert VERIFY_NAMES.issubset(metas)
    for tool_name, spec in VERIFICATION_SPECS.items():
        assert metas[tool_name].output_schema
        assert metas[tool_name].required_capabilities == [spec.method]
        assert metas[tool_name].target_types


def test_process_registry_registers_verification_implementations() -> None:
    assert tool_registry.get_tool("check_ip_block_status").tool_impl is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "response_tool",
        "verification_tool",
        "response_target_type",
        "response_target",
        "verification_target_type",
        "verification_target",
        "expected_method",
    ),
    [
        (
            "block_ip",
            "check_ip_block_status",
            "ip",
            "203.0.113.31",
            "ip",
            "203.0.113.31",
            "device_query",
        ),
        (
            "block_domain",
            "check_domain_block_status",
            "domain",
            "blocked.example",
            "domain",
            "blocked.example",
            "device_query",
        ),
        (
            "isolate_host",
            "check_host_isolation_status",
            "host",
            "host-31",
            "host",
            "host-31",
            "endpoint_query",
        ),
        (
            "quarantine_file",
            "check_file_quarantine_status",
            "file",
            "sha256:file-31",
            "file",
            "sha256:file-31",
            "endpoint_query",
        ),
        (
            "block_process",
            "check_process_block_status",
            "process",
            "sha256:process-31",
            "process",
            "sha256:process-31",
            "endpoint_query",
        ),
        (
            "scan_host_for_virus",
            "check_virus_scan_status",
            "host",
            "host-scan-31",
            "host",
            "host-scan-31",
            "endpoint_query",
        ),
        (
            "disable_account",
            "check_account_status",
            "account",
            "disabled-user",
            "account",
            "disabled-user",
            "endpoint_query",
        ),
        (
            "block_ip",
            "check_new_alerts",
            "ip",
            "203.0.113.32",
            "event",
            "__event_id__",
            "source_alert_delta",
        ),
        (
            "block_ip",
            "check_traffic_drop",
            "ip",
            "203.0.113.33",
            "ip",
            "203.0.113.33",
            "telemetry_observation",
        ),
    ],
)
async def test_each_verification_tool_reads_independent_projection(
    state: MockEnvironmentState,
    response_tool: str,
    verification_tool: str,
    response_target_type: str,
    response_target: str,
    verification_target_type: str,
    verification_target: str,
    expected_method: str,
) -> None:
    context = _context(verification_tool)
    resolved_target = (
        context.event_id if verification_target == "__event_id__" else verification_target
    )
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(
            observation_delay_ms=0,
            new_alert_events=(
                {context.event_id} if verification_tool == "check_new_alerts" else set()
            ),
        ),
    )
    completed = await _run_action(
        provider,
        response_tool,
        response_target_type,
        response_target,
        context,
    )
    registry = ToolRegistry()
    registry.auto_discover()
    runtime = MockVerificationRuntime(state)
    result = await _run_verify(
        registry,
        runtime,
        verification_tool,
        _target(
            verification_target_type,
            resolved_target,
            job_id=completed.job_id,
        ),
    )

    assert completed.status is ExecutionJobStatus.SUCCESS
    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["is_verified"] is True
    assert result.data["verification_method"] == expected_method
    assert result.data["observed_version"] == 1
    assert result.data["source_refs"] == []


@pytest.mark.asyncio
async def test_verification_waits_for_job_and_delayed_projection(
    state: MockEnvironmentState,
) -> None:
    observed_at = datetime.now(UTC)
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.34",
            status="blocked",
            observed_at=observed_at,
            available_at=observed_at,
            observed_version=1,
            action_id="act-stale-visible",
            job_id="job-stale-visible",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=30),
    )
    context = _context()
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.34"),
            context=context,
        )
    )
    registry = ToolRegistry()
    registry.auto_discover()
    runtime = MockVerificationRuntime(state, wait_timeout_ms=500, poll_interval_ms=5)

    async def complete_later() -> None:
        await asyncio.sleep(0.02)
        await provider.run_job(queued.job_id)

    completion = asyncio.create_task(complete_later())
    result = await _run_verify(
        registry,
        runtime,
        "check_ip_block_status",
        _target("ip", "203.0.113.34", job_id=queued.job_id),
    )
    await completion

    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["is_verified"] is True


@pytest.mark.asyncio
async def test_thin_wrapper_uses_the_bound_mock_provider_state(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    completed = await _run_action(
        provider,
        "block_ip",
        "ip",
        "203.0.113.45",
        _context(),
    )
    registry = ToolRegistry()
    registry.auto_discover()

    with bind_mock_tool_provider(provider):
        raw = await registry.get_tool("check_ip_block_status").execute(
            _target("ip", "203.0.113.45", job_id=completed.job_id)
        )
    registry.validate_output("check_ip_block_status", raw)

    result = ToolResult.model_validate(raw)
    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["is_verified"] is True


@pytest.mark.asyncio
async def test_pending_projection_timeout_is_not_reported_as_effect_failure(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=500),
    )
    completed = await _run_action(
        provider,
        "block_ip",
        "ip",
        "203.0.113.39",
        _context(),
    )
    registry = ToolRegistry()
    registry.auto_discover()
    timed_out = await _run_verify(
        registry,
        MockVerificationRuntime(state, wait_timeout_ms=1, poll_interval_ms=1),
        "check_ip_block_status",
        _target("ip", "203.0.113.39", job_id=completed.job_id),
    )

    assert timed_out.status is ToolResultStatus.TIMEOUT
    assert timed_out.data["is_verified"] is False
    assert timed_out.data["detail"] == "observation_not_visible"

    await asyncio.sleep(0.51)
    visible = await _run_verify(
        registry,
        MockVerificationRuntime(state),
        "check_ip_block_status",
        _target("ip", "203.0.113.39", job_id=completed.job_id),
    )
    assert visible.status is ToolResultStatus.SUCCESS
    assert visible.data["is_verified"] is True


@pytest.mark.asyncio
async def test_execution_state_alone_cannot_self_verify(
    state: MockEnvironmentState,
) -> None:
    await state.set_state(
        "blocked_ips",
        "203.0.113.35",
        {"status": "blocked", "version": 1},
    )
    registry = ToolRegistry()
    registry.auto_discover()
    result = await _run_verify(
        registry,
        MockVerificationRuntime(state),
        "check_ip_block_status",
        _target("ip", "203.0.113.35"),
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["is_verified"] is False
    assert result.data["detail"] == "observation_missing"


@pytest.mark.asyncio
async def test_unexecuted_or_observed_unblocked_state_returns_false(
    state: MockEnvironmentState,
) -> None:
    registry = ToolRegistry()
    registry.auto_discover()
    runtime = MockVerificationRuntime(state)
    missing = await _run_verify(
        registry,
        runtime,
        "check_ip_block_status",
        _target("ip", "203.0.113.36"),
    )
    assert missing.data["is_verified"] is False

    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    completed = await _run_action(
        provider,
        "block_ip",
        "ip",
        "203.0.113.36",
        _context("unblocked"),
    )
    verified = await _run_verify(
        registry,
        runtime,
        "check_ip_block_status",
        _target("ip", "203.0.113.36", job_id=completed.job_id),
    )
    assert verified.data["is_verified"] is True

    observed_at = datetime.now(UTC)
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.36",
            status="allowed",
            observed_at=observed_at,
            available_at=observed_at,
            observed_version=2,
            action_id="act-unblock",
            job_id="job-unblock",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    unblocked = await _run_verify(
        registry,
        runtime,
        "check_ip_block_status",
        _target("ip", "203.0.113.36"),
    )
    assert unblocked.data["is_verified"] is False
    assert unblocked.data["detail"] == "observed_status:allowed"


@pytest.mark.asyncio
async def test_false_override_forces_failure_without_mutating_projection(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    completed = await _run_action(
        provider,
        "block_domain",
        "domain",
        "override.example",
        _context(),
    )
    await state.set_verify_override(
        "check_domain_block_status",
        "override.example",
        False,
    )
    before = await state.list_observations()
    registry = ToolRegistry()
    registry.auto_discover()
    result = await _run_verify(
        registry,
        MockVerificationRuntime(state),
        "check_domain_block_status",
        _target("domain", "override.example", job_id=completed.job_id),
    )
    after = await state.list_observations()

    assert result.data["is_verified"] is False
    assert result.data["detail"] == "forced_failure_override"
    assert after == before


@pytest.mark.asyncio
async def test_true_override_cannot_create_ungrounded_success(
    state: MockEnvironmentState,
) -> None:
    await state.set_verify_override(
        "check_ip_block_status",
        "203.0.113.46",
        True,
    )
    registry = ToolRegistry()
    registry.auto_discover()
    result = await _run_verify(
        registry,
        MockVerificationRuntime(state),
        "check_ip_block_status",
        _target("ip", "203.0.113.46"),
    )

    assert result.data["is_verified"] is False
    assert result.data["detail"] == "observation_missing"
    assert result.data["observed_version"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "target", "expected_detail"),
    [
        (
            {"observation_never_targets": {"host-never"}},
            "host-never",
            "observation_missing",
        ),
        (
            {"observation_reversed_targets": {"host-reversed"}},
            "host-reversed",
            "observed_status:connected",
        ),
    ],
)
async def test_never_effective_and_reversed_observation_injection(
    state: MockEnvironmentState,
    config: dict[str, set[str]],
    target: str,
    expected_detail: str,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(
            **config,
            observation_delay_ms=0,
        ),
    )
    completed = await _run_action(
        provider,
        "isolate_host",
        "host",
        target,
        _context(target),
    )
    registry = ToolRegistry()
    registry.auto_discover()
    result = await _run_verify(
        registry,
        MockVerificationRuntime(state, wait_timeout_ms=10, poll_interval_ms=1),
        "check_host_isolation_status",
        _target("host", target, job_id=completed.job_id),
    )

    assert completed.status is ExecutionJobStatus.SUCCESS
    assert result.data["is_verified"] is False
    assert result.data["detail"] == expected_detail


@pytest.mark.asyncio
async def test_non_effectful_and_unknown_jobs_do_not_claim_verification(
    state: MockEnvironmentState,
) -> None:
    failed_provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(offline_targets={"host-failed"}),
    )
    failed_job = await _run_action(
        failed_provider,
        "isolate_host",
        "host",
        "host-failed",
        _context("failed"),
    )
    unknown_provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(late_success_targets={"host-unknown"}),
    )
    unknown_job = await _run_action(
        unknown_provider,
        "isolate_host",
        "host",
        "host-unknown",
        _context("unknown"),
    )
    registry = ToolRegistry()
    registry.auto_discover()
    runtime = MockVerificationRuntime(state, wait_timeout_ms=0)

    failed = await _run_verify(
        registry,
        runtime,
        "check_host_isolation_status",
        _target("host", "host-failed", job_id=failed_job.job_id),
    )
    unknown = await _run_verify(
        registry,
        runtime,
        "check_host_isolation_status",
        _target("host", "host-unknown", job_id=unknown_job.job_id),
    )

    assert failed.status is ToolResultStatus.SUCCESS
    assert failed.data["is_verified"] is False
    assert failed.data["detail"] == "execution_job_failed"
    assert unknown.status is ToolResultStatus.UNKNOWN
    assert unknown.data["is_verified"] is False
    assert unknown.data["detail"] == "execution_job_not_terminal:unknown"


@pytest.mark.asyncio
async def test_job_target_and_target_result_are_grounding_guards(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(
            observation_delay_ms=0,
            offline_targets={"203.0.113.42"},
        ),
    )
    context = _context()
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target(
                "ip",
                "203.0.113.41",
                targets=["203.0.113.41", "203.0.113.42"],
            ),
            context=context,
        )
    )
    completed = await provider.run_job(queued.job_id)
    registry = ToolRegistry()
    registry.auto_discover()
    runtime = MockVerificationRuntime(state)

    wrong_target = await _run_verify(
        registry,
        runtime,
        "check_ip_block_status",
        _target("ip", "203.0.113.99", job_id=completed.job_id),
    )
    failed_target = await _run_verify(
        registry,
        runtime,
        "check_ip_block_status",
        _target("ip", "203.0.113.42", job_id=completed.job_id),
    )

    assert completed.status is ExecutionJobStatus.PARTIAL_SUCCESS
    assert wrong_target.data["detail"] == "execution_job_target_mismatch"
    assert failed_target.data["detail"] == "execution_target_not_success:failed"
    assert wrong_target.data["is_verified"] is False
    assert failed_target.data["is_verified"] is False


@pytest.mark.asyncio
async def test_stale_projection_from_another_job_is_not_accepted(
    state: MockEnvironmentState,
) -> None:
    now = datetime.now(UTC)
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.43",
            status="blocked",
            observed_at=now,
            available_at=now,
            observed_version=1,
            action_id="act-old",
            job_id="job-old",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(
            observation_delay_ms=0,
            observation_never_targets={"203.0.113.43"},
        ),
    )
    completed = await _run_action(
        provider,
        "block_ip",
        "ip",
        "203.0.113.43",
        _context(),
    )
    registry = ToolRegistry()
    registry.auto_discover()
    result = await _run_verify(
        registry,
        MockVerificationRuntime(state),
        "check_ip_block_status",
        _target("ip", "203.0.113.43", job_id=completed.job_id),
    )

    assert result.data["is_verified"] is False
    assert result.data["detail"] == "observation_missing"


@pytest.mark.asyncio
async def test_already_applied_exemption_is_scoped_to_the_current_target(
    state: MockEnvironmentState,
) -> None:
    first_provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(observation_delay_ms=0),
    )
    await _run_action(
        first_provider,
        "block_ip",
        "ip",
        "203.0.113.47",
        _context("first-target"),
    )
    observed_at = datetime.now(UTC)
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.48",
            status="blocked",
            observed_at=observed_at,
            available_at=observed_at,
            observed_version=1,
            action_id="act-stale-target",
            job_id="job-stale-target",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    second_provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(
            observation_delay_ms=0,
            observation_never_targets={"203.0.113.48"},
        ),
    )
    context = _context("multi-target")
    queued = ActionExecutionJob.model_validate(
        await second_provider.execute(
            "block_ip",
            _target(
                "ip",
                "203.0.113.47",
                targets=["203.0.113.47", "203.0.113.48"],
            ),
            context=context,
        )
    )
    completed = await second_provider.run_job(queued.job_id)
    assert any(
        item.canonical_target == "ip:203.0.113.47" and item.code == "already_applied"
        for item in completed.target_results
    )

    registry = ToolRegistry()
    registry.auto_discover()
    result = await _run_verify(
        registry,
        MockVerificationRuntime(state),
        "check_ip_block_status",
        _target("ip", "203.0.113.48", job_id=completed.job_id),
    )

    assert result.data["is_verified"] is False
    assert result.data["detail"] == "observation_missing"


@pytest.mark.asyncio
async def test_pending_generation_does_not_hide_or_replace_newer_visible_state(
    state: MockEnvironmentState,
) -> None:
    now = datetime.now(UTC)
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.49",
            status="blocked",
            observed_at=now,
            available_at=now,
            observed_version=2,
            action_id="act-visible",
            job_id="job-visible",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.49",
            status="allowed",
            observed_at=now,
            available_at=now + timedelta(seconds=1),
            observed_version=1,
            action_id="act-delayed-old",
            job_id="job-delayed-old",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.49",
            status="allowed",
            observed_at=now,
            available_at=now + timedelta(milliseconds=500),
            observed_version=3,
            action_id="act-delayed-new",
            job_id="job-delayed-new",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )

    visible = await state.get_observation("ip_blocks", "203.0.113.49")
    future = await state.get_observation(
        "ip_blocks",
        "203.0.113.49",
        observed_at=now + timedelta(seconds=2),
    )

    assert visible is not None
    assert visible.status == "blocked"
    assert visible.observed_version == 2
    assert future is not None
    assert future.status == "allowed"
    assert future.observed_version == 3


@pytest.mark.asyncio
async def test_projection_generation_orders_shared_account_surface(
    state: MockEnvironmentState,
) -> None:
    now = datetime.now(UTC)
    for status, source_version, job_id in [
        ("password_reset", 5, "job-password-reset"),
        ("revoked", 1, "job-token-revoked"),
    ]:
        await state.set_observation(
            MockObservationRecord(
                surface="account_status",
                target="shared-account",
                status=status,
                observed_at=now,
                available_at=now,
                observed_version=source_version,
                action_id=f"act-{job_id}",
                job_id=job_id,
                provider="mock_tool_provider",
                connector="mock-tool-connector",
            )
        )

    latest = await state.get_observation("account_status", "shared-account")

    assert latest is not None
    assert latest.status == "revoked"
    assert latest.observed_version == 1
    assert latest.projection_generation == 2


@pytest.mark.asyncio
async def test_source_references_are_preserved_in_verification_output(
    state: MockEnvironmentState,
) -> None:
    now = datetime.now(UTC)
    source_ref = SourceReference(
        source_kind=SourceObjectKind.ASSET,
        source_product="mock_xdr",
        source_tenant_id="tenant-1",
        connector_id="mock-xdr",
        source_object_id="asset-31",
    )
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.37",
            status="blocked",
            observed_at=now,
            available_at=now,
            observed_version=3,
            source_refs=[source_ref],
            action_id="act-source-ref",
            job_id="job-source-ref",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    registry = ToolRegistry()
    registry.auto_discover()
    result = await _run_verify(
        registry,
        MockVerificationRuntime(state),
        "check_ip_block_status",
        _target("ip", "203.0.113.37"),
    )

    assert result.data["is_verified"] is True
    assert result.data["observed_version"] == 3
    assert result.data["source_refs"][0]["source_object_id"] == "asset-31"


@pytest.mark.asyncio
async def test_clear_all_removes_projection_and_override(
    state: MockEnvironmentState,
) -> None:
    now = datetime.now(UTC)
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.38",
            status="blocked",
            observed_at=now,
            available_at=now,
            observed_version=1,
            action_id="act-clear",
            job_id="job-clear",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    await state.set_verify_override(
        "check_ip_block_status",
        "203.0.113.38",
        False,
    )

    await state.clear_all()

    assert await state.get_observation("ip_blocks", "203.0.113.38") is None
    assert (
        await state.get_verify_override(
            "check_ip_block_status",
            "203.0.113.38",
        )
        is None
    )


@pytest.mark.asyncio
async def test_redis_api_path_uses_documented_override_hash_and_projection() -> None:
    redis_client = _FakeRedisClient()
    state = MockEnvironmentState(redis_client=cast(Any, redis_client))
    now = datetime.now(UTC)
    await state.set_observation(
        MockObservationRecord(
            surface="ip_blocks",
            target="203.0.113.44",
            status="blocked",
            observed_at=now,
            available_at=now,
            observed_version=1,
            action_id="act-redis",
            job_id="job-redis",
            provider="mock_tool_provider",
            connector="mock-tool-connector",
        )
    )
    await state.set_verify_override(
        "check_ip_block_status",
        "203.0.113.44",
        False,
    )

    override_field = "check_ip_block_status:203.0.113.44"
    assert redis_client.client.hashes[MOCK_VERIFY_OVERRIDE_KEY][override_field] == "false"
    assert "ip_blocks:203.0.113.44" in redis_client.client.hashes[MOCK_OBSERVATION_PROJECTION_KEY]
    assert await state.get_observation("ip_blocks", "203.0.113.44") is not None
    assert (
        await state.get_verify_override(
            "check_ip_block_status",
            "203.0.113.44",
        )
        is False
    )

    await state.clear_all()
    assert redis_client.client.hashes == {}
