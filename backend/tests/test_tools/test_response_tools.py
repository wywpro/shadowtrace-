from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.config import get_settings
from app.models.disposition import DispositionReceipt, TargetWritebackResult
from app.models.enums import (
    CapabilityState,
    ConfirmationEvidence,
    ExecutionJobStatus,
    ExecutionOwner,
    TargetExecutionStatus,
    TargetWritebackStatus,
    ToolCategory,
    WritebackStatus,
)
from app.models.execution import ActionExecutionJob
from app.models.tool_meta import ExecutionChannel, ToolResult, ToolResultStatus
from app.providers.tools.mock_provider import (
    MockToolProvider,
    MockToolProviderConfig,
    ToolExecutionContext,
    bind_mock_tool_provider,
    bind_tool_execution_context,
    get_mock_tool_provider,
    map_disposition_receipt_to_job,
)
from app.tools.mock_state import MockEnvironmentState
from app.tools.registry import ToolNotFoundError, ToolRegistry, tool_registry
from app.tools.specs import RESPONSE_TOOL_METAS


@pytest.fixture
async def state() -> MockEnvironmentState:
    store = MockEnvironmentState.in_memory()
    await store.clear_all()
    return store


def _context(
    suffix: str = "1",
    *,
    owner: ExecutionOwner = ExecutionOwner.DIRECT_TOOL,
    idempotency_key: str | None = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        event_id=f"evt-20260714-{suffix:0>8}",
        action_id=f"act-{suffix:0>8}",
        idempotency_key=idempotency_key or f"idem-{suffix}",
        execution_owner=owner,
    )


def _target(target_type: str, target: str, **parameters: Any) -> dict[str, Any]:
    return {
        "target_type": target_type,
        "target": target,
        "parameters": parameters,
    }


def test_registry_discovers_all_baseline_response_implementations() -> None:
    registry = ToolRegistry()
    discovered = set(registry.auto_discover())
    required = {meta.tool_name for meta in RESPONSE_TOOL_METAS}

    assert required.issubset(discovered)
    executable = {meta.tool_name for meta in RESPONSE_TOOL_METAS if meta.executable}
    assert executable.issubset(
        {
            entry.tool_meta.tool_name
            for entry in registry.list_registered_tools(ToolCategory.RESPONSE)
        }
    )
    assert registry.get_tool("update_source_event_disposition").tool_impl is None


def test_provider_manifest_registers_mutually_exclusive_owner_channels(
    state: MockEnvironmentState,
) -> None:
    registry = ToolRegistry()
    registry.auto_discover()
    provider = MockToolProvider(state)
    provider.register_bindings(registry)

    direct = registry.resolve_binding("block_ip", ExecutionOwner.DIRECT_TOOL, [])
    managed = registry.resolve_binding("block_ip", ExecutionOwner.XDR_MANAGED, [])
    assert direct.provider_name == "mock_tool_provider"
    assert direct.execution_channel is ExecutionChannel.TOOL_PROVIDER
    assert managed.provider_name == "mock_xdr"
    assert managed.execution_channel is ExecutionChannel.DISPOSITION_ADAPTER
    disposition = registry.resolve_binding(
        "update_source_event_disposition",
        ExecutionOwner.XDR_MANAGED,
        ["event_disposition"],
    )
    assert disposition.provider_name == "mock_xdr"
    assert disposition.execution_channel is ExecutionChannel.DISPOSITION_ADAPTER


def test_register_bindings_skips_missing_virtual_tool(
    state: MockEnvironmentState,
) -> None:
    """Partial discovery must still attach executable response bindings."""

    registry = ToolRegistry()
    registry.auto_discover(include_virtual=False)
    assert "update_source_event_disposition" not in {
        entry.tool_meta.tool_name for entry in registry.list_registered_tools()
    }

    MockToolProvider(state).register_bindings(registry)

    direct = registry.resolve_binding("block_ip", ExecutionOwner.DIRECT_TOOL, [])
    managed = registry.resolve_binding("block_ip", ExecutionOwner.XDR_MANAGED, [])
    assert direct.provider_name == "mock_tool_provider"
    assert managed.provider_name == "mock_xdr"
    with pytest.raises(ToolNotFoundError):
        registry.get_tool("update_source_event_disposition")


def test_process_registry_loads_mock_owner_bindings_at_startup() -> None:
    direct = tool_registry.resolve_binding("block_ip", ExecutionOwner.DIRECT_TOOL, [])
    managed = tool_registry.resolve_binding("block_ip", ExecutionOwner.XDR_MANAGED, [])

    assert direct.provider_name == "mock_tool_provider"
    assert managed.provider_name == "mock_xdr"


def test_mock_provider_fails_closed_when_simulation_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as context:
        context.setenv("TOOL_MODE", "mock")
        context.setenv("SIMULATION_ENABLED", "false")
        get_settings.cache_clear()
        with pytest.raises(RuntimeError, match="SIMULATION_ENABLED=true"):
            get_mock_tool_provider()
    get_settings.cache_clear()


def test_xdr_managed_receipt_maps_to_shared_job_without_provider_dispatch() -> None:
    receipt = DispositionReceipt(
        writeback_id="wbk-xdr-1",
        sequence=2,
        disposition_id="disp-1",
        action_id="act-xdr-1",
        source_record_id="src-1",
        status=WritebackStatus.PARTIAL,
        confirmation_evidence=ConfirmationEvidence.STATUS_QUERIED,
        provider_job_id="pjob-xdr-1",
        provider_code="partial",
        target_results=[
            TargetWritebackResult(
                canonical_target="host:host-a",
                status=TargetWritebackStatus.CONFIRMED,
                artifact_ref="artifact-a",
            ),
            TargetWritebackResult(
                canonical_target="host:host-b",
                status=TargetWritebackStatus.FAILED,
                provider_code="device_offline",
            ),
        ],
        raw_result={"fixture": "shadowtrace_mock_disposition"},
        simulated=True,
    )

    job = map_disposition_receipt_to_job(
        receipt,
        event_id="evt-20260714-00000001",
        idempotency_key="xdr-idem-1",
    )

    assert job.provider_name == "mock_xdr"
    assert job.provider_job_id == "pjob-xdr-1"
    assert job.status is ExecutionJobStatus.PARTIAL_SUCCESS
    assert [item.status for item in job.target_results] == [
        TargetExecutionStatus.SUCCESS,
        TargetExecutionStatus.FAILED,
    ]
    assert job.raw_result["writeback_id"] == "wbk-xdr-1"


def test_xdr_accepted_provider_job_remains_queued_and_raw_result_is_sanitized() -> None:
    receipt = DispositionReceipt(
        writeback_id="wbk-xdr-queued",
        sequence=1,
        disposition_id="disp-queued",
        action_id="",
        source_record_id="",
        status=WritebackStatus.ACCEPTED,
        provider_job_id="pjob-queued",
        provider_code="token=provider-code-secret",
        provider_message="Authorization: Bearer provider-message-secret",
        raw_result={
            "authorization": "secret",
            "nested": {"token": "secret"},
            "provider_note": "Bearer value-pattern-secret",
            "callback": "https://user:password@example.test/status",
        },
        simulated=True,
    )

    job = map_disposition_receipt_to_job(
        receipt,
        event_id="evt-20260714-00000001",
        action_id="act-correlated",
        idempotency_key="xdr-idem-queued",
    )

    assert job.action_id == "act-correlated"
    assert job.status is ExecutionJobStatus.QUEUED
    assert job.raw_result["authorization"] == "***"
    assert job.raw_result["nested"]["token"] == "***"
    assert "value-pattern-secret" not in job.raw_result["provider_note"]
    assert "password" not in job.raw_result["callback"]
    assert "provider-code-secret" not in str(job.provider_code)
    assert "provider-message-secret" not in str(job.provider_message)


@pytest.mark.parametrize(
    ("writeback_status", "provider_job_id", "expected_status"),
    [
        (WritebackStatus.PENDING, None, ExecutionJobStatus.QUEUED),
        (WritebackStatus.SENDING, None, ExecutionJobStatus.RUNNING),
        (WritebackStatus.ACCEPTED, None, ExecutionJobStatus.RUNNING),
        (WritebackStatus.ACCEPTED, "provider-job", ExecutionJobStatus.QUEUED),
        (WritebackStatus.CONFIRMED, None, ExecutionJobStatus.SUCCESS),
        (WritebackStatus.PARTIAL, None, ExecutionJobStatus.PARTIAL_SUCCESS),
        (WritebackStatus.FAILED, None, ExecutionJobStatus.FAILED),
        (WritebackStatus.CONFLICT, None, ExecutionJobStatus.FAILED),
        (WritebackStatus.UNKNOWN, None, ExecutionJobStatus.UNKNOWN),
    ],
)
def test_each_writeback_status_maps_to_shared_job_contract(
    writeback_status: WritebackStatus,
    provider_job_id: str | None,
    expected_status: ExecutionJobStatus,
) -> None:
    receipt = DispositionReceipt(
        writeback_id=f"wbk-{writeback_status.value}",
        sequence=1,
        disposition_id=f"disp-{writeback_status.value}",
        action_id="act-map-status",
        source_record_id="src-map-status",
        status=writeback_status,
        confirmation_evidence=(
            ConfirmationEvidence.READBACK_VERIFIED
            if writeback_status is WritebackStatus.CONFIRMED
            else None
        ),
        provider_job_id=provider_job_id,
        simulated=True,
    )

    job = map_disposition_receipt_to_job(
        receipt,
        event_id="evt-20260714-00000001",
        idempotency_key=f"idem-{writeback_status.value}",
    )

    assert job.status is expected_status


@pytest.mark.parametrize(
    ("writeback_status", "expected_status"),
    [
        (TargetWritebackStatus.PENDING, TargetExecutionStatus.UNKNOWN),
        (TargetWritebackStatus.ACCEPTED, TargetExecutionStatus.UNKNOWN),
        (TargetWritebackStatus.CONFIRMED, TargetExecutionStatus.SUCCESS),
        (TargetWritebackStatus.FAILED, TargetExecutionStatus.FAILED),
        (TargetWritebackStatus.CONFLICT, TargetExecutionStatus.FAILED),
        (TargetWritebackStatus.UNKNOWN, TargetExecutionStatus.UNKNOWN),
    ],
)
def test_each_target_writeback_status_maps_to_execution_status(
    writeback_status: TargetWritebackStatus,
    expected_status: TargetExecutionStatus,
) -> None:
    receipt = DispositionReceipt(
        writeback_id=f"wbk-target-{writeback_status.value}",
        sequence=1,
        disposition_id=f"disp-target-{writeback_status.value}",
        action_id="act-map-target",
        source_record_id="src-map-target",
        status=WritebackStatus.UNKNOWN,
        target_results=[
            TargetWritebackResult(
                canonical_target="host:host-map",
                status=writeback_status,
            )
        ],
        simulated=True,
    )

    job = map_disposition_receipt_to_job(
        receipt,
        event_id="evt-20260714-00000001",
        idempotency_key=f"idem-target-{writeback_status.value}",
    )

    assert job.target_results[0].status is expected_status


def test_missing_capability_is_not_registered_for_direct_execution(
    state: MockEnvironmentState,
) -> None:
    registry = ToolRegistry()
    registry.auto_discover()
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(disabled_tools={"block_ip"}),
    )
    provider.register_bindings(registry)

    with pytest.raises(ToolNotFoundError):
        registry.resolve_binding("block_ip", ExecutionOwner.DIRECT_TOOL, [])
    assert (
        registry.resolve_binding("block_ip", ExecutionOwner.XDR_MANAGED, []).provider_name
        == "mock_xdr"
    )


def test_capability_manifest_matches_enabled_direct_tools(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(disabled_tools={"block_domain"}),
    )
    manifest = provider.capability_manifest()

    assert manifest.provider_name == "mock_tool_provider"
    assert manifest.supports_idempotency is True
    assert manifest.supports_lookup_by_idempotency is True
    assert manifest.supports_concurrency_control is True
    assert manifest.supports_fencing is True
    assert manifest.allowed_execution_channels == [ExecutionChannel.TOOL_PROVIDER]
    assert (
        manifest.allows(
            intent_kind=manifest.allowed_intents[0],
            operation_code="block_ip",
        )
        is CapabilityState.SUPPORTED
    )
    assert "block_domain" not in manifest.allowed_operations


@pytest.mark.asyncio
async def test_invalid_params_return_validation_error_without_creating_job(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    result = ToolResult.model_validate(
        await provider.execute("block_ip", {"target_type": "ip"}, context=_context())
    )

    assert result.status is ToolResultStatus.VALIDATION_ERROR
    assert result.provider_code == "validation_error"
    assert await state.list_namespace("jobs") == {}
    assert await state.list_namespace("dispatch_intents") == {}


@pytest.mark.asyncio
async def test_xdr_managed_is_rejected_before_provider_dispatch(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    result = ToolResult.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.10"),
            context=_context(owner=ExecutionOwner.XDR_MANAGED),
        )
    )

    assert result.status is ToolResultStatus.VALIDATION_ERROR
    assert result.provider_code == "wrong_execution_owner"
    assert await state.list_namespace("jobs") == {}
    assert await state.get_state("blocked_ips", "203.0.113.10") is None


@pytest.mark.asyncio
async def test_action_owner_claim_prevents_cross_channel_dispatch(
    state: MockEnvironmentState,
) -> None:
    context = _context()
    await state.claim_execution_owner(context.action_id, ExecutionOwner.XDR_MANAGED.value)
    provider = MockToolProvider(state)

    result = ToolResult.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.14"),
            context=context,
        )
    )

    assert result.status is ToolResultStatus.VALIDATION_ERROR
    assert result.provider_code == "execution_owner_conflict"
    assert await state.list_namespace("jobs") == {}


@pytest.mark.asyncio
async def test_async_action_is_queued_before_effect_and_remains_traceable(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    params = _target("ip", "203.0.113.11")
    context = _context()

    accepted = ActionExecutionJob.model_validate(
        await provider.execute("block_ip", params, context=context)
    )
    assert accepted.status is ExecutionJobStatus.QUEUED
    assert await state.get_state("blocked_ips", "203.0.113.11") is None
    intent = await state.get_dispatch_intent(accepted.job_id)
    assert intent is not None
    assert intent["action_id"] == context.action_id
    assert intent["execution_owner"] == "direct_tool"

    completed = await provider.run_job(accepted.job_id)
    record = await state.get_state("blocked_ips", "203.0.113.11")
    assert completed.status is ExecutionJobStatus.SUCCESS
    assert isinstance(record, dict)
    assert record["action_id"] == context.action_id
    assert record["job_id"] == accepted.job_id
    assert record["provider"] == "mock_tool_provider"
    assert record["connector"] == "mock-tool-connector"
    assert record["version"] == 1
    assert record["effective_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "target_type", "target", "namespace"),
    [
        ("block_ip", "ip", "203.0.113.21", "blocked_ips"),
        ("block_domain", "domain", "blocked.example", "blocked_domains"),
        ("isolate_host", "host", "host-21", "isolated_hosts"),
        ("quarantine_file", "file", "sha256:file-21", "quarantined_files"),
        ("block_process", "process", "sha256:process-21", "blocked_processes"),
        ("scan_host_for_virus", "host", "host-scan-21", "scan_results"),
        ("disable_account", "account", "disabled-user", "accounts"),
        ("force_logout", "account", "logout-user", "sessions"),
        ("reset_password", "account", "reset-user", "accounts"),
        ("revoke_token", "account", "token-user", "tokens"),
    ],
)
async def test_each_async_baseline_tool_applies_traceable_state(
    state: MockEnvironmentState,
    tool_name: str,
    target_type: str,
    target: str,
    namespace: str,
) -> None:
    provider = MockToolProvider(state)
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            tool_name,
            _target(target_type, target),
            context=_context(tool_name),
        )
    )

    assert queued.status is ExecutionJobStatus.QUEUED
    completed = await provider.run_job(queued.job_id)
    record = await state.get_state(namespace, target)

    assert completed.status is ExecutionJobStatus.SUCCESS
    assert isinstance(record, dict)
    assert record["status"]
    assert record["reason"]
    assert record["executed_at"]
    assert record["executed_by"] == "shadowtrace"
    assert record["provider"] == "mock_tool_provider"
    assert record["connector"] == "mock-tool-connector"
    assert record["action_id"] == queued.action_id
    assert record["job_id"] == queued.job_id
    assert record["effective_at"]
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_provider_reuses_precreated_execution_job_id(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    context = _context().model_copy(update={"execution_job_id": "job-precreated"})

    accepted = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.19"),
            context=context,
        )
    )

    assert accepted.job_id == "job-precreated"
    assert set(await state.list_namespace("jobs")) == {"job-precreated"}


@pytest.mark.asyncio
async def test_same_idempotency_key_never_creates_second_job_or_effect(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    params = _target("ip", "203.0.113.12")
    context = _context(idempotency_key="stable-idem")

    accepted = ActionExecutionJob.model_validate(
        await provider.execute("block_ip", params, context=context)
    )
    await provider.run_job(accepted.job_id)
    replay = ActionExecutionJob.model_validate(
        await provider.execute("block_ip", params, context=context)
    )

    assert replay.job_id == accepted.job_id
    assert replay.status is ExecutionJobStatus.SUCCESS
    assert len(await state.list_namespace("jobs")) == 1
    record = await state.get_state("blocked_ips", "203.0.113.12")
    assert isinstance(record, dict)
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_lookup_by_idempotency_returns_the_reserved_job(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    context = _context(idempotency_key="lookup-idem")
    accepted = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.24"),
            context=context,
        )
    )

    looked_up = await provider.lookup_by_idempotency("block_ip", context.idempotency_key)

    assert looked_up is not None
    assert ActionExecutionJob.model_validate(looked_up).job_id == accepted.job_id
    assert await provider.lookup_by_idempotency("block_domain", context.idempotency_key) is None


@pytest.mark.asyncio
async def test_same_tool_idempotency_key_rejects_changed_payload(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    context = _context(idempotency_key="changed-payload-idem")
    first = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.25"),
            context=context,
        )
    )
    changed = ToolResult.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.26"),
            context=context,
        )
    )

    assert first.status is ExecutionJobStatus.QUEUED
    assert changed.status is ToolResultStatus.VALIDATION_ERROR
    assert changed.provider_code == "idempotency_key_reuse"
    assert len(await state.list_namespace("jobs")) == 1


@pytest.mark.asyncio
async def test_idempotency_key_cannot_be_reused_across_tools(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    context = _context(idempotency_key="shared-key")
    first = ActionExecutionJob.model_validate(
        await provider.execute(
            "disable_account",
            _target("account", "alice"),
            context=context,
        )
    )
    second = ToolResult.model_validate(
        await provider.execute(
            "reset_password",
            _target("account", "alice"),
            context=context,
        )
    )

    assert first.status is ExecutionJobStatus.QUEUED
    assert second.status is ToolResultStatus.VALIDATION_ERROR
    assert second.provider_code == "idempotency_key_reuse"
    assert len(await state.list_namespace("jobs")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        _target("domain", "example.test"),
        _target("ip", ""),
        _target("ip", "203.0.113.15", targets=[]),
        _target("ip", "203.0.113.15", targets=[{"bad": "target"}]),
    ],
)
async def test_invalid_target_contract_is_rejected_before_queue(
    state: MockEnvironmentState,
    params: dict[str, Any],
) -> None:
    provider = MockToolProvider(state)
    result = ToolResult.model_validate(
        await provider.execute("block_ip", params, context=_context())
    )

    assert result.status is ToolResultStatus.VALIDATION_ERROR
    assert await state.list_namespace("jobs") == {}


@pytest.mark.asyncio
async def test_duplicate_block_with_new_idempotency_is_already_applied(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    params = _target("ip", "203.0.113.13")
    first = ActionExecutionJob.model_validate(
        await provider.execute("block_ip", params, context=_context("1"))
    )
    await provider.run_job(first.job_id)
    second = ActionExecutionJob.model_validate(
        await provider.execute("block_ip", params, context=_context("2"))
    )
    completed = await provider.run_job(second.job_id)

    assert completed.status is ExecutionJobStatus.SUCCESS
    assert completed.target_results[0].code == "already_applied"
    record = await state.get_state("blocked_ips", "203.0.113.13")
    assert isinstance(record, dict)
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_multiple_targets_can_complete_with_partial_success(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(offline_targets={"host-b"}),
    )
    accepted = ActionExecutionJob.model_validate(
        await provider.execute(
            "isolate_host",
            _target("host", "host-a", targets=["host-a", "host-b"]),
            context=_context(),
        )
    )
    completed = await provider.run_job(accepted.job_id)

    assert completed.status is ExecutionJobStatus.PARTIAL_SUCCESS
    assert [item.status for item in completed.target_results] == [
        TargetExecutionStatus.SUCCESS,
        TargetExecutionStatus.FAILED,
    ]
    assert completed.target_results[1].code == "device_offline"
    assert completed.target_results[1].raw_result["code"] == "device_offline"
    assert await state.get_state("isolated_hosts", "host-a") is not None
    assert await state.get_state("isolated_hosts", "host-b") is None


@pytest.mark.asyncio
async def test_concurrent_workers_apply_one_effect(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.16"),
            context=_context(),
        )
    )

    await asyncio.gather(
        provider.run_job(queued.job_id, worker_id="worker-a"),
        provider.run_job(queued.job_id, worker_id="worker-b"),
    )
    completed = await provider.get_job(queued.job_id)
    record = await state.get_state("blocked_ips", "203.0.113.16")

    assert completed.status is ExecutionJobStatus.SUCCESS
    assert completed.attempt == 1
    assert isinstance(record, dict)
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_cancel_cannot_overwrite_a_concurrently_completed_job(
    state: MockEnvironmentState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockToolProvider(state)
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.22"),
            context=_context(),
        )
    )
    cancel_reached_intent = asyncio.Event()
    allow_cancel = asyncio.Event()
    original_get_intent = state.get_dispatch_intent

    async def gated_get_intent(job_id: str) -> dict[str, Any] | None:
        task = asyncio.current_task()
        if task is not None and task.get_name() == "cancel-job":
            cancel_reached_intent.set()
            await allow_cancel.wait()
        return await original_get_intent(job_id)

    monkeypatch.setattr(state, "get_dispatch_intent", gated_get_intent)
    cancel_task = asyncio.create_task(
        provider.cancel_job(queued.job_id),
        name="cancel-job",
    )
    await cancel_reached_intent.wait()
    completed = await provider.run_job(queued.job_id, worker_id="run-worker")
    allow_cancel.set()
    cancel_result = await cancel_task
    stored = await provider.get_job(queued.job_id)

    assert completed.status is ExecutionJobStatus.SUCCESS
    assert cancel_result.status is ExecutionJobStatus.SUCCESS
    assert stored.status is ExecutionJobStatus.SUCCESS
    assert await state.get_state("blocked_ips", "203.0.113.22") is not None


@pytest.mark.asyncio
async def test_expired_claim_uses_fencing_token(
    state: MockEnvironmentState,
) -> None:
    job = {
        "job_id": "job-fenced",
        "status": "running",
    }
    await state.set_job("job-fenced", job)
    first_token = await state.claim_job(
        "job-fenced",
        "worker-a",
        lease_seconds=0.001,
    )
    await asyncio.sleep(0.01)
    second_token = await state.claim_job(
        "job-fenced",
        "worker-b",
        lease_seconds=1,
    )

    stale_saved = await state.set_job_if_claimed(
        "job-fenced",
        {**job, "status": "failed"},
        worker_id="worker-a",
        token=first_token,
    )
    current_saved = await state.set_job_if_claimed(
        "job-fenced",
        {**job, "status": "success"},
        worker_id="worker-b",
        token=second_token,
    )

    assert first_token > 0
    assert second_token > first_token
    assert stale_saved is False
    assert current_saved is True
    stored = await state.get_job("job-fenced")
    assert stored is not None
    assert stored["status"] == "success"


@pytest.mark.asyncio
async def test_ticket_sequence_allocation_is_atomic_per_job(
    state: MockEnvironmentState,
) -> None:
    sequences = await asyncio.gather(
        state.allocate_ticket_sequence("job-ticket"),
        state.allocate_ticket_sequence("job-ticket"),
    )

    assert list(sequences) == [1, 1]


@pytest.mark.asyncio
async def test_concurrent_capacity_check_is_atomic(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(capacity_limits={"blocked_ips": 1}),
    )
    first = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.17"),
            context=_context("1"),
        )
    )
    second = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_ip",
            _target("ip", "203.0.113.18"),
            context=_context("2"),
        )
    )

    results = await asyncio.gather(
        provider.run_job(first.job_id),
        provider.run_job(second.job_id),
    )

    assert {item.status for item in results} == {
        ExecutionJobStatus.SUCCESS,
        ExecutionJobStatus.FAILED,
    }
    assert len(await state.list_namespace("blocked_ips")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected_status", "expected_code"),
    [
        ({"missing_targets": {"host-1"}}, ExecutionJobStatus.FAILED, "target_not_found"),
        ({"permission_denied_targets": {"host-1"}}, ExecutionJobStatus.FAILED, "permission_denied"),
        ({"transient_error_targets": {"host-1"}}, ExecutionJobStatus.FAILED, "transient_error"),
        ({"timed_out_targets": {"host-1"}}, ExecutionJobStatus.TIMED_OUT, "timed_out"),
        ({"cancelled_targets": {"host-1"}}, ExecutionJobStatus.CANCELLED, "cancelled"),
    ],
)
async def test_configurable_terminal_failures_preserve_original_error(
    state: MockEnvironmentState,
    config: dict[str, set[str]],
    expected_status: ExecutionJobStatus,
    expected_code: str,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig.model_validate(config),
    )
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "isolate_host",
            _target("host", "host-1"),
            context=_context(),
        )
    )
    completed = await provider.run_job(queued.job_id)

    assert completed.status is expected_status
    assert completed.target_results[0].code == expected_code
    assert completed.target_results[0].raw_result["code"] == expected_code
    assert completed.raw_result["target_codes"] == [expected_code]


@pytest.mark.asyncio
async def test_capacity_limit_fails_without_mutating_environment(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(capacity_limits={"blocked_domains": 0}),
    )
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "block_domain",
            _target("domain", "example.test"),
            context=_context(),
        )
    )
    completed = await provider.run_job(queued.job_id)

    assert completed.status is ExecutionJobStatus.FAILED
    assert completed.target_results[0].code == "capacity_exceeded"
    assert await state.get_state("blocked_domains", "example.test") is None


@pytest.mark.asyncio
async def test_effect_replay_does_not_invent_missing_state(
    state: MockEnvironmentState,
) -> None:
    record = {
        "status": "blocked",
        "reason": "fixture",
        "executed_at": "2026-07-14T00:00:00Z",
        "executed_by": "test",
        "provider": "mock_tool_provider",
        "connector": "mock-tool-connector",
        "version": 1,
        "action_id": "act-fixture",
        "job_id": "job-fixture",
        "effective_at": "2026-07-14T00:00:00Z",
        "value": {},
    }
    stored, applied, code = await state.apply_effect(
        job_id="job-fixture",
        namespace="blocked_ips",
        key="203.0.113.23",
        record=record,
        desired_status="blocked",
        allow_update=False,
        capacity=None,
    )
    assert isinstance(stored, dict)
    assert applied is True
    assert code == "applied"
    await state.delete_state("blocked_ips", "203.0.113.23")

    replayed, replay_applied, replay_code = await state.apply_effect(
        job_id="job-fixture",
        namespace="blocked_ips",
        key="203.0.113.23",
        record=record,
        desired_status="blocked",
        allow_update=False,
        capacity=None,
    )

    assert replayed is None
    assert replay_applied is False
    assert replay_code == "applied"


@pytest.mark.asyncio
async def test_lost_response_retry_looks_up_job_without_duplicate_side_effect(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(lost_response_targets={"198.51.100.20"}),
    )
    params = _target("ip", "198.51.100.20")
    context = _context(idempotency_key="lost-response-idem")

    unknown = ActionExecutionJob.model_validate(
        await provider.execute("block_ip", params, context=context)
    )
    replay = ActionExecutionJob.model_validate(
        await provider.execute("block_ip", params, context=context)
    )

    assert unknown.status is ExecutionJobStatus.UNKNOWN
    assert unknown.provider_code == "response_lost"
    assert replay.job_id == unknown.job_id
    assert replay.status is ExecutionJobStatus.SUCCESS
    assert len(await state.list_namespace("jobs")) == 1
    record = await state.get_state("blocked_ips", "198.51.100.20")
    assert isinstance(record, dict)
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_unknown_job_can_be_confirmed_as_late_success(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(late_success_targets={"host-late"}),
    )
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "isolate_host",
            _target("host", "host-late"),
            context=_context(),
        )
    )
    unknown = await provider.run_job(queued.job_id)
    assert unknown.status is ExecutionJobStatus.UNKNOWN
    assert await state.get_state("isolated_hosts", "host-late") is None

    confirmed = await provider.resolve_late_success(queued.job_id)
    assert confirmed.status is ExecutionJobStatus.SUCCESS
    assert confirmed.provider_code == "late_confirmation"
    assert await state.get_state("isolated_hosts", "host-late") is not None


@pytest.mark.asyncio
async def test_concurrent_late_success_resolution_uses_status_cas(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(late_success_targets={"host-late-cas"}),
    )
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "isolate_host",
            _target("host", "host-late-cas"),
            context=_context(),
        )
    )
    unknown = await provider.run_job(queued.job_id)
    assert unknown.status is ExecutionJobStatus.UNKNOWN

    results = await asyncio.gather(
        provider.resolve_late_success(queued.job_id),
        provider.resolve_late_success(queued.job_id),
    )
    record = await state.get_state("isolated_hosts", "host-late-cas")

    assert all(item.status is ExecutionJobStatus.SUCCESS for item in results)
    assert isinstance(record, dict)
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_late_success_is_scoped_to_configured_target(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(late_success_targets={"host-late"}),
    )
    queued = ActionExecutionJob.model_validate(
        await provider.execute(
            "isolate_host",
            _target("host", "host-now", targets=["host-now", "host-late"]),
            context=_context(),
        )
    )
    unknown = await provider.run_job(queued.job_id)

    assert unknown.status is ExecutionJobStatus.UNKNOWN
    assert [item.status for item in unknown.target_results] == [
        TargetExecutionStatus.SUCCESS,
        TargetExecutionStatus.UNKNOWN,
    ]
    assert await state.get_state("isolated_hosts", "host-now") is not None
    assert await state.get_state("isolated_hosts", "host-late") is None

    confirmed = await provider.resolve_late_success(queued.job_id)
    assert confirmed.status is ExecutionJobStatus.SUCCESS
    host_now = await state.get_state("isolated_hosts", "host-now")
    host_late = await state.get_state("isolated_hosts", "host-late")
    assert isinstance(host_now, dict)
    assert isinstance(host_late, dict)
    assert host_now["version"] == 1
    assert host_late["version"] == 1


@pytest.mark.asyncio
async def test_distinct_account_effects_update_state_version(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    disable = ActionExecutionJob.model_validate(
        await provider.execute(
            "disable_account",
            _target("account", "alice"),
            context=_context("1"),
        )
    )
    await provider.run_job(disable.job_id)
    reset = ActionExecutionJob.model_validate(
        await provider.execute(
            "reset_password",
            _target("account", "alice"),
            context=_context("2"),
        )
    )
    reset_result = await provider.run_job(reset.job_id)
    account = await state.get_state("accounts", "alice")

    assert reset_result.target_results[0].code == "applied"
    assert isinstance(account, dict)
    assert account["status"] == "password_reset"
    assert account["version"] == 2


@pytest.mark.asyncio
async def test_sync_tool_fault_and_capacity_are_configurable(
    state: MockEnvironmentState,
) -> None:
    failed_provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(permission_denied_targets={"create_ticket"}),
    )
    failed = ToolResult.model_validate(
        await failed_provider.execute(
            "create_ticket",
            {"title": "Denied"},
            context=_context("1"),
        )
    )
    assert failed.status is ToolResultStatus.FAILED
    assert failed.target_results[0].code == "permission_denied"

    capacity_provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(capacity_limits={"notifications": 0}),
    )
    capacity = ToolResult.model_validate(
        await capacity_provider.execute(
            "notify_security_team",
            {"message": "No capacity"},
            context=_context("2"),
        )
    )
    assert capacity.status is ToolResultStatus.FAILED
    assert capacity.target_results[0].code == "capacity_exceeded"
    assert await state.list_namespace("tickets") == {}
    assert await state.list_namespace("notifications") == {}


@pytest.mark.asyncio
async def test_sync_lost_response_recovers_without_duplicate_artifact(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(
        state,
        config=MockToolProviderConfig(lost_response_targets={"create_ticket"}),
    )
    context = _context(idempotency_key="sync-lost-response")
    params = {"title": "Recover ticket"}

    unknown = ToolResult.model_validate(
        await provider.execute("create_ticket", params, context=context)
    )
    replay = ToolResult.model_validate(
        await provider.execute("create_ticket", params, context=context)
    )

    assert unknown.status is ToolResultStatus.UNKNOWN
    assert replay.status is ToolResultStatus.SUCCESS
    assert replay.job_id == unknown.job_id
    assert len(await state.list_namespace("tickets")) == 1


def test_capacity_configuration_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MockToolProviderConfig(capacity_limits={"blocked_ips": -1})
    with pytest.raises(ValueError, match="unknown capacity namespaces"):
        MockToolProviderConfig(capacity_limits={"unknown": 1})


@pytest.mark.asyncio
async def test_sync_ticket_and_notification_use_required_identifiers(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    ticket = ToolResult.model_validate(
        await provider.execute(
            "create_ticket",
            {"title": "Investigate", "description": "Mock fixture"},
            context=_context("1"),
        )
    )
    notification = ToolResult.model_validate(
        await provider.execute(
            "notify_security_team",
            {"message": "Mock alert", "channels": ["soc"]},
            context=_context("2"),
        )
    )

    assert ticket.status is ToolResultStatus.SUCCESS
    assert notification.status is ToolResultStatus.SUCCESS
    assert re.fullmatch(r"TKT-\d{4}-\d{4}", ticket.data["artifact_ids"][0])
    assert re.fullmatch(r"ntf-[0-9a-f]{8}", notification.data["artifact_ids"][0])
    assert len(await state.list_namespace("tickets")) == 1
    assert len(await state.list_namespace("notifications")) == 1


@pytest.mark.asyncio
async def test_recovered_ticket_job_uses_creation_year(
    state: MockEnvironmentState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockToolProvider(state)
    context = _context()
    created_at = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)
    job = ActionExecutionJob(
        job_id="job-ticket-year",
        event_id=context.event_id,
        action_id=context.action_id,
        provider_name="mock_tool_provider",
        idempotency_key=context.idempotency_key,
        created_at=created_at,
        updated_at=created_at,
    )
    _, created = await state.reserve_dispatch(
        idempotency_key=context.idempotency_key,
        job_id=job.job_id,
        job=job.model_dump(mode="json"),
        intent={
            "job_id": job.job_id,
            "event_id": context.event_id,
            "action_id": context.action_id,
            "tool_name": "create_ticket",
            "execution_owner": "direct_tool",
            "connector": context.connector,
            "executed_by": context.executed_by,
            "parameters": {"title": "Recovered ticket", "parameters": {}},
            "payload_hash": "recovery-fixture",
            "status": "queued",
            "created_at": created_at.isoformat(),
        },
    )
    assert created is True
    monkeypatch.setattr(
        "app.providers.tools.mock_provider._utc_now",
        lambda: datetime(2027, 1, 1, tzinfo=UTC),
    )

    completed = await provider.run_job(job.job_id)

    assert completed.status is ExecutionJobStatus.SUCCESS
    assert completed.target_results[0].artifact_id == "TKT-2026-0001"


@pytest.mark.asyncio
async def test_discovered_thin_wrapper_uses_bound_provider(
    state: MockEnvironmentState,
) -> None:
    registry = ToolRegistry()
    registry.auto_discover()
    provider = MockToolProvider(state)
    context = _context(idempotency_key="wrapper-context-idem")

    with bind_mock_tool_provider(provider), bind_tool_execution_context(context):
        raw = await registry.get_tool("block_process").execute(_target("process", "sha256:abc"))
    job = ActionExecutionJob.model_validate(raw)
    intent = await state.get_dispatch_intent(job.job_id)

    assert job.provider_name == "mock_tool_provider"
    assert job.event_id == context.event_id
    assert job.action_id == context.action_id
    assert job.idempotency_key == context.idempotency_key
    assert intent is not None
    assert intent["action_id"] == context.action_id
    registry.validate_output("block_process", raw)
    assert await state.get_state("blocked_processes", "sha256:abc") is None


@pytest.mark.asyncio
async def test_clear_all_removes_state_jobs_intents_and_sequences(
    state: MockEnvironmentState,
) -> None:
    provider = MockToolProvider(state)
    result = ToolResult.model_validate(
        await provider.execute(
            "create_ticket",
            {"title": "First"},
            context=_context(),
        )
    )
    assert result.status is ToolResultStatus.SUCCESS

    await state.clear_all()
    assert await state.list_namespace("jobs") == {}
    assert await state.list_namespace("tickets") == {}
    assert await state.list_namespace("dispatch_intents") == {}
