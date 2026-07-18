"""Vendor-neutral ToolProvider and DispositionAdapter contracts (ISSUE-026)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest

from app.adapters.disposition.base import DispositionAdapterCapabilities
from app.adapters.disposition.http_adapter import HttpDispositionAdapter
from app.core.errors import (
    ShadowTraceError,
    WritebackConflictError,
    WritebackUnsupportedError,
)
from app.models.disposition import (
    DispositionCommand,
    DispositionReceipt,
    RecordExecutionResultParams,
    SourceObjectLocator,
    SubmitEntityActionParams,
)
from app.models.enums import (
    CapabilityState,
    ConfirmationEvidence,
    ConnectorStatus,
    DispositionIntentKind,
    ExecutionJobStatus,
    ExecutionOwner,
    SourceObjectKind,
    WritebackStatus,
)
from app.models.execution import ActionExecutionJob
from app.models.tool_meta import (
    CapabilityManifest,
    ExecutionChannel,
    ProviderToolBinding,
    ToolResult,
    ToolResultStatus,
    WrongExecutionChannelError,
)
from app.providers.tools.mock_provider import MockToolProvider
from app.tools.adapters.base import (
    AdapterConfig,
    BaseToolAdapter,
    ToolMode,
    configure_tool_registry,
)
from app.tools.adapters.file_state_firewall import FileStateFirewallAdapter
from app.tools.executor import InMemoryExecutionJobStore, ToolExecutor
from app.tools.registry import (
    ToolNotFoundError,
    ToolRegistry,
    ToolUnavailableReason,
    ToolValidationError,
)
from app.tools.retry import RetryPolicy


def _sfx() -> str:
    return uuid.uuid4().hex[:8]


def _file_adapter(path: Path) -> FileStateFirewallAdapter:
    return FileStateFirewallAdapter(
        AdapterConfig(
            endpoint=path.as_uri(),
            auth_type="none",
            enabled=True,
        )
    )


def test_adapter_config_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="must not embed credentials"):
        AdapterConfig(
            endpoint="https://user:secret-value@candidate.invalid/submit",
            auth_type="none",
            enabled=True,
        )
    with pytest.raises(ValueError, match="credential query parameters"):
        AdapterConfig(
            endpoint="https://candidate.invalid/submit?api_key=secret-value",
            auth_type="none",
            enabled=True,
        )


class _NoOptionalCapabilitiesAdapter(BaseToolAdapter):
    name = "no_optional_capabilities"
    tool_meta = FileStateFirewallAdapter.tool_meta.model_copy(deep=True)

    def capability_manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            provider_name=self.name,
            online=True,
            entity_response=CapabilityState.SUPPORTED,
            allowed_operations=["block_ip"],
            supports_idempotency=True,
            allowed_execution_channels=[ExecutionChannel.TOOL_PROVIDER],
        )

    async def execute(
        self,
        params: dict[str, Any],
        idempotency_key: str,
    ) -> ToolResult:
        _ = (params, idempotency_key)
        return self.unsupported_result(
            error_detail="not used",
            provider_code="not_used",
        )

    async def health_check(self) -> bool:
        return True


class _UnavailableLiveAdapter(_NoOptionalCapabilitiesAdapter):
    name = "unavailable_live_provider"

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self.execute_count = 0

    def capability_manifest(self) -> CapabilityManifest:
        return super().capability_manifest().model_copy(update={"provider_name": self.name})

    async def execute(
        self,
        params: dict[str, Any],
        idempotency_key: str,
    ) -> ToolResult:
        self.execute_count += 1
        return await super().execute(params, idempotency_key)

    async def health_check(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_undeclared_optional_tool_capabilities_return_unsupported(
    tmp_path: Path,
) -> None:
    adapter = _NoOptionalCapabilitiesAdapter(
        AdapterConfig(
            endpoint=(tmp_path / "unused").as_uri(),
            auth_type="none",
            enabled=True,
        )
    )
    status = await adapter.get_job_status("provider-job")
    lookup = await adapter.lookup_by_idempotency("idempotency-key")

    assert adapter.capability_manifest().supports_status_query is False
    assert adapter.capability_manifest().supports_lookup_by_idempotency is False
    assert status.status is ToolResultStatus.UNSUPPORTED
    assert lookup is not None and lookup.status is ToolResultStatus.UNSUPPORTED


async def _seed_job(
    store: InMemoryExecutionJobStore,
    *,
    provider_name: str,
) -> tuple[str, str, str, str]:
    event_id = f"evt-{_sfx()}"
    action_id = f"act-{_sfx()}"
    job_id = f"job-{_sfx()}"
    idempotency_key = f"idem-{_sfx()}"
    await store.seed_job(
        ActionExecutionJob(
            job_id=job_id,
            event_id=event_id,
            action_id=action_id,
            provider_name=provider_name,
            idempotency_key=idempotency_key,
            status=ExecutionJobStatus.QUEUED,
        )
    )
    return event_id, action_id, job_id, idempotency_key


def _block_params(target: str = "203.0.113.26") -> dict[str, Any]:
    return {
        "target_type": "ip",
        "target": target,
        "parameters": {"reason_code": "candidate-profile-test"},
    }


@pytest.mark.asyncio
async def test_file_adapter_is_discovered_and_called_through_executor(tmp_path: Path) -> None:
    state_path = tmp_path / "firewall.json"
    config = AdapterConfig(
        endpoint=state_path.as_uri(),
        auth_type="none",
        enabled=True,
    )
    registry = ToolRegistry()
    discovered = await registry.auto_discover_for_mode(
        tool_mode=ToolMode.MIXED,
        adapter_configs={FileStateFirewallAdapter.name: config},
        mixed_routes={"block_ip": FileStateFirewallAdapter.name},
    )
    store = InMemoryExecutionJobStore()
    event_id, action_id, job_id, idempotency_key = await _seed_job(
        store,
        provider_name=FileStateFirewallAdapter.name,
    )

    result = await ToolExecutor(registry=registry, job_store=store).call(
        "block_ip",
        _block_params(),
        event_id,
        action_id=action_id,
        execution_job_id=job_id,
        idempotency_key=idempotency_key,
        execution_owner=ExecutionOwner.DIRECT_TOOL,
        retry_policy=RetryPolicy(max_retries=0),
    )

    assert "block_ip" in discovered
    assert result.status is ToolResultStatus.ACCEPTED
    assert result.provider_name == FileStateFirewallAdapter.name
    assert result.job_id == job_id
    assert result.raw_result["simulated"] is True
    polled = await _file_adapter(state_path).get_job_status(result.provider_job_id or "")
    assert polled.status is ToolResultStatus.SUCCESS
    persisted = await store.get_job(job_id)
    assert persisted is not None
    assert persisted.provider_name == FileStateFirewallAdapter.name
    assert persisted.status is ExecutionJobStatus.QUEUED
    state = orjson.loads((tmp_path / "firewall.json").read_bytes())
    assert "203.0.113.26" in state["blocked_ips"]
    assert idempotency_key not in (tmp_path / "firewall.json").read_text()


@pytest.mark.asyncio
async def test_tool_modes_are_strict_and_mixed_routes_are_per_tool(tmp_path: Path) -> None:
    mock_registry = ToolRegistry()
    await mock_registry.auto_discover_for_mode(tool_mode="mock")
    assert mock_registry.list_bindings("block_ip")[0].provider_name == "mock_tool_provider"

    live_registry = ToolRegistry()
    adapter = _file_adapter(tmp_path / "live-firewall.json")
    live_adapter = _NoOptionalCapabilitiesAdapter(adapter.config)
    with pytest.raises(
        ToolValidationError,
        match="live ToolProvider side effects are disabled",
    ):
        await live_registry.auto_discover_for_mode(
            tool_mode="live",
            adapters=[live_adapter],
        )
    await live_registry.auto_discover_for_mode(
        tool_mode="live",
        adapters=[live_adapter],
        allow_live_side_effects=True,
    )
    assert live_registry.list_bindings("block_ip")[0].provider_name == live_adapter.name
    with pytest.raises(ToolNotFoundError):
        live_registry.get_tool("query_dns")
    with pytest.raises(
        ToolValidationError,
        match="mixed live ToolProvider side effects are disabled",
    ):
        await ToolRegistry().auto_discover_for_mode(
            tool_mode="mixed",
            adapters=[live_adapter],
            mixed_routes={"block_ip": live_adapter.name},
        )
    with pytest.raises(
        ToolValidationError,
        match="live tool mode forbids simulated Providers",
    ):
        await ToolRegistry().auto_discover_for_mode(
            tool_mode="live",
            adapters=[adapter],
        )

    mixed_registry = ToolRegistry()
    await mixed_registry.auto_discover_for_mode(
        tool_mode="mixed",
        adapter_configs={adapter.name: adapter.config},
        mixed_routes={
            "block_ip": adapter.name,
            "block_domain": "mock",
        },
        mock_provider=MockToolProvider(),
    )
    assert mixed_registry.list_bindings("block_ip")[0].provider_name == adapter.name
    assert {binding.provider_name for binding in mixed_registry.list_bindings("block_domain")} == {
        "mock_tool_provider",
        "mock_xdr",
    }
    with pytest.raises(ToolNotFoundError):
        mixed_registry.get_tool("query_dns")


@pytest.mark.asyncio
async def test_unhealthy_live_adapter_is_unavailable_and_never_submits(tmp_path: Path) -> None:
    adapter = _UnavailableLiveAdapter(
        AdapterConfig(
            endpoint=(tmp_path / "unavailable-live").as_uri(),
            auth_type="none",
            enabled=True,
        )
    )
    registry = ToolRegistry()
    await configure_tool_registry(
        registry,
        tool_mode="live",
        adapters=[adapter],
        allow_live_side_effects=True,
    )
    view = next(
        entry
        for entry in registry.list_registered_tools()
        if entry.tool_meta.tool_name == "block_ip"
    )
    assert view.available is False
    assert view.unavailable_reasons == (ToolUnavailableReason.UNHEALTHY,)

    store = InMemoryExecutionJobStore()
    event_id, action_id, job_id, idempotency_key = await _seed_job(
        store,
        provider_name=adapter.name,
    )
    result = await ToolExecutor(registry=registry, job_store=store).call(
        "block_ip",
        _block_params(),
        event_id,
        action_id=action_id,
        execution_job_id=job_id,
        idempotency_key=idempotency_key,
        execution_owner=ExecutionOwner.DIRECT_TOOL,
    )
    assert result.status is ToolResultStatus.UNSUPPORTED
    assert result.provider_name == adapter.name
    assert adapter.execute_count == 0


@pytest.mark.asyncio
async def test_file_adapter_idempotency_status_and_crash_recovery(tmp_path: Path) -> None:
    path = tmp_path / "recoverable-firewall.json"
    adapter = _file_adapter(path)
    idempotency_key = "idem-file-recovery"

    first = await adapter.execute(_block_params(), idempotency_key)
    replay = await adapter.execute(_block_params(), idempotency_key)
    recovered = await _file_adapter(path).lookup_by_idempotency(idempotency_key)
    conflict = await adapter.execute(_block_params("203.0.113.99"), idempotency_key)
    status = await adapter.get_job_status(first.provider_job_id or "")

    assert first.status is ToolResultStatus.SUCCESS
    assert replay.data["idempotent_replay"] is True
    assert recovered is not None and recovered.status is ToolResultStatus.SUCCESS
    assert recovered.provider_job_id == first.provider_job_id
    assert status.status is ToolResultStatus.SUCCESS
    assert conflict.status is ToolResultStatus.VALIDATION_ERROR
    assert conflict.provider_code == "idempotency_key_reuse"


def _candidate_capabilities() -> DispositionAdapterCapabilities:
    supported = CapabilityState.SUPPORTED
    return DispositionAdapterCapabilities(
        intents={
            DispositionIntentKind.ENTITY_ACTION_SUBMIT: supported,
            DispositionIntentKind.EXECUTION_RESULT_RECORD: supported,
        },
        operations={
            "submit_entity_action": supported,
            "record_execution_result": supported,
        },
        supports_idempotency=True,
        supports_status_query=True,
        supports_concurrency_token=True,
        supports_lookup_by_idempotency=True,
    )


def _source_locator() -> SourceObjectLocator:
    return SourceObjectLocator(
        source_product="candidate-profile",
        source_tenant_id="tenant-test",
        connector_id="connector-test",
        source_kind=SourceObjectKind.INCIDENT,
        source_object_id="incident-test",
    )


def _command(owner: ExecutionOwner) -> DispositionCommand:
    if owner is ExecutionOwner.XDR_MANAGED:
        intent = DispositionIntentKind.ENTITY_ACTION_SUBMIT
        operation_code = "submit_entity_action"
        params: SubmitEntityActionParams | RecordExecutionResultParams = SubmitEntityActionParams(
            entity_action_code="block_ip",
            canonical_target="203.0.113.26",
        )
    else:
        intent = DispositionIntentKind.EXECUTION_RESULT_RECORD
        operation_code = "record_execution_result"
        params = RecordExecutionResultParams(summary_code="completed")
    return DispositionCommand(
        disposition_id=f"disp-{_sfx()}",
        action_id=f"act-{_sfx()}",
        closure_cycle=1,
        intent_kind=intent,
        source_locator=_source_locator(),
        operation_code=operation_code,
        operation_params=params,
        operator_id="shadowtrace",
        idempotency_key=f"idem-{_sfx()}",
        source_concurrency_token="version-1",
        execution_owner=owner,
    )


def _receipt(
    command: DispositionCommand,
    *,
    status: WritebackStatus = WritebackStatus.ACCEPTED,
    provider_job_id: str | None = "candidate-job-1",
) -> DispositionReceipt:
    return DispositionReceipt(
        writeback_id=f"wbk-{_sfx()}",
        sequence=1,
        disposition_id=command.disposition_id,
        action_id=command.action_id,
        source_record_id=command.source_locator.source_object_id,
        status=status,
        confirmation_evidence=(
            ConfirmationEvidence.READBACK_VERIFIED if status is WritebackStatus.CONFIRMED else None
        ),
        provider_job_id=provider_job_id,
        simulated=False,
    )


def _http_config() -> AdapterConfig:
    return AdapterConfig(
        endpoint="https://candidate.invalid/submit",
        auth_type="bearer",
        credential_ref="ISSUE26_WRITE_TOKEN",
        timeout_s=1,
        tls_verify=True,
        enabled=True,
    )


@pytest.mark.asyncio
async def test_xdr_owner_uses_disposition_channel_and_direct_owner_syncs_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISSUE26_WRITE_TOKEN", "write-token-value")
    posts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = orjson.loads(request.content)
        posts.append(body)
        command = DispositionCommand.model_validate(body)
        return httpx.Response(
            200,
            json=_receipt(command).model_dump(mode="json"),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    disposition = HttpDispositionAdapter(
        _http_config(),
        capabilities=_candidate_capabilities(),
        allow_side_effects=True,
        client=client,
    )
    file_adapter = _file_adapter(tmp_path / "owners-firewall.json")
    registry = ToolRegistry()
    await configure_tool_registry(
        registry,
        tool_mode="mixed",
        adapters=[file_adapter],
        mixed_routes={"block_ip": file_adapter.name},
    )
    registry.register_binding(
        ProviderToolBinding(
            tool_name="block_ip",
            provider_name=disposition.name,
            execution_owner=ExecutionOwner.XDR_MANAGED,
            execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
            capabilities=["entity_response"],
        )
    )

    store = InMemoryExecutionJobStore()
    event_id, action_id, job_id, idempotency_key = await _seed_job(
        store,
        provider_name=file_adapter.name,
    )
    executor = ToolExecutor(registry=registry, job_store=store)
    with pytest.raises(WrongExecutionChannelError):
        await executor.call(
            "block_ip",
            _block_params("203.0.113.27"),
            event_id,
            action_id=action_id,
            execution_job_id=job_id,
            idempotency_key=idempotency_key,
            execution_owner=ExecutionOwner.XDR_MANAGED,
        )
    assert not (tmp_path / "owners-firewall.json").exists()

    await disposition.submit(_command(ExecutionOwner.XDR_MANAGED))
    direct_result = await executor.call(
        "block_ip",
        _block_params(),
        event_id,
        action_id=action_id,
        execution_job_id=job_id,
        idempotency_key=idempotency_key,
        execution_owner=ExecutionOwner.DIRECT_TOOL,
    )
    assert direct_result.status is ToolResultStatus.ACCEPTED
    completed = await file_adapter.get_job_status(direct_result.provider_job_id or "")
    assert completed.status is ToolResultStatus.SUCCESS
    await disposition.submit(_command(ExecutionOwner.DIRECT_TOOL))

    assert [row["intent_kind"] for row in posts] == [
        DispositionIntentKind.ENTITY_ACTION_SUBMIT.value,
        DispositionIntentKind.EXECUTION_RESULT_RECORD.value,
    ]
    assert all("reason" not in row and "raw_result" not in row for row in posts)
    await client.aclose()


@pytest.mark.asyncio
async def test_http_profile_status_lookup_and_lost_response_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISSUE26_WRITE_TOKEN", "write-token-value")
    command = _command(ExecutionOwner.XDR_MANAGED)
    receipt = _receipt(command, status=WritebackStatus.CONFIRMED)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/submit":
            raise httpx.ReadTimeout("response lost", request=request)
        if request.url.path == "/lookup":
            return httpx.Response(200, json=receipt.model_dump(mode="json"))
        if request.url.path == "/status/candidate-job-1":
            return httpx.Response(200, json=receipt.model_dump(mode="json"))
        if request.url.path == "/health":
            return httpx.Response(204)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpDispositionAdapter(
        _http_config(),
        capabilities=_candidate_capabilities(),
        status_endpoint_template="https://candidate.invalid/status/{provider_job_id}",
        idempotency_lookup_endpoint="https://candidate.invalid/lookup",
        health_endpoint="https://candidate.invalid/health",
        allow_side_effects=True,
        client=client,
    )

    recovered = await adapter.submit(command)
    status = await adapter.get_status("candidate-job-1")
    health = await adapter.health_check()
    lookup_request = next(item for item in requests if item.url.path == "/lookup")

    assert recovered.status is WritebackStatus.CONFIRMED
    assert status is not None and status.status is WritebackStatus.CONFIRMED
    assert health is ConnectorStatus.ONLINE
    assert command.idempotency_key not in str(lookup_request.url)
    assert command.source_locator.source_tenant_id not in str(lookup_request.url)
    assert "idempotency_key_sha256=" in str(lookup_request.url)
    await client.aclose()


@pytest.mark.asyncio
async def test_http_profile_unknown_capability_and_separated_credentials_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISSUE26_WRITE_TOKEN", "same-token")
    monkeypatch.setenv("ISSUE26_READ_TOKEN", "same-token")
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    unknown = HttpDispositionAdapter(
        _http_config(),
        allow_side_effects=True,
        client=client,
    )
    with pytest.raises(WritebackUnsupportedError):
        await unknown.submit(_command(ExecutionOwner.XDR_MANAGED))
    assert called is False

    shared = HttpDispositionAdapter(
        _http_config(),
        capabilities=_candidate_capabilities(),
        source_credential_ref="ISSUE26_READ_TOKEN",
        allow_side_effects=True,
        client=client,
    )
    assert shared.validate_config() is False
    assert await shared.health_check() is ConnectorStatus.UNKNOWN
    with pytest.raises(WritebackUnsupportedError):
        await shared.submit(_command(ExecutionOwner.XDR_MANAGED))
    assert called is False
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [
        (401, "auth_error"),
        (403, "permission_denied"),
        (409, "version_conflict"),
    ],
)
async def test_http_candidate_error_classification(
    status_code: int,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISSUE26_WRITE_TOKEN", "write-token-value")

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status_code)))
    adapter = HttpDispositionAdapter(
        _http_config(),
        capabilities=_candidate_capabilities(),
        allow_side_effects=True,
        client=client,
    )
    expected_exception = WritebackConflictError if status_code == 409 else ShadowTraceError
    with pytest.raises(expected_exception) as captured:
        await adapter.submit(_command(ExecutionOwner.XDR_MANAGED))
    assert captured.value.error_code == expected_code
    await client.aclose()


@pytest.mark.asyncio
async def test_http_5xx_without_idempotency_evidence_returns_unknown_not_mock_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISSUE26_WRITE_TOKEN", "write-token-value")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/lookup":
            return httpx.Response(404)
        return httpx.Response(503)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpDispositionAdapter(
        _http_config(),
        capabilities=_candidate_capabilities(),
        idempotency_lookup_endpoint="https://candidate.invalid/lookup",
        allow_side_effects=True,
        client=client,
    )
    command = _command(ExecutionOwner.XDR_MANAGED)
    result = await adapter.submit(command)
    repeated = await adapter.submit(command)

    assert result.status is WritebackStatus.UNKNOWN
    assert repeated.writeback_id == result.writeback_id
    assert result.provider_code == "unknown_delivery"
    assert result.simulated is False
    assert "mock" not in result.writeback_id
    await client.aclose()
