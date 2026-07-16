"""Adapter contract tests (ISSUE-012) — Mock facts, not vendor API facts."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.adapters._util import parse_source_item
from app.adapters.disposition.base import BaseDispositionAdapter
from app.adapters.file_source import FileSourceAdapter
from app.adapters.mock_xdr import (
    LiveDispositionAdapterStub,
    MockXDRDispositionAdapter,
    MockXDRSourceAdapter,
)
from app.adapters.normalizers import CHANNEL_NORMALIZERS, normalize_record
from app.adapters.registry import DispositionAdapterRegistry, SourceAdapterRegistry
from app.adapters.source.base import BaseSourceAdapter, InMemoryDataQualityRecorder
from app.core.errors import (
    AdapterNotFoundError,
    DependencyUnavailableError,
    WritebackUnsupportedError,
)
from app.data_generators.scenarios import build_scenario
from app.mock_xdr.models import MockFailureProfile
from app.mock_xdr.state import MockXDRState
from app.models.enums import (
    CapabilityState,
    ConnectorCapability,
    ConnectorStatus,
    SourceObjectKind,
    WritebackStatus,
)
from app.models.source import SourceAlert, SourceAsset, SourceIncident, SourceLog
from tests.test_adapters.conftest import event_disposition_command

REPO_ROOT = Path(__file__).resolve().parents[3]
MOCK_DIR = REPO_ROOT / "data" / "mock"


def test_agent_modules_do_not_import_adapters() -> None:
    agents_root = REPO_ROOT / "backend" / "app" / "agents"
    if not agents_root.exists():
        pytest.skip("agents package not present yet")
    for path in agents_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "adapters" not in alias.name, path
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith("app.adapters"), path


@pytest.mark.asyncio
async def test_source_round_trip_preserves_ids_and_raw(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    quality = InMemoryDataQualityRecorder()
    adapter = MockXDRSourceAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
        quality=quality,
    )
    pages = {
        kind: await adapter.list_objects([kind], limit=100)
        for kind in (
            SourceObjectKind.INCIDENT,
            SourceObjectKind.ALERT,
            SourceObjectKind.ASSET,
            SourceObjectKind.LOG,
        )
    }
    incidents = [i for i in pages[SourceObjectKind.INCIDENT].items if isinstance(i, SourceIncident)]
    alerts = [i for i in pages[SourceObjectKind.ALERT].items if isinstance(i, SourceAlert)]
    assets = [i for i in pages[SourceObjectKind.ASSET].items if isinstance(i, SourceAsset)]
    logs = [i for i in pages[SourceObjectKind.LOG].items if isinstance(i, SourceLog)]
    assert incidents and alerts and assets and logs

    scenario = build_scenario("insider_data_exfiltration", seed=42)
    assert {i.reference.source_object_id for i in incidents} == {
        i.reference.source_object_id for i in scenario.incidents
    }
    # Parent/child refs survive.
    for alert in alerts:
        if alert.incident_ref is not None:
            assert alert.incident_ref.source_object_id in {
                i.reference.source_object_id for i in incidents
            }
    for log in logs:
        parent = log.reference.parent_source_object_id
        if parent:
            assert parent in {a.reference.source_object_id for a in alerts} or parent in {
                a.reference.source_object_id for a in assets
            }
    # raw_payload / mock metadata retained
    assert any("_mock" in (i.raw_payload or {}) for i in incidents + alerts)

    fetched = await adapter.get_object(SourceObjectKind.INCIDENT, "88442201")
    assert isinstance(fetched, SourceIncident)
    assert fetched.reference.source_object_id == "88442201"
    assert fetched.raw_payload is not None


@pytest.mark.asyncio
async def test_disposition_idempotent_submit(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    adapter = MockXDRDispositionAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
    )
    # Use current concurrency token from store.
    stored = mock_state.objects[("incident", "88442201")]
    cmd = event_disposition_command(token=stored.concurrency_token)
    first = await adapter.submit(cmd)
    second = await adapter.submit(cmd)
    assert first.writeback_id == second.writeback_id
    assert first.status == second.status
    assert first.status in {WritebackStatus.ACCEPTED, WritebackStatus.PARTIAL}


@pytest.mark.asyncio
async def test_disposition_concurrency_token_conflict(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    adapter = MockXDRDispositionAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
    )
    cmd = event_disposition_command(
        token="stale-token",
        idempotency_key="idem-conflict-1",
        disposition_id="disp-conflict-1",
    )
    receipt = await adapter.submit(cmd)
    assert receipt.status is WritebackStatus.CONFLICT
    assert receipt.provider_code == "version_conflict"


@pytest.mark.asyncio
async def test_disposition_rejects_analysis_fields(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    stored = mock_state.objects[("incident", "88442201")]
    cmd = event_disposition_command(token=stored.concurrency_token)
    dumped = cmd.model_dump(mode="json")
    dumped["operation_params"]["decision_trace"] = "secret analysis"
    resp = await mock_client.post(
        "/mock-xdr/v1/dispositions",
        headers={"Authorization": f"Bearer {mock_state.write_token}"},
        json=dumped,
    )
    assert resp.status_code == 422
    body = resp.json()
    # FastAPI HTTPException wraps the Mock error envelope under ``detail``.
    detail = body.get("detail", body)
    assert detail["error_code"] == "unauthorized_field"


@pytest.mark.asyncio
async def test_lost_response_returns_unknown_then_lookup(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    adapter = MockXDRDispositionAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
    )
    stored = mock_state.objects[("incident", "88442201")]
    cmd = event_disposition_command(
        token=stored.concurrency_token,
        idempotency_key="idem-lost-1",
        disposition_id="disp-lost-1",
    )
    # First succeed so lookup can find it after a simulated loss.
    accepted = await adapter.submit(cmd)
    assert accepted.status is not WritebackStatus.FAILED

    lost = await adapter._unknown_after_loss(cmd)
    # Lookup recovers the prior acceptance — never re-executes entity action.
    assert lost.writeback_id == accepted.writeback_id
    assert lost.status == accepted.status


@pytest.mark.asyncio
async def test_submit_and_lookup_transport_failures_return_unknown(
    mock_state: MockXDRState,
) -> None:
    async def fail_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("deterministic transport failure", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(fail_transport),
        base_url="http://mock-xdr",
    ) as client:
        adapter = MockXDRDispositionAdapter(
            read_token=mock_state.read_token,
            write_token=mock_state.write_token,
            client=client,
        )
        stored = mock_state.objects[("incident", "88442201")]
        command = event_disposition_command(
            token=stored.concurrency_token,
            idempotency_key="idem-double-transport",
            disposition_id="disp-double-transport",
        )
        receipt = await adapter.submit(command)

    assert receipt.status is WritebackStatus.UNKNOWN
    assert receipt.confirmation_evidence is None
    assert receipt.provider_code == "unknown_delivery"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("future_vendor_field", {"revision": 2}),
        ("new_scalar", "opaque"),
        ("new_collection", [1, 2, 3]),
    ],
)
async def test_unknown_source_fields_are_folded_into_raw_payload(
    mock_client: httpx.AsyncClient,
    mock_state: MockXDRState,
    field_name: str,
    field_value: Any,
) -> None:
    stored = mock_state.objects[("incident", "88442201")]
    body = dict(stored.body)
    body[field_name] = field_value
    mock_state.upsert_object("incident", "88442201", body)
    adapter = MockXDRSourceAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
    )

    item = await adapter.get_object(SourceObjectKind.INCIDENT, "88442201")

    assert isinstance(item, SourceIncident)
    assert item.title
    assert item.raw_payload[field_name] == field_value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "expected_code"),
    [
        (MockFailureProfile(rate_limit_every_n=1), "rate_limited"),
        (MockFailureProfile(timeout_every_n=1), "timeout"),
    ],
)
async def test_source_fault_profiles_have_deterministic_errors(
    mock_client: httpx.AsyncClient,
    mock_state: MockXDRState,
    profile: MockFailureProfile,
    expected_code: str,
) -> None:
    mock_state.failure_profile = profile
    adapter = MockXDRSourceAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
        max_retries=0,
    )
    with pytest.raises(DependencyUnavailableError) as exc_info:
        await adapter.list_objects([SourceObjectKind.INCIDENT])
    assert exc_info.value.error_code == expected_code


@pytest.mark.asyncio
async def test_malformed_source_payload_marks_adapter_degraded(
    mock_client: httpx.AsyncClient,
    mock_state: MockXDRState,
) -> None:
    mock_state.failure_profile = MockFailureProfile(malformed_payload_every_n=1)
    quality = InMemoryDataQualityRecorder()
    adapter = MockXDRSourceAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
        quality=quality,
        max_retries=0,
    )

    page = await adapter.list_objects([SourceObjectKind.INCIDENT])

    assert page.items == []
    assert page.malformed_items == 1
    assert any(row["error_category"] == "malformed_payload" for row in quality.rows)
    assert await adapter.health_check() is ConnectorStatus.ONLINE


def test_source_schema_quality_error_does_not_record_rejected_input() -> None:
    quality = InMemoryDataQualityRecorder()
    secret = "Bearer source-quality-secret"

    item = parse_source_item(
        SourceObjectKind.INCIDENT.value,
        {"impacted_asset_refs": [secret]},
        quality=quality,
    )

    assert item is None
    assert quality.rows
    assert secret not in str(quality.rows)
    errors = quality.rows[0]["detail"]["errors"]
    assert all("input" not in error and "url" not in error for error in errors)


@pytest.mark.asyncio
async def test_malformed_submit_and_lookup_payloads_return_unknown(
    mock_client: httpx.AsyncClient,
    mock_state: MockXDRState,
) -> None:
    mock_state.failure_profile = MockFailureProfile(malformed_payload_every_n=1)
    adapter = MockXDRDispositionAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
    )
    stored = mock_state.objects[("incident", "88442201")]
    command = event_disposition_command(
        token=stored.concurrency_token,
        idempotency_key="idem-malformed",
        disposition_id="disp-malformed",
    )

    receipt = await adapter.submit(command)

    assert receipt.status is WritebackStatus.UNKNOWN
    assert receipt.confirmation_evidence is None


@pytest.mark.asyncio
async def test_file_source_has_no_writeback() -> None:
    adapter = FileSourceAdapter(scenario_id="insider_data_exfiltration", mock_dir=MOCK_DIR)
    assert adapter.writeback_required is False
    assert not isinstance(adapter, BaseDispositionAdapter)
    assert (
        adapter.capabilities()[ConnectorCapability.EVENT_DISPOSITION] is CapabilityState.UNSUPPORTED
    )
    page = await adapter.list_objects([SourceObjectKind.INCIDENT])
    assert page.items
    telemetry = adapter.load_telemetry()
    assert "identity" in telemetry
    assert any(normalize_record(r).channel == "identity" for r in telemetry["identity"])


def test_normalizers_cover_six_plus_threat_intel_channels() -> None:
    # Six deep-dive channels + threat_intel used by Evidence — not six XDR adapters.
    assert set(CHANNEL_NORMALIZERS) >= {
        "identity",
        "endpoint",
        "dlp",
        "network",
        "dns",
        "asset",
        "threat_intel",
    }


def test_registries_and_adapter_not_found() -> None:
    sources = SourceAdapterRegistry()
    dispositions = DispositionAdapterRegistry()
    file_adapter = FileSourceAdapter(scenario_id="account_anomaly_fp")
    sources.register("file", file_adapter)
    assert sources.get("file") is file_adapter
    with pytest.raises(AdapterNotFoundError):
        sources.get("missing")
    with pytest.raises(AdapterNotFoundError):
        dispositions.get("missing")


def test_live_stub_capabilities_are_unknown() -> None:
    stub = LiveDispositionAdapterStub()
    caps = stub.capabilities()
    assert all(v is CapabilityState.UNKNOWN for v in caps.intents.values())
    cmd = event_disposition_command()
    with pytest.raises(WritebackUnsupportedError):
        stub.validate_command(cmd)


def test_mock_requires_separated_credentials() -> None:
    with pytest.raises(ValueError, match="separated"):
        MockXDRSourceAdapter(read_token="same", write_token="same")


@pytest.mark.asyncio
async def test_schema_unsupported_halts_kind_watermark(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    quality = InMemoryDataQualityRecorder()
    adapter = MockXDRSourceAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
        quality=quality,
        supported_schema_versions=frozenset({"999"}),
    )
    page = await adapter.list_objects([SourceObjectKind.INCIDENT], limit=10)
    assert page.items == []
    assert any(r["error_category"] == "schema_unsupported" for r in quality.rows)
    assert await adapter.health_check() is ConnectorStatus.ONLINE


@pytest.mark.asyncio
async def test_list_objects_paginates_single_kind(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    adapter = MockXDRSourceAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
    )
    scenario = build_scenario("insider_data_exfiltration", seed=42)
    expected = {a.reference.source_object_id for a in scenario.alerts}
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(10):  # safety bound against runaway pagination
        page = await adapter.list_objects([SourceObjectKind.ALERT], cursor=cursor, limit=1)
        seen.extend(a.reference.source_object_id for a in page.items if isinstance(a, SourceAlert))
        if not page.has_more:
            break
        cursor = page.next_cursor
    assert set(seen) == expected  # every alert retrieved
    assert len(seen) == len(set(seen))  # no duplicates across pages


@pytest.mark.asyncio
async def test_list_objects_rejects_multi_kind_request(
    mock_client: httpx.AsyncClient, mock_state: MockXDRState
) -> None:
    adapter = MockXDRSourceAdapter(
        read_token=mock_state.read_token,
        write_token=mock_state.write_token,
        client=mock_client,
    )
    with pytest.raises(ValueError, match="exactly one"):
        await adapter.list_objects(
            [SourceObjectKind.ALERT, SourceObjectKind.ASSET],
            limit=1,
        )


def test_source_adapter_has_no_write_methods() -> None:
    writeish = {"submit", "write", "delete", "update", "dispose"}
    public = {name for name in dir(BaseSourceAdapter) if not name.startswith("_")}
    assert public.isdisjoint(writeish)
