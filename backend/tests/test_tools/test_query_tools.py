"""ISSUE-019 evidence projection and baseline query tool tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.adapters.file_source import FileSourceAdapter
from app.adapters.mock_xdr import MockXDRSourceAdapter
from app.core.errors import GuardrailViolationError
from app.data_generators.scenarios import build_scenario
from app.ingestion.source_ingester import IngestionSummary, SourceIngester
from app.mock_xdr.api import create_app
from app.mock_xdr.state import MockXDRState
from app.models.enums import ConnectorStatus, ToolCategory
from app.services.evidence_projection import (
    EvidenceProjection,
    EvidenceQueryScope,
    bind_evidence_projection,
)
from app.tools.query.fixture_loader import load_fixture_records
from app.tools.registry import ToolRegistry, ToolValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
MOCK_DATA = REPO_ROOT / "data" / "mock"
WINDOW = {
    "start": "2024-06-15T08:00:00Z",
    "end": "2024-06-15T10:00:00Z",
}
OUTSIDE_WINDOW = {
    "start": "2023-01-01T00:00:00Z",
    "end": "2023-01-01T01:00:00Z",
}
QUERY_NAMES = {
    "query_account_login",
    "query_edr_process",
    "query_file_access",
    "query_network_flow",
    "query_dns",
    "query_asset_info",
    "query_vuln_info",
    "query_threat_intel",
    "query_history_cases",
}
DEFAULT_SCOPE = EvidenceQueryScope(
    source_tenant_id="test-tenant",
    connector_ids=frozenset({"fixture-evidence"}),
)


def _scope(source_tenant_id: str, connector_id: str) -> EvidenceQueryScope:
    return EvidenceQueryScope(
        source_tenant_id=source_tenant_id,
        connector_ids=frozenset({connector_id}),
    )


class _EventScopeService:
    def __init__(self, scope: EvidenceQueryScope) -> None:
        self.scope = scope
        self.requested_event_ids: list[str] = []

    async def get_evidence_query_scope(self, event_id: str) -> EvidenceQueryScope:
        self.requested_event_ids.append(event_id)
        return self.scope


@pytest_asyncio.fixture
async def projection() -> EvidenceProjection:
    projection = EvidenceProjection.in_memory()
    loaded = await load_fixture_records(projection, MOCK_DATA)
    assert loaded > 0
    await projection.ingest_records(
        {
            "asset": [
                {
                    "record_id": "vuln-fixture-1",
                    "channel": "asset",
                    "logged_at": "2024-06-15T09:00:00Z",
                    "ip": "10.20.30.23",
                    "hostname": "PC-FIN-023",
                    "cve": "CVE-2024-0001",
                    "cvss": 8.1,
                }
            ],
            "history_cases": [
                {
                    "record_id": "case-fixture-1",
                    "channel": "history_cases",
                    "logged_at": "2024-06-15T09:00:00Z",
                    "case_id": "case-fixture-1",
                    "title": "Finance endpoint data exfiltration",
                    "description": "PowerShell archive upload to unknown infrastructure",
                    "final_verdict": "confirmed_threat",
                }
            ],
        },
        source_product="fixture",
        source_tenant_id="test-tenant",
        connector_id="fixture-evidence",
        watermark={"cursor": None, "scenario": "insider_data_exfiltration"},
    )
    return projection


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.auto_discover(include_virtual=False)
    return registry


async def _run_tool(
    registry: ToolRegistry,
    projection: EvidenceProjection,
    tool_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    event_service = _EventScopeService(DEFAULT_SCOPE)
    with bind_evidence_projection(projection):
        result = await registry.execute_event_query(
            "evt-query-test",
            tool_name,
            params,
            event_service=cast(Any, event_service),
        )
    assert event_service.requested_event_ids == ["evt-query-test"]
    return result


def test_registry_discovers_all_nine_baseline_query_implementations(
    registry: ToolRegistry,
) -> None:
    query_metas = [entry.tool_meta for entry in registry.list_registered_tools(ToolCategory.QUERY)]
    query_meta_by_name = {meta.tool_name: meta for meta in query_metas}
    assert QUERY_NAMES.issubset(query_meta_by_name)
    assert all(query_meta_by_name[name].output_schema for name in QUERY_NAMES)

    async def _third_party_query(params: dict[str, Any]) -> dict[str, Any]:
        return {"data": params}

    registry.register(
        query_metas[0].model_copy(
            update={
                "tool_name": "third_party_query_extension",
                "input_schema": {},
                "output_schema": {},
            }
        ),
        _third_party_query,
    )
    extended_names = {
        entry.tool_meta.tool_name for entry in registry.list_registered_tools(ToolCategory.QUERY)
    }
    assert QUERY_NAMES.issubset(extended_names)
    assert "third_party_query_extension" in extended_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "params", "expected_record"),
    [
        (
            "query_account_login",
            {"account": "zhangsan", "time_range": WINDOW},
            "id-conflict-42-0002",
        ),
        (
            "query_edr_process",
            {"host_id": "PC-FIN-023", "time_range": WINDOW},
            "ep-conflict-42-0003",
        ),
        (
            "query_file_access",
            {"account": "zhangsan", "time_range": WINDOW},
            "ep-key-42-0005",
        ),
        (
            "query_network_flow",
            {"src_ip": "10.20.30.23", "time_range": WINDOW},
            "net-key-42-0009",
        ),
        (
            "query_dns",
            {"domain": "unknown-upload-example.com", "time_range": WINDOW},
            "dns-key-42-0008",
        ),
        (
            "query_asset_info",
            {"ip": "10.20.30.23"},
            "asset-key-42-0013",
        ),
        (
            "query_vuln_info",
            {"hostname": "PC-FIN-023"},
            "vuln-fixture-1",
        ),
        (
            "query_threat_intel",
            {"indicator": "203.0.113.88"},
            "ti-key-42-0011",
        ),
        (
            "query_history_cases",
            {"pattern_description": "PowerShell data exfiltration"},
            "case-fixture-1",
        ),
    ],
)
async def test_each_query_returns_traceable_schema_valid_records(
    registry: ToolRegistry,
    projection: EvidenceProjection,
    tool_name: str,
    params: dict[str, Any],
    expected_record: str,
) -> None:
    result = await _run_tool(registry, projection, tool_name, params)

    assert result["status"] == "success"
    assert any(row.get("record_id") == expected_record for row in result["data"]["records"])
    assert result["data"]["source_references"]
    assert result["data"]["data_freshness"]["state"] == "fresh"
    assert result["data"]["watermark"] is not None
    assert result["confidence"] is not None
    if tool_name == "query_history_cases":
        assert result["data"]["degraded"] is True
        assert "vector_store_unavailable_keyword_fallback" in result["data"]["coverage"]["reasons"]


@pytest.mark.asyncio
async def test_direct_unbound_query_fails_closed_and_request_cannot_supply_scope(
    registry: ToolRegistry,
    projection: EvidenceProjection,
) -> None:
    params = {"domain": "unknown-upload-example.com", "time_range": WINDOW}
    with bind_evidence_projection(projection):
        with pytest.raises(GuardrailViolationError, match="trusted event scope"):
            await registry.get_tool("query_dns").execute(params)

        with pytest.raises(ToolValidationError, match="cannot be supplied"):
            await registry.execute_event_query(
                "evt-query-test",
                "query_dns",
                {**params, "source_tenant_id": "attacker", "connector_ids": ["all"]},
                event_service=cast(Any, _EventScopeService(DEFAULT_SCOPE)),
            )


@pytest.mark.asyncio
async def test_main_scenario_account_evidence_preserves_no_record_fact(
    registry: ToolRegistry,
    projection: EvidenceProjection,
) -> None:
    result = await _run_tool(
        registry,
        projection,
        "query_account_login",
        {"account": "zhangsan", "time_range": WINDOW},
    )
    records = result["data"]["records"]
    assert {row["result"] for row in records} == {"no_record"}
    assert not any(row.get("result") == "success" for row in records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "params"),
    [
        ("query_account_login", {"account": "nobody", "time_range": WINDOW}),
        ("query_edr_process", {"host_id": "missing-host", "time_range": WINDOW}),
        ("query_file_access", {"account": "nobody", "time_range": WINDOW}),
        (
            "query_network_flow",
            {"dst_ip": "198.51.100.254", "time_range": WINDOW},
        ),
        ("query_dns", {"domain": "missing.invalid", "time_range": WINDOW}),
        ("query_asset_info", {"hostname": "missing-host"}),
        ("query_vuln_info", {"ip": "198.51.100.254"}),
        ("query_threat_intel", {"indicator": "missing.invalid"}),
        ("query_history_cases", {"pattern_description": "unrelated zebra token"}),
    ],
)
async def test_nonexistent_entity_is_successful_empty_result(
    registry: ToolRegistry,
    projection: EvidenceProjection,
    tool_name: str,
    params: dict[str, Any],
) -> None:
    result = await _run_tool(registry, projection, tool_name, params)
    assert result["status"] == "success"
    assert result["data"]["records"] == []
    assert result["data"]["coverage"]["state"] in {"complete", "partial"}
    if tool_name != "query_history_cases":
        assert result["data"]["degraded"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "params"),
    [
        (
            "query_account_login",
            {"account": "zhangsan", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_edr_process",
            {"host_id": "PC-FIN-023", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_file_access",
            {"account": "zhangsan", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_network_flow",
            {"src_ip": "10.20.30.23", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_dns",
            {"domain": "unknown-upload-example.com", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_asset_info",
            {"ip": "10.20.30.23", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_vuln_info",
            {"hostname": "PC-FIN-023", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_threat_intel",
            {"indicator": "203.0.113.88", "time_range": OUTSIDE_WINDOW},
        ),
        (
            "query_history_cases",
            {
                "pattern_description": "data exfiltration",
                "time_range": OUTSIDE_WINDOW,
            },
        ),
    ],
)
async def test_each_query_applies_time_filter(
    registry: ToolRegistry,
    projection: EvidenceProjection,
    tool_name: str,
    params: dict[str, Any],
) -> None:
    result = await _run_tool(registry, projection, tool_name, params)
    assert result["data"]["records"] == []


@pytest.mark.asyncio
async def test_projection_pagination_uses_opaque_cursor(
    projection: EvidenceProjection,
) -> None:
    first = await projection.query(
        "network_flow",
        {"src_ip": "10.20.30.23"},
        None,
        cursor=None,
        limit=1,
        scope=DEFAULT_SCOPE,
    )
    assert len(first.records) == 1
    assert first.next_cursor is not None
    assert first.next_cursor.startswith("evp1:")

    second = await projection.query(
        "network_flow",
        {"src_ip": "10.20.30.23"},
        None,
        cursor=first.next_cursor,
        limit=1,
        scope=DEFAULT_SCOPE,
    )
    assert len(second.records) == 1
    assert second.records[0]["record_id"] != first.records[0]["record_id"]

    with pytest.raises(ValueError, match="invalid evidence projection cursor"):
        await projection.query(
            "network_flow",
            {},
            None,
            cursor="invalid",
            limit=1,
            scope=DEFAULT_SCOPE,
        )


@pytest.mark.asyncio
async def test_projection_enforces_tenant_connector_scope_and_cursor_fingerprint() -> None:
    projection = EvidenceProjection.in_memory()
    for tenant_id, connector_id, record_id in (
        ("tenant-a", "connector-a", "allowed"),
        ("tenant-b", "connector-a", "wrong-tenant"),
        ("tenant-a", "connector-b", "wrong-connector"),
    ):
        await projection.ingest_records(
            {
                "dns": [
                    {
                        "record_id": record_id,
                        "channel": "dns",
                        "logged_at": datetime.now(UTC).isoformat(),
                        "query": "isolated.example",
                    }
                ]
            },
            source_product="fixture",
            source_tenant_id=tenant_id,
            connector_id=connector_id,
        )

    allowed_scope = _scope("tenant-a", "connector-a")
    result = await projection.query(
        "dns",
        {"domain": "isolated.example"},
        None,
        None,
        1,
        scope=allowed_scope,
    )
    assert [row["record_id"] for row in result.records] == ["allowed"]

    await projection.ingest_records(
        {
            "dns": [
                {
                    "record_id": "allowed-second",
                    "channel": "dns",
                    "logged_at": (datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
                    "query": "isolated.example",
                }
            ]
        },
        source_product="fixture",
        source_tenant_id="tenant-a",
        connector_id="connector-a",
    )
    first = await projection.query(
        "dns",
        {"domain": "isolated.example"},
        None,
        None,
        1,
        scope=allowed_scope,
    )
    assert first.next_cursor is not None
    with pytest.raises(ValueError, match="invalid evidence projection cursor"):
        await projection.query(
            "dns",
            {"domain": "isolated.example"},
            None,
            first.next_cursor,
            1,
            scope=_scope("tenant-b", "connector-a"),
        )


@pytest.mark.asyncio
async def test_projection_cursor_freezes_rows_inserted_after_first_page() -> None:
    projection = EvidenceProjection.in_memory()
    base = datetime.now(UTC) - timedelta(minutes=10)
    await projection.ingest_records(
        {
            "dns": [
                {
                    "record_id": f"original-{index}",
                    "channel": "dns",
                    "logged_at": (base + timedelta(seconds=index)).isoformat(),
                    "query": "stable.example",
                }
                for index in range(3)
            ]
        },
        source_product="fixture",
        source_tenant_id="stable-tenant",
        connector_id="stable-connector",
        ingested_at=base + timedelta(seconds=3),
    )
    scope = _scope("stable-tenant", "stable-connector")
    first = await projection.query(
        "dns",
        {"domain": "stable.example"},
        None,
        None,
        1,
        scope=scope,
    )
    assert [row["record_id"] for row in first.records] == ["original-0"]

    await projection.ingest_records(
        {
            "dns": [
                {
                    "record_id": "late-indexed",
                    "channel": "dns",
                    "logged_at": (base + timedelta(milliseconds=500)).isoformat(),
                    "query": "stable.example",
                }
            ]
        },
        source_product="fixture",
        source_tenant_id="stable-tenant",
        connector_id="stable-connector",
        # A caller-controlled timestamp predating page 1 cannot evade the
        # projection's trustworthy insertion/indexed cutoff.
        ingested_at=base - timedelta(days=1),
    )
    second = await projection.query(
        "dns",
        {"domain": "stable.example"},
        None,
        first.next_cursor,
        1,
        scope=scope,
    )

    assert [row["record_id"] for row in second.records] == ["original-1"]


@pytest.mark.asyncio
async def test_missing_stale_and_offline_projection_are_degraded() -> None:
    missing = EvidenceProjection.in_memory()
    missing_result = await missing.query(
        "dns",
        {"domain": "example.invalid"},
        None,
        None,
        10,
        scope=_scope("missing-tenant", "missing-connector"),
    )
    assert missing_result.degraded is True
    assert missing_result.data_freshness.state == "missing"
    assert missing_result.coverage.state == "missing"

    asset_only = EvidenceProjection.in_memory()
    await load_fixture_records(asset_only, MOCK_DATA)
    vuln_gap = await asset_only.query(
        "vuln_info",
        {"ip": "10.20.30.23"},
        None,
        None,
        10,
        scope=DEFAULT_SCOPE,
    )
    assert vuln_gap.records == []
    assert vuln_gap.degraded is True
    assert vuln_gap.coverage.state == "missing"

    stale = EvidenceProjection.in_memory(stale_after=timedelta(minutes=5))
    await stale.ingest_records(
        {
            "dns": [
                {
                    "record_id": "stale-dns",
                    "channel": "dns",
                    "logged_at": "2024-06-15T09:00:00Z",
                    "query": "stale.example",
                }
            ]
        },
        source_product="fixture",
        source_tenant_id="test",
        connector_id="stale",
        ingested_at=datetime(2024, 6, 15, 9, 1, tzinfo=UTC),
    )
    stale_result = await stale.query(
        "dns",
        {"domain": "stale.example"},
        None,
        None,
        10,
        scope=_scope("test", "stale"),
    )
    assert stale_result.degraded is True
    assert stale_result.data_freshness.state == "stale"
    assert stale_result.coverage.reasons == ["projection_stale"]

    offline = EvidenceProjection.in_memory()
    await offline.ingest_records(
        {
            "dns": [
                {
                    "record_id": "offline-dns",
                    "channel": "dns",
                    "logged_at": datetime.now(UTC).isoformat(),
                    "query": "offline.example",
                }
            ]
        },
        source_product="fixture",
        source_tenant_id="test",
        connector_id="offline",
        connector_status=ConnectorStatus.OFFLINE,
    )
    await offline.ingest_records(
        {
            "dns": [
                {
                    "record_id": "offline-dns",
                    "channel": "dns",
                    "logged_at": datetime.now(UTC).isoformat(),
                    "query": "offline.example",
                }
            ]
        },
        source_product="fixture",
        source_tenant_id="test",
        connector_id="offline",
        connector_status=ConnectorStatus.ONLINE,
    )
    offline_result = await offline.query(
        "dns",
        {"domain": "offline.example"},
        None,
        None,
        10,
        scope=_scope("test", "offline"),
    )
    assert offline_result.degraded is True
    assert offline_result.coverage.state == "partial"
    assert offline_result.coverage.unavailable_connectors == ["offline"]


@pytest.mark.asyncio
async def test_fixture_loader_is_idempotent(projection: EvidenceProjection) -> None:
    replay = await load_fixture_records(projection, MOCK_DATA)
    assert replay == 0


@pytest.mark.asyncio
async def test_source_ingester_projects_adapter_telemetry_through_shared_hook() -> None:
    projection = EvidenceProjection.in_memory()
    ingester = SourceIngester(
        cast(Any, object()),
        cast(Any, object()),
        source_mode="file",
        evidence_projection=projection,
    )
    adapter = FileSourceAdapter(
        scenario_path=MOCK_DATA / "insider_data_exfiltration.scenario.json",
        mock_dir=MOCK_DATA,
    )
    summary = IngestionSummary(
        accepted=10,
        watermark_after={"cursor": None, "updated_after": "2024-06-15T10:00:00Z"},
    )

    await ingester._project_adapter_evidence(
        adapter,
        summary=summary,
    )
    evidence_page = await adapter.list_evidence_records()
    assert evidence_page is not None

    assert summary.degraded is False
    result = await projection.query(
        "account_login",
        {"account": "zhangsan"},
        None,
        None,
        10,
        scope=_scope(evidence_page.source_tenant_id, evidence_page.connector_id),
    )
    assert [row["record_id"] for row in result.records] == ["id-conflict-42-0002"]


@pytest.mark.asyncio
async def test_object_reject_still_projects_adapter_evidence() -> None:
    """Alert/object page rejection must not skip independent evidence projection."""
    from app.models.enums import SourceObjectKind

    projection = EvidenceProjection.in_memory()
    adapter = FileSourceAdapter(
        scenario_path=MOCK_DATA / "insider_data_exfiltration.scenario.json",
        mock_dir=MOCK_DATA,
    )
    ingester = SourceIngester(
        cast(Any, object()),
        cast(Any, object()),
        source_mode="file",
        evidence_projection=projection,
    )

    async def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _no_connectors(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    async def _reject_kind(*_args: Any, **_kwargs: Any) -> IngestionSummary:
        return IngestionSummary(
            rejected=1,
            degraded=True,
            errors=[{"stage": "source_ingest", "error_category": "object_rejected"}],
        )

    ingester._refresh_adapter_connectors = _noop  # type: ignore[method-assign]
    ingester._adapter_connectors = _no_connectors  # type: ignore[method-assign]
    ingester._poll_kind = _reject_kind  # type: ignore[method-assign]
    ingester._mark_adapter_status = _noop  # type: ignore[method-assign]

    summary = await ingester.poll(
        adapter,
        [SourceObjectKind.ALERT, SourceObjectKind.ASSET, SourceObjectKind.LOG],
        batch_size=50,
    )
    assert summary.degraded is True
    assert summary.rejected >= 1
    evidence_page = await adapter.list_evidence_records()
    assert evidence_page is not None

    result = await projection.query(
        "account_login",
        {"account": "zhangsan"},
        None,
        None,
        10,
        scope=_scope(evidence_page.source_tenant_id, evidence_page.connector_id),
    )
    assert [row["record_id"] for row in result.records] == ["id-conflict-42-0002"]


@pytest.mark.asyncio
async def test_mock_xdr_exposes_the_same_normalized_evidence_page() -> None:
    state = MockXDRState()
    state.load_scenario(build_scenario("insider_data_exfiltration", seed=42))
    transport = ASGITransport(app=create_app(state=state))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://mock-xdr",
    ) as client:
        adapter = MockXDRSourceAdapter(
            base_url="http://mock-xdr",
            read_token="mock-read-token",
            write_token="mock-write-token",
            client=client,
            max_retries=0,
        )
        page = await adapter.list_evidence_records()

    assert page is not None
    assert page.source_product == "mock_xdr"
    assert set(page.records_by_source) == {
        "identity",
        "endpoint",
        "dlp",
        "network",
        "dns",
        "asset",
        "threat_intel",
    }
    projection = EvidenceProjection.in_memory()
    await projection.ingest_records(
        page.records_by_source,
        source_product=page.source_product,
        source_tenant_id=page.source_tenant_id,
        connector_id=page.connector_id,
    )
    result = await projection.query(
        "dns",
        {"domain": "unknown-upload-example.com"},
        None,
        None,
        10,
        scope=_scope(page.source_tenant_id, page.connector_id),
    )
    assert result.records[0]["record_id"] == "dns-key-42-0008"
