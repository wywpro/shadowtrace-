"""Shared helpers for Mock XDR scenario packs (ISSUE-011)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.data_generators.base import TELEMETRY_FILENAMES, offset_time
from app.mock_xdr.models import MockFailureProfile, ScenarioVariant
from app.models.enums import (
    CapabilityState,
    ConnectorCapability,
    ConnectorStatus,
    DispositionPolicy,
    SourceDisposition,
    SourceObjectKind,
)
from app.models.source import SourceConnector, SourceReference

PRODUCT = "mock_xdr"
DEFAULT_TENANT = "tenant-demo"
DEFAULT_BASE_TIME = datetime(2024, 6, 15, 9, 0, 0, tzinfo=UTC)
SCENARIO_VARIANTS: tuple[str, ...] = tuple(variant.value for variant in ScenarioVariant)


def normalize_variant(variant: ScenarioVariant | str) -> ScenarioVariant:
    return variant if isinstance(variant, ScenarioVariant) else ScenarioVariant(variant)


def failure_profile_for_variant(
    *,
    seed: int,
    variant: ScenarioVariant,
) -> MockFailureProfile:
    return MockFailureProfile(
        seed=seed,
        force_partial_targets=variant is ScenarioVariant.PARTIAL_SUCCESS,
        rate_limit_every_n=1 if variant is ScenarioVariant.RATE_LIMIT else None,
        timeout_every_n=1 if variant is ScenarioVariant.TIMEOUT else None,
        malformed_payload_every_n=(1 if variant is ScenarioVariant.MALFORMED_PAYLOAD else None),
        control_plane_enabled=True,
    )


def telemetry_for_variant(
    timeline: list[dict[str, Any]],
    *,
    variant: ScenarioVariant,
) -> list[dict[str, Any]]:
    return [
        row for row in timeline if row.get("variant") is None or row.get("variant") == variant.value
    ]


def make_ref(
    kind: SourceObjectKind,
    object_id: str,
    *,
    tenant: str = DEFAULT_TENANT,
    connector_id: str,
    parent: str | None = None,
    disposition: SourceDisposition = SourceDisposition.PENDING,
    status_raw: str | None = None,
    updated_at: datetime | None = None,
    object_type: str | None = None,
) -> SourceReference:
    """Build an opaque external identity — never parse id prefixes in production."""
    return SourceReference(
        source_kind=kind,
        source_product=PRODUCT,
        source_tenant_id=tenant,
        connector_id=connector_id,
        source_object_type=object_type,
        source_object_id=object_id,
        parent_source_object_id=parent,
        source_status_raw=status_raw,
        source_disposition=disposition,
        source_updated_at=updated_at,
        schema_version="1",
    )


def log_only_connector(*, connector_id: str = "conn-log-only") -> SourceConnector:
    return SourceConnector(
        connector_id=connector_id,
        source_product=PRODUCT,
        display_name="Mock Log Ingestion Only",
        device_type="syslog",
        status=ConnectorStatus.ONLINE,
        capabilities={
            ConnectorCapability.LOG_INGESTION: CapabilityState.SUPPORTED,
            ConnectorCapability.QUERY: CapabilityState.UNSUPPORTED,
            ConnectorCapability.EVENT_DISPOSITION: CapabilityState.UNSUPPORTED,
            ConnectorCapability.ENTITY_RESPONSE: CapabilityState.UNSUPPORTED,
        },
        disposition_policy_default=DispositionPolicy.NOT_REQUIRED,
        schema_version="1",
        metadata={"role": "log_only"},
    )


def disposition_connector(
    *,
    connector_id: str = "conn-disposition",
    status: ConnectorStatus = ConnectorStatus.ONLINE,
) -> SourceConnector:
    return SourceConnector(
        connector_id=connector_id,
        source_product=PRODUCT,
        display_name="Mock Disposition / Response Connector",
        device_type="xdr_console",
        status=status,
        capabilities={
            ConnectorCapability.LOG_INGESTION: CapabilityState.UNSUPPORTED,
            ConnectorCapability.QUERY: CapabilityState.SUPPORTED,
            ConnectorCapability.EVENT_DISPOSITION: CapabilityState.SUPPORTED,
            ConnectorCapability.ENTITY_RESPONSE: CapabilityState.SUPPORTED,
        },
        disposition_policy_default=DispositionPolicy.REQUIRED,
        disposition_endpoint="mock://disposition",
        schema_version="1",
        disposition_credential_ref="secret://mock/disposition",
        metadata={"role": "disposition"},
    )


def capability_gap_connector(*, connector_id: str = "conn-capability-gap") -> SourceConnector:
    """Connector missing ENTITY_RESPONSE — capability-gap variant."""
    return SourceConnector(
        connector_id=connector_id,
        source_product=PRODUCT,
        display_name="Mock Query-Only (capability gap)",
        status=ConnectorStatus.DEGRADED,
        capabilities={
            ConnectorCapability.QUERY: CapabilityState.SUPPORTED,
            ConnectorCapability.EVENT_DISPOSITION: CapabilityState.UNKNOWN,
            ConnectorCapability.ENTITY_RESPONSE: CapabilityState.UNSUPPORTED,
        },
        disposition_policy_default=DispositionPolicy.NOT_REQUIRED,
        schema_version="1",
        metadata={
            "role": "capability_gap",
            "provider_error_codes": ["capacity_limit_exceeded"],
        },
    )


def event(
    *,
    channel: str,
    record_id: str,
    offset_s: int,
    base_time: datetime,
    is_key_event: bool = True,
    is_conflict_seed: bool = False,
    **fields: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record_id,
        "channel": channel,
        "logged_at": offset_time(base_time, offset_s).isoformat(),
        "is_key_event": is_key_event,
        "is_conflict_seed": is_conflict_seed,
    }
    row.update(fields)
    return row


def split_telemetry_by_channel(
    timeline: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group timeline rows into the seven fixed telemetry files."""
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in TELEMETRY_FILENAMES}
    for row in timeline:
        channel = str(row.get("channel", ""))
        if channel not in buckets:
            raise ValueError(f"unknown telemetry channel: {channel!r}")
        buckets[channel].append(row)
    return buckets


def count_key_events(timeline: list[dict[str, Any]]) -> int:
    return sum(1 for row in timeline if row.get("is_key_event") is True)


def count_conflict_seeds(timeline: list[dict[str, Any]]) -> int:
    return sum(1 for row in timeline if row.get("is_conflict_seed") is True)
