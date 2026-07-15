"""Scenario models for MockXDRServer (ISSUE-010)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.source import (
    SourceAlert,
    SourceAsset,
    SourceConnector,
    SourceIncident,
    SourceLog,
)


class TickOperation(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"
    CONNECTOR_CHANGE = "connector_change"


class ScenarioVariant(StrEnum):
    NORMAL = "normal"
    AGENT_NOT_INSTALLED = "agent_not_installed"
    DEVICE_OFFLINE = "device_offline"
    CAPABILITY_GAP = "capability_gap"
    PARTIAL_SUCCESS = "partial_success"
    CAPACITY_LIMIT_EXCEEDED = "capacity_limit_exceeded"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    MALFORMED_PAYLOAD = "malformed_payload"


class ScenarioTick(BaseModel):
    """One deterministic timeline mutation against the Mock store."""

    model_config = ConfigDict(extra="forbid")

    offset_seconds: int = Field(ge=0)
    operation: TickOperation
    object_type: str
    object_id: str
    patch: dict[str, Any] = Field(default_factory=dict)


class MockFailureProfile(BaseModel):
    """Injectable integration-fault profile (deterministic under ``seed``)."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    fixed_delay_ms: int = Field(default=0, ge=0)
    jitter_delay_ms: int = Field(default=0, ge=0)
    rate_limit_every_n: int | None = Field(default=None, gt=0)
    server_error_every_n: int | None = Field(default=None, gt=0)
    timeout_every_n: int | None = Field(default=None, gt=0)
    malformed_payload_every_n: int | None = Field(default=None, gt=0)
    duplicate_page: bool = False
    late_data: bool = False
    out_of_order_updates: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    schema_version_override: str | None = None
    async_disposition: bool = False
    force_token_conflict: bool = False
    force_partial_targets: bool = False
    reject_unauthorized_fields: bool = True
    # Test/demo control plane (never for production live adapters).
    control_plane_enabled: bool = True


class MockXDRScenario(BaseModel):
    """Self-contained Mock XDR scenario seed (ISSUE-010 naming)."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    variant: ScenarioVariant = ScenarioVariant.NORMAL
    base_time: datetime
    source_tenant_id: str
    incidents: list[SourceIncident] = Field(default_factory=list)
    alerts: list[SourceAlert] = Field(default_factory=list)
    assets: list[SourceAsset] = Field(default_factory=list)
    logs: list[SourceLog] = Field(default_factory=list)
    connectors: list[SourceConnector] = Field(default_factory=list)
    telemetry_timeline: list[dict[str, Any]] = Field(default_factory=list)
    ticks: list[ScenarioTick] = Field(default_factory=list)
    failure_profile: MockFailureProfile = Field(default_factory=MockFailureProfile)
    expected_outcome: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_referential_consistency(self) -> MockXDRScenario:
        incident_ids = {i.reference.source_object_id for i in self.incidents}
        alert_ids = {a.reference.source_object_id for a in self.alerts}
        asset_ids = {a.reference.source_object_id for a in self.assets}
        log_ids = {log.reference.source_object_id for log in self.logs}
        connector_ids = {connector.connector_id for connector in self.connectors}

        collections = (
            ("incident", self.incidents, incident_ids),
            ("alert", self.alerts, alert_ids),
            ("asset", self.assets, asset_ids),
            ("log", self.logs, log_ids),
        )
        for kind, objects, ids in collections:
            if len(ids) != len(objects):
                raise ValueError(f"duplicate {kind} source_object_id")
            for obj in objects:
                ref = obj.reference
                if ref.source_kind.value != kind:
                    raise ValueError(
                        f"{kind} {ref.source_object_id} has source_kind={ref.source_kind.value}"
                    )
                if ref.source_tenant_id != self.source_tenant_id:
                    raise ValueError(f"{kind} {ref.source_object_id} belongs to a different tenant")
                if ref.connector_id not in connector_ids:
                    raise ValueError(
                        f"{kind} {ref.source_object_id} references missing connector "
                        f"{ref.connector_id}"
                    )
        if len(connector_ids) != len(self.connectors):
            raise ValueError("duplicate connector_id")

        for incident in self.incidents:
            for ref in incident.related_alert_refs:
                if ref.source_kind.value != "alert":
                    raise ValueError(
                        f"incident {incident.reference.source_object_id} has non-alert relation"
                    )
                if ref.source_object_id not in alert_ids:
                    raise ValueError(
                        f"incident {incident.reference.source_object_id} references "
                        f"missing alert {ref.source_object_id}"
                    )
            for ref in incident.impacted_asset_refs:
                if ref.source_kind.value != "asset":
                    raise ValueError(
                        f"incident {incident.reference.source_object_id} has non-asset relation"
                    )
                if ref.source_object_id not in asset_ids:
                    raise ValueError(
                        f"incident {incident.reference.source_object_id} references "
                        f"missing asset {ref.source_object_id}"
                    )

        for alert in self.alerts:
            if alert.incident_ref is not None:
                if alert.incident_ref.source_kind.value != "incident":
                    raise ValueError(
                        f"alert {alert.reference.source_object_id} has non-incident parent"
                    )
                iid = alert.incident_ref.source_object_id
                if iid not in incident_ids:
                    raise ValueError(
                        f"alert {alert.reference.source_object_id} points to missing incident {iid}"
                    )
            for ref in alert.related_log_refs:
                if ref.source_kind.value != "log":
                    raise ValueError(
                        f"alert {alert.reference.source_object_id} has non-log relation"
                    )
                if ref.source_object_id not in log_ids:
                    raise ValueError(
                        f"alert {alert.reference.source_object_id} references "
                        f"missing log {ref.source_object_id}"
                    )
            for ref in alert.sub_alert_refs:
                if ref.source_kind.value != "alert":
                    raise ValueError(
                        f"alert {alert.reference.source_object_id} has non-alert child"
                    )
                if ref.source_object_id not in alert_ids:
                    raise ValueError(
                        f"alert {alert.reference.source_object_id} references "
                        f"missing sub-alert {ref.source_object_id}"
                    )

        for log in self.logs:
            parent = log.reference.parent_source_object_id
            if parent is None:
                continue
            if parent not in alert_ids and parent not in asset_ids and parent not in incident_ids:
                raise ValueError(
                    f"log {log.reference.source_object_id} parent {parent} does not exist"
                )

        known_by_kind = {
            "incident": incident_ids,
            "alert": alert_ids,
            "asset": asset_ids,
            "log": log_ids,
            "connector": connector_ids,
        }
        for tick in self.ticks:
            if tick.object_type not in known_by_kind:
                raise ValueError(f"tick references unknown object_type {tick.object_type}")
            if tick.operation is TickOperation.CONNECTOR_CHANGE:
                if tick.object_type != "connector" or tick.object_id not in connector_ids:
                    raise ValueError("connector_change tick must reference an existing connector")
            elif tick.object_type == "connector":
                raise ValueError("connector ticks must use connector_change")
            elif (
                tick.operation is TickOperation.DELETE
                and tick.object_id not in known_by_kind[tick.object_type]
            ):
                raise ValueError(
                    f"delete tick references missing {tick.object_type} {tick.object_id}"
                )

        return self
