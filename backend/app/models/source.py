"""External source models (intro §4.3).

These models may carry ``raw_payload`` and are mapped into internal models by the
SourceAdapter. External fields must never leak directly into the Agent business
layer. The investigation-snapshot reference is immutable; only the mutable
``SourceObjectState`` (``current_*`` / ``source_sync_state``) is updated over time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    CapabilityState,
    ConnectorCapability,
    ConnectorStatus,
    DispositionPolicy,
    SourceDisposition,
    SourceObjectKind,
)


class SourceReference(BaseModel):
    """Immutable snapshot reference to an external source object (intro §4.3).

    Identity five-tuple is
    ``(source_product, source_tenant_id, connector_id, source_kind, source_object_id)``.
    The adapter-native ``source_object_type`` and opaque ``source_concurrency_token``
    do NOT participate in identity.
    """

    model_config = ConfigDict(extra="forbid")

    source_kind: SourceObjectKind
    source_product: str
    source_tenant_id: str
    connector_id: str
    source_object_type: str | None = None
    source_object_id: str
    parent_source_object_id: str | None = None
    source_status_raw: str | None = None
    source_disposition: SourceDisposition = SourceDisposition.UNKNOWN
    source_concurrency_token: str | None = None
    source_updated_at: datetime | None = None
    schema_version: str = "1"
    ingested_at: datetime | None = None
    raw_payload_hash: str | None = None

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        """Return the canonical identity five-tuple."""
        return (
            self.source_product,
            self.source_tenant_id,
            self.connector_id,
            self.source_kind.value,
            self.source_object_id,
        )


class SourceObjectState(BaseModel):
    """Mutable current state of a source object; never overwrites the snapshot."""

    model_config = ConfigDict(extra="forbid")

    current_source_status_raw: str | None = None
    current_source_disposition: SourceDisposition = SourceDisposition.UNKNOWN
    current_concurrency_token: str | None = None
    source_sync_state: str | None = None
    updated_at: datetime | None = None


class SourceIncident(BaseModel):
    """External incident. Related-object references are nullable and only filled
    when the adapter actually obtained the relationship (never inferred)."""

    model_config = ConfigDict(extra="forbid")

    reference: SourceReference
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)

    title: str | None = None
    level: str | None = None
    gpt_verdict_label: str | None = None
    impacted_asset_refs: list[SourceReference] = Field(default_factory=list)
    related_alert_refs: list[SourceReference] = Field(default_factory=list)


class SourceAlert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: SourceReference
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)

    incident_ref: SourceReference | None = None
    xff: str | None = None
    source_ip: str | None = None
    related_log_refs: list[SourceReference] = Field(default_factory=list)
    sub_alert_refs: list[SourceReference] = Field(default_factory=list)


class SourceAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: SourceReference
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)

    numeric_asset_id: str | None = None
    ip: str | None = None
    hostname: str | None = None
    asset_name: str | None = None
    asset_group: str | None = None
    owner: str | None = None
    business_system: str | None = None
    importance: str | None = None
    agent_status: str | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None


class SourceLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: SourceReference
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)

    device_source: str | None = None
    logged_at: datetime | None = None
    src_ip: str | None = None
    src_port: int | None = None
    dst_ip: str | None = None
    dst_port: int | None = None
    category: str | None = None


class SourceConnector(BaseModel):
    """Connector descriptor. Secrets are stored only as references (intro §4.3.5)."""

    model_config = ConfigDict(extra="forbid")

    connector_id: str
    source_product: str
    display_name: str
    device_type: str | None = None
    status: ConnectorStatus = ConnectorStatus.UNKNOWN
    read_endpoint: str | None = None
    disposition_endpoint: str | None = None
    capabilities: dict[ConnectorCapability, CapabilityState] = Field(default_factory=dict)
    # No default: live connectors MUST set this explicitly (fail-closed). Mock/file
    # connector factories set an explicit value at construction time.
    disposition_policy_default: DispositionPolicy | None = None
    last_sync_at: datetime | None = None
    schema_version: str = "1"
    read_credential_ref: str | None = None
    disposition_credential_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
