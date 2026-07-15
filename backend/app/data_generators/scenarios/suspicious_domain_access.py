"""Ambiguous scenario: new-domain access without exfil (ISSUE-011)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.data_generators.scenarios._common import (
    DEFAULT_BASE_TIME,
    DEFAULT_TENANT,
    SCENARIO_VARIANTS,
    capability_gap_connector,
    disposition_connector,
    event,
    failure_profile_for_variant,
    log_only_connector,
    make_ref,
    normalize_variant,
    telemetry_for_variant,
)
from app.mock_xdr.models import MockXDRScenario, ScenarioVariant
from app.models.enums import DispositionPolicy, SourceObjectKind
from app.models.source import SourceAlert, SourceAsset, SourceIncident, SourceLog

SCENARIO_ID = "suspicious_domain_access"

OFFICE_HOST = "PC-OFFICE-014"
OFFICE_ACCOUNT = "office-user-014"
NEW_DOMAIN = "brand-new-cdn-example.net"
# Opaque IDs
INCIDENT_ID = "c3d4e5f6-a7b8-9012-cdef-1234567890ab"  # UUID incident
ALERT_ID = "80808080"  # numeric alert
ASSET_ID = "xdr_asset_office_endpoint_014_long_id"  # long unprefixed
LOG_ID = "9091"


def build_suspicious_domain_access(
    *,
    seed: int = 42,
    variant: ScenarioVariant | str = ScenarioVariant.NORMAL,
) -> MockXDRScenario:
    selected_variant = normalize_variant(variant)
    base = DEFAULT_BASE_TIME
    tenant = DEFAULT_TENANT
    conn_log = log_only_connector(connector_id="conn-log-domain")
    conn_disp = disposition_connector(connector_id="conn-disp-domain")
    conn_disp = conn_disp.model_copy(
        update={"disposition_policy_default": DispositionPolicy.NOT_REQUIRED}
    )
    conn_gap = capability_gap_connector(connector_id="conn-gap-domain")

    asset_ref = make_ref(
        SourceObjectKind.ASSET,
        ASSET_ID,
        connector_id=conn_disp.connector_id,
        status_raw="managed",
        updated_at=base,
    )
    asset_no_agent = make_ref(
        SourceObjectKind.ASSET,
        "8010",
        connector_id=conn_disp.connector_id,
        status_raw="unmanaged",
        updated_at=base,
    )
    asset_offline = make_ref(
        SourceObjectKind.ASSET,
        "8011",
        connector_id=conn_disp.connector_id,
        status_raw="offline",
        updated_at=base,
    )

    assets = [
        SourceAsset(
            reference=asset_ref,
            numeric_asset_id="8014",
            hostname=OFFICE_HOST,
            ip="10.40.14.14",
            owner=OFFICE_ACCOUNT,
            agent_status="online",
            asset_group="office",
        )
    ]
    if selected_variant is ScenarioVariant.AGENT_NOT_INSTALLED:
        assets.append(
            SourceAsset(
                reference=asset_no_agent,
                numeric_asset_id="8010",
                hostname="PC-OFFICE-LEGACY",
                agent_status="not_installed",
                normalized={"variant": "agent_not_installed"},
            )
        )
    if selected_variant is ScenarioVariant.DEVICE_OFFLINE:
        assets.append(
            SourceAsset(
                reference=asset_offline,
                numeric_asset_id="8011",
                hostname="PC-OFFICE-DR",
                agent_status="offline",
                normalized={"variant": "device_offline"},
            )
        )

    log_ref = make_ref(
        SourceObjectKind.LOG,
        LOG_ID,
        connector_id=conn_log.connector_id,
        parent=ALERT_ID,
        status_raw="indexed",
        updated_at=base,
    )
    logs = [
        SourceLog(
            reference=log_ref,
            device_source="proxy",
            category="web",
            logged_at=base,
            src_ip="10.40.14.14",
            dst_port=443,
            raw_payload={"domain": NEW_DOMAIN, "bytes_out": 12_000},
        )
    ]

    incident_ref = make_ref(
        SourceObjectKind.INCIDENT,
        INCIDENT_ID,
        connector_id=conn_disp.connector_id,
        status_raw="open",
        updated_at=base,
    )
    alert_ref = make_ref(
        SourceObjectKind.ALERT,
        ALERT_ID,
        connector_id=conn_disp.connector_id,
        status_raw="open",
        updated_at=base,
    )
    alert = SourceAlert(
        reference=alert_ref,
        incident_ref=incident_ref,
        source_ip="10.40.14.14",
        related_log_refs=[log_ref],
        normalized={
            "level": "medium",
            "gpt_tag": "new_domain_access",
            "domain": NEW_DOMAIN,
        },
    )
    # risk_score in 40–69 → expected_verdict must stay ``none`` (not confirmed_threat).
    risk_score = 55
    incident = SourceIncident(
        reference=incident_ref,
        title="Office host accessed newly registered domain",
        level="medium",
        gpt_verdict_label="needs_more_evidence",
        related_alert_refs=[alert_ref],
        impacted_asset_refs=[asset_ref],
        normalized={"risk_score": risk_score, "exfil_observed": False},
    )

    timeline = telemetry_for_variant(
        _build_timeline(base=base, seed=seed, risk_score=risk_score),
        variant=selected_variant,
    )

    return MockXDRScenario(
        scenario_id=SCENARIO_ID,
        name=f"Suspicious domain access without data exfiltration [{selected_variant.value}]",
        variant=selected_variant,
        base_time=base,
        source_tenant_id=tenant,
        incidents=[incident],
        alerts=[alert],
        assets=assets,
        logs=logs,
        connectors=[
            conn_log,
            conn_disp,
            *([conn_gap] if selected_variant is ScenarioVariant.CAPABILITY_GAP else []),
        ],
        telemetry_timeline=timeline,
        failure_profile=failure_profile_for_variant(
            seed=seed,
            variant=selected_variant,
        ),
        expected_outcome={
            "expected_verdict": "none",
            "expected_severity": "medium",
            "risk_score": risk_score,
            "disposition_policy": "not_required",
            "exfil_observed": False,
            "active_variant": selected_variant.value,
            "variants": list(SCENARIO_VARIANTS),
            "provider_error_codes": (
                ["capacity_limit_exceeded"]
                if selected_variant is ScenarioVariant.CAPACITY_LIMIT_EXCEEDED
                else []
            ),
        },
    )


def _build_timeline(*, base: datetime, seed: int, risk_score: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        event(
            channel="dns",
            record_id=f"dns-dom-{seed}-0001",
            offset_s=10,
            base_time=base,
            query=NEW_DOMAIN,
            qtype="A",
            rcode="NOERROR",
            hostname=OFFICE_HOST,
            domain_age_days=3,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="network",
            record_id=f"net-dom-{seed}-0002",
            offset_s=20,
            base_time=base,
            src_ip="10.40.14.14",
            dst_ip="198.51.100.44",
            dst_port=443,
            bytes_out=12_000,
            hostname=OFFICE_HOST,
            domain=NEW_DOMAIN,
            is_key_event=True,
        )
    )
    # Explicitly no large egress / no archive upload.
    rows.append(
        event(
            channel="dlp",
            record_id=f"dlp-dom-{seed}-0003",
            offset_s=30,
            base_time=base,
            file_name="readme.html",
            action="browse",
            bytes=4_096,
            account=OFFICE_ACCOUNT,
            hostname=OFFICE_HOST,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="endpoint",
            record_id=f"ep-dom-{seed}-0004",
            offset_s=40,
            base_time=base,
            hostname=OFFICE_HOST,
            process="chrome.exe",
            account=OFFICE_ACCOUNT,
            action="process_create",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="identity",
            record_id=f"id-dom-{seed}-0005",
            offset_s=0,
            base_time=base,
            account=OFFICE_ACCOUNT,
            event_type="login",
            src_ip="10.40.14.14",
            result="success",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-dom-{seed}-0006",
            offset_s=0,
            base_time=base,
            numeric_asset_id="8014",
            hostname=OFFICE_HOST,
            agent_status="online",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-dom-{seed}-0007",
            offset_s=0,
            base_time=base,
            numeric_asset_id="8010",
            agent_status="not_installed",
            variant="agent_not_installed",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-dom-{seed}-0008",
            offset_s=0,
            base_time=base,
            numeric_asset_id="8011",
            agent_status="offline",
            variant="device_offline",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="threat_intel",
            record_id=f"ti-dom-{seed}-0009",
            offset_s=50,
            base_time=base,
            indicator=NEW_DOMAIN,
            indicator_type="domain",
            confidence=0.45,
            tags=["newly_registered"],
            risk_score=risk_score,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="network",
            record_id=f"net-dom-{seed}-0010",
            offset_s=80,
            base_time=base,
            src_ip="10.40.14.14",
            dst_ip="198.51.100.44",
            dst_port=443,
            bytes_out=8_000,
            domain=NEW_DOMAIN,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="dns",
            record_id=f"dns-dom-{seed}-0011",
            offset_s=90,
            base_time=base,
            query=f"www.{NEW_DOMAIN}",
            qtype="A",
            rcode="NOERROR",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="identity",
            record_id=f"id-dom-{seed}-0012",
            offset_s=120,
            base_time=base,
            account="system",
            event_type="provider_error",
            result="capacity_limit_exceeded",
            provider_error_code="capacity_limit_exceeded",
            variant="capacity_limit_exceeded",
            is_key_event=True,
        )
    )
    return rows
