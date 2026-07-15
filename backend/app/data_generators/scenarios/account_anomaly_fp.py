"""False-positive scenario: ops bulk login in a change window (ISSUE-011)."""

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

SCENARIO_ID = "account_anomaly_fp"

OPS_ACCOUNT = "ops-change-bot"
OPS_HOST = "PC-OPS-JUMP-01"
# Opaque IDs: numeric / UUID / long unprefixed
INCIDENT_ID = "3001"
ALERT_ID = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
ASSET_ID = "7001"
LOG_ID = "xdr_log_ops_bulk_login_change_window_001"


def build_account_anomaly_fp(
    *,
    seed: int = 42,
    variant: ScenarioVariant | str = ScenarioVariant.NORMAL,
) -> MockXDRScenario:
    selected_variant = normalize_variant(variant)
    base = DEFAULT_BASE_TIME
    tenant = DEFAULT_TENANT
    conn_log = log_only_connector(connector_id="conn-log-fp")
    conn_disp = disposition_connector(connector_id="conn-disp-fp")
    conn_disp = conn_disp.model_copy(
        update={"disposition_policy_default": DispositionPolicy.NOT_REQUIRED}
    )
    conn_gap = capability_gap_connector(connector_id="conn-gap-fp")

    asset_ref = make_ref(
        SourceObjectKind.ASSET,
        ASSET_ID,
        connector_id=conn_disp.connector_id,
        status_raw="managed",
        updated_at=base,
    )
    asset_no_agent = make_ref(
        SourceObjectKind.ASSET,
        "7002",
        connector_id=conn_disp.connector_id,
        status_raw="unmanaged",
        updated_at=base,
    )
    asset_offline = make_ref(
        SourceObjectKind.ASSET,
        "7003",
        connector_id=conn_disp.connector_id,
        status_raw="offline",
        updated_at=base,
    )

    assets = [
        SourceAsset(
            reference=asset_ref,
            numeric_asset_id=ASSET_ID,
            hostname=OPS_HOST,
            ip="10.50.1.10",
            owner=OPS_ACCOUNT,
            agent_status="online",
            asset_group="ops",
            normalized={"change_window": True},
        )
    ]
    if selected_variant is ScenarioVariant.AGENT_NOT_INSTALLED:
        assets.append(
            SourceAsset(
                reference=asset_no_agent,
                numeric_asset_id="7002",
                hostname="PC-OPS-LEGACY",
                agent_status="not_installed",
                normalized={"variant": "agent_not_installed"},
            )
        )
    if selected_variant is ScenarioVariant.DEVICE_OFFLINE:
        assets.append(
            SourceAsset(
                reference=asset_offline,
                numeric_asset_id="7003",
                hostname="PC-OPS-DR",
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
            device_source="iam",
            category="auth",
            logged_at=base,
            src_ip="10.50.1.10",
            raw_payload={"account": OPS_ACCOUNT, "batch": True},
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
        source_ip="10.50.1.10",
        related_log_refs=[log_ref],
        normalized={
            "level": "low",
            "gpt_tag": "bulk_login_change_window",
            "fp_rule": "ops_change_window_bulk_login",
        },
    )
    incident = SourceIncident(
        reference=incident_ref,
        title="Bulk login by ops account during change window",
        level="low",
        gpt_verdict_label="likely_false_positive",
        related_alert_refs=[alert_ref],
        impacted_asset_refs=[asset_ref],
        normalized={"risk_score": 18, "fp_rule_match": True},
    )

    timeline = telemetry_for_variant(
        _build_timeline(base=base, seed=seed),
        variant=selected_variant,
    )

    return MockXDRScenario(
        scenario_id=SCENARIO_ID,
        name=(
            f"Account anomaly false positive (change-window bulk login) [{selected_variant.value}]"
        ),
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
            "expected_verdict": "false_positive",
            "expected_severity": "low",
            "risk_score": 18,
            "disposition_policy": "not_required",
            "fp_rule": "ops_change_window_bulk_login",
            "active_variant": selected_variant.value,
            "variants": list(SCENARIO_VARIANTS),
            "provider_error_codes": (
                ["capacity_limit_exceeded"]
                if selected_variant is ScenarioVariant.CAPACITY_LIMIT_EXCEEDED
                else []
            ),
        },
    )


def _build_timeline(*, base: datetime, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Change window: 10 consecutive successful logins by ops account.
    for i in range(10):
        rows.append(
            event(
                channel="identity",
                record_id=f"id-fp-{seed}-{i:04d}",
                offset_s=i * 45,
                base_time=base,
                account=OPS_ACCOUNT,
                event_type="login",
                src_ip=f"10.50.1.{10 + (i % 5)}",
                result="success",
                change_window=True,
                is_key_event=True,
            )
        )
    rows.append(
        event(
            channel="endpoint",
            record_id=f"ep-fp-{seed}-0010",
            offset_s=500,
            base_time=base,
            hostname=OPS_HOST,
            process="ssh.exe",
            account=OPS_ACCOUNT,
            action="process_create",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="network",
            record_id=f"net-fp-{seed}-0011",
            offset_s=520,
            base_time=base,
            src_ip="10.50.1.10",
            dst_ip="10.50.2.20",
            dst_port=22,
            bytes_out=4096,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="dns",
            record_id=f"dns-fp-{seed}-0012",
            offset_s=530,
            base_time=base,
            query="repo.internal.example",
            qtype="A",
            rcode="NOERROR",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="dlp",
            record_id=f"dlp-fp-{seed}-0013",
            offset_s=540,
            base_time=base,
            file_name="runbook.txt",
            action="read",
            bytes=2048,
            account=OPS_ACCOUNT,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-fp-{seed}-0014",
            offset_s=0,
            base_time=base,
            numeric_asset_id=ASSET_ID,
            hostname=OPS_HOST,
            agent_status="online",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-fp-{seed}-0015",
            offset_s=0,
            base_time=base,
            numeric_asset_id="7002",
            agent_status="not_installed",
            variant="agent_not_installed",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-fp-{seed}-0016",
            offset_s=0,
            base_time=base,
            numeric_asset_id="7003",
            agent_status="offline",
            variant="device_offline",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="threat_intel",
            record_id=f"ti-fp-{seed}-0017",
            offset_s=600,
            base_time=base,
            indicator="10.50.1.10",
            indicator_type="ip",
            confidence=0.1,
            tags=["internal"],
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="identity",
            record_id=f"id-fp-{seed}-0018",
            offset_s=700,
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
