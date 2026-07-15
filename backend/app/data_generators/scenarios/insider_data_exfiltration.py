"""Primary demo scenario: insider data exfiltration (ISSUE-011)."""

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
from app.mock_xdr.models import MockXDRScenario, ScenarioTick, ScenarioVariant, TickOperation
from app.models.enums import SourceDisposition, SourceObjectKind
from app.models.source import SourceAlert, SourceAsset, SourceIncident, SourceLog

SCENARIO_ID = "insider_data_exfiltration"

# Fixed demo entities (only allowed inside scenario packs / generated artifacts).
ACCOUNT = "zhangsan"
HOST = "PC-FIN-023"
# RFC5737 documentation range — never ship a real-looking IOC in fixtures/logs.
EXFIL_IP = "203.0.113.88"
EXFIL_DOMAIN = "unknown-upload-example.com"
PROC_PS = "powershell.exe"
PROC_7Z = "7z.exe"
FILE_ZIP = "finance_report.zip"

# Opaque external IDs — fixtures cover pure digit / UUID / unprefixed long string.
INCIDENT_ID = "88442201"  # pure numeric
ALERT_DLP_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  # UUID
ALERT_EDR_ID = "xdr_alert_fin_exfil_endpoint_activity_20240615"  # long unprefixed
ALERT_NET_ID = "99110022"
ASSET_PRIMARY_ID = "1023"
ASSET_NO_AGENT_ID = "2048"
ASSET_OFFLINE_ID = "4096"
LOG_DLP_ID = "log_dlp_finance_upload_001"
LOG_EDR_ID = "log_edr_proc_chain_001"
LOG_NET_ID = "550011"


def build_insider_data_exfiltration(
    *,
    seed: int = 42,
    variant: ScenarioVariant | str = ScenarioVariant.NORMAL,
) -> MockXDRScenario:
    """Build the insider exfiltration scenario (deterministic under ``seed``)."""
    selected_variant = normalize_variant(variant)
    base = DEFAULT_BASE_TIME
    tenant = DEFAULT_TENANT
    conn_log = log_only_connector()
    conn_disp = disposition_connector()
    conn_gap = capability_gap_connector()

    asset_primary_ref = make_ref(
        SourceObjectKind.ASSET,
        ASSET_PRIMARY_ID,
        connector_id=conn_disp.connector_id,
        status_raw="managed",
        updated_at=base,
        object_type="endpoint",
    )
    asset_no_agent_ref = make_ref(
        SourceObjectKind.ASSET,
        ASSET_NO_AGENT_ID,
        connector_id=conn_disp.connector_id,
        status_raw="unmanaged",
        updated_at=base,
    )
    asset_offline_ref = make_ref(
        SourceObjectKind.ASSET,
        ASSET_OFFLINE_ID,
        connector_id=conn_disp.connector_id,
        status_raw="offline",
        updated_at=base,
        disposition=SourceDisposition.UNKNOWN,
    )

    assets = [
        SourceAsset(
            reference=asset_primary_ref,
            numeric_asset_id=ASSET_PRIMARY_ID,
            hostname=HOST,
            ip="10.20.30.23",
            asset_name=HOST,
            asset_group="finance",
            owner=ACCOUNT,
            business_system="finance_erp",
            importance="critical",
            agent_status="online",
            first_seen_at=base,
            last_seen_at=base,
            normalized={"variant": "primary"},
        )
    ]
    if selected_variant is ScenarioVariant.AGENT_NOT_INSTALLED:
        assets.append(
            SourceAsset(
                reference=asset_no_agent_ref,
                numeric_asset_id=ASSET_NO_AGENT_ID,
                hostname="PC-FIN-099",
                ip="10.20.30.99",
                agent_status="not_installed",
                normalized={"variant": "agent_not_installed"},
            )
        )
    if selected_variant is ScenarioVariant.DEVICE_OFFLINE:
        assets.append(
            SourceAsset(
                reference=asset_offline_ref,
                numeric_asset_id=ASSET_OFFLINE_ID,
                hostname="PC-FIN-077",
                ip="10.20.30.77",
                agent_status="offline",
                normalized={"variant": "device_offline"},
            )
        )

    log_dlp_ref = make_ref(
        SourceObjectKind.LOG,
        LOG_DLP_ID,
        connector_id=conn_log.connector_id,
        parent=ALERT_DLP_ID,
        status_raw="indexed",
        updated_at=base,
    )
    log_edr_ref = make_ref(
        SourceObjectKind.LOG,
        LOG_EDR_ID,
        connector_id=conn_disp.connector_id,
        parent=ALERT_EDR_ID,
        status_raw="indexed",
        updated_at=base,
    )
    log_net_ref = make_ref(
        SourceObjectKind.LOG,
        LOG_NET_ID,
        connector_id=conn_log.connector_id,
        parent=ALERT_NET_ID,
        status_raw="indexed",
        updated_at=base,
    )
    logs = [
        SourceLog(
            reference=log_dlp_ref,
            device_source="dlp",
            category="data_exfiltration",
            logged_at=base,
            src_ip="10.20.30.23",
            dst_ip=EXFIL_IP,
            dst_port=443,
            raw_payload={"file": FILE_ZIP, "account": ACCOUNT},
        ),
        SourceLog(
            reference=log_edr_ref,
            device_source="edr",
            category="process",
            logged_at=base,
            src_ip="10.20.30.23",
            raw_payload={"process": PROC_PS, "account": ACCOUNT, "host": HOST},
        ),
        SourceLog(
            reference=log_net_ref,
            device_source="nfw",
            category="egress",
            logged_at=base,
            src_ip="10.20.30.23",
            dst_ip=EXFIL_IP,
            dst_port=443,
            raw_payload={"domain": EXFIL_DOMAIN},
        ),
    ]

    alert_dlp_ref = make_ref(
        SourceObjectKind.ALERT,
        ALERT_DLP_ID,
        connector_id=conn_disp.connector_id,
        status_raw="open",
        updated_at=base,
        object_type="dlp_alert",
    )
    alert_edr_ref = make_ref(
        SourceObjectKind.ALERT,
        ALERT_EDR_ID,
        connector_id=conn_disp.connector_id,
        status_raw="open",
        updated_at=base,
        object_type="endpoint_alert",
    )
    alert_net_ref = make_ref(
        SourceObjectKind.ALERT,
        ALERT_NET_ID,
        connector_id=conn_disp.connector_id,
        status_raw="open",
        updated_at=base,
        object_type="network_alert",
    )

    incident_ref = make_ref(
        SourceObjectKind.INCIDENT,
        INCIDENT_ID,
        connector_id=conn_disp.connector_id,
        status_raw="investigating",
        updated_at=base,
        object_type="security_incident",
    )

    alerts = [
        SourceAlert(
            reference=alert_dlp_ref,
            incident_ref=incident_ref,
            source_ip="10.20.30.23",
            related_log_refs=[log_dlp_ref],
            normalized={
                "level": "critical",
                "gpt_tag": "suspected_data_exfil",
                "account": ACCOUNT,
                "file": FILE_ZIP,
            },
            raw_payload={"rule": "sensitive_file_upload"},
        ),
        SourceAlert(
            reference=alert_edr_ref,
            incident_ref=incident_ref,
            source_ip="10.20.30.23",
            related_log_refs=[log_edr_ref],
            normalized={
                "level": "high",
                "gpt_tag": "suspicious_process_chain",
                "processes": [PROC_PS, PROC_7Z],
            },
        ),
        SourceAlert(
            reference=alert_net_ref,
            incident_ref=incident_ref,
            source_ip="10.20.30.23",
            related_log_refs=[log_net_ref],
            normalized={
                "level": "high",
                "gpt_tag": "c2_like_egress",
                "dst_ip": EXFIL_IP,
                "domain": EXFIL_DOMAIN,
            },
        ),
    ]

    impacted_asset_refs = [asset_primary_ref]
    if selected_variant is ScenarioVariant.AGENT_NOT_INSTALLED:
        impacted_asset_refs.append(asset_no_agent_ref)
    if selected_variant is ScenarioVariant.DEVICE_OFFLINE:
        impacted_asset_refs.append(asset_offline_ref)
    incident = SourceIncident(
        reference=incident_ref,
        title="Finance endpoint suspected data exfiltration",
        level="critical",
        gpt_verdict_label="suspected_insider_threat",
        related_alert_refs=[alert_dlp_ref, alert_edr_ref, alert_net_ref],
        impacted_asset_refs=impacted_asset_refs,
        normalized={
            "account": ACCOUNT,
            "hostname": HOST,
            "risk_score": 92,
        },
    )

    timeline = telemetry_for_variant(
        _build_timeline(base=base, seed=seed),
        variant=selected_variant,
    )
    ticks = [
        ScenarioTick(
            offset_seconds=3600,
            operation=TickOperation.UPSERT,
            object_type="alert",
            object_id=ALERT_DLP_ID,
            patch={"reference": {"source_status_raw": "in_progress"}},
        ),
        # Disposition connector starts ONLINE; degrade later to exercise the edge.
        ScenarioTick(
            offset_seconds=7200,
            operation=TickOperation.CONNECTOR_CHANGE,
            object_type="connector",
            object_id=conn_disp.connector_id,
            patch={"status": "degraded"},
        ),
    ]

    return MockXDRScenario(
        scenario_id=SCENARIO_ID,
        name=f"Insider data exfiltration (finance endpoint) [{selected_variant.value}]",
        variant=selected_variant,
        base_time=base,
        source_tenant_id=tenant,
        incidents=[incident],
        alerts=alerts,
        assets=assets,
        logs=logs,
        connectors=[
            conn_log,
            conn_disp,
            *([conn_gap] if selected_variant is ScenarioVariant.CAPABILITY_GAP else []),
        ],
        telemetry_timeline=timeline,
        ticks=ticks,
        failure_profile=failure_profile_for_variant(
            seed=seed,
            variant=selected_variant,
        ),
        expected_outcome={
            "expected_verdict": "confirmed_threat",
            "expected_severity": "critical",
            "risk_score": 92,
            "disposition_policy": "required",
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
    """≥20 key events spanning identity/endpoint/dlp/network/dns/asset/ti."""
    rows: list[dict[str, Any]] = []
    # Background identity noise for other users — zhangsan has NO successful login.
    rows.append(
        event(
            channel="identity",
            record_id=f"id-noise-{seed}-0001",
            offset_s=0,
            base_time=base,
            account="ops-bot",
            event_type="login",
            src_ip="10.0.0.8",
            result="success",
            is_key_event=True,
        )
    )
    # Conflict seed 1: explicit absence of login for zhangsan (identity side).
    rows.append(
        event(
            channel="identity",
            record_id=f"id-conflict-{seed}-0002",
            offset_s=60,
            base_time=base,
            account=ACCOUNT,
            event_type="login_lookup",
            src_ip=None,
            result="no_record",
            note="no interactive login for account in window",
            is_key_event=True,
            is_conflict_seed=True,
        )
    )
    # Conflict seed 2: endpoint shows process activity under zhangsan.
    rows.append(
        event(
            channel="endpoint",
            record_id=f"ep-conflict-{seed}-0003",
            offset_s=120,
            base_time=base,
            hostname=HOST,
            process=PROC_PS,
            account=ACCOUNT,
            action="process_create",
            cmdline=f"{PROC_PS} -enc <compressed>",
            is_key_event=True,
            is_conflict_seed=True,
        )
    )
    rows.append(
        event(
            channel="endpoint",
            record_id=f"ep-key-{seed}-0004",
            offset_s=180,
            base_time=base,
            hostname=HOST,
            process=PROC_7Z,
            account=ACCOUNT,
            action="process_create",
            cmdline=f"{PROC_7Z} a {FILE_ZIP}",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="endpoint",
            record_id=f"ep-key-{seed}-0005",
            offset_s=240,
            base_time=base,
            hostname=HOST,
            process=PROC_PS,
            account=ACCOUNT,
            action="file_access",
            file_name=FILE_ZIP,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="dlp",
            record_id=f"dlp-key-{seed}-0006",
            offset_s=300,
            base_time=base,
            file_name=FILE_ZIP,
            action="archive",
            bytes=52_428_800,
            account=ACCOUNT,
            hostname=HOST,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="dlp",
            record_id=f"dlp-key-{seed}-0007",
            offset_s=360,
            base_time=base,
            file_name=FILE_ZIP,
            action="upload",
            bytes=52_428_800,
            account=ACCOUNT,
            hostname=HOST,
            destination=EXFIL_DOMAIN,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="dns",
            record_id=f"dns-key-{seed}-0008",
            offset_s=370,
            base_time=base,
            query=EXFIL_DOMAIN,
            qtype="A",
            rcode="NOERROR",
            answer=EXFIL_IP,
            hostname=HOST,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="network",
            record_id=f"net-key-{seed}-0009",
            offset_s=400,
            base_time=base,
            src_ip="10.20.30.23",
            dst_ip=EXFIL_IP,
            dst_port=443,
            bytes_out=52_000_000,
            hostname=HOST,
            domain=EXFIL_DOMAIN,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="network",
            record_id=f"net-key-{seed}-0010",
            offset_s=460,
            base_time=base,
            src_ip="10.20.30.23",
            dst_ip=EXFIL_IP,
            dst_port=443,
            bytes_out=400_000,
            hostname=HOST,
            domain=EXFIL_DOMAIN,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="threat_intel",
            record_id=f"ti-key-{seed}-0011",
            offset_s=500,
            base_time=base,
            indicator=EXFIL_IP,
            indicator_type="ip",
            confidence=0.91,
            tags=["exfil", "unknown_infra"],
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="threat_intel",
            record_id=f"ti-key-{seed}-0012",
            offset_s=510,
            base_time=base,
            indicator=EXFIL_DOMAIN,
            indicator_type="domain",
            confidence=0.88,
            tags=["newly_observed"],
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-key-{seed}-0013",
            offset_s=0,
            base_time=base,
            numeric_asset_id=ASSET_PRIMARY_ID,
            hostname=HOST,
            ip="10.20.30.23",
            agent_status="online",
            owner=ACCOUNT,
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-key-{seed}-0014",
            offset_s=0,
            base_time=base,
            numeric_asset_id=ASSET_NO_AGENT_ID,
            hostname="PC-FIN-099",
            agent_status="not_installed",
            variant="agent_not_installed",
            is_key_event=True,
        )
    )
    rows.append(
        event(
            channel="asset",
            record_id=f"asset-key-{seed}-0015",
            offset_s=0,
            base_time=base,
            numeric_asset_id=ASSET_OFFLINE_ID,
            hostname="PC-FIN-077",
            agent_status="offline",
            variant="device_offline",
            is_key_event=True,
        )
    )
    # Additional key events to exceed 20.
    for i, offset in enumerate(range(600, 600 + 8 * 30, 30), start=16):
        rows.append(
            event(
                channel="endpoint",
                record_id=f"ep-key-{seed}-{i:04d}",
                offset_s=offset,
                base_time=base,
                hostname=HOST,
                process=PROC_PS if i % 2 == 0 else PROC_7Z,
                account=ACCOUNT,
                action="network_connect",
                dst_ip=EXFIL_IP,
                is_key_event=True,
            )
        )
    # Non-key noise + provider business-error observation (Mock-custom only).
    rows.append(
        event(
            channel="network",
            record_id=f"net-noise-{seed}-0090",
            offset_s=900,
            base_time=base,
            src_ip="10.20.30.23",
            dst_ip="192.0.2.53",
            dst_port=53,
            bytes_out=120,
            is_key_event=False,
        )
    )
    rows.append(
        event(
            channel="identity",
            record_id=f"id-provider-{seed}-0091",
            offset_s=920,
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
