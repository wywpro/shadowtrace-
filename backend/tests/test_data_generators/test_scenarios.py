"""Demo scenario pack tests (ISSUE-011)."""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.data_generators.base import TELEMETRY_FILENAMES
from app.data_generators.scenarios import (
    SCENARIO_REGISTRY,
    SCENARIO_VARIANTS,
    build_scenario,
    telemetry_for_scenario,
    write_scenario_artifacts,
)
from app.data_generators.scenarios._common import count_conflict_seeds, count_key_events
from app.data_generators.scenarios.insider_data_exfiltration import (
    ACCOUNT,
    EXFIL_DOMAIN,
    EXFIL_IP,
    FILE_ZIP,
    HOST,
    PROC_7Z,
    PROC_PS,
)
from app.mock_xdr.models import MockXDRScenario, ScenarioVariant
from app.mock_xdr.state import MockXDRState
from app.models.enums import FinalVerdict, Severity

SCENARIO_IDS = (
    "insider_data_exfiltration",
    "account_anomaly_fp",
    "suspicious_domain_access",
)

# Opaque external id fixtures must cover digit-only / UUID / long unprefixed forms.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_registry_contains_three_scenarios(scenario_id: str) -> None:
    assert scenario_id in SCENARIO_REGISTRY
    assert set(SCENARIO_REGISTRY) == set(SCENARIO_IDS)
    assert SCENARIO_REGISTRY[scenario_id].variant is ScenarioVariant.NORMAL


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_passes_framework_and_schema_validation(scenario_id: str) -> None:
    built = build_scenario(scenario_id, seed=42)
    # Round-trip through Pydantic (= schema) + referential consistency validator.
    restored = MockXDRScenario.model_validate(built.model_dump(mode="json"))
    assert restored.scenario_id == scenario_id
    assert restored.incidents
    assert restored.alerts
    assert restored.assets
    assert restored.logs
    assert len(restored.connectors) >= 2
    roles = {c.metadata.get("role") for c in restored.connectors}
    assert "log_only" in roles
    assert "disposition" in roles


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_generation_is_deterministic(scenario_id: str) -> None:
    a = build_scenario(scenario_id, seed=42).model_dump(mode="json")
    b = build_scenario(scenario_id, seed=42).model_dump(mode="json")
    assert a == b
    # Telemetry channel split is also stable.
    assert telemetry_for_scenario(build_scenario(scenario_id, seed=7)) == telemetry_for_scenario(
        build_scenario(scenario_id, seed=7)
    )


def test_insider_key_events_and_entities() -> None:
    scenario = build_scenario("insider_data_exfiltration", seed=42)
    assert count_key_events(scenario.telemetry_timeline) >= 20
    assert count_conflict_seeds(scenario.telemetry_timeline) == 2

    blob = json.dumps(scenario.model_dump(mode="json"))
    for token in (ACCOUNT, HOST, EXFIL_IP, EXFIL_DOMAIN, PROC_PS, PROC_7Z, FILE_ZIP):
        assert token in blob

    # Cross-channel entity consistency for the primary host/account.
    channels = telemetry_for_scenario(scenario)
    assert any(r.get("account") == ACCOUNT for r in channels["endpoint"])
    assert any(r.get("hostname") == HOST for r in channels["endpoint"])
    assert any(r.get("file_name") == FILE_ZIP for r in channels["dlp"])
    assert any(r.get("dst_ip") == EXFIL_IP for r in channels["network"])
    assert any(r.get("query") == EXFIL_DOMAIN for r in channels["dns"])

    # Conflict: identity has no successful login for zhangsan; endpoint does.
    id_rows = [r for r in channels["identity"] if r.get("account") == ACCOUNT]
    assert any(
        r.get("is_conflict_seed") is True and r.get("result") == "no_record" for r in id_rows
    )
    assert not any(r.get("event_type") == "login" and r.get("result") == "success" for r in id_rows)
    assert any(
        r.get("is_conflict_seed") is True and r.get("account") == ACCOUNT
        for r in channels["endpoint"]
    )

    outcome = scenario.expected_outcome
    assert outcome["expected_verdict"] == FinalVerdict.CONFIRMED_THREAT.value
    assert outcome["expected_severity"] == Severity.CRITICAL.value


def test_account_anomaly_fp_outcome() -> None:
    scenario = build_scenario("account_anomaly_fp", seed=42)
    outcome = scenario.expected_outcome
    assert outcome["expected_verdict"] == FinalVerdict.FALSE_POSITIVE.value
    assert outcome["expected_severity"] == Severity.LOW.value
    assert outcome.get("fp_rule") == "ops_change_window_bulk_login"
    assert any(r.get("change_window") is True for r in scenario.telemetry_timeline)


def test_suspicious_domain_access_outcome_and_risk_band() -> None:
    scenario = build_scenario("suspicious_domain_access", seed=42)
    outcome = scenario.expected_outcome
    assert outcome["expected_verdict"] == FinalVerdict.NONE.value
    assert outcome["expected_severity"] == Severity.MEDIUM.value
    risk = int(outcome["risk_score"])
    assert 40 <= risk <= 69
    assert outcome["expected_verdict"] != FinalVerdict.CONFIRMED_THREAT.value
    assert outcome.get("exfil_observed") is False


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
@pytest.mark.parametrize("variant", SCENARIO_VARIANTS)
def test_normal_and_degraded_variants_execute_independently(
    scenario_id: str,
    variant: str,
) -> None:
    selected = ScenarioVariant(variant)
    scenario = build_scenario(scenario_id, seed=42, variant=selected)
    state = MockXDRState()
    state.load_scenario(scenario)

    assert scenario.variant is selected
    assert scenario.expected_outcome["active_variant"] == selected.value
    assert set(scenario.expected_outcome["variants"]) == set(SCENARIO_VARIANTS)
    assert bool([asset for asset in scenario.assets if asset.agent_status == "not_installed"]) is (
        selected is ScenarioVariant.AGENT_NOT_INSTALLED
    )
    assert bool([asset for asset in scenario.assets if asset.agent_status == "offline"]) is (
        selected is ScenarioVariant.DEVICE_OFFLINE
    )
    assert bool(
        [
            connector
            for connector in scenario.connectors
            if connector.metadata.get("role") == "capability_gap"
        ]
    ) is (selected is ScenarioVariant.CAPABILITY_GAP)
    assert scenario.failure_profile.force_partial_targets is (
        selected is ScenarioVariant.PARTIAL_SUCCESS
    )
    assert scenario.failure_profile.rate_limit_every_n == (
        1 if selected is ScenarioVariant.RATE_LIMIT else None
    )
    assert scenario.failure_profile.timeout_every_n == (
        1 if selected is ScenarioVariant.TIMEOUT else None
    )
    assert scenario.failure_profile.malformed_payload_every_n == (
        1 if selected is ScenarioVariant.MALFORMED_PAYLOAD else None
    )
    provider_rows = [
        row
        for row in scenario.telemetry_timeline
        if row.get("provider_error_code") == "capacity_limit_exceeded"
    ]
    assert bool(provider_rows) is (selected is ScenarioVariant.CAPACITY_LIMIT_EXCEEDED)
    assert scenario.expected_outcome["provider_error_codes"] == (
        ["capacity_limit_exceeded"] if selected is ScenarioVariant.CAPACITY_LIMIT_EXCEEDED else []
    )
    assert state.scenario is scenario
    blob = json.dumps(scenario.model_dump(mode="json"), ensure_ascii=False)
    assert "黑名单总数超出" not in blob


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["alerts"][0]["reference"].update(
            {"connector_id": "missing-connector"}
        ),
        lambda payload: payload["alerts"][0]["reference"].update({"source_kind": "asset"}),
        lambda payload: payload["incidents"].append(payload["incidents"][0]),
    ],
)
def test_scenario_semantic_reference_validation_rejects_invalid_structures(
    mutate,
) -> None:
    payload = build_scenario("insider_data_exfiltration", seed=42).model_dump(mode="json")
    mutate(payload)
    with pytest.raises(ValueError):
        MockXDRScenario.model_validate(payload)


def test_opaque_external_id_shapes_across_fixtures() -> None:
    """Fixtures cover pure digit, UUID, and unprefixed long-string object ids."""
    seen_digit = seen_uuid = seen_long = False
    for scenario_id in SCENARIO_IDS:
        scenario = build_scenario(scenario_id, seed=42)
        ids = [
            *(i.reference.source_object_id for i in scenario.incidents),
            *(a.reference.source_object_id for a in scenario.alerts),
            *(a.reference.source_object_id for a in scenario.assets),
            *(log.reference.source_object_id for log in scenario.logs),
        ]
        for oid in ids:
            assert not oid.startswith("incident-")
            assert not oid.startswith("alert-")
            if oid.isdigit():
                seen_digit = True
            elif _UUID_RE.match(oid):
                seen_uuid = True
            elif len(oid) >= 20 and "-" not in oid[:8]:
                seen_long = True
    assert seen_digit and seen_uuid and seen_long


_DOC_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]


def _iter_ipv4(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            found.extend(_iter_ipv4(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_iter_ipv4(item))
    elif isinstance(obj, str):
        try:
            ipaddress.IPv4Address(obj)
        except ipaddress.AddressValueError:
            return found
        found.append(obj)
    return found


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_fixtures_use_documentation_or_private_ips(scenario_id: str) -> None:
    """No real-looking public IOC ships in fixtures — only RFC5737/private IPs."""
    scenario = build_scenario(scenario_id, seed=42)
    for ip_str in _iter_ipv4(scenario.model_dump(mode="json")):
        ip = ipaddress.IPv4Address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_multicast:
            continue
        assert any(ip in net for net in _DOC_NETS), (
            f"non-documentation public IP {ip_str} in scenario {scenario_id}"
        )


def test_write_seven_telemetry_files(tmp_path: Path) -> None:
    scenario = build_scenario("insider_data_exfiltration", seed=42)
    written = write_scenario_artifacts(scenario, tmp_path, write_scenario_json=False)
    names = {p.name for p in written}
    assert names == set(TELEMETRY_FILENAMES.values())
    assert len(written) == 7
    for path in written:
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(rows, list)
        assert rows
