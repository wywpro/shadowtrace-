"""Edge tables + terminal lineage rules (ISSUE-010 §验收4)."""

from __future__ import annotations

import pytest

from app.mock_xdr.state import MockValidationError, MockXDRState
from app.models.enums import (
    ConnectorStatus,
    SourceDisposition,
    WritebackStatus,
)
from tests.test_mock_xdr.conftest import disposition_command


def test_source_disposition_legal_and_illegal(state: MockXDRState) -> None:
    state.transition_source_disposition(
        "incident", "INC-1", SourceDisposition.PROCESSING, allow_unknown_recovery=True
    )
    rb = state.readback_source_disposition("incident", "INC-1")
    assert rb["source_disposition"] == "processing"

    # Terminal cannot regress
    state.transition_source_disposition(
        "incident", "INC-1", SourceDisposition.CONTAINED, allow_unknown_recovery=True
    )
    with pytest.raises(MockValidationError, match="terminal"):
        state.transition_source_disposition(
            "incident", "INC-1", SourceDisposition.PROCESSING, allow_unknown_recovery=True
        )


def test_unknown_recovery_requires_flag(state: MockXDRState) -> None:
    state.transition_source_disposition(
        "incident", "INC-1", SourceDisposition.UNKNOWN, allow_unknown_recovery=True
    )
    with pytest.raises(MockValidationError, match="unknown"):
        state.transition_source_disposition(
            "incident", "INC-1", SourceDisposition.PROCESSING, allow_unknown_recovery=False
        )
    # Authoritative recovery with new token
    before = state.objects[("incident", "INC-1")].concurrency_token
    state.transition_source_disposition(
        "incident", "INC-1", SourceDisposition.PROCESSING, allow_unknown_recovery=True
    )
    after = state.objects[("incident", "INC-1")].concurrency_token
    assert after != before


def test_connector_online_requires_health(state: MockXDRState) -> None:
    state.transition_connector("conn-1", ConnectorStatus.DEGRADED, health_ok=False)
    with pytest.raises(MockValidationError, match="health"):
        state.transition_connector("conn-1", ConnectorStatus.ONLINE, health_ok=False)
    state.transition_connector("conn-1", ConnectorStatus.ONLINE, health_ok=True)
    assert state.connectors["conn-1"].status is ConnectorStatus.ONLINE


def test_parallel_active_heads_rejected(state: MockXDRState) -> None:
    token = state.objects[("incident", "INC-1")].concurrency_token
    first = disposition_command(
        disposition_id="disp-a",
        idempotency_key="idem-a",
        token=token,
        target=SourceDisposition.CONTAINED,
    )
    state.submit_disposition(first)
    # Second active head without supersedes
    token2 = state.objects[("incident", "INC-1")].concurrency_token
    second = disposition_command(
        disposition_id="disp-b",
        idempotency_key="idem-b",
        token=token2,
        target=SourceDisposition.COMPLETED,
    )
    with pytest.raises(MockValidationError, match="parallel active"):
        state.submit_disposition(second)


def test_legal_supersede_keeps_history_one_confirmed_head(state: MockXDRState) -> None:
    token = state.objects[("incident", "INC-1")].concurrency_token
    first = disposition_command(
        disposition_id="disp-old",
        idempotency_key="idem-old",
        token=token,
        target=SourceDisposition.CONTAINED,
    )
    state.submit_disposition(first)
    # Supersede with approved different terminal payload
    token2 = state.objects[("incident", "INC-1")].concurrency_token
    # Reset disposition to allow new transition for demo: use processing via control
    # After first submit, object may already be CONTAINED — supersede is about lineage,
    # not requiring another disposition transition success.
    second = disposition_command(
        disposition_id="disp-new",
        idempotency_key="idem-new",
        token=token2,
        target=SourceDisposition.IGNORED,
        supersedes="disp-old",
    )
    # Object is already CONTAINED; transition to IGNORED is illegal — submit should
    # still accept lineage supersede but may mark FAILED on transition. For lineage
    # test we only care about active head bookkeeping.
    # Soften: put object back to PROCESSING via direct body patch for this unit test.
    body = dict(state.objects[("incident", "INC-1")].body)
    body["reference"] = dict(body["reference"])
    body["reference"]["source_disposition"] = SourceDisposition.PROCESSING.value
    state.upsert_object("incident", "INC-1", body)
    token3 = state.objects[("incident", "INC-1")].concurrency_token
    second = disposition_command(
        disposition_id="disp-new",
        idempotency_key="idem-new",
        token=token3,
        target=SourceDisposition.IGNORED,
        supersedes="disp-old",
    )
    state.submit_disposition(second)
    assert state.disposition_by_id["disp-old"].superseded is True
    assert state.disposition_by_id["disp-old"].active is False
    assert state.active_terminal_heads[("INC-1", 1)] == "disp-new"
    state.transition_source_disposition(
        "incident",
        "INC-1",
        SourceDisposition.IGNORED,
        allow_unknown_recovery=True,
    )
    state.confirm_via_readback("disp-new")
    assert state.disposition_by_id["disp-new"].latest_status is WritebackStatus.CONFIRMED
    # History retained
    assert len(state.disposition_by_id["disp-old"].receipts) >= 1


def test_required_missing_terminal_lineage_detected(state: MockXDRState) -> None:
    missing = state.required_events_missing_terminal_lineage()
    assert any(m.startswith("INC-1") for m in missing)
    token = state.objects[("incident", "INC-1")].concurrency_token
    cmd = disposition_command(token=token)
    state.submit_disposition(cmd)
    state.transition_source_disposition(
        "incident",
        "INC-1",
        SourceDisposition.CONTAINED,
        allow_unknown_recovery=True,
    )
    state.confirm_via_readback(cmd.disposition_id)
    missing_after = state.required_events_missing_terminal_lineage()
    assert not any(m == "INC-1" or m.startswith("INC-1@") for m in missing_after)
