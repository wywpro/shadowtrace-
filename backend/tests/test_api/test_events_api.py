"""API contract tests for event lifecycle endpoints (ISSUE-038).

Tests the 11 core event endpoints with real database-backed services.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.v1.deps import reset_deps
from app.db import models as orm
from app.main import app
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    WritebackReadiness,
    WritebackStatus,
)
from app.services.event_service import EventService

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("clean_state")]

_DEV_TOKENS = json.dumps(
    {
        "analyst-token": {"subject": "analyst-1", "roles": ["analyst"]},
        "approver-token": {"subject": "approver-1", "roles": ["approver"]},
        "operator-token": {"subject": "op-1", "roles": ["disposition_operator"]},
        "admin-token": {"subject": "admin-1", "roles": ["admin"]},
    }
)


@pytest.fixture(autouse=True)
def _dev_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_AUTH_TOKENS", _DEV_TOKENS)
    monkeypatch.setenv("ALLOW_LIVE_SIDE_EFFECTS", "false")
    monkeypatch.setenv("ALLOW_XDR_WRITEBACK", "false")
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("TOOL_MODE", "mock")
    monkeypatch.setenv("SOURCE_MODE", "mock_xdr")
    monkeypatch.setenv("DISPOSITION_MODE", "mock_xdr")
    monkeypatch.setenv("SIMULATION_ENABLED", "true")


def _hdr(role: str = "analyst") -> dict[str, str]:
    return {"Authorization": f"Bearer {role}-token"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_services() -> None:
    """Reset lazy singletons between tests so each test gets clean state."""
    reset_deps()
    app.dependency_overrides.clear()


@pytest.fixture
def client(
    session_factory: async_sessionmaker[AsyncSession],
    event_service: EventService,
) -> TestClient:
    """Inject test services into the app via dependency overrides."""

    from app.api.v1.deps import get_event_service

    async def _override_event_service() -> EventService:
        return event_service

    app.dependency_overrides[get_event_service] = _override_event_service
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Helper: create a test event
# --------------------------------------------------------------------------- #


async def _create_test_event(
    event_service: EventService,
    *,
    title: str = "Test event",
    event_type: EventType = EventType.INSIDER_THREAT,
    severity: Severity = Severity.HIGH,
) -> str:
    event = await event_service.create_event(
        {"title": title, "description": "Test event created by API test"},
        source_type="manual",
        title=title,
        event_type=event_type,
        severity=severity,
    )
    return event.event_id


async def _seed_reporting_required_event(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    title: str = "Reporting required event",
    writeback_readiness: WritebackReadiness = WritebackReadiness.READY,
    outbox_status: WritebackStatus | None = None,
    include_action: bool = True,
) -> str:
    """Insert a REPORTING event with optional writeback action/outbox rows."""
    import hashlib
    from datetime import UTC, datetime
    from uuid import uuid4

    from app.models.action import TERMINAL_DISPOSITION_TOOL
    from app.models.enums import ActionExecutionPhase, DispositionIntentKind, SourceDisposition

    sfx = uuid4().hex[:8]
    event_id = f"evt-{sfx}"
    now = datetime.now(UTC)

    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SecurityEvent(
                    event_id=event_id,
                    event_type="data_exfiltration",
                    title=title,
                    description="Writeback gate test fixture",
                    status=EventStatus.REPORTING.value,
                    severity=Severity.HIGH.value,
                    final_verdict=FinalVerdict.CONFIRMED_THREAT.value,
                    risk_score=85,
                    entities={},
                    creation_source_ref={
                        "source_kind": "incident",
                        "source_product": "mock_xdr",
                        "source_tenant_id": "t1",
                        "connector_id": f"conn-{sfx}",
                        "source_object_id": f"INC-{sfx}",
                        "raw_payload_hash": hashlib.sha256(b"wb").hexdigest(),
                        "ingested_at": now.isoformat(),
                    },
                    source_reference_snapshots=[],
                    disposition_policy=DispositionPolicy.REQUIRED.value,
                    source_type="mock_xdr",
                    occurred_at=now,
                    row_version=1,
                )
            )
            session.add(
                orm.EventAuditLog(
                    event_id=event_id,
                    from_status="new",
                    to_status=EventStatus.REPORTING.value,
                    operator="test",
                    reason="test_setup:reporting",
                )
            )
            await session.flush()
            if include_action:
                connector_id = f"conn-{sfx}"
                source_record_id = f"src-{sfx}"
                session.add(
                    orm.SourceConnector(
                        connector_id=connector_id,
                        source_product="mock_xdr",
                        display_name="Writeback test connector",
                    )
                )
                session.add(
                    orm.SourceObject(
                        source_record_id=source_record_id,
                        source_product="mock_xdr",
                        source_tenant_id="t1",
                        connector_id=connector_id,
                        source_kind="incident",
                        source_object_id=f"INC-{sfx}",
                    )
                )
                session.add(
                    orm.Action(
                        action_id=f"act-{sfx}",
                        event_id=event_id,
                        plan_revision=1,
                        action_fingerprint=f"fp-{sfx}",
                        action_category="response",
                        action_name="block ip",
                        tool_name="block_ip",
                        action_level="l2",
                        execution_owner="direct_tool",
                        writeback_required=True,
                        writeback_applicable=True,
                        writeback_readiness=writeback_readiness.value,
                        writeback_status=(
                            WritebackStatus.CONFIRMED.value
                            if outbox_status is WritebackStatus.CONFIRMED
                            else None
                        ),
                    )
                )
                await session.flush()
                if outbox_status is not None:
                    session.add(
                        orm.DispositionOutbox(
                            outbox_id=f"obx-{sfx}",
                            writeback_id=f"wbk-{sfx}",
                            disposition_id=f"disp-{sfx}",
                            action_id=f"act-{sfx}",
                            event_id=event_id,
                            closure_cycle=1,
                            source_record_id=source_record_id,
                            source_locator_hash="h" * 64,
                            source_sequence=1,
                            intent_kind=DispositionIntentKind.ENTITY_ACTION_SUBMIT.value,
                            logical_slot="slot-1",
                            idempotency_key=f"idem-{sfx}",
                            command_payload={},
                            command_payload_sha256="a" * 64,
                            delivery_status="delivered",
                            latest_writeback_status=outbox_status.value,
                        )
                    )
                    if outbox_status is WritebackStatus.CONFIRMED:
                        session.add(
                            orm.Action(
                                action_id=f"act-term-{sfx}",
                                event_id=event_id,
                                plan_revision=1,
                                action_fingerprint=f"fp-term-{sfx}",
                                action_category="response",
                                action_name=TERMINAL_DISPOSITION_TOOL,
                                tool_name="",
                                action_level="l1",
                                execution_owner="xdr_managed",
                                execution_phase=ActionExecutionPhase.POST_VERIFY.value,
                                writeback_required=True,
                                writeback_applicable=True,
                                writeback_readiness=WritebackReadiness.READY.value,
                                writeback_status=WritebackStatus.CONFIRMED.value,
                                approved_terminal_dispositions=[SourceDisposition.CONTAINED.value],
                            )
                        )
                        await session.flush()
                        session.add(
                            orm.DispositionOutbox(
                                outbox_id=f"obx-term-{sfx}",
                                writeback_id=f"wbk-term-{sfx}",
                                disposition_id=f"disp-term-{sfx}",
                                action_id=f"act-term-{sfx}",
                                event_id=event_id,
                                closure_cycle=1,
                                source_record_id=source_record_id,
                                source_locator_hash="h" * 64,
                                source_sequence=2,
                                intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE.value,
                                logical_slot="terminal",
                                idempotency_key=f"idem-term-{sfx}",
                                command_payload={
                                    "target_disposition": SourceDisposition.CONTAINED.value
                                },
                                command_payload_sha256="b" * 64,
                                delivery_status="delivered",
                                latest_writeback_status=WritebackStatus.CONFIRMED.value,
                            )
                        )
            await session.flush()
    return event_id


async def _seed_report_with_event(
    session_factory: async_sessionmaker[AsyncSession],
    event_id: str,
) -> None:
    """Insert a minimal report row so REPORTING events can close when gate passes."""
    from datetime import UTC, datetime

    from app.models.ids import report_id_for_event

    now = datetime.now(UTC)
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.Report(
                    report_id=report_id_for_event(event_id),
                    event_id=event_id,
                    title="Gate test report",
                    summary="fixture",
                    sections=[],
                    final_verdict=FinalVerdict.CONFIRMED_THREAT.value,
                    risk_score=85,
                    severity=Severity.HIGH.value,
                    version=1,
                    generated_by="test",
                    generated_at=now,
                    updated_at=now,
                )
            )
            await session.flush()


async def _seed_investigation_report(
    session_factory: async_sessionmaker[AsyncSession],
    event_id: str,
    *,
    final_verdict: FinalVerdict = FinalVerdict.CONFIRMED_THREAT,
) -> list[dict[str, str]]:
    """Insert a full investigation-style report (not quick_close)."""
    from datetime import UTC, datetime

    from app.models.ids import report_id_for_event

    sections = [
        {"key": "overview", "title": "Overview", "content": "Detailed investigation overview."},
        {"key": "evidence", "title": "Evidence", "content": "Collected DNS and asset evidence."},
    ]
    now = datetime.now(UTC)
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.Report(
                    report_id=report_id_for_event(event_id),
                    event_id=event_id,
                    title="Investigation Report",
                    summary="Full analysis report fixture",
                    sections=sections,
                    final_verdict=final_verdict.value,
                    risk_score=85,
                    severity=Severity.HIGH.value,
                    version=1,
                    generated_by="template",
                    generated_at=now,
                    updated_at=now,
                )
            )
            await session.flush()
    return sections


# --------------------------------------------------------------------------- #
# Tests: POST /events
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_event_returns_201(
    client: TestClient,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """POST /events creates an event and returns 201 with valid summary."""
    resp = client.post(
        "/api/v1/events",
        json={
            "event_type": "insider_threat",
            "title": "Test insider threat",
            "description": "API test event",
            "severity": "high",
            "creation_source_ref": {
                "source_kind": "alert",
                "source_product": "mock_xdr",
                "source_tenant_id": "t1",
                "connector_id": "conn-mock-1",
                "source_object_id": "ALT-99901",
                "source_status_raw": "open",
                "source_disposition": "pending",
                "schema_version": "1",
            },
        },
        headers=_hdr(),
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["event_id"].startswith("evt-")
    assert data["status"] == "new"
    assert data["event_type"] == "insider_threat"


@pytest.mark.asyncio
async def test_create_event_rejects_unknown_fields(
    client: TestClient,
) -> None:
    """Extra fields are rejected (extra='forbid' on request model)."""
    resp = client.post(
        "/api/v1/events",
        json={
            "event_type": "insider_threat",
            "title": "Test",
            "severity": "high",
            "unknown_field": "should_reject",
            "creation_source_ref": {
                "source_kind": "alert",
                "source_product": "mock_xdr",
                "source_tenant_id": "t1",
                "connector_id": "conn-mock-1",
                "source_object_id": "ALT-99902",
                "source_status_raw": "open",
                "source_disposition": "pending",
                "schema_version": "1",
            },
        },
        headers=_hdr(),
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Tests: GET /events
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_events_returns_paginated(
    client: TestClient,
    event_service: EventService,
) -> None:
    """GET /events returns correct pagination structure."""
    await _create_test_event(event_service, title="List test 1")
    await _create_test_event(event_service, title="List test 2")

    resp = client.get("/api/v1/events", headers=_hdr())
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "items" in data
    assert data["page"] == 1
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_list_events_filters_by_status(
    client: TestClient,
    event_service: EventService,
) -> None:
    """Filtering by status works."""
    await _create_test_event(event_service, title="Status test")

    resp = client.get("/api/v1/events?status=new", headers=_hdr())
    assert resp.status_code == 200
    data = resp.json()
    for item in data["items"]:
        assert item["status"] == "new"


# --------------------------------------------------------------------------- #
# Tests: GET /events/{id}
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_event_returns_detail(
    client: TestClient,
    event_service: EventService,
) -> None:
    """GET /events/{id} returns full event detail."""
    event_id = await _create_test_event(event_service, title="Detail test")

    resp = client.get(f"/api/v1/events/{event_id}", headers=_hdr())
    assert resp.status_code == 200
    data = resp.json()
    assert data["event"]["event_id"] == event_id
    assert data["event"]["title"] == "Detail test"
    assert "writeback_required" in data
    assert "writeback_readiness" in data


@pytest.mark.asyncio
async def test_get_event_surfaces_failed_writeback_status(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """FAILED outbox rows must appear in writeback_overall_status (not omitted)."""
    event_id = await _seed_reporting_required_event(
        session_factory,
        outbox_status=WritebackStatus.FAILED,
    )

    resp = client.get(f"/api/v1/events/{event_id}", headers=_hdr())
    assert resp.status_code == 200
    data = resp.json()
    assert data["writeback_overall_status"] == WritebackStatus.FAILED.value


@pytest.mark.asyncio
async def test_get_event_404_for_unknown_id(
    client: TestClient,
) -> None:
    """GET /events/{id} returns 404 for unknown ids."""
    resp = client.get("/api/v1/events/evt-99999999-ffffffff", headers=_hdr())
    assert resp.status_code == 404
    data = resp.json()
    assert data["error_code"] == "event_not_found"


# --------------------------------------------------------------------------- #
# Tests: GET /events/{id}/report
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_report_404_when_no_report(
    client: TestClient,
    event_service: EventService,
) -> None:
    """GET /events/{id}/report returns 404 when report doesn't exist."""
    event_id = await _create_test_event(event_service, title="No report")

    resp = client.get(f"/api/v1/events/{event_id}/report", headers=_hdr())
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Tests: GET /events/{id}/traces and audit-logs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_traces_returns_empty_for_new_event(
    client: TestClient,
    event_service: EventService,
) -> None:
    """GET /events/{id}/traces returns empty list for new event."""
    event_id = await _create_test_event(event_service, title="Traces test")

    resp = client.get(f"/api/v1/events/{event_id}/traces", headers=_hdr())
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


@pytest.mark.asyncio
async def test_audit_logs_returns_entries(
    client: TestClient,
    event_service: EventService,
) -> None:
    """GET /events/{id}/audit-logs returns creation audit entry."""
    event_id = await _create_test_event(event_service, title="Audit test")

    resp = client.get(f"/api/v1/events/{event_id}/audit-logs", headers=_hdr())
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert any(entry["reason"] == "event_created" for entry in data["items"])


# --------------------------------------------------------------------------- #
# Tests: GET /events/{id}/actions
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_actions_paginated(
    client: TestClient,
    event_service: EventService,
) -> None:
    """GET /events/{id}/actions returns paginated list."""
    event_id = await _create_test_event(event_service, title="Actions test")

    resp = client.get(
        f"/api/v1/events/{event_id}/actions?page=1&page_size=10",
        headers=_hdr(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "items" in data


# --------------------------------------------------------------------------- #
# Tests: GET /events/{id}/tool-calls and GET /tool-calls
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_event_tool_calls_empty(
    client: TestClient,
    event_service: EventService,
) -> None:
    """GET /events/{id}/tool-calls returns empty for new event."""
    event_id = await _create_test_event(event_service, title="Tool calls test")

    resp = client.get(f"/api/v1/events/{event_id}/tool-calls", headers=_hdr())
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_global_tool_calls_paginated(
    client: TestClient,
) -> None:
    """GET /tool-calls returns paginated list with optional filters."""
    resp = client.get(
        "/api/v1/tool-calls?page=1&page_size=10",
        headers=_hdr(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.asyncio
async def test_global_tool_calls_filter_by_tool_name(
    client: TestClient,
) -> None:
    """GET /tool-calls?tool_name=query_asset_info filters correctly."""
    resp = client.get(
        "/api/v1/tool-calls?tool_name=query_asset_info",
        headers=_hdr(),
    )
    assert resp.status_code == 200
    data = resp.json()
    for item in data["items"]:
        assert item["tool_name"] == "query_asset_info"


# --------------------------------------------------------------------------- #
# Tests: POST /events/{id}/close
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_close_event_404(
    client: TestClient,
) -> None:
    """POST /events/{id}/close returns 404 for unknown id."""
    resp = client.post(
        "/api/v1/events/evt-99999999-ffffffff/close",
        json={"reason": "test"},
        headers=_hdr(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_close_event_invalid_transition_from_new(
    client: TestClient,
    event_service: EventService,
) -> None:
    """Closing a NEW event directly must fail — invalid transition."""
    event_id = await _create_test_event(event_service, title="Close from NEW")

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "test close"},
        headers=_hdr(),
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error_code"] == "invalid_state_transition"


@pytest.mark.asyncio
async def test_force_close_requires_admin(
    client: TestClient,
    event_service: EventService,
) -> None:
    """Force local close requires admin role."""
    event_id = await _create_test_event(event_service, title="Force close test")

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "forced", "force_local_close": True},
        headers=_hdr("analyst"),
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_close_triaging_not_required_succeeds(
    client: TestClient,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Close a TRIAGING not_required event succeeds after generating report."""
    event_id = await _create_test_event(
        event_service,
        title="Close TRIAGING test",
        severity=Severity.LOW,
    )

    # Transition to TRIAGING directly via DB.
    async with session_factory() as session:
        async with session.begin():
            row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
            assert row is not None
            row.status = EventStatus.TRIAGING.value
            row.row_version = int(row.row_version or 1) + 1
            session.add(
                orm.EventAuditLog(
                    event_id=event_id,
                    from_status="new",
                    to_status="triaging",
                    operator="test",
                    reason="test_setup:triaging",
                )
            )
            await session.flush()

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "quick close test"},
        headers=_hdr(),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["event_id"] == event_id
    assert data["status"] == "closed"

    # Verify report was generated and is queryable.
    report_resp = client.get(
        f"/api/v1/events/{event_id}/report",
        headers=_hdr(),
    )
    assert report_resp.status_code == 200


@pytest.mark.asyncio
async def test_close_failed_succeeds_with_report(
    client: TestClient,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Close a FAILED event succeeds after generating report."""
    event_id = await _create_test_event(
        event_service,
        title="Close FAILED test",
        severity=Severity.LOW,
    )

    # Transition to FAILED directly via DB.
    async with session_factory() as session:
        async with session.begin():
            row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
            assert row is not None
            row.status = EventStatus.FAILED.value
            row.row_version = int(row.row_version or 1) + 1
            session.add(
                orm.EventAuditLog(
                    event_id=event_id,
                    from_status="new",
                    to_status="failed",
                    operator="test",
                    reason="test_setup:failed",
                )
            )
            await session.flush()

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "close failed test"},
        headers=_hdr(),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["event_id"] == event_id
    assert data["status"] == "closed"

    # Verify report was generated and is queryable.
    report_resp = client.get(
        f"/api/v1/events/{event_id}/report",
        headers=_hdr(),
    )
    assert report_resp.status_code == 200


@pytest.mark.asyncio
async def test_close_reporting_writeback_not_configured_rejected(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """REPORTING + required policy without disposition actions is blocked."""
    event_id = await _seed_reporting_required_event(
        session_factory,
        include_action=False,
    )
    await _seed_report_with_event(session_factory, event_id)

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "writeback gate test"},
        headers=_hdr(),
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "writeback_unsupported"


@pytest.mark.asyncio
async def test_close_reporting_writeback_pending_rejected(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_reporting_required_event(
        session_factory,
        outbox_status=WritebackStatus.PENDING,
    )
    await _seed_report_with_event(session_factory, event_id)

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "writeback pending test"},
        headers=_hdr(),
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "writeback_pending"


@pytest.mark.asyncio
async def test_close_reporting_writeback_failed_rejected(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_reporting_required_event(
        session_factory,
        outbox_status=WritebackStatus.FAILED,
    )
    await _seed_report_with_event(session_factory, event_id)

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "writeback failed test"},
        headers=_hdr(),
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "writeback_failed"


@pytest.mark.asyncio
async def test_close_reporting_writeback_conflict_rejected(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_reporting_required_event(
        session_factory,
        outbox_status=WritebackStatus.CONFLICT,
    )
    await _seed_report_with_event(session_factory, event_id)

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "writeback conflict test"},
        headers=_hdr(),
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "writeback_conflict"


@pytest.mark.asyncio
async def test_close_reporting_writeback_unsupported_readiness_rejected(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_reporting_required_event(
        session_factory,
        writeback_readiness=WritebackReadiness.CAPABILITY_UNKNOWN,
        outbox_status=WritebackStatus.PENDING,
    )
    await _seed_report_with_event(session_factory, event_id)

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "writeback unsupported test"},
        headers=_hdr(),
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "writeback_unsupported"


@pytest.mark.asyncio
async def test_close_reporting_writeback_unknown_rejected(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _seed_reporting_required_event(
        session_factory,
        outbox_status=WritebackStatus.UNKNOWN,
    )
    await _seed_report_with_event(session_factory, event_id)

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={"reason": "writeback unknown test"},
        headers=_hdr(),
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "writeback_pending"


@pytest.mark.asyncio
async def test_close_reporting_verdict_change_preserves_report_sections(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
    event_service: EventService,
) -> None:
    """Changing verdict on a full report must not replace sections with quick-close placeholders."""
    event_id = await _seed_reporting_required_event(
        session_factory,
        outbox_status=WritebackStatus.CONFIRMED,
    )
    original_sections = await _seed_investigation_report(session_factory, event_id)

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={
            "reason": "verdict change test",
            "final_verdict": "false_positive",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text

    report = await event_service.get_report(event_id=event_id)
    assert report is not None
    assert report.final_verdict == FinalVerdict.FALSE_POSITIVE
    assert report.generated_by == "template"
    assert len(report.sections) == len(original_sections)
    assert report.sections[0].content == original_sections[0]["content"]


@pytest.mark.asyncio
async def test_close_triaging_applies_requested_final_verdict(
    client: TestClient,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event_id = await _create_test_event(
        event_service,
        title="TRIAGING verdict test",
        severity=Severity.LOW,
    )

    async with session_factory() as session:
        async with session.begin():
            row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
            assert row is not None
            row.status = EventStatus.TRIAGING.value
            row.row_version = int(row.row_version or 1) + 1
            session.add(
                orm.EventAuditLog(
                    event_id=event_id,
                    from_status="new",
                    to_status="triaging",
                    operator="test",
                    reason="test_setup:triaging",
                )
            )
            await session.flush()

    resp = client.post(
        f"/api/v1/events/{event_id}/close",
        json={
            "reason": "triaging fp close",
            "final_verdict": "false_positive",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text

    report = await event_service.get_report(event_id=event_id)
    assert report is not None
    assert report.final_verdict == FinalVerdict.FALSE_POSITIVE


@pytest.mark.asyncio
async def test_investigate_http_low_risk_polls_to_closed(
    client: TestClient,
    event_service: EventService,
) -> None:
    """POST investigate (202) on a low-risk event completes at CLOSED via HTTP."""
    event_id = await _create_test_event(
        event_service,
        title="Investigate HTTP low risk",
        severity=Severity.LOW,
    )

    resp = client.post(
        f"/api/v1/events/{event_id}/investigate",
        headers=_hdr(),
    )
    assert resp.status_code == 202, resp.text

    detail = client.get(f"/api/v1/events/{event_id}", headers=_hdr())
    assert detail.status_code == 200
    assert detail.json()["event"]["status"] == "closed"

    report_resp = client.get(f"/api/v1/events/{event_id}/report", headers=_hdr())
    assert report_resp.status_code == 200


@pytest.mark.asyncio
async def test_investigate_high_risk_http_polls_to_reporting(
    client: TestClient,
    event_service: EventService,
) -> None:
    """High-risk required events stay at REPORTING when started via HTTP investigate."""
    from app.models.enums import SourceDisposition, SourceObjectKind
    from app.models.source import SourceReference
    from app.services.event_service import IngestableSource

    ref = SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="t1",
        connector_id="conn-mock-http-high",
        source_object_id="INC-HTTP-HIGH-001",
        source_status_raw="open",
        source_disposition=SourceDisposition.PENDING,
        schema_version="1",
    )
    ingest = IngestableSource(
        reference=ref,
        title="HTTP high risk incident",
        description="Serious incident for HTTP investigate test",
        event_type=EventType.DATA_EXFILTRATION,
        severity=Severity.HIGH,
    )
    result = await event_service.ingest_source_object(ingest)
    assert result.event_id is not None
    event_id = result.event_id

    resp = client.post(
        f"/api/v1/events/{event_id}/investigate",
        headers=_hdr(),
    )
    assert resp.status_code == 202, resp.text

    detail = client.get(f"/api/v1/events/{event_id}", headers=_hdr())
    assert detail.status_code == 200
    assert detail.json()["event"]["status"] == "reporting"

    report_resp = client.get(f"/api/v1/events/{event_id}/report", headers=_hdr())
    assert report_resp.status_code == 200


@pytest.mark.asyncio
async def test_investigate_http_flow_polls_to_completion(
    client: TestClient,
    event_service: EventService,
) -> None:
    """Backward-compatible alias for the low-risk HTTP investigate path."""
    await test_investigate_http_low_risk_polls_to_closed(client, event_service)


# --------------------------------------------------------------------------- #
# Tests: POST /events/{id}/investigate
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_investigate_404(
    client: TestClient,
) -> None:
    """POST /events/{id}/investigate returns 404 for unknown id."""
    resp = client.post(
        "/api/v1/events/evt-99999999-ffffffff/investigate",
        headers=_hdr(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_investigate_closed_rejected(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Cannot investigate a CLOSED event."""
    # Directly insert a closed event via session.
    async with session_factory() as session:
        async with session.begin():
            import hashlib
            from datetime import UTC, datetime

            now = datetime.now(UTC)
            eid = "evt-20260101-closed99"
            session.add(
                orm.SecurityEvent(
                    event_id=eid,
                    event_type="insider_threat",
                    title="Closed event",
                    description="Already closed",
                    status="closed",
                    severity="high",
                    final_verdict="none",
                    entities={},
                    creation_source_ref={
                        "source_kind": "alert",
                        "source_product": "file",
                        "source_tenant_id": "local",
                        "connector_id": "file-local",
                        "source_object_id": "file-closed99",
                        "raw_payload_hash": hashlib.sha256(b"closed").hexdigest(),
                        "ingested_at": now.isoformat(),
                    },
                    source_reference_snapshots=[],
                    disposition_policy="not_required",
                    source_type="manual",
                    occurred_at=now,
                    row_version=1,
                )
            )
            session.add(
                orm.EventAuditLog(
                    event_id=eid,
                    from_status=None,
                    to_status="new",
                    operator="test",
                    reason="test_setup",
                )
            )
            session.add(
                orm.EventAuditLog(
                    event_id=eid,
                    from_status="new",
                    to_status="closed",
                    operator="test",
                    reason="test_setup",
                )
            )
            await session.flush()

    resp = client.post(
        f"/api/v1/events/{eid}/investigate",
        headers=_hdr(),
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error_code"] == "invalid_state_transition"


@pytest.mark.asyncio
async def test_investigate_returns_202(
    client: TestClient,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """POST /events/{id}/investigate returns 202 with task_id matching event_id."""
    event_id = await _create_test_event(event_service, title="Investigate 202 test")

    resp = client.post(
        f"/api/v1/events/{event_id}/investigate",
        headers=_hdr(),
    )
    assert resp.status_code == 202, f"Expected 202 Accepted, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["task_id"] == event_id
    assert data["event_id"] == event_id


# --------------------------------------------------------------------------- #
# Helper: integration pipeline with tool executor + evidence projection
# --------------------------------------------------------------------------- #


def _build_integration_pipeline(
    *,
    event_service: EventService,
    state_machine_service,
    session_factory: async_sessionmaker[AsyncSession],
    context_store: Any | None = None,
):
    """Build AnalysisOnlyPipeline wired like production deps (ISSUE-039)."""
    from app.agents.evidence_agent import EvidenceAgent
    from app.agents.rag_agent import RAGAgent
    from app.agents.report_agent import ReportAgent
    from app.agents.risk_agent import RiskAgent
    from app.agents.triage_agent import TriageAgent
    from app.core.config import get_settings
    from app.core.redis_client import RedisClient
    from app.services.analysis_only_pipeline import AnalysisOnlyPipeline
    from app.services.context_service import EventContextStore
    from app.services.degraded_flag_service import DegradedFlagService
    from app.services.evidence_projection import EvidenceProjection, bind_evidence_projection
    from app.services.working_memory import WorkingMemory
    from app.tools.executor import get_tool_executor

    settings = get_settings()
    redis = RedisClient(url=settings.redis_url)
    store = context_store or EventContextStore(redis, session_factory)
    degraded = DegradedFlagService(store, session_factory)
    wm = WorkingMemory(store=store, redis=redis, degraded_flags=degraded)
    tool_executor = get_tool_executor()

    triage = TriageAgent(
        llm_client=None,
        working_memory=wm.for_writer("TriageAgent"),
    )
    evidence = EvidenceAgent(
        llm_client=None,
        tool_executor=tool_executor,
        working_memory=wm.for_writer("EvidenceAgent"),
        event_service=event_service,
        session_factory=session_factory,
    )
    rag = RAGAgent(
        working_memory=wm.for_writer("RAGAgent"),
        pipeline=None,
    )
    risk = RiskAgent(
        llm_client=None,
        working_memory=wm.for_writer("RiskAgent"),
        event_service=event_service,
        scenario_id="insider_data_exfiltration",
    )
    report = ReportAgent(
        llm_client=None,
        working_memory=wm.for_writer("ReportAgent"),
        event_service=event_service,
        scenario_id="insider_data_exfiltration",
    )

    pipeline = AnalysisOnlyPipeline(
        event_service=event_service,
        state_machine=state_machine_service,
        triage_agent=triage,
        evidence_agent=evidence,
        rag_agent=rag,
        risk_agent=risk,
        report_agent=report,
        context_store=store,
        settings=settings,
    )
    projection = EvidenceProjection(session_factory)
    return pipeline, projection, bind_evidence_projection, store


# --------------------------------------------------------------------------- #
# Conftest-level integration: run the full analysis pipeline end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_full_analysis_pipeline_happy_path(
    client: TestClient,
    event_service: EventService,
    state_machine_service,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """End-to-end: create → investigate → poll → report → close.

    For a not_required event, the pipeline should complete with the event CLOSED.
    """
    pipeline, projection, bind_projection, _store = _build_integration_pipeline(
        event_service=event_service,
        state_machine_service=state_machine_service,
        session_factory=session_factory,
    )

    # Create a not_required low-severity event.
    event = await event_service.create_event(
        {"title": "Pipeline test", "description": "Low risk event"},
        source_type="manual",
        title="Pipeline test",
        event_type=EventType.ACCOUNT_ANOMALY,
        severity=Severity.LOW,
    )
    event_id = event.event_id
    assert event.status == EventStatus.NEW

    with bind_projection(projection):
        result = await pipeline.run(event_id)

    assert result.event_id == event_id
    assert result.analysis_only_complete is True

    # After pipeline: should be CLOSED (not_required + low severity = short-circuit close).
    event = await event_service.get_event(event_id)
    assert event is not None
    assert event.status == EventStatus.CLOSED


@pytest.mark.asyncio
async def test_high_risk_event_stays_reporting(
    client: TestClient,
    event_service: EventService,
    state_machine_service,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """High-risk required events stay at REPORTING after analysis."""
    from app.models.enums import SourceDisposition, SourceObjectKind
    from app.models.source import SourceReference
    from app.services.event_service import IngestableSource

    pipeline, projection, bind_projection, _store = _build_integration_pipeline(
        event_service=event_service,
        state_machine_service=state_machine_service,
        session_factory=session_factory,
    )

    ref = SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="t1",
        connector_id="conn-mock-high",
        source_object_id="INC-HIGH-001",
        source_status_raw="open",
        source_disposition=SourceDisposition.PENDING,
        schema_version="1",
    )
    ingest = IngestableSource(
        reference=ref,
        title="High risk incident",
        description="A serious data exfiltration incident",
        event_type=EventType.DATA_EXFILTRATION,
        severity=Severity.HIGH,
    )
    result = await event_service.ingest_source_object(ingest)
    assert result.event_id is not None
    event_id = result.event_id

    event = await event_service.get_event(event_id)
    assert event is not None

    if event.status == EventStatus.NEW:
        with bind_projection(projection):
            pipeline_result = await pipeline.run(event_id)
        assert pipeline_result.disposition_policy == "required"
        assert pipeline_result.analysis_only_complete is True

        event = await event_service.get_event(event_id)
        assert event is not None
        assert event.status == EventStatus.REPORTING


@pytest.mark.asyncio
async def test_analysis_only_complete_persisted_in_context(
    client: TestClient,
    event_service: EventService,
    state_machine_service,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """analysis_only_complete is persisted to EventContextStore after pipeline runs."""
    pipeline, projection, bind_projection, store = _build_integration_pipeline(
        event_service=event_service,
        state_machine_service=state_machine_service,
        session_factory=session_factory,
    )

    event = await event_service.create_event(
        {"title": "Persistence test", "description": "Low risk"},
        source_type="manual",
        title="Persistence test",
        event_type=EventType.ACCOUNT_ANOMALY,
        severity=Severity.LOW,
    )
    event_id = event.event_id

    with bind_projection(projection):
        result = await pipeline.run(event_id)
    assert result.analysis_only_complete is True

    stored_value = await store.get(event_id, "analysis_only_complete")
    assert stored_value is True, (
        f"Expected analysis_only_complete=True in context, got {stored_value!r}"
    )
