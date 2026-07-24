"""AuthN / AuthZ tests (ISSUE-004 step 6)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import schemas as s
from app.core.config import get_settings
from app.main import app

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
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from app.api.v1.deps import get_approval_engine
    from app.api.v1.deps import get_disposition_sync as _real_get_disposition_sync
    from app.api.v1.deps import get_event_service as _real_get_event_service
    from app.api.v1.deps import get_state_machine as _real_get_state_machine
    from tests.test_api.test_contracts import (
        _MockDispositionSyncService,
        _MockEventService,
        _MockStateMachine,
    )

    class _StubApprovalEngine:
        async def approve(self, *args: object, **kwargs: object) -> None:
            return None

        async def reject(self, *args: object, **kwargs: object) -> None:
            return None

        async def scan_timeouts(self) -> list[str]:
            return []

    async def _stub_engine() -> _StubApprovalEngine:
        return _StubApprovalEngine()

    async def _mock_event_service() -> _MockEventService:
        return _MockEventService()

    async def _mock_state_machine() -> _MockStateMachine:
        return _MockStateMachine()

    async def _mock_disposition_sync() -> _MockDispositionSyncService:
        return _MockDispositionSyncService()

    app.dependency_overrides[get_approval_engine] = _stub_engine
    app.dependency_overrides[_real_get_event_service] = _mock_event_service
    app.dependency_overrides[_real_get_state_machine] = _mock_state_machine
    app.dependency_overrides[_real_get_disposition_sync] = _mock_disposition_sync
    yield TestClient(app)
    app.dependency_overrides.pop(get_approval_engine, None)
    app.dependency_overrides.pop(_real_get_event_service, None)
    app.dependency_overrides.pop(_real_get_state_machine, None)
    app.dependency_overrides.pop(_real_get_disposition_sync, None)


def _hdr(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {role}-token"}


def test_anonymous_is_rejected(client: TestClient) -> None:
    resp = client.get("/api/v1/events")
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "unauthorized"


def test_wrong_role_is_forbidden(client: TestClient) -> None:
    # analyst cannot approve (needs approver).
    resp = client.post(
        "/api/v1/actions/act-1/approve", headers=_hdr("analyst"), json={"comment": "ok"}
    )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "forbidden"


def test_approver_can_approve(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/actions/act-1/approve", headers=_hdr("approver"), json={"comment": "ok"}
    )
    assert resp.status_code == 200


def test_body_cannot_forge_operator(client: TestClient) -> None:
    # extra="forbid" on the request model rejects a client-supplied operator.
    resp = client.post(
        "/api/v1/actions/act-1/approve",
        headers=_hdr("approver"),
        json={"comment": "ok", "operator": "root"},
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "validation_error"


def test_retry_requires_disposition_operator(client: TestClient) -> None:
    forbidden = client.post("/api/v1/writebacks/wbk-0a1b2c3d/retry", headers=_hdr("analyst"))
    assert forbidden.status_code == 403
    ok = client.post("/api/v1/writebacks/wbk-0a1b2c3d/retry", headers=_hdr("operator"))
    assert ok.status_code == 200


def test_resolve_writeback_requires_admin(client: TestClient) -> None:
    body = {
        "resolution": "manual_confirmed",
        "comment": "verified",
        "evidence_ref": "evidence://verified",
    }
    forbidden = client.post(
        "/api/v1/writebacks/wbk-0a1b2c3d/resolve", headers=_hdr("operator"), json=body
    )
    assert forbidden.status_code == 403
    ok = client.post("/api/v1/writebacks/wbk-0a1b2c3d/resolve", headers=_hdr("admin"), json=body)
    assert ok.status_code == 200


def test_force_local_close_requires_admin(client: TestClient) -> None:
    body = {"reason": "manual", "force_local_close": True}
    forbidden = client.post(
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/close", headers=_hdr("analyst"), json=body
    )
    assert forbidden.status_code == 403
    ok = client.post(f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/close", headers=_hdr("admin"), json=body)
    assert ok.status_code == 200
    assert ok.json()["external_unsynced"] is True


def test_trusted_proxy_headers_only_honored_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers = {"X-Auth-Subject": "proxied-user", "X-Auth-Roles": "analyst"}
    # Disabled proxy: identity headers are ignored -> anonymous -> 401.
    disabled = client.get("/api/v1/events", headers=headers)
    assert disabled.status_code == 401

    # Enabled + client host allowlisted: headers are honored.
    monkeypatch.setenv("TRUSTED_AUTH_PROXY_ENABLED", "true")
    monkeypatch.setenv("TRUSTED_PROXY_ALLOWLIST", "testclient")
    enabled = client.get("/api/v1/events", headers=headers)
    assert enabled.status_code == 200

    # Enabled but client host NOT allowlisted: headers ignored -> 401.
    monkeypatch.setenv("TRUSTED_PROXY_ALLOWLIST", "10.0.0.1")
    blocked = client.get("/api/v1/events", headers=headers)
    assert blocked.status_code == 401


def test_dev_token_rejected_in_production(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    # ISSUE-093 §5 + ISSUE-027: production Settings fail-closes on mock
    # source/tool/disposition/LLM or simulation modes, so use live-shaped
    # runtime modes to isolate the assertion from that unrelated gate.
    monkeypatch.setenv("SOURCE_MODE", "live_edr")
    monkeypatch.setenv("TOOL_MODE", "live")
    monkeypatch.setenv("DISPOSITION_MODE", "live_xdr")
    monkeypatch.setenv("DISPOSITION_ADAPTER_KIND", "http")
    monkeypatch.setenv("LLM_MODE", "openai_compatible")
    monkeypatch.setenv("SIMULATION_ENABLED", "false")
    get_settings.cache_clear()
    resp = client.get("/api/v1/events", headers=_hdr("admin"))
    assert resp.status_code == 401
