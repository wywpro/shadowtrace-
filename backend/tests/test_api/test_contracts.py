"""API contract tests (ISSUE-004 acceptance 1-4 + step 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.v1 import schemas as s
from app.api.v1.errors import register_exception_handlers
from app.core.errors import ValidationError as DomainValidationError
from app.main import app
from app.models.disposition import DispositionCommand

# (method, path) pairs for every core endpoint in intro §4.2.2.
CORE_ENDPOINTS = {
    ("post", "/api/v1/events"),
    ("get", "/api/v1/events"),
    ("get", "/api/v1/events/{event_id}"),
    ("post", "/api/v1/events/{event_id}/investigate"),
    ("post", "/api/v1/events/{event_id}/close"),
    ("get", "/api/v1/events/{event_id}/report"),
    ("get", "/api/v1/events/{event_id}/traces"),
    ("get", "/api/v1/events/{event_id}/audit-logs"),
    ("get", "/api/v1/events/{event_id}/tool-calls"),
    ("get", "/api/v1/events/{event_id}/timeline"),
    ("get", "/api/v1/events/{event_id}/graph"),
    ("get", "/api/v1/events/{event_id}/decision-trace"),
    ("get", "/api/v1/events/{event_id}/actions"),
    ("post", "/api/v1/actions/{action_id}/approve"),
    ("post", "/api/v1/actions/{action_id}/reject"),
    ("post", "/api/v1/actions/{action_id}/resolve-unknown"),
    ("post", "/api/v1/ingestion/source-records"),
    ("get", "/api/v1/source-records/{source_record_id}"),
    ("get", "/api/v1/connectors"),
    ("put", "/api/v1/events/{event_id}/disposition-source"),
    ("post", "/api/v1/events/{event_id}/disposition-readiness/recheck"),
    ("get", "/api/v1/events/{event_id}/dispositions"),
    ("get", "/api/v1/dispositions/{disposition_id}"),
    ("get", "/api/v1/writebacks/{writeback_id}"),
    ("post", "/api/v1/writebacks/{writeback_id}/retry"),
    ("post", "/api/v1/writebacks/{writeback_id}/resolve"),
    ("get", "/api/v1/execution-jobs/{job_id}"),
    ("get", "/api/v1/tool-calls"),
    ("get", "/api/v1/tasks/{task_id}"),
    ("get", "/api/v1/tools"),
    ("get", "/api/v1/knowledge"),
    ("get", "/api/v1/health"),
    ("get", "/api/v1/stats"),
}

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


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _hdr(role: str = "analyst") -> dict[str, str]:
    return {"Authorization": f"Bearer {role}-token"}


def test_openapi_has_all_core_paths_and_methods() -> None:
    schema = app.openapi()
    assert schema["openapi"].startswith("3.")
    for method, path in CORE_ENDPOINTS:
        assert path in schema["paths"], f"missing path {path}"
        assert method in schema["paths"][path], f"missing {method.upper()} {path}"


def test_export_openapi_writes_valid_json(tmp_path: Path) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[3] / "scripts" / "export_openapi.py"
    spec = importlib.util.spec_from_file_location("export_openapi", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "openapi.json"
    mod.export_openapi(out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["openapi"].startswith("3.")
    assert "/api/v1/events" in doc["paths"]


@pytest.mark.parametrize(
    "path",
    [
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/report",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/traces",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/audit-logs",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/tool-calls",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/timeline",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/graph",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/decision-trace",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/actions",
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/dispositions",
        "/api/v1/events?page=1&page_size=20",
        "/api/v1/connectors",
        "/api/v1/source-records/src-associated-1",
        "/api/v1/dispositions/disp-0a1b2c3d",
        "/api/v1/writebacks/wbk-0a1b2c3d",
        "/api/v1/execution-jobs/job-0a1b2c3d",
        "/api/v1/tasks/task-1",
        "/api/v1/tools",
        "/api/v1/tool-calls",
        "/api/v1/knowledge",
        "/api/v1/stats",
    ],
)
def test_placeholder_get_endpoints_validate(client: TestClient, path: str) -> None:
    # 200 implies the placeholder passed its response_model validation.
    resp = client.get(path, headers=_hdr("analyst"))
    assert resp.status_code == 200, resp.text


def test_event_list_declares_mandated_query_params() -> None:
    # intro §4.2 / ISSUE-004 naming §3: the event list contract must expose the
    # full documented filter/sort/pagination parameter set.
    schema = app.openapi()
    params = {p["name"] for p in schema["paths"]["/api/v1/events"]["get"].get("parameters", [])}
    expected = {
        "page",
        "page_size",
        "status",
        "severity",
        "event_type",
        "final_verdict",
        "keyword",
        "start_time",
        "end_time",
        "sort_by",
        "sort_order",
    }
    assert expected <= params, {"missing": expected - params}


def test_actions_list_is_paginated() -> None:
    # GET /events/{event_id}/actions must be a paginated list (contract-stable
    # for the ISSUE-038 real implementation).
    op = app.openapi()["paths"]["/api/v1/events/{event_id}/actions"]["get"]
    params = {p["name"] for p in op.get("parameters", [])}
    assert {"page", "page_size", "status"} <= params, {"present": params}


def test_event_not_found_error_body(client: TestClient) -> None:
    resp = client.get("/api/v1/events/evt-does-not-exist", headers=_hdr("analyst"))
    assert resp.status_code == 404
    body = resp.json()
    assert set(body) >= {"error_code", "error_message", "details"}
    assert body["error_code"] == "event_not_found"


def test_invalid_state_transition_error_body(client: TestClient) -> None:
    resp = client.post(
        f"/api/v1/events/{s.EXAMPLE_CLOSED_EVENT_ID}/investigate",
        headers=_hdr("analyst"),
        json={},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert set(body) >= {"error_code", "error_message", "details"}
    assert body["error_code"] == "invalid_state_transition"


def test_validation_error_does_not_echo_rejected_payload_or_pydantic_url(
    client: TestClient,
) -> None:
    secrets = {
        "password": "password-value-must-not-leak",
        "token": "token-value-must-not-leak",
        "cookie": "cookie-value-must-not-leak",
        "Authorization": "Bearer authorization-value-must-not-leak",
    }
    response = client.post(
        "/api/v1/actions/act-0a1b2c3d/approve",
        headers=_hdr("approver"),
        json={"comment": "safe", "decision_id": "decision-1", **secrets},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "validation_error"
    assert body["error_message"] == "request validation failed"
    assert body["details"]["errors"]
    assert all(set(error) == {"loc", "type", "msg"} for error in body["details"]["errors"])
    serialized = json.dumps(body)
    assert '"input"' not in serialized
    assert "errors.pydantic.dev" not in serialized
    assert all(secret not in serialized for secret in secrets.values())


def test_domain_error_details_are_redacted_before_api_response() -> None:
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/error")
    async def _error() -> None:
        raise DomainValidationError(
            "provider rejected Authorization: Bearer domain-message-secret",
            details={
                "password": "domain-password-secret",
                "note": "token=domain-note-secret",
            },
        )

    response = TestClient(test_app).get("/error")
    assert response.status_code == 422
    serialized = response.text
    assert "domain-message-secret" not in serialized
    assert "domain-password-secret" not in serialized
    assert "domain-note-secret" not in serialized
    assert "[REDACTED]" in serialized


def test_disposition_command_rejects_analysis_fields() -> None:
    # Outbound envelope must never carry Action.parameters/reason/raw etc.
    valid = s.example_disposition_command().model_dump()
    with pytest.raises(ValidationError):
        DispositionCommand(**valid, reason="leaked analysis text")


def test_disposition_command_outbound_keys_are_allowlisted() -> None:
    payload = s.example_disposition_command().model_dump()
    forbidden = {"parameters", "reason", "raw_result", "prompt", "evidence"}
    assert forbidden.isdisjoint(payload.keys())


def test_writeback_response_never_exposes_raw_result(client: TestClient) -> None:
    resp = client.get("/api/v1/writebacks/wbk-0a1b2c3d", headers=_hdr("analyst"))
    assert resp.status_code == 200
    assert "raw_result" not in resp.json()


def test_execution_job_partial_success(client: TestClient) -> None:
    resp = client.get("/api/v1/execution-jobs/job-0a1b2c3d", headers=_hdr("analyst"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "partial_success"
    statuses = {t["status"] for t in body["target_results"]}
    assert statuses == {"success", "failed"}


def test_writeback_retry_requires_verification_then_idempotent(client: TestClient) -> None:
    # UNKNOWN must be verified before retry.
    unknown = client.post("/api/v1/writebacks/wbk-unknown/retry", headers=_hdr("operator"))
    assert unknown.status_code == 409
    assert unknown.json()["error_code"] == "writeback_conflict"

    # A known confirmed writeback re-enqueues idempotently (repeatable).
    first = client.post("/api/v1/writebacks/wbk-0a1b2c3d/retry", headers=_hdr("operator"))
    second = client.post("/api/v1/writebacks/wbk-0a1b2c3d/retry", headers=_hdr("operator"))
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_readiness_recheck_is_idempotent(client: TestClient) -> None:
    body = {"expected_event_version": 1}
    r1 = client.post(
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/disposition-readiness/recheck",
        headers=_hdr("operator"),
        json=body,
    )
    r2 = client.post(
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/disposition-readiness/recheck",
        headers=_hdr("operator"),
        json=body,
    )
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()


def test_select_disposition_source_rejects_unassociated_source(client: TestClient) -> None:
    resp = client.put(
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/disposition-source",
        headers=_hdr("operator"),
        json={"source_record_id": "src-other-tenant", "expected_event_version": 1},
    )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "disposition_permission_denied"


def test_select_disposition_source_version_cas(client: TestClient) -> None:
    resp = client.put(
        f"/api/v1/events/{s.EXAMPLE_EVENT_ID}/disposition-source",
        headers=_hdr("operator"),
        json={"source_record_id": "src-associated-1", "expected_event_version": 999},
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "writeback_conflict"
