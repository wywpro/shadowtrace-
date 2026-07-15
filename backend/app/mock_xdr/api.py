"""HTTP routes for MockXDRServer under ``/mock-xdr/v1`` (ISSUE-010)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.mock_xdr.models import MockXDRScenario
from app.mock_xdr.state import (
    MockAuthError,
    MockConflictError,
    MockValidationError,
    MockXDRState,
    find_forbidden_analysis_keys,
)
from app.models.disposition import DispositionCommand
from app.models.enums import ExecutionJobStatus, SourceDisposition

ObjectKind = Literal["incident", "alert", "asset", "log"]

# Shared process-local state for the standalone Mock server / TestClient.
GLOBAL_STATE = MockXDRState()


class MockMalformedPayloadError(Exception):
    """Control-flow signal for deterministic HTTP-200 malformed responses."""


def create_app(*, state: MockXDRState | None = None) -> FastAPI:
    """Build a standalone Mock XDR FastAPI application."""
    app = FastAPI(title="ShadowTrace MockXDRServer", version="0.1.0")
    runtime = state or GLOBAL_STATE

    def _state() -> MockXDRState:
        return runtime

    def _require_read(
        authorization: str | None = Header(default=None),
    ) -> None:
        # Auth is token-only. Read is weaker, so the write token also grants read;
        # any missing/unknown token is rejected (no header-based bypass).
        token = _bearer(authorization)
        if token not in (runtime.read_token, runtime.write_token):
            raise HTTPException(status_code=401, detail={"error_code": "unauthorized"})

    def _require_write(
        authorization: str | None = Header(default=None),
    ) -> None:
        if _bearer(authorization) != runtime.write_token:
            raise HTTPException(status_code=401, detail={"error_code": "unauthorized"})

    @app.exception_handler(MockValidationError)
    async def _validation(_: Request, exc: MockValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error_code": exc.error_code,
                "error_message": exc.message,
                "details": {},
            },
        )

    @app.exception_handler(MockAuthError)
    async def _auth(_: Request, exc: MockAuthError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"error_code": exc.error_code, "error_message": exc.message, "details": {}},
        )

    @app.exception_handler(MockConflictError)
    async def _conflict(_: Request, exc: MockConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error_code": exc.error_code, "error_message": exc.message, "details": {}},
        )

    @app.exception_handler(MockMalformedPayloadError)
    async def _malformed(_: Request, __: MockMalformedPayloadError) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "items": {"unexpected": "not-a-list"},
                "malformed_payload": True,
            },
        )

    # ---- health / meta -----------------------------------------------------

    @app.get("/mock-xdr/v1/health")
    def health(st: MockXDRState = Depends(_state)) -> dict[str, Any]:
        return {
            "status": "ok",
            "clock": st.clock.isoformat(),
            "object_counts": _counts(st),
            "schema_version": st.failure_profile.schema_version_override or "1",
            "seed": st.failure_profile.seed,
        }

    # ---- read routes (read client) ----------------------------------------

    @app.get("/mock-xdr/v1/incidents")
    def list_incidents(
        page_size: int = Query(default=100, ge=1, le=1000),
        cursor: str | None = None,
        updated_after: datetime | None = None,
        commit: bool = False,
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _maybe_fault(st)
        return st.list_page(
            "incident",
            page_size=page_size,
            cursor=cursor,
            updated_after=updated_after,
            commit_watermark=commit,
        )

    @app.get("/mock-xdr/v1/alerts")
    def list_alerts(
        page_size: int = Query(default=100, ge=1, le=1000),
        cursor: str | None = None,
        updated_after: datetime | None = None,
        commit: bool = False,
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _maybe_fault(st)
        return st.list_page(
            "alert",
            page_size=page_size,
            cursor=cursor,
            updated_after=updated_after,
            commit_watermark=commit,
        )

    @app.get("/mock-xdr/v1/assets")
    def list_assets(
        page_size: int = Query(default=100, ge=1, le=1000),
        cursor: str | None = None,
        updated_after: datetime | None = None,
        commit: bool = False,
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _maybe_fault(st)
        return st.list_page(
            "asset",
            page_size=page_size,
            cursor=cursor,
            updated_after=updated_after,
            commit_watermark=commit,
        )

    @app.get("/mock-xdr/v1/logs")
    def list_logs(
        page_size: int = Query(default=100, ge=1, le=1000),
        cursor: str | None = None,
        updated_after: datetime | None = None,
        commit: bool = False,
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _maybe_fault(st)
        return st.list_page(
            "log",
            page_size=page_size,
            cursor=cursor,
            updated_after=updated_after,
            commit_watermark=commit,
        )

    @app.get("/mock-xdr/v1/evidence")
    def list_evidence(
        updated_after: datetime | None = None,
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _maybe_fault(st)
        scenario = st.scenario
        records_by_source: dict[str, list[dict[str, Any]]] = {}
        if scenario is not None:
            for record in scenario.telemetry_timeline:
                logged_at = _telemetry_time(record)
                if (
                    updated_after is not None
                    and logged_at is not None
                    and logged_at <= updated_after
                ):
                    continue
                channel = str(record.get("channel") or "unknown")
                records_by_source.setdefault(channel, []).append(dict(record))
        return {
            "records_by_source": records_by_source,
            "source_product": "mock_xdr",
            "source_tenant_id": (scenario.source_tenant_id if scenario is not None else "unknown"),
            "connector_id": "mock_xdr-evidence",
            "schema_version": st.failure_profile.schema_version_override or "1",
        }

    @app.get("/mock-xdr/v1/connectors")
    def list_connectors(
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        return {
            "items": [c.model_dump(mode="json") for c in st.connectors.values()],
            "page_size": len(st.connectors),
            "next_cursor": None,
        }

    # ---- write routes (write client) --------------------------------------

    @app.post("/mock-xdr/v1/dispositions")
    def post_disposition(
        payload: dict[str, Any],
        _: None = Depends(_require_write),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _maybe_fault(st)
        leaks = find_forbidden_analysis_keys(payload)
        if leaks:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "unauthorized_field",
                    "error_message": "analysis/report/prompt/evidence fields forbidden",
                    "details": {"paths": leaks},
                },
            )
        try:
            command = DispositionCommand.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — surface pydantic as mock validation
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "validation_error",
                    "error_message": str(exc),
                    "details": {},
                },
            ) from exc
        receipt = st.submit_disposition(command)
        return receipt.model_dump(mode="json")

    @app.get("/mock-xdr/v1/disposition-jobs/{provider_job_id}")
    def get_disposition_job(
        provider_job_id: str,
        _: None = Depends(_require_write),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        job = st.get_job(provider_job_id)
        return {
            "provider_job_id": job.provider_job_id,
            "disposition_id": job.disposition_id,
            "status": job.status.value,
            "writeback_id": job.writeback_id,
            "terminal_writeback_status": (
                job.terminal_writeback_status.value if job.terminal_writeback_status else None
            ),
            # Explicit: job status is NOT a WritebackStatus.
            "status_domain": "ExecutionJobStatus",
        }

    @app.get("/mock-xdr/v1/dispositions/by-idempotency/{key_hash}")
    def lookup_idempotency(
        key_hash: str,
        _: None = Depends(_require_write),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        receipt = st.lookup_by_idempotency(key_hash)
        if receipt is None:
            raise HTTPException(
                status_code=404,
                detail={"error_code": "not_found", "error_message": "no submission", "details": {}},
            )
        return receipt.model_dump(mode="json")

    # ---- test/demo control plane ------------------------------------------

    @app.post("/mock-xdr/v1/control/seed")
    def control_seed(
        scenario: dict[str, Any],
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _require_control(st)
        loaded = MockXDRScenario.model_validate(scenario)
        st.load_scenario(loaded)
        return {
            "scenario_id": loaded.scenario_id,
            "object_counts": _counts(st),
            "seed": st.failure_profile.seed,
            "schema_version": st.failure_profile.schema_version_override or "1",
        }

    @app.post("/mock-xdr/v1/control/advance-clock")
    def control_advance_clock(
        seconds: float = 1.0,
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _require_control(st)
        clock = st.advance_clock(seconds)
        return {"clock": clock.isoformat()}

    @app.post("/mock-xdr/v1/control/confirm/{disposition_id}")
    def control_confirm(
        disposition_id: str,
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _require_control(st)
        receipt = st.confirm_via_readback(disposition_id)
        return receipt.model_dump(mode="json")

    @app.post("/mock-xdr/v1/control/jobs/{provider_job_id}/advance")
    def control_advance_job(
        provider_job_id: str,
        target: str,
        provider_confirmed_terminal: bool = False,
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _require_control(st)
        job = st.advance_job(
            provider_job_id,
            ExecutionJobStatus(target),
            provider_confirmed_terminal=provider_confirmed_terminal,
        )
        return {
            "provider_job_id": job.provider_job_id,
            "status": job.status.value,
            "terminal_writeback_status": (
                job.terminal_writeback_status.value if job.terminal_writeback_status else None
            ),
        }

    @app.post("/mock-xdr/v1/control/source-disposition")
    def control_source_disposition(
        kind: ObjectKind,
        object_id: str,
        target: str,
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _require_control(st)
        st.transition_source_disposition(
            kind,
            object_id,
            SourceDisposition(target),
            allow_unknown_recovery=True,
        )
        return st.readback_source_disposition(kind, object_id)

    @app.get("/mock-xdr/v1/control/captured-requests")
    def control_captured(st: MockXDRState = Depends(_state)) -> dict[str, Any]:
        _require_control(st)
        return {"items": list(st.captured_requests)}

    # Parameterized object routes MUST come after fixed paths like /control/...
    @app.get("/mock-xdr/v1/{kind}/{object_id}")
    def get_object(
        kind: ObjectKind,
        object_id: str,
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        _maybe_fault(st)
        return st.get_object(kind, object_id)

    @app.get("/mock-xdr/v1/{kind}/{object_id}/disposition")
    def readback_disposition(
        kind: ObjectKind,
        object_id: str,
        _: None = Depends(_require_read),
        st: MockXDRState = Depends(_state),
    ) -> dict[str, Any]:
        return st.readback_source_disposition(kind, object_id)

    return app


def _telemetry_time(record: dict[str, Any]) -> datetime | None:
    raw = record.get("logged_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return authorization.strip()


def _require_control(st: MockXDRState) -> None:
    if not st.failure_profile.control_plane_enabled:
        raise HTTPException(status_code=403, detail={"error_code": "forbidden"})


def _counts(st: MockXDRState) -> dict[str, int]:
    counts = {"incident": 0, "alert": 0, "asset": 0, "log": 0, "connector": len(st.connectors)}
    for (kind, _), obj in st.objects.items():
        if not obj.deleted and kind in counts:
            counts[kind] += 1
    return counts


def _maybe_fault(st: MockXDRState) -> None:
    st.request_counter += 1
    n = st.request_counter
    profile = st.failure_profile
    if profile.rate_limit_every_n and n % profile.rate_limit_every_n == 0:
        raise HTTPException(
            status_code=429,
            detail={"error_code": "rate_limited", "error_message": "mock 429", "details": {}},
        )
    if profile.server_error_every_n and n % profile.server_error_every_n == 0:
        raise HTTPException(
            status_code=500,
            detail={"error_code": "remote_error", "error_message": "mock 500", "details": {}},
        )
    if profile.timeout_every_n and n % profile.timeout_every_n == 0:
        raise HTTPException(
            status_code=504,
            detail={"error_code": "timeout", "error_message": "mock timeout", "details": {}},
        )
    if profile.malformed_payload_every_n and n % profile.malformed_payload_every_n == 0:
        raise MockMalformedPayloadError


# Module-level app for ``uvicorn app.mock_xdr.api:app``
app = create_app()
