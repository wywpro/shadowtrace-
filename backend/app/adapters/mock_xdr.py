"""Mock XDR Source + Disposition adapters (ISSUE-012).

These wrap the Mock HTTP contract under ``/mock-xdr/v1``. Relationships and
error codes are ShadowTrace Mock facts — not 深信服 backend facts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from app.adapters._util import (
    kind_to_path,
    parse_connector,
    parse_source_item,
    require_separated_credentials,
    sanitize_raw_result,
)
from app.adapters.disposition.base import (
    BaseDispositionAdapter,
    DispositionAdapterCapabilities,
)
from app.adapters.source.base import (
    BaseSourceAdapter,
    DataQualityRecorder,
    InMemoryDataQualityRecorder,
    SourceEvidencePage,
    SourcePage,
)
from app.core.errors import (
    DependencyUnavailableError,
    WritebackConflictError,
    WritebackUnsupportedError,
)
from app.core.errors import (
    ValidationError as ShadowTraceValidationError,
)
from app.mock_xdr.state import find_forbidden_analysis_keys, idempotency_key_hash
from app.models.disposition import DispositionCommand, DispositionReceipt, SourceObjectLocator
from app.models.enums import (
    CapabilityState,
    ConnectorCapability,
    ConnectorStatus,
    DispositionIntentKind,
    SourceObjectKind,
    WritebackStatus,
)

logger = logging.getLogger(__name__)

_ALLOWED_OPERATIONS = frozenset(
    {
        "set_event_disposition",
        "submit_entity_action",
        "record_execution_result",
        "record_compensation",
    }
)


def _decode_cursor(cursor: str | None) -> dict[str, str]:
    """Decode a per-kind composite cursor ({kind: mock_cursor}).

    Mock cursors are bound to a single object kind, so a multi-kind ``list_objects``
    call must page each kind independently. ``None`` means "start fresh for all".
    """
    if not cursor:
        return {}
    try:
        data = json.loads(cursor)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


class MockXDRSourceAdapter(BaseSourceAdapter):
    """Read-only Mock XDR client. Uses the read credential only."""

    name = "mock_xdr"

    def __init__(
        self,
        *,
        base_url: str = "http://mock-xdr",
        read_token: str,
        write_token: str,
        client: httpx.AsyncClient | None = None,
        quality: DataQualityRecorder | None = None,
        supported_schema_versions: frozenset[str] | None = None,
        max_retries: int = 3,
    ) -> None:
        require_separated_credentials(read_token=read_token, write_token=write_token)
        self._base_url = base_url.rstrip("/")
        self._read_token = read_token
        # write_token retained only to enforce separation at construction; never sent.
        self._write_token = write_token
        self._client = client
        self._owns_client = client is None
        self._quality = quality or InMemoryDataQualityRecorder()
        self._supported_schema_versions = supported_schema_versions or frozenset({"1"})
        self._max_retries = max_retries
        self._degraded_kinds: set[str] = set()

    def capabilities(self) -> dict[ConnectorCapability, CapabilityState]:
        return {
            ConnectorCapability.LOG_INGESTION: CapabilityState.SUPPORTED,
            ConnectorCapability.QUERY: CapabilityState.SUPPORTED,
            ConnectorCapability.EVENT_DISPOSITION: CapabilityState.UNSUPPORTED,
            ConnectorCapability.ENTITY_RESPONSE: CapabilityState.UNSUPPORTED,
        }

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._read_token}"}

    async def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        client = await self._http()
        delay = 0.05
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.get(path, headers=self._auth_headers(), params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    raise DependencyUnavailableError(
                        "mock source transport failure",
                        error_code="remote_error",
                        details={"path": path},
                    ) from exc
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 429:
                if attempt >= self._max_retries:
                    raise DependencyUnavailableError(
                        "mock source rate limited",
                        error_code="rate_limited",
                        details={"path": path},
                    )
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if resp.status_code >= 500:
                if attempt >= self._max_retries:
                    raise DependencyUnavailableError(
                        "mock source remote error",
                        error_code="remote_error",
                        details={"status": resp.status_code, "path": path},
                    )
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if resp.status_code >= 400:
                body = resp.json() if resp.content else {}
                code = body.get("error_code", "adapter_validation_error")
                raise ShadowTraceValidationError(
                    body.get("error_message", f"HTTP {resp.status_code}"),
                    error_code=code,
                    details=body.get("details") or {"status": resp.status_code},
                )
            return resp.json()
        raise DependencyUnavailableError(
            "mock source exhausted retries",
            details={"path": path, "last": str(last_exc)},
        )

    async def list_objects(
        self,
        object_types: Sequence[SourceObjectKind | str],
        *,
        cursor: str | None = None,
        updated_after: datetime | None = None,
        limit: int = 100,
    ) -> SourcePage:
        items: list[Any] = []
        schema_version = "1"
        server_time = datetime.now(UTC)
        per_kind_cursor = _decode_cursor(cursor)
        paging_continuation = cursor is not None and bool(per_kind_cursor)
        # Per-kind continuation cursors for the NEXT page (composite, kind-scoped).
        next_cursors: dict[str, str] = {}

        for raw_kind in object_types:
            kind = raw_kind.value if isinstance(raw_kind, SourceObjectKind) else str(raw_kind)
            if kind in self._degraded_kinds:
                self._quality.record(
                    stage="source_list",
                    error_category="schema_unsupported",
                    detail={"kind": kind, "reason": "watermark_halted"},
                )
                continue
            # On a continuation call, a kind absent from the composite cursor is
            # already exhausted — skip it instead of restarting from page 1.
            if paging_continuation and kind not in per_kind_cursor:
                continue
            path = f"/mock-xdr/v1/{kind_to_path(kind)}"
            params: dict[str, Any] = {"page_size": limit}
            # Watermark commit is owned by the ingester after persist — never auto-commit.
            params["commit"] = False
            kind_cursor = per_kind_cursor.get(kind)
            if kind_cursor is not None:
                params["cursor"] = kind_cursor
            if updated_after is not None:
                params["updated_after"] = updated_after.isoformat()

            payload = await self._get_json(path, params=params)
            page_items = payload.get("items") or []
            kind_next = payload.get("next_cursor")
            halted = False
            for body in page_items:
                if not isinstance(body, dict):
                    continue
                sv = "1"
                mock_meta = body.get("_mock")
                if isinstance(mock_meta, dict) and mock_meta.get("schema_version"):
                    sv = str(mock_meta["schema_version"])
                schema_version = sv
                if sv not in self._supported_schema_versions:
                    self._degraded_kinds.add(kind)
                    self._quality.record(
                        stage="source_list",
                        error_category="schema_unsupported",
                        detail={"kind": kind, "schema_version": sv},
                    )
                    # Stop advancing this object type (no watermark commit by design).
                    halted = True
                    break
                parsed = parse_source_item(kind, body, quality=self._quality)
                if parsed is not None:
                    items.append(parsed)
            # Only advance this kind if it did not halt on an unsupported schema.
            if not halted and kind_next:
                next_cursors[kind] = kind_next

        next_cursor = json.dumps(next_cursors, sort_keys=True) if next_cursors else None
        return SourcePage(
            items=items,
            next_cursor=next_cursor,
            has_more=bool(next_cursors),
            server_time=server_time,
            schema_version=schema_version,
        )

    async def get_object(
        self,
        source_kind: SourceObjectKind | str,
        source_object_id: str,
    ) -> Any:
        kind = source_kind.value if isinstance(source_kind, SourceObjectKind) else str(source_kind)
        path = f"/mock-xdr/v1/{kind}/{source_object_id}"
        try:
            body = await self._get_json(path)
        except ShadowTraceValidationError as exc:
            if exc.error_code == "not_found":
                return None
            raise
        if not isinstance(body, dict):
            return None
        return parse_source_item(kind, body, quality=self._quality)

    async def list_evidence_records(
        self,
        *,
        updated_after: datetime | None = None,
    ) -> SourceEvidencePage | None:
        params = {"updated_after": updated_after.isoformat()} if updated_after is not None else None
        payload = await self._get_json("/mock-xdr/v1/evidence", params=params)
        if not isinstance(payload, dict):
            return None
        page = SourceEvidencePage.model_validate(payload)
        return page if any(page.records_by_source.values()) else None

    async def health_check(self) -> ConnectorStatus:
        try:
            await self._get_json("/mock-xdr/v1/health")
        except Exception:  # noqa: BLE001 — health must never raise to callers
            return ConnectorStatus.OFFLINE
        if self._degraded_kinds:
            return ConnectorStatus.DEGRADED
        return ConnectorStatus.ONLINE

    async def list_connectors(self) -> list[Any]:
        payload = await self._get_json("/mock-xdr/v1/connectors")
        return [parse_connector(item) for item in payload.get("items") or []]


class MockXDRDispositionAdapter(BaseDispositionAdapter):
    """Write-only Mock disposition client. Uses the write credential only."""

    name = "mock_xdr"

    def __init__(
        self,
        *,
        base_url: str = "http://mock-xdr",
        read_token: str,
        write_token: str,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
    ) -> None:
        require_separated_credentials(read_token=read_token, write_token=write_token)
        self._base_url = base_url.rstrip("/")
        self._read_token = read_token
        self._write_token = write_token
        self._client = client
        self._owns_client = client is None
        self._max_retries = max_retries

    def capabilities(self) -> DispositionAdapterCapabilities:
        supported = CapabilityState.SUPPORTED
        return DispositionAdapterCapabilities(
            intents={
                DispositionIntentKind.EVENT_STATUS_UPDATE: supported,
                DispositionIntentKind.ENTITY_ACTION_SUBMIT: supported,
                DispositionIntentKind.EXECUTION_RESULT_RECORD: supported,
                DispositionIntentKind.COMPENSATION_RECORD: supported,
            },
            operations={op: supported for op in _ALLOWED_OPERATIONS},
            supports_idempotency=True,
            supports_status_query=True,
            supports_concurrency_token=True,
            supports_lookup_by_idempotency=True,
        )

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._write_token}"}

    def validate_command(self, command: DispositionCommand) -> None:
        # Re-validate extra=forbid envelope; reject analysis/report keys.
        DispositionCommand.model_validate(command.model_dump(mode="json"))
        forbidden = find_forbidden_analysis_keys(command.model_dump(mode="json"))
        if forbidden:
            raise ShadowTraceValidationError(
                "analysis fields forbidden on disposition command",
                error_code="unauthorized_field",
                details={"paths": forbidden},
            )
        caps = self.capabilities()
        intent_state = caps.intents.get(command.intent_kind, CapabilityState.UNKNOWN)
        if intent_state is CapabilityState.UNSUPPORTED:
            raise WritebackUnsupportedError(
                f"intent {command.intent_kind.value} unsupported",
                details={"intent_kind": command.intent_kind.value},
            )
        if intent_state is CapabilityState.UNKNOWN:
            raise WritebackUnsupportedError(
                f"intent {command.intent_kind.value} capability UNKNOWN",
                details={"intent_kind": command.intent_kind.value},
            )
        op_state = caps.operations.get(command.operation_code, CapabilityState.UNKNOWN)
        if op_state is not CapabilityState.SUPPORTED:
            raise WritebackUnsupportedError(
                f"operation {command.operation_code} not supported",
                details={"operation_code": command.operation_code},
            )
        if command.operation_code not in _ALLOWED_OPERATIONS:
            raise ShadowTraceValidationError(
                f"operation_code {command.operation_code!r} not allowlisted",
                error_code="adapter_validation_error",
            )

    async def submit(self, command: DispositionCommand) -> DispositionReceipt:
        self.validate_command(command)
        client = await self._http()
        payload = command.model_dump(mode="json")
        try:
            resp = await client.post(
                "/mock-xdr/v1/dispositions",
                headers=self._auth_headers(),
                json=payload,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Lost response: do NOT invent FAILED; mark UNKNOWN and verify via idempotency.
            logger.warning("disposition submit transport lost: %s", type(exc).__name__)
            return await self._unknown_after_loss(command)

        if resp.status_code == 409:
            body = resp.json() if resp.content else {}
            raise WritebackConflictError(
                body.get("error_message", "version conflict"),
                error_code=body.get("error_code", "version_conflict"),
                details=body.get("details") or {},
            )
        if resp.status_code >= 500 or resp.status_code == 504:
            return await self._unknown_after_loss(command)
        if resp.status_code >= 400:
            body = resp.json() if resp.content else {}
            code = body.get("error_code", "adapter_validation_error")
            raise ShadowTraceValidationError(
                body.get("error_message", f"HTTP {resp.status_code}"),
                error_code=code,
                details=body.get("details") or {},
            )

        receipt = DispositionReceipt.model_validate(resp.json())
        return receipt.model_copy(
            update={"raw_result": sanitize_raw_result(dict(receipt.raw_result))}
        )

    async def _unknown_after_loss(self, command: DispositionCommand) -> DispositionReceipt:
        caps = self.capabilities()
        if caps.supports_lookup_by_idempotency:
            found = await self.lookup_submission(command.idempotency_key, command.source_locator)
            if found is not None:
                return found
        # No automatic re-execution of entity actions — UNKNOWN + manual/verify path.
        now = datetime.now(UTC)
        return DispositionReceipt(
            writeback_id=f"wbk-unknown-{command.disposition_id}",
            sequence=1,
            disposition_id=command.disposition_id,
            action_id=command.action_id,
            source_record_id=command.source_locator.source_object_id,
            status=WritebackStatus.UNKNOWN,
            provider_code="unknown_delivery",
            provider_message="response lost; verify via idempotency before retry",
            submitted_at=now,
            observed_at=now,
            raw_result=sanitize_raw_result({"lost_response": True}),
            simulated=True,
        )

    async def get_status(self, provider_job_id: str) -> DispositionReceipt | None:
        if not self.capabilities().supports_status_query:
            return None
        client = await self._http()
        resp = await client.get(
            f"/mock-xdr/v1/disposition-jobs/{provider_job_id}",
            headers=self._auth_headers(),
        )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            return None
        body = resp.json()
        # Job status domain is ExecutionJobStatus — not WritebackStatus.
        terminal = body.get("terminal_writeback_status")
        status = WritebackStatus(terminal) if terminal else WritebackStatus.ACCEPTED
        return DispositionReceipt(
            writeback_id=body.get("writeback_id") or f"wbk-job-{provider_job_id}",
            sequence=1,
            disposition_id=body.get("disposition_id") or "",
            action_id="",
            source_record_id="",
            status=status,
            provider_job_id=provider_job_id,
            provider_code=body.get("status"),
            raw_result=sanitize_raw_result(body),
            simulated=True,
        )

    async def lookup_submission(
        self,
        idempotency_key: str,
        source_locator: SourceObjectLocator,
    ) -> DispositionReceipt | None:
        if not self.capabilities().supports_lookup_by_idempotency:
            return None
        _ = source_locator  # Mock lookup is by key hash; locator is for live adapters.
        key_hash = idempotency_key_hash(idempotency_key)
        client = await self._http()
        resp = await client.get(
            f"/mock-xdr/v1/dispositions/by-idempotency/{key_hash}",
            headers=self._auth_headers(),
        )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            return None
        receipt = DispositionReceipt.model_validate(resp.json())
        return receipt.model_copy(
            update={"raw_result": sanitize_raw_result(dict(receipt.raw_result))}
        )

    async def health_check(self) -> ConnectorStatus:
        # Health is a read endpoint; use write token (also grants read on Mock).
        client = await self._http()
        try:
            resp = await client.get(
                "/mock-xdr/v1/health",
                headers=self._auth_headers(),
            )
        except httpx.HTTPError:
            return ConnectorStatus.OFFLINE
        if resp.status_code >= 400:
            return ConnectorStatus.OFFLINE
        return ConnectorStatus.ONLINE


class LiveDispositionAdapterStub(BaseDispositionAdapter):
    """Live stub: every capability stays UNKNOWN until evidenced."""

    name = "live_stub"

    def capabilities(self) -> DispositionAdapterCapabilities:
        unknown = CapabilityState.UNKNOWN
        return DispositionAdapterCapabilities(
            intents={intent: unknown for intent in DispositionIntentKind},
            operations={op: unknown for op in _ALLOWED_OPERATIONS},
            supports_idempotency=False,
            supports_status_query=False,
            supports_concurrency_token=False,
            supports_lookup_by_idempotency=False,
        )

    def validate_command(self, command: DispositionCommand) -> None:
        raise WritebackUnsupportedError(
            "live disposition adapter capabilities are UNKNOWN",
            details={"intent_kind": command.intent_kind.value},
        )

    async def submit(self, command: DispositionCommand) -> DispositionReceipt:
        self.validate_command(command)
        raise AssertionError("unreachable")

    async def health_check(self) -> ConnectorStatus:
        return ConnectorStatus.UNKNOWN
