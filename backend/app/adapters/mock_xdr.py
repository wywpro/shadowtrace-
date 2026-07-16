"""Mock XDR Source + Disposition adapters (ISSUE-012).

These wrap the Mock HTTP contract under ``/mock-xdr/v1``. Relationships and
error codes are ShadowTrace Mock facts — not 深信服 backend facts.
"""

from __future__ import annotations

import asyncio
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
    sanitize_disposition_receipt,
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
from app.models.source import SourceConnector

logger = logging.getLogger(__name__)

_ALLOWED_OPERATIONS = frozenset(
    {
        "set_event_disposition",
        "submit_entity_action",
        "record_execution_result",
        "record_compensation",
    }
)


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
            if resp.status_code == 504:
                if attempt >= self._max_retries:
                    raise DependencyUnavailableError(
                        "mock source timeout",
                        error_code="timeout",
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
        connector_id: str | None = None,
        cursor: str | None = None,
        updated_after: datetime | None = None,
        limit: int = 100,
    ) -> SourcePage:
        if len(object_types) != 1:
            raise ValueError("SourceAdapter.list_objects requires exactly one object kind")
        raw_kind = object_types[0]
        kind = (
            raw_kind if isinstance(raw_kind, SourceObjectKind) else SourceObjectKind(str(raw_kind))
        )
        items: list[Any] = []
        schema_version = "1"
        server_time = datetime.now(UTC)
        path = f"/mock-xdr/v1/{kind_to_path(kind.value)}"
        params: dict[str, Any] = {"page_size": limit, "commit": False}
        if connector_id is not None:
            params["connector_id"] = connector_id
        if cursor is not None:
            params["cursor"] = cursor
        if updated_after is not None:
            params["updated_after"] = updated_after.isoformat()

        payload = await self._get_json(path, params=params)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            self._quality.record(
                stage="source_list",
                error_category="malformed_payload",
                detail={"kind": kind.value},
            )
            return SourcePage(
                object_kind=kind,
                connector_id=connector_id,
                schema_version=schema_version,
                server_time=server_time,
                malformed_items=1,
            )

        halted = False
        malformed_items = 0
        for body in payload["items"]:
            if not isinstance(body, dict):
                malformed_items += 1
                self._quality.record(
                    stage="source_list",
                    error_category="malformed_payload",
                    detail={"kind": kind.value, "reason": "item_not_object"},
                )
                continue
            sv = "1"
            mock_meta = body.get("_mock")
            if isinstance(mock_meta, dict) and mock_meta.get("schema_version"):
                sv = str(mock_meta["schema_version"])
            schema_version = sv
            if sv not in self._supported_schema_versions:
                self._quality.record(
                    stage="source_list",
                    error_category="schema_unsupported",
                    detail={"kind": kind.value, "schema_version": sv},
                )
                halted = True
                break
            parsed = parse_source_item(kind.value, body, quality=self._quality)
            if parsed is not None:
                items.append(parsed)
            else:
                malformed_items += 1
        next_cursor = None if halted else payload.get("next_cursor")
        return SourcePage(
            items=items,
            object_kind=kind,
            connector_id=connector_id,
            next_cursor=str(next_cursor) if next_cursor is not None else None,
            has_more=next_cursor is not None,
            server_time=server_time,
            schema_version=schema_version,
            malformed_items=malformed_items,
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
            self._quality.record(
                stage="source_get",
                error_category="malformed_payload",
                detail={"kind": kind},
            )
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
        return ConnectorStatus.ONLINE

    async def list_connectors(self) -> list[SourceConnector]:
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

        try:
            receipt = DispositionReceipt.model_validate(resp.json())
        except ValueError:
            logger.warning("disposition submit returned malformed payload")
            return await self._unknown_after_loss(command)
        return sanitize_disposition_receipt(receipt)

    async def _unknown_after_loss(self, command: DispositionCommand) -> DispositionReceipt:
        caps = self.capabilities()
        if caps.supports_lookup_by_idempotency:
            try:
                found = await self.lookup_submission(
                    command.idempotency_key,
                    command.source_locator,
                )
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning(
                    "disposition idempotency lookup inconclusive: %s",
                    type(exc).__name__,
                )
                found = None
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
        return sanitize_disposition_receipt(receipt)

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
