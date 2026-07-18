"""Generic, vendor-neutral candidate HTTP DispositionAdapter profile.

This module defines no vendor URL, operation mapping, authentication detail, or
proven write capability. Every capability defaults to UNKNOWN and submission is
blocked unless an explicit, tested profile enables it.
"""

from __future__ import annotations

import hashlib
import os
from urllib.parse import quote

import httpx
import orjson

from app.adapters._util import sanitize_disposition_receipt
from app.adapters.disposition.base import (
    BaseDispositionAdapter,
    DispositionAdapterCapabilities,
)
from app.core.errors import (
    DependencyUnavailableError,
    WritebackConflictError,
    WritebackUnsupportedError,
)
from app.core.errors import (
    ValidationError as ShadowTraceValidationError,
)
from app.mock_xdr.state import find_forbidden_analysis_keys
from app.models.disposition import DispositionCommand, DispositionReceipt, SourceObjectLocator
from app.models.enums import (
    CapabilityState,
    ConnectorStatus,
    DispositionIntentKind,
    WritebackStatus,
)
from app.tools.adapters.base import AdapterConfig

_INTERNAL_OPERATION_CODES = frozenset(
    {
        "set_event_disposition",
        "submit_entity_action",
        "record_execution_result",
        "record_compensation",
    }
)


def candidate_disposition_capabilities() -> DispositionAdapterCapabilities:
    """Return fail-closed candidate capabilities; no live fact is implied."""

    unknown = CapabilityState.UNKNOWN
    return DispositionAdapterCapabilities(
        intents={intent: unknown for intent in DispositionIntentKind},
        operations={operation: unknown for operation in _INTERNAL_OPERATION_CODES},
        supports_idempotency=False,
        supports_status_query=False,
        supports_concurrency_token=False,
        supports_lookup_by_idempotency=False,
    )


class HttpDispositionAdapter(BaseDispositionAdapter):
    """Configurable candidate profile operating on exact caller-supplied URLs."""

    name = "generic_http_disposition"

    def __init__(
        self,
        config: AdapterConfig,
        *,
        capabilities: DispositionAdapterCapabilities | None = None,
        status_endpoint_template: str | None = None,
        idempotency_lookup_endpoint: str | None = None,
        health_endpoint: str | None = None,
        source_credential_ref: str | None = None,
        shared_credential_scope_verified: bool = False,
        allow_side_effects: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._capabilities = capabilities or candidate_disposition_capabilities()
        self._status_endpoint_template = status_endpoint_template
        self._idempotency_lookup_endpoint = idempotency_lookup_endpoint
        self._health_endpoint = health_endpoint
        self._source_credential_ref = source_credential_ref
        self._shared_credential_scope_verified = shared_credential_scope_verified
        self._allow_side_effects = allow_side_effects
        self._client = client
        self._owns_client = client is None

    def capabilities(self) -> DispositionAdapterCapabilities:
        return self._capabilities.model_copy(deep=True)

    def validate_config(self) -> bool:
        if not self.config.enabled or not self.config.endpoint.startswith(("http://", "https://")):
            return False
        if self.config.auth_type == "none":
            if self.config.credential_ref:
                return False
        elif not self.config.credential_ref or self.config.credential_ref not in os.environ:
            return False

        source_ref = self._source_credential_ref
        write_ref = self.config.credential_ref
        if source_ref and write_ref and not self._shared_credential_scope_verified:
            if source_ref == write_ref:
                return False
            source_value = os.environ.get(source_ref)
            write_value = os.environ.get(write_ref)
            if source_value and write_value and source_value == write_value:
                return False
        return True

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_s,
                verify=self.config.tls_verify,
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        if self.config.auth_type == "none":
            return {}
        credential = os.environ.get(self.config.credential_ref)
        if not credential:
            raise WritebackUnsupportedError(
                "disposition credential reference is unavailable",
                details={"credential_ref": self.config.credential_ref},
            )
        scheme = "Bearer" if self.config.auth_type == "bearer" else "Basic"
        return {"Authorization": f"{scheme} {credential}"}

    def validate_command(self, command: DispositionCommand) -> None:
        DispositionCommand.model_validate(command.model_dump(mode="json"))
        if not self.validate_config():
            raise WritebackUnsupportedError("generic HTTP disposition profile is not configured")
        if not self._allow_side_effects:
            raise WritebackUnsupportedError(
                "live disposition side effects are disabled",
                details={"allow_side_effects": False},
            )
        forbidden = find_forbidden_analysis_keys(command.model_dump(mode="json"))
        if forbidden:
            raise ShadowTraceValidationError(
                "analysis fields are forbidden on disposition commands",
                error_code="unauthorized_field",
                details={"paths": forbidden},
            )
        capabilities = self.capabilities()
        if (
            capabilities.intents.get(
                command.intent_kind,
                CapabilityState.UNKNOWN,
            )
            is not CapabilityState.SUPPORTED
        ):
            raise WritebackUnsupportedError(
                "disposition intent capability is not verified",
                details={"intent_kind": command.intent_kind.value},
            )
        if (
            capabilities.operations.get(
                command.operation_code,
                CapabilityState.UNKNOWN,
            )
            is not CapabilityState.SUPPORTED
        ):
            raise WritebackUnsupportedError(
                "disposition operation capability is not verified",
                details={"operation_code": command.operation_code},
            )

    async def submit(self, command: DispositionCommand) -> DispositionReceipt:
        self.validate_command(command)
        client = await self._http()
        try:
            response = await client.post(
                self.config.endpoint,
                headers={
                    **self._auth_headers(),
                    "Idempotency-Key": command.idempotency_key,
                },
                json=command.model_dump(mode="json"),
            )
        except (httpx.TimeoutException, httpx.TransportError):
            return await self._recover_or_unknown(command)

        if response.status_code == 409:
            raise WritebackConflictError(
                "candidate HTTP profile reported a conflict",
                error_code="version_conflict",
                details={"status": 409},
            )
        if response.status_code == 401:
            raise ShadowTraceValidationError(
                "candidate HTTP profile authentication failed",
                error_code="auth_error",
                details={"status": 401},
            )
        if response.status_code == 403:
            raise ShadowTraceValidationError(
                "candidate HTTP profile denied the operation",
                error_code="permission_denied",
                details={"status": 403},
            )
        if response.status_code == 429:
            raise DependencyUnavailableError(
                "candidate HTTP profile is rate limited",
                error_code="rate_limited",
                details={"status": 429},
            )
        if response.status_code >= 500:
            return await self._recover_or_unknown(command)
        if response.status_code >= 400:
            raise ShadowTraceValidationError(
                "candidate HTTP profile rejected the operation",
                error_code="invalid_operation",
                details={"status": response.status_code},
            )
        try:
            receipt = DispositionReceipt.model_validate(response.json())
        except ValueError:
            return await self._recover_or_unknown(command)
        return sanitize_disposition_receipt(receipt)

    async def get_status(self, provider_job_id: str) -> DispositionReceipt | None:
        capabilities = self.capabilities()
        if not capabilities.supports_status_query or self._status_endpoint_template is None:
            return None
        try:
            endpoint = self._status_endpoint_template.format(
                provider_job_id=quote(provider_job_id, safe=""),
            )
        except (KeyError, ValueError):
            return None
        client = await self._http()
        try:
            response = await client.get(endpoint, headers=self._auth_headers())
        except (httpx.HTTPError, WritebackUnsupportedError):
            return None
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            return None
        try:
            return sanitize_disposition_receipt(DispositionReceipt.model_validate(response.json()))
        except ValueError:
            return None

    async def lookup_submission(
        self,
        idempotency_key: str,
        source_locator: SourceObjectLocator,
    ) -> DispositionReceipt | None:
        capabilities = self.capabilities()
        if (
            not capabilities.supports_lookup_by_idempotency
            or self._idempotency_lookup_endpoint is None
        ):
            return None
        locator_bytes = orjson.dumps(
            source_locator.model_dump(mode="json"),
            option=orjson.OPT_SORT_KEYS,
        )
        client = await self._http()
        try:
            response = await client.get(
                self._idempotency_lookup_endpoint,
                headers=self._auth_headers(),
                params={
                    "idempotency_key_sha256": hashlib.sha256(idempotency_key.encode()).hexdigest(),
                    "source_locator_sha256": hashlib.sha256(locator_bytes).hexdigest(),
                },
            )
        except (httpx.HTTPError, WritebackUnsupportedError):
            return None
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            return None
        try:
            return sanitize_disposition_receipt(DispositionReceipt.model_validate(response.json()))
        except ValueError:
            return None

    async def health_check(self) -> ConnectorStatus:
        if not self.validate_config() or self._health_endpoint is None:
            return ConnectorStatus.UNKNOWN
        client = await self._http()
        try:
            response = await client.get(
                self._health_endpoint,
                headers=self._auth_headers(),
            )
        except (httpx.HTTPError, WritebackUnsupportedError):
            return ConnectorStatus.OFFLINE
        return (
            ConnectorStatus.ONLINE if 200 <= response.status_code < 300 else ConnectorStatus.OFFLINE
        )

    async def _recover_or_unknown(
        self,
        command: DispositionCommand,
    ) -> DispositionReceipt:
        if self.capabilities().supports_lookup_by_idempotency:
            found = await self.lookup_submission(
                command.idempotency_key,
                command.source_locator,
            )
            if found is not None:
                return found
        return DispositionReceipt(
            writeback_id=(
                "wbk-unknown-"
                + hashlib.sha256(
                    f"{command.disposition_id}|{command.idempotency_key}".encode()
                ).hexdigest()[:16]
            ),
            sequence=1,
            disposition_id=command.disposition_id,
            action_id=command.action_id,
            source_record_id=command.source_locator.source_object_id,
            status=WritebackStatus.UNKNOWN,
            provider_code="unknown_delivery",
            provider_message="delivery could not be confirmed; manual verification required",
            simulated=False,
        )


__all__ = [
    "HttpDispositionAdapter",
    "candidate_disposition_capabilities",
]
