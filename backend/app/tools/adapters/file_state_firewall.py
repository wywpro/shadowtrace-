"""Local JSON firewall example proving ToolProvider replacement mechanics."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import ipaddress
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import orjson
from pydantic import ValidationError

from app.models.enums import CapabilityState, DispositionIntentKind, TargetExecutionStatus
from app.models.execution import TargetExecutionResult
from app.models.ids import new_call_id
from app.models.tool_meta import (
    CapabilityBindingEntry,
    CapabilityManifest,
    ExecutionChannel,
    ToolResult,
    ToolResultStatus,
)
from app.tools.adapters.base import AdapterConfig, BaseToolAdapter
from app.tools.inputs import BlockIpInput
from app.tools.specs import baseline_tool_index

_STATE_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


class FileStateFirewallAdapter(BaseToolAdapter):
    """Simulated ``block_ip`` adapter backed by an atomic local state file.

    This is intentionally not a real firewall client. Its Provider identity and
    every result explicitly carry ``simulated=true``.
    """

    name = "file_state_firewall"
    tool_meta = baseline_tool_index()["block_ip"].model_copy(deep=True)
    simulated = True

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._path = self._path_from_endpoint(config.endpoint)

    @staticmethod
    def _path_from_endpoint(endpoint: str) -> Path:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"", "file"}:
            raise ValueError(
                "FileStateFirewallAdapter endpoint must be a local path or file:// URL"
            )
        if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
            raise ValueError("remote file:// hosts are not supported")
        raw_path = unquote(parsed.path) if parsed.scheme == "file" else endpoint
        if not raw_path:
            raise ValueError("state file path is required")
        return Path(raw_path).expanduser().resolve()

    def capability_manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            provider_name=self.name,
            online=True,
            entity_response=CapabilityState.SUPPORTED,
            allowed_intents=[DispositionIntentKind.EXECUTION_RESULT_RECORD],
            allowed_operations=[self.tool_meta.tool_name],
            allowed_target_types=["ip"],
            supports_status_query=True,
            supports_lookup_by_idempotency=True,
            supports_idempotency=True,
            supports_concurrency_control=True,
            supports_fencing=False,
            allowed_execution_channels=[ExecutionChannel.TOOL_PROVIDER],
            bindings=[
                CapabilityBindingEntry(
                    intent_kind=DispositionIntentKind.EXECUTION_RESULT_RECORD,
                    operation_code=self.tool_meta.tool_name,
                    state=CapabilityState.SUPPORTED,
                )
            ],
        )

    def validate_config(self) -> bool:
        return (
            self.config.enabled
            and self.config.auth_type == "none"
            and not self.config.credential_ref
            and bool(self.config.endpoint.strip())
        )

    async def health_check(self) -> bool:
        return await asyncio.to_thread(self._health_check_sync)

    def _health_check_sync(self) -> bool:
        if not self.validate_config():
            return False
        parent = self._path.parent
        if not parent.is_dir() or not os.access(parent, os.R_OK | os.W_OK):
            return False
        if not self._path.exists():
            return True
        try:
            self._read_state()
        except (OSError, ValueError):
            return False
        return os.access(self._path, os.R_OK | os.W_OK)

    async def execute(
        self,
        params: dict[str, Any],
        idempotency_key: str,
    ) -> ToolResult:
        try:
            parsed = BlockIpInput.model_validate(params)
            ipaddress.ip_address(parsed.target)
        except (ValidationError, ValueError):
            return self._result(
                status=ToolResultStatus.VALIDATION_ERROR,
                provider_code="invalid_block_ip_parameters",
                error_detail="block_ip parameters failed validation",
            )
        if not idempotency_key:
            return self._result(
                status=ToolResultStatus.VALIDATION_ERROR,
                provider_code="idempotency_key_required",
                error_detail="idempotency_key is required",
            )
        try:
            return await asyncio.to_thread(
                self._execute_sync,
                parsed.target,
                params,
                idempotency_key,
            )
        except (OSError, ValueError) as exc:
            return self._result(
                status=ToolResultStatus.REMOTE_ERROR,
                provider_code="file_state_unavailable",
                error_detail=f"local state update failed: {type(exc).__name__}",
            )

    async def lookup_by_idempotency(self, idempotency_key: str) -> ToolResult | None:
        if not self.capability_manifest().supports_lookup_by_idempotency:
            return await super().lookup_by_idempotency(idempotency_key)
        try:
            return await asyncio.to_thread(self._lookup_sync, idempotency_key)
        except (OSError, ValueError):
            return self._result(
                status=ToolResultStatus.REMOTE_ERROR,
                provider_code="file_state_unavailable",
                error_detail="local state lookup failed",
            )

    async def get_job_status(self, provider_job_id: str) -> ToolResult:
        if not self.capability_manifest().supports_status_query:
            return await super().get_job_status(provider_job_id)
        try:
            return await asyncio.to_thread(self._status_sync, provider_job_id)
        except (OSError, ValueError):
            return self._result(
                status=ToolResultStatus.REMOTE_ERROR,
                provider_code="file_state_unavailable",
                error_detail="local state status lookup failed",
            )

    def _execute_sync(
        self,
        target: str,
        params: dict[str, Any],
        idempotency_key: str,
    ) -> ToolResult:
        idem_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        params_hash = hashlib.sha256(_canonical_bytes(params)).hexdigest()
        provider_job_id = f"file-job-{idem_hash[:24]}"
        lock_path = self._path.with_name(f"{self._path.name}.lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            state = self._read_state()
            existing = state["idempotency"].get(idem_hash)
            if existing is not None:
                if existing["parameters_sha256"] != params_hash:
                    return self._result(
                        status=ToolResultStatus.VALIDATION_ERROR,
                        provider_code="idempotency_key_reuse",
                        error_detail="idempotency key was already used with different parameters",
                    )
                return self._success_result(
                    target=existing["target"],
                    provider_job_id=existing["provider_job_id"],
                    replayed=True,
                )

            now = datetime.now(UTC).isoformat()
            state["blocked_ips"][target] = {
                "provider_job_id": provider_job_id,
                "updated_at": now,
            }
            state["idempotency"][idem_hash] = {
                "parameters_sha256": params_hash,
                "provider_job_id": provider_job_id,
                "target": target,
                "completed_at": now,
            }
            self._write_state(state)
            return self._success_result(
                target=target,
                provider_job_id=provider_job_id,
                replayed=False,
            )

    def _lookup_sync(self, idempotency_key: str) -> ToolResult | None:
        state = self._read_state()
        idem_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        existing = state["idempotency"].get(idem_hash)
        if existing is None:
            return None
        return self._success_result(
            target=existing["target"],
            provider_job_id=existing["provider_job_id"],
            replayed=True,
        )

    def _status_sync(self, provider_job_id: str) -> ToolResult:
        state = self._read_state()
        existing = next(
            (
                item
                for item in state["idempotency"].values()
                if item["provider_job_id"] == provider_job_id
            ),
            None,
        )
        if existing is None:
            return self._result(
                status=ToolResultStatus.UNKNOWN,
                provider_code="provider_job_not_found",
                error_detail="provider job was not found",
            )
        return self._success_result(
            target=existing["target"],
            provider_job_id=provider_job_id,
            replayed=True,
        )

    def _read_state(self) -> dict[str, Any]:
        if not self._path.exists():
            return {
                "version": _STATE_VERSION,
                "blocked_ips": {},
                "idempotency": {},
            }
        raw = orjson.loads(self._path.read_bytes())
        if (
            not isinstance(raw, dict)
            or raw.get("version") != _STATE_VERSION
            or not isinstance(raw.get("blocked_ips"), dict)
            or not isinstance(raw.get("idempotency"), dict)
        ):
            raise ValueError("invalid file-state firewall document")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("provider_job_id"), str)
            or not isinstance(item.get("updated_at"), str)
            for item in raw["blocked_ips"].values()
        ):
            raise ValueError("invalid blocked_ips state")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("parameters_sha256"), str)
            or not isinstance(item.get("provider_job_id"), str)
            or not isinstance(item.get("target"), str)
            or not isinstance(item.get("completed_at"), str)
            for item in raw["idempotency"].values()
        ):
            raise ValueError("invalid idempotency state")
        return raw

    def _write_state(self, state: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=False, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{self._path.name}.",
                dir=self._path.parent,
                delete=False,
            ) as temp_file:
                temp_name = temp_file.name
                temp_file.write(_canonical_bytes(state))
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self._path)
            temp_name = None
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)

    def _success_result(
        self,
        *,
        target: str,
        provider_job_id: str,
        replayed: bool,
    ) -> ToolResult:
        return ToolResult(
            call_id=new_call_id(),
            tool_name=self.tool_meta.tool_name,
            provider_name=self.name,
            status=ToolResultStatus.SUCCESS,
            provider_job_id=provider_job_id,
            data={
                "target": target,
                "state": "blocked",
                "idempotent_replay": replayed,
                "simulated": True,
            },
            target_results=[
                TargetExecutionResult(
                    canonical_target=target,
                    status=TargetExecutionStatus.SUCCESS,
                    code="file_state_blocked",
                    message="local example state updated",
                )
            ],
            provider_code="file_state_blocked",
            raw_result={"simulated": True},
        )

    def _result(
        self,
        *,
        status: ToolResultStatus,
        provider_code: str,
        error_detail: str,
    ) -> ToolResult:
        return ToolResult(
            call_id=new_call_id(),
            tool_name=self.tool_meta.tool_name,
            provider_name=self.name,
            status=status,
            data={"simulated": True},
            provider_code=provider_code,
            error_detail=error_detail,
            raw_result={"simulated": True},
        )


ADAPTER_CLASS = FileStateFirewallAdapter


__all__ = ["ADAPTER_CLASS", "FileStateFirewallAdapter"]
