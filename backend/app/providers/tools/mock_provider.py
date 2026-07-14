"""Deterministic asynchronous ToolProvider used by mock and test environments."""

from __future__ import annotations

import contextvars
import hashlib
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.adapters._util import sanitize_raw_result
from app.core.config import get_settings
from app.models.disposition import DispositionReceipt
from app.models.enums import (
    CapabilityState,
    ConfirmationEvidence,
    DispositionIntentKind,
    ExecutionJobStatus,
    ExecutionOwner,
    TargetExecutionStatus,
    TargetWritebackStatus,
    ToolCategory,
    WritebackStatus,
)
from app.models.execution import ActionExecutionJob, TargetExecutionResult
from app.models.ids import new_call_id, new_event_id, new_job_id
from app.models.tool_meta import (
    TERMINAL_DISPOSITION_TOOL,
    CapabilityBindingEntry,
    CapabilityManifest,
    ExecutionChannel,
    ProviderToolBinding,
    ToolResult,
    ToolResultStatus,
)
from app.models.workflow import validate_job_status_transition
from app.tools.inputs import TOOL_INPUT_MODELS
from app.tools.mock_state import (
    MOCK_STATE_NAMESPACES,
    MockEnvironmentState,
    MockObservationRecord,
    MockStateRecord,
)
from app.tools.specs import baseline_tool_index

PROVIDER_NAME = "mock_tool_provider"
XDR_PROVIDER_NAME = "mock_xdr"
DEFAULT_CONNECTOR = "mock-tool-connector"

_TOOL_STATE: dict[str, tuple[str, str]] = {
    "block_ip": ("blocked_ips", "blocked"),
    "block_domain": ("blocked_domains", "blocked"),
    "isolate_host": ("isolated_hosts", "isolated"),
    "quarantine_file": ("quarantined_files", "quarantined"),
    "block_process": ("blocked_processes", "blocked"),
    "scan_host_for_virus": ("scan_results", "completed"),
    "disable_account": ("accounts", "disabled"),
    "force_logout": ("sessions", "terminated"),
    "reset_password": ("accounts", "password_reset"),
    "revoke_token": ("tokens", "revoked"),
}
_TOOL_OBSERVATION_SURFACE: dict[str, str] = {
    "block_ip": "ip_blocks",
    "block_domain": "domain_blocks",
    "isolate_host": "host_isolation",
    "quarantine_file": "file_quarantine",
    "block_process": "process_blocks",
    "scan_host_for_virus": "virus_scans",
    "disable_account": "account_status",
    "force_logout": "account_status",
    "reset_password": "account_status",
    "revoke_token": "account_status",
}
_REVERSED_OBSERVATION_STATUS = {
    "blocked": "allowed",
    "isolated": "connected",
    "quarantined": "present",
    "completed": "pending",
    "disabled": "enabled",
    "terminated": "active",
    "password_reset": "unchanged",
    "revoked": "active",
    "dropped": "flowing",
    "detected": "none",
}
_SYNC_TOOLS = frozenset({"create_ticket", "notify_security_team"})
_TERMINAL_JOB_STATUSES = frozenset(
    {
        ExecutionJobStatus.PARTIAL_SUCCESS,
        ExecutionJobStatus.SUCCESS,
        ExecutionJobStatus.FAILED,
        ExecutionJobStatus.TIMED_OUT,
        ExecutionJobStatus.CANCELLED,
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_json(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def map_disposition_receipt_to_job(
    receipt: DispositionReceipt,
    *,
    event_id: str,
    idempotency_key: str,
    job_id: str | None = None,
    action_id: str | None = None,
) -> ActionExecutionJob:
    """Map the XDR_MANAGED receipt path to the shared execution-job contract.

    This is deliberately a pure mapping: it never invokes ``MockToolProvider``
    and therefore cannot duplicate the entity action submitted by the adapter.
    """

    status_map = {
        WritebackStatus.PENDING: ExecutionJobStatus.QUEUED,
        WritebackStatus.SENDING: ExecutionJobStatus.RUNNING,
        WritebackStatus.ACCEPTED: ExecutionJobStatus.RUNNING,
        WritebackStatus.CONFIRMED: ExecutionJobStatus.SUCCESS,
        WritebackStatus.PARTIAL: ExecutionJobStatus.PARTIAL_SUCCESS,
        WritebackStatus.FAILED: ExecutionJobStatus.FAILED,
        WritebackStatus.CONFLICT: ExecutionJobStatus.FAILED,
        WritebackStatus.UNKNOWN: ExecutionJobStatus.UNKNOWN,
    }
    target_status_map = {
        TargetWritebackStatus.PENDING: TargetExecutionStatus.UNKNOWN,
        TargetWritebackStatus.ACCEPTED: TargetExecutionStatus.UNKNOWN,
        TargetWritebackStatus.CONFIRMED: TargetExecutionStatus.SUCCESS,
        TargetWritebackStatus.FAILED: TargetExecutionStatus.FAILED,
        TargetWritebackStatus.CONFLICT: TargetExecutionStatus.FAILED,
        TargetWritebackStatus.UNKNOWN: TargetExecutionStatus.UNKNOWN,
    }
    status = (
        ExecutionJobStatus.QUEUED
        if receipt.status is WritebackStatus.ACCEPTED and receipt.provider_job_id is not None
        else status_map[receipt.status]
    )
    stable_job_id = job_id or f"job-{hashlib.sha256(receipt.writeback_id.encode()).hexdigest()[:8]}"
    finished_at = (
        receipt.confirmed_at or receipt.observed_at if status in _TERMINAL_JOB_STATUSES else None
    )
    target_results = [
        TargetExecutionResult(
            canonical_target=item.canonical_target,
            status=target_status_map[item.status],
            code=item.provider_code,
            message=item.message_code,
            artifact_id=item.artifact_ref,
        )
        for item in receipt.target_results
    ]
    return ActionExecutionJob(
        job_id=stable_job_id,
        event_id=event_id,
        action_id=action_id or receipt.action_id,
        provider_name=XDR_PROVIDER_NAME,
        idempotency_key=idempotency_key,
        provider_job_id=receipt.provider_job_id,
        status=status,
        created_at=receipt.submitted_at,
        updated_at=receipt.observed_at,
        started_at=receipt.submitted_at if status is not ExecutionJobStatus.QUEUED else None,
        finished_at=finished_at,
        target_results=target_results,
        provider_code=receipt.provider_code,
        provider_message=receipt.provider_message,
        raw_result=sanitize_raw_result(
            {
                **receipt.raw_result,
                "simulated": receipt.simulated,
                "writeback_id": receipt.writeback_id,
                "confirmation_evidence": (
                    ConfirmationEvidence(receipt.confirmation_evidence).value
                    if receipt.confirmation_evidence is not None
                    else None
                ),
            }
        ),
    )


class ToolExecutionContext(BaseModel):
    """Execution envelope supplied by ToolExecutor without polluting tool params."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    action_id: str
    idempotency_key: str
    execution_job_id: str | None = None
    execution_owner: ExecutionOwner = ExecutionOwner.DIRECT_TOOL
    connector: str = DEFAULT_CONNECTOR
    executed_by: str = "shadowtrace"


class MockToolProviderConfig(BaseModel):
    """Deterministic fault and capacity controls for contract tests."""

    model_config = ConfigDict(extra="forbid")

    disabled_tools: set[str] = Field(default_factory=set)
    capacity_limits: dict[str, int] = Field(default_factory=dict)
    missing_targets: set[str] = Field(default_factory=set)
    offline_targets: set[str] = Field(default_factory=set)
    permission_denied_targets: set[str] = Field(default_factory=set)
    transient_error_targets: set[str] = Field(default_factory=set)
    timed_out_targets: set[str] = Field(default_factory=set)
    cancelled_targets: set[str] = Field(default_factory=set)
    late_success_targets: set[str] = Field(default_factory=set)
    lost_response_targets: set[str] = Field(default_factory=set)
    observation_never_targets: set[str] = Field(default_factory=set)
    observation_reversed_targets: set[str] = Field(default_factory=set)
    new_alert_events: set[str] = Field(default_factory=set)
    observation_delay_ms: int = Field(default=25, ge=0)
    poll_after_ms: int = Field(default=25, ge=0)
    job_lease_seconds: float = Field(default=30.0, gt=0)

    @field_validator("capacity_limits")
    @classmethod
    def _capacity_limits_are_valid(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = set(value) - MOCK_STATE_NAMESPACES
        if unknown:
            raise ValueError(f"unknown capacity namespaces: {sorted(unknown)!r}")
        if any(limit < 0 for limit in value.values()):
            raise ValueError("capacity limits must be non-negative")
        return value


class MockToolProvider:
    """A stateful provider that separates acceptance from effect application."""

    name = PROVIDER_NAME

    def __init__(
        self,
        state: MockEnvironmentState | None = None,
        *,
        config: MockToolProviderConfig | None = None,
    ) -> None:
        self.state = state or MockEnvironmentState()
        self.config = config or MockToolProviderConfig()
        self._metas = {
            name: meta
            for name, meta in baseline_tool_index().items()
            if meta.tool_category is ToolCategory.RESPONSE and meta.executable
        }

    def capability_manifest(self) -> CapabilityManifest:
        operations = sorted(set(self._metas) - self.config.disabled_tools)
        return CapabilityManifest(
            provider_name=self.name,
            online=True,
            entity_response=CapabilityState.SUPPORTED,
            allowed_intents=[DispositionIntentKind.EXECUTION_RESULT_RECORD],
            allowed_operations=operations,
            allowed_target_types=sorted(
                {target for meta in self._metas.values() for target in meta.target_types}
            ),
            supports_status_query=True,
            supports_lookup_by_idempotency=True,
            supports_idempotency=True,
            bindings=[
                CapabilityBindingEntry(
                    intent_kind=DispositionIntentKind.EXECUTION_RESULT_RECORD,
                    operation_code=name,
                    state=CapabilityState.SUPPORTED,
                )
                for name in operations
            ],
        )

    def provider_bindings(self) -> list[ProviderToolBinding]:
        """Declare both mutually exclusive channels; only DIRECT_TOOL points here."""

        bindings: list[ProviderToolBinding] = []
        enabled = set(self.capability_manifest().allowed_operations)
        for tool_name in sorted(self._metas):
            if tool_name in enabled:
                bindings.append(
                    ProviderToolBinding(
                        tool_name=tool_name,
                        provider_name=self.name,
                        execution_owner=ExecutionOwner.DIRECT_TOOL,
                        execution_channel=ExecutionChannel.TOOL_PROVIDER,
                        capabilities=["entity_response"],
                    )
                )
            bindings.append(
                ProviderToolBinding(
                    tool_name=tool_name,
                    provider_name=XDR_PROVIDER_NAME,
                    execution_owner=ExecutionOwner.XDR_MANAGED,
                    execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
                    capabilities=["entity_response"],
                )
            )
        bindings.append(
            ProviderToolBinding(
                tool_name=TERMINAL_DISPOSITION_TOOL,
                provider_name=XDR_PROVIDER_NAME,
                execution_owner=ExecutionOwner.XDR_MANAGED,
                execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
                capabilities=["event_disposition"],
            )
        )
        return bindings

    def register_bindings(self, registry: Any) -> None:
        """Attach manifest bindings idempotently to an already-discovered registry.

        Bindings for tools that are not present (for example disposition-only
        virtual metas omitted by ``include_virtual=False``) are skipped so a
        partial catalog still receives the executable response channels.
        """

        from app.tools.registry import ToolNotFoundError

        for binding in self.provider_bindings():
            try:
                existing = registry.list_bindings(binding.tool_name)
            except ToolNotFoundError:
                continue
            if any(item == binding for item in existing):
                continue
            registry.register_binding(binding)

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Validate, reserve the dispatch, and return either a job or ToolResult."""

        context = context or get_tool_execution_context(tool_name, params)
        meta = self._metas.get(tool_name)
        if meta is None:
            return self._error_result(
                tool_name,
                ToolResultStatus.UNSUPPORTED,
                "tool is not an executable response capability",
                code="capability_missing",
            )
        if context.execution_owner is not ExecutionOwner.DIRECT_TOOL:
            return self._error_result(
                tool_name,
                ToolResultStatus.VALIDATION_ERROR,
                "MockToolProvider accepts only execution_owner=direct_tool",
                code="wrong_execution_owner",
            )
        if tool_name in self.config.disabled_tools:
            return self._error_result(
                tool_name,
                ToolResultStatus.UNSUPPORTED,
                "provider capability is unavailable",
                code="capability_missing",
            )

        input_model = TOOL_INPUT_MODELS[tool_name]
        try:
            parsed = input_model.model_validate(params)
        except ValidationError as exc:
            return self._error_result(
                tool_name,
                ToolResultStatus.VALIDATION_ERROR,
                self._validation_detail(exc),
                code="validation_error",
            )
        try:
            targets = self._validated_targets(meta.target_types, parsed)
        except ValueError as exc:
            return self._error_result(
                tool_name,
                ToolResultStatus.VALIDATION_ERROR,
                str(exc),
                code="validation_error",
            )

        frozen_owner = await self.state.claim_execution_owner(
            context.action_id,
            context.execution_owner.value,
        )
        if frozen_owner != context.execution_owner.value:
            return self._error_result(
                tool_name,
                ToolResultStatus.VALIDATION_ERROR,
                f"action is already frozen to execution_owner={frozen_owner}",
                code="execution_owner_conflict",
            )

        now = _utc_now()
        job = ActionExecutionJob(
            job_id=context.execution_job_id or new_job_id(),
            event_id=context.event_id,
            action_id=context.action_id,
            provider_name=self.name,
            idempotency_key=context.idempotency_key,
            provider_job_id=f"mjob-{secrets.token_hex(6)}",
            created_at=now,
            updated_at=now,
            poll_after_ms=self.config.poll_after_ms,
        )
        normalized_params = parsed.model_dump(mode="json")
        dispatch_identity = {
            "tool_name": tool_name,
            "event_id": context.event_id,
            "action_id": context.action_id,
            "execution_owner": context.execution_owner.value,
            "connector": context.connector,
            "parameters": normalized_params,
        }
        intent = {
            "job_id": job.job_id,
            "event_id": context.event_id,
            "action_id": context.action_id,
            "tool_name": tool_name,
            "execution_owner": context.execution_owner.value,
            "connector": context.connector,
            "executed_by": context.executed_by,
            "parameters": normalized_params,
            "payload_hash": _payload_hash(dispatch_identity),
            "status": ExecutionJobStatus.QUEUED.value,
            "created_at": now.isoformat(),
        }
        reserved_job_id, created = await self.state.reserve_dispatch(
            idempotency_key=context.idempotency_key,
            job_id=job.job_id,
            job=job.model_dump(mode="json"),
            intent=intent,
        )
        if not created:
            previous_intent = await self.state.get_dispatch_intent(reserved_job_id)
            if (
                previous_intent is None
                or previous_intent.get("payload_hash") != intent["payload_hash"]
            ):
                return self._error_result(
                    tool_name,
                    ToolResultStatus.VALIDATION_ERROR,
                    "idempotency key reused with a different payload",
                    code="idempotency_key_reuse",
                )
            stored = await self.get_job(reserved_job_id)
            return self._result_for(meta.async_mode, tool_name, stored)

        fault_targets = targets or [("operation", tool_name)]
        if self._matches_any(self.config.lost_response_targets, fault_targets):
            stored = await self.run_job(job.job_id)
            unknown = stored.model_copy(
                update={
                    "status": ExecutionJobStatus.UNKNOWN,
                    "provider_code": "response_lost",
                    "provider_message": "response lost; lookup by idempotency before retry",
                    "raw_result": {
                        "fixture": "shadowtrace_mock_tool",
                        "response_lost": True,
                    },
                }
            )
            return (
                unknown.model_dump(mode="json")
                if meta.async_mode
                else self._tool_result(tool_name, unknown, status=ToolResultStatus.UNKNOWN)
            )

        if meta.async_mode:
            return job.model_dump(mode="json")
        completed = await self.run_job(job.job_id)
        return self._tool_result(tool_name, completed)

    async def get_job(self, job_id: str) -> ActionExecutionJob:
        raw = await self.state.get_job(job_id)
        if raw is None:
            raise KeyError(f"job {job_id!r} not found")
        return ActionExecutionJob.model_validate(raw)

    async def lookup_by_idempotency(
        self,
        tool_name: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
        job_id = await self.state.get_state("idempotency", digest)
        if not isinstance(job_id, str):
            return None
        intent = await self.state.get_dispatch_intent(job_id)
        if intent is None or intent.get("tool_name") != tool_name:
            return None
        job = await self.get_job(job_id)
        meta = self._metas[tool_name]
        return self._result_for(meta.async_mode, tool_name, job)

    async def run_job(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
    ) -> ActionExecutionJob:
        """Run one accepted job. Calling this again never repeats a side effect."""

        job = await self.get_job(job_id)
        if job.status in _TERMINAL_JOB_STATUSES or job.status is ExecutionJobStatus.UNKNOWN:
            return job
        claimant = worker_id or f"mock-worker-{secrets.token_hex(4)}"
        claim_token = await self.state.claim_job(
            job_id,
            claimant,
            lease_seconds=self.config.job_lease_seconds,
        )
        if not claim_token:
            return await self.get_job(job_id)
        try:
            job = await self.get_job(job_id)
            if job.status in _TERMINAL_JOB_STATUSES or job.status is ExecutionJobStatus.UNKNOWN:
                return job
            if job.status is ExecutionJobStatus.QUEUED:
                validate_job_status_transition(job.status, ExecutionJobStatus.RUNNING)
            elif job.status is not ExecutionJobStatus.RUNNING:
                return job

            now = _utc_now()
            job = job.model_copy(
                update={
                    "status": ExecutionJobStatus.RUNNING,
                    "claimed_by": claimant,
                    "lease_expires_at": now + timedelta(seconds=self.config.job_lease_seconds),
                    "started_at": job.started_at or now,
                    "updated_at": now,
                    "attempt": job.attempt + 1,
                }
            )
            if not await self.state.set_job_if_claimed(
                job_id,
                job.model_dump(mode="json"),
                worker_id=claimant,
                token=claim_token,
            ):
                return await self.get_job(job_id)

            intent = await self.state.get_dispatch_intent(job_id)
            if intent is None:
                raise RuntimeError(f"dispatch intent for {job_id!r} is missing")
            tool_name = str(intent["tool_name"])
            params = dict(intent["parameters"])
            context = ToolExecutionContext(
                event_id=str(intent["event_id"]),
                action_id=str(intent["action_id"]),
                idempotency_key=job.idempotency_key,
                connector=str(intent["connector"]),
                executed_by=str(intent["executed_by"]),
            )

            if tool_name == "create_ticket":
                results = [await self._create_ticket(job, params, context)]
            elif tool_name == "notify_security_team":
                results = [await self._create_notification(job, params, context)]
            else:
                meta = self._metas[tool_name]
                parsed = TOOL_INPUT_MODELS[tool_name].model_validate(params)
                targets = self._validated_targets(meta.target_types, parsed)
                results = []
                has_pending_confirmation = False
                for target_type, target in targets:
                    if self._matches_any(
                        self.config.late_success_targets,
                        [(target_type, target)],
                    ):
                        has_pending_confirmation = True
                        results.append(
                            self._target_result(
                                target_type,
                                target,
                                TargetExecutionStatus.UNKNOWN,
                                "pending_confirmation",
                            )
                        )
                    else:
                        results.append(
                            await self._apply_target(
                                tool_name,
                                target_type,
                                target,
                                job,
                                context,
                            )
                        )
                if has_pending_confirmation:
                    return await self._finish_job(
                        job,
                        ExecutionJobStatus.UNKNOWN,
                        results,
                        provider_code="pending_confirmation",
                        provider_message="effect awaits provider status confirmation",
                        claim=(claimant, claim_token),
                    )

            status = self._aggregate_status(results)
            return await self._finish_job(
                job,
                status,
                results,
                claim=(claimant, claim_token),
            )
        finally:
            await self.state.release_job_claim(job_id, claimant, claim_token)

    async def resolve_late_success(self, job_id: str) -> ActionExecutionJob:
        """Confirm an UNKNOWN job and apply its delayed provider effect once."""

        job = await self.get_job(job_id)
        if job.status is not ExecutionJobStatus.UNKNOWN:
            return job
        intent = await self.state.get_dispatch_intent(job_id)
        if intent is None:
            raise RuntimeError(f"dispatch intent for {job_id!r} is missing")
        tool_name = str(intent["tool_name"])
        params = TOOL_INPUT_MODELS[tool_name].model_validate(intent["parameters"])
        context = ToolExecutionContext(
            event_id=str(intent["event_id"]),
            action_id=str(intent["action_id"]),
            idempotency_key=job.idempotency_key,
            connector=str(intent["connector"]),
            executed_by=str(intent["executed_by"]),
        )
        meta = self._metas[tool_name]
        prior_results = {item.canonical_target: item for item in job.target_results}
        results: list[TargetExecutionResult] = []
        for target_type, target in self._validated_targets(meta.target_types, params):
            canonical = f"{target_type}:{target}"
            prior = prior_results.get(canonical)
            if prior is not None and prior.status is not TargetExecutionStatus.UNKNOWN:
                results.append(prior)
                continue
            results.append(
                await self._apply_target(
                    tool_name,
                    target_type,
                    target,
                    job,
                    context,
                    ignore_late=True,
                )
            )
        status = self._aggregate_status(results)
        return await self._finish_job(
            job,
            status,
            results,
            provider_code="late_confirmation",
            provider_message="provider confirmed a previously unknown result",
            expected_status=ExecutionJobStatus.UNKNOWN,
        )

    async def cancel_job(self, job_id: str) -> ActionExecutionJob:
        job = await self.get_job(job_id)
        if job.status is not ExecutionJobStatus.QUEUED:
            return job
        intent = await self.state.get_dispatch_intent(job_id)
        results: list[TargetExecutionResult] = []
        if intent is not None:
            tool_name = str(intent["tool_name"])
            meta = self._metas[tool_name]
            parsed = TOOL_INPUT_MODELS[tool_name].model_validate(intent["parameters"])
            targets = self._validated_targets(meta.target_types, parsed)
            if not targets:
                targets = [("operation", tool_name)]
            results = [
                self._target_result(
                    target_type,
                    target,
                    TargetExecutionStatus.SKIPPED,
                    "cancelled",
                )
                for target_type, target in targets
            ]
        return await self._finish_job(
            job,
            ExecutionJobStatus.CANCELLED,
            results,
            expected_status=ExecutionJobStatus.QUEUED,
        )

    async def _apply_target(
        self,
        tool_name: str,
        target_type: str,
        target: str,
        job: ActionExecutionJob,
        context: ToolExecutionContext,
        *,
        ignore_late: bool = False,
    ) -> TargetExecutionResult:
        canonical = f"{target_type}:{target}"
        fault = self._target_fault(canonical, target, ignore_late=ignore_late)
        if fault is not None:
            status = (
                TargetExecutionStatus.SKIPPED
                if fault == "cancelled"
                else TargetExecutionStatus.UNKNOWN
                if fault == "timed_out"
                else TargetExecutionStatus.FAILED
            )
            return self._target_result(target_type, target, status, fault)

        namespace, state_status = _TOOL_STATE[tool_name]
        artifact_id = (
            f"scan-{hashlib.sha256(f'{job.job_id}:{target}'.encode()).hexdigest()[:8]}"
            if tool_name == "scan_host_for_virus"
            else None
        )
        record = MockStateRecord(
            status=state_status,
            reason="mock provider effect",
            executed_by=context.executed_by,
            provider=self.name,
            connector=context.connector,
            action_id=context.action_id,
            job_id=job.job_id,
            value={
                "target_type": target_type,
                "target": target,
                "artifact_id": artifact_id,
            },
        )
        stored, _, code = await self.state.apply_effect(
            job_id=job.job_id,
            namespace=namespace,
            key=target,
            record=record.model_dump(mode="json"),
            desired_status=state_status,
            allow_update=tool_name == "scan_host_for_virus",
            capacity=self.config.capacity_limits.get(namespace),
        )
        if code == "capacity_exceeded":
            return self._target_result(
                target_type,
                target,
                TargetExecutionStatus.FAILED,
                code,
            )
        if isinstance(stored, dict):
            stored_artifact = stored.get("value", {}).get("artifact_id")
            if isinstance(stored_artifact, str):
                artifact_id = stored_artifact
            await self._copy_effect_to_observation(
                tool_name,
                target_type,
                target,
                stored,
                job,
                context,
            )
        return self._target_result(
            target_type,
            target,
            TargetExecutionStatus.SUCCESS,
            code,
            artifact_id=artifact_id,
        )

    async def _copy_effect_to_observation(
        self,
        tool_name: str,
        target_type: str,
        target: str,
        stored: dict[str, Any],
        job: ActionExecutionJob,
        context: ToolExecutionContext,
    ) -> None:
        """Schedule independent observation records without exposing effect namespaces."""

        configured_target = [(target_type, target)]
        never_visible = self._matches_any(
            self.config.observation_never_targets,
            configured_target,
        )
        reverse = self._matches_any(
            self.config.observation_reversed_targets,
            configured_target,
        )
        now = _utc_now()
        available_at = now + timedelta(milliseconds=self.config.observation_delay_ms)
        status = str(stored.get("status") or "")
        if reverse:
            status = _REVERSED_OBSERVATION_STATUS.get(status, f"not_{status}")
        common: dict[str, Any] = {
            "observed_at": available_at,
            "available_at": available_at,
            "observed_version": int(stored.get("version", 1)),
            "action_id": str(stored.get("action_id") or context.action_id),
            "job_id": str(stored.get("job_id") or job.job_id),
            "provider": str(stored.get("provider") or self.name),
            "connector": str(stored.get("connector") or context.connector),
            "value": dict(stored.get("value") or {}),
        }
        if not never_visible:
            await self.state.set_observation(
                MockObservationRecord(
                    surface=_TOOL_OBSERVATION_SURFACE[tool_name],
                    target=target,
                    status=status,
                    **common,
                )
            )
        if not never_visible and tool_name in {"block_ip", "isolate_host"}:
            traffic_status = "flowing" if reverse else "dropped"
            await self.state.set_observation(
                MockObservationRecord(
                    surface="traffic",
                    target=target,
                    status=traffic_status,
                    **common,
                )
            )
        if context.event_id in self.config.new_alert_events:
            alert_status = _REVERSED_OBSERVATION_STATUS["detected"] if reverse else "detected"
            await self.state.set_observation(
                MockObservationRecord(
                    surface="new_alerts",
                    target=context.event_id,
                    status=alert_status,
                    **{
                        **common,
                        "action_id": context.action_id,
                        "job_id": job.job_id,
                    },
                )
            )

    async def _create_ticket(
        self,
        job: ActionExecutionJob,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> TargetExecutionResult:
        fault = self._target_fault(
            "operation:create_ticket",
            "create_ticket",
            ignore_late=False,
        )
        if fault is not None:
            return self._fault_result("ticket", "pending", fault)
        sequence = await self.state.allocate_ticket_sequence(job.job_id)
        if sequence > 9_999:
            return self._target_result(
                "ticket",
                "capacity",
                TargetExecutionStatus.FAILED,
                "capacity_exceeded",
            )
        ticket_year = (job.created_at or _utc_now()).year
        ticket_id = f"TKT-{ticket_year}-{sequence:04d}"
        record = MockStateRecord(
            status="open",
            reason="ticket created",
            executed_by=context.executed_by,
            provider=self.name,
            connector=context.connector,
            action_id=context.action_id,
            job_id=job.job_id,
            value={"ticket_id": ticket_id, **params},
        )
        _, _, code = await self.state.apply_effect(
            job_id=job.job_id,
            namespace="tickets",
            key=ticket_id,
            record=record.model_dump(mode="json"),
            desired_status="open",
            allow_update=False,
            capacity=self.config.capacity_limits.get("tickets"),
        )
        if code == "capacity_exceeded":
            return self._target_result(
                "ticket",
                ticket_id,
                TargetExecutionStatus.FAILED,
                code,
            )
        return self._target_result(
            "ticket",
            ticket_id,
            TargetExecutionStatus.SUCCESS,
            "created" if code == "applied" else code,
            artifact_id=ticket_id,
        )

    async def _create_notification(
        self,
        job: ActionExecutionJob,
        params: dict[str, Any],
        context: ToolExecutionContext,
    ) -> TargetExecutionResult:
        fault = self._target_fault(
            "operation:notify_security_team",
            "notify_security_team",
            ignore_late=False,
        )
        if fault is not None:
            return self._fault_result("notification", "pending", fault)
        notification_id = f"ntf-{hashlib.sha256(job.job_id.encode()).hexdigest()[:8]}"
        record = MockStateRecord(
            status="sent",
            reason="notification sent",
            executed_by=context.executed_by,
            provider=self.name,
            connector=context.connector,
            action_id=context.action_id,
            job_id=job.job_id,
            value={"notification_id": notification_id, **params},
        )
        _, _, code = await self.state.apply_effect(
            job_id=job.job_id,
            namespace="notifications",
            key=notification_id,
            record=record.model_dump(mode="json"),
            desired_status="sent",
            allow_update=False,
            capacity=self.config.capacity_limits.get("notifications"),
        )
        if code == "capacity_exceeded":
            return self._target_result(
                "notification",
                notification_id,
                TargetExecutionStatus.FAILED,
                code,
            )
        return self._target_result(
            "notification",
            notification_id,
            TargetExecutionStatus.SUCCESS,
            "sent" if code == "applied" else code,
            artifact_id=notification_id,
        )

    async def _finish_job(
        self,
        job: ActionExecutionJob,
        status: ExecutionJobStatus,
        results: list[TargetExecutionResult],
        *,
        provider_code: str | None = None,
        provider_message: str | None = None,
        claim: tuple[str, int] | None = None,
        expected_status: ExecutionJobStatus | None = None,
    ) -> ActionExecutionJob:
        validate_job_status_transition(
            job.status,
            status,
            provider_confirmed_terminal=job.status is ExecutionJobStatus.UNKNOWN,
        )
        now = _utc_now()
        code = provider_code or (next((item.code for item in results if item.code), None))
        updated = job.model_copy(
            update={
                "status": status,
                "target_results": results,
                "provider_code": code,
                "provider_message": provider_message,
                "claimed_by": None,
                "lease_expires_at": None,
                "updated_at": now,
                "finished_at": now if status in _TERMINAL_JOB_STATUSES else None,
                "raw_result": {
                    "fixture": "shadowtrace_mock_tool",
                    "outcome": status.value,
                    "target_codes": [item.code for item in results],
                },
            }
        )
        if claim is not None:
            worker_id, token = claim
            saved = await self.state.set_job_if_claimed(
                job.job_id,
                updated.model_dump(mode="json"),
                worker_id=worker_id,
                token=token,
            )
            if not saved:
                return await self.get_job(job.job_id)
        elif expected_status is not None:
            saved = await self.state.set_job_if_status(
                job.job_id,
                updated.model_dump(mode="json"),
                expected_status=expected_status.value,
            )
            if not saved:
                return await self.get_job(job.job_id)
        else:
            await self.state.set_job(job.job_id, updated.model_dump(mode="json"))
        return updated

    def _target_fault(self, canonical: str, target: str, *, ignore_late: bool) -> str | None:
        checks: tuple[tuple[set[str], str], ...] = (
            (self.config.missing_targets, "target_not_found"),
            (self.config.offline_targets, "device_offline"),
            (self.config.permission_denied_targets, "permission_denied"),
            (self.config.transient_error_targets, "transient_error"),
            (self.config.timed_out_targets, "timed_out"),
            (self.config.cancelled_targets, "cancelled"),
        )
        for configured, code in checks:
            if canonical in configured or target in configured:
                return code
        if not ignore_late and (
            canonical in self.config.late_success_targets
            or target in self.config.late_success_targets
        ):
            return "pending_confirmation"
        return None

    @staticmethod
    def _validated_targets(
        allowed_target_types: list[str],
        parsed: BaseModel,
    ) -> list[tuple[str, str]]:
        target = getattr(parsed, "target", None)
        target_type = getattr(parsed, "target_type", None)
        if target is None or target_type is None:
            return []
        if not isinstance(target_type, str) or target_type not in allowed_target_types:
            raise ValueError(f"target_type must be one of {sorted(allowed_target_types)!r}")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")
        parameters = getattr(parsed, "parameters", {})
        extras = parameters.get("targets") if isinstance(parameters, dict) else None
        if extras is None:
            values = [target]
        elif not isinstance(extras, list) or not extras:
            raise ValueError("parameters.targets must be a non-empty list of strings")
        else:
            values = extras
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("each parameters.targets item must be a non-empty string")
        return [(target_type, item) for item in values]

    @staticmethod
    def _matches_any(configured: set[str], targets: list[tuple[str, str]]) -> bool:
        return any(
            target in configured or f"{kind}:{target}" in configured for kind, target in targets
        )

    @staticmethod
    def _target_result(
        target_type: str,
        target: str,
        status: TargetExecutionStatus,
        code: str,
        *,
        artifact_id: str | None = None,
    ) -> TargetExecutionResult:
        canonical = f"{target_type}:{target}"
        return TargetExecutionResult(
            canonical_target=canonical,
            status=status,
            code=code,
            message=code.replace("_", " "),
            artifact_id=artifact_id,
            raw_result={
                "fixture": "shadowtrace_mock_tool",
                "target": canonical,
                "code": code,
            },
        )

    def _fault_result(
        self,
        target_type: str,
        target: str,
        fault: str,
    ) -> TargetExecutionResult:
        status = (
            TargetExecutionStatus.SKIPPED
            if fault == "cancelled"
            else TargetExecutionStatus.UNKNOWN
            if fault == "timed_out"
            else TargetExecutionStatus.FAILED
        )
        return self._target_result(target_type, target, status, fault)

    @staticmethod
    def _aggregate_status(results: list[TargetExecutionResult]) -> ExecutionJobStatus:
        if not results:
            return ExecutionJobStatus.CANCELLED
        success_count = sum(item.status is TargetExecutionStatus.SUCCESS for item in results)
        codes = {item.code for item in results}
        if success_count == len(results):
            return ExecutionJobStatus.SUCCESS
        if success_count:
            return ExecutionJobStatus.PARTIAL_SUCCESS
        if codes == {"timed_out"}:
            return ExecutionJobStatus.TIMED_OUT
        if codes == {"cancelled"}:
            return ExecutionJobStatus.CANCELLED
        return ExecutionJobStatus.FAILED

    def _result_for(
        self,
        async_mode: bool,
        tool_name: str,
        job: ActionExecutionJob,
    ) -> dict[str, Any]:
        return job.model_dump(mode="json") if async_mode else self._tool_result(tool_name, job)

    def _tool_result(
        self,
        tool_name: str,
        job: ActionExecutionJob,
        *,
        status: ToolResultStatus | None = None,
    ) -> dict[str, Any]:
        mapped = {
            ExecutionJobStatus.QUEUED: ToolResultStatus.ACCEPTED,
            ExecutionJobStatus.RUNNING: ToolResultStatus.ACCEPTED,
            ExecutionJobStatus.PARTIAL_SUCCESS: ToolResultStatus.PARTIAL_SUCCESS,
            ExecutionJobStatus.SUCCESS: ToolResultStatus.SUCCESS,
            ExecutionJobStatus.FAILED: ToolResultStatus.FAILED,
            ExecutionJobStatus.TIMED_OUT: ToolResultStatus.TIMEOUT,
            ExecutionJobStatus.CANCELLED: ToolResultStatus.FAILED,
            ExecutionJobStatus.UNKNOWN: ToolResultStatus.UNKNOWN,
        }
        result = ToolResult(
            call_id=new_call_id(),
            tool_name=tool_name,
            provider_name=self.name,
            status=status or mapped[job.status],
            job_id=job.job_id,
            provider_job_id=job.provider_job_id,
            target_results=job.target_results,
            provider_code=job.provider_code,
            provider_message=job.provider_message,
            raw_result=job.raw_result,
            data={
                "artifact_ids": [
                    item.artifact_id for item in job.target_results if item.artifact_id is not None
                ]
            },
        )
        return result.model_dump(mode="json")

    def _error_result(
        self,
        tool_name: str,
        status: ToolResultStatus,
        detail: str,
        *,
        code: str,
    ) -> dict[str, Any]:
        return ToolResult(
            call_id=new_call_id(),
            tool_name=tool_name,
            provider_name=self.name,
            status=status,
            provider_code=code,
            provider_message=detail,
            error_detail=detail,
            raw_result=sanitize_raw_result(
                {
                    "fixture": "shadowtrace_mock_tool",
                    "code": code,
                    "message": detail,
                }
            ),
        ).model_dump(mode="json")

    @staticmethod
    def _validation_detail(exc: ValidationError) -> str:
        return "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )


_provider_override: contextvars.ContextVar[MockToolProvider | None] = contextvars.ContextVar(
    "mock_tool_provider_override",
    default=None,
)
_execution_context: contextvars.ContextVar[ToolExecutionContext | None] = contextvars.ContextVar(
    "tool_execution_context",
    default=None,
)
_default_provider: MockToolProvider | None = None


def get_mock_tool_provider() -> MockToolProvider:
    override = _provider_override.get()
    if override is not None:
        return override
    settings = get_settings()
    if settings.tool_mode != "mock" or not settings.simulation_enabled:
        raise RuntimeError(
            "implicit MockToolProvider is available only when "
            "TOOL_MODE=mock and SIMULATION_ENABLED=true"
        )
    global _default_provider
    if _default_provider is None:
        _default_provider = MockToolProvider()
    return _default_provider


def get_tool_execution_context(
    tool_name: str,
    params: dict[str, Any],
) -> ToolExecutionContext:
    bound = _execution_context.get()
    if bound is not None:
        return bound
    digest = _payload_hash({"tool_name": tool_name, "params": params})
    now = _utc_now()
    return ToolExecutionContext(
        event_id=new_event_id(f"mock-tool|{digest}", now),
        action_id=f"act-{digest[:8]}",
        idempotency_key=f"mock-tool:{digest}",
    )


@contextmanager
def bind_mock_tool_provider(provider: MockToolProvider) -> Iterator[None]:
    token = _provider_override.set(provider)
    try:
        yield
    finally:
        _provider_override.reset(token)


@contextmanager
def bind_tool_execution_context(context: ToolExecutionContext) -> Iterator[None]:
    token = _execution_context.set(context)
    try:
        yield
    finally:
        _execution_context.reset(token)


async def execute_mock_response_tool(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    return await get_mock_tool_provider().execute(tool_name, params)


__all__ = [
    "MockToolProvider",
    "MockToolProviderConfig",
    "ToolExecutionContext",
    "bind_mock_tool_provider",
    "bind_tool_execution_context",
    "execute_mock_response_tool",
    "get_mock_tool_provider",
    "map_disposition_receipt_to_job",
]
