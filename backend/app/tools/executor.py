"""Tool execution engine — sole entry for Agent tool calls (ISSUE-024)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from app.core.errors import is_retryable
from app.core.event_bus import EventBus
from app.models.enums import ExecutionJobStatus, ExecutionOwner, ToolCategory
from app.models.execution import ActionExecutionJob
from app.models.ids import new_call_id
from app.models.tool_meta import (
    ExecutionChannel,
    RoutingKind,
    ToolMeta,
    ToolResult,
    ToolResultStatus,
    WrongExecutionChannelError,
    ensure_tool_provider_executable,
)
from app.models.workflow import validate_job_status_transition
from app.providers.tools.mock_provider import (
    ToolExecutionContext,
    bind_tool_execution_context,
)
from app.services.tool_call_log_service import ToolCallLogService
from app.tools.circuit_breaker import CircuitBreakerRegistry
from app.tools.registry import RegisteredTool, ToolRegistry, ToolValidationError
from app.tools.retry import RetryPolicy

logger = logging.getLogger(__name__)


class CallNature(StrEnum):
    """Derived from trusted registry metadata — never caller-supplied."""

    QUERY = "query"
    VERIFICATION = "verification"
    SIDE_EFFECT = "side_effect"
    VIRTUAL = "virtual"


def derive_call_nature(meta: ToolMeta) -> CallNature:
    if meta.routing_kind is RoutingKind.DISPOSITION_ONLY or not meta.executable:
        return CallNature.VIRTUAL
    if meta.tool_category is ToolCategory.QUERY:
        return CallNature.QUERY
    if meta.tool_category is ToolCategory.VERIFICATION:
        return CallNature.VERIFICATION
    return CallNature.SIDE_EFFECT


@runtime_checkable
class ConvergenceGuardPort(Protocol):
    async def record_step(self, event_id: str, *, tool_name: str) -> None: ...

    async def should_stop(self, event_id: str) -> bool: ...


@runtime_checkable
class BudgetServicePort(Protocol):
    async def charge_tool(self, event_id: str, agent_name: str, tool_name: str) -> None: ...


@runtime_checkable
class ExecutionJobStorePort(Protocol):
    async def get_job(self, job_id: str) -> ActionExecutionJob | None: ...

    async def cas_update_job(
        self,
        job_id: str,
        updated: ActionExecutionJob,
        *,
        expected_status: ExecutionJobStatus,
    ) -> bool: ...


class InMemoryExecutionJobStore:
    """Lightweight job store for tests and mock side-effect CAS writeback."""

    def __init__(self) -> None:
        self._jobs: dict[str, ActionExecutionJob] = {}

    async def seed_job(self, job: ActionExecutionJob) -> None:
        self._jobs[job.job_id] = job.model_copy(deep=True)

    async def get_job(self, job_id: str) -> ActionExecutionJob | None:
        job = self._jobs.get(job_id)
        return job.model_copy(deep=True) if job is not None else None

    async def cas_update_job(
        self,
        job_id: str,
        updated: ActionExecutionJob,
        *,
        expected_status: ExecutionJobStatus,
    ) -> bool:
        current = self._jobs.get(job_id)
        if current is None or current.status is not expected_status:
            return False
        if current.status is updated.status:
            self._jobs[job_id] = updated.model_copy(deep=True)
            return True
        validate_job_status_transition(current.status, updated.status)
        self._jobs[job_id] = updated.model_copy(deep=True)
        return True


class NoopConvergenceGuard:
    async def record_step(self, event_id: str, *, tool_name: str) -> None:
        return None

    async def should_stop(self, event_id: str) -> bool:
        return False


class NoopBudgetService:
    async def charge_tool(self, event_id: str, agent_name: str, tool_name: str) -> None:
        return None


class NullEventBus:
    async def publish_event(
        self,
        event_id: str,
        message_type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        return True


class NullAuditService:
    async def log_start(self, *args: Any, **kwargs: Any) -> str:
        return str(kwargs.get("call_id") or args[0] if args else new_call_id())

    async def log_finish(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass(slots=True)
class ToolExecutor:
    """Validate, guard, retry, break, audit, and dispatch tool calls."""

    registry: ToolRegistry
    audit_service: ToolCallLogService | NullAuditService = field(default_factory=NullAuditService)
    event_bus: EventBus | NullEventBus = field(default_factory=NullEventBus)
    convergence_guard: ConvergenceGuardPort | None = field(default_factory=NoopConvergenceGuard)
    budget_service: BudgetServicePort | None = field(default_factory=NoopBudgetService)
    job_store: ExecutionJobStorePort | None = None
    breaker_registry: CircuitBreakerRegistry = field(default_factory=CircuitBreakerRegistry)
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    provider_context: Callable[[], AbstractContextManager[None]] | None = None

    async def call(
        self,
        tool_name: str,
        params: dict[str, Any],
        event_id: str,
        *,
        action_id: str | None = None,
        execution_job_id: str | None = None,
        idempotency_key: str | None = None,
        execution_owner: ExecutionOwner | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
        agent_name: str = "tool_agent",
    ) -> ToolResult:
        registered = self.registry.get_tool(tool_name)
        meta = registered.tool_meta
        call_nature = derive_call_nature(meta)
        policy = retry_policy or RetryPolicy()
        effective_timeout = float(timeout) if timeout is not None else float(meta.default_timeout_s)

        if call_nature is CallNature.VIRTUAL:
            ensure_tool_provider_executable(meta)
            raise WrongExecutionChannelError(tool_name, routing_kind=meta.routing_kind)

        binding_provider = "tool_provider"
        if call_nature is CallNature.SIDE_EFFECT:
            self._validate_side_effect_envelope(
                tool_name,
                action_id=action_id,
                execution_job_id=execution_job_id,
                idempotency_key=idempotency_key,
                execution_owner=execution_owner,
            )
            binding = self.registry.resolve_binding(
                tool_name,
                execution_owner,  # type: ignore[arg-type]
                [],
            )
            if binding.execution_channel is not ExecutionChannel.TOOL_PROVIDER:
                raise WrongExecutionChannelError(tool_name, routing_kind=meta.routing_kind)
            binding_provider = binding.provider_name
            await self._assert_precreated_job(
                execution_job_id=execution_job_id,  # type: ignore[arg-type]
                event_id=event_id,
                action_id=action_id,  # type: ignore[arg-type]
            )

        self.registry.validate_input(tool_name, params)

        breaker = self.breaker_registry.get(tool_name)
        call_id = new_call_id()
        audit_started = await self._safe_audit_start(
            call_id=call_id,
            event_id=event_id,
            action_id=action_id,
            tool_name=tool_name,
            tool_category=meta.tool_category.value,
            parameters=params,
        )
        await self._safe_publish(
            event_id,
            "tool_call_started",
            {
                "call_id": call_id,
                "tool_name": tool_name,
                "call_nature": call_nature.value,
            },
        )
        if not registered.submission_ready:
            result = self._failure_result(
                call_id=call_id,
                tool_name=tool_name,
                provider_name=binding_provider,
                status=ToolResultStatus.UNSUPPORTED,
                error_detail="configured Provider is unavailable; manual handling required",
                execution_time_ms=0,
            )
            await self._finalize(
                call_id=call_id,
                event_id=event_id,
                result=result,
                retry_count=0,
                audit_started=audit_started,
            )
            return result

        attempt = 0
        retry_count = 0
        dispatch_started = False
        started_monotonic = time.monotonic()

        while True:
            guard = self.convergence_guard
            if guard is not None and await guard.should_stop(event_id):
                result = self._failure_result(
                    call_id=call_id,
                    tool_name=tool_name,
                    provider_name=binding_provider,
                    status=ToolResultStatus.FAILED,
                    error_detail="convergence guard stopped execution",
                    execution_time_ms=self._elapsed_ms(started_monotonic),
                )
                await self._finalize(
                    call_id=call_id,
                    event_id=event_id,
                    result=result,
                    retry_count=retry_count,
                    audit_started=audit_started,
                )
                return result

            if attempt > 0:
                await self.sleep(policy.delay_for_attempt(attempt))

            if guard is not None:
                await guard.record_step(event_id, tool_name=tool_name)

            if not breaker.allow_request():
                result = self._circuit_open_result(
                    tool_name,
                    provider_name=binding_provider,
                    call_id=call_id,
                )
                await self._finalize(
                    call_id=call_id,
                    event_id=event_id,
                    result=result,
                    retry_count=retry_count,
                    audit_started=audit_started,
                )
                return result

            try:
                dispatch_started = True
                raw = await asyncio.wait_for(
                    self._dispatch(
                        registered=registered,
                        tool_name=tool_name,
                        params=params,
                        call_nature=call_nature,
                        event_id=event_id,
                        action_id=action_id,
                        execution_job_id=execution_job_id,
                        idempotency_key=idempotency_key,
                        execution_owner=execution_owner,
                    ),
                    timeout=effective_timeout,
                )
                try:
                    raw_result = ToolResult.model_validate(raw)
                except ValidationError:
                    raw_result = None
                if raw_result is None or raw_result.status in {
                    ToolResultStatus.ACCEPTED,
                    ToolResultStatus.SUCCESS,
                    ToolResultStatus.PARTIAL_SUCCESS,
                }:
                    self.registry.validate_output(tool_name, raw)
                result = self._coerce_tool_result(
                    raw,
                    call_id=call_id,
                    tool_name=tool_name,
                    provider_name=binding_provider,
                    execution_time_ms=self._elapsed_ms(started_monotonic),
                    force_provider_name=call_nature is CallNature.SIDE_EFFECT,
                )

                if result.status in {
                    ToolResultStatus.SUCCESS,
                    ToolResultStatus.PARTIAL_SUCCESS,
                    ToolResultStatus.ACCEPTED,
                }:
                    breaker.record_success()
                elif result.status in {
                    ToolResultStatus.FAILED,
                    ToolResultStatus.REMOTE_ERROR,
                    ToolResultStatus.RATE_LIMITED,
                    ToolResultStatus.AUTH_ERROR,
                    ToolResultStatus.VALIDATION_ERROR,
                }:
                    breaker.record_failure()

                if (
                    call_nature is CallNature.SIDE_EFFECT
                    and execution_job_id is not None
                    and self.job_store is not None
                    and result.job_id is not None
                ):
                    await self._cas_writeback_job(
                        execution_job_id,
                        result,
                        provider_name=binding_provider,
                    )

                if self.budget_service is not None:
                    await self.budget_service.charge_tool(event_id, agent_name, tool_name)

                await self._finalize(
                    call_id=call_id,
                    event_id=event_id,
                    result=result,
                    retry_count=retry_count,
                    audit_started=audit_started,
                )
                return result

            except TimeoutError:
                status = (
                    ToolResultStatus.UNKNOWN
                    if call_nature is CallNature.SIDE_EFFECT and dispatch_started
                    else ToolResultStatus.TIMEOUT
                )
                breaker.record_failure()
                if self._should_retry(
                    call_nature=call_nature,
                    meta=meta,
                    exc=TimeoutError(),
                    attempt=attempt,
                    policy=policy,
                    side_effect_dispatched=dispatch_started,
                ):
                    attempt += 1
                    retry_count += 1
                    continue
                result = self._failure_result(
                    call_id=call_id,
                    tool_name=tool_name,
                    provider_name=binding_provider,
                    status=status,
                    error_detail="tool execution timed out",
                    execution_time_ms=self._elapsed_ms(started_monotonic),
                )
                await self._finalize(
                    call_id=call_id,
                    event_id=event_id,
                    result=result,
                    retry_count=retry_count,
                    audit_started=audit_started,
                )
                return result

            except WrongExecutionChannelError:
                raise

            except Exception as exc:
                breaker.record_failure()
                if self._should_retry(
                    call_nature=call_nature,
                    meta=meta,
                    exc=exc,
                    attempt=attempt,
                    policy=policy,
                    side_effect_dispatched=dispatch_started,
                ):
                    attempt += 1
                    retry_count += 1
                    continue
                status = ToolResultStatus.FAILED
                if call_nature is CallNature.SIDE_EFFECT and dispatch_started:
                    status = ToolResultStatus.UNKNOWN
                result = self._failure_result(
                    call_id=call_id,
                    tool_name=tool_name,
                    provider_name=binding_provider,
                    status=status,
                    error_detail=str(exc),
                    execution_time_ms=self._elapsed_ms(started_monotonic),
                )
                await self._finalize(
                    call_id=call_id,
                    event_id=event_id,
                    result=result,
                    retry_count=retry_count,
                    audit_started=audit_started,
                )
                return result

    async def _dispatch(
        self,
        *,
        registered: RegisteredTool,
        tool_name: str,
        params: dict[str, Any],
        call_nature: CallNature,
        event_id: str,
        action_id: str | None,
        execution_job_id: str | None,
        idempotency_key: str | None,
        execution_owner: ExecutionOwner | None,
    ) -> dict[str, Any]:
        if call_nature is CallNature.SIDE_EFFECT:
            context = ToolExecutionContext(
                event_id=event_id,
                action_id=action_id or "",
                idempotency_key=idempotency_key or "",
                execution_job_id=execution_job_id,
                execution_owner=execution_owner or ExecutionOwner.DIRECT_TOOL,
            )
            provider_cm = (
                self.provider_context() if self.provider_context is not None else nullcontext()
            )
            with bind_tool_execution_context(context), provider_cm:
                return await registered.execute(params)
        return await registered.execute(params)

    @staticmethod
    def _validate_side_effect_envelope(
        tool_name: str,
        *,
        action_id: str | None,
        execution_job_id: str | None,
        idempotency_key: str | None,
        execution_owner: ExecutionOwner | None,
    ) -> None:
        missing = [
            name
            for name, value in (
                ("action_id", action_id),
                ("execution_job_id", execution_job_id),
                ("idempotency_key", idempotency_key),
                ("execution_owner", execution_owner),
            )
            if not value
        ]
        if missing:
            raise ToolValidationError(
                f"side-effect tool {tool_name!r} requires {', '.join(missing)}",
                details={"tool_name": tool_name, "missing": missing},
            )

    async def _assert_precreated_job(
        self,
        *,
        execution_job_id: str,
        event_id: str,
        action_id: str,
    ) -> None:
        if self.job_store is None:
            return
        job = await self.job_store.get_job(execution_job_id)
        if job is None:
            raise ToolValidationError(
                "pre-created execution job not found",
                details={"execution_job_id": execution_job_id},
            )
        if job.event_id != event_id or job.action_id != action_id:
            raise ToolValidationError(
                "execution job binding mismatch",
                details={
                    "execution_job_id": execution_job_id,
                    "event_id": event_id,
                    "action_id": action_id,
                },
            )

    async def _cas_writeback_job(
        self,
        execution_job_id: str,
        result: ToolResult,
        *,
        provider_name: str,
    ) -> None:
        if self.job_store is None:
            return
        current = await self.job_store.get_job(execution_job_id)
        if current is None:
            return
        updated = current.model_copy(
            update={
                "provider_name": provider_name,
                "provider_job_id": result.provider_job_id or current.provider_job_id,
                "target_results": result.target_results or current.target_results,
                "provider_code": result.provider_code,
                "provider_message": result.provider_message,
                "raw_result": result.raw_result or current.raw_result,
            }
        )
        if result.status is ToolResultStatus.ACCEPTED:
            updated.status = ExecutionJobStatus.QUEUED
        elif result.status is ToolResultStatus.SUCCESS:
            updated.status = ExecutionJobStatus.SUCCESS
        elif result.status is ToolResultStatus.PARTIAL_SUCCESS:
            updated.status = ExecutionJobStatus.PARTIAL_SUCCESS
        elif result.status is ToolResultStatus.UNKNOWN:
            updated.status = ExecutionJobStatus.UNKNOWN
        elif result.status is ToolResultStatus.TIMEOUT:
            updated.status = ExecutionJobStatus.TIMED_OUT
        elif result.status is ToolResultStatus.FAILED:
            updated.status = ExecutionJobStatus.FAILED
        await self.job_store.cas_update_job(
            execution_job_id,
            updated,
            expected_status=current.status,
        )

    @staticmethod
    def _should_retry(
        *,
        call_nature: CallNature,
        meta: ToolMeta,
        exc: BaseException,
        attempt: int,
        policy: RetryPolicy,
        side_effect_dispatched: bool,
    ) -> bool:
        if attempt >= policy.max_retries:
            return False
        if call_nature is CallNature.SIDE_EFFECT:
            if side_effect_dispatched:
                return False
            if not meta.idempotency:
                return False
            return isinstance(exc, TimeoutError) and meta.idempotency
        return is_retryable(exc)

    @staticmethod
    def _coerce_tool_result(
        raw: dict[str, Any],
        *,
        call_id: str,
        tool_name: str,
        provider_name: str,
        execution_time_ms: int | None,
        force_provider_name: bool = False,
    ) -> ToolResult:
        payload = dict(raw)
        payload.setdefault("call_id", call_id)
        payload.setdefault("tool_name", tool_name)
        if force_provider_name:
            payload["provider_name"] = provider_name
            if provider_name in {"mock_tool_provider", "mock_xdr"}:
                if isinstance(payload.get("data"), dict):
                    payload["data"] = {**payload["data"], "simulated": True}
                payload["raw_result"] = {
                    **(payload.get("raw_result") or {}),
                    "simulated": True,
                }
        else:
            payload.setdefault("provider_name", provider_name)
        payload.setdefault("execution_time_ms", execution_time_ms)
        try:
            return ToolResult.model_validate(payload)
        except ValidationError:
            job_payload = dict(raw)
            if force_provider_name:
                job_payload["provider_name"] = provider_name
                if provider_name in {"mock_tool_provider", "mock_xdr"}:
                    job_payload["raw_result"] = {
                        **(job_payload.get("raw_result") or {}),
                        "simulated": True,
                    }
            job = ActionExecutionJob.model_validate(job_payload)
            status = ToolResultStatus.ACCEPTED
            if job.status is ExecutionJobStatus.SUCCESS:
                status = ToolResultStatus.SUCCESS
            elif job.status is ExecutionJobStatus.PARTIAL_SUCCESS:
                status = ToolResultStatus.PARTIAL_SUCCESS
            elif job.status is ExecutionJobStatus.FAILED:
                status = ToolResultStatus.FAILED
            elif job.status is ExecutionJobStatus.UNKNOWN:
                status = ToolResultStatus.UNKNOWN
            elif job.status is ExecutionJobStatus.TIMED_OUT:
                status = ToolResultStatus.TIMEOUT
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                provider_name=provider_name if force_provider_name else job.provider_name,
                status=status,
                job_id=job.job_id,
                provider_job_id=job.provider_job_id,
                data={},
                target_results=job.target_results,
                provider_code=job.provider_code,
                provider_message=job.provider_message,
                raw_result=job.raw_result,
                execution_time_ms=execution_time_ms,
            )

    @staticmethod
    def _failure_result(
        *,
        call_id: str,
        tool_name: str,
        provider_name: str,
        status: ToolResultStatus,
        error_detail: str,
        execution_time_ms: int | None,
    ) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            provider_name=provider_name,
            status=status,
            error_detail=error_detail,
            execution_time_ms=execution_time_ms,
        )

    @staticmethod
    def _circuit_open_result(
        tool_name: str,
        *,
        provider_name: str,
        call_id: str,
    ) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            provider_name=provider_name,
            status=ToolResultStatus.CIRCUIT_OPEN,
            error_detail="circuit breaker is open",
        )

    async def _safe_audit_start(self, **kwargs: Any) -> bool:
        try:
            await self.audit_service.log_start(**kwargs)
            return True
        except Exception:  # noqa: BLE001 - audit must not block execution
            logger.exception("tool_call_log start failed for call_id=%s", kwargs.get("call_id"))
            return False

    async def _safe_audit_finish(
        self,
        *,
        call_id: str,
        result: ToolResult,
        retry_count: int,
        audit_started: bool,
    ) -> None:
        if not audit_started:
            return
        try:
            await self.audit_service.log_finish(
                call_id=call_id,
                status=result.status.value,
                result=result.model_dump(mode="json"),
                error_detail=result.error_detail,
                retry_count=retry_count,
            )
        except Exception:  # noqa: BLE001
            logger.exception("tool_call_log finish failed for call_id=%s", call_id)

    async def _safe_publish(
        self,
        event_id: str,
        message_type: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            await self.event_bus.publish_event(event_id, message_type, payload)
        except Exception:  # noqa: BLE001
            logger.exception(
                "event bus publish failed event_id=%s message_type=%s",
                event_id,
                message_type,
            )

    async def _finalize(
        self,
        *,
        call_id: str,
        event_id: str,
        result: ToolResult,
        retry_count: int,
        audit_started: bool,
    ) -> None:
        await self._safe_audit_finish(
            call_id=call_id,
            result=result,
            retry_count=retry_count,
            audit_started=audit_started,
        )
        await self._safe_publish(
            event_id,
            "tool_call_completed",
            {
                "call_id": call_id,
                "tool_name": result.tool_name,
                "status": result.status.value,
                "retry_count": retry_count,
            },
        )

    @staticmethod
    def _elapsed_ms(started_monotonic: float) -> int:
        return max(0, int((time.monotonic() - started_monotonic) * 1000))


tool_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    """FastAPI dependency returning the process executor singleton.

    Production wiring must replace ``NullAuditService`` with a real
    ``ToolCallLogService`` (and inject EventBus / job store as needed).
    """
    global tool_executor
    if tool_executor is None:
        from app.tools.registry import tool_registry

        tool_executor = ToolExecutor(registry=tool_registry)
    return tool_executor


__all__ = [
    "BudgetServicePort",
    "CallNature",
    "ConvergenceGuardPort",
    "ExecutionJobStorePort",
    "InMemoryExecutionJobStore",
    "NoopBudgetService",
    "NoopConvergenceGuard",
    "NullAuditService",
    "NullEventBus",
    "ToolExecutor",
    "derive_call_nature",
    "get_tool_executor",
    "tool_executor",
]
