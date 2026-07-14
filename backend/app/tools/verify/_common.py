"""Read-only execution path for Mock observation verification tools."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.models.enums import ExecutionJobStatus, TargetExecutionStatus
from app.models.execution import ActionExecutionJob, TargetExecutionResult
from app.models.ids import new_call_id
from app.models.source import SourceReference
from app.models.tool_meta import ToolMeta, ToolResult, ToolResultStatus
from app.tools.inputs import CheckStatusInput
from app.tools.mock_state import MockEnvironmentState, MockObservationRecord
from app.tools.specs import baseline_tool_index

VerificationMethod = Literal[
    "device_query",
    "endpoint_query",
    "telemetry_observation",
    "source_alert_delta",
]

PROVIDER_NAME = "mock_observation"

_TERMINAL_JOB_STATUSES = frozenset(
    {
        ExecutionJobStatus.PARTIAL_SUCCESS,
        ExecutionJobStatus.SUCCESS,
        ExecutionJobStatus.FAILED,
        ExecutionJobStatus.TIMED_OUT,
        ExecutionJobStatus.CANCELLED,
    }
)
_EFFECTFUL_JOB_STATUSES = frozenset(
    {
        ExecutionJobStatus.PARTIAL_SUCCESS,
        ExecutionJobStatus.SUCCESS,
    }
)


@dataclass(frozen=True, slots=True)
class VerificationSpec:
    surface: str
    method: VerificationMethod
    expected_statuses: frozenset[str]


VERIFICATION_SPECS: dict[str, VerificationSpec] = {
    "check_ip_block_status": VerificationSpec(
        "ip_blocks",
        "device_query",
        frozenset({"blocked"}),
    ),
    "check_domain_block_status": VerificationSpec(
        "domain_blocks",
        "device_query",
        frozenset({"blocked"}),
    ),
    "check_host_isolation_status": VerificationSpec(
        "host_isolation",
        "endpoint_query",
        frozenset({"isolated"}),
    ),
    "check_file_quarantine_status": VerificationSpec(
        "file_quarantine",
        "endpoint_query",
        frozenset({"quarantined"}),
    ),
    "check_process_block_status": VerificationSpec(
        "process_blocks",
        "endpoint_query",
        frozenset({"blocked"}),
    ),
    "check_virus_scan_status": VerificationSpec(
        "virus_scans",
        "endpoint_query",
        frozenset({"completed"}),
    ),
    "check_account_status": VerificationSpec(
        "account_status",
        "endpoint_query",
        frozenset({"disabled", "terminated", "password_reset", "revoked"}),
    ),
    "check_new_alerts": VerificationSpec(
        "new_alerts",
        "source_alert_delta",
        frozenset({"detected"}),
    ),
    "check_traffic_drop": VerificationSpec(
        "traffic",
        "telemetry_observation",
        frozenset({"dropped"}),
    ),
}


class VerificationData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_verified: bool
    detail: str
    verified_at: datetime
    verification_method: VerificationMethod
    observed_version: int | None
    source_refs: list[SourceReference]


def verification_output_schema() -> dict[str, Any]:
    schema = ToolResult.model_json_schema()
    data_schema = VerificationData.model_json_schema()
    data_definitions = data_schema.pop("$defs", {})
    schema.setdefault("$defs", {}).update(data_definitions)
    schema["properties"]["data"] = data_schema
    required = set(schema.get("required", []))
    required.add("data")
    schema["required"] = sorted(required)
    return schema


def verification_tool_meta(tool_name: str) -> ToolMeta:
    spec = VERIFICATION_SPECS[tool_name]
    return baseline_tool_index()[tool_name].model_copy(
        deep=True,
        update={
            "required_capabilities": [spec.method],
            "output_schema": verification_output_schema(),
        },
    )


class MockVerificationRuntime:
    """Wait for job completion, then read only the independent observation surface."""

    def __init__(
        self,
        state: MockEnvironmentState,
        *,
        wait_timeout_ms: int = 1_000,
        poll_interval_ms: int = 10,
    ) -> None:
        if wait_timeout_ms < 0:
            raise ValueError("wait_timeout_ms must be non-negative")
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be positive")
        self.state = state
        self.wait_timeout_ms = wait_timeout_ms
        self.poll_interval_ms = poll_interval_ms

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        meta = verification_tool_meta(tool_name)
        parsed = CheckStatusInput.model_validate(params)
        target_type = str(parsed.target_type)
        target = str(parsed.target)
        if target_type not in meta.target_types:
            raise ValueError(f"target_type must be one of {sorted(meta.target_types)!r}")
        options = dict(parsed.parameters)
        wait_timeout_ms = _integer_option(
            options,
            "wait_timeout_ms",
            self.wait_timeout_ms,
            minimum=0,
        )
        poll_interval_ms = _integer_option(
            options,
            "poll_interval_ms",
            self.poll_interval_ms,
            minimum=1,
        )
        job_id = options.get("job_id")
        if job_id is not None and (not isinstance(job_id, str) or not job_id):
            raise ValueError("parameters.job_id must be a non-empty string")

        job: ActionExecutionJob | None = None
        if isinstance(job_id, str):
            job = await self._wait_for_job(
                job_id,
                wait_timeout_ms=wait_timeout_ms,
                poll_interval_ms=poll_interval_ms,
            )
            if job is None:
                return self._result(
                    tool_name,
                    VerificationData(
                        is_verified=False,
                        detail="execution_job_not_found",
                        verified_at=datetime.now(UTC),
                        verification_method=VERIFICATION_SPECS[tool_name].method,
                        observed_version=None,
                        source_refs=[],
                    ),
                    status=ToolResultStatus.FAILED,
                )
            if job.status not in _TERMINAL_JOB_STATUSES:
                status = (
                    ToolResultStatus.UNKNOWN
                    if job.status is ExecutionJobStatus.UNKNOWN
                    else ToolResultStatus.TIMEOUT
                )
                return self._result(
                    tool_name,
                    VerificationData(
                        is_verified=False,
                        detail=f"execution_job_not_terminal:{job.status.value}",
                        verified_at=datetime.now(UTC),
                        verification_method=VERIFICATION_SPECS[tool_name].method,
                        observed_version=None,
                        source_refs=[],
                    ),
                    status=status,
                )
            if job.status not in _EFFECTFUL_JOB_STATUSES:
                return self._result(
                    tool_name,
                    VerificationData(
                        is_verified=False,
                        detail=f"execution_job_{job.status.value}",
                        verified_at=datetime.now(UTC),
                        verification_method=VERIFICATION_SPECS[tool_name].method,
                        observed_version=None,
                        source_refs=[],
                    ),
                )
            job_precondition = _job_precondition_detail(
                tool_name,
                target_type,
                target,
                job,
            )
            if job_precondition is not None:
                return self._result(
                    tool_name,
                    VerificationData(
                        is_verified=False,
                        detail=job_precondition,
                        verified_at=datetime.now(UTC),
                        verification_method=VERIFICATION_SPECS[tool_name].method,
                        observed_version=None,
                        source_refs=[],
                    ),
                )

        override = await self.state.get_verify_override(tool_name, target)
        if override is False:
            return self._result(
                tool_name,
                VerificationData(
                    is_verified=False,
                    detail="forced_failure_override",
                    verified_at=datetime.now(UTC),
                    verification_method=VERIFICATION_SPECS[tool_name].method,
                    observed_version=None,
                    source_refs=[],
                ),
            )

        allow_prior_observation = job is not None and _target_allows_prior_observation(
            tool_name,
            target_type,
            target,
            job,
        )
        observation_job_id = job.job_id if job is not None and not allow_prior_observation else None
        observation = await self._wait_for_observation(
            VERIFICATION_SPECS[tool_name].surface,
            target,
            job_id=observation_job_id,
            wait_for_copy=job is not None,
            wait_timeout_ms=wait_timeout_ms,
            poll_interval_ms=poll_interval_ms,
        )
        if observation is None and job is not None:
            pending = await self.state.get_observation(
                VERIFICATION_SPECS[tool_name].surface,
                target,
                include_pending=True,
                job_id=observation_job_id,
            )
            if pending is not None:
                return self._result(
                    tool_name,
                    VerificationData(
                        is_verified=False,
                        detail="observation_not_visible",
                        verified_at=datetime.now(UTC),
                        verification_method=VERIFICATION_SPECS[tool_name].method,
                        observed_version=None,
                        source_refs=[],
                    ),
                    status=ToolResultStatus.TIMEOUT,
                )
        expected = _expected_statuses(tool_name, options)
        observation_matches_job = (
            job is None
            or observation is None
            or observation.job_id == job.job_id
            or allow_prior_observation
        )
        is_verified = (
            observation is not None and observation_matches_job and observation.status in expected
        )
        detail = (
            "observation_job_mismatch"
            if observation is not None and not observation_matches_job
            else (
                f"observed_status:{observation.status}"
                if observation is not None
                else "observation_missing"
            )
        )
        return self._result(
            tool_name,
            VerificationData(
                is_verified=is_verified,
                detail=detail,
                verified_at=datetime.now(UTC),
                verification_method=VERIFICATION_SPECS[tool_name].method,
                observed_version=(
                    observation.observed_version if observation is not None else None
                ),
                source_refs=(observation.source_refs if observation is not None else []),
            ),
        )

    async def _wait_for_job(
        self,
        job_id: str,
        *,
        wait_timeout_ms: int,
        poll_interval_ms: int,
    ) -> ActionExecutionJob | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout_ms / 1_000
        while True:
            raw = await self.state.get_job(job_id)
            if raw is None:
                return None
            job = ActionExecutionJob.model_validate(raw)
            if job.status in _TERMINAL_JOB_STATUSES:
                return job
            if loop.time() >= deadline:
                return job
            await asyncio.sleep(poll_interval_ms / 1_000)

    async def _wait_for_observation(
        self,
        surface: str,
        target: str,
        *,
        job_id: str | None,
        wait_for_copy: bool,
        wait_timeout_ms: int,
        poll_interval_ms: int,
    ) -> MockObservationRecord | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout_ms / 1_000
        while True:
            record = await self.state.get_observation(
                surface,
                target,
                job_id=job_id,
            )
            if record is not None or not wait_for_copy or loop.time() >= deadline:
                return record
            await asyncio.sleep(poll_interval_ms / 1_000)

    @staticmethod
    def _result(
        tool_name: str,
        data: VerificationData,
        *,
        status: ToolResultStatus = ToolResultStatus.SUCCESS,
    ) -> dict[str, Any]:
        return ToolResult(
            call_id=new_call_id(),
            tool_name=tool_name,
            provider_name=PROVIDER_NAME,
            status=status,
            data=data.model_dump(mode="json"),
        ).model_dump(mode="json")


def _integer_option(
    options: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = options.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"parameters.{key} must be an integer >= {minimum}")
    return value


def _expected_statuses(tool_name: str, options: dict[str, Any]) -> frozenset[str]:
    expected = options.get("expected_status")
    allowed = VERIFICATION_SPECS[tool_name].expected_statuses
    if expected is None:
        return allowed
    if not isinstance(expected, str) or expected not in allowed:
        raise ValueError(f"parameters.expected_status must be one of {sorted(allowed)!r}")
    return frozenset({expected})


def _job_precondition_detail(
    tool_name: str,
    target_type: str,
    target: str,
    job: ActionExecutionJob,
) -> str | None:
    if tool_name == "check_new_alerts":
        return None if job.event_id == target else "execution_job_event_mismatch"
    if not job.target_results:
        return None
    matching = _matching_target_result(target_type, target, job)
    if matching is None:
        return "execution_job_target_mismatch"
    if matching.status is not TargetExecutionStatus.SUCCESS:
        return f"execution_target_not_success:{matching.status.value}"
    return None


def _matching_target_result(
    target_type: str,
    target: str,
    job: ActionExecutionJob,
) -> TargetExecutionResult | None:
    canonical_target = f"{target_type}:{target}"
    return next(
        (item for item in job.target_results if item.canonical_target == canonical_target),
        None,
    )


def _target_allows_prior_observation(
    tool_name: str,
    target_type: str,
    target: str,
    job: ActionExecutionJob,
) -> bool:
    if tool_name == "check_new_alerts":
        return False
    matching = _matching_target_result(target_type, target, job)
    return matching is not None and matching.code == "already_applied"


_runtime_override: contextvars.ContextVar[MockVerificationRuntime | None] = contextvars.ContextVar(
    "mock_verification_runtime_override", default=None
)
_default_runtime: MockVerificationRuntime | None = None


def get_mock_verification_runtime() -> MockVerificationRuntime:
    override = _runtime_override.get()
    if override is not None:
        return override
    from app.providers.tools.mock_provider import get_mock_tool_provider

    provider_state = get_mock_tool_provider().state
    global _default_runtime
    if _default_runtime is None or _default_runtime.state is not provider_state:
        _default_runtime = MockVerificationRuntime(provider_state)
    return _default_runtime


@contextmanager
def bind_mock_verification_runtime(
    runtime: MockVerificationRuntime,
) -> Iterator[None]:
    token = _runtime_override.set(runtime)
    try:
        yield
    finally:
        _runtime_override.reset(token)


async def execute_verification_tool(
    tool_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return await get_mock_verification_runtime().execute(tool_name, params)


__all__ = [
    "MockVerificationRuntime",
    "VerificationData",
    "VerificationMethod",
    "VERIFICATION_SPECS",
    "bind_mock_verification_runtime",
    "execute_verification_tool",
    "get_mock_verification_runtime",
    "verification_output_schema",
    "verification_tool_meta",
]
