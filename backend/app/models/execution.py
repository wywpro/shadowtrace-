"""Async execution models (intro §4.3 / ISSUE-002 field spec).

``ActionExecutionJob`` doubles as the pre-persisted dispatch intent for the
DIRECT_TOOL path: a QUEUED job must be committed transactionally BEFORE calling
the external Provider, and recovery must query the Provider by the stable
idempotency key rather than inferring "not executed" from a missing local result.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.disposition import WritebackSummary
from app.models.enums import (
    ActionExecutionPhase,
    ActionStatus,
    ExecutionJobStatus,
    TargetExecutionStatus,
    WritebackReadiness,
    WritebackStatus,
)


class TargetExecutionResult(BaseModel):
    """Per-target execution result kept on a job (internal; may retain raw_result)."""

    model_config = ConfigDict(extra="forbid")

    canonical_target: str
    status: TargetExecutionStatus
    code: str | None = None
    message: str | None = None
    artifact_id: str | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)


class ActionExecutionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    event_id: str
    action_id: str
    provider_name: str
    idempotency_key: str
    provider_job_id: str | None = None
    status: ExecutionJobStatus = ExecutionJobStatus.QUEUED
    claimed_by: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    poll_after_ms: int | None = None
    attempt: int = 0
    target_results: list[TargetExecutionResult] = Field(default_factory=list)
    provider_code: str | None = None
    provider_message: str | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)


class ExecutionActionView(BaseModel):
    """Per-action projection used inside ExecutionSummary."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    action_status: ActionStatus
    execution_phase: ActionExecutionPhase
    writeback_required: bool
    writeback_applicable: bool
    writeback_readiness: WritebackReadiness
    writeback_status: WritebackStatus | None = None
    target_results: list[TargetExecutionResult] = Field(default_factory=list)


class ExecutionSummary(BaseModel):
    """Single source of truth reused by API / Verify / UI / CLOSED gate / stats."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    plan_revision: int
    action_counts: dict[str, int] = Field(default_factory=dict)
    jobs: list[ActionExecutionJob] = Field(default_factory=list)
    actions: list[ExecutionActionView] = Field(default_factory=list)
    writeback_counts: dict[str, int] = Field(default_factory=dict)
    writeback_ids: list[str] = Field(default_factory=list)
    writeback_summary: WritebackSummary | None = None
    updated_at: datetime | None = None
