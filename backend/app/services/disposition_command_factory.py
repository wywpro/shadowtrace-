"""Strict allowlisted disposition command assembly (ISSUE-059)."""

from __future__ import annotations

from typing import Any

from app.agents.response_agent import compute_source_locator_hash
from app.models.action import Action
from app.models.disposition import (
    DispositionCommand,
    RecordExecutionResultParams,
    SourceObjectLocator,
    SubmitEntityActionParams,
    TargetDispositionResult,
)
from app.models.enums import (
    DispositionIntentKind,
    ExecutionOwner,
    TargetExecutionStatus,
)
from app.models.execution import ActionExecutionJob


class DispositionCommandFactory:
    """Rebuild outbound commands from approved Action fields only.

    Never copies ``Action.reason``, free-form ``parameters``, or Provider
    ``raw_result`` into outbound payloads.
    """

    def build_entity_action_submit(
        self,
        action: Action,
        *,
        source_locator: SourceObjectLocator,
        source_concurrency_token: str | None,
        operator_id: str,
        disposition_id: str,
        writeback_id: str,
        closure_cycle: int,
        entity_action_code: str,
    ) -> DispositionCommand:
        canonical_target = action.target or ""
        return DispositionCommand(
            disposition_id=disposition_id,
            action_id=action.action_id,
            closure_cycle=closure_cycle,
            intent_kind=DispositionIntentKind.ENTITY_ACTION_SUBMIT,
            source_locator=source_locator,
            operation_code="submit_entity_action",
            operation_params=SubmitEntityActionParams(
                entity_action_code=entity_action_code,
                canonical_target=canonical_target,
            ),
            target_results=[
                TargetDispositionResult(
                    canonical_target=canonical_target,
                    status=TargetExecutionStatus.UNKNOWN,
                )
            ],
            operator_id=operator_id,
            idempotency_key=action.idempotency_key or f"{action.action_id}:entity",
            source_concurrency_token=source_concurrency_token,
            execution_owner=ExecutionOwner.XDR_MANAGED,
        )

    def build_execution_result_record(
        self,
        action: Action,
        job: ActionExecutionJob,
        *,
        source_locator: SourceObjectLocator,
        source_concurrency_token: str | None,
        operator_id: str,
        disposition_id: str,
        closure_cycle: int,
    ) -> DispositionCommand:
        target_results = [
            TargetDispositionResult(
                canonical_target=result.canonical_target,
                status=(
                    TargetExecutionStatus.SUCCESS
                    if result.status.value == "success"
                    else TargetExecutionStatus.FAILED
                ),
                provider_code=result.code,
                message_code=result.message,
                artifact_ref=result.artifact_id,
            )
            for result in job.target_results
        ]
        summary_code = _execution_summary_code(job)
        return DispositionCommand(
            disposition_id=disposition_id,
            action_id=action.action_id,
            closure_cycle=closure_cycle,
            intent_kind=DispositionIntentKind.EXECUTION_RESULT_RECORD,
            source_locator=source_locator,
            operation_code="record_execution_result",
            operation_params=RecordExecutionResultParams(summary_code=summary_code),
            target_results=target_results,
            operator_id=operator_id,
            idempotency_key=action.idempotency_key or f"{action.action_id}:result",
            source_concurrency_token=source_concurrency_token,
            execution_owner=ExecutionOwner.DIRECT_TOOL,
        )

    @staticmethod
    def locator_hash(locator: SourceObjectLocator) -> str:
        return compute_source_locator_hash(locator)


def _execution_summary_code(job: ActionExecutionJob) -> str:
    if job.status.value == "partial_success":
        return "partial_success"
    if job.status.value == "success":
        return "success"
    if job.status.value in {"failed", "timed_out", "cancelled"}:
        return "failed"
    return "unknown"


def entity_action_code_for(action: Action) -> str:
    """Map approved tool metadata to a stable Mock operation code."""
    mapping: dict[str, str] = {
        "block_ip": "block_ip",
        "block_domain": "block_domain",
        "block_process": "block_process",
        "isolate_host": "isolate_host",
    }
    return mapping.get(action.tool_name, action.tool_name)


__all__ = [
    "DispositionCommandFactory",
    "entity_action_code_for",
]
