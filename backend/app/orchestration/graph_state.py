"""LangGraph investigation state (ISSUE-048)."""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


def _merge_trace(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Append node names while preserving execution order."""
    return [*(left or []), *(right or [])]


def _merge_flags(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Merge degraded flags without duplicating entries during replay."""
    return list(dict.fromkeys([*(left or []), *(right or [])]))


class InvestigationState(TypedDict, total=False):
    """Checkpoint-safe state for one investigation workflow."""

    event_id: str
    event_status: str
    disposition_policy: str
    severity: str
    final_verdict: str | None
    confidence: float
    need_investigation: bool | None
    triage_result: dict[str, Any] | None
    false_positive_match: dict[str, Any] | None
    source_snapshot: dict[str, Any] | None
    disposition_only_intent: bool
    execution_substate: str
    execution_plan: dict[str, Any] | None
    event_status_update_readiness: str
    degraded_flags: Annotated[list[str], _merge_flags]
    node_trace: Annotated[list[str], _merge_trace]
    halted: bool
    error: str | None
    verify_need_manual_resolution: bool
    verify_need_writeback_recovery: bool
    verify_need_action_replan: bool
    include_rag: bool
    evidence_output: dict[str, Any] | None
    rag_output: dict[str, Any] | None
    risk_assessment: dict[str, Any] | None
    report_generated: bool
    needs_approval_wait: bool
