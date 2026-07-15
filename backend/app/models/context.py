"""EventContext: the working-memory aggregate for one investigation (intro §4.11).

The field set is fixed by the ISSUE-002 spec. Concrete models defined in this
issue are typed directly; Agent stage outputs (TriageResult, EvidenceOutput, ...)
are modeled in later issues and are declared here as structural placeholders
(``dict | None``) so this baseline pre-declares the full field set without pulling
in models that do not yet exist. The four state families (source / internal
orchestration / action effect / external writeback) are kept separate.

``disposition_only_intent`` is a boolean set by a trusted workflow service before
action generation; it must never be self-reported by API/LLM.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.action import ImpactAssessment
from app.models.disposition import (
    DispositionCommand,
    DispositionReceipt,
    WritebackSummary,
)
from app.models.enums import ExecutionSubstate
from app.models.execution import ActionExecutionJob, ExecutionSummary
from app.models.report import InvestigationReport
from app.models.security_event import EventSummary
from app.models.source import SourceObjectState


class EventContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # EventSummary (never the full SecurityEvent, ISSUE-094 §2): agents/API
    # consumers must only see the redacted projection, and the store always
    # validates through this type — no ``model_construct`` bypass.
    event: EventSummary | None = None

    # --- source state family ---
    source_snapshot: dict[str, Any] | None = None
    source_sync_state: SourceObjectState | None = None

    # --- internal orchestration / agent outputs (typed in later issues) ---
    triage_result: dict[str, Any] | None = None
    false_positive_match: dict[str, Any] | None = None
    evidence_output: dict[str, Any] | None = None
    storyline: dict[str, Any] | None = None
    graph_output: dict[str, Any] | None = None
    rag_output: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None
    response_plan: dict[str, Any] | None = None
    approval_records: list[dict[str, Any]] = Field(default_factory=list)
    disposition_only_intent: bool = False
    execution_substate: ExecutionSubstate = ExecutionSubstate.NONE

    # --- action effect family ---
    execution_summary: ExecutionSummary | None = None
    execution_jobs: list[ActionExecutionJob] = Field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    rollback_results: list[dict[str, Any]] = Field(default_factory=list)
    impact_assessments: list[ImpactAssessment] = Field(default_factory=list)
    report: InvestigationReport | None = None
    memory_output: dict[str, Any] | None = None

    # --- external writeback family ---
    disposition_commands: list[DispositionCommand] = Field(default_factory=list)
    disposition_receipts: list[DispositionReceipt] = Field(default_factory=list)
    writeback_summary: WritebackSummary | None = None

    # --- orchestration bookkeeping ---
    state_history: list[dict[str, Any]] = Field(default_factory=list)
    replan_count: int = 0
    budget_usage: dict[str, Any] = Field(default_factory=dict)
    guard_violations: list[dict[str, Any]] = Field(default_factory=list)
    convergence_state: dict[str, Any] | None = None
    quality_scores: list[dict[str, Any]] = Field(default_factory=list)
    scratchpad: list[dict[str, Any]] = Field(default_factory=list)
    degraded_flags: list[str] = Field(default_factory=list)
