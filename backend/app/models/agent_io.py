"""Agent stage input/output schemas (ISSUE-005).

These models lock the data-passing contract between the 12 Agents named in
intro §4.4. Later Agent implementation Issues must not add or rename fields.
Nested structures reuse ISSUE-002 models (``Evidence``, ``Action``,
``EntitySet``, writeback enums) where applicable.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.action import Action
from app.models.entities import EntitySet
from app.models.enums import (
    EventStatus,
    EventType,
    FinalVerdict,
    QualityVerdict,
    Severity,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.evidence import Evidence, EvidenceConflict, EvidenceGap

# --------------------------------------------------------------------------- #
# Agent-IO-local enumerations (not part of intro §4.6 DECLARED_ENUMS)
# --------------------------------------------------------------------------- #


class CollectionStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL_DONE = "partial_done"
    DEGRADED = "degraded"
    FAILED = "failed"


class StorylineGeneratedBy(StrEnum):
    LLM = "llm"
    RULE = "rule"


class StorylinePhaseName(StrEnum):
    INITIAL_ACCESS = "initial_access"
    COLLECTION = "collection"
    STAGING = "staging"
    EXFILTRATION = "exfiltration"
    POST_ACTION = "post_action"


class ScoringMode(StrEnum):
    LLM_AND_RULE = "llm_and_rule"
    RULE_ONLY = "rule_only"


class ResponsePlanGeneratedBy(StrEnum):
    LLM = "llm"
    TEMPLATE = "template"


class EffectStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNVERIFIABLE = "unverifiable"


class VerificationOverallStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    WAITING = "waiting"
    MANUAL_RESOLUTION = "manual_resolution"


class VerificationPhase(StrEnum):
    EFFECT = "effect"
    DISPOSITION = "disposition"


class GraphRelationType(StrEnum):
    LOGGED_IN_FROM = "logged_in_from"
    LOGGED_IN_TO = "logged_in_to"
    EXECUTED = "executed"
    ACCESSED = "accessed"
    CONNECTED_TO = "connected_to"
    RESOLVED = "resolved"
    REQUESTED = "requested"
    UPLOADED_TO = "uploaded_to"


AgentName = Literal[
    "super_agent",
    "planner_agent",
    "triage_agent",
    "evidence_agent",
    "graph_agent",
    "rag_agent",
    "risk_agent",
    "response_agent",
    "verify_agent",
    "report_agent",
    "memory_agent",
    "tool_agent",
]


# --------------------------------------------------------------------------- #
# Triage
# --------------------------------------------------------------------------- #


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    severity: Severity
    need_investigation: bool
    entities: EntitySet = Field(default_factory=EntitySet)
    ioc_list: list[str] = Field(default_factory=list)
    reasoning: str = ""
    degraded: bool = False


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class EvidenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_list: list[Evidence] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    success_sources: list[str] = Field(default_factory=list)
    failed_sources: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    collection_status: CollectionStatus


# --------------------------------------------------------------------------- #
# Attack storyline
# --------------------------------------------------------------------------- #


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    description: str
    evidence_id: str
    technique_id: str | None = None
    severity_hint: Severity | None = None


class StorylinePhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_order: int
    phase_name: StorylinePhaseName
    tactic: str | None = None
    narrative: str = ""
    entries: list[TimelineEntry] = Field(default_factory=list)


class AttackStoryline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storyline_id: str
    event_id: str
    narrative_summary: str
    phases: list[StorylinePhase] = Field(default_factory=list)
    generated_by: StorylineGeneratedBy


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    event_id: str
    entity_type: str
    entity_value: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    event_id: str
    source_node_id: str
    target_node_id: str
    relation_type: GraphRelationType
    evidence_id: str
    occurred_at: datetime | None = None


class GraphOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    central_entities: list[str] = Field(default_factory=list)
    # Each candidate is a time-ordered chain of node_id values.
    attack_path_candidates: list[list[str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #


class AttackTechniqueMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technique_id: str
    technique_name: str
    tactics: list[str] = Field(default_factory=list)
    match_confidence: float = Field(ge=0.0, le=1.0)
    citation_id: str


class FpSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_case_id: str | None = None
    matched_pattern: str | None = None


class SimilarCaseSummary(BaseModel):
    """Compact HistoryCase digest for RAG similar_cases (full model lands later)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    event_type: EventType | None = None
    summary: str = ""
    final_verdict: FinalVerdict | None = None
    risk_score: int | None = Field(default=None, ge=0, le=100)
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chunk_id: str
    kb_name: str
    quoted_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class RAGOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack_techniques: list[AttackTechniqueMatch] = Field(default_factory=list)
    fp_similarity: FpSimilarity = Field(default_factory=FpSimilarity)
    similar_cases: list[SimilarCaseSummary] = Field(default_factory=list)
    playbook_refs: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    degraded: bool = False


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #


class RiskFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_name: str
    weight: float = Field(ge=0.0, le=1.0)
    raw_score: float
    weighted_score: float
    reasoning: str = ""


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_score: int = Field(ge=0, le=100)
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    possible_false_positive: bool = False
    scoring_mode: ScoringMode


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


class ResponsePlan(BaseModel):
    """Generated disposition plan.

    ``actions`` is a generation-time snapshot only. Approval / execution /
    verification stages must re-load each Action by ``action_id``; they must
    not rely on the embedded ``status`` field here.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    actions: list[Action] = Field(default_factory=list)
    strategy_summary: str = ""
    generated_by: ResponsePlanGeneratedBy


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


class VerificationActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    effect_status: EffectStatus
    writeback_required: bool
    writeback_readiness: WritebackReadiness
    # Only the eight WritebackStatus values; null when no command exists or
    # writeback is not required. Never use not_required/unsupported as status.
    writeback_status: WritebackStatus | None = None
    writeback_ids: list[str] = Field(default_factory=list)
    verification_action_id: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _writeback_fields_are_consistent(self) -> VerificationActionResult:
        if not self.writeback_required:
            if self.writeback_readiness is not WritebackReadiness.NOT_REQUIRED:
                raise ValueError(
                    "writeback_required=false requires writeback_readiness=NOT_REQUIRED"
                )
            if self.writeback_status is not None:
                raise ValueError("writeback_required=false requires writeback_status=null")
        elif self.writeback_readiness is WritebackReadiness.NOT_REQUIRED:
            # required must never be silently downgraded to "not required".
            raise ValueError("writeback_required=true forbids writeback_readiness=NOT_REQUIRED")
        return self


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[VerificationActionResult] = Field(default_factory=list)
    overall_status: VerificationOverallStatus
    failed_actions: list[str] = Field(default_factory=list)
    failed_writebacks: list[str] = Field(default_factory=list)
    blocked_writebacks: list[str] = Field(default_factory=list)
    need_action_replan: bool = False
    need_writeback_recovery: bool = False
    need_manual_resolution: bool = False
    verification_phase: VerificationPhase

    @model_validator(mode="after")
    def _deferred_skipped_not_in_failed_actions(self) -> VerificationResult:
        # Unactivated POST_VERIFY deferred Actions are skipped with a fixed detail
        # and must never be treated as effect failures.
        deferred = {
            item.action_id
            for item in self.results
            if item.effect_status is EffectStatus.SKIPPED
            and item.detail == "deferred_pending_activation"
        }
        leaked = deferred.intersection(self.failed_actions)
        if leaked:
            raise ValueError(
                "deferred skipped actions must not appear in failed_actions: "
                + ", ".join(sorted(leaked))
            )
        return self


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


class CaseRecordSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    event_id: str | None = None
    summary: str = ""
    archived: bool = False


class FpRuleCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_summary: str
    alert_signature: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_event_id: str
    pending_review: bool = True


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    entity_value: str
    event_id: str
    risk_score: int | None = Field(default=None, ge=0, le=100)
    behavior_tags: list[str] = Field(default_factory=list)


class MemoryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_records: list[CaseRecordSummary] = Field(default_factory=list)
    fp_rules: list[FpRuleCandidate] = Field(default_factory=list)
    profile_updates: list[ProfileUpdate] = Field(default_factory=list)
    sigma_drafts: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class PlanBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tool_calls: int = 30
    max_llm_calls: int = 20
    max_duration_s: int = 300


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_order: int
    step_goal: str
    assigned_agent: AgentName
    required_tools: list[str] = Field(default_factory=list)
    success_criteria: str = ""


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    event_id: str
    steps: list[PlanStep] = Field(default_factory=list)
    budget: PlanBudget = Field(default_factory=PlanBudget)
    revision: int = 0
    revise_reason: str | None = None
    degraded: bool = False


# --------------------------------------------------------------------------- #
# SuperAgent investigation aggregate
# --------------------------------------------------------------------------- #


class InvestigationResult(BaseModel):
    """Final investigation summary produced by SuperAgent.

    ``final_status=CLOSED`` is a *local* EventStatus only — it must never be
    interpreted as proof that an external XDR disposition completed. Use the
    writeback_* fields (and ``external_unsynced``) for external sync state.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    final_status: EventStatus
    final_verdict: FinalVerdict = FinalVerdict.NONE
    escalated: bool = False
    external_unsynced: bool = False
    report_id: str | None = None
    writeback_required: bool = False
    writeback_readiness: WritebackReadiness = WritebackReadiness.NOT_REQUIRED
    writeback_overall_status: WritebackStatus | None = None
    pending_writeback_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _writeback_null_when_not_required(self) -> InvestigationResult:
        if not self.writeback_required:
            if self.writeback_readiness is not WritebackReadiness.NOT_REQUIRED:
                raise ValueError(
                    "writeback_required=false requires writeback_readiness=NOT_REQUIRED"
                )
            if self.writeback_overall_status is not None:
                raise ValueError("writeback_required=false requires writeback_overall_status=null")
        elif self.writeback_readiness is WritebackReadiness.NOT_REQUIRED:
            raise ValueError("writeback_required=true forbids writeback_readiness=NOT_REQUIRED")
        return self


# --------------------------------------------------------------------------- #
# Agent inputs (ISSUE-094 §1)
#
# Each of the 12 Agents (intro §4.4) gets a dedicated, strictly-validated
# input model instead of the generic ``AgentInput(event_id, data: dict)``
# envelope. Fields carry the *typed* upstream stage output(s) that Agent
# consumes; ``extra="forbid"`` rejects unknown/typo'd fields so a caller can
# never smuggle an untyped payload through the inter-agent boundary.
# The base ``AgentInput`` contains only ``event_id``; BaseAgent rejects that
# base type at runtime and accepts only the dedicated class bound to its name.
# --------------------------------------------------------------------------- #


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str


class SuperAgentInput(AgentInput):
    """Top-level investigation kickoff — the only Agent that starts from a bare event."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    triggered_by: str = "ingestion"


class PlannerAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    triage_result: TriageResult | None = None
    previous_plan: ExecutionPlan | None = None
    revise_reason: str | None = None


class TriageAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    raw_event_summary: str = ""
    hint_entities: EntitySet = Field(default_factory=EntitySet)


class EvidenceAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    triage_result: TriageResult
    plan_step_goal: str = ""
    required_tools: list[str] = Field(default_factory=list)


class GraphAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    evidence_output: EvidenceOutput


class RAGAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    triage_result: TriageResult
    evidence_output: EvidenceOutput | None = None


class RiskAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    triage_result: TriageResult
    evidence_output: EvidenceOutput
    graph_output: GraphOutput | None = None
    rag_output: RAGOutput | None = None


class ResponseAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    risk_assessment: RiskAssessment
    evidence_output: EvidenceOutput | None = None


class VerifyAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    response_plan: ResponsePlan
    verification_phase: VerificationPhase


class ReportAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    evidence_output: EvidenceOutput
    risk_assessment: RiskAssessment
    response_plan: ResponsePlan | None = None
    verification_result: VerificationResult | None = None


class MemoryAgentInput(AgentInput):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    investigation_result: InvestigationResult


class ToolAgentInput(AgentInput):
    """ToolExecutor dispatch (ISSUE-006 owns actual execution).

    ``tool_params`` stays a dict because tool argument shapes are defined by
    the per-tool Pydantic schemas in ``app.tools.inputs``, not by this
    envelope — ToolAgent must validate ``tool_params`` against the named
    tool's own input model before dispatch.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    tool_name: str
    tool_params: dict[str, Any] = Field(default_factory=dict)
    action_id: str | None = None


# Mapping of the 12 Agents (intro §4.4) to their locked input model — mirrors
# the output-side mapping tests build against ``agent_io`` classes.
AGENT_INPUT_MODELS: dict[AgentName, type[AgentInput]] = {
    "super_agent": SuperAgentInput,
    "planner_agent": PlannerAgentInput,
    "triage_agent": TriageAgentInput,
    "evidence_agent": EvidenceAgentInput,
    "graph_agent": GraphAgentInput,
    "rag_agent": RAGAgentInput,
    "risk_agent": RiskAgentInput,
    "response_agent": ResponseAgentInput,
    "verify_agent": VerifyAgentInput,
    "report_agent": ReportAgentInput,
    "memory_agent": MemoryAgentInput,
    "tool_agent": ToolAgentInput,
}

AGENT_INPUT_BY_NAME = AGENT_INPUT_MODELS


# --------------------------------------------------------------------------- #
# Output quality evaluation (ISSUE-065)
# --------------------------------------------------------------------------- #


class OutputQualityScore(BaseModel):
    """Per-agent output quality score computed by OutputQualityEvaluator.

    Fields match intro §4.13 ``OutputQualityScore`` and §4.6 ``QualityVerdict``.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: QualityVerdict
    metrics: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evaluated_by: Literal["rule", "llm"]
