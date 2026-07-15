"""Core data models package (ISSUE-002).

``MODEL_REGISTRY`` maps model name -> Pydantic model class for every model whose
JSON Schema is exported to ``contracts/schemas/``. The schema-export test compares
the registry key set against the exported file set (no brittle fixed count).
"""

from __future__ import annotations

from pydantic import BaseModel

from app.models.action import Action, ImpactAssessment
from app.models.agent_io import (
    AgentInput,
    AttackStoryline,
    AttackTechniqueMatch,
    CaseRecordSummary,
    Citation,
    EvidenceAgentInput,
    EvidenceOutput,
    ExecutionPlan,
    FpRuleCandidate,
    FpSimilarity,
    GraphAgentInput,
    GraphEdge,
    GraphNode,
    GraphOutput,
    InvestigationResult,
    MemoryAgentInput,
    MemoryOutput,
    PlanBudget,
    PlannerAgentInput,
    PlanStep,
    ProfileUpdate,
    RAGAgentInput,
    RAGOutput,
    ReportAgentInput,
    ResponseAgentInput,
    ResponsePlan,
    RiskAgentInput,
    RiskAssessment,
    RiskFactor,
    SimilarCaseSummary,
    StorylinePhase,
    SuperAgentInput,
    TimelineEntry,
    ToolAgentInput,
    TriageAgentInput,
    TriageResult,
    VerificationActionResult,
    VerificationResult,
    VerifyAgentInput,
)
from app.models.context import EventContext
from app.models.disposition import (
    DispositionCommand,
    DispositionOutboxRecord,
    DispositionReceipt,
    RecordCompensationParams,
    RecordExecutionResultParams,
    SetEventDispositionParams,
    SourceObjectLocator,
    SubmitEntityActionParams,
    TargetDispositionResult,
    TargetWritebackResult,
    WritebackSummary,
)
from app.models.entities import (
    AccountEntity,
    DomainEntity,
    EntitySet,
    FileEntity,
    HostEntity,
    IPEntity,
    ProcessEntity,
)
from app.models.evidence import Evidence, EvidenceConflict, EvidenceGap
from app.models.execution import (
    ActionExecutionJob,
    ExecutionActionView,
    ExecutionSummary,
    TargetExecutionResult,
)
from app.models.report import InvestigationReport, ReportSection
from app.models.security_event import SecurityEvent
from app.models.source import (
    SourceAlert,
    SourceAsset,
    SourceConnector,
    SourceIncident,
    SourceLog,
    SourceObjectState,
    SourceReference,
)
from app.models.tool_meta import (
    CapabilityBindingEntry,
    CapabilityManifest,
    ProviderToolBinding,
    ToolMeta,
    ToolResult,
)

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    # entities
    "AccountEntity": AccountEntity,
    "HostEntity": HostEntity,
    "IPEntity": IPEntity,
    "DomainEntity": DomainEntity,
    "ProcessEntity": ProcessEntity,
    "FileEntity": FileEntity,
    "EntitySet": EntitySet,
    # source
    "SourceReference": SourceReference,
    "SourceObjectState": SourceObjectState,
    "SourceIncident": SourceIncident,
    "SourceAlert": SourceAlert,
    "SourceAsset": SourceAsset,
    "SourceLog": SourceLog,
    "SourceConnector": SourceConnector,
    # evidence
    "Evidence": Evidence,
    "EvidenceConflict": EvidenceConflict,
    "EvidenceGap": EvidenceGap,
    # execution
    "TargetExecutionResult": TargetExecutionResult,
    "ActionExecutionJob": ActionExecutionJob,
    "ExecutionActionView": ExecutionActionView,
    "ExecutionSummary": ExecutionSummary,
    # disposition
    "SourceObjectLocator": SourceObjectLocator,
    "SetEventDispositionParams": SetEventDispositionParams,
    "SubmitEntityActionParams": SubmitEntityActionParams,
    "RecordExecutionResultParams": RecordExecutionResultParams,
    "RecordCompensationParams": RecordCompensationParams,
    "TargetDispositionResult": TargetDispositionResult,
    "DispositionCommand": DispositionCommand,
    "TargetWritebackResult": TargetWritebackResult,
    "DispositionReceipt": DispositionReceipt,
    "DispositionOutboxRecord": DispositionOutboxRecord,
    "WritebackSummary": WritebackSummary,
    # action
    "ImpactAssessment": ImpactAssessment,
    "Action": Action,
    # report
    "ReportSection": ReportSection,
    "InvestigationReport": InvestigationReport,
    # security event + context
    "SecurityEvent": SecurityEvent,
    "EventContext": EventContext,
    # agent stage I/O (ISSUE-005)
    "AgentInput": AgentInput,
    "SuperAgentInput": SuperAgentInput,
    "PlannerAgentInput": PlannerAgentInput,
    "TriageAgentInput": TriageAgentInput,
    "EvidenceAgentInput": EvidenceAgentInput,
    "GraphAgentInput": GraphAgentInput,
    "RAGAgentInput": RAGAgentInput,
    "RiskAgentInput": RiskAgentInput,
    "ResponseAgentInput": ResponseAgentInput,
    "VerifyAgentInput": VerifyAgentInput,
    "ReportAgentInput": ReportAgentInput,
    "MemoryAgentInput": MemoryAgentInput,
    "ToolAgentInput": ToolAgentInput,
    "TriageResult": TriageResult,
    "EvidenceOutput": EvidenceOutput,
    "TimelineEntry": TimelineEntry,
    "StorylinePhase": StorylinePhase,
    "AttackStoryline": AttackStoryline,
    "GraphNode": GraphNode,
    "GraphEdge": GraphEdge,
    "GraphOutput": GraphOutput,
    "AttackTechniqueMatch": AttackTechniqueMatch,
    "FpSimilarity": FpSimilarity,
    "SimilarCaseSummary": SimilarCaseSummary,
    "Citation": Citation,
    "RAGOutput": RAGOutput,
    "RiskFactor": RiskFactor,
    "RiskAssessment": RiskAssessment,
    "ResponsePlan": ResponsePlan,
    "VerificationActionResult": VerificationActionResult,
    "VerificationResult": VerificationResult,
    "CaseRecordSummary": CaseRecordSummary,
    "FpRuleCandidate": FpRuleCandidate,
    "ProfileUpdate": ProfileUpdate,
    "MemoryOutput": MemoryOutput,
    "PlanBudget": PlanBudget,
    "PlanStep": PlanStep,
    "ExecutionPlan": ExecutionPlan,
    "InvestigationResult": InvestigationResult,
    # tool contracts (ISSUE-006)
    "ToolMeta": ToolMeta,
    "ToolResult": ToolResult,
    "ProviderToolBinding": ProviderToolBinding,
    "CapabilityBindingEntry": CapabilityBindingEntry,
    "CapabilityManifest": CapabilityManifest,
}

__all__ = ["MODEL_REGISTRY", *sorted(MODEL_REGISTRY.keys())]
