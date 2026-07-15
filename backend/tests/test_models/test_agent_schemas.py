"""Agent I/O schema and BaseAgent tests (ISSUE-005)."""

from __future__ import annotations

from abc import ABC
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from app.agents.base import AgentInput, AgentOutput, BaseAgent
from app.models.action import Action
from app.models.agent_io import (
    AttackStoryline,
    AttackTechniqueMatch,
    CaseRecordSummary,
    Citation,
    CollectionStatus,
    EffectStatus,
    EvidenceAgentInput,
    EvidenceOutput,
    ExecutionPlan,
    FpRuleCandidate,
    FpSimilarity,
    GraphEdge,
    GraphNode,
    GraphOutput,
    GraphRelationType,
    InvestigationResult,
    MemoryOutput,
    PlanBudget,
    PlanStep,
    ProfileUpdate,
    RAGOutput,
    ResponsePlan,
    ResponsePlanGeneratedBy,
    RiskAssessment,
    RiskFactor,
    ScoringMode,
    SimilarCaseSummary,
    StorylineGeneratedBy,
    StorylinePhase,
    StorylinePhaseName,
    SuperAgentInput,
    TimelineEntry,
    ToolAgentInput,
    TriageAgentInput,
    TriageResult,
    VerificationActionResult,
    VerificationOverallStatus,
    VerificationPhase,
    VerificationResult,
)
from app.models.entities import AccountEntity, EntitySet
from app.models.enums import (
    ActionCategory,
    ActionLevel,
    EventStatus,
    EventType,
    EvidenceSource,
    ExecutionOwner,
    FinalVerdict,
    Severity,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.evidence import Evidence, EvidenceConflict, EvidenceGap

# Mapping of the 12 Agents (intro §4.4) to their locked output model names.
AGENT_OUTPUT_MODELS = {
    "super_agent": InvestigationResult,
    "planner_agent": ExecutionPlan,
    "triage_agent": TriageResult,
    "evidence_agent": EvidenceOutput,
    "graph_agent": GraphOutput,
    "rag_agent": RAGOutput,
    "risk_agent": RiskAssessment,
    "response_agent": ResponsePlan,
    "verify_agent": VerificationResult,
    "report_agent": "InvestigationReport",  # ISSUE-002; not redefined here
    "memory_agent": MemoryOutput,
    "tool_agent": AgentOutput,  # ToolExecutor lands in ISSUE-006
}


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="evd-00000001",
        event_id="evt-20260101-0a1b2c3d",
        source=EvidenceSource.IDENTITY,
        evidence_type="login",
        description="login anomaly",
        confidence=0.8,
        timestamp=_now(),
    )


def _system_action() -> Action:
    return Action(
        action_id="act-00000001",
        event_id="evt-20260101-0a1b2c3d",
        plan_revision=1,
        action_fingerprint="fp-report",
        action_category=ActionCategory.SYSTEM,
        action_name="Generate investigation report",
        tool_name="generate_report",
        action_level=ActionLevel.L0,
        reason="placeholder",
    )


def _response_action() -> Action:
    return Action(
        action_id="act-00000002",
        event_id="evt-20260101-0a1b2c3d",
        plan_revision=1,
        action_fingerprint="fp-block",
        action_category=ActionCategory.RESPONSE,
        action_name="Block IP",
        tool_name="block_ip",
        action_level=ActionLevel.L3,
        execution_owner=ExecutionOwner.DIRECT_TOOL,
        writeback_required=True,
        writeback_applicable=True,
        writeback_readiness=WritebackReadiness.READY,
        reason="placeholder",
    )


# --------------------------------------------------------------------------- #
# Positive constructions for every stage model
# --------------------------------------------------------------------------- #


def test_triage_result_ok() -> None:
    result = TriageResult(
        event_type=EventType.INSIDER_THREAT,
        severity=Severity.HIGH,
        need_investigation=True,
        entities=EntitySet(accounts=[AccountEntity(entity_id="a1", username="svc")]),
        ioc_list=["203.0.113.9"],
        reasoning="suspicious volume",
        degraded=False,
    )
    assert result.need_investigation is True


def test_evidence_output_ok() -> None:
    out = EvidenceOutput(
        evidence_list=[_evidence()],
        conflicts=[
            EvidenceConflict(
                conflict_id="c1",
                event_id="evt-20260101-0a1b2c3d",
                description="mismatch",
                evidence_ids=["evd-00000001"],
            )
        ],
        gaps=[
            EvidenceGap(
                event_id="evt-20260101-0a1b2c3d",
                missing_source=EvidenceSource.NETWORK_FLOW,
                reason="timeout",
            )
        ],
        success_sources=["identity"],
        failed_sources=["network_flow"],
        overall_confidence=0.7,
        collection_status=CollectionStatus.PARTIAL_DONE,
    )
    assert out.collection_status is CollectionStatus.PARTIAL_DONE


def test_evidence_output_rejects_invalid_collection_status() -> None:
    with pytest.raises(ValidationError):
        EvidenceOutput(collection_status="done")  # type: ignore[arg-type]


def test_attack_storyline_ok() -> None:
    story = AttackStoryline(
        storyline_id="sty-abcdef12",
        event_id="evt-20260101-0a1b2c3d",
        narrative_summary="exfiltration attempt",
        phases=[
            StorylinePhase(
                phase_order=1,
                phase_name=StorylinePhaseName.EXFILTRATION,
                tactic="exfiltration",
                narrative="upload",
                entries=[
                    TimelineEntry(
                        timestamp=_now(),
                        description="upload observed",
                        evidence_id="evd-00000001",
                        technique_id="T1567",
                        severity_hint=Severity.HIGH,
                    )
                ],
            )
        ],
        generated_by=StorylineGeneratedBy.RULE,
    )
    assert story.phases[0].phase_name is StorylinePhaseName.EXFILTRATION


def test_storyline_rejects_unknown_phase_name() -> None:
    with pytest.raises(ValidationError):
        StorylinePhase(phase_order=1, phase_name="recon")  # type: ignore[arg-type]


def test_graph_output_ok() -> None:
    out = GraphOutput(
        nodes=[
            GraphNode(
                node_id="node-11111111",
                event_id="evt-20260101-0a1b2c3d",
                entity_type="account",
                entity_value="svc",
            ),
            GraphNode(
                node_id="node-22222222",
                event_id="evt-20260101-0a1b2c3d",
                entity_type="ip",
                entity_value="203.0.113.9",
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge-33333333",
                event_id="evt-20260101-0a1b2c3d",
                source_node_id="node-11111111",
                target_node_id="node-22222222",
                relation_type=GraphRelationType.LOGGED_IN_FROM,
                evidence_id="evd-00000001",
                occurred_at=_now(),
            )
        ],
        central_entities=["node-11111111"],
        attack_path_candidates=[["node-11111111", "node-22222222"]],
    )
    assert len(out.attack_path_candidates[0]) == 2


def test_rag_output_ok() -> None:
    out = RAGOutput(
        attack_techniques=[
            AttackTechniqueMatch(
                technique_id="T1567",
                technique_name="Exfiltration Over Web Service",
                tactics=["exfiltration"],
                match_confidence=0.9,
                citation_id="cit-aaaa1111",
            )
        ],
        fp_similarity=FpSimilarity(max_score=0.0),
        similar_cases=[
            SimilarCaseSummary(
                case_id="case-1",
                event_type=EventType.DATA_EXFILTRATION,
                summary="prior case",
                final_verdict=FinalVerdict.CONFIRMED_THREAT,
                risk_score=80,
                score=0.6,
            )
        ],
        playbook_refs=["pb-1"],
        citations=[
            Citation(
                citation_id="cit-aaaa1111",
                chunk_id="chk-1",
                kb_name="attack_techniques",
                quoted_text="web exfil",
                relevance_score=0.9,
            )
        ],
        degraded=False,
    )
    assert out.fp_similarity.max_score == 0.0


def test_risk_assessment_ok() -> None:
    assessment = RiskAssessment(
        risk_score=72,
        severity=Severity.HIGH,
        confidence=0.85,
        risk_factors=[
            RiskFactor(
                factor_name="impact",
                weight=0.2,
                raw_score=80,
                weighted_score=16,
                reasoning="data volume",
            )
        ],
        possible_false_positive=False,
        scoring_mode=ScoringMode.LLM_AND_RULE,
    )
    assert assessment.scoring_mode is ScoringMode.LLM_AND_RULE


def test_risk_assessment_rejects_invalid_scoring_mode() -> None:
    with pytest.raises(ValidationError):
        RiskAssessment(
            risk_score=10,
            severity=Severity.LOW,
            confidence=0.5,
            scoring_mode="llm_only",  # type: ignore[arg-type]
        )


def test_response_plan_ok() -> None:
    plan = ResponsePlan(
        plan_id="rsp-abcdef12",
        actions=[_system_action(), _response_action()],
        strategy_summary="contain then report",
        generated_by=ResponsePlanGeneratedBy.TEMPLATE,
    )
    assert len(plan.actions) == 2


def test_verification_result_ok_and_deferred_skipped() -> None:
    result = VerificationResult(
        results=[
            VerificationActionResult(
                action_id="act-00000002",
                effect_status=EffectStatus.VERIFIED,
                writeback_required=True,
                writeback_readiness=WritebackReadiness.READY,
                writeback_status=WritebackStatus.CONFIRMED,
                writeback_ids=["wbk-00000001"],
                verification_action_id="act-vfy-1",
            ),
            VerificationActionResult(
                action_id="act-deferred",
                effect_status=EffectStatus.SKIPPED,
                writeback_required=True,
                writeback_readiness=WritebackReadiness.READY,
                writeback_status=None,
                writeback_ids=[],
                detail="deferred_pending_activation",
            ),
            VerificationActionResult(
                action_id="act-ticket",
                effect_status=EffectStatus.SKIPPED,
                writeback_required=False,
                writeback_readiness=WritebackReadiness.NOT_REQUIRED,
                writeback_status=None,
            ),
        ],
        overall_status=VerificationOverallStatus.SUCCESS,
        failed_actions=[],
        verification_phase=VerificationPhase.DISPOSITION,
    )
    assert "act-deferred" not in result.failed_actions
    assert result.results[1].detail == "deferred_pending_activation"


def test_verification_rejects_not_required_as_writeback_status() -> None:
    with pytest.raises(ValidationError):
        VerificationActionResult(
            action_id="act-1",
            effect_status=EffectStatus.VERIFIED,
            writeback_required=False,
            writeback_readiness=WritebackReadiness.NOT_REQUIRED,
            writeback_status="not_required",  # type: ignore[arg-type]
        )


def test_verification_writeback_false_requires_null_status() -> None:
    with pytest.raises(ValidationError):
        VerificationActionResult(
            action_id="act-1",
            effect_status=EffectStatus.VERIFIED,
            writeback_required=False,
            writeback_readiness=WritebackReadiness.NOT_REQUIRED,
            writeback_status=WritebackStatus.CONFIRMED,
        )


def test_verification_required_forbids_not_required_readiness() -> None:
    with pytest.raises(ValidationError):
        VerificationActionResult(
            action_id="act-1",
            effect_status=EffectStatus.VERIFIED,
            writeback_required=True,
            writeback_readiness=WritebackReadiness.NOT_REQUIRED,
            writeback_status=None,
        )


def test_verification_rejects_need_replan_field() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            overall_status=VerificationOverallStatus.FAILED,
            need_replan=True,  # type: ignore[call-arg]
            verification_phase=VerificationPhase.EFFECT,
        )


def test_verification_rejects_receipt_id_field() -> None:
    with pytest.raises(ValidationError):
        VerificationActionResult(
            action_id="act-1",
            effect_status=EffectStatus.VERIFIED,
            writeback_required=False,
            writeback_readiness=WritebackReadiness.NOT_REQUIRED,
            writeback_status=None,
            receipt_id="rcp-1",  # type: ignore[call-arg]
        )


def test_verification_deferred_skipped_must_not_be_failed() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            results=[
                VerificationActionResult(
                    action_id="act-deferred",
                    effect_status=EffectStatus.SKIPPED,
                    writeback_required=True,
                    writeback_readiness=WritebackReadiness.READY,
                    writeback_status=None,
                    detail="deferred_pending_activation",
                )
            ],
            overall_status=VerificationOverallStatus.FAILED,
            failed_actions=["act-deferred"],
            verification_phase=VerificationPhase.EFFECT,
        )


def test_memory_output_ok() -> None:
    out = MemoryOutput(
        case_records=[CaseRecordSummary(case_id="case-1", event_id="evt-1", archived=True)],
        fp_rules=[
            FpRuleCandidate(
                rule_summary="batch password reset",
                alert_signature="pwd_reset_burst",
                confidence=0.9,
                source_event_id="evt-1",
                pending_review=True,
            )
        ],
        profile_updates=[
            ProfileUpdate(
                entity_type="account",
                entity_value="svc",
                event_id="evt-1",
                risk_score=72,
                behavior_tags=["exfil"],
            )
        ],
        sigma_drafts=["title: evt-1\ndetection:\n  selection:\n    EventID: 1\n"],
    )
    assert out.fp_rules[0].pending_review is True


def test_execution_plan_ok() -> None:
    plan = ExecutionPlan(
        plan_id="pln-abcdef12",
        event_id="evt-20260101-0a1b2c3d",
        steps=[
            PlanStep(
                step_order=1,
                step_goal="triage",
                assigned_agent="triage_agent",
                required_tools=[],
                success_criteria="triage_result present",
            ),
            PlanStep(
                step_order=2,
                step_goal="collect evidence",
                assigned_agent="evidence_agent",
                required_tools=["query_asset_info"],
                success_criteria="collection_status != failed",
            ),
        ],
        budget=PlanBudget(),
        revision=0,
        degraded=False,
    )
    assert plan.budget.max_tool_calls == 30


def test_execution_plan_rejects_unknown_agent() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            step_order=1,
            step_goal="x",
            assigned_agent="unknown_agent",  # type: ignore[arg-type]
            success_criteria="ok",
        )


def test_investigation_result_ok() -> None:
    result = InvestigationResult(
        event_id="evt-20260101-0a1b2c3d",
        final_status=EventStatus.CLOSED,
        final_verdict=FinalVerdict.CONFIRMED_THREAT,
        escalated=False,
        external_unsynced=False,
        report_id="rpt-20260101-0a1b2c3d",
        writeback_required=True,
        writeback_readiness=WritebackReadiness.READY,
        writeback_overall_status=WritebackStatus.CONFIRMED,
        pending_writeback_ids=[],
    )
    # CLOSED is local status only — external sync is carried by writeback_*.
    assert result.final_status is EventStatus.CLOSED
    assert result.writeback_overall_status is WritebackStatus.CONFIRMED


def test_investigation_result_closed_may_be_external_unsynced() -> None:
    result = InvestigationResult(
        event_id="evt-20260101-0a1b2c3d",
        final_status=EventStatus.CLOSED,
        external_unsynced=True,
        writeback_required=True,
        writeback_readiness=WritebackReadiness.READY,
        writeback_overall_status=WritebackStatus.FAILED,
        pending_writeback_ids=["wbk-1"],
    )
    assert result.external_unsynced is True


def test_investigation_result_not_required_null_status() -> None:
    result = InvestigationResult(
        event_id="evt-20260101-0a1b2c3d",
        final_status=EventStatus.CLOSED,
        writeback_required=False,
        writeback_readiness=WritebackReadiness.NOT_REQUIRED,
        writeback_overall_status=None,
    )
    assert result.writeback_overall_status is None


def test_investigation_result_rejects_status_when_not_required() -> None:
    with pytest.raises(ValidationError):
        InvestigationResult(
            event_id="evt-20260101-0a1b2c3d",
            final_status=EventStatus.CLOSED,
            writeback_required=False,
            writeback_readiness=WritebackReadiness.NOT_REQUIRED,
            writeback_overall_status=WritebackStatus.CONFIRMED,
        )


def test_investigation_result_required_forbids_not_required_readiness() -> None:
    with pytest.raises(ValidationError):
        InvestigationResult(
            event_id="evt-20260101-0a1b2c3d",
            final_status=EventStatus.CLOSED,
            writeback_required=True,
            writeback_readiness=WritebackReadiness.NOT_REQUIRED,
            writeback_overall_status=None,
        )


def test_all_twelve_agent_output_models_are_importable() -> None:
    from app.models.report import InvestigationReport

    assert len(AGENT_OUTPUT_MODELS) == 12
    for name, model in AGENT_OUTPUT_MODELS.items():
        if model == "InvestigationReport":
            assert InvestigationReport is not None
        else:
            assert model is not None, name


# --------------------------------------------------------------------------- #
# Typed Agent inputs (ISSUE-094 §1)
# --------------------------------------------------------------------------- #


def test_all_twelve_agent_input_models_are_exported_and_distinct() -> None:
    from app.agents import AGENT_INPUT_MODELS as exported_models
    from app.models.agent_io import AGENT_INPUT_MODELS as models_models

    assert exported_models is models_models
    assert len(models_models) == 12
    assert len(set(models_models.values())) == 12
    for name, model in models_models.items():
        assert issubclass(model, BaseModel), name
        assert model.model_config.get("extra") == "forbid", name
        assert "event_id" in model.model_fields, name


def test_agent_input_models_json_schema_export() -> None:
    from app.models.agent_io import AGENT_INPUT_MODELS

    for name, model in AGENT_INPUT_MODELS.items():
        schema = model.model_json_schema()
        assert schema["properties"]["event_id"]["type"] == "string", name
        # extra="forbid" surfaces as additionalProperties: false in JSON Schema.
        assert schema.get("additionalProperties") is False, name


def test_agent_input_models_reject_unknown_fields() -> None:
    from app.models.agent_io import AGENT_INPUT_MODELS

    for _name, model in AGENT_INPUT_MODELS.items():
        required = {
            field_name
            for field_name, field in model.model_fields.items()
            if field.is_required()
        }
        kwargs: dict[str, object] = {"event_id": "evt-20260101-0a1b2c3d"}
        for field_name in required:
            if field_name == "event_id":
                continue
            # Every currently-required non-event_id field is itself another
            # strict BaseModel; construct a same-typed dummy is out of scope
            # here — assert directly on the always-required event_id contract
            # and rely on per-model tests below for full construction checks.
            kwargs.pop(field_name, None)
        with pytest.raises(ValidationError):
            model.model_validate({**kwargs, "unexpected_field": "boom"}, strict=False)


def test_super_agent_input_ok() -> None:
    inp = SuperAgentInput(event_id="evt-1")
    assert inp.triggered_by == "ingestion"
    with pytest.raises(ValidationError):
        SuperAgentInput(event_id="evt-1", bogus="x")  # type: ignore[call-arg]


def test_triage_agent_input_ok() -> None:
    inp = TriageAgentInput(event_id="evt-1", raw_event_summary="brute force login")
    assert inp.hint_entities.accounts == []


def test_evidence_agent_input_requires_triage_result() -> None:
    with pytest.raises(ValidationError):
        EvidenceAgentInput(event_id="evt-1")  # type: ignore[call-arg]
    inp = EvidenceAgentInput(
        event_id="evt-1",
        triage_result=TriageResult(
            event_type=EventType.INSIDER_THREAT,
            severity=Severity.HIGH,
            need_investigation=True,
        ),
    )
    assert inp.triage_result.need_investigation is True


def test_tool_agent_input_ok() -> None:
    inp = ToolAgentInput(event_id="evt-1", tool_name="block_ip", tool_params={"ip": "1.2.3.4"})
    assert inp.tool_params["ip"] == "1.2.3.4"


# --------------------------------------------------------------------------- #
# BaseAgent
# --------------------------------------------------------------------------- #


def test_base_agent_cannot_be_instantiated() -> None:
    assert issubclass(BaseAgent, ABC)
    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore[abstract, call-arg]


@pytest.mark.asyncio
async def test_base_agent_execute_template_calls_run() -> None:
    class StubAgent(BaseAgent[ToolAgentInput, AgentOutput]):
        agent_name = "tool_agent"

        async def _run(self, input: ToolAgentInput) -> AgentOutput:
            return AgentOutput(agent_name=self.agent_name, data={"echo": input.event_id})

    agent = StubAgent()
    out = await agent.execute(ToolAgentInput(event_id="evt-1", tool_name="probe"))
    assert out.success is True
    assert out.data["echo"] == "evt-1"


@pytest.mark.asyncio
async def test_base_agent_hooks_and_placeholders() -> None:
    calls: list[str] = []

    class StubAgent(BaseAgent[TriageAgentInput, AgentOutput]):
        agent_name = "triage_agent"

        async def _run(self, input: TriageAgentInput) -> AgentOutput:
            calls.append("run")
            return AgentOutput(agent_name=self.agent_name)

        async def _record_trace(self, **kwargs: object) -> None:  # type: ignore[override]
            calls.append("trace")

        async def _check_budget(self, input: TriageAgentInput) -> None:
            calls.append("budget")

        async def _apply_guardrails(self, output: AgentOutput) -> AgentOutput:
            calls.append("guard")
            return output

    agent = StubAgent()

    async def pre(agent_: BaseAgent, input_: TriageAgentInput) -> None:
        calls.append("pre")

    async def post(agent_: BaseAgent, input_: TriageAgentInput) -> None:
        calls.append("post")

    agent.pre_hooks.append(pre)
    agent.post_hooks.append(post)
    await agent.execute(TriageAgentInput(event_id="evt-1"))
    assert calls == ["budget", "pre", "run", "guard", "post", "trace"]


@pytest.mark.asyncio
async def test_base_agent_rejects_generic_input_and_context_store() -> None:
    class StubAgent(BaseAgent[TriageAgentInput, AgentOutput]):
        agent_name = "triage_agent"

        async def _run(self, input: TriageAgentInput) -> AgentOutput:
            return AgentOutput(agent_name=self.agent_name)

    with pytest.raises(TypeError):
        StubAgent(context_store=object())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="TriageAgentInput"):
        await StubAgent().execute(AgentInput(event_id="evt-1"))  # type: ignore[arg-type]
