"""Rule-based default investigation plans per EventType (ISSUE-049).

These are used when LLM is unavailable. Each plan provides a minimal but safe
investigation path that keeps the main pipeline running (降级策略).

Evidence tools align with ``EVIDENCE_QUERY_ORDER`` / ``QUERY_TOOL_METAS`` on main.
"""

from __future__ import annotations

from collections.abc import Callable

from app.agents.evidence_agent import EVIDENCE_QUERY_ORDER
from app.models.agent_io import ExecutionPlan, PlanBudget, PlanStep
from app.models.enums import EventType

PlanBuilder = Callable[[str, str, bool], ExecutionPlan]

# Canonical seven-source evidence queries (ISSUE-033).
SEVEN_EVIDENCE_TOOLS: list[str] = list(EVIDENCE_QUERY_ORDER)

MIN_PLAN_STEPS = 4


def _build_standard_plan(
    event_id: str,
    plan_id: str,
    evidence_steps: list[tuple[str, str, list[str], str]],
    *,
    include_rag: bool = False,
    include_graph: bool = False,
) -> ExecutionPlan:
    """Build a standard plan from evidence step specs plus risk + response."""
    steps: list[PlanStep] = []
    for idx, (goal, agent, tools, criteria) in enumerate(evidence_steps, start=1):
        steps.append(
            PlanStep(
                step_order=idx,
                step_goal=goal,
                assigned_agent=agent,  # type: ignore[arg-type]
                required_tools=tools,
                success_criteria=criteria,
            )
        )
    offset = len(steps)

    if include_graph:
        offset += 1
        steps.append(
            PlanStep(
                step_order=offset,
                step_goal="构建实体关系图：从证据中派生节点与边",
                assigned_agent="graph_agent",
                required_tools=[],
                success_criteria="产出至少3个节点和2条边",
            )
        )

    if include_rag:
        offset += 1
        steps.append(
            PlanStep(
                step_order=offset,
                step_goal="匹配ATT&CK技术与历史案例：检索知识库",
                assigned_agent="rag_agent",
                required_tools=[],
                success_criteria="返回至少1条技术匹配或相似案例",
            )
        )

    offset += 1
    steps.append(
        PlanStep(
            step_order=offset,
            step_goal="综合风险评估：基于全部证据计算风险分数",
            assigned_agent="risk_agent",
            required_tools=[],
            success_criteria="产出risk_score >= 0且包含至少3个风险因子",
        )
    )
    offset += 1
    steps.append(
        PlanStep(
            step_order=offset,
            step_goal="生成处置方案：根据风险等级规划响应动作",
            assigned_agent="response_agent",
            required_tools=[],
            success_criteria="产出至少1个处置Action",
        )
    )

    return ExecutionPlan(
        plan_id=plan_id,
        event_id=event_id,
        steps=steps,
        budget=PlanBudget(),
        revision=0,
        degraded=True,
    )


def _seven_query_evidence_steps() -> list[tuple[str, str, list[str], str]]:
    """Split the seven canonical queries across three evidence steps."""
    q = SEVEN_EVIDENCE_TOOLS
    return [
        (
            "身份与终端取证：登录、进程、文件访问",
            "evidence_agent",
            [q[0], q[1], q[2]],
            "获取账号登录、EDR进程与文件访问记录",
        ),
        (
            "网络与资产取证：流量、DNS、资产信息",
            "evidence_agent",
            [q[3], q[4], q[5]],
            "获取网络流量、DNS与资产信息",
        ),
        (
            "威胁情报查询",
            "evidence_agent",
            [q[6]],
            "获取IOC威胁情报数据",
        ),
    ]


# --------------------------------------------------------------------------- #
# Per-EventType default plans
# --------------------------------------------------------------------------- #


def _plan_data_exfiltration(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    """数据外泄：七路查询 + 可选 graph/rag"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=_seven_query_evidence_steps(),
        include_graph=True,
        include_rag=rag_enabled,
    )


def _plan_account_anomaly(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    q = SEVEN_EVIDENCE_TOOLS
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "账号活动审计",
                "evidence_agent",
                [q[0], q[2]],
                "获取账号登录与文件访问记录",
            ),
            (
                "终端与网络关联取证",
                "evidence_agent",
                [q[1], q[3], q[4]],
                "获取进程、网络与DNS记录",
            ),
            (
                "威胁情报验证",
                "evidence_agent",
                [q[5], q[6]],
                "获取资产与IOC情报",
            ),
        ],
        include_rag=rag_enabled,
    )


def _plan_host_compromise(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    q = SEVEN_EVIDENCE_TOOLS
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "终端深度取证",
                "evidence_agent",
                [q[1], q[2]],
                "获取进程树与文件事件",
            ),
            (
                "网络连接检测",
                "evidence_agent",
                [q[3], q[4]],
                "获取网络连接与DNS记录",
            ),
            (
                "身份与情报",
                "evidence_agent",
                [q[0], q[5], q[6]],
                "获取登录、资产与威胁情报",
            ),
        ],
        include_graph=True,
        include_rag=rag_enabled,
    )


def _plan_insider_threat(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    q = SEVEN_EVIDENCE_TOOLS
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "数据访问行为审计",
                "evidence_agent",
                [q[2], q[0]],
                "获取文件访问与账号登录记录",
            ),
            (
                "终端与网络操作取证",
                "evidence_agent",
                [q[1], q[3]],
                "获取进程与网络记录",
            ),
            (
                "资产与情报",
                "evidence_agent",
                [q[5], q[6]],
                "获取资产与威胁情报",
            ),
        ],
        include_rag=rag_enabled,
    )


def _plan_malicious_process(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    q = SEVEN_EVIDENCE_TOOLS
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "进程链完整取证",
                "evidence_agent",
                [q[1], q[2]],
                "获取进程树和文件操作",
            ),
            (
                "网络与DNS检测",
                "evidence_agent",
                [q[3], q[4]],
                "获取网络连接和DNS记录",
            ),
            (
                "威胁情报查询",
                "evidence_agent",
                [q[5], q[6]],
                "获取资产与IOC情报",
            ),
        ],
        include_rag=rag_enabled,
    )


def _plan_suspicious_domain(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    q = SEVEN_EVIDENCE_TOOLS
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "域名情报与DNS溯源",
                "evidence_agent",
                [q[4], q[6], q[5]],
                "获取DNS、威胁情报与资产数据",
            ),
            (
                "关联主机访问记录",
                "evidence_agent",
                [q[3], q[1]],
                "获取网络连接与进程记录",
            ),
        ],
        include_graph=True,
        include_rag=rag_enabled,
    )


def _plan_lateral_movement(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    q = SEVEN_EVIDENCE_TOOLS
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "横向移动与登录链分析",
                "evidence_agent",
                [q[0], q[3], q[1]],
                "获取登录、网络与进程记录",
            ),
            (
                "多主机取证与情报",
                "evidence_agent",
                [q[2], q[5], q[6]],
                "获取文件、资产与威胁情报",
            ),
        ],
        include_graph=True,
        include_rag=rag_enabled,
    )


def _plan_other(event_id: str, plan_id: str, rag_enabled: bool) -> ExecutionPlan:
    """通用/未知类型：保守只读调查模板"""
    q = SEVEN_EVIDENCE_TOOLS
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "基础威胁情报与DNS",
                "evidence_agent",
                [q[4], q[6]],
                "获取DNS与IOC基础情报",
            ),
            (
                "终端与网络基础信息",
                "evidence_agent",
                [q[1], q[3]],
                "获取进程和网络基础信息",
            ),
            (
                "账号活动基础审计",
                "evidence_agent",
                [q[0], q[2]],
                "获取账号登录和文件访问基础信息",
            ),
        ],
        include_rag=rag_enabled,
    )


# --------------------------------------------------------------------------- #
# Master lookup
# --------------------------------------------------------------------------- #

DEFAULT_PLANS: dict[EventType, PlanBuilder] = {
    EventType.DATA_EXFILTRATION: _plan_data_exfiltration,
    EventType.ACCOUNT_ANOMALY: _plan_account_anomaly,
    EventType.HOST_COMPROMISE: _plan_host_compromise,
    EventType.INSIDER_THREAT: _plan_insider_threat,
    EventType.MALICIOUS_PROCESS: _plan_malicious_process,
    EventType.SUSPICIOUS_DOMAIN: _plan_suspicious_domain,
    EventType.LATERAL_MOVEMENT: _plan_lateral_movement,
    EventType.OTHER: _plan_other,
}


def get_default_plan(
    event_id: str,
    event_type: EventType,
    plan_id: str,
    *,
    rag_enabled: bool = False,
) -> ExecutionPlan:
    """Return the rule-based default plan for the given event type."""
    builder = DEFAULT_PLANS[event_type]
    return builder(event_id, plan_id, rag_enabled)


def get_revised_default_plan(
    event_id: str,
    event_type: EventType,
    plan_id: str,
    revision: int,
    failure_reason: str,
    *,
    rag_enabled: bool = False,
) -> ExecutionPlan:
    """Deterministic revised plan when LLM revision fails.

    Prepends a full seven-query evidence sweep so step goals differ from the
    initial DEFAULT_PLANS output (acceptance criterion #2).
    """
    base = get_default_plan(event_id, event_type, plan_id, rag_enabled=rag_enabled)
    expansion = PlanStep(
        step_order=1,
        step_goal=f"Revised evidence sweep (rev {revision}): {failure_reason[:120]}",
        assigned_agent="evidence_agent",
        required_tools=list(SEVEN_EVIDENCE_TOOLS),
        success_criteria="Seven canonical query tools re-executed",
    )
    steps: list[PlanStep] = [expansion]
    order = 2
    for step in base.steps:
        if step.assigned_agent == "evidence_agent":
            continue
        steps.append(
            PlanStep(
                step_order=order,
                step_goal=step.step_goal,
                assigned_agent=step.assigned_agent,
                required_tools=list(step.required_tools),
                success_criteria=step.success_criteria,
            )
        )
        order += 1
    return ExecutionPlan(
        plan_id=plan_id,
        event_id=event_id,
        steps=steps,
        budget=PlanBudget(),
        revision=revision,
        revise_reason=failure_reason,
        degraded=True,
    )


__all__ = [
    "DEFAULT_PLANS",
    "MIN_PLAN_STEPS",
    "SEVEN_EVIDENCE_TOOLS",
    "get_default_plan",
    "get_revised_default_plan",
]
