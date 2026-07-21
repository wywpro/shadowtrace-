"""Rule-based default investigation plans per EventType (ISSUE-049).

These are used when LLM is unavailable. Each plan provides a minimal but safe
investigation path that keeps the main pipeline running (降级策略).
"""

from __future__ import annotations

from collections.abc import Callable

from app.models.agent_io import ExecutionPlan, PlanBudget, PlanStep
from app.models.enums import EventType

PlanBuilder = Callable[[str, str], ExecutionPlan]


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
                required_tools=["search_kb", "match_techniques"],
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


# --------------------------------------------------------------------------- #
# Per-EventType default plans
# --------------------------------------------------------------------------- #


def _plan_data_exfiltration(event_id: str, plan_id: str) -> ExecutionPlan:
    """数据外泄：七路查询覆盖IOC、终端、网络、身份、数据安全"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "威胁情报查询：IOC信誉与关联域名",
                "evidence_agent",
                ["query_threat_intel", "query_dns", "query_whois"],
                "获取IOC外部情报数据",
            ),
            (
                "终端进程与网络连接采集",
                "evidence_agent",
                ["query_process_tree", "query_network_connections"],
                "获取进程树和网络连接记录",
            ),
            (
                "文件访问与外传检测",
                "evidence_agent",
                ["query_file_events", "query_dlp_events"],
                "获取文件访问和外传检测记录",
            ),
            (
                "身份登录与账号活动审计",
                "evidence_agent",
                ["query_login_history", "query_account_activity"],
                "获取账号登录和活动记录",
            ),
            (
                "数据访问日志采集",
                "evidence_agent",
                ["query_data_access_logs"],
                "获取数据访问记录",
            ),
        ],
        include_graph=True,
        include_rag=True,
    )


def _plan_account_anomaly(event_id: str, plan_id: str) -> ExecutionPlan:
    """账号异常：重点身份溯源与登录审计"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "账号活动审计：登录历史、权限变更",
                "evidence_agent",
                ["query_login_history", "query_account_activity", "query_privilege_changes"],
                "获取账号完整活动记录",
            ),
            (
                "关联终端取证：登录来源主机进程与网络",
                "evidence_agent",
                ["query_process_tree", "query_network_connections"],
                "获取关联主机活动记录",
            ),
            (
                "威胁情报验证：关联IP/域名信誉",
                "evidence_agent",
                ["query_threat_intel", "query_dns"],
                "获取关联IOC情报",
            ),
        ],
    )


def _plan_host_compromise(event_id: str, plan_id: str) -> ExecutionPlan:
    """主机失陷：终端取证为主、网络与身份为辅"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "终端进程深度取证",
                "evidence_agent",
                ["query_process_tree", "query_file_events"],
                "获取进程树和文件事件",
            ),
            (
                "网络连接与横向移动检测",
                "evidence_agent",
                ["query_network_connections", "query_lateral_movement"],
                "获取网络连接和横向移动记录",
            ),
            (
                "登录来源与账号关联审计",
                "evidence_agent",
                ["query_login_history", "query_account_activity"],
                "获取关联账号活动",
            ),
            (
                "威胁情报查询：进程哈希与连接IP信誉",
                "evidence_agent",
                ["query_threat_intel", "query_dns"],
                "获取IOC情报",
            ),
        ],
        include_graph=True,
    )


def _plan_insider_threat(event_id: str, plan_id: str) -> ExecutionPlan:
    """内部威胁：数据访问与账号行为并重"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "数据访问行为审计",
                "evidence_agent",
                ["query_data_access_logs", "query_dlp_events", "query_file_events"],
                "获取数据访问和外传记录",
            ),
            (
                "账号活动与权限变更审计",
                "evidence_agent",
                ["query_account_activity", "query_privilege_changes", "query_login_history"],
                "获取账号活动和权限变更记录",
            ),
            (
                "终端操作取证",
                "evidence_agent",
                ["query_process_tree", "query_network_connections"],
                "获取终端操作记录",
            ),
        ],
    )


def _plan_malicious_process(event_id: str, plan_id: str) -> ExecutionPlan:
    """恶意进程：进程链分析与威胁情报"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "进程链完整取证",
                "evidence_agent",
                ["query_process_tree", "query_file_events"],
                "获取完整进程树和文件操作",
            ),
            (
                "网络连接与C2通信检测",
                "evidence_agent",
                ["query_network_connections", "query_dns"],
                "获取网络连接和DNS记录",
            ),
            (
                "威胁情报查询：进程哈希、IP、域名信誉",
                "evidence_agent",
                ["query_threat_intel", "query_whois"],
                "获取IOC情报",
            ),
        ],
        include_rag=True,
    )


def _plan_suspicious_domain(event_id: str, plan_id: str) -> ExecutionPlan:
    """可疑域名：DNS溯源与关联分析"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "域名情报查询",
                "evidence_agent",
                ["query_threat_intel", "query_dns", "query_whois", "query_passive_dns"],
                "获取域名注册、解析与情报数据",
            ),
            (
                "关联主机访问记录",
                "evidence_agent",
                ["query_network_connections", "query_process_tree"],
                "获取访问该域名的内网主机记录",
            ),
        ],
        include_graph=True,
    )


def _plan_lateral_movement(event_id: str, plan_id: str) -> ExecutionPlan:
    """横向移动：多主机关联与登录链分析"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "横向移动路径检测",
                "evidence_agent",
                ["query_lateral_movement", "query_login_history", "query_privilege_changes"],
                "获取横向移动和登录记录",
            ),
            (
                "多主机进程与网络取证",
                "evidence_agent",
                ["query_process_tree", "query_network_connections"],
                "获取多主机活动记录",
            ),
            (
                "威胁情报验证",
                "evidence_agent",
                ["query_threat_intel", "query_dns"],
                "获取IOC情报",
            ),
        ],
        include_graph=True,
        include_rag=True,
    )


def _plan_other(event_id: str, plan_id: str) -> ExecutionPlan:
    """通用/未知类型：保守只读调查模板"""
    return _build_standard_plan(
        event_id,
        plan_id,
        evidence_steps=[
            (
                "基础威胁情报查询",
                "evidence_agent",
                ["query_threat_intel", "query_dns"],
                "获取IOC基础情报",
            ),
            (
                "终端基础信息采集",
                "evidence_agent",
                ["query_process_tree", "query_network_connections"],
                "获取终端进程和网络基础信息",
            ),
            (
                "账号活动基础审计",
                "evidence_agent",
                ["query_login_history", "query_account_activity"],
                "获取账号登录和活动基础信息",
            ),
        ],
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

# --------------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------------- #


def get_default_plan(event_id: str, event_type: EventType, plan_id: str) -> ExecutionPlan:
    """Return the rule-based default plan for the given event type.

    Raises KeyError if the event_type has no registered default plan — this is
    intentional: any new EventType added without a default plan will cause a
    test failure until the gap is filled.
    """
    builder = DEFAULT_PLANS[event_type]
    return builder(event_id, plan_id)


__all__ = [
    "DEFAULT_PLANS",
    "get_default_plan",
]
