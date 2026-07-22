"""Deterministic 15-section report builder (ISSUE-036)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models.action import Action
from app.models.agent_io import (
    EvidenceOutput,
    ResponsePlan,
    RiskAssessment,
    TriageResult,
    VerificationResult,
)
from app.models.enums import ActionCategory, FinalVerdict, Severity
from app.models.report import ReportSection

PLACEHOLDER_NO_ACTIONS = "暂无处置动作"
PLACEHOLDER_NO_VERIFICATION = "暂无验证结果"
PLACEHOLDER_LOW_RISK_NO_EVIDENCE = "低危快结案：未执行证据采集"

SECTION_SPECS: tuple[tuple[str, str], ...] = (
    ("overview", "事件概述"),
    ("severity_level", "严重级别"),
    ("risk_scoring", "风险评分"),
    ("involved_accounts", "涉及账号"),
    ("involved_assets", "涉及资产"),
    ("involved_processes", "涉及进程"),
    ("involved_files", "涉及文件"),
    ("involved_external_addresses", "涉及外部地址"),
    ("evidence_chain", "证据链"),
    ("attack_storyline", "攻击故事线"),
    ("attack_mapping", "攻击映射"),
    ("executed_actions", "已执行处置"),
    ("verification_results", "验证结果"),
    ("recommendations", "处置建议"),
    ("appendix_index", "附录索引"),
)

SECTION_KEYS: tuple[str, ...] = tuple(key for key, _ in SECTION_SPECS)
SECTION_TITLES: dict[str, str] = dict(SECTION_SPECS)


def _fmt_ts(value: datetime | None) -> str:
    if value is None:
        return "unknown"
    return value.isoformat()


def _bullet(lines: list[str], empty: str) -> str:
    cleaned = [line.strip() for line in lines if line and str(line).strip()]
    if not cleaned:
        return empty
    return "\n".join(f"- {line}" for line in cleaned)


class ReportSectionBuilder:
    """Build the locked 15-section skeleton from EventContext facts."""

    def build(
        self,
        *,
        event_id: str,
        evidence_output: EvidenceOutput,
        risk_assessment: RiskAssessment,
        triage_result: TriageResult | None = None,
        response_plan: ResponsePlan | None = None,
        verification_result: VerificationResult | None = None,
        rag_output: dict[str, Any] | None = None,
        final_verdict: FinalVerdict = FinalVerdict.NONE,
        content_sha256: str | None = None,
    ) -> list[ReportSection]:
        # Prefer triage entities; otherwise derive labels from evidence raw/related.
        account_lines, asset_lines, process_lines, file_lines, external_lines = self._entity_lines(
            triage_result, evidence_output
        )
        response_actions = self._response_actions(response_plan)

        overview = self._overview(
            event_id=event_id,
            triage_result=triage_result,
            risk_assessment=risk_assessment,
            final_verdict=final_verdict,
            evidence_output=evidence_output,
        )
        severity_level = (
            f"severity={risk_assessment.severity.value}\n"
            f"risk_score={risk_assessment.risk_score}\n"
            f"confidence={risk_assessment.confidence:.4f}\n"
            f"possible_false_positive={risk_assessment.possible_false_positive}\n"
            f"scoring_mode={risk_assessment.scoring_mode.value}\n"
            f"final_verdict={final_verdict.value}"
        )
        risk_scoring = self._risk_scoring(risk_assessment)
        evidence_chain = self._evidence_chain(evidence_output)
        storyline = self._attack_storyline(evidence_output)
        attack_mapping = self._attack_mapping(evidence_output, rag_output)
        executed = self._executed_actions(response_actions)
        verification = self._verification_results(verification_result)
        recommendations = self._recommendations(
            risk_assessment=risk_assessment,
            response_actions=response_actions,
            final_verdict=final_verdict,
        )
        appendix = self._appendix(
            event_id=event_id,
            evidence_output=evidence_output,
            response_actions=response_actions,
            content_sha256=content_sha256,
        )

        contents: dict[str, str] = {
            "overview": overview,
            "severity_level": severity_level,
            "risk_scoring": risk_scoring,
            "involved_accounts": _bullet(account_lines, "暂无涉及账号"),
            "involved_assets": _bullet(asset_lines, "暂无涉及资产"),
            "involved_processes": _bullet(process_lines, "暂无涉及进程"),
            "involved_files": _bullet(file_lines, "暂无涉及文件"),
            "involved_external_addresses": _bullet(external_lines, "暂无涉及外部地址"),
            "evidence_chain": evidence_chain,
            "attack_storyline": storyline,
            "attack_mapping": attack_mapping,
            "executed_actions": executed,
            "verification_results": verification,
            "recommendations": recommendations,
            "appendix_index": appendix,
        }
        data_by_key: dict[str, dict[str, Any]] = {
            "risk_scoring": {
                "risk_score": risk_assessment.risk_score,
                "factors": [
                    {
                        "factor_name": f.factor_name,
                        "weight": f.weight,
                        "raw_score": f.raw_score,
                        "weighted_score": f.weighted_score,
                        "reasoning": f.reasoning,
                    }
                    for f in risk_assessment.risk_factors
                ],
            },
            "executed_actions": {
                "response_action_count": len(response_actions),
                "action_ids": [a.action_id for a in response_actions],
            },
            "verification_results": {
                "overall_status": (
                    verification_result.overall_status.value
                    if verification_result is not None
                    else None
                ),
            },
            "appendix_index": {
                "content_sha256": content_sha256,
                "evidence_count": len(evidence_output.evidence_list),
                "response_action_count": len(response_actions),
            },
        }

        sections: list[ReportSection] = []
        for key, title in SECTION_SPECS:
            sections.append(
                ReportSection(
                    key=key,
                    title=title,
                    content=contents[key],
                    data=data_by_key.get(key, {}),
                )
            )
        return sections

    def default_title(self, triage_result: TriageResult | None, event_id: str) -> str:
        if triage_result is not None:
            return f"调查报告 · {triage_result.event_type.value} · {event_id}"
        return f"调查报告 · {event_id}"

    def default_summary(
        self,
        *,
        risk_assessment: RiskAssessment,
        final_verdict: FinalVerdict,
        triage_result: TriageResult | None,
    ) -> str:
        event_type = triage_result.event_type.value if triage_result else "unknown"
        return (
            f"event_type={event_type}; severity={risk_assessment.severity.value}; "
            f"risk_score={risk_assessment.risk_score}; verdict={final_verdict.value}"
        )

    def _entity_lines(
        self,
        triage_result: TriageResult | None,
        evidence_output: EvidenceOutput,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
        accounts: list[str] = []
        assets: list[str] = []
        processes: list[str] = []
        files: list[str] = []
        externals: list[str] = []

        if triage_result is not None:
            for acc in triage_result.entities.accounts:
                accounts.append(acc.username or acc.entity_id)
            for host in triage_result.entities.hosts:
                label = host.hostname or host.ip or host.entity_id
                assets.append(label)
            for proc in triage_result.entities.processes:
                processes.append(proc.name or proc.entity_id)
            for file_ent in triage_result.entities.files:
                files.append(file_ent.path or file_ent.name or file_ent.entity_id)
            for ip in triage_result.entities.ips:
                if ip.scope == "external" or ip.scope == "unknown":
                    externals.append(ip.address or ip.entity_id)
            for domain in triage_result.entities.domains:
                externals.append(domain.fqdn or domain.entity_id)
            for ioc in triage_result.ioc_list:
                if ioc not in externals:
                    externals.append(ioc)

        # Supplement from evidence when triage entities are sparse.
        for item in evidence_output.evidence_list:
            raw = item.raw_data or {}
            if isinstance(raw, dict):
                if raw.get("account"):
                    accounts.append(str(raw["account"]))
                if raw.get("hostname"):
                    assets.append(str(raw["hostname"]))
                if raw.get("process_name") or raw.get("process"):
                    processes.append(str(raw.get("process_name") or raw.get("process")))
                if raw.get("file_path") or raw.get("path"):
                    files.append(str(raw.get("file_path") or raw.get("path")))
                if raw.get("dst_ip"):
                    externals.append(str(raw["dst_ip"]))
                if raw.get("indicator"):
                    externals.append(str(raw["indicator"]))
            for related in item.related_entities:
                text = str(related)
                if text.startswith("PC-") or "FIN" in text:
                    assets.append(text)

        def _uniq(values: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for value in values:
                key = value.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(key)
            return out

        return (
            _uniq(accounts),
            _uniq(assets),
            _uniq(processes),
            _uniq(files),
            _uniq(externals),
        )

    def _overview(
        self,
        *,
        event_id: str,
        triage_result: TriageResult | None,
        risk_assessment: RiskAssessment,
        final_verdict: FinalVerdict,
        evidence_output: EvidenceOutput,
    ) -> str:
        event_type = triage_result.event_type.value if triage_result else "unknown"
        reasoning = (triage_result.reasoning if triage_result else "") or ""
        lines = [
            f"event_id: {event_id}",
            f"event_type: {event_type}",
            f"severity: {risk_assessment.severity.value}",
            f"risk_score: {risk_assessment.risk_score}",
            f"final_verdict: {final_verdict.value}",
            f"evidence_count: {len(evidence_output.evidence_list)}",
            f"collection_status: {evidence_output.collection_status.value}",
        ]
        if reasoning:
            lines.append(f"triage_reasoning: {reasoning}")
        if risk_assessment.severity is Severity.LOW and not evidence_output.evidence_list:
            lines.append(PLACEHOLDER_LOW_RISK_NO_EVIDENCE)
        return "\n".join(lines)

    def _risk_scoring(self, risk_assessment: RiskAssessment) -> str:
        lines = [
            f"total_score={risk_assessment.risk_score}",
            f"severity={risk_assessment.severity.value}",
            f"scoring_mode={risk_assessment.scoring_mode.value}",
            "six_dimension_breakdown:",
        ]
        for factor in risk_assessment.risk_factors:
            lines.append(
                f"- {factor.factor_name}: raw={factor.raw_score:.1f} "
                f"weight={factor.weight:.2f} weighted={factor.weighted_score:.1f} "
                f"| {factor.reasoning}"
            )
        if len(risk_assessment.risk_factors) < 6:
            lines.append("- note: fewer than six factors present in assessment")
        return "\n".join(lines)

    def _evidence_chain(self, evidence_output: EvidenceOutput) -> str:
        if not evidence_output.evidence_list:
            return PLACEHOLDER_LOW_RISK_NO_EVIDENCE
        lines: list[str] = []
        # Stable sort: missing timestamps first, then chronological.
        ordered = sorted(
            evidence_output.evidence_list,
            key=lambda e: (
                e.timestamp is None,
                e.timestamp or datetime(1970, 1, 1),
            ),
        )
        for item in ordered:
            lines.append(
                f"{_fmt_ts(item.timestamp)} | {item.source.value} | "
                f"{item.evidence_type} | conf={item.confidence:.2f} | "
                f"{item.description}"
            )
        if evidence_output.conflicts:
            lines.append(f"conflicts={len(evidence_output.conflicts)}")
        if evidence_output.gaps:
            lines.append(f"gaps={len(evidence_output.gaps)}")
        return "\n".join(lines)

    def _attack_storyline(self, evidence_output: EvidenceOutput) -> str:
        """Fallback storyline from evidence timeline (StorylineService is post-report)."""
        if not evidence_output.evidence_list:
            return PLACEHOLDER_LOW_RISK_NO_EVIDENCE
        # Stable sort: missing timestamps first, then chronological.
        ordered = sorted(
            evidence_output.evidence_list,
            key=lambda e: (
                e.timestamp is None,
                e.timestamp or datetime(1970, 1, 1),
            ),
        )
        lines = ["证据时间线（StorylineService 后置，此处使用证据兜底）："]
        for idx, item in enumerate(ordered, start=1):
            tech = f" [{item.mitre_technique}]" if item.mitre_technique else ""
            lines.append(f"{idx}. {_fmt_ts(item.timestamp)} — {item.description}{tech}")
        return "\n".join(lines)

    def _attack_mapping(
        self,
        evidence_output: EvidenceOutput,
        rag_output: dict[str, Any] | None,
    ) -> str:
        techniques: list[str] = []
        for item in evidence_output.evidence_list:
            if item.mitre_technique:
                techniques.append(item.mitre_technique)
        if isinstance(rag_output, dict):
            for match in rag_output.get("attack_techniques") or []:
                if isinstance(match, dict) and match.get("technique_id"):
                    name = match.get("technique_name") or ""
                    techniques.append(f"{match['technique_id']} {name}".strip())
        techniques = list(dict.fromkeys(techniques))
        if not techniques:
            return "暂无 ATT&CK 技术映射"
        return _bullet(techniques, "暂无 ATT&CK 技术映射")

    def _response_actions(self, response_plan: ResponsePlan | None) -> list[Action]:
        if response_plan is None:
            return []
        # Count disposition by ActionCategory.RESPONSE — never hard-code tool names.
        return [
            action
            for action in response_plan.actions
            if action.action_category is ActionCategory.RESPONSE
        ]

    def _executed_actions(self, response_actions: list[Action]) -> str:
        if not response_actions:
            return PLACEHOLDER_NO_ACTIONS
        lines: list[str] = []
        for action in response_actions:
            wb = action.writeback_status.value if action.writeback_status is not None else "null"
            effect = action.effect_verification_status or "unset"
            lines.append(
                f"{action.action_id} | {action.action_name} | tool={action.tool_name} | "
                f"status={action.status.value} | effect_verification={effect} | "
                f"writeback_status={wb} | target={action.target or '-'}"
            )
        return "\n".join(lines)

    def _verification_results(self, verification_result: VerificationResult | None) -> str:
        if verification_result is None or not verification_result.results:
            return PLACEHOLDER_NO_VERIFICATION
        lines = [
            f"overall_status={verification_result.overall_status.value}",
            f"verification_phase={verification_result.verification_phase.value}",
        ]
        for item in verification_result.results:
            wb = item.writeback_status.value if item.writeback_status is not None else "null"
            receipt = ",".join(item.writeback_ids) if item.writeback_ids else "-"
            lines.append(
                f"{item.action_id} | effect={item.effect_status.value} | "
                f"writeback_status={wb} | readiness={item.writeback_readiness.value} | "
                f"receipt_refs={receipt} | detail={item.detail or '-'}"
            )
        return "\n".join(lines)

    def _recommendations(
        self,
        *,
        risk_assessment: RiskAssessment,
        response_actions: list[Action],
        final_verdict: FinalVerdict,
    ) -> str:
        tips: list[str] = []
        if risk_assessment.severity in {Severity.HIGH, Severity.CRITICAL}:
            tips.append("对高价值主机执行隔离或进程阻断，并复核外联阻断生效。")
            tips.append("冻结涉事账号会话并强制改密，排查横向移动痕迹。")
            tips.append("保全敏感文件访问与外传日志，评估数据泄露范围。")
        elif final_verdict in {
            FinalVerdict.FALSE_POSITIVE,
            FinalVerdict.POSSIBLE_FALSE_POSITIVE,
        }:
            tips.append("按误报案例沉淀规则，降低同类告警噪音。")
            tips.append("复核检测阈值与基线，避免重复误报。")
            tips.append("保留审计记录后关闭事件，并同步来源处置状态。")
        else:
            tips.append("持续观察账号与主机行为，补充缺失证据源。")
            tips.append("核对威胁情报命中与资产重要性后再决定升级。")
            tips.append("若风险上升，按 playbook 启动处置与写回闭环。")
        if not response_actions:
            tips.append("当前无 RESPONSE 处置动作；确认 disposition_policy 后再规划。")
        tips.append("报告仅存 ShadowTrace 本地，禁止写入 DispositionCommand。")
        # Keep 3–5 recommendations.
        return "\n".join(f"{idx}. {tip}" for idx, tip in enumerate(tips[:5], start=1))

    def _appendix(
        self,
        *,
        event_id: str,
        evidence_output: EvidenceOutput,
        response_actions: list[Action],
        content_sha256: str | None,
    ) -> str:
        lines = [
            f"event_id={event_id}",
            f"evidence_ids={','.join(e.evidence_id for e in evidence_output.evidence_list) or '-'}",
            f"response_action_ids={','.join(a.action_id for a in response_actions) or '-'}",
            f"success_sources={','.join(evidence_output.success_sources) or '-'}",
            f"failed_sources={','.join(evidence_output.failed_sources) or '-'}",
        ]
        if content_sha256:
            lines.append(f"content_sha256={content_sha256}")
        return "\n".join(lines)
