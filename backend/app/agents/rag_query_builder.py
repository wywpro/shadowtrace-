"""RAGQueryBuilder: generate per-KB query strings from triage + evidence context."""

from __future__ import annotations

from app.models.agent_io import EvidenceOutput, TriageResult


class RAGQueryBuilder:
    """Build four knowledge-base queries from triage result and evidence output."""

    @staticmethod
    def build_queries(
        triage_result: TriageResult,
        evidence_output: EvidenceOutput | None = None,
    ) -> dict[str, str]:
        """Return ``{kb_name: query_string}`` for the four knowledge bases."""

        # attack_kb: 攻击技术查询拼证据行为摘要
        attack_parts: list[str] = [
            f"Event type: {triage_result.event_type.value}.",
            f"Alert severity: {triage_result.severity.value}.",
        ]
        if evidence_output and evidence_output.evidence_list:
            behaviors = [e.description for e in evidence_output.evidence_list if e.description]
            if behaviors:
                attack_parts.append("Behavior evidence: " + "; ".join(behaviors[:3]))
        attack_query = " ".join(attack_parts)

        # fp_case_kb: 误报查询拼告警特征
        fp_parts: list[str] = [
            f"False positive pattern for event type {triage_result.event_type.value},",
            f"severity {triage_result.severity.value}.",
        ]
        if triage_result.reasoning:
            fp_parts.append(f"Analysis: {triage_result.reasoning[:200]}")
        fp_query = " ".join(fp_parts)

        # history_case_kb: 案例查询拼事件类型与实体特征
        history_parts: list[str] = [
            f"Historical case with event type {triage_result.event_type.value}."
        ]
        entity_descs: list[str] = []
        for ip_e in triage_result.entities.ips[:5]:
            entity_descs.append(f"IP:{ip_e.address}")
        for host_e in triage_result.entities.hosts[:5]:
            entity_descs.append(f"Host:{host_e.hostname}")
        for proc_e in triage_result.entities.processes[:3]:
            entity_descs.append(f"Process:{proc_e.name}")
        if entity_descs:
            history_parts.append("Entities: " + ", ".join(entity_descs))
        history_query = " ".join(history_parts)

        # playbook_kb: 剧本查询拼事件类型与严重度
        playbook_query = (
            f"SOAR playbook for event type {triage_result.event_type.value}, "
            f"severity {triage_result.severity.value}."
        )

        return {
            "attack_kb": attack_query,
            "fp_case_kb": fp_query,
            "history_case_kb": history_query,
            "playbook_kb": playbook_query,
        }
