"""ReportAgent: 15-section investigation report (ISSUE-036).

Maps to the locked ``InvestigationReport`` model (ISSUE-002):
markdown is derived from section contents; JSON is the structured sections
payload; ``content_sha256`` is published on EventBus and stored in
``appendix_index.data`` (not a first-class model field).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.agents.base import BaseAgent
from app.agents.prompts.report_prompt import build_report_messages
from app.agents.report_section_builder import (
    SECTION_KEYS,
    SECTION_SPECS,
    SECTION_TITLES,
    ReportSectionBuilder,
)
from app.core.errors import LLMError
from app.models.agent_io import ReportAgentInput, TriageResult
from app.models.enums import (
    ActionCategory,
    ActionLevel,
    ActionStatus,
    FinalVerdict,
    WritebackReadiness,
)
from app.models.ids import new_action_id, report_id_for_event
from app.models.report import InvestigationReport, ReportSection

logger = logging.getLogger(__name__)

GENERATED_BY_LLM = "llm"
GENERATED_BY_TEMPLATE = "template"
LLM_TIMEOUT_SECONDS = 30.0
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class ReportAgent(BaseAgent[ReportAgentInput, InvestigationReport]):
    """Generate, persist, and publish a 15-section investigation report."""

    agent_name = "report_agent"

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        tool_executor: Any | None = None,
        working_memory: Any | None = None,
        budget_service: Any | None = None,
        output_guard: Any | None = None,
        trace_service: Any | None = None,
        audit_service: Any | None = None,
        event_bus: Any | None = None,
        event_service: Any | None = None,
        section_builder: ReportSectionBuilder | None = None,
        scenario_id: str | None = None,
        llm_timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            tool_executor=tool_executor,
            working_memory=working_memory,
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
            audit_service=audit_service,
            event_bus=event_bus,
        )
        self.event_service = event_service
        self.section_builder = section_builder or ReportSectionBuilder()
        self.scenario_id = scenario_id
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.last_content_sha256: str | None = None
        self.last_report_markdown: str | None = None
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def _run(self, input: ReportAgentInput) -> InvestigationReport:
        triage = await self._load_triage(input.event_id)
        rag = await self._read_optional(input.event_id, "rag_output")
        if not isinstance(rag, dict):
            rag = None
        final_verdict = await self._resolve_final_verdict(input.event_id)

        draft_sections = self.section_builder.build(
            event_id=input.event_id,
            evidence_output=input.evidence_output,
            risk_assessment=input.risk_assessment,
            triage_result=triage,
            response_plan=input.response_plan,
            verification_result=input.verification_result,
            rag_output=rag,
            final_verdict=final_verdict,
        )
        title = self.section_builder.default_title(triage, input.event_id)
        summary = self.section_builder.default_summary(
            risk_assessment=input.risk_assessment,
            final_verdict=final_verdict,
            triage_result=triage,
        )
        generated_by = GENERATED_BY_TEMPLATE

        if self.llm_client is not None:
            try:
                llm_title, llm_summary, llm_sections = await self._generate_with_llm(
                    input=input,
                    triage=triage,
                    draft_sections=draft_sections,
                    final_verdict=final_verdict,
                    rag=rag,
                )
                title = llm_title or title
                summary = llm_summary or summary
                draft_sections = self._merge_sections(draft_sections, llm_sections)
                generated_by = GENERATED_BY_LLM
            except Exception as exc:
                logger.warning(
                    "ReportAgent LLM path failed; using Jinja2 template event=%s err=%s",
                    input.event_id,
                    exc,
                )
                generated_by = GENERATED_BY_TEMPLATE

        # Hash canonical body (pre-fingerprint); final markdown includes appendix sha line.
        body_markdown = self._render_markdown(
            title=title,
            summary=summary,
            sections=draft_sections,
        )
        content_sha256 = hashlib.sha256(body_markdown.encode("utf-8")).hexdigest()
        self.last_content_sha256 = content_sha256
        sections = self._stamp_sha(draft_sections, content_sha256)
        self.last_report_markdown = self._render_markdown(
            title=title,
            summary=summary,
            sections=sections,
        )

        now = datetime.now(UTC)
        report = InvestigationReport(
            report_id=report_id_for_event(input.event_id),
            event_id=input.event_id,
            title=title,
            summary=summary,
            sections=sections,
            final_verdict=final_verdict,
            risk_score=input.risk_assessment.risk_score,
            severity=input.risk_assessment.severity,
            version=1,
            generated_by=generated_by,
            generated_at=now,
            updated_at=now,
        )

        await self._persist_report(report)
        await self._write_context(input.event_id, report)
        await self._record_generate_report_action(input)
        await self._publish_report_generated(report)
        return report

    async def _generate_with_llm(
        self,
        *,
        input: ReportAgentInput,
        triage: TriageResult | None,
        draft_sections: list[ReportSection],
        final_verdict: FinalVerdict,
        rag: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, str]]:
        assert self.llm_client is not None
        context_summary = self._context_summary(
            input=input,
            triage=triage,
            final_verdict=final_verdict,
            rag=rag,
        )
        draft_map = {section.key: section.content for section in draft_sections}
        messages = build_report_messages(
            event_id=input.event_id,
            context_summary=context_summary,
            draft_sections=draft_map,
        )
        response = await asyncio.wait_for(
            self.llm_client.chat(
                messages,
                event_id=input.event_id,
                agent_name=self.agent_name,
                prompt_key="report_generate",
                scenario_id=self.scenario_id,
                json_mode=True,
                max_tokens=8192,
            ),
            timeout=self.llm_timeout_seconds,
        )
        payload = response.parsed
        if payload is not None and hasattr(payload, "model_dump"):
            data = payload.model_dump(mode="json")
        else:
            data = json.loads(response.content)
        if not isinstance(data, dict):
            raise LLMError("report_generate LLM response is not an object")

        title = str(data.get("title") or "").strip()
        summary = str(data.get("summary") or "").strip()
        sections_raw = data.get("sections") or {}
        if not isinstance(sections_raw, dict):
            raise LLMError("report_generate sections must be an object")

        parsed: dict[str, str] = {}
        for key in SECTION_KEYS:
            value = sections_raw.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                parsed[key] = text
        if len(parsed) < len(SECTION_KEYS):
            raise LLMError(
                "report_generate LLM returned too few sections",
                details={"present": sorted(parsed), "required": len(SECTION_KEYS)},
            )
        return title, summary, parsed

    def _merge_sections(
        self,
        base: list[ReportSection],
        overrides: dict[str, str],
    ) -> list[ReportSection]:
        merged: list[ReportSection] = []
        for section in base:
            content = overrides.get(section.key) or section.content
            merged.append(
                ReportSection(
                    key=section.key,
                    title=section.title,
                    content=content,
                    data=dict(section.data),
                )
            )
        return merged

    def _stamp_sha(self, sections: list[ReportSection], content_sha256: str) -> list[ReportSection]:
        out: list[ReportSection] = []
        for section in sections:
            data = dict(section.data)
            if section.key == "appendix_index":
                data["content_sha256"] = content_sha256
                content = section.content
                if "content_sha256=" not in content:
                    content = f"{content}\ncontent_sha256={content_sha256}"
                else:
                    lines = [
                        line
                        for line in content.splitlines()
                        if not line.startswith("content_sha256=")
                    ]
                    lines.append(f"content_sha256={content_sha256}")
                    content = "\n".join(lines)
                out.append(
                    ReportSection(
                        key=section.key,
                        title=section.title,
                        content=content,
                        data=data,
                    )
                )
            else:
                out.append(section)
        return out

    def _render_markdown(
        self,
        *,
        title: str,
        summary: str,
        sections: list[ReportSection],
    ) -> str:
        template = self._jinja.get_template("report_template.md.j2")
        return (
            template.render(
                title=title,
                summary=summary,
                sections=[
                    {
                        "key": s.key,
                        "title": s.title,
                        "content": s.content,
                    }
                    for s in sections
                ],
            ).strip()
            + "\n"
        )

    def _context_summary(
        self,
        *,
        input: ReportAgentInput,
        triage: TriageResult | None,
        final_verdict: FinalVerdict,
        rag: dict[str, Any] | None,
    ) -> dict[str, Any]:
        evidence_sample = [
            {
                "source": item.source.value,
                "evidence_type": item.evidence_type,
                "description": item.description[:240],
                "confidence": item.confidence,
                "mitre_technique": item.mitre_technique,
            }
            for item in input.evidence_output.evidence_list[:20]
        ]
        response_actions = []
        if input.response_plan is not None:
            response_actions = [
                {
                    "action_id": a.action_id,
                    "tool_name": a.tool_name,
                    "status": a.status.value,
                    "target": a.target,
                }
                for a in input.response_plan.actions
                if a.action_category is ActionCategory.RESPONSE
            ]
        return {
            "event_id": input.event_id,
            "triage": (
                {
                    "event_type": triage.event_type.value,
                    "severity": triage.severity.value,
                    "ioc_list": list(triage.ioc_list),
                    "reasoning": triage.reasoning,
                    "accounts": [a.username for a in triage.entities.accounts],
                    "hosts": [h.hostname for h in triage.entities.hosts],
                    "external_ips": [
                        ip.address
                        for ip in triage.entities.ips
                        if ip.scope in {"external", "unknown"}
                    ],
                }
                if triage is not None
                else {}
            ),
            "risk": {
                "risk_score": input.risk_assessment.risk_score,
                "severity": input.risk_assessment.severity.value,
                "confidence": input.risk_assessment.confidence,
                "factors": [
                    {
                        "name": f.factor_name,
                        "raw_score": f.raw_score,
                        "weighted_score": f.weighted_score,
                        "reasoning": f.reasoning,
                    }
                    for f in input.risk_assessment.risk_factors
                ],
            },
            "final_verdict": final_verdict.value,
            "evidence_sample": evidence_sample,
            "response_actions": response_actions,
            "verification": (
                input.verification_result.model_dump(mode="json")
                if input.verification_result is not None
                else None
            ),
            "rag": rag or {},
        }

    async def _load_triage(self, event_id: str) -> TriageResult | None:
        raw = await self._read_optional(event_id, "triage_result")
        if raw is None:
            return None
        try:
            if isinstance(raw, TriageResult):
                return raw
            if isinstance(raw, dict):
                return TriageResult.model_validate(raw)
        except Exception:
            logger.debug("triage_result parse failed event=%s", event_id, exc_info=True)
        return None

    async def _resolve_final_verdict(self, event_id: str) -> FinalVerdict:
        if self.event_service is not None:
            getter = getattr(self.event_service, "get_event", None)
            if getter is not None:
                try:
                    event = await getter(event_id)
                    if event is not None:
                        verdict = getattr(event, "final_verdict", None)
                        if isinstance(verdict, FinalVerdict):
                            return verdict
                        if isinstance(verdict, str):
                            return FinalVerdict(verdict)
                except Exception:
                    logger.debug(
                        "get_event for verdict failed event=%s",
                        event_id,
                        exc_info=True,
                    )
            # Unit-test helpers may expose last verdict directly.
            stored = getattr(self.event_service, "final_verdicts", None)
            if isinstance(stored, dict) and event_id in stored:
                value = stored[event_id]
                if isinstance(value, FinalVerdict):
                    return value
                return FinalVerdict(str(value))
        return FinalVerdict.NONE

    async def _read_optional(self, event_id: str, key: str) -> Any:
        if self.working_memory is None:
            return None
        try:
            return await self.working_memory.read(event_id, key)
        except Exception:
            logger.debug("optional WM read failed key=%s", key, exc_info=True)
            return None

    async def _write_context(self, event_id: str, report: InvestigationReport) -> None:
        if self.working_memory is None:
            return
        try:
            await self.working_memory.write(
                event_id,
                "report",
                report.model_dump(mode="json"),
            )
        except Exception:
            logger.warning(
                "failed to write report to working memory event=%s",
                event_id,
                exc_info=True,
            )

    async def _persist_report(self, report: InvestigationReport) -> None:
        if self.event_service is None:
            return
        upsert = getattr(self.event_service, "upsert_report", None)
        if upsert is None:
            logger.debug("event_service lacks upsert_report; skip DB report sync")
            return
        try:
            persisted = await upsert(report)
            if isinstance(persisted, InvestigationReport):
                report.version = persisted.version
                report.updated_at = persisted.updated_at or report.updated_at
        except Exception:
            logger.warning(
                "failed to upsert report event=%s report_id=%s",
                report.event_id,
                report.report_id,
                exc_info=True,
            )
            raise
    async def _record_generate_report_action(self, input: ReportAgentInput) -> None:
        if self.event_service is None:
            return
        upsert = getattr(self.event_service, "upsert_generate_report_action", None)
        if upsert is None:
            logger.debug("event_service lacks upsert_generate_report_action; skip")
            return
        plan_revision = 1
        if input.response_plan is not None and input.response_plan.actions:
            plan_revision = max(a.plan_revision for a in input.response_plan.actions)
        try:
            await upsert(input.event_id, plan_revision=plan_revision)
        except Exception:
            logger.warning(
                "failed to upsert generate_report action event=%s",
                input.event_id,
                exc_info=True,
            )

    async def _publish_report_generated(
        self,
        report: InvestigationReport,
    ) -> None:
        if self.event_bus is None:
            return
        try:
            payload: dict[str, Any] = {
                "report_id": report.report_id,
                "sections": len(report.sections),
            }
            if report.generated_at is not None:
                payload["generated_at"] = report.generated_at.isoformat()
            await self.event_bus.publish_event(
                report.event_id,
                "report_generated",
                payload,
            )
        except Exception:
            logger.warning(
                "event_bus report_generated failed event=%s",
                report.event_id,
                exc_info=True,
            )


def generate_report_action_fingerprint(event_id: str, plan_revision: int) -> str:
    """Stable fingerprint for the system generate_report Action."""
    material = f"{event_id}|{plan_revision}|generate_report|system|system||immediate|"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_generate_report_action(
    *,
    event_id: str,
    plan_revision: int = 1,
    action_id: str | None = None,
    executed_at: datetime | None = None,
) -> dict[str, Any]:
    """ORM-friendly payload for the system generate_report Action.

    ISSUE-036 prose allows empty ``tool_name``; locked Action / agent schema
    tests use ``generate_report``. Empty string is avoided.
    """
    now = executed_at or datetime.now(UTC)
    return {
        "action_id": action_id or new_action_id(),
        "event_id": event_id,
        "plan_revision": plan_revision,
        "action_fingerprint": generate_report_action_fingerprint(event_id, plan_revision),
        "action_category": ActionCategory.SYSTEM.value,
        "action_name": "generate_report",
        "tool_name": "generate_report",
        "action_level": ActionLevel.L0.value,
        "target_type": "system",
        "target": "system",
        "parameters": {},
        "status": ActionStatus.SUCCESS.value,
        "auto_execute": True,
        "reason": "报告自动生成",
        "impact_assessment": None,
        "execution_owner": None,
        "writeback_required": False,
        "writeback_applicable": False,
        "writeback_readiness": WritebackReadiness.NOT_REQUIRED.value,
        "writeback_status": None,
        "executed_at": now,
        "source_action_id": None,
    }


__all__ = [
    "GENERATED_BY_LLM",
    "GENERATED_BY_TEMPLATE",
    "ReportAgent",
    "SECTION_KEYS",
    "SECTION_SPECS",
    "SECTION_TITLES",
    "build_generate_report_action",
    "generate_report_action_fingerprint",
]
