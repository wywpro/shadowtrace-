"""StorylineService: attack storyline generation (ISSUE-051).

Dual-path: LLM (JSON mode) with rule-based fallback.  Consumes EventContext
fields (evidence_output, rag_output, graph_output) and writes the generated
``AttackStoryline`` to the ``storyline`` field via WorkingMemory.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from app.agents.prompts.storyline_prompt import build_storyline_messages
from app.core.errors import ShadowTraceError

# LLMError / LLMProviderError are runtime-importable from app.core.llm.base
# even though only LLMProviderError is listed in __all__.  Catch Exception
# broadly so the fallback path runs on any LLM failure.
from app.models.agent_io import (
    AttackStoryline,
    StorylineGeneratedBy,
    StorylinePhase,
    StorylinePhaseName,
    TimelineEntry,
)
from app.models.enums import Severity
from app.models.ids import new_storyline_id

logger = logging.getLogger(__name__)

_TS_MIN = datetime.min.replace(tzinfo=UTC)

_PHASE_ORDER: dict[StorylinePhaseName, int] = {
    StorylinePhaseName.INITIAL_ACCESS: 1,
    StorylinePhaseName.COLLECTION: 2,
    StorylinePhaseName.STAGING: 3,
    StorylinePhaseName.EXFILTRATION: 4,
    StorylinePhaseName.POST_ACTION: 5,
}

# Evidence type keywords → phase bucket
_PHASE_KEYWORDS: dict[StorylinePhaseName, list[str]] = {
    StorylinePhaseName.INITIAL_ACCESS: [
        "login",
        "logon",
        "authentication",
        "credential",
        "登录",
    ],
    StorylinePhaseName.COLLECTION: [
        "file_access",
        "file_read",
        "access",
        "枚举",
        "browse",
        "directory",
        "search",
        "query",
        "read",
    ],
    StorylinePhaseName.STAGING: [
        "archive",
        "compress",
        "encrypt",
        "rar",
        "zip",
        "7z",
        "打包",
        "压缩",
        "加密",
        "staging",
    ],
    StorylinePhaseName.EXFILTRATION: [
        "upload",
        "exfil",
        "dns",
        "network_flow",
        "connection",
        "outbound",
        "外传",
        "上传",
        "resolve",
        "connect",
    ],
    StorylinePhaseName.POST_ACTION: [
        "cleanup",
        "delete",
        "remove",
        "清除",
        "删除",
    ],
}


class StorylineService:
    """Attack storyline generator.

    Tries the LLM path first (JSON-mode, prompt_key ``storyline_generate``).
    On any LLM error falls back to a deterministic rule-based pipeline that
    sorts evidence, buckets it into five phases, and generates narrative
    from templates.
    """

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        working_memory: Any | None = None,
    ) -> None:
        self._llm_client = llm_client
        if working_memory is not None:
            self._bound_wm = working_memory.for_writer("StorylineService")
        else:
            self._bound_wm = None
        self.last_degraded_reason: str | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def generate(self, event_context: dict[str, Any]) -> AttackStoryline:
        """Generate an attack storyline from full EventContext.

        Returns an ``AttackStoryline`` whose ``evidence_id`` values all
        reference real evidence records in the input.
        """
        event_id = _resolve_event_id(event_context)
        evidence_list = _extract_evidence(event_context)
        technique_matches = _extract_techniques(event_context)
        graph_paths = _extract_graph_paths(event_context)
        entity_names = _extract_entity_names(event_context)

        # --- LLM path ---
        if self._llm_client is not None and evidence_list:
            try:
                storyline = await self._generate_llm(
                    event_id=event_id,
                    evidence_list=evidence_list,
                    technique_matches=technique_matches,
                    graph_paths=graph_paths,
                    entity_names=entity_names,
                )
                if storyline is not None:
                    await self._write(event_id, storyline)
                    return storyline
            except Exception as exc:
                logger.warning(
                    "StorylineService LLM path failed for event=%s: %s",
                    event_id,
                    exc,
                )

        # --- Rule fallback ---
        storyline = self._generate_rule(
            event_id=event_id,
            evidence_list=evidence_list,
            technique_matches=technique_matches,
        )
        await self._write(event_id, storyline)
        return storyline

    # ------------------------------------------------------------------ #
    # LLM path
    # ------------------------------------------------------------------ #

    async def _generate_llm(
        self,
        *,
        event_id: str,
        evidence_list: list[dict[str, Any]],
        technique_matches: list[dict[str, Any]],
        graph_paths: list[list[str]],
        entity_names: list[str],
    ) -> AttackStoryline | None:
        """Call LLM, validate evidence_ids, backfill technique_ids, return."""
        if self._llm_client is None:
            return None
        evidence_entries = _build_evidence_entries(evidence_list)
        messages = build_storyline_messages(
            evidence_entries=evidence_entries,
            technique_matches=technique_matches[:10],
            graph_paths=graph_paths[:3],
            entity_names=entity_names[:10],
        )
        response = await self._llm_client.chat(
            messages,
            event_id=event_id,
            agent_name="storyline_service",
            prompt_key="storyline_generate",
            json_mode=True,
        )
        import json as _json

        # Parse from response.content (str) — never rely on response.parsed
        # because MockLLMClient returns a raw dict when json_mode=True without
        # a response_model, and LLMResponse.parsed is typed BaseModel | None.
        if isinstance(response.content, str):
            try:
                llm_data: Any = _json.loads(response.content)
            except (_json.JSONDecodeError, TypeError):
                return None
            if not isinstance(llm_data, dict):
                return None
        elif isinstance(response.content, dict):
            llm_data = response.content
        else:
            return None

        narrative = str(llm_data.get("narrative_summary", ""))[:300]
        raw_phases: list[dict[str, Any]] = llm_data.get("phases") or []
        if not isinstance(raw_phases, list):
            raw_phases = []

        # Build evidence_id lookup
        valid_evidence_ids: set[str] = {e.get("evidence_id", "") for e in evidence_list}

        phases: list[StorylinePhase] = []
        for rp in raw_phases:
            if not isinstance(rp, dict):
                continue
            phase_name_str = str(rp.get("phase_name", ""))
            phase_name = _parse_phase_name(phase_name_str)
            if phase_name is None:
                logger.warning(
                    "LLM returned unrecognized phase_name=%r for event=%s, skipping phase",
                    phase_name_str,
                    event_id,
                )
                continue

            entries: list[TimelineEntry] = []
            raw_entries: list[dict[str, Any]] = rp.get("entries") or []
            if not isinstance(raw_entries, list):
                raw_entries = []
            for re_entry in raw_entries:
                if not isinstance(re_entry, dict):
                    continue
                evidence_id = str(re_entry.get("evidence_id", ""))
                # Remove entries whose evidence_id doesn't exist in input (spec: 剔除)
                if not evidence_id or evidence_id not in valid_evidence_ids:
                    continue
                entries.append(
                    TimelineEntry(
                        timestamp=_parse_ts(re_entry.get("timestamp")) or _TS_MIN,
                        description=str(re_entry.get("description", ""))[:500],
                        evidence_id=evidence_id,
                        technique_id=re_entry.get("technique_id"),
                        severity_hint=_parse_severity(re_entry.get("severity_hint")),
                    )
                )

            if not entries:
                continue

            tactic = rp.get("tactic")
            phases.append(
                StorylinePhase(
                    phase_order=_PHASE_ORDER.get(phase_name, len(phases) + 1),
                    phase_name=phase_name,
                    tactic=str(tactic) if tactic else None,
                    narrative=str(rp.get("narrative", ""))[:500],
                    entries=entries,
                )
            )

        if not phases:
            return None  # signal fallback

        # Backfill technique_id from RAGOutput matches
        _backfill_technique_ids(phases, technique_matches)

        return AttackStoryline(
            storyline_id=new_storyline_id(),
            event_id=event_id,
            narrative_summary=narrative,
            phases=phases,
            generated_by=StorylineGeneratedBy.LLM,
        )

    # ------------------------------------------------------------------ #
    # Rule path
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_rule(
        *,
        event_id: str,
        evidence_list: list[dict[str, Any]],
        technique_matches: list[dict[str, Any]],
    ) -> AttackStoryline:
        """Deterministic rule-based storyline from evidence alone."""
        # Sort by timestamp
        sorted_evidence = sorted(
            evidence_list,
            key=lambda e: _parse_ts(e.get("timestamp")) or _TS_MIN,
        )

        if not sorted_evidence:
            return AttackStoryline(
                storyline_id=new_storyline_id(),
                event_id=event_id,
                narrative_summary="无足够证据构建攻击链。",
                phases=[],
                generated_by=StorylineGeneratedBy.RULE,
            )

        if len(sorted_evidence) < 3:
            scarce_phases = _build_scarce_single_phase(sorted_evidence)
            _backfill_technique_ids(scarce_phases, technique_matches)
            return AttackStoryline(
                storyline_id=new_storyline_id(),
                event_id=event_id,
                narrative_summary=_summary_for_phases(scarce_phases),
                phases=scarce_phases,
                generated_by=StorylineGeneratedBy.RULE,
            )

        # Bucket evidence into phases
        phase_buckets: dict[StorylinePhaseName, list[dict[str, Any]]] = defaultdict(list)
        for ev in sorted_evidence:
            phase = _bucket_evidence(ev)
            phase_buckets[phase].append(ev)

        storyline_phases: list[StorylinePhase] = []
        for phase_name in StorylinePhaseName:
            bucket = phase_buckets.get(phase_name, [])
            if not bucket:
                continue
            entries = [
                TimelineEntry(
                    timestamp=_parse_ts(e.get("timestamp")) or _TS_MIN,
                    description=e.get("description", "")[:500],
                    evidence_id=e.get("evidence_id", ""),
                )
                for e in bucket
            ]
            storyline_phases.append(
                StorylinePhase(
                    phase_order=_PHASE_ORDER[phase_name],
                    phase_name=phase_name,
                    narrative=_template_narrative(phase_name, bucket),
                    entries=entries,
                )
            )

        # Defence-in-depth: under current _bucket_evidence (POST_ACTION fallback)
        # this branch is unreachable; kept as a safety net for future refactors.
        if not storyline_phases and sorted_evidence:
            entries = [
                TimelineEntry(
                    timestamp=_parse_ts(e.get("timestamp")) or _TS_MIN,
                    description=e.get("description", "")[:500],
                    evidence_id=e.get("evidence_id", ""),
                )
                for e in sorted_evidence
            ]
            storyline_phases = [
                StorylinePhase(
                    phase_order=5,
                    phase_name=StorylinePhaseName.POST_ACTION,
                    narrative="未分类证据汇总",
                    entries=entries,
                )
            ]

        # Backfill technique_ids
        _backfill_technique_ids(storyline_phases, technique_matches)

        return AttackStoryline(
            storyline_id=new_storyline_id(),
            event_id=event_id,
            narrative_summary=_summary_for_phases(storyline_phases),
            phases=storyline_phases,
            generated_by=StorylineGeneratedBy.RULE,
        )

    # ------------------------------------------------------------------ #
    # WorkingMemory
    # ------------------------------------------------------------------ #

    async def _write(self, event_id: str, storyline: AttackStoryline) -> None:
        if self._bound_wm is None:
            return
        try:
            await self._bound_wm.write(
                event_id,
                "storyline",
                storyline.model_dump(mode="json"),
            )
        except Exception as exc:
            logger.warning("StorylineService WM write failed event=%s", event_id, exc_info=True)
            await self._mark_degraded(
                event_id,
                reason=f"storyline_write_failed: {exc}",
            )

    async def _mark_degraded(self, event_id: str, *, reason: str) -> None:
        """Best-effort degraded marker when storyline WM write fails."""
        self.last_degraded_reason = reason
        if self._bound_wm is None:
            return
        try:
            await self._bound_wm.write(
                event_id,
                "storyline_degraded",
                {
                    "degraded": True,
                    "reason": reason,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except ShadowTraceError:
            logger.exception(
                "Failed to persist storyline_degraded flag for event=%s",
                event_id,
            )


# ====================================================================== #
# Extraction helpers
# ====================================================================== #


def _resolve_event_id(event_context: dict[str, Any]) -> str:
    event_summary = event_context.get("event")
    if isinstance(event_summary, dict):
        return str(event_summary.get("event_id", ""))
    return str(event_context.get("event_id", ""))


def _extract_evidence(event_context: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_output = event_context.get("evidence_output")
    if isinstance(evidence_output, dict):
        evidence_list = evidence_output.get("evidence_list")
        if isinstance(evidence_list, list):
            return [e for e in evidence_list if isinstance(e, dict)]
    return []


def _extract_techniques(event_context: dict[str, Any]) -> list[dict[str, Any]]:
    rag_output = event_context.get("rag_output")
    if isinstance(rag_output, dict):
        techniques = rag_output.get("attack_techniques")
        if isinstance(techniques, list):
            return [t for t in techniques if isinstance(t, dict)]
    return []


def _extract_graph_paths(event_context: dict[str, Any]) -> list[list[str]]:
    graph_output = event_context.get("graph_output")
    if isinstance(graph_output, dict):
        paths = graph_output.get("attack_path_candidates")
        if isinstance(paths, list):
            return [[str(n) for n in p] for p in paths if isinstance(p, list)]
    return []


def _extract_entity_names(event_context: dict[str, Any]) -> list[str]:
    graph_output = event_context.get("graph_output")
    if isinstance(graph_output, dict):
        entities = graph_output.get("central_entities")
        if isinstance(entities, list):
            return [str(e) for e in entities]
    return []


# ====================================================================== #
# LLM helpers
# ====================================================================== #


def _build_evidence_entries(
    evidence_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for ev in evidence_list:
        entries.append(
            {
                "evidence_id": ev.get("evidence_id", ""),
                "source": ev.get("source", ""),
                "evidence_type": ev.get("evidence_type", ""),
                "description": ev.get("description", ""),
                "confidence": ev.get("confidence", 0),
                "timestamp": ev.get("timestamp"),
            }
        )
    return entries


# ====================================================================== #
# Rule helpers
# ====================================================================== #


def _build_scarce_single_phase(
    sorted_evidence: list[dict[str, Any]],
) -> list[StorylinePhase]:
    """Merge fewer than three evidence records into a single phase timeline."""
    entries = [
        TimelineEntry(
            timestamp=_parse_ts(e.get("timestamp")) or _TS_MIN,
            description=str(e.get("description", ""))[:500],
            evidence_id=str(e.get("evidence_id", "")),
        )
        for e in sorted_evidence
    ]
    return [
        StorylinePhase(
            phase_order=1,
            phase_name=StorylinePhaseName.POST_ACTION,
            narrative="证据数量有限，以下活动合并为单阶段时间线。",
            entries=entries,
        )
    ]


def _bucket_evidence(evidence: dict[str, Any]) -> StorylinePhaseName:
    """Assign evidence to a phase based on source, type and description keywords.

    Source-level checks run first to prevent keyword collisions (e.g. ``dns``
    source must be EXFILTRATION even when evidence_type contains ``query``).
    """
    source = str(evidence.get("source", "")).lower()
    evidence_type = str(evidence.get("evidence_type", "")).lower()
    description = str(evidence.get("description", "")).lower()
    combined = f"{source} {evidence_type} {description}"

    # Source-level priority: dns / network_flow → EXFILTRATION
    if source in ("dns", "network_flow"):
        return StorylinePhaseName.EXFILTRATION

    # Source-level: identity → INITIAL_ACCESS
    if source == "identity":
        return StorylinePhaseName.INITIAL_ACCESS

    for phase_name in (
        StorylinePhaseName.COLLECTION,
        StorylinePhaseName.STAGING,
        StorylinePhaseName.EXFILTRATION,
        StorylinePhaseName.INITIAL_ACCESS,
    ):
        keywords = _PHASE_KEYWORDS.get(phase_name, [])
        for kw in keywords:
            if kw.lower() in combined:
                return phase_name

    return StorylinePhaseName.POST_ACTION


def _template_narrative(
    phase_name: StorylinePhaseName,
    evidence_list: list[dict[str, Any]],
) -> str:
    """Generate a short Chinese narrative for a phase from evidence."""
    entity_hints = _collect_entity_hints(evidence_list)
    entity_text = "、".join(entity_hints[:3]) if entity_hints else "未知实体"

    templates: dict[StorylinePhaseName, str] = {
        StorylinePhaseName.INITIAL_ACCESS: (
            f"攻击者通过 {entity_text} 获得初始访问权限，涉及 {len(evidence_list)} 条证据记录。"
        ),
        StorylinePhaseName.COLLECTION: (
            f"在 {entity_text} 上检测到数据收集活动，涉及 {len(evidence_list)} 条证据记录。"
        ),
        StorylinePhaseName.STAGING: (
            f"在 {entity_text} 上发现数据暂存或打包行为，涉及 {len(evidence_list)} 条证据记录。"
        ),
        StorylinePhaseName.EXFILTRATION: (
            f"检测到数据经 {entity_text} 向外传输，涉及 {len(evidence_list)} 条证据记录。"
        ),
        StorylinePhaseName.POST_ACTION: (
            f"后续活动涉及 {entity_text}，共 {len(evidence_list)} 条证据记录。"
        ),
    }
    return templates.get(phase_name, f"包含 {len(evidence_list)} 条证据记录。")


def _collect_entity_hints(evidence_list: list[dict[str, Any]]) -> list[str]:
    """Extract human-readable entity hints from evidence descriptions."""
    hints: list[str] = []
    for ev in evidence_list:
        desc = str(ev.get("description", ""))
        # Quick extraction: values after common patterns in descriptions
        for prefix in (
            "账号 ",
            "主机 ",
            "进程 ",
            "文件 ",
            "user ",
            "host ",
            "process ",
            "file ",
        ):
            if prefix in desc:
                start = desc.index(prefix) + len(prefix)
                end = desc.find(" ", start) if " " in desc[start:] else len(desc)
                hint = desc[start:end].strip()
                if hint and hint not in hints:
                    hints.append(hint)
        related = ev.get("related_entities")
        if isinstance(related, list):
            for r in related:
                if isinstance(r, str) and r not in hints:
                    hints.append(r)
    return hints


def _summary_for_phases(phases: list[StorylinePhase]) -> str:
    """Generate a short narrative summary from phase list."""
    phase_names_cn = {
        StorylinePhaseName.INITIAL_ACCESS: "初始访问",
        StorylinePhaseName.COLLECTION: "数据收集",
        StorylinePhaseName.STAGING: "数据暂存",
        StorylinePhaseName.EXFILTRATION: "数据外传",
        StorylinePhaseName.POST_ACTION: "后续活动",
    }
    parts: list[str] = []
    for p in phases:
        label = phase_names_cn.get(p.phase_name, p.phase_name.value)
        parts.append(label)
    desc = " → ".join(parts)
    return f"攻击链：{desc}。" if desc else "无足够证据构建攻击链。"


# ====================================================================== #
# Shared helpers
# ====================================================================== #


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw
    try:
        text = str(raw).replace("Z", "+00:00")
        result = datetime.fromisoformat(text)
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        return result
    except (ValueError, TypeError):
        return None


def _parse_phase_name(raw: str) -> StorylinePhaseName | None:
    raw_lower = raw.strip().lower()
    for pn in StorylinePhaseName:
        if pn.value == raw_lower:
            return pn
    return None


def _parse_severity(raw: Any) -> Severity | None:
    if raw is None:
        return None
    if isinstance(raw, Severity):
        return raw
    raw_lower = str(raw).strip().lower()
    for sev in Severity:
        if sev.value == raw_lower:
            return sev
    return None


def _backfill_technique_ids(
    phases: list[StorylinePhase],
    technique_matches: list[dict[str, Any]],
) -> None:
    """Backfill technique_id in entries by matching description text against
    ATT&CK technique names from RAGOutput."""
    if not technique_matches:
        return
    for phase in phases:
        for entry in phase.entries:
            if entry.technique_id:
                continue
            desc_lower = entry.description.lower()
            for tm in technique_matches:
                tech_name = str(tm.get("technique_name", "")).lower()
                if tech_name and tech_name in desc_lower:
                    entry.technique_id = str(tm.get("technique_id", ""))
                    break
