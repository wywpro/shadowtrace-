"""AgentTraceService with TraceProjection for decision_trace audit (ISSUE-028).

Stores redacted, bounded input/output projections so the audit trail reveals
*what* an Agent decided and *which* evidence it cited, without persisting raw
payloads, secrets, prompts, or hidden reasoning chains.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any

import orjson
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.sanitization import REDACTED, is_sensitive_key, redact_sensitive_text
from app.db import models as orm

MAX_AUDIT_FIELD_BYTES = 1_048_576
_MAX_DECISION_TEXT_CHARS = 4_096
_MAX_AUDIT_DEPTH = 32
_RAW_KEYS = frozenset({"raw_payload", "raw_data", "source_snapshot", "raw_result", "prompt"})

# Fields that TraceProjection extracts for the structured decision_basis summary.
_DECISION_ID_FIELDS = frozenset({
    "event_id", "evidence_id", "action_id", "plan_id", "storyline_id",
    "report_id", "case_id", "trace_id",
})
_DECISION_CONCLUSION_FIELDS = frozenset({
    "reasoning", "narrative_summary", "strategy_summary", "summary",
    "structured_conclusion", "verdict", "final_verdict",
})
_DECISION_EVIDENCE_FIELDS = frozenset({
    "evidence_list", "evidence_refs", "evidence_output",
    "success_sources", "failed_sources",
})
_DECISION_RULES_FIELDS = frozenset({
    "rules_applied", "playbook_refs", "attack_techniques",
    "mitre_technique", "technique_id",
})
_DECISION_MODEL_FIELDS = frozenset({
    "model_name", "scoring_mode", "generated_by", "llm_model",
})
_DECISION_CONFIDENCE_FIELDS = frozenset({
    "confidence", "overall_confidence", "risk_score",
})
_DECISION_WARNING_FIELDS = frozenset({
    "warnings", "degraded", "degraded_flags", "error_detail",
    "possible_false_positive",
})


def _canonical_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def _hasher(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return redact_sensitive_text(value) if isinstance(value, str) else value
    if isinstance(value, bytes):
        return redact_sensitive_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _normalize_scalar(value.value)
    return redact_sensitive_text(str(value))


def _audit_hash_reference(value: Any, *, reason: str) -> dict[str, Any]:
    projected = _project_tree(value, project_raw=False)
    encoded = _canonical_bytes(projected)
    return {
        "_redacted": True,
        "reason": reason,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _is_raw_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _RAW_KEYS
        or "raw_payload" in lowered
        or "raw_data" in lowered
        or "prompt" in lowered
    )


def _project_tree(value: Any, *, project_raw: bool = True, depth: int = 0) -> Any:
    """Recursively sanitize a value: redact secrets, hash raw payloads, bound size."""
    if depth > _MAX_AUDIT_DEPTH:
        return {"_redacted": True, "reason": "max_depth_exceeded"}
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if project_raw and _is_raw_key(key):
                projected[key] = _audit_hash_reference(item, reason="raw_block")
            elif is_sensitive_key(key):
                projected[key] = REDACTED
            else:
                projected[key] = _project_tree(item, project_raw=project_raw, depth=depth + 1)
        return projected
    if isinstance(value, list | tuple):
        return [_project_tree(item, project_raw=project_raw, depth=depth + 1) for item in value]
    if isinstance(value, set | frozenset):
        projected_items = [
            _project_tree(item, project_raw=project_raw, depth=depth + 1) for item in value
        ]
        return sorted(projected_items, key=_canonical_bytes)
    return _normalize_scalar(value)


def _truncate_text(value: str, max_chars: int = _MAX_DECISION_TEXT_CHARS) -> str:
    cleaned = redact_sensitive_text(value)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}[TRUNCATED sha256={_hasher(cleaned)}]"


def _extract_scalar(data: dict[str, Any], keys: frozenset[str]) -> Any:
    """Extract the first matching value from a dict by key name."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _collect_refs(data: dict[str, Any], keys: frozenset[str]) -> list[str]:
    """Collect reference IDs from named list/dict fields."""
    refs: list[str] = []
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    _ref_keys = (
                        "evidence_id", "action_id", "case_id",
                        "technique_id", "citation_id",
                    )
                    for id_key in _ref_keys:
                        if id_key in item:
                            refs.append(str(item[id_key]))
        elif isinstance(value, Mapping):
            for id_key in ("evidence_id", "action_id", "case_id"):
                if id_key in value:
                    refs.append(str(value[id_key]))
    return refs[:100]


class TraceProjection:
    """Safe projection of Agent I/O for the audit trail.

    Strips raw payloads, secrets, and prompts; produces a bounded ``decision_basis``
    summary suitable for the ``agent_trace`` input_data / output_data columns.
    """

    @staticmethod
    def project(value: Any) -> dict[str, Any]:
        """Return a sanitised, size-bounded dict suitable for JSONB persistence."""
        if isinstance(value, BaseModel):
            raw = value.model_dump(mode="json")
        elif isinstance(value, Mapping):
            raw = dict(value)
        else:
            raw = {"_value": _normalize_scalar(value)}

        projected = _project_tree(raw)
        assert isinstance(projected, dict)
        encoded = _canonical_bytes(projected)
        if len(encoded) <= MAX_AUDIT_FIELD_BYTES:
            return projected

        return {
            "_truncated": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "original_size_bytes": len(encoded),
            "top_level_keys": sorted(projected)[:100],
        }

    @staticmethod
    def decision_basis(value: Any) -> dict[str, Any]:
        """Extract a compact structured summary from a projected model.

        Fields: input_summary, evidence_refs, rules_applied, model_name,
        structured_conclusion, selected_action, confidence, warnings.
        """
        if isinstance(value, BaseModel):
            data = value.model_dump(mode="json")
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            return {}

        id_scalar = _extract_scalar(data, _DECISION_ID_FIELDS)
        input_summary = (
            str(id_scalar) if id_scalar is not None
            else f"keys={sorted(data)[:20]}"
        )

        raw_conclusion = _extract_scalar(data, _DECISION_CONCLUSION_FIELDS)
        structured_conclusion = (
            _truncate_text(str(raw_conclusion)) if raw_conclusion is not None else ""
        )

        evidence_refs = _collect_refs(data, _DECISION_EVIDENCE_FIELDS)

        raw_rules = _extract_scalar(data, _DECISION_RULES_FIELDS)
        rules_applied = (
            [str(raw_rules)] if raw_rules is not None and not isinstance(raw_rules, (list, dict))
            else raw_rules if isinstance(raw_rules, list)
            else []
        )

        raw_model = _extract_scalar(data, _DECISION_MODEL_FIELDS)
        model_name = str(raw_model) if raw_model is not None else None

        raw_action = _extract_scalar(
            data, frozenset({"selected_action", "actions", "response_plan"}),
        )
        selected_action = str(raw_action)[:1000] if raw_action is not None else None

        confidence = None
        for key in _DECISION_CONFIDENCE_FIELDS:
            v = data.get(key)
            if isinstance(v, (int, float)):
                confidence = float(v)
                break

        raw_warnings = _extract_scalar(data, _DECISION_WARNING_FIELDS)
        warnings: list[str] = []
        if isinstance(raw_warnings, list):
            warnings = [str(w)[:500] for w in raw_warnings[:20]]
        elif raw_warnings is not None:
            warnings = [str(raw_warnings)[:500]]

        return {
            "input_summary": input_summary,
            "evidence_refs": evidence_refs,
            "rules_applied": rules_applied,
            "model_name": model_name,
            "structured_conclusion": structured_conclusion,
            "selected_action": selected_action,
            "confidence": confidence,
            "warnings": warnings,
        }


class AgentTraceService:
    """Writes and queries ``agent_trace`` rows with redacted I/O projections."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def new_trace_id() -> str:
        return f"trc-{uuid.uuid4().hex[:8]}"

    async def log_trace(
        self,
        event_id: str,
        agent_name: str,
        input_data: Any,
        output_data: Any | None,
        status: str,
        started_at: datetime,
        completed_at: datetime | None,
        error_detail: str | None = None,
        llm_model: str | None = None,
        llm_tokens_used: int | None = None,
    ) -> str:
        trace_id = self.new_trace_id()
        input_projected = TraceProjection.project(input_data)
        output_projected = (
            TraceProjection.project(output_data) if output_data is not None else {}
        )
        decision_basis = (
            TraceProjection.decision_basis(output_data)
            if output_data is not None else {}
        )
        output_projected["_decision_basis"] = decision_basis

        duration_ms: int | None = None
        if started_at is not None and completed_at is not None:
            duration_ms = max(0, int((completed_at - started_at).total_seconds() * 1_000))

        row = orm.AgentTrace(
            trace_id=trace_id,
            event_id=event_id,
            agent_name=agent_name,
            input_data=input_projected,
            output_data=output_projected,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error_detail=(
                redact_sensitive_text(error_detail)[:MAX_AUDIT_FIELD_BYTES]
                if error_detail else None
            ),
            llm_model=llm_model,
            llm_tokens_used=llm_tokens_used,
        )
        async with self._session_factory() as session:
            async with session.begin():
                session.add(row)
                await session.flush()
        return trace_id

    async def get_traces_by_event(self, event_id: str) -> list[orm.AgentTrace]:
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(orm.AgentTrace)
                .where(orm.AgentTrace.event_id == event_id)
                .order_by(
                    orm.AgentTrace.started_at.asc().nulls_last(),
                    orm.AgentTrace.trace_id.asc(),
                )
            )
            return list(rows)

    async def get_trace(self, trace_id: str) -> orm.AgentTrace | None:
        async with self._session_factory() as session:
            return await session.get(orm.AgentTrace, trace_id)


__all__ = ["AgentTraceService", "MAX_AUDIT_FIELD_BYTES", "TraceProjection"]
