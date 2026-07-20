"""FalsePositiveCase and HistoryCase domain models (ISSUE-043)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import CaseLabel, EventType


class FalsePositiveCase(BaseModel):
    """A confirmed false-positive pattern stored in fp_case_kb."""

    case_id: str = Field(..., description="case-{8 hex}")
    pattern_summary: str = Field(..., description="Human-readable summary of the FP pattern")
    alert_signature: str = Field(..., description="Alert text / signature that triggers this FP")
    entity_pattern: str = Field(..., description="Involved entity types and characteristics")
    fp_reason: str = Field(..., description="Detailed reason why this is a false positive")
    confirmed_by: str = Field(..., description="Role or person who confirmed the FP")
    confirmed_at: datetime | str = Field(..., description="ISO-8601 timestamp of confirmation")


class HistoryCase(BaseModel):
    """A resolved historical investigation case stored in history_case_kb."""

    case_id: str = Field(..., description="case-{8 hex}")
    event_id: str | None = Field(default=None, description="Source event_id; null for seed data")
    event_type: EventType = Field(..., description="One of the 8 EventType values")
    case_label: CaseLabel = Field(..., description="true_positive | false_positive | uncertain")
    summary: str = Field(..., description="Narrative summary of the investigation")
    key_entities: str = Field(..., description="Key entities involved, semicolon-delimited")
    final_verdict: str = Field(..., description="Original FinalVerdict value")
    risk_score: int = Field(default=0, ge=0, le=100)
    resolution: str = Field(..., description="How the case was resolved / closed")
    closed_at: datetime | str | None = Field(default=None, description="ISO-8601 case closure time")


def make_chunk_id(kb_name: str, case_id: str) -> str:
    """Deterministic chunk_id from kb_name + case_id via SHA-256 prefix."""
    import hashlib

    digest = hashlib.sha256(f"{kb_name}:{case_id}".encode()).hexdigest()
    return f"chk-{digest[:8]}"


def fp_case_to_text(case: FalsePositiveCase) -> str:
    """Flatten an FP case into a single searchable text for embedding."""
    return " | ".join(
        [
            case.pattern_summary,
            case.alert_signature,
            case.entity_pattern,
            case.fp_reason,
        ]
    )


def history_case_to_text(case: HistoryCase) -> str:
    """Flatten a history case into a single searchable text for embedding."""
    return " | ".join([case.summary, case.key_entities])


def fp_case_metadata(case: FalsePositiveCase) -> dict[str, Any]:
    """Build metadata dict for an FP case chunk."""
    return {
        "case_id": case.case_id,
        "pattern_summary": case.pattern_summary,
        "alert_signature": case.alert_signature,
        "entity_pattern": case.entity_pattern,
        "fp_reason": case.fp_reason,
        "confirmed_by": case.confirmed_by,
        "confirmed_at": (
            case.confirmed_at.isoformat()
            if isinstance(case.confirmed_at, datetime)
            else case.confirmed_at
        ),
    }


def history_case_metadata(case: HistoryCase) -> dict[str, Any]:
    """Build metadata dict for a history case chunk."""
    et = case.event_type
    cl = case.case_label
    return {
        "case_id": case.case_id,
        "event_id": case.event_id,
        "event_type": et.value if isinstance(et, EventType) else et,
        "case_label": cl.value if isinstance(cl, CaseLabel) else cl,
        "summary": case.summary,
        "key_entities": case.key_entities,
        "final_verdict": case.final_verdict,
        "risk_score": case.risk_score,
        "resolution": case.resolution,
        "closed_at": (
            case.closed_at.isoformat() if isinstance(case.closed_at, datetime) else case.closed_at
        ),
    }
