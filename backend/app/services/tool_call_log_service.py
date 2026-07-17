"""Persistent, redacted audit log for tool calls (ISSUE-023)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import orjson
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.sanitization import REDACTED, is_sensitive_key, redact_sensitive_text
from app.db import models as orm

MAX_AUDIT_FIELD_BYTES = 1_048_576
_MAX_CREDENTIAL_REFERENCE_BYTES = 2_048
_MAX_AUDIT_DEPTH = 32
_RAW_PAYLOAD_KEYS = frozenset({"raw", "raw_payload", "raw_result"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


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


def _is_credential_reference(key: str) -> bool:
    lowered = key.lower()
    return is_sensitive_key(lowered) and lowered.endswith(("_ref", "_reference", "_id"))


def _is_raw_payload_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _RAW_PAYLOAD_KEYS or "raw_payload" in lowered or "raw_result" in lowered


def _audit_reference(value: Any, *, reason: str) -> dict[str, Any]:
    projected = _sanitize_tree(value, project_raw=False)
    encoded = _canonical_bytes(projected)
    return {
        "_redacted": True,
        "reason": reason,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _sanitize_tree(value: Any, *, project_raw: bool = True, depth: int = 0) -> Any:
    if depth > _MAX_AUDIT_DEPTH:
        return {"_redacted": True, "reason": "max_depth_exceeded"}
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if project_raw and _is_raw_payload_key(key):
                projected[key] = _audit_reference(item, reason="raw_payload")
            elif _is_credential_reference(key):
                reference = _sanitize_tree(item, project_raw=False, depth=depth + 1)
                encoded = _canonical_bytes(reference)
                projected[key] = (
                    reference
                    if len(encoded) <= _MAX_CREDENTIAL_REFERENCE_BYTES
                    else {
                        "_redacted": True,
                        "reason": "credential_reference_too_large",
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "size_bytes": len(encoded),
                    }
                )
            elif is_sensitive_key(key):
                projected[key] = REDACTED
            else:
                projected[key] = _sanitize_tree(
                    item,
                    project_raw=project_raw,
                    depth=depth + 1,
                )
        return projected
    if isinstance(value, list | tuple):
        return [_sanitize_tree(item, project_raw=project_raw, depth=depth + 1) for item in value]
    if isinstance(value, set | frozenset):
        projected_items = [
            _sanitize_tree(item, project_raw=project_raw, depth=depth + 1) for item in value
        ]
        return sorted(projected_items, key=_canonical_bytes)
    return _normalize_scalar(value)


def _key_summary(key: str) -> str:
    cleaned = redact_sensitive_text(key)
    if len(cleaned) <= 128:
        return cleaned
    return f"{cleaned[:128]}...[sha256={hashlib.sha256(cleaned.encode()).hexdigest()}]"


def _bounded_projection(value: Mapping[str, Any] | None) -> dict[str, Any]:
    projected = _sanitize_tree(dict(value or {}))
    assert isinstance(projected, dict)
    encoded = _canonical_bytes(projected)
    if len(encoded) <= MAX_AUDIT_FIELD_BYTES:
        return projected
    return {
        "_truncated": True,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "original_size_bytes": len(encoded),
        "top_level_keys": [_key_summary(key) for key in sorted(projected)[:100]],
    }


def _bounded_error_detail(value: str | None) -> str | None:
    if value is None:
        return None
    projected = redact_sensitive_text(value)
    encoded = projected.encode("utf-8")
    if len(encoded) <= MAX_AUDIT_FIELD_BYTES:
        return projected
    return (
        "[TRUNCATED "
        f"original_size_bytes={len(encoded)} "
        f"sha256={hashlib.sha256(encoded).hexdigest()}]"
    )


def _string_value(value: str | Enum) -> str:
    normalized = value.value if isinstance(value, Enum) else value
    return str(normalized)


class ToolCallLogService:
    """Two-phase writer and query service for ``tool_call_log``."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def log_start(
        self,
        call_id: str,
        event_id: str,
        action_id: str | None,
        tool_name: str,
        tool_category: str | Enum,
        parameters: Mapping[str, Any] | None,
    ) -> str:
        started_at = _utc_now()
        row = orm.ToolCallLog(
            call_id=call_id,
            event_id=event_id,
            action_id=action_id,
            tool_name=tool_name,
            tool_category=_string_value(tool_category),
            parameters=_bounded_projection(parameters),
            result={},
            status="running",
            started_at=started_at,
            retry_count=0,
        )
        async with self._session_factory() as session:
            async with session.begin():
                session.add(row)
                await session.flush()
        return call_id

    async def log_finish(
        self,
        call_id: str,
        status: str | Enum,
        result: Mapping[str, Any] | None,
        error_detail: str | None,
        retry_count: int,
    ) -> None:
        if retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        completed_at = _utc_now()
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(orm.ToolCallLog, call_id, with_for_update=True)
                if row is None:
                    raise KeyError(f"tool call log not found: {call_id}")
                row.completed_at = completed_at
                row.duration_ms = (
                    max(0, int((completed_at - row.started_at).total_seconds() * 1_000))
                    if row.started_at is not None
                    else None
                )
                row.status = _string_value(status)
                row.result = _bounded_projection(result)
                row.error_detail = _bounded_error_detail(error_detail)
                row.retry_count = retry_count
                await session.flush()

    async def get_logs_by_event(self, event_id: str) -> list[orm.ToolCallLog]:
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(orm.ToolCallLog)
                .where(orm.ToolCallLog.event_id == event_id)
                .order_by(
                    orm.ToolCallLog.started_at.asc().nulls_last(),
                    orm.ToolCallLog.call_id.asc(),
                )
            )
            return list(rows)

    async def get_logs_by_tool(
        self,
        tool_name: str,
        limit: int = 50,
    ) -> list[orm.ToolCallLog]:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(orm.ToolCallLog)
                .where(orm.ToolCallLog.tool_name == tool_name)
                .order_by(
                    orm.ToolCallLog.started_at.asc().nulls_last(),
                    orm.ToolCallLog.call_id.asc(),
                )
                .limit(limit)
            )
            return list(rows)

    async def get_log(self, call_id: str) -> orm.ToolCallLog | None:
        async with self._session_factory() as session:
            return await session.get(orm.ToolCallLog, call_id)


__all__ = ["MAX_AUDIT_FIELD_BYTES", "ToolCallLogService"]
