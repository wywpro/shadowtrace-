"""EventAuditLogService for persisting EventStatus transitions (ISSUE-028).

Every controlled state change is recorded as an append-only audit log entry
keyed by event_id, ordered by created_at.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.sanitization import redact_sensitive_text
from app.db import models as orm


def _utc_now() -> datetime:
    return datetime.now(UTC)


class EventAuditLogService:
    """Appends status-transition audit entries to ``event_audit_log``."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def log_transition(
        self,
        event_id: str,
        from_status: str | None,
        to_status: str | None,
        operator: str | None,
        reason: str | None,
    ) -> str:
        """Persist one state transition and return its generated id as a string.

        The caller (StateMachineService) is responsible for calling this inside
        the same transaction that performs the status change so the log entry is
        atomically consistent with the new state.
        """
        row = orm.EventAuditLog(
            event_id=event_id,
            from_status=from_status,
            to_status=to_status,
            operator=redact_sensitive_text(operator) if operator else None,
            reason=redact_sensitive_text(reason)[:4096] if reason else None,
            created_at=_utc_now(),
        )
        async with self._session_factory() as session:
            async with session.begin():
                session.add(row)
                await session.flush()
        return str(row.id)

    async def log_transition_in_session(
        self,
        session: AsyncSession,
        event_id: str,
        from_status: str | None,
        to_status: str | None,
        operator: str | None,
        reason: str | None,
    ) -> str:
        """Same as ``log_transition`` but within a caller-provided transaction."""
        row = orm.EventAuditLog(
            event_id=event_id,
            from_status=from_status,
            to_status=to_status,
            operator=redact_sensitive_text(operator) if operator else None,
            reason=redact_sensitive_text(reason)[:4096] if reason else None,
            created_at=_utc_now(),
        )
        session.add(row)
        await session.flush()
        return str(row.id)

    async def get_logs_by_event(self, event_id: str) -> list[orm.EventAuditLog]:
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(orm.EventAuditLog)
                .where(orm.EventAuditLog.event_id == event_id)
                .order_by(
                    orm.EventAuditLog.created_at.asc(),
                    orm.EventAuditLog.id.asc(),
                )
            )
            return list(rows)


__all__ = ["EventAuditLogService"]
