"""Trusted workflow side effects for the ISSUE-048 StateGraph."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import ValidationError
from app.db import models as orm
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    ExecutionSubstate,
    FinalVerdict,
    Severity,
    WritebackReadiness,
)
from app.models.workflow import (
    TransitionContext,
    validate_execution_substate,
    validate_transition,
)
from app.services.context_service import append_context_journal_in_session

logger = logging.getLogger(__name__)

_RUNTIME_OPERATOR = "WorkflowRuntimeService"


class _EventServicePort(Protocol):
    async def apply_final_verdict_in_session(
        self,
        session: AsyncSession,
        event_id: str,
        verdict: FinalVerdict,
        *,
        operator: str | None = None,
    ) -> tuple[bool, Any, Any]: ...

    async def publish_final_verdict_mutation(
        self,
        event_id: str,
        verdict: FinalVerdict,
        *,
        result: Any,
        summary: Any,
    ) -> None: ...

    async def sync_event_summary_mutation(
        self,
        event_id: str,
        *,
        result: Any,
        summary: Any,
    ) -> None: ...


class WorkflowRuntimeService:
    """Sole writer for disposition-only intent and execution substate."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        event_service: _EventServicePort,
        readiness_resolver: (Callable[[str], Awaitable[WritebackReadiness]] | None) = None,
    ) -> None:
        self._session_factory = session_factory
        self._event_service = event_service
        self._readiness_resolver = readiness_resolver

    async def begin_disposition_only(self, event_id: str) -> None:
        """Atomically persist FP verdict, confidence floor, and trusted intent."""
        readiness = await self.get_event_status_update_readiness(event_id)
        if readiness is not WritebackReadiness.READY:
            raise ValidationError(
                "EVENT_STATUS_UPDATE is not ready for disposition-only",
                details={"event_id": event_id, "readiness": readiness.value},
            )
        verdict_changed = False
        confidence_changed = False
        result: Any = None
        summary: Any = None

        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
                if row is None:
                    raise KeyError(f"security_event not found: {event_id}")
                if DispositionPolicy(row.disposition_policy) is not DispositionPolicy.REQUIRED:
                    raise ValidationError(
                        "disposition-only requires disposition_policy=required",
                        details={"event_id": event_id},
                    )
                if EventStatus(row.status) is not EventStatus.TRIAGING:
                    raise ValidationError(
                        "disposition-only must begin from TRIAGING",
                        details={"event_id": event_id, "status": row.status},
                    )

                fp = await self._journal_dict(session, event_id, "false_positive_match")
                if not isinstance(fp, dict) or fp.get("recommendation") != "close_as_fp":
                    raise ValidationError(
                        "begin_disposition_only requires close_as_fp false_positive_match",
                        details={"event_id": event_id},
                    )
                try:
                    fp_score = max(0.0, min(1.0, float(fp.get("max_score") or 0.0)))
                except (TypeError, ValueError):
                    fp_score = 0.0

                previous_confidence = float(row.confidence or 0.0)
                confidence = max(previous_confidence, fp_score)
                confidence_changed = confidence != previous_confidence
                if confidence_changed:
                    row.confidence = confidence
                    row.row_version = int(row.row_version or 1) + 1
                    row.updated_at = datetime.now(UTC)
                    session.add(
                        orm.EventAuditLog(
                            event_id=event_id,
                            from_status=row.status,
                            to_status=row.status,
                            operator=_RUNTIME_OPERATOR,
                            reason=(
                                f"disposition_only_confidence:{previous_confidence}->{confidence}"
                            ),
                        )
                    )

                (
                    verdict_changed,
                    result,
                    summary,
                ) = await self._event_service.apply_final_verdict_in_session(
                    session,
                    event_id,
                    FinalVerdict.FALSE_POSITIVE,
                    operator=_RUNTIME_OPERATOR,
                )
                if not bool(
                    await self._journal_scalar(
                        session,
                        event_id,
                        "disposition_only_intent",
                    )
                ):
                    await append_context_journal_in_session(
                        session,
                        event_id,
                        "disposition_only_intent",
                        True,
                    )
                await session.flush()

        if verdict_changed:
            await self._event_service.publish_final_verdict_mutation(
                event_id,
                FinalVerdict.FALSE_POSITIVE,
                result=result,
                summary=summary,
            )
        elif confidence_changed:
            await self._event_service.sync_event_summary_mutation(
                event_id,
                result=result,
                summary=summary,
            )

    async def get_event_status_update_readiness(
        self,
        event_id: str,
    ) -> WritebackReadiness:
        """Resolve Adapter readiness server-side; missing resolver fails closed."""
        if self._readiness_resolver is None:
            return WritebackReadiness.CAPABILITY_UNKNOWN
        try:
            return WritebackReadiness(await self._readiness_resolver(event_id))
        except Exception:
            logger.warning(
                "EVENT_STATUS_UPDATE readiness lookup failed event=%s",
                event_id,
                exc_info=True,
            )
            return WritebackReadiness.CAPABILITY_UNKNOWN

    async def read_disposition_only_intent(self, event_id: str) -> bool:
        """Read the server-persisted intent, never a client or LLM claim."""
        async with self._session_factory() as session:
            value = await self._journal_scalar(session, event_id, "disposition_only_intent")
        return bool(value)

    async def set_execution_substate(
        self,
        event_id: str,
        substate: ExecutionSubstate,
        *,
        event_status: EventStatus,
    ) -> None:
        """Validate against locked EventStatus and persist the resumable substate."""
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(orm.SecurityEvent, event_id, with_for_update=True)
                if row is None:
                    raise KeyError(f"security_event not found: {event_id}")
                authoritative_status = EventStatus(row.status)
                if event_status is not authoritative_status:
                    raise ValidationError(
                        "caller EventStatus does not match authoritative state",
                        details={
                            "event_id": event_id,
                            "caller_status": event_status.value,
                            "authoritative_status": authoritative_status.value,
                        },
                    )
                raw = await self._journal_scalar(
                    session,
                    event_id,
                    "execution_substate",
                )
                try:
                    current = ExecutionSubstate(raw or ExecutionSubstate.NONE.value)
                except ValueError:
                    current = ExecutionSubstate.NONE
                validate_execution_substate(authoritative_status, current, substate)
                if current is not substate:
                    await append_context_journal_in_session(
                        session,
                        event_id,
                        "execution_substate",
                        substate.value,
                    )

    async def assert_disposition_only_transition_allowed(
        self,
        event_id: str,
        *,
        current: EventStatus,
        target: EventStatus,
    ) -> None:
        """Reject forged intent by rebuilding transition context from PostgreSQL."""
        async with self._session_factory() as session:
            row = await session.get(orm.SecurityEvent, event_id)
            if row is None:
                raise KeyError(f"security_event not found: {event_id}")
            intent = await self._journal_scalar(session, event_id, "disposition_only_intent")
            fp = await self._journal_dict(session, event_id, "false_positive_match")
        validate_transition(
            current,
            target,
            TransitionContext(
                final_verdict=FinalVerdict(row.final_verdict),
                disposition_only_intent=bool(intent),
                disposition_policy=DispositionPolicy(row.disposition_policy),
                severity=Severity(row.severity),
                recommendation=fp.get("recommendation") if isinstance(fp, dict) else None,
            ),
        )

    async def _journal_scalar(
        self,
        session: AsyncSession,
        event_id: str,
        field_name: str,
    ) -> Any:
        row = await session.scalar(
            select(orm.EventContextJournal)
            .where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name == field_name,
            )
            .order_by(orm.EventContextJournal.version.desc())
            .limit(1)
        )
        if row is None:
            return None
        value = row.value
        if isinstance(value, dict) and set(value) == {"_scalar"}:
            return value["_scalar"]
        return value

    async def _journal_dict(
        self,
        session: AsyncSession,
        event_id: str,
        field_name: str,
    ) -> dict[str, Any] | None:
        value = await self._journal_scalar(session, event_id, field_name)
        return value if isinstance(value, dict) else None


__all__ = ["WorkflowRuntimeService"]
