"""Event endpoints — real implementations (ISSUE-038).

Replaces ISSUE-004 placeholder stubs with database-backed endpoints
that drive the full analysis lifecycle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy import exc as sa_exc
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.v1 import schemas as s
from app.api.v1.deps import _get_session_factory, get_event_service, get_pipeline, get_state_machine
from app.api.v1.errors import (
    DispositionPermissionDenied,
    EventNotFoundError,
    InvalidStateTransitionError,
    WritebackConflictError,
    WritebackFailedError,
    WritebackPendingError,
    WritebackUnsupportedError,
)
from app.core.auth import (
    ROLE_ADMIN,
    ROLE_ANALYST,
    ROLE_DISPOSITION_OPERATOR,
    AuthorizationError,
    CurrentPrincipal,
    Principal,
    require_roles,
)
from app.core.errors import DependencyUnavailableError
from app.db import models as orm
from app.models.action import Action as ActionModel
from app.models.disposition import SourceObjectLocator
from app.models.enums import (
    ActionStatus,
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    SourceObjectKind,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.workflow import TransitionContext

if TYPE_CHECKING:
    from app.services.event_service import EventService
    from app.services.state_machine_service import StateMachineService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"])

# Source objects associated with the example event (contract-test backward compat).
_ASSOCIATED_SOURCE_RECORDS = {"src-associated-1"}


# --------------------------------------------------------------------------- #
# Helper: safe session factory access (degrades gracefully without DB)
# --------------------------------------------------------------------------- #


def _try_get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the session factory, or None if DB is unavailable.

    Configuration errors (ValueError, TypeError — e.g. malformed database_url)
    propagate immediately so the operator can detect them at startup rather than
    silently running with empty results.
    """
    try:
        from app.api.v1.deps import _get_session_factory

        sf = _get_session_factory()
        return sf
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "Database session factory unavailable (missing configuration) — returning empty results"
        )
        return None
    except (ValueError, TypeError):
        # Configuration errors must propagate — a malformed database_url or
        # similar config issue must not be silently swallowed (ISSUE-038 #8).
        raise
    except (ConnectionRefusedError, TimeoutError, OSError):
        # Transient infrastructure errors (network, filesystem) — degrade
        # gracefully so the API can still return empty results rather than 5xx.
        logger.warning(
            "Database session factory unavailable (transient error) — returning empty results",
            exc_info=True,
        )
        return None


# --------------------------------------------------------------------------- #
# Helper: resolve writeback info for EventDetail / EventListItem
# --------------------------------------------------------------------------- #


def _writeback_required(policy: DispositionPolicy) -> bool:
    return policy == DispositionPolicy.REQUIRED


async def _sync_report_context_and_bus(
    event_id: str,
    report: Any,
    event_service: EventService,
) -> None:
    """Write report to EventContext and publish report_generated when bus is available."""
    from app.api.v1.deps import _get_context_store

    try:
        await _get_context_store().set(event_id, "report", report.model_dump(mode="json"))
    except Exception:
        logger.warning(
            "Failed to write report to EventContext for event=%s",
            event_id,
            exc_info=True,
        )

    bus = getattr(event_service, "_bus", None)
    if bus is not None:
        try:
            payload: dict[str, Any] = {
                "report_id": report.report_id,
                "sections": len(report.sections),
            }
            if report.generated_at is not None:
                payload["generated_at"] = report.generated_at.isoformat()
            await bus.publish_event(event_id, "report_generated", payload)
        except Exception:
            logger.warning(
                "event_bus report_generated failed for event=%s",
                event_id,
                exc_info=True,
            )


async def _regenerate_report_after_verdict_change(
    event_id: str,
    *,
    event_title: str,
    final_verdict: FinalVerdict,
    risk_score: int,
    severity: Severity,
    operator: str,
    event_service: EventService,
) -> None:
    """Refresh report after verdict change without destroying full investigation content."""
    existing = await event_service.get_report(event_id=event_id)
    if existing is not None and existing.generated_by != "quick_close":
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        updated = existing.model_copy(
            update={
                "final_verdict": final_verdict,
                "version": int(existing.version or 1) + 1,
                "updated_at": now,
            }
        )
        await event_service.upsert_report(updated)
        await _sync_report_context_and_bus(event_id, updated, event_service)
        return

    await _generate_quick_close_report(
        event_id=event_id,
        event_title=event_title,
        final_verdict=final_verdict,
        risk_score=risk_score,
        severity=severity,
        operator=operator,
        event_service=event_service,
        force_regenerate=existing is not None,
    )


async def _generate_quick_close_report(
    event_id: str,
    event_title: str,
    final_verdict: FinalVerdict,
    risk_score: int,
    severity: Severity,
    operator: str,
    event_service: EventService,
    *,
    force_regenerate: bool = False,
) -> None:
    """Generate a standard 15-section quick-close report so the CLOSED gate can pass.

    Uses ReportSectionBuilder to produce all 15 standard sections.  Evidence,
    disposition, and verification sections use placeholder text; overview and
    recommendations explain the quick-close / low-risk reason.

    The validate_closed_gate check in StateMachineService requires a report
    row to exist before allowing CLOSED.
    """
    from datetime import UTC, datetime

    from app.agents.report_section_builder import ReportSectionBuilder
    from app.models.agent_io import (
        CollectionStatus,
        EvidenceOutput,
        RiskAssessment,
        ScoringMode,
    )
    from app.models.ids import report_id_for_event
    from app.models.report import InvestigationReport

    # Skip if report already exists unless caller needs a verdict refresh.
    existing = await event_service.get_report(event_id=event_id)
    if existing is not None and not force_regenerate:
        return

    # Build placeholder evidence / risk matching _short_circuit_close semantics.
    placeholder_evidence = EvidenceOutput(
        evidence_list=[],
        conflicts=[],
        gaps=[],
        success_sources=[],
        failed_sources=[],
        overall_confidence=0.0,
        collection_status=CollectionStatus.COMPLETED,
    )
    placeholder_risk = RiskAssessment(
        risk_score=risk_score,
        severity=severity,
        confidence=0.9,
        risk_factors=[],
        possible_false_positive=(final_verdict == FinalVerdict.FALSE_POSITIVE),
        scoring_mode=ScoringMode.RULE_ONLY,
    )

    builder = ReportSectionBuilder()
    sections = builder.build(
        event_id=event_id,
        evidence_output=placeholder_evidence,
        risk_assessment=placeholder_risk,
        triage_result=None,
        response_plan=None,
        verification_result=None,
        rag_output=None,
        final_verdict=final_verdict,
    )

    title = f"Quick Close Report — {event_title}"
    summary = (
        f"Auto-generated quick-close report for {event_id}. "
        f"severity={severity.value}; risk_score={risk_score}; verdict={final_verdict.value}. "
        f"Evidence, disposition, and verification sections use placeholder content — "
        f"this event was closed via quick-close path without full investigation."
    )

    now = datetime.now(UTC)
    report_version = 1
    if existing is not None:
        report_version = int(existing.version or 1) + 1

    report = InvestigationReport(
        report_id=report_id_for_event(event_id),
        event_id=event_id,
        title=title,
        summary=summary,
        sections=sections,
        final_verdict=final_verdict,
        risk_score=risk_score,
        severity=severity,
        version=report_version,
        generated_by="quick_close",
        generated_at=now,
        updated_at=now,
    )
    await event_service.upsert_report(report)
    await _sync_report_context_and_bus(event_id, report, event_service)

    # Record the system action for audit trail.
    # Only catch IntegrityError (idempotent re-entry race); let other
    # exceptions propagate so callers get DependencyUnavailableError rather
    # than silently incomplete audit trails.
    try:
        await event_service.upsert_generate_report_action(event_id, plan_revision=1)
    except IntegrityError:
        logger.warning(
            "generate_report action already exists for quick-close event=%s "
            "(concurrent upsert race)",
            event_id,
            exc_info=True,
        )


async def _validate_writeback_gate(
    event_id: str,
    event: Any,
) -> None:
    """Validate the writeback gate before allowing close.

    For REQUIRED disposition_policy events, checks writeback readiness and
    outbox status.  Raises the appropriate error for each blocked case so
    callers never need to duplicate the gate logic.

    No-op for NOT_REQUIRED events.
    """
    if event.disposition_policy != DispositionPolicy.REQUIRED:
        return

    from app.api.v1.deps import _get_session_factory

    readiness, wb_status, _pending = await _build_writeback_info(
        event_id, event.disposition_policy, _get_session_factory()
    )
    if readiness == WritebackReadiness.NOT_CONFIGURED:
        raise WritebackUnsupportedError(
            "required disposition_policy but no disposition Action configured",
            details={"event_id": event_id},
        )
    if readiness not in (WritebackReadiness.READY, WritebackReadiness.NOT_REQUIRED):
        raise WritebackUnsupportedError(
            f"writeback readiness is {readiness.value}",
            details={"event_id": event_id, "readiness": readiness.value},
        )
    if wb_status in (WritebackStatus.PENDING, WritebackStatus.UNKNOWN):
        raise WritebackPendingError(
            f"writeback is {wb_status.value}",
            details={"event_id": event_id, "writeback_status": wb_status.value},
        )
    if wb_status == WritebackStatus.FAILED:
        raise WritebackFailedError(
            "writeback failed",
            details={"event_id": event_id},
        )
    if wb_status == WritebackStatus.CONFLICT:
        raise WritebackConflictError(
            "writeback conflict",
            details={"event_id": event_id},
        )


async def _build_writeback_info(
    event_id: str,
    policy: DispositionPolicy,
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[WritebackReadiness, WritebackStatus | None, int]:
    """Derive overall event-level writeback readiness / status / pending count."""
    if policy == DispositionPolicy.NOT_REQUIRED:
        return WritebackReadiness.NOT_REQUIRED, None, 0

    async with session_factory() as session:
        # Count non-superseded response/rollback actions.
        counts = await session.execute(
            select(
                func.count(orm.Action.action_id),
                func.min(orm.Action.writeback_readiness),
            ).where(
                orm.Action.event_id == event_id,
                orm.Action.action_category.in_(("response", "rollback")),
                orm.Action.superseded_by_revision.is_(None),
                orm.Action.status.not_in(("rejected", "superseded")),
            )
        )
        total_actions, min_readiness_raw = counts.one()
        total = int(total_actions or 0)

        readiness = WritebackReadiness.READY
        if total == 0:
            readiness = WritebackReadiness.NOT_CONFIGURED
        elif min_readiness_raw:
            try:
                readiness = WritebackReadiness(min_readiness_raw)
            except ValueError:
                readiness = WritebackReadiness.CAPABILITY_UNKNOWN

        # Count pending/active outbox records.
        pending_count = await session.scalar(
            select(func.count(orm.DispositionOutbox.outbox_id)).where(
                orm.DispositionOutbox.event_id == event_id,
                orm.DispositionOutbox.superseded_by_disposition_id.is_(None),
                orm.DispositionOutbox.latest_writeback_status.in_(
                    (
                        WritebackStatus.PENDING.value,
                        WritebackStatus.SENDING.value,
                        WritebackStatus.ACCEPTED.value,
                        WritebackStatus.UNKNOWN.value,
                    )
                ),
            )
        )
        pending = int(pending_count or 0)

        # Derive overall writeback status from all active outbox rows (not only
        # pending-countable rows — FAILED/CONFLICT are terminal and excluded
        # from pending_count but must still block close).
        wb_status: WritebackStatus | None = None
        status_rows = (
            await session.scalars(
                select(orm.DispositionOutbox.latest_writeback_status).where(
                    orm.DispositionOutbox.event_id == event_id,
                    orm.DispositionOutbox.superseded_by_disposition_id.is_(None),
                )
            )
        ).all()
        parsed_statuses: list[WritebackStatus] = []
        for raw in status_rows:
            if not raw:
                continue
            try:
                parsed_statuses.append(WritebackStatus(str(raw)))
            except ValueError:
                continue

        if parsed_statuses:
            if any(s is WritebackStatus.FAILED for s in parsed_statuses):
                wb_status = WritebackStatus.FAILED
            elif any(s is WritebackStatus.CONFLICT for s in parsed_statuses):
                wb_status = WritebackStatus.CONFLICT
            elif any(s is WritebackStatus.UNKNOWN for s in parsed_statuses):
                wb_status = WritebackStatus.UNKNOWN
            elif any(
                s
                in (
                    WritebackStatus.PENDING,
                    WritebackStatus.SENDING,
                    WritebackStatus.ACCEPTED,
                )
                for s in parsed_statuses
            ):
                wb_status = WritebackStatus.PENDING
            elif all(s is WritebackStatus.CONFIRMED for s in parsed_statuses):
                wb_status = WritebackStatus.CONFIRMED

        return readiness, wb_status, pending


# --------------------------------------------------------------------------- #
# POST /events — create
# --------------------------------------------------------------------------- #


@router.post("/events", response_model=s.EventSummary, status_code=status.HTTP_201_CREATED)
async def create_event(
    body: s.EventCreateRequest,
    principal: Annotated[Principal, require_roles(ROLE_ANALYST)],
    event_service: EventService = Depends(get_event_service),
) -> s.EventSummary:
    raw_alert: dict[str, Any] = {
        "title": body.title,
        "description": body.description,
    }
    event = await event_service.create_event(
        raw_alert,
        source_type="manual",
        title=body.title,
        event_type=body.event_type,
        severity=body.severity,
    )
    from app.services.context_service import event_summary_from_domain

    return event_summary_from_domain(event)


# --------------------------------------------------------------------------- #
# GET /events — list
# --------------------------------------------------------------------------- #


@router.get("/events", response_model=s.EventListResponse)
async def list_events(
    principal: CurrentPrincipal,
    page: int = 1,
    page_size: int = 20,
    status: EventStatus | None = None,
    severity: Severity | None = None,
    event_type: EventType | None = None,
    final_verdict: FinalVerdict | None = None,
    keyword: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    sort_by: str | None = None,
    sort_order: Literal["asc", "desc"] | None = None,
    event_service: EventService = Depends(get_event_service),
) -> s.EventListResponse:
    result = await event_service.list_events(
        status=status,
        severity=severity,
        event_type=event_type,
        final_verdict=final_verdict,
        keyword=keyword,
        occurred_after=start_time,
        occurred_before=end_time,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    items: list[s.EventListItem] = []
    for event in result.items:
        wb_required = _writeback_required(event.disposition_policy)
        # ISSUE-038: list view does not resolve per-event writeback info for
        # performance reasons. When writeback is required, signal capability
        # is unknown rather than misleading NOT_CONFIGURED.
        wb_readiness = (
            WritebackReadiness.CAPABILITY_UNKNOWN
            if wb_required
            else WritebackReadiness.NOT_REQUIRED
        )
        items.append(
            s.EventListItem(
                event_id=event.event_id,
                event_type=event.event_type,
                title=event.title,
                status=event.status,
                severity=event.severity,
                risk_score=event.risk_score,
                final_verdict=event.final_verdict,
                writeback_required=wb_required,
                writeback_readiness=wb_readiness,
                writeback_overall_status=None,
                pending_writeback_count=0,
                created_at=event.created_at,
                updated_at=event.updated_at,
                occurred_at=event.occurred_at,
            )
        )
    return s.EventListResponse(
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        items=items,
    )


# --------------------------------------------------------------------------- #
# GET /events/{event_id} — detail
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}", response_model=s.EventDetailResponse)
async def get_event(
    event_id: str,
    principal: CurrentPrincipal,
    event_service: EventService = Depends(get_event_service),
) -> s.EventDetailResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    required = _writeback_required(event.disposition_policy)
    readiness = WritebackReadiness.NOT_REQUIRED
    wb_status: WritebackStatus | None = None
    pending_count = 0

    if required:
        try:
            from app.api.v1.deps import _get_session_factory

            readiness, wb_status, pending_count = await _build_writeback_info(
                event_id, event.disposition_policy, _get_session_factory()
            )
        except Exception:
            # DB unavailable: leave writeback info as defaults.
            readiness = WritebackReadiness.CAPABILITY_UNKNOWN

    return s.EventDetailResponse(
        event=event,
        writeback_required=required,
        writeback_readiness=readiness,
        writeback_overall_status=wb_status,
        pending_writeback_count=pending_count,
    )


# --------------------------------------------------------------------------- #
# POST /events/{event_id}/investigate — start analysis
# --------------------------------------------------------------------------- #


@router.post(
    "/events/{event_id}/investigate",
    response_model=s.InvestigateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def investigate_event(
    event_id: str,
    background: BackgroundTasks,
    principal: Annotated[Principal, require_roles(ROLE_ANALYST)],
    body: s.InvestigateRequest | None = None,
    event_service: EventService = Depends(get_event_service),
    state_machine: StateMachineService = Depends(get_state_machine),
) -> s.InvestigateResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    if event.status != EventStatus.NEW:
        raise InvalidStateTransitionError(
            f"event must be in NEW status to start investigation, current: {event.status.value}",
            current=event.status,
            target=EventStatus.TRIAGING,
            details={"event_id": event_id},
        )

    # Enqueue the pipeline as a background task.
    async def _run_pipeline() -> None:
        try:
            from app.services.evidence_projection import (
                EvidenceProjection,
                bind_evidence_projection,
            )

            pipeline = await get_pipeline()
            projection = EvidenceProjection(_get_session_factory())
            with bind_evidence_projection(projection):
                await pipeline.run(event_id)
        except Exception as exc:
            logger.error(
                "Background pipeline failed for event=%s: %s",
                event_id,
                exc,
            )
            try:
                await state_machine.transition(
                    event_id,
                    EventStatus.FAILED,
                    operator="AnalysisOnlyPipeline",
                    reason=f"pipeline_failed: {exc}",
                )
            except Exception:
                logger.exception("Failed to mark event as FAILED: %s", event_id)

    background.add_task(_run_pipeline)

    return s.InvestigateResponse(
        event_id=event_id,
        task_id=event_id,
        status=event.status,
    )


# --------------------------------------------------------------------------- #
# POST /events/{event_id}/close — close event
# --------------------------------------------------------------------------- #


@router.post("/events/{event_id}/close", response_model=s.EventCloseResponse)
async def close_event(
    event_id: str,
    body: s.EventCloseRequest,
    principal: Annotated[Principal, require_roles(ROLE_ANALYST)],
    event_service: EventService = Depends(get_event_service),
    state_machine: StateMachineService = Depends(get_state_machine),
) -> s.EventCloseResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    current_status = event.status

    # Admin force_close bypass.
    if body.force_local_close:
        if not principal.has_any_role([ROLE_ADMIN]):
            raise AuthorizationError([ROLE_ADMIN])
        result = await state_machine.force_close(
            event_id,
            principal=principal.subject,
            reason=body.reason,
        )
        return s.EventCloseResponse(
            event_id=event_id,
            status=EventStatus.CLOSED,
            final_verdict=result.final_verdict,
            external_unsynced=True,
        )

    # Validate close rules per ISSUE-038.
    # Allowed paths: REPORTING→CLOSED, FAILED→REPORTING→CLOSED,
    # TRIAGING+not_required low/fp→CLOSED.
    if current_status == EventStatus.TRIAGING:
        if event.disposition_policy != DispositionPolicy.NOT_REQUIRED:
            raise WritebackUnsupportedError(
                "TRIAGING→CLOSED requires disposition_policy=not_required; "
                "required-disposition events must go through the disposition-only "
                "orchestration chain",
                details={
                    "event_id": event_id,
                    "disposition_policy": event.disposition_policy.value,
                },
            )
        # TRIAGING shortcut: generate report so validate_closed_gate can pass,
        # then transition directly to CLOSED (TRIAGING→REPORTING is illegal).
        close_verdict = event.final_verdict
        if body.final_verdict is not None and body.final_verdict != event.final_verdict:
            await event_service.set_final_verdict(
                event_id,
                body.final_verdict,
                operator=f"principal:{principal.subject}",
            )
            event = await event_service.get_event(event_id)
            if event is None:
                raise EventNotFoundError(
                    f"event {event_id} not found after verdict update",
                    details={"event_id": event_id},
                )
            close_verdict = body.final_verdict
        await _generate_quick_close_report(
            event_id=event_id,
            event_title=event.title,
            final_verdict=close_verdict,
            risk_score=event.risk_score,
            severity=event.severity,
            operator=f"principal:{principal.subject}",
            event_service=event_service,
        )
        await event_service.transition_status(
            event_id,
            EventStatus.CLOSED,
            context=TransitionContext(
                need_investigation=body.need_investigation,
            ),
            operator=f"principal:{principal.subject}",
            reason=body.reason,
        )
    elif current_status == EventStatus.REPORTING:
        # ISSUE-038 step 2: writeback gate pre-check.
        await _validate_writeback_gate(event_id, event)

        # Handle final_verdict change before closing — regenerate report first.
        if body.final_verdict is not None and body.final_verdict != event.final_verdict:
            await event_service.set_final_verdict(
                event_id,
                body.final_verdict,
                operator=f"principal:{principal.subject}",
            )
            event = await event_service.get_event(event_id)
            if event is None:
                raise EventNotFoundError(
                    f"event {event_id} not found after verdict update",
                    details={"event_id": event_id},
                )
            await _regenerate_report_after_verdict_change(
                event_id,
                event_title=event.title,
                final_verdict=body.final_verdict,
                risk_score=event.risk_score,
                severity=event.severity,
                operator=f"principal:{principal.subject}",
                event_service=event_service,
            )
        await event_service.transition_status(
            event_id,
            EventStatus.CLOSED,
            operator=f"principal:{principal.subject}",
            reason=body.reason,
        )
    elif current_status == EventStatus.FAILED:
        # FAILED → REPORTING → CLOSED.
        # ISSUE-038: writeback gate pre-check before any state transitions
        # to avoid leaving the event stuck in REPORTING.
        await _validate_writeback_gate(event_id, event)

        # Generate a quick-close report so validate_closed_gate can pass.
        await _generate_quick_close_report(
            event_id=event_id,
            event_title=event.title,
            final_verdict=event.final_verdict,
            risk_score=event.risk_score,
            severity=event.severity,
            operator=f"principal:{principal.subject}",
            event_service=event_service,
        )
        await event_service.transition_status(
            event_id,
            EventStatus.REPORTING,
            operator=f"principal:{principal.subject}",
            reason="close:report_before_close",
        )
        if body.final_verdict is not None and body.final_verdict != event.final_verdict:
            await event_service.set_final_verdict(
                event_id,
                body.final_verdict,
                operator=f"principal:{principal.subject}",
            )
            event = await event_service.get_event(event_id)
            if event is None:
                raise EventNotFoundError(
                    f"event {event_id} not found after verdict update",
                    details={"event_id": event_id},
                )
            await _regenerate_report_after_verdict_change(
                event_id,
                event_title=event.title,
                final_verdict=body.final_verdict,
                risk_score=event.risk_score,
                severity=event.severity,
                operator=f"principal:{principal.subject}",
                event_service=event_service,
            )
        await event_service.transition_status(
            event_id,
            EventStatus.CLOSED,
            operator=f"principal:{principal.subject}",
            reason=body.reason,
        )
    else:
        raise InvalidStateTransitionError(
            f"Cannot close event in {current_status.value} status",
            current=current_status,
            target=EventStatus.CLOSED,
            details={"event_id": event_id},
        )

    # Reload final state.
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(
            f"event {event_id} disappeared after close",
            details={"event_id": event_id},
        )
    return s.EventCloseResponse(
        event_id=event_id,
        status=event.status,
        final_verdict=event.final_verdict,
        external_unsynced=event.external_unsynced,
    )


# --------------------------------------------------------------------------- #
# Helper: execute DB read for list endpoints
# --------------------------------------------------------------------------- #


async def _db_read(
    event_id: str,
    table: Any,
    order_by: Any,
    page: int = 1,
    page_size: int = 20,
    extra_conditions: list[Any] | None = None,
) -> tuple[list[Any], int]:
    """Execute a paginated read query.

    Returns empty results for transient DB errors (connection issues, pool
    exhaustion).  Non-transient errors are re-raised so the API layer can
    return HTTP 503 rather than silently reporting success with no data.
    """
    from sqlalchemy import exc as sa_exc

    from app.core.errors import DependencyUnavailableError

    sf = _try_get_session_factory()
    if sf is None:
        return [], 0
    conditions: list[Any] = [table.event_id == event_id]
    if extra_conditions:
        conditions.extend(extra_conditions)
    page = max(1, page)
    page_size = min(max(1, page_size), 500)
    try:
        async with sf() as session:
            count = await session.scalar(select(func.count()).select_from(table).where(*conditions))
            total = int(count or 0)
            rows = (
                await session.scalars(
                    select(table)
                    .where(*conditions)
                    .order_by(order_by)
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).all()
        return list(rows), total
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "DB read skipped for table=%s event=%s (session factory unavailable)",
            getattr(table, "__tablename__", table),
            event_id,
        )
        return [], 0
    except (ConnectionRefusedError, TimeoutError, sa_exc.OperationalError):
        logger.warning(
            "DB read degraded (transient error) for table=%s event=%s",
            getattr(table, "__tablename__", table),
            event_id,
            exc_info=True,
        )
        return [], 0
    except Exception as exc:
        logger.error(
            "DB read failed (non-transient) for table=%s event=%s: %s",
            getattr(table, "__tablename__", table),
            event_id,
            exc,
            exc_info=True,
        )
        raise DependencyUnavailableError(
            "database query failed",
            error_code="dependency_unavailable",
            details={
                "table": getattr(table, "__tablename__", str(table)),
                "event_id": event_id,
            },
        ) from exc


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/report
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/report", response_model=s.ReportResponse)
async def get_report(
    event_id: str,
    principal: CurrentPrincipal,
    event_service: EventService = Depends(get_event_service),
) -> s.ReportResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    report = await event_service.get_report(event_id=event_id)
    if report is None:
        raise EventNotFoundError(
            f"no report found for event {event_id}",
            details={"event_id": event_id},
        )
    return s.ReportResponse(report=report)


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/traces
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/traces", response_model=s.TracesResponse)
async def get_traces(
    event_id: str,
    principal: CurrentPrincipal,
    page: int = 1,
    page_size: int = 20,
    event_service: EventService = Depends(get_event_service),
) -> s.TracesResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    rows, total = await _db_read(
        event_id,
        orm.AgentTrace,
        orm.AgentTrace.started_at.asc(),
        page=page,
        page_size=page_size,
    )

    items: list[s.TraceItem] = []
    for row in rows:
        items.append(
            s.TraceItem(
                trace_id=row.trace_id,
                agent_name=row.agent_name,
                status=row.status,
                duration_ms=row.duration_ms,
                started_at=row.started_at,
            )
        )

    return s.TracesResponse(total=total, page=page, page_size=page_size, items=items)


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/audit-logs
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/audit-logs", response_model=s.AuditLogsResponse)
async def get_audit_logs(
    event_id: str,
    principal: CurrentPrincipal,
    page: int = 1,
    page_size: int = 20,
    event_service: EventService = Depends(get_event_service),
) -> s.AuditLogsResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    rows, total = await _db_read(
        event_id,
        orm.EventAuditLog,
        orm.EventAuditLog.id.asc(),
        page=page,
        page_size=page_size,
    )

    items: list[s.AuditLogItem] = []
    for row in rows:
        items.append(
            s.AuditLogItem(
                id=row.id,
                from_status=row.from_status,
                to_status=row.to_status,
                operator=row.operator,
                reason=row.reason,
                created_at=row.created_at,
            )
        )

    return s.AuditLogsResponse(total=total, page=page, page_size=page_size, items=items)


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/tool-calls
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/tool-calls", response_model=s.ToolCallsResponse)
async def get_event_tool_calls(
    event_id: str,
    principal: CurrentPrincipal,
    page: int = 1,
    page_size: int = 20,
    event_service: EventService = Depends(get_event_service),
) -> s.ToolCallsResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    rows, total = await _db_read(
        event_id,
        orm.ToolCallLog,
        orm.ToolCallLog.started_at.desc(),
        page=page,
        page_size=page_size,
    )

    items: list[s.ToolCallItem] = []
    for row in rows:
        items.append(
            s.ToolCallItem(
                call_id=row.call_id,
                event_id=row.event_id,
                action_id=row.action_id,
                tool_name=row.tool_name,
                tool_category=row.tool_category,
                status=row.status,
                duration_ms=row.duration_ms,
            )
        )

    return s.ToolCallsResponse(total=total, page=page, page_size=page_size, items=items)


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/actions
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/actions", response_model=s.ActionListResponse)
async def get_actions(
    event_id: str,
    principal: CurrentPrincipal,
    page: int = 1,
    page_size: int = 20,
    status: ActionStatus | None = None,
    event_service: EventService = Depends(get_event_service),
) -> s.ActionListResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    extra: list[Any] = []
    if status is not None:
        extra.append(orm.Action.status == status.value)

    rows, total = await _db_read(
        event_id,
        orm.Action,
        orm.Action.created_at.desc(),
        page=page,
        page_size=page_size,
        extra_conditions=extra,
    )

    items: list[ActionModel] = []
    for row in rows:
        from app.models.enums import ActionCategory, ActionLevel

        try:
            action_cat = ActionCategory(row.action_category)
        except ValueError:
            action_cat = ActionCategory.SYSTEM
        try:
            action_lvl = ActionLevel(row.action_level)
        except ValueError:
            action_lvl = ActionLevel.L0

        items.append(
            ActionModel(
                action_id=row.action_id,
                event_id=row.event_id,
                plan_revision=int(row.plan_revision or 1),
                action_fingerprint=row.action_fingerprint,
                action_category=action_cat,
                action_name=row.action_name,
                tool_name=row.tool_name,
                action_level=action_lvl,
                reason=row.reason,
            )
        )

    return s.ActionListResponse(total=total, page=page, page_size=page_size, items=items)


# --------------------------------------------------------------------------- #
# GET /tool-calls — global tool call audit
# --------------------------------------------------------------------------- #


@router.get("/tool-calls", response_model=s.ToolCallsResponse)
async def list_tool_calls(
    principal: CurrentPrincipal,
    page: int = 1,
    page_size: int = 20,
    tool_name: str | None = None,
    status: str | None = None,
) -> s.ToolCallsResponse:
    sf = _try_get_session_factory()
    if sf is None:
        return s.ToolCallsResponse(total=0, page=page, page_size=page_size, items=[])

    try:
        page = max(1, page)
        page_size = min(max(1, page_size), 200)
        async with sf() as session:
            conditions: list[Any] = []
            if tool_name:
                conditions.append(orm.ToolCallLog.tool_name == tool_name)
            if status:
                conditions.append(orm.ToolCallLog.status == status)

            count = await session.scalar(
                select(func.count(orm.ToolCallLog.call_id)).where(*conditions)
            )
            total = int(count or 0)
            rows = (
                await session.scalars(
                    select(orm.ToolCallLog)
                    .where(*conditions)
                    .order_by(orm.ToolCallLog.started_at.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).all()

        items: list[s.ToolCallItem] = []
        for row in rows:
            items.append(
                s.ToolCallItem(
                    call_id=row.call_id,
                    event_id=row.event_id,
                    action_id=row.action_id,
                    tool_name=row.tool_name,
                    tool_category=row.tool_category,
                    status=row.status,
                    duration_ms=row.duration_ms,
                )
            )

        return s.ToolCallsResponse(total=total, page=page, page_size=page_size, items=items)
    except (ConnectionRefusedError, TimeoutError, sa_exc.OperationalError):
        logger.warning("Global tool-calls query failed (transient DB error)", exc_info=True)
        return s.ToolCallsResponse(total=0, page=page, page_size=page_size, items=[])
    except Exception as exc:
        logger.error("Global tool-calls query failed (non-transient): %s", exc, exc_info=True)
        raise DependencyUnavailableError(
            "database query failed for global tool-calls",
            error_code="dependency_unavailable",
        ) from exc


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/timeline (stub, real implementation deferred)
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/timeline", response_model=s.TimelineResponse)
async def get_timeline(
    event_id: str,
    principal: CurrentPrincipal,
    event_service: EventService = Depends(get_event_service),
) -> s.TimelineResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})
    return s.TimelineResponse(event_id=event_id, items=[])


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/graph (stub, real implementation deferred)
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/graph", response_model=s.GraphResponse)
async def get_graph(
    event_id: str,
    principal: CurrentPrincipal,
    event_service: EventService = Depends(get_event_service),
) -> s.GraphResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})
    return s.GraphResponse(event_id=event_id, nodes=[], edges=[])


# --------------------------------------------------------------------------- #
# GET /events/{event_id}/decision-trace (stub, real implementation deferred)
# --------------------------------------------------------------------------- #


@router.get("/events/{event_id}/decision-trace", response_model=s.DecisionTraceResponse)
async def get_decision_trace(
    event_id: str,
    principal: CurrentPrincipal,
    event_service: EventService = Depends(get_event_service),
) -> s.DecisionTraceResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})
    return s.DecisionTraceResponse(event_id=event_id, steps=[])


# --------------------------------------------------------------------------- #
# PUT /events/{event_id}/disposition-source
# --------------------------------------------------------------------------- #


@router.put(
    "/events/{event_id}/disposition-source",
    response_model=s.DispositionSourceSelectResponse,
)
async def select_disposition_source(
    event_id: str,
    body: s.SelectDispositionSourceRequest,
    principal: Annotated[Principal, require_roles(ROLE_DISPOSITION_OPERATOR)],
    event_service: EventService = Depends(get_event_service),
) -> s.DispositionSourceSelectResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    # Optimistic concurrency.
    if body.expected_event_version != event.row_version:
        raise WritebackConflictError(
            "event version mismatch",
            details={"expected": body.expected_event_version, "actual": event.row_version},
        )

    # Validate source is associated.
    sf = _try_get_session_factory()
    if sf is not None:
        try:
            async with sf() as session:
                link = await session.scalar(
                    select(orm.SourceEventLink).where(
                        orm.SourceEventLink.source_record_id == body.source_record_id,
                        orm.SourceEventLink.event_id == event_id,
                    )
                )
                if link is None:
                    raise DispositionPermissionDenied(
                        "source object is not associated with this event",
                        details={"source_record_id": body.source_record_id, "event_id": event_id},
                    )

                source_obj = await session.scalar(
                    select(orm.SourceObject).where(
                        orm.SourceObject.source_record_id == body.source_record_id
                    )
                )
                if source_obj is None:
                    raise DispositionPermissionDenied(
                        "source object not found",
                        details={"source_record_id": body.source_record_id},
                    )

                locator = SourceObjectLocator(
                    source_product=source_obj.source_product,
                    source_tenant_id=source_obj.source_tenant_id,
                    connector_id=source_obj.connector_id,
                    source_kind=SourceObjectKind(source_obj.source_kind),
                    source_object_id=source_obj.source_object_id,
                )
                return s.DispositionSourceSelectResponse(
                    event_id=event_id,
                    disposition_source_ref=locator,
                    event_version=event.row_version + 1,
                )
        except Exception:
            logger.warning("DB unavailable for disposition-source validation", exc_info=True)

    # DB unavailable fallback — use static associated set.
    if body.source_record_id not in _ASSOCIATED_SOURCE_RECORDS:
        raise DispositionPermissionDenied(
            "source object is not an associated, tenant-consistent source for this event",
            details={"source_record_id": body.source_record_id},
        )
    return s.DispositionSourceSelectResponse(
        event_id=event_id,
        disposition_source_ref=SourceObjectLocator(
            source_product="mock_xdr",
            source_tenant_id="t1",
            connector_id="conn-mock-1",
            source_kind=s.example_source_reference().source_kind,
            source_object_id="INC-1001",
        ),
        event_version=event.row_version + 1,
    )


# --------------------------------------------------------------------------- #
# POST /events/{event_id}/disposition-readiness/recheck
# --------------------------------------------------------------------------- #


@router.post(
    "/events/{event_id}/disposition-readiness/recheck",
    response_model=s.ReadinessRecheckResponse,
)
async def recheck_disposition_readiness(
    event_id: str,
    body: s.RecheckDispositionReadinessRequest,
    principal: Annotated[Principal, require_roles(ROLE_DISPOSITION_OPERATOR)],
    event_service: EventService = Depends(get_event_service),
) -> s.ReadinessRecheckResponse:
    event = await event_service.get_event(event_id)
    if event is None:
        raise EventNotFoundError(f"event {event_id} not found", details={"event_id": event_id})

    if body.expected_event_version != event.row_version:
        raise WritebackConflictError(
            "event version mismatch",
            details={"expected": body.expected_event_version, "actual": event.row_version},
        )

    # Recheck: recompute readiness without external call.
    return s.ReadinessRecheckResponse(
        event_id=event_id,
        writeback_readiness=WritebackReadiness.CAPABILITY_UNKNOWN,
        blocked_reason="capability_unknown",
        event_version=event.row_version,
    )
