"""EventService — unified internal event creation & query (ISSUE-015).

Does **not** call external systems and does **not** set ``Action.writeback_required``.
Status mutations are delegated exclusively to ``StateMachineService.transition``
(ISSUE-037); this service never writes ``security_event.status`` directly.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import orjson
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, case, delete, func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.v1.schemas import EventSummary
from app.core.config import get_settings
from app.core.errors import DependencyUnavailableError, EventNotFoundError, ValidationError
from app.core.event_bus import EventBus
from app.db import models as orm
from app.models.disposition import SourceObjectLocator
from app.models.entities import EntitySet
from app.models.enums import (
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    SourceObjectKind,
)
from app.models.ids import canonical_source_identity, new_action_id, new_event_id
from app.models.report import InvestigationReport
from app.models.security_event import SecurityEvent
from app.models.source import SourceReference
from app.models.tool_meta import TERMINAL_DISPOSITION_TOOL
from app.models.workflow import TransitionContext, validate_verdict_status
from app.services.context_service import EventContextStore, event_summary_from_security_event
from app.services.degraded_flag_service import DegradedFlagService
from app.services.evidence_projection import EvidenceQueryScope
from app.services.source_policy_resolver import (
    SourcePolicyResolver,
    connector_policy_from_row,
)

logger = logging.getLogger(__name__)

FILE_DEDUP_WINDOW = timedelta(hours=1)
LINK_ROLE_PRIMARY = "primary"
LINK_ROLE_RELATED = "related"
LINK_ROLE_PROVISIONAL = "provisional"
PROMOTION_NONE = "none"
PROMOTION_PROMOTED = "promoted"


def should_apply_source_update(
    *,
    stored_updated_at: datetime | None,
    stored_token: str | None,
    incoming_updated_at: datetime | None,
    incoming_token: str | None,
) -> bool:
    """Accept only demonstrably newer mutable source state."""
    if stored_updated_at is not None and incoming_updated_at is not None:
        stored = (
            stored_updated_at.astimezone(UTC)
            if stored_updated_at.tzinfo is not None
            else stored_updated_at.replace(tzinfo=UTC)
        )
        incoming = (
            incoming_updated_at.astimezone(UTC)
            if incoming_updated_at.tzinfo is not None
            else incoming_updated_at.replace(tzinfo=UTC)
        )
        if incoming != stored:
            return incoming > stored
        return stored_token is None and incoming_token is not None
    if stored_updated_at is not None:
        return False
    if incoming_updated_at is not None:
        return True
    return stored_token is None and incoming_token is not None


class StateMachinePort(Protocol):
    """ISSUE-037 surface used by EventService (injected when available)."""

    async def transition(
        self,
        event_id: str,
        target: EventStatus,
        *,
        context: TransitionContext | None = None,
        operator: str | None = None,
        reason: str | None = None,
    ) -> SecurityEvent: ...


class IngestableSource(BaseModel):
    """Normalized ingest envelope for one external source object."""

    model_config = ConfigDict(extra="forbid")

    reference: SourceReference
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    normalized: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    description: str = ""
    event_type: EventType | None = None
    severity: Severity | None = None
    occurred_at: datetime | None = None
    # Explicit Adapter-verified associations only (never inferred).
    incident_ref: SourceReference | None = None
    related_alert_refs: list[SourceReference] = Field(default_factory=list)
    source_type: str | None = None  # mock_xdr / file / manual / …


@dataclass(frozen=True, slots=True)
class IngestResult:
    source_record_id: str
    event_id: str | None
    accepted: bool = True
    created: bool = False
    promoted: bool = False
    related_only: bool = False
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class EventListResult:
    items: list[SecurityEvent]
    total: int
    page: int
    page_size: int


@dataclass
class _CreateBundle:
    event: orm.SecurityEvent
    source_record_id: str
    created: bool
    promoted: bool = False
    related_only: bool = False
    idempotent: bool = False
    merged_event_ids: tuple[str, ...] = ()


def stable_source_record_id(*, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"src-{digest}"


def locator_from_reference(ref: SourceReference) -> SourceObjectLocator:
    return SourceObjectLocator(
        source_product=ref.source_product,
        source_tenant_id=ref.source_tenant_id,
        connector_id=ref.connector_id,
        source_kind=ref.source_kind,
        source_object_type=ref.source_object_type,
        source_object_id=ref.source_object_id,
    )


def _ref_dump(ref: SourceReference) -> dict[str, Any]:
    return ref.model_dump(mode="json")


def _source_snapshot_from_row(row: orm.SecurityEvent) -> dict[str, Any]:
    """Return immutable source evidence only; never include mutable current_* state."""
    return {
        "creation_source_ref": dict(row.creation_source_ref),
        "source_reference_snapshots": [
            dict(item) for item in (row.source_reference_snapshots or [])
        ],
        "raw_alert_snapshot": (
            dict(row.raw_alert_snapshot) if row.raw_alert_snapshot is not None else None
        ),
    }


def _security_event_from_row(row: orm.SecurityEvent) -> SecurityEvent:
    creation = SourceReference.model_validate(row.creation_source_ref)
    snapshots = [SourceReference.model_validate(s) for s in (row.source_reference_snapshots or [])]
    disposition = None
    if row.disposition_source_ref:
        disposition = SourceObjectLocator.model_validate(row.disposition_source_ref)
    entities_raw = row.entities or {}
    try:
        entities = EntitySet.model_validate(entities_raw)
    except Exception:  # noqa: BLE001 — tolerate sparse ORM JSON
        entities = EntitySet()
    return SecurityEvent(
        event_id=row.event_id,
        event_type=EventType(row.event_type),
        title=row.title,
        description=row.description or "",
        status=EventStatus(row.status),
        severity=Severity(row.severity),
        risk_score=int(row.risk_score or 0),
        confidence=float(row.confidence or 0.0),
        final_verdict=FinalVerdict(row.final_verdict),
        entities=entities,
        creation_source_ref=creation,
        source_reference_snapshots=snapshots,
        current_primary_source_record_id=row.current_primary_source_record_id,
        disposition_source_ref=disposition,
        disposition_policy=DispositionPolicy(row.disposition_policy),
        raw_alert_ids=list(row.raw_alert_ids or []),
        raw_alert_snapshot=row.raw_alert_snapshot,
        source_type=row.source_type,
        occurred_at=row.occurred_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        closed_at=row.closed_at,
        replan_count=int(row.replan_count or 0),
        degraded_flags=[str(f) for f in (row.degraded_flags or [])],
        escalated=bool(row.escalated),
        external_unsynced=bool(row.external_unsynced),
        event_context_snapshot=row.event_context_snapshot,
        row_version=int(row.row_version or 1),
    )


class EventService:
    """Create / query SecurityEvent records; never mutate status directly."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        store: EventContextStore,
        *,
        degraded_flags: DegradedFlagService,
        event_bus: EventBus | None = None,
        policy_resolver: SourcePolicyResolver | None = None,
        state_machine: StateMachinePort | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._store = store
        self._bus = event_bus
        self._degraded = degraded_flags
        self._policy = policy_resolver or SourcePolicyResolver()
        self._state_machine = state_machine

    # ------------------------------------------------------------------ #
    # Ingest / create
    # ------------------------------------------------------------------ #

    async def ingest_source_object(self, source_object: IngestableSource) -> IngestResult:
        """Upsert source_object and attach / create / promote an internal event."""
        bundle = await self._ingest_with_unique_retry(source_object)
        await self._post_create_side_effects(
            bundle.event,
            force_context_refresh=not bundle.idempotent,
            publish_event=bundle.created or bundle.promoted,
        )
        for merged_event_id in bundle.merged_event_ids:
            await self._store.delete_cached_context(merged_event_id)
        return IngestResult(
            source_record_id=bundle.source_record_id,
            event_id=bundle.event.event_id,
            accepted=True,
            created=bundle.created,
            promoted=bundle.promoted,
            related_only=bundle.related_only,
            idempotent=bundle.idempotent,
        )

    async def create_event_from_source(
        self, primary_ref: SourceReference, **kwargs: Any
    ) -> SecurityEvent:
        """Create (or idempotently return) an event for a primary source reference."""
        ingest = IngestableSource(reference=primary_ref, **kwargs)
        bundle = await self._ingest_with_unique_retry(ingest)
        await self._post_create_side_effects(
            bundle.event,
            force_context_refresh=not bundle.idempotent,
            publish_event=bundle.created or bundle.promoted,
        )
        for merged_event_id in bundle.merged_event_ids:
            await self._store.delete_cached_context(merged_event_id)
        return _security_event_from_row(bundle.event)

    async def create_event(
        self,
        raw_alert: dict[str, Any],
        source_type: str = "file",
        *,
        title: str | None = None,
        event_type: EventType = EventType.OTHER,
        severity: Severity = Severity.LOW,
        occurred_at: datetime | None = None,
        primary_entity: str | None = None,
    ) -> SecurityEvent:
        """File / manual fallback create path (no stable external source ID)."""
        now = occurred_at or datetime.now(UTC)
        payload_hash = hashlib.sha256(
            orjson.dumps(raw_alert, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()
        entity_key = primary_entity or str(raw_alert.get("entity") or "unknown")
        identity = f"file|{entity_key}|{payload_hash}"
        event_id = new_event_id(identity, now)

        lock_material = f"{source_type}|{entity_key}|{payload_hash}".encode()
        advisory_lock_key = int.from_bytes(
            hashlib.sha256(lock_material).digest()[:8], byteorder="big", signed=True
        )
        created = False
        async with self._session_factory() as session:
            async with session.begin():
                # Serialize the file/manual soft-dedup key across workers. Unlike a
                # process lock, this also covers multiple API workers and midnight
                # crossings where deterministic event IDs can differ.
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_key)"),
                    {"lock_key": advisory_lock_key},
                )
                row = await session.get(orm.SecurityEvent, event_id)
                if row is None:
                    # Soft dedup: same payload_hash + primary entity within 1h.
                    window_start = now - FILE_DEDUP_WINDOW
                    candidates = (
                        await session.scalars(
                            select(orm.SecurityEvent).where(
                                orm.SecurityEvent.source_type == source_type,
                                orm.SecurityEvent.occurred_at >= window_start,
                                orm.SecurityEvent.occurred_at <= now + timedelta(seconds=1),
                            )
                        )
                    ).all()
                    row = next(
                        (
                            candidate
                            for candidate in candidates
                            if (candidate.raw_alert_snapshot or {}).get("payload_hash")
                            == payload_hash
                            and (candidate.raw_alert_snapshot or {}).get("primary_entity")
                            == entity_key
                        ),
                        None,
                    )

                if row is None:
                    creation_ref = SourceReference(
                        source_kind=SourceObjectKind.ALERT,
                        source_product="file",
                        source_tenant_id="local",
                        connector_id="file-local",
                        source_object_id=f"file-{payload_hash[:12]}",
                        raw_payload_hash=payload_hash,
                        ingested_at=now,
                    )
                    row = orm.SecurityEvent(
                        event_id=event_id,
                        event_type=event_type.value,
                        title=title or str(raw_alert.get("title") or "file alert"),
                        description=str(raw_alert.get("description") or ""),
                        status=EventStatus.NEW.value,
                        severity=severity.value,
                        final_verdict=FinalVerdict.NONE.value,
                        entities={},
                        creation_source_ref=_ref_dump(creation_ref),
                        source_reference_snapshots=[_ref_dump(creation_ref)],
                        disposition_source_ref=None,
                        disposition_policy=DispositionPolicy.NOT_REQUIRED.value,
                        raw_alert_ids=[creation_ref.source_object_id],
                        raw_alert_snapshot={
                            "payload_hash": payload_hash,
                            "primary_entity": entity_key,
                            "raw": raw_alert,
                        },
                        source_type=source_type,
                        occurred_at=now,
                    )
                    session.add(row)
                    session.add(
                        orm.EventAuditLog(
                            event_id=event_id,
                            from_status=None,
                            to_status=EventStatus.NEW.value,
                            operator="EventService",
                            reason="event_created",
                        )
                    )
                    await session.flush()
                    await session.refresh(row)
                    created = True

        await self._post_create_side_effects(
            row,
            force_context_refresh=created,
            publish_event=created,
        )
        return _security_event_from_row(row)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    async def get_event(self, event_id: str) -> SecurityEvent | None:
        async with self._session_factory() as session:
            row = await session.get(orm.SecurityEvent, event_id)
            return _security_event_from_row(row) if row else None

    async def get_evidence_query_scope(self, event_id: str) -> EvidenceQueryScope:
        """Derive the only permitted evidence tenant/connectors from trusted event state."""
        event = await self.get_event(event_id)
        if event is None:
            raise EventNotFoundError(
                f"security_event not found: {event_id}",
                details={"event_id": event_id},
            )
        tenant_id = event.creation_source_ref.source_tenant_id
        references = [event.creation_source_ref, *event.source_reference_snapshots]
        products_by_connector: dict[str, str] = {}
        for reference in references:
            if reference.source_tenant_id != tenant_id:
                raise ValidationError(
                    "event source references span multiple source tenants",
                    error_code="adapter_validation_error",
                    details={
                        "event_id": event_id,
                        "expected_source_tenant_id": tenant_id,
                        "conflicting_source_tenant_id": reference.source_tenant_id,
                        "connector_id": reference.connector_id,
                    },
                )
            existing_product = products_by_connector.get(reference.connector_id)
            if existing_product not in (None, reference.source_product):
                raise ValidationError(
                    "event connector has conflicting source product ownership",
                    error_code="adapter_validation_error",
                    details={
                        "event_id": event_id,
                        "connector_id": reference.connector_id,
                        "existing_source_product": existing_product,
                        "conflicting_source_product": reference.source_product,
                    },
                )
            products_by_connector[reference.connector_id] = reference.source_product

        connector_ids = frozenset(products_by_connector)
        async with self._session_factory() as session:
            connectors = (
                await session.scalars(
                    select(orm.SourceConnector).where(
                        orm.SourceConnector.connector_id.in_(connector_ids)
                    )
                )
            ).all()
        for connector in connectors:
            expected_product = products_by_connector[connector.connector_id]
            metadata_tenant = (connector.connector_metadata or {}).get("source_tenant_id")
            if connector.source_product != expected_product or metadata_tenant not in (
                None,
                tenant_id,
            ):
                raise ValidationError(
                    "event connector ownership conflicts with trusted event scope",
                    error_code="adapter_validation_error",
                    details={
                        "event_id": event_id,
                        "connector_id": connector.connector_id,
                        "expected_source_product": expected_product,
                        "existing_source_product": connector.source_product,
                        "expected_source_tenant_id": tenant_id,
                        "existing_source_tenant_id": metadata_tenant,
                    },
                )
        return EvidenceQueryScope(
            source_tenant_id=tenant_id,
            connector_ids=connector_ids,
        )

    # Whitelist of columns allowed for sort_by in list_events.
    _SORT_COLUMN_MAP: dict[str, Any] = {
        "created_at": orm.SecurityEvent.created_at,
        "updated_at": orm.SecurityEvent.updated_at,
        "occurred_at": orm.SecurityEvent.occurred_at,
        "severity": orm.SecurityEvent.severity,
        "risk_score": orm.SecurityEvent.risk_score,
        "status": orm.SecurityEvent.status,
        "event_type": orm.SecurityEvent.event_type,
    }

    async def list_events(
        self,
        *,
        status: EventStatus | str | None = None,
        severity: Severity | str | None = None,
        event_type: EventType | str | None = None,
        final_verdict: FinalVerdict | str | None = None,
        keyword: str | None = None,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> EventListResult:
        page = max(1, page)
        page_size = min(max(1, page_size), 200)
        filters: list[Any] = []
        if status is not None:
            filters.append(
                orm.SecurityEvent.status
                == (status.value if isinstance(status, EventStatus) else status)
            )
        if severity is not None:
            filters.append(
                orm.SecurityEvent.severity
                == (severity.value if isinstance(severity, Severity) else severity)
            )
        if event_type is not None:
            filters.append(
                orm.SecurityEvent.event_type
                == (event_type.value if isinstance(event_type, EventType) else event_type)
            )
        if final_verdict is not None:
            filters.append(
                orm.SecurityEvent.final_verdict
                == (
                    final_verdict.value
                    if isinstance(final_verdict, FinalVerdict)
                    else final_verdict
                )
            )
        if keyword:
            like = f"%{keyword}%"
            filters.append(
                or_(
                    orm.SecurityEvent.title.ilike(like),
                    orm.SecurityEvent.description.ilike(like),
                )
            )
        if occurred_after is not None:
            filters.append(orm.SecurityEvent.occurred_at >= occurred_after)
        if occurred_before is not None:
            filters.append(orm.SecurityEvent.occurred_at <= occurred_before)

        # Resolve sort column (whitelist only; default to created_at).
        sort_col = self._SORT_COLUMN_MAP.get(
            sort_by or "created_at", orm.SecurityEvent.created_at
        )
        descending = (sort_order or "desc") != "asc"

        async with self._session_factory() as session:
            count_stmt = select(func.count()).select_from(orm.SecurityEvent)
            list_stmt = select(orm.SecurityEvent).order_by(
                sort_col.desc() if descending else sort_col.asc()
            )
            if filters:
                count_stmt = count_stmt.where(and_(*filters))
                list_stmt = list_stmt.where(and_(*filters))
            total = int(await session.scalar(count_stmt) or 0)
            rows = (
                await session.scalars(list_stmt.offset((page - 1) * page_size).limit(page_size))
            ).all()
            items = [_security_event_from_row(r) for r in rows]
        return EventListResult(items=items, total=total, page=page, page_size=page_size)

    # ------------------------------------------------------------------ #
    # Verdict / status (status via StateMachineService only)
    # ------------------------------------------------------------------ #

    async def set_final_verdict(
        self,
        event_id: str,
        verdict: FinalVerdict,
        *,
        operator: str | None = None,
        context: TransitionContext | None = None,
    ) -> SecurityEvent:
        """Sole path for writing ``final_verdict`` + publishing ``final_verdict_updated``."""
        # ``context`` remains in the public signature for compatibility, but trusted
        # gate projections are always rebuilt below from PostgreSQL.
        _ = context
        changed = False
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(
                    orm.SecurityEvent,
                    event_id,
                    with_for_update=True,
                )
                if row is None:
                    raise KeyError(f"security_event not found: {event_id}")

                ctx = await self._authoritative_verdict_context(session, event_id)

                validate_verdict_status(verdict, EventStatus(row.status), ctx)

                previous = row.final_verdict
                if previous != verdict.value:
                    changed = True
                    row.final_verdict = verdict.value
                    row.row_version = int(row.row_version or 1) + 1
                    session.add(
                        orm.EventAuditLog(
                            event_id=event_id,
                            from_status=row.status,
                            to_status=row.status,
                            operator=operator or "EventService",
                            reason=f"final_verdict:{previous}->{verdict.value}",
                        )
                    )
                    await session.flush()
                    await session.refresh(row)
                result = _security_event_from_row(row)
                summary = event_summary_from_security_event(row)

        if changed:
            await self._sync_event_summary_after_mutation(
                event_id,
                committed_version=result.row_version,
                summary=summary,
            )
            if self._bus is not None:
                await self._bus.publish_event(
                    event_id,
                    "final_verdict_updated",
                    {"final_verdict": verdict.value, "operator": operator},
                )
        return result

    async def update_risk_fields(
        self,
        event_id: str,
        *,
        risk_score: int,
        severity: Severity,
        confidence: float,
        operator: str | None = None,
        factor_names: list[str] | None = None,
    ) -> SecurityEvent:
        """Persist RiskAgent score fields onto ``security_event`` (ISSUE-035).

        Does **not** write ``final_verdict`` — that remains ``set_final_verdict`` only.
        Publishes ``risk_updated`` (locked Socket payload: ``RiskUpdatedPayload``).
        """
        score = max(0, min(100, int(risk_score)))
        conf = max(0.0, min(1.0, float(confidence)))
        previous_score = 0
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(
                    orm.SecurityEvent,
                    event_id,
                    with_for_update=True,
                )
                if row is None:
                    raise KeyError(f"security_event not found: {event_id}")
                previous_score = int(row.risk_score or 0)
                row.risk_score = score
                row.severity = severity.value if isinstance(severity, Severity) else str(severity)
                row.confidence = conf
                row.row_version = int(row.row_version or 1) + 1
                session.add(
                    orm.EventAuditLog(
                        event_id=event_id,
                        from_status=row.status,
                        to_status=row.status,
                        operator=operator or "RiskAgent",
                        reason=(
                            f"risk_fields:score={score},"
                            f"severity={row.severity},confidence={conf:.4f}"
                        ),
                    )
                )
                await session.flush()
                await session.refresh(row)
                result = _security_event_from_row(row)
                summary = event_summary_from_security_event(row)

        await self._sync_event_summary_after_mutation(
            event_id,
            committed_version=result.row_version,
            summary=summary,
        )
        if self._bus is not None:
            payload: dict[str, Any] = {"risk_score": score}
            if previous_score != score:
                payload["previous_score"] = previous_score
            if factor_names:
                payload["factors"] = list(factor_names)
            await self._bus.publish_event(event_id, "risk_updated", payload)
        return result

    async def upsert_report(self, report: InvestigationReport) -> InvestigationReport:
        """Idempotent upsert of InvestigationReport by stable ``report_id`` (ISSUE-036)."""
        now = datetime.now(UTC)
        sections_payload = [section.model_dump(mode="json") for section in report.sections]
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.get(
                    orm.Report,
                    report.report_id,
                    with_for_update=True,
                )
                if row is None:
                    row = orm.Report(
                        report_id=report.report_id,
                        event_id=report.event_id,
                        title=report.title,
                        summary=report.summary,
                        sections=sections_payload,
                        final_verdict=report.final_verdict.value,
                        risk_score=int(report.risk_score),
                        severity=report.severity.value,
                        version=1,
                        generated_by=report.generated_by,
                        generated_at=report.generated_at or now,
                        updated_at=now,
                    )
                    session.add(row)
                else:
                    if row.event_id != report.event_id:
                        raise ValidationError(
                            "report_id already bound to a different event_id",
                            details={
                                "report_id": report.report_id,
                                "existing_event_id": row.event_id,
                                "incoming_event_id": report.event_id,
                            },
                        )
                    row.title = report.title
                    row.summary = report.summary
                    row.sections = sections_payload
                    row.final_verdict = report.final_verdict.value
                    row.risk_score = int(report.risk_score)
                    row.severity = report.severity.value
                    row.version = int(row.version or 1) + 1
                    row.generated_by = report.generated_by
                    if report.generated_at is not None:
                        row.generated_at = report.generated_at
                    row.updated_at = now
                await session.flush()
                await session.refresh(row)
                return InvestigationReport(
                    report_id=row.report_id,
                    event_id=row.event_id,
                    title=row.title,
                    summary=row.summary,
                    sections=report.sections,
                    final_verdict=FinalVerdict(row.final_verdict),
                    risk_score=int(row.risk_score),
                    severity=Severity(row.severity),
                    version=int(row.version),
                    generated_by=row.generated_by,
                    generated_at=row.generated_at,
                    updated_at=row.updated_at,
                )

    async def get_report(
        self,
        *,
        report_id: str | None = None,
        event_id: str | None = None,
    ) -> InvestigationReport | None:
        """Load a persisted report by ``report_id`` or ``event_id``."""
        if report_id is None and event_id is None:
            raise ValidationError("get_report requires report_id or event_id")
        async with self._session_factory() as session:
            row: orm.Report | None = None
            if report_id is not None:
                row = await session.get(orm.Report, report_id)
            elif event_id is not None:
                row = await session.scalar(
                    select(orm.Report)
                    .where(orm.Report.event_id == event_id)
                    .order_by(orm.Report.updated_at.desc())
                    .limit(1)
                )
            if row is None:
                return None
            from app.models.report import ReportSection

            sections = [ReportSection.model_validate(item) for item in (row.sections or [])]
            return InvestigationReport(
                report_id=row.report_id,
                event_id=row.event_id,
                title=row.title,
                summary=row.summary,
                sections=sections,
                final_verdict=FinalVerdict(row.final_verdict),
                risk_score=int(row.risk_score),
                severity=Severity(row.severity),
                version=int(row.version),
                generated_by=row.generated_by,
                generated_at=row.generated_at,
                updated_at=row.updated_at,
            )

    async def upsert_generate_report_action(
        self,
        event_id: str,
        *,
        plan_revision: int = 1,
    ) -> str:
        """Idempotent system Action for local report generation (ISSUE-036)."""
        now = datetime.now(UTC)
        material = f"{event_id}|{int(plan_revision)}|generate_report|system|system||immediate|"
        fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()
        async with self._session_factory() as session:
            async with session.begin():
                existing = await session.scalar(
                    select(orm.Action).where(orm.Action.action_fingerprint == fingerprint)
                )
                if existing is not None:
                    existing.status = "success"
                    existing.executed_at = now
                    existing.updated_at = now
                    existing.reason = "报告自动生成"
                    await session.flush()
                    return existing.action_id

                action_id = new_action_id()
                session.add(
                    orm.Action(
                        action_id=action_id,
                        event_id=event_id,
                        plan_revision=int(plan_revision),
                        action_fingerprint=fingerprint,
                        action_category="system",
                        action_name="generate_report",
                        tool_name="generate_report",
                        action_level="l0",
                        target_type="system",
                        target="system",
                        parameters={},
                        status="success",
                        auto_execute=True,
                        reason="报告自动生成",
                        impact_assessment=None,
                        execution_owner=None,
                        writeback_required=False,
                        writeback_applicable=False,
                        writeback_readiness="not_required",
                        writeback_status=None,
                        executed_at=now,
                        source_action_id=None,
                    )
                )
                await session.flush()
                return action_id

    async def transition_status(
        self,
        event_id: str,
        target: EventStatus,
        *,
        context: TransitionContext | None = None,
        operator: str | None = None,
        reason: str | None = None,
    ) -> SecurityEvent:
        """Delegate status change to StateMachineService (ISSUE-037).

        EventService never writes ``security_event.status`` itself.  All
        validation — including the CLOSED writeback gate — happens inside
        ``StateMachineService.transition()`` under ``SELECT … FOR UPDATE``.
        No pre-validation is done here; the single authoritative path avoids
        TOCTOU windows, duplicate DB queries, and stale context projections.
        """
        if self._state_machine is None:
            raise DependencyUnavailableError(
                "StateMachineService is required for status transitions",
                details={"event_id": event_id, "target": target.value},
            )
        return await self._state_machine.transition(
            event_id,
            target,
            context=context,
            operator=operator,
            reason=reason,
        )

    # Intentionally NO update_event_status — status writes live in ISSUE-037 only.

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _sync_event_summary_after_mutation(
        self,
        event_id: str,
        *,
        committed_version: int,
        summary: EventSummary,
    ) -> None:
        """Sync Context, then reconcile if a newer event commit won the race."""
        context_result = await self._store.set(event_id, "event", summary)
        redis_ok = context_result.redis_ok

        # Another transaction may commit after this caller releases its row lock
        # but before this Context write. Re-read PG and make the newest row win.
        async with self._session_factory() as session:
            latest = await session.get(orm.SecurityEvent, event_id)
        if latest is not None and int(latest.row_version or 1) != committed_version:
            latest_result = await self._store.set(
                event_id,
                "event",
                event_summary_from_security_event(latest),
            )
            redis_ok = redis_ok and latest_result.redis_ok

        if not redis_ok:
            logger.warning(
                "Redis context event sync failed for event_id=%s; marking degraded",
                event_id,
            )
            await self._degraded.set_flag(
                event_id,
                "redis_context_unavailable",
                True,
                writer="EventService",
            )

    async def _authoritative_verdict_context(
        self, session: AsyncSession, event_id: str
    ) -> TransitionContext:
        """Build trusted verdict gates from PostgreSQL, never caller input."""
        journal_value = await session.scalar(
            select(orm.EventContextJournal.value)
            .where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name == "disposition_only_intent",
            )
            .order_by(orm.EventContextJournal.version.desc())
            .limit(1)
        )
        if isinstance(journal_value, dict) and set(journal_value) == {"_scalar"}:
            journal_value = journal_value["_scalar"]
        disposition_only_intent = journal_value is True

        current_revision = await session.scalar(
            select(func.max(orm.Action.plan_revision)).where(orm.Action.event_id == event_id)
        )
        response_actions: list[orm.Action] = []
        if current_revision is not None:
            response_actions = list(
                (
                    await session.scalars(
                        select(orm.Action).where(
                            orm.Action.event_id == event_id,
                            orm.Action.plan_revision == current_revision,
                            orm.Action.action_category == "response",
                            orm.Action.superseded_by_revision.is_(None),
                        )
                    )
                ).all()
            )

        response_actions_are_disposition_only: bool | None = None
        has_entity_side_effect_actions = False
        if response_actions:
            response_actions_are_disposition_only = all(
                action.action_name == TERMINAL_DISPOSITION_TOOL for action in response_actions
            )
            has_entity_side_effect_actions = any(
                action.action_name != TERMINAL_DISPOSITION_TOOL for action in response_actions
            )

        return TransitionContext(
            disposition_only_intent=disposition_only_intent,
            response_actions_are_disposition_only=response_actions_are_disposition_only,
            has_entity_side_effect_actions=has_entity_side_effect_actions,
        )

    async def _post_create_side_effects(
        self,
        row: orm.SecurityEvent,
        *,
        force_context_refresh: bool,
        publish_event: bool,
    ) -> None:
        """Idempotently ensure Context after the authoritative event commit.

        An earlier request may have committed ``security_event`` and then failed
        before ``init_context``. Repeated delivery must repair that partial state
        rather than returning a permanently context-less event.

        ``event_created`` is published only for created/promoted paths — never for
        context-repair / idempotent losers — to avoid duplicate bus events.
        """
        summary = event_summary_from_security_event(row)
        init_result = await self._store.init_context(row.event_id, summary)
        initialized_now = init_result.initialized
        redis_ok = init_result.redis_ok
        if not initialized_now and force_context_refresh:
            set_result = await self._store.set(row.event_id, "event", summary)
            redis_ok = redis_ok and set_result.redis_ok

        snapshot_result = await self._ensure_source_snapshot(
            row,
            overwrite=force_context_refresh,
        )
        redis_ok = redis_ok and snapshot_result

        if not redis_ok:
            logger.warning(
                "Redis context sync failed for event_id=%s; marking degraded",
                row.event_id,
            )
            await self._degraded.set_flag(
                row.event_id,
                "redis_context_unavailable",
                True,
                writer="EventService",
            )
        if self._bus is not None and publish_event:
            await self._bus.publish_event(
                row.event_id,
                "event_created",
                {
                    "status": row.status,
                    "event_type": row.event_type,
                    "title": row.title,
                },
            )

    async def _ensure_source_snapshot(
        self,
        row: orm.SecurityEvent,
        *,
        overwrite: bool,
    ) -> bool:
        """Write immutable source evidence; repair when the field was never initialized.

        ``overwrite=True`` (create/promote) refreshes snapshots after association
        changes. ``overwrite=False`` (idempotent replay) only fills a missing
        field so a crash between ``event`` init and snapshot write can heal.
        """
        snapshot = _source_snapshot_from_row(row)
        if not overwrite:
            async with self._session_factory() as session:
                exists = await session.scalar(
                    select(orm.EventContextFieldVersion.current_version).where(
                        orm.EventContextFieldVersion.event_id == row.event_id,
                        orm.EventContextFieldVersion.field_name == "source_snapshot",
                    )
                )
            if exists is not None:
                return True
        result = await self._store.set(row.event_id, "source_snapshot", snapshot)
        return result.redis_ok

    async def _ingest_with_unique_retry(self, source: IngestableSource) -> _CreateBundle:
        """Resolve concurrent delivery races by rereading the canonical row/link."""
        try:
            return await self._ingest(source)
        except IntegrityError as exc:
            if getattr(exc.orig, "sqlstate", None) != "23505":
                raise
            logger.info(
                "Concurrent source ingest won by another transaction; rereading "
                "canonical link connector_id=%s source_object_id=%s",
                source.reference.connector_id,
                source.reference.source_object_id,
            )
            return await self._ingest(source)

    @staticmethod
    def _validate_explicit_associations(source: IngestableSource) -> None:
        """Reject malformed or cross-connector explicit associations."""
        primary = source.reference
        associations: list[tuple[SourceReference, SourceObjectKind]] = []
        if source.incident_ref is not None:
            associations.append((source.incident_ref, SourceObjectKind.INCIDENT))
        associations.extend(
            (related, SourceObjectKind.ALERT) for related in source.related_alert_refs
        )
        primary_scope = (
            primary.source_product,
            primary.source_tenant_id,
            primary.connector_id,
        )
        for related, expected_kind in associations:
            related_scope = (
                related.source_product,
                related.source_tenant_id,
                related.connector_id,
            )
            if related.source_kind is not expected_kind or related_scope != primary_scope:
                raise ValidationError(
                    "explicit source association has invalid kind or source scope",
                    error_code="adapter_validation_error",
                    details={
                        "source_object_id": primary.source_object_id,
                        "related_source_object_id": related.source_object_id,
                        "expected_kind": expected_kind.value,
                    },
                )

    async def _ingest(self, source: IngestableSource) -> _CreateBundle:
        self._validate_explicit_associations(source)
        ref = source.reference
        identity = canonical_source_identity(
            source_product=ref.source_product,
            source_tenant_id=ref.source_tenant_id,
            connector_id=ref.connector_id,
            source_kind=ref.source_kind.value,
            source_object_id=ref.source_object_id,
        )
        source_record_id = stable_source_record_id(identity=identity)
        occurred = source.occurred_at or ref.source_updated_at or datetime.now(UTC)

        async with self._session_factory() as session:
            async with session.begin():
                await self._ensure_connector(session, source)
                obj = await self._upsert_source_object(session, source, source_record_id)

                # Idempotent: same source object already linked.
                existing_link = await session.scalar(
                    select(orm.SourceEventLink)
                    .where(orm.SourceEventLink.source_record_id == source_record_id)
                    .order_by(
                        case(
                            (orm.SourceEventLink.role == LINK_ROLE_PRIMARY, 0),
                            (orm.SourceEventLink.role == LINK_ROLE_PROVISIONAL, 1),
                            else_=2,
                        ),
                        orm.SourceEventLink.id,
                    )
                )
                if existing_link is not None:
                    event = await session.get(orm.SecurityEvent, existing_link.event_id)
                    assert event is not None
                    return _CreateBundle(
                        event=event,
                        source_record_id=source_record_id,
                        created=False,
                        idempotent=True,
                    )

                # Alert with verified parent Incident → try merge into parent event.
                if ref.source_kind is SourceObjectKind.ALERT and source.incident_ref is not None:
                    parent_bundle = await self._link_alert_to_incident_event(
                        session, source, obj, source_record_id
                    )
                    if parent_bundle is not None:
                        return parent_bundle

                # Incident with verified related alerts → promote provisional children.
                if ref.source_kind is SourceObjectKind.INCIDENT and source.related_alert_refs:
                    promoted = await self._promote_or_relate_alerts(
                        session, source, obj, source_record_id, occurred
                    )
                    if promoted is not None:
                        return promoted

                # Fresh event (provisional for orphan alert; primary for incident/other).
                event = await self._create_new_event(
                    session, source, obj, source_record_id, occurred
                )
                return _CreateBundle(
                    event=event,
                    source_record_id=source_record_id,
                    created=True,
                )

    async def _ensure_connector(
        self, session: AsyncSession, source: IngestableSource
    ) -> orm.SourceConnector:
        ref = source.reference
        connector = await session.get(orm.SourceConnector, ref.connector_id)
        if connector is not None:
            metadata = dict(connector.connector_metadata or {})
            metadata_tenant = metadata.get("source_tenant_id")
            if metadata_tenant is None:
                existing_tenants = set(
                    (
                        await session.scalars(
                            select(orm.SourceObject.source_tenant_id)
                            .where(orm.SourceObject.connector_id == ref.connector_id)
                            .distinct()
                        )
                    ).all()
                )
            else:
                existing_tenants = {str(metadata_tenant)}
            if connector.source_product != ref.source_product or existing_tenants - {
                ref.source_tenant_id
            }:
                raise ValidationError(
                    "connector tenant or product ownership conflicts with source reference",
                    error_code="adapter_validation_error",
                    details={
                        "connector_id": ref.connector_id,
                        "existing_source_product": connector.source_product,
                        "incoming_source_product": ref.source_product,
                        "existing_source_tenant_ids": sorted(existing_tenants),
                        "incoming_source_tenant_id": ref.source_tenant_id,
                    },
                )
            metadata["source_tenant_id"] = ref.source_tenant_id
            if source.source_type:
                existing_adapter = metadata.get("ingestion_adapter")
                if existing_adapter not in (None, source.source_type):
                    raise ValidationError(
                        "connector cannot be reassigned to a different ingestion adapter",
                        error_code="adapter_validation_error",
                        details={
                            "connector_id": ref.connector_id,
                            "existing_adapter": existing_adapter,
                            "incoming_adapter": source.source_type,
                        },
                    )
                metadata["ingestion_adapter"] = source.source_type
            connector.connector_metadata = metadata
            return connector
        settings = get_settings()
        is_mock = ref.source_product == "mock_xdr" or settings.source_mode == "mock_xdr"
        is_file_or_manual = (source.source_type or "").strip().lower() in {
            "file",
            "manual",
        }
        is_live = (source.source_type or "").strip().lower() == "live" or (
            settings.source_mode == "live" and not is_mock and not is_file_or_manual
        )
        if is_live:
            raise ValidationError(
                "live connector must be provisioned with explicit disposition_policy_default",
                error_code="adapter_validation_error",
                details={
                    "connector_id": ref.connector_id,
                    "source_product": ref.source_product,
                },
            )
        connector = orm.SourceConnector(
            connector_id=ref.connector_id,
            source_product=ref.source_product,
            display_name=ref.connector_id,
            disposition_policy_default=(
                DispositionPolicy.REQUIRED.value
                if is_mock
                else DispositionPolicy.NOT_REQUIRED.value
            ),
            connector_metadata=(
                {
                    **({"ingestion_adapter": source.source_type} if source.source_type else {}),
                    "source_tenant_id": ref.source_tenant_id,
                }
            ),
        )
        session.add(connector)
        await session.flush()
        return connector

    async def _upsert_source_object(
        self,
        session: AsyncSession,
        source: IngestableSource,
        source_record_id: str,
    ) -> orm.SourceObject:
        ref = source.reference
        existing = await session.scalar(
            select(orm.SourceObject)
            .where(
                orm.SourceObject.source_product == ref.source_product,
                orm.SourceObject.source_tenant_id == ref.source_tenant_id,
                orm.SourceObject.connector_id == ref.connector_id,
                orm.SourceObject.source_kind == ref.source_kind.value,
                orm.SourceObject.source_object_id == ref.source_object_id,
            )
            .with_for_update()
        )
        if existing is not None:
            if not should_apply_source_update(
                stored_updated_at=existing.current_source_updated_at,
                stored_token=existing.current_concurrency_token,
                incoming_updated_at=ref.source_updated_at,
                incoming_token=ref.source_concurrency_token,
            ):
                return existing
            # Mutable current_* only — never overwrite investigation snapshot.
            existing.current_source_status_raw = ref.source_status_raw
            existing.current_source_disposition = ref.source_disposition.value
            existing.current_concurrency_token = ref.source_concurrency_token
            existing.current_source_updated_at = ref.source_updated_at
            existing.current_state_version += 1
            if source.normalized:
                existing.normalized = source.normalized
            await session.flush()
            return existing

        obj = orm.SourceObject(
            source_record_id=source_record_id,
            source_product=ref.source_product,
            source_tenant_id=ref.source_tenant_id,
            connector_id=ref.connector_id,
            source_kind=ref.source_kind.value,
            source_object_id=ref.source_object_id,
            source_object_type=ref.source_object_type,
            parent_source_object_id=ref.parent_source_object_id,
            source_status_raw=ref.source_status_raw,
            source_disposition=ref.source_disposition.value,
            source_concurrency_token=ref.source_concurrency_token,
            source_updated_at=ref.source_updated_at,
            schema_version=ref.schema_version,
            ingested_at=ref.ingested_at or datetime.now(UTC),
            raw_payload_hash=ref.raw_payload_hash,
            normalized=source.normalized or {},
            raw_payload=source.raw_payload or {},
            current_source_status_raw=ref.source_status_raw,
            current_source_disposition=ref.source_disposition.value,
            current_concurrency_token=ref.source_concurrency_token,
            current_source_updated_at=ref.source_updated_at,
            current_state_version=1,
        )
        session.add(obj)
        await session.flush()
        return obj

    async def _resolve_policy(
        self, session: AsyncSession, source: IngestableSource
    ) -> DispositionPolicy:
        connector = await session.get(orm.SourceConnector, source.reference.connector_id)
        settings = get_settings()
        source_type = source.source_type
        if source_type is None and source.reference.source_product == "file":
            source_type = "file"
        normalized_type = (source_type or "").strip().lower()
        product = (source.reference.source_product or "").strip().lower()
        mode = (settings.source_mode or "").strip().lower()
        is_mock = product == "mock_xdr" or mode == "mock_xdr"
        is_file_or_manual = normalized_type in {"file", "manual"}
        is_live = normalized_type == "live" or (
            mode == "live" and not is_mock and not is_file_or_manual
        )
        connector_policy = connector_policy_from_row(connector)
        try:
            policy = self._policy.resolve(
                source_type=source_type,
                source_kind=source.reference.source_kind,
                source_product=source.reference.source_product,
                connector_policy_default=connector_policy,
                source_mode=settings.source_mode,
                live_configured=is_live and connector_policy is None,
            )
        except ValueError as exc:
            raise ValidationError(
                str(exc),
                error_code="adapter_validation_error",
                details={
                    "connector_id": source.reference.connector_id,
                    "source_product": source.reference.source_product,
                    "source_mode": settings.source_mode,
                },
            ) from exc
        return policy

    async def _create_new_event(
        self,
        session: AsyncSession,
        source: IngestableSource,
        obj: orm.SourceObject,
        source_record_id: str,
        occurred: datetime,
    ) -> orm.SecurityEvent:
        ref = source.reference
        identity = canonical_source_identity(
            source_product=ref.source_product,
            source_tenant_id=ref.source_tenant_id,
            connector_id=ref.connector_id,
            source_kind=ref.source_kind.value,
            source_object_id=ref.source_object_id,
        )
        event_id = new_event_id(identity, occurred)
        existing = await session.get(orm.SecurityEvent, event_id)
        if existing is not None:
            # Rare: event_id collision with different source — attach related link.
            session.add(
                orm.SourceEventLink(
                    source_record_id=source_record_id,
                    event_id=event_id,
                    role=LINK_ROLE_RELATED,
                    promotion_status=PROMOTION_NONE,
                )
            )
            await session.flush()
            return existing

        policy = await self._resolve_policy(session, source)
        role = (
            LINK_ROLE_PROVISIONAL
            if ref.source_kind is SourceObjectKind.ALERT
            else LINK_ROLE_PRIMARY
        )
        disposition_ref: dict[str, Any] | None
        if ref.source_kind is SourceObjectKind.INCIDENT:
            disposition_ref = locator_from_reference(ref).model_dump(mode="json")
        elif ref.source_kind is SourceObjectKind.ALERT:
            disposition_ref = locator_from_reference(ref).model_dump(mode="json")
        else:
            disposition_ref = None

        title = source.title or f"{ref.source_kind.value}:{ref.source_object_id}"
        event_type = source.event_type or EventType.OTHER
        severity = source.severity or Severity.LOW
        raw_alert_ids = [ref.source_object_id] if ref.source_kind is SourceObjectKind.ALERT else []

        row = orm.SecurityEvent(
            event_id=event_id,
            event_type=event_type.value,
            title=title,
            description=source.description,
            status=EventStatus.NEW.value,
            severity=severity.value,
            final_verdict=FinalVerdict.NONE.value,
            entities={},
            creation_source_ref=_ref_dump(ref),
            source_reference_snapshots=[_ref_dump(ref)],
            current_primary_source_record_id=source_record_id,
            disposition_source_ref=disposition_ref,
            disposition_policy=policy.value,
            raw_alert_ids=raw_alert_ids,
            source_type=source.source_type or ref.source_product,
            occurred_at=occurred,
        )
        session.add(row)
        session.add(
            orm.SourceEventLink(
                source_record_id=source_record_id,
                event_id=event_id,
                role=role,
                promotion_status=PROMOTION_NONE,
            )
        )
        session.add(
            orm.EventAuditLog(
                event_id=event_id,
                from_status=None,
                to_status=EventStatus.NEW.value,
                operator="EventService",
                reason="event_created",
            )
        )
        await session.flush()
        await session.refresh(row)
        return row

    async def _find_source_by_ref(
        self, session: AsyncSession, ref: SourceReference
    ) -> orm.SourceObject | None:
        obj: orm.SourceObject | None = await session.scalar(
            select(orm.SourceObject).where(
                orm.SourceObject.source_product == ref.source_product,
                orm.SourceObject.source_tenant_id == ref.source_tenant_id,
                orm.SourceObject.connector_id == ref.connector_id,
                orm.SourceObject.source_kind == ref.source_kind.value,
                orm.SourceObject.source_object_id == ref.source_object_id,
            )
        )
        return obj

    async def _link_alert_to_incident_event(
        self,
        session: AsyncSession,
        source: IngestableSource,
        alert_obj: orm.SourceObject,
        alert_record_id: str,
    ) -> _CreateBundle | None:
        """When Alert carries verified incident_ref and Incident event exists → merge."""
        assert source.incident_ref is not None
        parent_obj = await self._find_source_by_ref(session, source.incident_ref)
        if parent_obj is None:
            return None
        parent_link = await session.scalar(
            select(orm.SourceEventLink).where(
                orm.SourceEventLink.source_record_id == parent_obj.source_record_id,
                orm.SourceEventLink.role.in_([LINK_ROLE_PRIMARY, LINK_ROLE_PROVISIONAL]),
            )
        )
        if parent_link is None:
            return None
        event = await session.get(orm.SecurityEvent, parent_link.event_id)
        if event is None:
            return None

        snapshots = list(event.source_reference_snapshots or [])
        snapshots.append(_ref_dump(source.reference))
        event.source_reference_snapshots = snapshots
        alert_ids = list(event.raw_alert_ids or [])
        if source.reference.source_object_id not in alert_ids:
            alert_ids.append(source.reference.source_object_id)
            event.raw_alert_ids = alert_ids
        event.row_version = int(event.row_version or 1) + 1

        session.add(
            orm.SourceEventLink(
                source_record_id=alert_record_id,
                event_id=event.event_id,
                role=LINK_ROLE_RELATED,
                promotion_status=PROMOTION_NONE,
            )
        )
        session.add(
            orm.EventAuditLog(
                event_id=event.event_id,
                from_status=event.status,
                to_status=event.status,
                operator="EventService",
                reason="alert_linked_to_incident_event",
            )
        )
        await session.flush()
        await session.refresh(event)
        return _CreateBundle(
            event=event,
            source_record_id=alert_record_id,
            created=False,
            idempotent=False,
        )

    async def _event_has_merge_blockers(self, session: AsyncSession, event_id: str) -> bool:
        action = await session.scalar(
            select(orm.Action.action_id).where(orm.Action.event_id == event_id).limit(1)
        )
        if action is not None:
            return True

        # Do not destructively merge an event once investigation artifacts exist.
        activity_queries = (
            select(orm.Evidence.evidence_id).where(orm.Evidence.event_id == event_id).limit(1),
            select(orm.Report.report_id).where(orm.Report.event_id == event_id).limit(1),
            select(orm.AgentTrace.trace_id).where(orm.AgentTrace.event_id == event_id).limit(1),
            select(orm.ToolCallLog.call_id).where(orm.ToolCallLog.event_id == event_id).limit(1),
            select(orm.LLMCallLog.id).where(orm.LLMCallLog.event_id == event_id).limit(1),
        )
        for query in activity_queries:
            if await session.scalar(query) is not None:
                return True

        # ``event`` and ``source_snapshot`` are ingestion initialization records.
        # Any other context field means an Agent/Service has started work
        # (including approval_records).
        context_activity = await session.scalar(
            select(orm.EventContextJournal.id)
            .where(
                orm.EventContextJournal.event_id == event_id,
                orm.EventContextJournal.field_name.not_in(("event", "source_snapshot")),
            )
            .limit(1)
        )
        return context_activity is not None

    async def _merge_provisional_event(
        self,
        session: AsyncSession,
        *,
        target: orm.SecurityEvent,
        secondary: orm.SecurityEvent,
    ) -> None:
        """Move a pristine provisional event into ``target`` and remove it."""
        snapshots = list(target.source_reference_snapshots or [])
        seen_snapshots = {
            (
                str(item.get("source_product")),
                str(item.get("source_tenant_id")),
                str(item.get("connector_id")),
                str(item.get("source_kind")),
                str(item.get("source_object_id")),
            )
            for item in snapshots
        }
        for item in secondary.source_reference_snapshots or []:
            identity = (
                str(item.get("source_product")),
                str(item.get("source_tenant_id")),
                str(item.get("connector_id")),
                str(item.get("source_kind")),
                str(item.get("source_object_id")),
            )
            if identity not in seen_snapshots:
                seen_snapshots.add(identity)
                snapshots.append(item)
        target.source_reference_snapshots = snapshots
        target.raw_alert_ids = list(
            dict.fromkeys([*(target.raw_alert_ids or []), *(secondary.raw_alert_ids or [])])
        )

        links = (
            await session.scalars(
                select(orm.SourceEventLink).where(
                    orm.SourceEventLink.event_id == secondary.event_id
                )
            )
        ).all()
        for link in links:
            duplicate = await session.scalar(
                select(orm.SourceEventLink.id).where(
                    orm.SourceEventLink.source_record_id == link.source_record_id,
                    orm.SourceEventLink.event_id == target.event_id,
                )
            )
            if duplicate is not None:
                await session.delete(link)
            else:
                link.event_id = target.event_id
                link.role = LINK_ROLE_RELATED
                link.promotion_status = PROMOTION_PROMOTED

        # Preserve audit/data-quality history under the surviving event.
        await session.execute(
            update(orm.EventAuditLog)
            .where(orm.EventAuditLog.event_id == secondary.event_id)
            .values(event_id=target.event_id)
        )
        await session.execute(
            update(orm.DataQualityError)
            .where(orm.DataQualityError.event_id == secondary.event_id)
            .values(event_id=target.event_id)
        )
        await session.execute(
            delete(orm.EventContextJournal).where(
                orm.EventContextJournal.event_id == secondary.event_id
            )
        )
        await session.execute(
            delete(orm.EventContextFieldVersion).where(
                orm.EventContextFieldVersion.event_id == secondary.event_id
            )
        )
        await session.delete(secondary)

    async def _promote_or_relate_alerts(
        self,
        session: AsyncSession,
        source: IngestableSource,
        incident_obj: orm.SourceObject,
        incident_record_id: str,
        occurred: datetime,
    ) -> _CreateBundle | None:
        """Incident arrives with verified related_alert_refs → promote or relate."""
        provisional_events: list[orm.SecurityEvent] = []
        seen_source_records: set[str] = set()
        seen_event_ids: set[str] = set()
        for alert_ref in source.related_alert_refs:
            alert_obj = await self._find_source_by_ref(session, alert_ref)
            if alert_obj is None or alert_obj.source_record_id in seen_source_records:
                continue
            seen_source_records.add(alert_obj.source_record_id)
            link = await session.scalar(
                select(orm.SourceEventLink).where(
                    orm.SourceEventLink.source_record_id == alert_obj.source_record_id
                )
            )
            if link is None:
                continue
            event = await session.get(orm.SecurityEvent, link.event_id)
            if event is None:
                continue
            if link.role == LINK_ROLE_PROVISIONAL and event.event_id not in seen_event_ids:
                seen_event_ids.add(event.event_id)
                provisional_events.append(event)

        if not provisional_events:
            return None

        # Promote one pristine event, merge other pristine provisional events,
        # and keep only events with investigation state as separate related cases.
        target: orm.SecurityEvent | None = None
        blocked: list[orm.SecurityEvent] = []
        mergeable: list[orm.SecurityEvent] = []
        for event in provisional_events:
            if await self._event_has_merge_blockers(session, event.event_id):
                blocked.append(event)
            elif target is None:
                target = event
            else:
                mergeable.append(event)

        if target is None:
            # All provisional children blocked — create new incident event + related links.
            created = await self._create_new_event(
                session, source, incident_obj, incident_record_id, occurred
            )
            for event in blocked:
                await self._add_related_link_if_missing(session, incident_record_id, event.event_id)
                # Also link child event ↔ keep separate; mark related between events via audit.
                session.add(
                    orm.EventAuditLog(
                        event_id=event.event_id,
                        from_status=event.status,
                        to_status=event.status,
                        operator="EventService",
                        reason=f"related_to_incident_event:{created.event_id}",
                    )
                )
            await session.flush()
            return _CreateBundle(
                event=created,
                source_record_id=incident_record_id,
                created=True,
                related_only=True,
            )

        merged_event_ids: list[str] = []
        for secondary in mergeable:
            merged_event_ids.append(secondary.event_id)
            await self._merge_provisional_event(
                session,
                target=target,
                secondary=secondary,
            )

        # Atomic promotion: keep event_id / creation_source_ref; append Incident snapshot.
        snapshots = list(target.source_reference_snapshots or [])
        snapshots.append(_ref_dump(source.reference))
        target.source_reference_snapshots = snapshots
        target.current_primary_source_record_id = incident_record_id
        target.disposition_source_ref = locator_from_reference(source.reference).model_dump(
            mode="json"
        )
        policy = await self._resolve_policy(session, source)
        target.disposition_policy = policy.value
        if source.title:
            target.title = source.title
        target.row_version = int(target.row_version or 1) + 1

        session.add(
            orm.SourceEventLink(
                source_record_id=incident_record_id,
                event_id=target.event_id,
                role=LINK_ROLE_PRIMARY,
                promotion_status=PROMOTION_PROMOTED,
            )
        )
        # Flip original provisional alert link promotion marker.
        alert_links = (
            await session.scalars(
                select(orm.SourceEventLink).where(
                    orm.SourceEventLink.event_id == target.event_id,
                    orm.SourceEventLink.role == LINK_ROLE_PROVISIONAL,
                )
            )
        ).all()
        for link in alert_links:
            link.promotion_status = PROMOTION_PROMOTED

        for other in blocked:
            session.add(
                orm.SourceEventLink(
                    source_record_id=incident_record_id,
                    event_id=other.event_id,
                    role=LINK_ROLE_RELATED,
                    promotion_status=PROMOTION_NONE,
                )
            )
            session.add(
                orm.EventAuditLog(
                    event_id=other.event_id,
                    from_status=other.status,
                    to_status=other.status,
                    operator="EventService",
                    reason=f"related_not_merged:{target.event_id}",
                )
            )

        session.add(
            orm.EventAuditLog(
                event_id=target.event_id,
                from_status=target.status,
                to_status=target.status,
                operator="EventService",
                reason="promoted_to_incident",
            )
        )
        await session.flush()
        await session.refresh(target)
        return _CreateBundle(
            event=target,
            source_record_id=incident_record_id,
            created=False,
            promoted=True,
            merged_event_ids=tuple(merged_event_ids),
        )

    async def _add_related_link_if_missing(
        self, session: AsyncSession, source_record_id: str, event_id: str
    ) -> None:
        existing = await session.scalar(
            select(orm.SourceEventLink).where(
                orm.SourceEventLink.source_record_id == source_record_id,
                orm.SourceEventLink.event_id == event_id,
            )
        )
        if existing is None:
            session.add(
                orm.SourceEventLink(
                    source_record_id=source_record_id,
                    event_id=event_id,
                    role=LINK_ROLE_RELATED,
                    promotion_status=PROMOTION_NONE,
                )
            )
