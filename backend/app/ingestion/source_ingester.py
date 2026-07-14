"""Incremental SourceAdapter ingestion with durable watermarks (ISSUE-016)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.source.base import BaseSourceAdapter, SourcePage
from app.core.config import get_settings
from app.db import models as orm
from app.models.enums import (
    ConnectorStatus,
    DispositionPolicy,
    EventType,
    Severity,
    SourceObjectKind,
)
from app.models.ids import canonical_source_identity
from app.models.source import (
    SourceAlert,
    SourceAsset,
    SourceConnector,
    SourceIncident,
    SourceLog,
    SourceReference,
)
from app.services.event_service import EventService, IngestableSource, stable_source_record_id
from app.services.evidence_projection import EvidenceProjection

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1"})
_EVENT_SOURCE_TYPES = (SourceIncident, SourceAlert)
_SUPPORTING_SOURCE_TYPES = (SourceAsset, SourceLog)


class IngestionSummary(BaseModel):
    """One poll/push result with committed watermark boundaries."""

    model_config = ConfigDict(extra="forbid")

    accepted: int = 0
    duplicate: int = 0
    rejected: int = 0
    watermark_before: dict[str, Any] | None = None
    watermark_after: dict[str, Any] | None = None
    degraded: bool = False
    errors: list[dict[str, Any]] = Field(default_factory=list)


def source_identity(ref: SourceReference) -> str:
    return canonical_source_identity(
        source_product=ref.source_product,
        source_tenant_id=ref.source_tenant_id,
        connector_id=ref.connector_id,
        source_kind=ref.source_kind.value,
        source_object_id=ref.source_object_id,
    )


def source_to_ingestable(
    item: SourceIncident | SourceAlert,
    *,
    source_type: str,
) -> IngestableSource:
    """Project a validated SourceIncident/Alert into EventService input."""
    normalized = item.normalized or {}
    event_type = _event_type(normalized, item)
    severity = _severity(normalized, item)
    title: str | None
    description = str(normalized.get("description") or "")

    if isinstance(item, SourceIncident):
        title = item.title or _optional_text(normalized.get("title"))
        incident_ref = None
        related_alert_refs = list(item.related_alert_refs)
    else:
        title = (
            _optional_text(normalized.get("title"))
            or _optional_text(normalized.get("alert_type"))
            or f"alert:{item.reference.source_object_id}"
        )
        incident_ref = item.incident_ref
        related_alert_refs = []

    return IngestableSource(
        reference=item.reference,
        raw_payload=item.raw_payload,
        normalized=normalized,
        title=title,
        description=description,
        event_type=event_type,
        severity=severity,
        occurred_at=item.reference.source_updated_at,
        incident_ref=incident_ref,
        related_alert_refs=related_alert_refs,
        source_type=source_type,
    )


class SourceIngester:
    """Pull SourceAdapter pages, persist objects, then durably advance watermark."""

    def __init__(
        self,
        event_service: EventService,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        source_mode: str | None = None,
        supported_schema_versions: frozenset[str] = SUPPORTED_SCHEMA_VERSIONS,
        evidence_projection: EvidenceProjection | None = None,
    ) -> None:
        self._events = event_service
        self._session_factory = session_factory
        self._source_mode = source_mode or get_settings().source_mode
        self._supported_schema_versions = supported_schema_versions
        self._evidence_projection = evidence_projection or EvidenceProjection(session_factory)

    async def poll(
        self,
        adapter: BaseSourceAdapter,
        object_types: Sequence[SourceObjectKind | str],
        batch_size: int,
    ) -> IngestionSummary:
        """Poll all pages and commit a watermark only after each persisted page."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._assert_file_mode(adapter)

        checkpoint_key = str(getattr(adapter, "checkpoint_key", adapter.name))
        before = await self._load_watermark(adapter.name, checkpoint_key)
        summary = IngestionSummary(
            watermark_before=_copy_watermark(before),
            watermark_after=_copy_watermark(before),
        )

        try:
            health = await adapter.health_check()
        except Exception as exc:  # noqa: BLE001 — health failure is degradation
            health = ConnectorStatus.OFFLINE
            summary.errors.append(
                {
                    "stage": "connector_health",
                    "error_category": "health_check_failed",
                    "detail": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
        if health is not ConnectorStatus.ONLINE:
            summary.degraded = True
            summary.errors.append(
                {
                    "stage": "connector_health",
                    "error_category": "connector_unavailable",
                    "status": health.value,
                }
            )
            await self._mark_adapter_status(
                adapter.name,
                ConnectorStatus.DEGRADED,
                error_category="connector_unavailable",
            )
            return summary

        cursor = _watermark_cursor(before)
        updated_after = _watermark_time(before)
        seen_connectors: set[str] = set()
        seen_cursors: set[str | None] = set()

        while True:
            if cursor in seen_cursors:
                await self._reject_page(
                    summary,
                    adapter.name,
                    "cursor_loop",
                    {"cursor": cursor},
                    rejected=1,
                )
                break
            seen_cursors.add(cursor)

            try:
                page = await adapter.list_objects(
                    object_types,
                    cursor=cursor,
                    updated_after=updated_after,
                    limit=batch_size,
                )
            except Exception as exc:  # noqa: BLE001 — poll reports degradation
                summary.degraded = True
                summary.errors.append(
                    {
                        "stage": "adapter_poll",
                        "error_category": "adapter_unavailable",
                        "detail": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
                await self._record_quality(
                    stage="adapter_poll",
                    error_category="adapter_unavailable",
                    detail={"adapter": adapter.name, "type": type(exc).__name__},
                )
                await self._mark_adapter_status(
                    adapter.name,
                    ConnectorStatus.DEGRADED,
                    error_category="adapter_unavailable",
                )
                break

            if page.schema_version not in self._supported_schema_versions:
                await self._reject_page(
                    summary,
                    adapter.name,
                    "schema_unsupported",
                    {"schema_version": page.schema_version},
                    rejected=max(1, len(page.items)),
                )
                break

            page_summary, page_connectors = await self.ingest_items(
                page.items,
                source_type=adapter.name,
            )
            _merge_counts(summary, page_summary)
            seen_connectors.update(page_connectors)

            if page_summary.rejected:
                summary.degraded = True
                summary.errors.extend(page_summary.errors)
                await self._mark_connectors(
                    seen_connectors,
                    ConnectorStatus.DEGRADED,
                    error_category="object_rejected",
                )
                # Accepted objects remain idempotent; no watermark advance means
                # a retry can safely replay the page and recover rejected items.
                break

            if page.has_more and not page.next_cursor:
                await self._reject_page(
                    summary,
                    adapter.name,
                    "invalid_pagination",
                    {"reason": "has_more_without_next_cursor"},
                    rejected=1,
                )
                break

            after = _next_watermark(
                before=before,
                page=page,
            )
            await self._commit_watermark(
                adapter_name=adapter.name,
                checkpoint_key=checkpoint_key,
                connector_ids=seen_connectors,
                watermark=after,
                schema_version=page.schema_version,
            )
            summary.watermark_after = _copy_watermark(after)

            if not page.has_more:
                break
            cursor = page.next_cursor

        # Evidence projection is independent of alert/object page success: a
        # partial object_reject must not erase otherwise queryable telemetry.
        # Projection failures degrade on their own inside the helper.
        await self._project_adapter_evidence(
            adapter,
            summary=summary,
        )
        return summary

    async def ingest_items(
        self,
        items: list[Any],
        *,
        source_type: str,
    ) -> tuple[IngestionSummary, set[str]]:
        """Process validated Source* items independently (partial acceptance)."""
        summary = IngestionSummary()
        connector_ids: set[str] = set()

        # Incident first gives later linked alerts an existing parent event.
        ordered = sorted(items, key=_source_processing_order)
        for item in ordered:
            connector_id = _connector_id(item)
            if connector_id:
                connector_ids.add(connector_id)
            try:
                duplicate = await self._ingest_one(item, source_type=source_type)
            except Exception as exc:  # noqa: BLE001 — partial batch acceptance
                summary.rejected += 1
                detail: dict[str, Any] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "connector_id": connector_id,
                }
                error = {
                    "stage": "source_ingest",
                    "error_category": "object_rejected",
                    "detail": detail,
                }
                summary.errors.append(error)
                await self._record_quality(
                    stage="source_ingest",
                    error_category="object_rejected",
                    detail=detail,
                )
                continue

            if duplicate:
                summary.duplicate += 1
            else:
                summary.accepted += 1

        return summary, connector_ids

    async def ingest_telemetry(
        self,
        records_by_source: dict[str, list[dict[str, Any]]],
        *,
        source_type: str,
        connector_id: str | None = None,
        source_tenant_id: str = "local",
        watermark: dict[str, Any] | None = None,
    ) -> int:
        """Project adapter-normalized telemetry through the shared evidence store."""
        return await self._evidence_projection.ingest_records(
            records_by_source,
            source_product=source_type,
            source_tenant_id=source_tenant_id,
            connector_id=connector_id or f"{source_type}-evidence",
            watermark=watermark,
        )

    async def _project_adapter_evidence(
        self,
        adapter: BaseSourceAdapter,
        *,
        summary: IngestionSummary,
    ) -> None:
        try:
            # Evidence has its own idempotent identities. Until adapters expose
            # a dedicated evidence watermark, replay the page rather than reuse
            # the SourceObject watermark and risk permanently skipping a failed
            # projection write.
            page = await adapter.list_evidence_records(updated_after=None)
            if page is None:
                return
            if page.schema_version not in self._supported_schema_versions:
                raise ValueError(f"unsupported evidence schema_version={page.schema_version}")
            await self._evidence_projection.ingest_records(
                page.records_by_source,
                source_product=page.source_product,
                source_tenant_id=page.source_tenant_id,
                connector_id=page.connector_id,
                schema_version=page.schema_version,
                watermark=summary.watermark_after,
            )
        except Exception as exc:  # noqa: BLE001 — project gap degrades, never fabricates
            summary.degraded = True
            summary.errors.append(
                {
                    "stage": "evidence_projection",
                    "error_category": "projection_failed",
                    "detail": {
                        "adapter": adapter.name,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            await self._record_quality(
                stage="evidence_projection",
                error_category="projection_failed",
                detail={"adapter": adapter.name, "type": type(exc).__name__},
            )
            await self._mark_adapter_status(
                adapter.name,
                ConnectorStatus.DEGRADED,
                error_category="projection_failed",
            )

    async def _ingest_one(self, item: Any, *, source_type: str) -> bool:
        if isinstance(item, _EVENT_SOURCE_TYPES):
            result = await self._events.ingest_source_object(
                source_to_ingestable(item, source_type=source_type)
            )
            return result.idempotent
        if isinstance(item, _SUPPORTING_SOURCE_TYPES):
            return await self._persist_supporting_object(
                item,
                source_type=source_type,
            )
        if isinstance(item, SourceConnector):
            return await self._persist_connector(item, adapter_name=source_type)
        raise TypeError(f"unsupported source object type: {type(item).__name__}")

    async def _persist_supporting_object(
        self,
        item: SourceAsset | SourceLog,
        *,
        source_type: str,
    ) -> bool:
        ref = item.reference
        identity = source_identity(ref)
        record_id = stable_source_record_id(identity=identity)
        projected = _supporting_projection(item)
        async with self._session_factory() as session:
            async with session.begin():
                await self._ensure_connector_for_ref(
                    session,
                    ref,
                    source_type=source_type,
                )
                existing = await session.scalar(
                    select(orm.SourceObject).where(
                        orm.SourceObject.source_product == ref.source_product,
                        orm.SourceObject.source_tenant_id == ref.source_tenant_id,
                        orm.SourceObject.connector_id == ref.connector_id,
                        orm.SourceObject.source_kind == ref.source_kind.value,
                        orm.SourceObject.source_object_id == ref.source_object_id,
                    )
                )
                if existing is not None:
                    existing.current_source_status_raw = ref.source_status_raw
                    existing.current_source_disposition = ref.source_disposition.value
                    existing.current_concurrency_token = ref.source_concurrency_token
                    existing.source_sync_state = "synced"
                    if projected:
                        existing.normalized = projected
                    await session.flush()
                    return True

                session.add(
                    orm.SourceObject(
                        source_record_id=record_id,
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
                        normalized=projected,
                        raw_payload=item.raw_payload,
                        current_source_status_raw=ref.source_status_raw,
                        current_source_disposition=ref.source_disposition.value,
                        current_concurrency_token=ref.source_concurrency_token,
                        source_sync_state="synced",
                    )
                )
                await session.flush()
                return False

    async def _persist_connector(
        self,
        item: SourceConnector,
        *,
        adapter_name: str,
    ) -> bool:
        async with self._session_factory() as session:
            async with session.begin():
                existing = await session.get(orm.SourceConnector, item.connector_id)
                duplicate = existing is not None
                row = existing or orm.SourceConnector(
                    connector_id=item.connector_id,
                    source_product=item.source_product,
                    display_name=item.display_name,
                )
                row.source_product = item.source_product
                row.display_name = item.display_name
                row.device_type = item.device_type
                row.status = item.status.value
                row.read_endpoint = item.read_endpoint
                row.disposition_endpoint = item.disposition_endpoint
                row.capabilities = {
                    key.value: value.value for key, value in item.capabilities.items()
                }
                row.disposition_policy_default = item.disposition_policy_default.value
                row.last_sync_at = item.last_sync_at
                row.schema_version = item.schema_version
                metadata = dict(item.metadata)
                metadata["ingestion_adapter"] = adapter_name
                row.connector_metadata = metadata
                if existing is None:
                    session.add(row)
                await session.flush()
                return duplicate

    async def _ensure_connector_for_ref(
        self,
        session: AsyncSession,
        ref: SourceReference,
        *,
        source_type: str,
    ) -> orm.SourceConnector:
        row = await session.get(orm.SourceConnector, ref.connector_id)
        if row is not None:
            return row
        row = orm.SourceConnector(
            connector_id=ref.connector_id,
            source_product=ref.source_product,
            display_name=ref.connector_id,
            status=ConnectorStatus.ONLINE.value,
            disposition_policy_default=(
                DispositionPolicy.NOT_REQUIRED.value
                if source_type in {"file", "manual"}
                else DispositionPolicy.REQUIRED.value
                if ref.source_product == "mock_xdr"
                else None
            ),
            connector_metadata={"ingestion_adapter": source_type},
        )
        session.add(row)
        await session.flush()
        return row

    async def _load_watermark(
        self,
        adapter_name: str,
        checkpoint_key: str,
    ) -> dict[str, Any] | None:
        rows = await self._adapter_connectors(adapter_name)
        for row in rows:
            metadata = row.connector_metadata or {}
            scoped = metadata.get("ingestion_watermarks")
            if isinstance(scoped, dict) and isinstance(scoped.get(checkpoint_key), dict):
                return dict(scoped[checkpoint_key])
            if checkpoint_key == adapter_name and row.watermark:
                return dict(row.watermark)
        return None

    async def _commit_watermark(
        self,
        *,
        adapter_name: str,
        checkpoint_key: str,
        connector_ids: set[str],
        watermark: dict[str, Any],
        schema_version: str,
    ) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                rows: list[orm.SourceConnector] = []
                if connector_ids:
                    rows = list(
                        (
                            await session.scalars(
                                select(orm.SourceConnector).where(
                                    orm.SourceConnector.connector_id.in_(connector_ids)
                                )
                            )
                        ).all()
                    )
                if not rows:
                    candidates = (await session.scalars(select(orm.SourceConnector))).all()
                    rows = [row for row in candidates if _row_matches_adapter(row, adapter_name)]
                now = datetime.now(UTC)
                for row in rows:
                    metadata = dict(row.connector_metadata or {})
                    metadata["ingestion_adapter"] = adapter_name
                    scoped = dict(metadata.get("ingestion_watermarks") or {})
                    scoped[checkpoint_key] = dict(watermark)
                    metadata["ingestion_watermarks"] = scoped
                    row.connector_metadata = metadata
                    row.watermark = dict(watermark)
                    row.last_sync_at = now
                    row.schema_version = schema_version
                    row.status = ConnectorStatus.ONLINE.value
                await session.flush()

    async def _mark_adapter_status(
        self,
        adapter_name: str,
        status: ConnectorStatus,
        *,
        error_category: str,
    ) -> None:
        rows = await self._adapter_connectors(adapter_name)
        await self._mark_connectors(
            {row.connector_id for row in rows},
            status,
            error_category=error_category,
        )

    async def _mark_connectors(
        self,
        connector_ids: set[str],
        status: ConnectorStatus,
        *,
        error_category: str,
    ) -> None:
        if not connector_ids:
            return
        async with self._session_factory() as session:
            async with session.begin():
                rows = (
                    await session.scalars(
                        select(orm.SourceConnector).where(
                            orm.SourceConnector.connector_id.in_(connector_ids)
                        )
                    )
                ).all()
                for row in rows:
                    metadata = dict(row.connector_metadata or {})
                    metadata["last_ingestion_error"] = error_category
                    row.connector_metadata = metadata
                    row.status = status.value
                await session.flush()

    async def _adapter_connectors(self, adapter_name: str) -> list[orm.SourceConnector]:
        async with self._session_factory() as session:
            rows = (await session.scalars(select(orm.SourceConnector))).all()
            return [row for row in rows if _row_matches_adapter(row, adapter_name)]

    async def _reject_page(
        self,
        summary: IngestionSummary,
        adapter_name: str,
        category: str,
        detail: dict[str, Any],
        *,
        rejected: int,
    ) -> None:
        summary.rejected += rejected
        summary.degraded = True
        summary.errors.append(
            {
                "stage": "adapter_page",
                "error_category": category,
                "detail": detail,
            }
        )
        await self._record_quality(
            stage="adapter_page",
            error_category=category,
            detail={"adapter": adapter_name, **detail},
        )
        await self._mark_adapter_status(
            adapter_name,
            ConnectorStatus.DEGRADED,
            error_category=category,
        )

    async def _record_quality(
        self,
        *,
        stage: str,
        error_category: str,
        detail: dict[str, Any],
        **_: Any,
    ) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                session.add(
                    orm.DataQualityError(
                        event_id=None,
                        stage=stage,
                        error_category=error_category,
                        detail=detail,
                    )
                )

    def _assert_file_mode(self, adapter: BaseSourceAdapter) -> None:
        if adapter.name == "file" and self._source_mode != "file":
            raise RuntimeError(
                "FileSourceAdapter requires explicit SOURCE_MODE=file; "
                "automatic fallback is forbidden"
            )


def _next_watermark(
    *,
    before: dict[str, Any] | None,
    page: SourcePage,
) -> dict[str, Any]:
    previous_updated = (before or {}).get("updated_after")
    if page.has_more:
        updated_after = previous_updated
    else:
        updated_after = (
            page.server_time.isoformat() if page.server_time is not None else previous_updated
        )
    return {
        "cursor": page.next_cursor,
        "updated_after": updated_after,
    }


def _watermark_cursor(watermark: dict[str, Any] | None) -> str | None:
    if not watermark:
        return None
    cursor = watermark.get("cursor")
    return str(cursor) if cursor else None


def _watermark_time(watermark: dict[str, Any] | None) -> datetime | None:
    if not watermark:
        return None
    raw = watermark.get("updated_after")
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def _copy_watermark(watermark: dict[str, Any] | None) -> dict[str, Any] | None:
    return dict(watermark) if watermark is not None else None


def _merge_counts(target: IngestionSummary, source: IngestionSummary) -> None:
    target.accepted += source.accepted
    target.duplicate += source.duplicate
    target.rejected += source.rejected


def _connector_id(item: Any) -> str | None:
    if isinstance(item, SourceConnector):
        return item.connector_id
    ref = getattr(item, "reference", None)
    return ref.connector_id if isinstance(ref, SourceReference) else None


def _source_processing_order(item: Any) -> int:
    if isinstance(item, SourceConnector):
        return 0
    if isinstance(item, SourceIncident):
        return 1
    if isinstance(item, SourceAlert):
        return 2
    if isinstance(item, SourceAsset):
        return 3
    if isinstance(item, SourceLog):
        return 4
    return 99


def _supporting_projection(item: SourceAsset | SourceLog) -> dict[str, Any]:
    """Preserve typed SourceAsset/SourceLog fields in the query projection."""
    projected = dict(item.normalized)
    typed = item.model_dump(
        mode="json",
        exclude={"reference", "raw_payload", "normalized"},
        exclude_none=True,
    )
    for key, value in typed.items():
        projected.setdefault(key, value)
    if isinstance(item, SourceAsset):
        projected.setdefault("channel", "asset")
    else:
        device_source = str(item.device_source or "").lower()
        channel = {
            "edr": "endpoint",
            "iam": "identity",
            "nfw": "network",
            "proxy": "network",
        }.get(device_source, device_source or "log")
        projected.setdefault("channel", channel)
    return projected


def _row_matches_adapter(row: orm.SourceConnector, adapter_name: str) -> bool:
    metadata = row.connector_metadata or {}
    return metadata.get("ingestion_adapter") == adapter_name or row.source_product == adapter_name


def _event_type(normalized: dict[str, Any], item: SourceIncident | SourceAlert) -> EventType:
    candidates = [
        normalized.get("event_type"),
        normalized.get("alert_type"),
        getattr(item, "gpt_verdict_label", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        raw = str(candidate).lower()
        try:
            return EventType(raw)
        except ValueError:
            if "insider" in raw:
                return EventType.INSIDER_THREAT
            if "exfil" in raw:
                return EventType.DATA_EXFILTRATION
            if "domain" in raw:
                return EventType.SUSPICIOUS_DOMAIN
            if "account" in raw or "login" in raw:
                return EventType.ACCOUNT_ANOMALY
    return EventType.OTHER


def _severity(normalized: dict[str, Any], item: SourceIncident | SourceAlert) -> Severity:
    candidates = [normalized.get("severity"), getattr(item, "level", None)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return Severity(str(candidate).lower())
        except ValueError:
            continue
    return Severity.LOW


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
