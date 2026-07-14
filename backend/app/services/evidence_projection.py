"""Read-only evidence projection shared by file, Mock, and live query tools."""

from __future__ import annotations

import contextvars
import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import models as orm
from app.models.enums import ConnectorStatus, SourceDisposition, SourceObjectKind
from app.models.source import SourceReference
from app.models.tool_meta import ToolResult, ToolResultStatus

ProjectionSource = Literal[
    "account_login",
    "edr_process",
    "file_access",
    "network_flow",
    "dns",
    "asset_info",
    "vuln_info",
    "threat_intel",
    "history_cases",
]
FreshnessState = Literal["fresh", "stale", "missing"]
CoverageState = Literal["complete", "partial", "missing"]

_SOURCE_CHANNELS: dict[ProjectionSource, frozenset[str]] = {
    "account_login": frozenset({"identity"}),
    "edr_process": frozenset({"endpoint", "edr"}),
    "file_access": frozenset({"endpoint", "dlp", "data_security"}),
    "network_flow": frozenset({"network", "network_flow", "nfw"}),
    "dns": frozenset({"dns"}),
    "asset_info": frozenset({"asset"}),
    "vuln_info": frozenset({"asset", "vulnerability"}),
    "threat_intel": frozenset({"threat_intel"}),
    "history_cases": frozenset({"history_cases"}),
}
_FILE_ACTIONS = frozenset({"file_access", "archive", "upload", "download", "delete"})
_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)


class DataFreshness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: FreshnessState
    latest_record_at: datetime | None = None
    last_sync_at: datetime | None = None
    stale_after_seconds: int


class EvidenceCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: CoverageState
    requested_sources: list[str]
    available_sources: list[str] = Field(default_factory=list)
    unavailable_connectors: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class EvidenceQueryData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]]
    source_references: list[SourceReference]
    data_freshness: DataFreshness
    watermark: dict[str, Any] | None
    coverage: EvidenceCoverage
    next_cursor: str | None
    degraded: bool


@dataclass(slots=True)
class _ProjectionRow:
    source_record_id: str
    channel: str
    record: dict[str, Any]
    source_reference: SourceReference
    event_time: datetime | None
    ingested_at: datetime
    connector_status: ConnectorStatus
    last_sync_at: datetime | None
    watermark: dict[str, Any] | None


class EvidenceProjection:
    """Query normalized SourceObject evidence without touching Adapters or fixtures."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        *,
        stale_after: timedelta = timedelta(hours=1),
    ) -> None:
        self._session_factory = session_factory
        self._stale_after = stale_after
        self._memory_rows: dict[str, _ProjectionRow] = {}

    @classmethod
    def in_memory(cls, *, stale_after: timedelta = timedelta(hours=1)) -> EvidenceProjection:
        """Create an isolated projection for unit tests and fixture seeding."""
        return cls(stale_after=stale_after)

    async def ingest_records(
        self,
        records_by_source: Mapping[str, Sequence[dict[str, Any]]],
        *,
        source_product: str,
        source_tenant_id: str,
        connector_id: str,
        schema_version: str = "1",
        connector_status: ConnectorStatus = ConnectorStatus.ONLINE,
        watermark: dict[str, Any] | None = None,
        ingested_at: datetime | None = None,
    ) -> int:
        """Normalize raw telemetry into traceable SourceLog/SourceAsset rows."""
        observed_at = _as_utc(ingested_at or datetime.now(UTC))
        prepared = [
            self._prepare_row(
                channel=str(record.get("channel") or source),
                record=record,
                source_product=source_product,
                source_tenant_id=source_tenant_id,
                connector_id=connector_id,
                schema_version=schema_version,
                connector_status=connector_status,
                watermark=watermark,
                ingested_at=observed_at,
            )
            for source, records in records_by_source.items()
            for record in records
        ]
        if self._session_factory is None:
            inserted = 0
            for row in prepared:
                existing = self._memory_rows.get(row.source_record_id)
                if existing is not None:
                    existing.connector_status = connector_status
                    existing.last_sync_at = observed_at
                    existing.watermark = dict(watermark) if watermark is not None else None
                    continue
                self._memory_rows[row.source_record_id] = row
                inserted += 1
            return inserted
        return await self._persist_rows(
            prepared,
            source_product=source_product,
            connector_id=connector_id,
            schema_version=schema_version,
            connector_status=connector_status,
            watermark=watermark,
            observed_at=observed_at,
        )

    async def query(
        self,
        source: ProjectionSource,
        entity: Mapping[str, Any],
        time_range: tuple[datetime, datetime] | None,
        cursor: str | None,
        limit: int,
    ) -> EvidenceQueryData:
        """Filter and page one logical evidence source."""
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        offset = _decode_cursor(cursor)
        rows = (
            await self._history_rows()
            if source == "history_cases"
            else await self._source_rows(_SOURCE_CHANNELS[source])
        )
        rows = [row for row in rows if _eligible_for_source(source, row.record)]
        availability_rows = rows
        filtered = [
            row
            for row in rows
            if _matches_entity(source, row.record, entity)
            and _within_range(row.event_time, time_range)
        ]
        if source == "history_cases":
            filtered = _rank_history_rows(filtered, str(entity.get("pattern_description") or ""))
        else:
            filtered.sort(
                key=lambda row: (
                    row.event_time or datetime.min.replace(tzinfo=UTC),
                    row.source_record_id,
                )
            )

        page = filtered[offset : offset + limit]
        next_cursor = f"evp:{offset + len(page)}" if offset + len(page) < len(filtered) else None
        freshness, coverage, degraded = self._quality_projection(
            source,
            availability_rows,
        )
        records = [
            {
                **row.record,
                "source_record_id": row.source_record_id,
            }
            for row in page
        ]
        references = _unique_references(page)
        watermark = _watermark_projection(availability_rows)
        if source == "history_cases":
            degraded = True
            if "vector_store_unavailable_keyword_fallback" not in coverage.reasons:
                coverage.reasons.append("vector_store_unavailable_keyword_fallback")
            if coverage.state == "complete":
                coverage.state = "partial"

        return EvidenceQueryData(
            records=records,
            source_references=references,
            data_freshness=freshness,
            watermark=watermark,
            coverage=coverage,
            next_cursor=next_cursor,
            degraded=degraded,
        )

    def _prepare_row(
        self,
        *,
        channel: str,
        record: dict[str, Any],
        source_product: str,
        source_tenant_id: str,
        connector_id: str,
        schema_version: str,
        connector_status: ConnectorStatus,
        watermark: dict[str, Any] | None,
        ingested_at: datetime,
    ) -> _ProjectionRow:
        normalized_channel = channel.strip().lower()
        payload = orjson.loads(orjson.dumps(dict(record)))
        payload["channel"] = normalized_channel
        object_id = str(payload.get("record_id") or _payload_hash(payload)[:20])
        kind = SourceObjectKind.ASSET if normalized_channel == "asset" else SourceObjectKind.LOG
        identity = "|".join(
            (
                source_product,
                source_tenant_id,
                connector_id,
                kind.value,
                object_id,
            )
        )
        source_record_id = f"src-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
        event_time = _parse_datetime(payload.get("logged_at"))
        reference = SourceReference(
            source_kind=kind,
            source_product=source_product,
            source_tenant_id=source_tenant_id,
            connector_id=connector_id,
            source_object_type=normalized_channel,
            source_object_id=object_id,
            source_status_raw="indexed",
            source_disposition=SourceDisposition.UNKNOWN,
            source_updated_at=event_time,
            schema_version=schema_version,
            ingested_at=ingested_at,
            raw_payload_hash=_payload_hash(payload),
        )
        return _ProjectionRow(
            source_record_id=source_record_id,
            channel=normalized_channel,
            record=payload,
            source_reference=reference,
            event_time=event_time,
            ingested_at=ingested_at,
            connector_status=connector_status,
            last_sync_at=ingested_at,
            watermark=dict(watermark) if watermark is not None else None,
        )

    async def _persist_rows(
        self,
        rows: Sequence[_ProjectionRow],
        *,
        source_product: str,
        connector_id: str,
        schema_version: str,
        connector_status: ConnectorStatus,
        watermark: dict[str, Any] | None,
        observed_at: datetime,
    ) -> int:
        assert self._session_factory is not None
        async with self._session_factory() as session:
            async with session.begin():
                connector = await session.get(orm.SourceConnector, connector_id)
                if connector is None:
                    connector = orm.SourceConnector(
                        connector_id=connector_id,
                        source_product=source_product,
                        display_name=f"Evidence projection: {connector_id}",
                        status=connector_status.value,
                        schema_version=schema_version,
                        last_sync_at=observed_at,
                        watermark=dict(watermark) if watermark is not None else None,
                        connector_metadata={"evidence_projection": True},
                    )
                    session.add(connector)
                    await session.flush()
                else:
                    connector.status = connector_status.value
                    connector.schema_version = schema_version
                    connector.last_sync_at = observed_at
                    if watermark is not None:
                        connector.watermark = dict(watermark)

                inserted = 0
                for row in rows:
                    existing = await session.get(orm.SourceObject, row.source_record_id)
                    if existing is not None:
                        continue
                    ref = row.source_reference
                    session.add(
                        orm.SourceObject(
                            source_record_id=row.source_record_id,
                            source_product=ref.source_product,
                            source_tenant_id=ref.source_tenant_id,
                            connector_id=ref.connector_id,
                            source_kind=ref.source_kind.value,
                            source_object_id=ref.source_object_id,
                            source_object_type=ref.source_object_type,
                            source_status_raw=ref.source_status_raw,
                            source_disposition=ref.source_disposition.value,
                            source_updated_at=ref.source_updated_at,
                            schema_version=ref.schema_version,
                            ingested_at=ref.ingested_at,
                            raw_payload_hash=ref.raw_payload_hash,
                            normalized=dict(row.record),
                            raw_payload=dict(row.record),
                            current_source_status_raw=ref.source_status_raw,
                            current_source_disposition=ref.source_disposition.value,
                            source_sync_state="synced",
                        )
                    )
                    inserted += 1
                await session.flush()
                return inserted

    async def _source_rows(self, channels: frozenset[str]) -> list[_ProjectionRow]:
        if self._session_factory is None:
            return [row for row in self._memory_rows.values() if row.channel in channels]

        async with self._session_factory() as session:
            objects = (
                await session.scalars(
                    select(orm.SourceObject).where(
                        orm.SourceObject.source_kind.in_(
                            (SourceObjectKind.LOG.value, SourceObjectKind.ASSET.value)
                        )
                    )
                )
            ).all()
            connector_ids = {row.connector_id for row in objects}
            connectors = {}
            if connector_ids:
                connectors = {
                    row.connector_id: row
                    for row in (
                        await session.scalars(
                            select(orm.SourceConnector).where(
                                orm.SourceConnector.connector_id.in_(connector_ids)
                            )
                        )
                    ).all()
                }
            projected = [_row_from_orm(row, connectors.get(row.connector_id)) for row in objects]
            return [row for row in projected if row.channel in channels]

    async def _history_rows(self) -> list[_ProjectionRow]:
        memory = [row for row in self._memory_rows.values() if row.channel == "history_cases"]
        if self._session_factory is None:
            return memory

        async with self._session_factory() as session:
            events = (await session.scalars(select(orm.SecurityEvent))).all()
            rows: list[_ProjectionRow] = []
            for event in events:
                try:
                    reference = SourceReference.model_validate(event.creation_source_ref)
                except (TypeError, ValueError):
                    continue
                event_time = _as_utc(event.occurred_at or event.created_at)
                rows.append(
                    _ProjectionRow(
                        source_record_id=f"case:{event.event_id}",
                        channel="history_cases",
                        record={
                            "case_id": event.event_id,
                            "title": event.title,
                            "description": event.description,
                            "event_type": event.event_type,
                            "severity": event.severity,
                            "final_verdict": event.final_verdict,
                            "entities": event.entities,
                            "occurred_at": event_time.isoformat(),
                        },
                        source_reference=reference,
                        event_time=event_time,
                        ingested_at=_as_utc(event.created_at),
                        connector_status=ConnectorStatus.ONLINE,
                        last_sync_at=_as_utc(event.updated_at),
                        watermark=None,
                    )
                )
            return rows

    def _quality_projection(
        self,
        source: ProjectionSource,
        rows: Sequence[_ProjectionRow],
    ) -> tuple[DataFreshness, EvidenceCoverage, bool]:
        stale_after_seconds = int(self._stale_after.total_seconds())
        if not rows:
            return (
                DataFreshness(
                    state="missing",
                    stale_after_seconds=stale_after_seconds,
                ),
                EvidenceCoverage(
                    state="missing",
                    requested_sources=[source],
                    reasons=["projection_source_missing"],
                ),
                True,
            )

        latest_record_at = _max_datetime(row.event_time for row in rows)
        last_sync_at = _max_datetime((row.last_sync_at or row.ingested_at) for row in rows)
        stale = (
            last_sync_at is None or datetime.now(UTC) - _as_utc(last_sync_at) > self._stale_after
        )
        freshness = DataFreshness(
            state="stale" if stale else "fresh",
            latest_record_at=latest_record_at,
            last_sync_at=last_sync_at,
            stale_after_seconds=stale_after_seconds,
        )
        unavailable = sorted(
            {
                row.source_reference.connector_id
                for row in rows
                if row.connector_status is not ConnectorStatus.ONLINE
            }
        )
        reasons: list[str] = []
        if stale:
            reasons.append("projection_stale")
        if unavailable:
            reasons.append("connector_unavailable")
        coverage = EvidenceCoverage(
            state="partial" if unavailable else "complete",
            requested_sources=[source],
            available_sources=sorted({row.channel for row in rows}),
            unavailable_connectors=unavailable,
            reasons=reasons,
        )
        return freshness, coverage, stale or bool(unavailable)


def build_query_tool_result(
    *,
    call_id: str,
    tool_name: str,
    data: EvidenceQueryData,
    confidence: float,
) -> dict[str, Any]:
    """Build the normalized success envelope used by all nine query tools."""
    result = ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        provider_name="evidence_projection",
        status=ToolResultStatus.SUCCESS,
        data=data.model_dump(mode="json"),
        confidence=confidence,
    )
    return result.model_dump(mode="json")


def query_output_schema() -> dict[str, Any]:
    """Return ToolResult with a strict EvidenceQueryData payload schema."""
    schema = ToolResult.model_json_schema()
    data_schema = EvidenceQueryData.model_json_schema()
    data_definitions = data_schema.pop("$defs", {})
    schema.setdefault("$defs", {}).update(data_definitions)
    schema["properties"]["data"] = data_schema
    schema["properties"]["confidence"] = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
    }
    required = set(schema.get("required", []))
    required.update({"data", "confidence"})
    schema["required"] = sorted(required)
    return schema


def confidence_for_query_data(data: EvidenceQueryData) -> float:
    """Derive query confidence exclusively from freshness and coverage."""
    return _quality_confidence(data.data_freshness, data.coverage)


_projection_override: contextvars.ContextVar[EvidenceProjection | None] = contextvars.ContextVar(
    "evidence_projection_override", default=None
)
_default_projection: EvidenceProjection | None = None


def get_evidence_projection() -> EvidenceProjection:
    """Return the current async-context projection or the process default."""
    override = _projection_override.get()
    if override is not None:
        return override
    global _default_projection
    if _default_projection is None:
        from app.db.session import get_session_factory

        _default_projection = EvidenceProjection(get_session_factory())
    return _default_projection


@contextmanager
def bind_evidence_projection(projection: EvidenceProjection) -> Iterator[None]:
    """Temporarily bind an isolated projection for one test/request context."""
    token = _projection_override.set(projection)
    try:
        yield
    finally:
        _projection_override.reset(token)


def _row_from_orm(
    row: orm.SourceObject,
    connector: orm.SourceConnector | None,
) -> _ProjectionRow:
    payload = {**(row.raw_payload or {}), **(row.normalized or {})}
    channel = str(
        payload.get("channel")
        or row.source_object_type
        or payload.get("device_source")
        or row.source_kind
    ).lower()
    payload["channel"] = channel
    reference = SourceReference(
        source_kind=SourceObjectKind(row.source_kind),
        source_product=row.source_product,
        source_tenant_id=row.source_tenant_id,
        connector_id=row.connector_id,
        source_object_type=row.source_object_type,
        source_object_id=row.source_object_id,
        parent_source_object_id=row.parent_source_object_id,
        source_status_raw=row.source_status_raw,
        source_disposition=SourceDisposition(row.source_disposition),
        source_concurrency_token=row.source_concurrency_token,
        source_updated_at=row.source_updated_at,
        schema_version=row.schema_version,
        ingested_at=row.ingested_at,
        raw_payload_hash=row.raw_payload_hash,
    )
    status = ConnectorStatus.UNKNOWN
    if connector is not None:
        try:
            status = ConnectorStatus(connector.status)
        except ValueError:
            status = ConnectorStatus.UNKNOWN
    return _ProjectionRow(
        source_record_id=row.source_record_id,
        channel=channel,
        record=payload,
        source_reference=reference,
        event_time=_record_time(payload, row.source_updated_at),
        ingested_at=_as_utc(row.ingested_at or row.created_at),
        connector_status=status,
        last_sync_at=(
            _as_utc(connector.last_sync_at)
            if connector is not None and connector.last_sync_at is not None
            else None
        ),
        watermark=(
            dict(connector.watermark)
            if connector is not None and connector.watermark is not None
            else None
        ),
    )


def _matches_entity(
    source: ProjectionSource,
    record: Mapping[str, Any],
    entity: Mapping[str, Any],
) -> bool:
    if source == "account_login":
        return _same(record.get("account"), entity.get("account"))
    if source == "edr_process":
        return any(
            _same(record.get(key), entity.get("host_id")) for key in ("host_id", "hostname", "host")
        )
    if source == "file_access":
        account_match = _same(record.get("account"), entity.get("account"))
        action = str(record.get("action") or "").lower()
        has_file = bool(record.get("file") or record.get("file_name"))
        return account_match and (action in _FILE_ACTIONS or has_file)
    if source == "network_flow":
        src = entity.get("src_ip")
        dst = entity.get("dst_ip")
        return (src is None or _same(record.get("src_ip"), src)) and (
            dst is None or _same(record.get("dst_ip"), dst)
        )
    if source == "dns":
        return any(_same(record.get(key), entity.get("domain")) for key in ("domain", "query"))
    if source in {"asset_info", "vuln_info"}:
        key_match = (
            entity.get("ip") is not None and _same(record.get("ip"), entity.get("ip"))
        ) or (
            entity.get("hostname") is not None
            and _same(record.get("hostname"), entity.get("hostname"))
        )
        if source == "asset_info":
            return key_match
        return key_match and _has_vulnerability(record)
    if source == "threat_intel":
        return _same(record.get("indicator"), entity.get("indicator"))
    if source == "history_cases":
        return True
    return False


def _eligible_for_source(
    source: ProjectionSource,
    record: Mapping[str, Any],
) -> bool:
    if source == "account_login":
        event_type = str(record.get("event_type") or "").lower()
        return (
            "login" in event_type
            or str(record.get("category") or "").lower() == "auth"
            or str(record.get("device_source") or "").lower() == "iam"
        )
    if source == "edr_process":
        action = str(record.get("action") or "").lower()
        return action == "process_create" or record.get("process") is not None
    if source == "file_access":
        action = str(record.get("action") or "").lower()
        return action in _FILE_ACTIONS or bool(record.get("file") or record.get("file_name"))
    if source == "vuln_info":
        return _has_vulnerability(record)
    return True


def _has_vulnerability(record: Mapping[str, Any]) -> bool:
    return any(
        record.get(key) is not None
        for key in ("cve", "cves", "vulnerability", "vulnerabilities", "cvss")
    )


def _rank_history_rows(
    rows: Sequence[_ProjectionRow],
    description: str,
) -> list[_ProjectionRow]:
    query_terms = _terms(description)
    ranked: list[tuple[float, _ProjectionRow]] = []
    for row in rows:
        searchable_fields = (
            "title",
            "description",
            "event_type",
            "final_verdict",
            "pattern_description",
        )
        haystack = " ".join(str(row.record.get(key) or "") for key in searchable_fields)
        terms = _terms(haystack)
        if query_terms and not query_terms.intersection(terms):
            continue
        score = len(query_terms.intersection(terms)) / len(query_terms) if query_terms else 0.0
        ranked.append(
            (
                score,
                replace(
                    row,
                    record={**row.record, "keyword_score": round(score, 4)},
                ),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[0],
            -(item[1].event_time or datetime(1970, 1, 1, tzinfo=UTC)).timestamp(),
            item[1].source_record_id,
        )
    )
    return [row for _, row in ranked]


def _unique_references(rows: Sequence[_ProjectionRow]) -> list[SourceReference]:
    seen: set[tuple[str, str, str, str, str]] = set()
    result: list[SourceReference] = []
    for row in rows:
        identity = row.source_reference.identity
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row.source_reference)
    return result


def _watermark_projection(rows: Sequence[_ProjectionRow]) -> dict[str, Any] | None:
    if not rows:
        return None
    connectors: dict[str, dict[str, Any] | None] = {}
    for row in rows:
        connectors[row.source_reference.connector_id] = (
            dict(row.watermark) if row.watermark is not None else None
        )
    latest = _max_datetime(row.event_time for row in rows)
    return {
        "connectors": connectors,
        "latest_record_at": latest.isoformat() if latest is not None else None,
    }


def _quality_confidence(
    freshness: DataFreshness,
    coverage: EvidenceCoverage,
) -> float:
    score = {
        "complete": 0.95,
        "partial": 0.65,
        "missing": 0.0,
    }[coverage.state]
    if freshness.state == "stale":
        score -= 0.2
    if coverage.unavailable_connectors:
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 4)


def _within_range(
    value: datetime | None,
    time_range: tuple[datetime, datetime] | None,
) -> bool:
    if time_range is None:
        return True
    if value is None:
        return False
    start, end = time_range
    return _as_utc(start) <= _as_utc(value) <= _as_utc(end)


def _record_time(
    record: Mapping[str, Any],
    fallback: datetime | None = None,
) -> datetime | None:
    return (
        _parse_datetime(record.get("logged_at"))
        or _parse_datetime(record.get("occurred_at"))
        or (_as_utc(fallback) if fallback is not None else None)
    )


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_utc(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _max_datetime(values: Iterator[datetime | None]) -> datetime | None:
    present = [_as_utc(value) for value in values if value is not None]
    return max(present, default=None)


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return str(left).casefold() == str(right).casefold()


def _terms(value: str) -> set[str]:
    return {term.casefold() for term in _WORD_RE.findall(value) if len(term) > 1}


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    if not cursor.startswith("evp:"):
        raise ValueError("invalid evidence projection cursor")
    try:
        offset = int(cursor.removeprefix("evp:"))
    except ValueError as exc:
        raise ValueError("invalid evidence projection cursor") from exc
    if offset < 0:
        raise ValueError("invalid evidence projection cursor")
    return offset


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(orjson.dumps(dict(payload), option=orjson.OPT_SORT_KEYS)).hexdigest()


__all__ = [
    "DataFreshness",
    "EvidenceCoverage",
    "EvidenceProjection",
    "EvidenceQueryData",
    "ProjectionSource",
    "bind_evidence_projection",
    "build_query_tool_result",
    "confidence_for_query_data",
    "get_evidence_projection",
    "query_output_schema",
]
