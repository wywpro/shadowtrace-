"""ISSUE-017 data-foundation integration quality gate."""

from __future__ import annotations

import copy
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.mock_xdr import MockXDRSourceAdapter
from app.adapters.source.base import BaseSourceAdapter, SourcePage
from app.core.redis_client import RedisClient
from app.db import models as orm
from app.ingestion.file_ingester import FileIngester
from app.ingestion.push_receiver import PushBatchEnvelope, PushReceiver
from app.ingestion.source_ingester import SourceIngester
from app.mock_xdr.state import MockXDRState
from app.models.enums import (
    CapabilityState,
    ConnectorCapability,
    ConnectorStatus,
    EventStatus,
    SourceObjectKind,
)
from app.services.context_service import EventContextStore, ctx_key
from app.services.event_service import EventService

pytestmark = pytest.mark.integration

ALL_SOURCE_KINDS = [
    SourceObjectKind.INCIDENT,
    SourceObjectKind.ALERT,
    SourceObjectKind.ASSET,
    SourceObjectKind.LOG,
]
EVIDENCE_CONNECTOR_ID = "mock_xdr-evidence"


async def _count(session: AsyncSession, model: type[Any]) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def _count_ingested_source_objects(session: AsyncSession) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(orm.SourceObject)
            .where(orm.SourceObject.connector_id != EVIDENCE_CONNECTOR_ID)
        )
        or 0
    )


def _mock_object_count(
    state: MockXDRState,
    kinds: Sequence[SourceObjectKind] = ALL_SOURCE_KINDS,
) -> int:
    selected = {kind.value if isinstance(kind, SourceObjectKind) else str(kind) for kind in kinds}
    return sum(
        1 for (kind, _), stored in state.objects.items() if kind in selected and not stored.deleted
    )


@pytest.mark.asyncio
async def test_mock_xdr_http_pipeline_persists_queryable_frozen_context(
    source_adapter: MockXDRSourceAdapter,
    source_ingester: SourceIngester,
    event_service: EventService,
    context_store: EventContextStore,
    db_session: AsyncSession,
    redis_client: RedisClient,
    mock_xdr_state: MockXDRState,
) -> None:
    summary = await source_ingester.poll(
        source_adapter,
        ALL_SOURCE_KINDS,
        batch_size=1,
    )
    expected_object_count = _mock_object_count(mock_xdr_state)
    assert summary.accepted == expected_object_count
    assert summary.duplicate == 0
    assert summary.rejected == 0
    assert summary.degraded is False
    assert summary.watermark_before is None
    assert summary.watermark_after is not None

    assert await _count(db_session, orm.SecurityEvent) == 1
    assert await _count_ingested_source_objects(db_session) >= expected_object_count
    assert await _count(db_session, orm.SourceEventLink) == 4

    expected_assets = {
        object_id
        for (kind, object_id), stored in mock_xdr_state.objects.items()
        if kind == SourceObjectKind.ASSET.value and not stored.deleted
    }
    expected_logs = {
        object_id
        for (kind, object_id), stored in mock_xdr_state.objects.items()
        if kind == SourceObjectKind.LOG.value and not stored.deleted
    }
    assets = set(
        (
            await db_session.scalars(
                select(orm.SourceObject.source_object_id).where(
                    orm.SourceObject.source_kind == SourceObjectKind.ASSET.value,
                    orm.SourceObject.connector_id != EVIDENCE_CONNECTOR_ID,
                )
            )
        ).all()
    )
    logs = (
        await db_session.scalars(
            select(orm.SourceObject).where(
                orm.SourceObject.source_kind == SourceObjectKind.LOG.value,
                orm.SourceObject.connector_id != EVIDENCE_CONNECTOR_ID,
            )
        )
    ).all()
    assert assets == expected_assets
    assert {row.source_object_id for row in logs} == expected_logs
    assert all(row.parent_source_object_id for row in logs)

    assert mock_xdr_state.scenario is not None
    projected_ids = set(
        (
            await db_session.scalars(
                select(orm.SourceObject.source_object_id).where(
                    orm.SourceObject.connector_id == EVIDENCE_CONNECTOR_ID
                )
            )
        ).all()
    )
    assert projected_ids == {
        str(record["record_id"]) for record in mock_xdr_state.scenario.telemetry_timeline
    }

    listed = await event_service.list_events(status=EventStatus.NEW)
    assert listed.total == 1
    event = listed.items[0]
    assert await event_service.get_event(event.event_id) == event
    assert event.status is EventStatus.NEW
    assert len(event.source_reference_snapshots) == 4
    assert {ref.source_kind for ref in event.source_reference_snapshots} == {
        SourceObjectKind.INCIDENT,
        SourceObjectKind.ALERT,
    }

    source_snapshot = await context_store.get(event.event_id, "source_snapshot")
    frozen = copy.deepcopy(source_snapshot)
    assert source_snapshot["creation_source_ref"]["source_kind"] == "incident"
    assert len(source_snapshot["source_reference_snapshots"]) == 4
    assert await redis_client.get_client().exists(ctx_key(event.event_id)) == 1

    journals = (
        await db_session.scalars(
            select(orm.EventContextJournal).where(
                orm.EventContextJournal.event_id == event.event_id,
                orm.EventContextJournal.field_name == "source_snapshot",
            )
        )
    ).all()
    assert journals

    audit_reasons = (
        await db_session.scalars(
            select(orm.EventAuditLog.reason).where(orm.EventAuditLog.event_id == event.event_id)
        )
    ).all()
    assert audit_reasons.count("event_created") == 1
    assert audit_reasons.count("alert_linked_to_incident_event") == 3

    incident_source = await db_session.scalar(
        select(orm.SourceObject).where(
            orm.SourceObject.source_kind == SourceObjectKind.INCIDENT.value
        )
    )
    assert incident_source is not None
    immutable_status = incident_source.source_status_raw
    incident_source.current_source_status_raw = "closed-after-ingestion"
    await db_session.commit()
    assert incident_source.source_status_raw == immutable_status
    assert await context_store.get(event.event_id, "source_snapshot") == frozen


@pytest.mark.asyncio
async def test_partial_file_sources_still_create_events(
    mock_data_dir: Path,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    partial_dir = mock_data_dir.parent / "partial-data"
    partial_dir.mkdir()
    for filename in (
        "identity_logs.json",
        "endpoint_logs.json",
        "network_logs.json",
    ):
        shutil.copy2(mock_data_dir / filename, partial_dir / filename)

    source_ingester = SourceIngester(
        event_service,
        session_factory,
        source_mode="file",
    )
    file_ingester = FileIngester(
        source_ingester,
        event_service,
        source_mode="file",
    )
    summary = await file_ingester.ingest(partial_dir)

    assert summary.accepted > 0
    assert summary.rejected == 0
    assert summary.degraded is False
    listed = await event_service.list_events()
    assert listed.total >= 1
    assert all(event.source_type == "file" for event in listed.items)
    assert all(event.status is EventStatus.NEW for event in listed.items)


@pytest.mark.asyncio
async def test_bad_schema_records_quality_degrades_connector_and_halts_watermark(
    source_adapter: MockXDRSourceAdapter,
    source_ingester: SourceIngester,
    event_service: EventService,
    db_session: AsyncSession,
    mock_xdr_state: MockXDRState,
) -> None:
    alert_key = sorted(
        key for key in mock_xdr_state.objects if key[0] == SourceObjectKind.ALERT.value
    )[0]
    bad_alert = mock_xdr_state.objects[alert_key]
    bad_alert.schema_version = "2"
    reference = bad_alert.body.get("reference")
    assert isinstance(reference, dict)
    reference["schema_version"] = "2"

    source_adapter.checkpoint_key = "mock_xdr:valid-kinds"  # type: ignore[attr-defined]
    valid = await source_ingester.poll(
        source_adapter,
        [
            SourceObjectKind.INCIDENT,
            SourceObjectKind.ASSET,
            SourceObjectKind.LOG,
        ],
        batch_size=2,
    )
    valid_kinds = [
        SourceObjectKind.INCIDENT,
        SourceObjectKind.ASSET,
        SourceObjectKind.LOG,
    ]
    expected_valid_count = _mock_object_count(mock_xdr_state, valid_kinds)
    assert valid.accepted == expected_valid_count
    assert valid.rejected == 0

    source_adapter.checkpoint_key = "mock_xdr:bad-alerts"  # type: ignore[attr-defined]
    bad = await source_ingester.poll(
        source_adapter,
        [SourceObjectKind.ALERT],
        batch_size=10,
    )
    assert bad.accepted == 0
    assert bad.rejected == 1
    assert bad.degraded is True
    assert bad.watermark_before is None
    assert bad.watermark_after is None

    assert await _count_ingested_source_objects(db_session) == expected_valid_count
    assert (await event_service.list_events()).total == 1
    quality = await db_session.scalar(
        select(orm.DataQualityError).where(
            orm.DataQualityError.error_category == "schema_unsupported"
        )
    )
    assert quality is not None
    connectors = (
        await db_session.scalars(
            select(orm.SourceConnector).where(
                orm.SourceConnector.connector_id != EVIDENCE_CONNECTOR_ID
            )
        )
    ).all()
    assert connectors
    assert all(row.status == ConnectorStatus.DEGRADED.value for row in connectors)


class _FailAfterFirstPage(BaseSourceAdapter):
    name = "mock_xdr"

    def __init__(self, delegate: MockXDRSourceAdapter) -> None:
        self._delegate = delegate
        self.calls: list[str | None] = []

    def capabilities(self) -> dict[ConnectorCapability, CapabilityState]:
        return self._delegate.capabilities()

    async def list_objects(
        self,
        object_types: Sequence[SourceObjectKind | str],
        *,
        cursor: str | None = None,
        updated_after=None,
        limit: int = 100,
    ) -> SourcePage:
        self.calls.append(cursor)
        if len(self.calls) > 1:
            raise RuntimeError("simulated process interruption")
        return await self._delegate.list_objects(
            object_types,
            cursor=cursor,
            updated_after=updated_after,
            limit=limit,
        )

    async def health_check(self) -> ConnectorStatus:
        return await self._delegate.health_check()


@pytest.mark.asyncio
async def test_cursor_resume_and_delivery_replay_do_not_duplicate_event(
    source_adapter: MockXDRSourceAdapter,
    source_ingester: SourceIngester,
    event_service: EventService,
    session_factory: async_sessionmaker[AsyncSession],
    db_session: AsyncSession,
    mock_xdr_state: MockXDRState,
) -> None:
    mock_xdr_state.failure_profile.duplicate_page = True
    interrupted_adapter = _FailAfterFirstPage(source_adapter)
    interrupted = await source_ingester.poll(
        interrupted_adapter,
        ALL_SOURCE_KINDS,
        batch_size=1,
    )
    assert interrupted.accepted == 4
    assert interrupted.degraded is True
    assert interrupted.watermark_after is not None
    assert interrupted.watermark_after["cursor"] is not None
    assert await _count(db_session, orm.SecurityEvent) == 1

    restarted = SourceIngester(
        event_service,
        session_factory,
        source_mode="mock_xdr",
    )
    resumed = await restarted.poll(
        source_adapter,
        ALL_SOURCE_KINDS,
        batch_size=1,
    )
    assert resumed.watermark_before == interrupted.watermark_after
    assert resumed.rejected == 0
    assert resumed.duplicate > 0
    assert resumed.degraded is False
    assert await _count(db_session, orm.SecurityEvent) == 1
    assert await _count_ingested_source_objects(db_session) == _mock_object_count(mock_xdr_state)

    incident = next(
        stored.body
        for (kind, _), stored in mock_xdr_state.objects.items()
        if kind == SourceObjectKind.INCIDENT.value and not stored.deleted
    )
    envelope = PushBatchEnvelope(
        connector_id="conn-disposition",
        delivery_id="integration-delivery-1",
        source_product="mock_xdr",
        objects=[{"source_kind": "incident", "payload": incident}],
    )
    receiver = PushReceiver(restarted, event_service, session_factory)
    first_delivery = await receiver.receive(envelope)
    replayed_delivery = await receiver.receive(envelope)
    assert first_delivery.accepted == 0
    assert first_delivery.duplicate == 1
    assert replayed_delivery.accepted == 0
    assert replayed_delivery.duplicate == 1
    assert await _count(db_session, orm.SecurityEvent) == 1
