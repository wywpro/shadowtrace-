"""Explicit offline file fallback using the same EventService path (ISSUE-016)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from app.adapters.file_source import FileSourceAdapter
from app.core.config import get_settings
from app.ingestion.alert_builder import AlertBuilder
from app.ingestion.source_ingester import IngestionSummary, SourceIngester
from app.models.enums import (
    EventType,
    Severity,
    SourceDisposition,
    SourceObjectKind,
)
from app.models.source import SourceAlert, SourceReference
from app.services.event_service import EventService, IngestableSource

_OBJECT_TYPES = [
    SourceObjectKind.INCIDENT,
    SourceObjectKind.ALERT,
    SourceObjectKind.ASSET,
    SourceObjectKind.LOG,
]
_TELEMETRY_CHANNEL_BY_FILE = {
    "identity_logs.json": "identity",
    "endpoint_logs.json": "endpoint",
    "dlp_logs.json": "dlp",
    "network_logs.json": "network",
    "dns_logs.json": "dns",
    "asset_data.json": "asset",
    "threat_intel.json": "threat_intel",
}


class FileIngester:
    """Ingest a scenario snapshot or legacy telemetry only in explicit file mode."""

    def __init__(
        self,
        source_ingester: SourceIngester,
        event_service: EventService,
        *,
        alert_builder: AlertBuilder | None = None,
        source_mode: str | None = None,
    ) -> None:
        self._source_ingester = source_ingester
        self._events = event_service
        self._builder = alert_builder or AlertBuilder()
        self._source_mode = source_mode or get_settings().source_mode

    async def ingest(
        self,
        path: Path,
        *,
        scenario: str | None = None,
        batch_size: int = 10_000,
    ) -> IngestionSummary:
        """Ingest an offline scenario; telemetry-only directories use AlertBuilder."""
        if self._source_mode != "file":
            raise RuntimeError(
                "file fallback is disabled unless SOURCE_MODE=file is explicitly selected"
            )
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"mock data path is not a directory: {path}")

        scenario_path = _scenario_path(path, scenario)
        if scenario_path is not None:
            adapter = FileSourceAdapter(
                scenario_path=scenario_path,
                mock_dir=path,
            )
            _scope_file_checkpoint(adapter, path, scenario_path.name)
            return await self._ingest_scenario_adapter(adapter, batch_size)

        if scenario is not None:
            adapter = FileSourceAdapter(
                scenario_id=scenario,
                mock_dir=path,
            )
            _scope_file_checkpoint(adapter, path, scenario)
            return await self._ingest_scenario_adapter(adapter, batch_size)

        scenario_files = sorted(path.glob("*.scenario.json"))
        if len(scenario_files) == 1:
            adapter = FileSourceAdapter(
                scenario_path=scenario_files[0],
                mock_dir=path,
            )
            _scope_file_checkpoint(adapter, path, scenario_files[0].name)
            return await self._ingest_scenario_adapter(adapter, batch_size)

        return await self._ingest_legacy_telemetry(path)

    async def _ingest_scenario_adapter(
        self,
        adapter: FileSourceAdapter,
        batch_size: int,
    ) -> IngestionSummary:
        summary = await self._source_ingester.poll(
            adapter,
            _OBJECT_TYPES,
            batch_size,
        )
        return summary

    async def _ingest_legacy_telemetry(self, path: Path) -> IngestionSummary:
        records: list[dict[str, Any]] = []
        records_by_source: dict[str, list[dict[str, Any]]] = {}
        for file_path in sorted(path.glob("*.json")):
            if file_path.name.endswith(".scenario.json"):
                continue
            try:
                raw = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return IngestionSummary(
                    rejected=1,
                    degraded=True,
                    errors=[
                        {
                            "stage": "file_read",
                            "error_category": "invalid_json",
                            "detail": {
                                "path": str(file_path),
                                "type": type(exc).__name__,
                            },
                        }
                    ],
                )
            if isinstance(raw, list):
                valid = [item for item in raw if isinstance(item, dict)]
                records.extend(valid)
                channel = _TELEMETRY_CHANNEL_BY_FILE.get(file_path.name)
                if channel is not None:
                    records_by_source[channel] = valid

        raw_alerts = self._builder.build(records)
        summary = IngestionSummary()
        await self._project_telemetry(records_by_source, summary)
        for raw_alert in raw_alerts:
            try:
                source = _synthetic_source_alert(raw_alert)
                result = await self._events.ingest_source_object(source)
            except Exception as exc:  # noqa: BLE001 — partial file acceptance
                summary.rejected += 1
                summary.errors.append(
                    {
                        "stage": "file_ingest",
                        "error_category": "object_rejected",
                        "detail": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
                continue
            if result.idempotent:
                summary.duplicate += 1
            else:
                summary.accepted += 1
        summary.watermark_after = {
            "cursor": None,
            "updated_after": (
                max(
                    (str(alert["occurred_at"]) for alert in raw_alerts),
                    default=None,
                )
            ),
        }
        return summary

    async def _project_telemetry(
        self,
        records_by_source: dict[str, list[dict[str, Any]]],
        summary: IngestionSummary,
    ) -> None:
        if not records_by_source:
            return
        try:
            await self._source_ingester.ingest_telemetry(
                records_by_source,
                source_type="file",
                connector_id="file-evidence",
                watermark=summary.watermark_after,
            )
        except Exception as exc:  # noqa: BLE001 — event ingest remains usable
            summary.degraded = True
            summary.errors.append(
                {
                    "stage": "evidence_projection",
                    "error_category": "projection_failed",
                    "detail": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )


def _scenario_path(path: Path, scenario: str | None) -> Path | None:
    if scenario is None:
        return None
    candidate = path / f"{scenario}.scenario.json"
    return candidate if candidate.is_file() else None


def _scope_file_checkpoint(
    adapter: FileSourceAdapter,
    path: Path,
    scenario_key: str,
) -> None:
    # One physical connector can host several offline scenario snapshots. Keep
    # their cursors independent while preserving the required poll signature.
    adapter.checkpoint_key = f"file:{path}:{scenario_key}"  # type: ignore[attr-defined]


def _synthetic_source_alert(raw_alert: dict[str, Any]) -> IngestableSource:
    encoded = orjson.dumps(raw_alert, option=orjson.OPT_SORT_KEYS)
    digest = hashlib.sha256(encoded).hexdigest()
    occurred = datetime.fromisoformat(str(raw_alert["occurred_at"]).replace("Z", "+00:00"))
    alert_type_raw = str(raw_alert.get("alert_type") or EventType.OTHER.value)
    try:
        event_type = EventType(alert_type_raw)
    except ValueError:
        event_type = EventType.OTHER
    entities = [str(value) for value in raw_alert.get("primary_entities") or []]
    title = f"file fallback: {alert_type_raw}"
    reference = SourceReference(
        source_kind=SourceObjectKind.ALERT,
        source_product="file",
        source_tenant_id="local",
        connector_id="file-local",
        source_object_type="synthetic_alert",
        source_object_id=f"synthetic-{digest[:16]}",
        source_disposition=SourceDisposition.UNKNOWN,
        source_updated_at=occurred,
        schema_version="1",
        ingested_at=occurred,
        raw_payload_hash=digest,
    )
    alert = SourceAlert(
        reference=reference,
        raw_payload=raw_alert,
        normalized={
            "alert_type": alert_type_raw,
            "primary_entities": entities,
            "record_count": len(raw_alert.get("records") or []),
        },
    )
    return IngestableSource(
        reference=alert.reference,
        raw_payload=alert.raw_payload,
        normalized=alert.normalized,
        title=title,
        event_type=event_type,
        severity=Severity.LOW,
        occurred_at=occurred,
        source_type="file",
    )
