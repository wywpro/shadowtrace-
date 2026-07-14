"""Offline file SourceAdapter — no writeback capability (ISSUE-012).

Loads a MockXDRScenario (from registry or scenario JSON) and/or telemetry files
under ``data/mock/``. Never registers disposition operations; callers must treat
``writeback_required=false`` for file-sourced events.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters.source.base import (
    BaseSourceAdapter,
    DataQualityRecorder,
    InMemoryDataQualityRecorder,
    SourceEvidencePage,
    SourcePage,
)
from app.data_generators.scenarios import SCENARIO_REGISTRY, build_scenario
from app.mock_xdr.models import MockXDRScenario
from app.models.enums import (
    CapabilityState,
    ConnectorCapability,
    ConnectorStatus,
    SourceObjectKind,
)
from app.models.source import SourceAlert, SourceAsset, SourceConnector, SourceIncident, SourceLog


class FileSourceAdapter(BaseSourceAdapter):
    """Read-only offline fallback. Disposition writeback is intentionally absent."""

    name = "file"
    writeback_required: bool = False

    def __init__(
        self,
        *,
        scenario: MockXDRScenario | None = None,
        scenario_id: str | None = None,
        scenario_path: Path | None = None,
        mock_dir: Path | None = None,
        quality: DataQualityRecorder | None = None,
    ) -> None:
        self._quality = quality or InMemoryDataQualityRecorder()
        self._mock_dir = mock_dir
        if scenario is not None:
            self._scenario = scenario
        elif scenario_path is not None:
            raw = json.loads(scenario_path.read_text(encoding="utf-8"))
            self._scenario = MockXDRScenario.model_validate(raw)
        elif scenario_id is not None:
            if scenario_id in SCENARIO_REGISTRY:
                self._scenario = build_scenario(scenario_id, seed=42)
            else:
                raise ValueError(f"unknown scenario_id {scenario_id!r}")
        else:
            # Default: insider pack when present, else empty scaffold.
            self._scenario = build_scenario("insider_data_exfiltration", seed=42)

    def capabilities(self) -> dict[ConnectorCapability, CapabilityState]:
        return {
            ConnectorCapability.LOG_INGESTION: CapabilityState.SUPPORTED,
            ConnectorCapability.QUERY: CapabilityState.SUPPORTED,
            ConnectorCapability.EVENT_DISPOSITION: CapabilityState.UNSUPPORTED,
            ConnectorCapability.ENTITY_RESPONSE: CapabilityState.UNSUPPORTED,
        }

    async def list_objects(
        self,
        object_types: Sequence[SourceObjectKind | str],
        *,
        cursor: str | None = None,
        updated_after: datetime | None = None,
        limit: int = 100,
    ) -> SourcePage:
        """Return a page from the offline snapshot.

        File adapters are snapshot sources, but still speak the shared cursor
        contract so ``SourceIngester`` can page without ``invalid_pagination``.
        Cursor is a decimal offset into the filtered object list.
        """
        items: list[SourceIncident | SourceAlert | SourceAsset | SourceLog] = []
        for raw_kind in object_types:
            kind = raw_kind.value if isinstance(raw_kind, SourceObjectKind) else str(raw_kind)
            bucket = self._bucket(kind)
            for obj in bucket:
                if updated_after is not None and obj.reference.source_updated_at is not None:
                    if obj.reference.source_updated_at <= updated_after:
                        continue
                items.append(obj)

        offset = 0
        if cursor is not None and str(cursor).strip():
            try:
                offset = max(0, int(str(cursor).strip()))
            except ValueError:
                offset = 0
        if limit < 1:
            limit = 1
        page_items: list[
            SourceIncident | SourceAlert | SourceAsset | SourceLog | SourceConnector
        ] = list(items[offset : offset + limit])
        next_offset = offset + len(page_items)
        has_more = next_offset < len(items)
        return SourcePage(
            items=page_items,
            next_cursor=str(next_offset) if has_more else None,
            has_more=has_more,
            server_time=datetime.now(UTC),
            schema_version="1",
        )

    async def get_object(
        self,
        source_kind: SourceObjectKind | str,
        source_object_id: str,
    ) -> SourceIncident | SourceAlert | SourceAsset | SourceLog | None:
        kind = source_kind.value if isinstance(source_kind, SourceObjectKind) else str(source_kind)
        for obj in self._bucket(kind):
            if obj.reference.source_object_id == source_object_id:
                return obj
        return None

    async def list_evidence_records(
        self,
        *,
        updated_after: datetime | None = None,
    ) -> SourceEvidencePage | None:
        records = self.load_telemetry()
        if updated_after is not None:
            records = {
                channel: [record for record in rows if _after_watermark(record, updated_after)]
                for channel, rows in records.items()
            }
        if not any(records.values()):
            return None
        return SourceEvidencePage(
            records_by_source=records,
            source_product="file",
            source_tenant_id=self._scenario.source_tenant_id,
            connector_id="file-evidence",
            schema_version="1",
        )

    async def health_check(self) -> ConnectorStatus:
        return ConnectorStatus.ONLINE

    def load_telemetry(self) -> dict[str, list[dict[str, Any]]]:
        """Load deep-dive telemetry JSON files for Evidence normalizers."""
        if self._mock_dir is None:
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        mapping = {
            "identity": "identity_logs.json",
            "endpoint": "endpoint_logs.json",
            "dlp": "dlp_logs.json",
            "network": "network_logs.json",
            "dns": "dns_logs.json",
            "asset": "asset_data.json",
            "threat_intel": "threat_intel.json",
        }
        for channel, filename in mapping.items():
            path = self._mock_dir / filename
            if not path.exists():
                self._quality.record(
                    stage="file_telemetry",
                    error_category="missing_file",
                    detail={"path": str(path)},
                )
                continue
            out[channel] = json.loads(path.read_text(encoding="utf-8"))
        return out

    def _bucket(self, kind: str) -> list[SourceIncident | SourceAlert | SourceAsset | SourceLog]:
        if kind == "incident":
            return list(self._scenario.incidents)
        if kind == "alert":
            return list(self._scenario.alerts)
        if kind == "asset":
            return list(self._scenario.assets)
        if kind == "log":
            return list(self._scenario.logs)
        self._quality.record(
            stage="file_list",
            error_category="unknown_object_kind",
            detail={"kind": kind},
        )
        return []


def _record_time(record: dict[str, Any]) -> datetime | None:
    raw = record.get("logged_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _after_watermark(record: dict[str, Any], updated_after: datetime) -> bool:
    observed_at = _record_time(record)
    return observed_at is None or observed_at > updated_after
