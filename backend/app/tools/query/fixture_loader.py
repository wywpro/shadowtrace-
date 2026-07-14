"""Test-only fixture seeding for EvidenceProjection.

Runtime tools never import or call this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models.enums import ConnectorStatus
from app.services.evidence_projection import EvidenceProjection

_FIXTURE_FILES = {
    "identity": "identity_logs.json",
    "endpoint": "endpoint_logs.json",
    "dlp": "dlp_logs.json",
    "network": "network_logs.json",
    "dns": "dns_logs.json",
    "asset": "asset_data.json",
    "threat_intel": "threat_intel.json",
}


async def load_fixture_records(
    projection: EvidenceProjection,
    fixture_dir: Path,
    *,
    source_product: str = "fixture",
    source_tenant_id: str = "test-tenant",
    connector_id: str = "fixture-evidence",
    connector_status: ConnectorStatus = ConnectorStatus.ONLINE,
) -> int:
    """Load known test files through the same projection ingestion API."""
    records: dict[str, list[dict[str, Any]]] = {}
    for channel, filename in _FIXTURE_FILES.items():
        path = fixture_dir / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"fixture {path} must contain a JSON array")
        records[channel] = [dict(item) for item in payload if isinstance(item, dict)]
    return await projection.ingest_records(
        records,
        source_product=source_product,
        source_tenant_id=source_tenant_id,
        connector_id=connector_id,
        connector_status=connector_status,
        watermark={"cursor": None, "fixture": fixture_dir.name},
    )


__all__ = ["load_fixture_records"]
