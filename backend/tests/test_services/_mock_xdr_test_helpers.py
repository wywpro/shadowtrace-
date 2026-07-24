"""Shared Mock XDR helpers for service integration tests (ISSUE-059).

PostgreSQL ``SourceObject.current_concurrency_token`` must match the in-memory
Mock XDR scenario object or disposition submit returns conflict / HTTP errors and
outbox delivery never terminalizes the action.
"""

from __future__ import annotations

import httpx

from app.models.enums import SourceObjectKind

# Fixed id from ``insider_data_exfiltration`` scenario (seed=42).
SCENARIO_INCIDENT_ID = "88442201"

_MOCK_READ_HEADERS = {"Authorization": "Bearer mock-read-token"}


async def fetch_mock_concurrency_token(
    client: httpx.AsyncClient,
    *,
    object_id: str = SCENARIO_INCIDENT_ID,
    source_kind: SourceObjectKind = SourceObjectKind.INCIDENT,
) -> str:
    response = await client.get(
        f"/mock-xdr/v1/{source_kind.value}/{object_id}",
        headers=_MOCK_READ_HEADERS,
    )
    response.raise_for_status()
    body = response.json()
    token = body.get("concurrency_token")
    if not isinstance(token, str) or not token:
        raise AssertionError(f"mock object {object_id} missing concurrency_token")
    return token
