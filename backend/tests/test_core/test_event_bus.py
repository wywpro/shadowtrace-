"""EventBus publish/subscribe tests against Compose Redis (ISSUE-013)."""

from __future__ import annotations

import asyncio
import io
import logging
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.core.event_bus import SOCKET_MESSAGE_TYPES, EventBus, sanitize_payload
from app.core.redis_client import RedisClient
from app.core.sanitization import RedactingFormatter

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@pytest_asyncio.fixture
async def bus() -> AsyncIterator[tuple[EventBus, RedisClient]]:
    client = RedisClient(url=REDIS_URL)
    if not await client.ping():
        await client.aclose()
        pytest.skip("Redis not reachable; start Compose redis first")
    yield EventBus(client), client
    await client.aclose()


def test_socket_message_types_are_exactly_sixteen() -> None:
    assert len(SOCKET_MESSAGE_TYPES) == 16
    assert "state_change" in SOCKET_MESSAGE_TYPES
    assert "disposition_submitted" in SOCKET_MESSAGE_TYPES
    assert "writeback_updated" in SOCKET_MESSAGE_TYPES


def test_sanitize_redacts_secrets_and_raw_result() -> None:
    cleaned = sanitize_payload(
        {
            "status": "closed",
            "api_key": "secret-key",
            "nested": {"password": "p", "ok": 1},
            "raw_result": {"vendor": "leak"},
            "items": [{"token": "t", "id": "1"}],
            "message": "Authorization: Bearer value-pattern-secret",
            "diagnostic": 'password="value with spaces" cookie=session-secret; Path=/',
            "jwt_note": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyLTEifQ.signature123",
            "endpoint": "https://user:password@example.test/api",
        }
    )
    assert cleaned["status"] == "closed"
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["password"] == "[REDACTED]"
    assert cleaned["nested"]["ok"] == 1
    assert cleaned["raw_result"] == "[REDACTED]"
    assert cleaned["items"][0]["token"] == "[REDACTED]"
    assert cleaned["items"][0]["id"] == "1"
    assert "value-pattern-secret" not in cleaned["message"]
    assert "value with spaces" not in cleaned["diagnostic"]
    assert "session-secret" not in cleaned["diagnostic"]
    assert "eyJhbGci" not in cleaned["jwt_note"]
    assert "password" not in cleaned["endpoint"]


def test_redacting_log_formatter_removes_credential_values() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(levelname)s %(message)s"))
    logger = logging.getLogger("shadowtrace.tests.redaction")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        logger.info(
            "provider rejected Authorization: Bearer log-secret-token "
            "endpoint=https://user:password@example.test"
        )
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    output = stream.getvalue()
    assert "log-secret-token" not in output
    assert "user:password" not in output
    assert "[REDACTED]" in output


@pytest.mark.asyncio
async def test_publish_state_change_received_within_one_second(
    bus: tuple[EventBus, RedisClient],
) -> None:
    event_bus, _redis = bus
    event_id = "evt-20260712-bus00001"
    received: asyncio.Queue[dict] = asyncio.Queue()

    async def _reader() -> None:
        async for envelope in event_bus.subscribe(event_id):
            await received.put(envelope)
            break

    task = asyncio.create_task(_reader())
    await asyncio.sleep(0.05)
    ok = await event_bus.publish_event(
        event_id,
        "state_change",
        {"from_status": "new", "to_status": "triaging", "api_key": "should-not-leak"},
    )
    assert ok is True
    envelope = await asyncio.wait_for(received.get(), timeout=1.0)
    await asyncio.wait_for(task, timeout=1.0)

    assert envelope["event_id"] == event_id
    assert envelope["message_type"] == "state_change"
    assert "timestamp" in envelope
    assert envelope["payload"]["to_status"] == "triaging"
    assert envelope["payload"]["api_key"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_publish_unknown_type_raises(bus: tuple[EventBus, RedisClient]) -> None:
    event_bus, _ = bus
    with pytest.raises(ValueError, match="unknown socket message_type"):
        await event_bus.publish_event("evt-x", "not_a_real_type", {})


@pytest.mark.asyncio
async def test_publish_failure_returns_false_without_raising() -> None:
    client = RedisClient(url="redis://127.0.0.1:1/0", max_connections=1)
    event_bus = EventBus(client)
    try:
        ok = await event_bus.publish_event("evt-x", "state_change", {"a": 1})
        assert ok is False
    finally:
        await client.aclose()
