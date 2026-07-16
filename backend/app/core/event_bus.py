"""Redis Pub/Sub event bus — sole publisher for the 16 Socket event types (ISSUE-013)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any

from app.core.redis_client import RedisClient
from app.core.sanitization import sanitize_data

logger = logging.getLogger(__name__)

# Intro §4.2.4 SocketEventEnvelope message types (exactly 16).
SOCKET_MESSAGE_TYPES: frozenset[str] = frozenset(
    {
        "event_created",
        "state_change",
        "agent_progress",
        "agent_completed",
        "agent_failed",
        "tool_call_started",
        "tool_call_completed",
        "approval_required",
        "approval_updated",
        "action_executed",
        "action_verified",
        "risk_updated",
        "report_generated",
        "final_verdict_updated",
        "disposition_submitted",
        "writeback_updated",
    }
)

_REDACTED = "[REDACTED]"


def events_channel(event_id: str) -> str:
    """Pub/Sub channel name for one investigation event (intro §4.7)."""
    return f"shadowtrace:events:{event_id}"


def sanitize_payload(value: Any) -> Any:
    """Recursively redact secret keys, raw blobs, and credential-shaped values."""

    return sanitize_data(value, replacement=_REDACTED)


class EventBus:
    """Publish/subscribe façade over ``shadowtrace:events:{event_id}``."""

    def __init__(self, redis: RedisClient) -> None:
        self._redis = redis

    async def publish_event(
        self,
        event_id: str,
        message_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        """Publish one Socket envelope. Returns False on Redis failure (warn only)."""
        if message_type not in SOCKET_MESSAGE_TYPES:
            raise ValueError(f"unknown socket message_type: {message_type!r}")

        try:
            safe_payload = sanitize_payload(dict(payload or {}))
        except Exception:  # noqa: BLE001 - publish only a generic fail-closed payload
            safe_payload = {"detail": "payload unavailable after redaction failure"}
        envelope = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_id": event_id,
            "message_type": message_type,
            "payload": safe_payload,
        }
        channel = events_channel(event_id)
        try:
            client = self._redis.get_client()
            await client.publish(channel, RedisClient.dumps(envelope))
            return True
        except Exception:  # noqa: BLE001 — never block the main workflow
            logger.warning(
                "EventBus publish failed event_id=%s message_type=%s",
                event_id,
                message_type,
                exc_info=True,
            )
            return False

    async def subscribe(self, event_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded envelopes from ``shadowtrace:events:{event_id}``."""
        client = self._redis.get_client()
        pubsub = client.pubsub()
        channel = events_channel(event_id)
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message is None:
                    continue
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                if data is None:
                    continue
                try:
                    yield RedisClient.loads(data)
                except Exception:  # noqa: BLE001 — skip malformed bus payloads
                    logger.warning(
                        "EventBus received undecodable message on %s",
                        channel,
                        exc_info=True,
                    )
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()  # type: ignore[no-untyped-call]
