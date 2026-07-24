"""Redis-backed LangGraph checkpoints for ISSUE-048."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.core.redis_client import RedisClient

logger = logging.getLogger(__name__)

CHECKPOINT_KEY_PREFIX = "shadowtrace:checkpoint:"
CHECKPOINT_TTL_SECONDS = 7 * 24 * 60 * 60


def checkpoint_key_for_event(event_id: str) -> str:
    return f"{CHECKPOINT_KEY_PREFIX}{event_id}"


class RedisCheckpointer(BaseCheckpointSaver[str]):
    """LangGraph saver persisted as one JSON-safe Redis envelope per event."""

    def __init__(
        self,
        redis_client: RedisClient | None,
        *,
        ttl_seconds: int = CHECKPOINT_TTL_SECONDS,
    ) -> None:
        super().__init__()
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._memory = InMemorySaver()
        self._serde = JsonPlusSerializer()
        self.memory_fallback = redis_client is None
        self._fallback_warned = False

    @property
    def recoverable(self) -> bool:
        """Whether this run qualifies as process-recoverable P0 execution."""
        return not self.memory_fallback

    @classmethod
    async def create(
        cls,
        redis_client: RedisClient | None,
        *,
        ttl_seconds: int = CHECKPOINT_TTL_SECONDS,
    ) -> RedisCheckpointer:
        saver = cls(redis_client, ttl_seconds=ttl_seconds)
        try:
            available = redis_client is not None and await redis_client.ping()
        except Exception:
            available = False
        if not available:
            saver._enable_memory_fallback("Redis checkpoint unavailable")
        return saver

    def _mark_sync_nonrecoverable(self) -> None:
        self._enable_memory_fallback(
            "Synchronous LangGraph checkpoint API cannot perform async Redis I/O"
        )

    # The synchronous protocol remains usable in-process, but explicitly
    # downgrades the saver so callers cannot mistake it for Redis-recoverable.
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._mark_sync_nonrecoverable()
        return self._memory.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        self._mark_sync_nonrecoverable()
        yield from self._memory.list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._mark_sync_nonrecoverable()
        return self._memory.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._mark_sync_nonrecoverable()
        self._memory.put_writes(config, writes, task_id, task_path=task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._mark_sync_nonrecoverable()
        self._memory.delete_thread(thread_id)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        await self._hydrate(thread_id)
        return self._memory.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is not None:
            await self._hydrate(str(config["configurable"]["thread_id"]))
        for item in self._memory.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        result = self._memory.put(config, checkpoint, metadata, new_versions)
        await self._persist(str(config["configurable"]["thread_id"]))
        return result

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._memory.put_writes(config, writes, task_id, task_path=task_path)
        await self._persist(str(config["configurable"]["thread_id"]))

    async def adelete_thread(self, thread_id: str) -> None:
        self._memory.delete_thread(thread_id)
        if self.memory_fallback or self._redis is None:
            return
        try:
            await self._redis.get_client().delete(checkpoint_key_for_event(thread_id))
        except Exception:
            self._enable_memory_fallback("Redis checkpoint delete failed", exc_info=True)
            raise

    def _export(self, thread_id: str) -> bytes | None:
        if thread_id not in self._memory.storage:
            return None
        payload = {
            "storage": self._memory.storage[thread_id],
            "writes": {
                key: value for key, value in self._memory.writes.items() if key[0] == thread_id
            },
            "blobs": {
                key: value for key, value in self._memory.blobs.items() if key[0] == thread_id
            },
        }
        type_tag, raw = self._serde.dumps_typed(payload)
        envelope = {
            "format": 1,
            "serde": type_tag,
            "payload": base64.b64encode(raw).decode("ascii"),
        }
        return json.dumps(envelope, separators=(",", ":")).encode()

    def _import(self, thread_id: str, raw: bytes) -> None:
        envelope = json.loads(raw.decode())
        if envelope.get("format") != 1:
            raise ValueError("unsupported Redis checkpoint envelope format")
        payload = self._serde.loads_typed(
            (
                envelope["serde"],
                base64.b64decode(envelope["payload"]),
            )
        )
        self._memory.storage[thread_id] = payload["storage"]
        self._memory.writes.update(payload.get("writes", {}))
        self._memory.blobs.update(payload.get("blobs", {}))

    async def _hydrate(self, thread_id: str) -> None:
        if self.memory_fallback or self._redis is None or thread_id in self._memory.storage:
            return
        try:
            raw = await self._redis.get_client().get(checkpoint_key_for_event(thread_id))
            if raw is not None:
                value = raw if isinstance(raw, bytes) else str(raw).encode()
                self._import(thread_id, value)
        except Exception:
            self._enable_memory_fallback("Redis checkpoint load failed", exc_info=True)

    async def _persist(self, thread_id: str) -> None:
        if self.memory_fallback or self._redis is None:
            return
        raw = self._export(thread_id)
        if raw is None:
            return
        try:
            await self._redis.get_client().set(
                checkpoint_key_for_event(thread_id),
                raw,
                ex=self._ttl_seconds,
            )
        except Exception:
            self._enable_memory_fallback("Redis checkpoint persist failed", exc_info=True)

    def _enable_memory_fallback(self, message: str, *, exc_info: bool = False) -> None:
        if not self._fallback_warned:
            logger.warning(
                "%s; using in-memory fallback (process restart cannot recover)",
                message,
                exc_info=exc_info,
            )
            self._fallback_warned = True
        self.memory_fallback = True


async def build_checkpointer(
    redis_client: RedisClient | None,
) -> RedisCheckpointer:
    return await RedisCheckpointer.create(redis_client)


__all__ = [
    "CHECKPOINT_KEY_PREFIX",
    "CHECKPOINT_TTL_SECONDS",
    "RedisCheckpointer",
    "build_checkpointer",
    "checkpoint_key_for_event",
]
