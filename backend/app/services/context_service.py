"""EventContext Hash store with PostgreSQL journal + Redis cache (ISSUE-013)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.v1.schemas import EventSummary
from app.core.redis_client import RedisClient
from app.db import models as orm
from app.models.context import EventContext
from app.models.disposition import WritebackSummary
from app.models.enums import (
    DispositionIntentKind,
    DispositionPolicy,
    EventStatus,
    EventType,
    FinalVerdict,
    Severity,
    SourceDisposition,
    WritebackReadiness,
    WritebackStatus,
)
from app.models.security_event import SecurityEvent

logger = logging.getLogger(__name__)

CTX_KEY_PREFIX = "shadowtrace:ctx:"
CTX_LOG_PREFIX = "shadowtrace:ctx_log:"
CLOSED_TTL_SECONDS = 24 * 60 * 60
DEGRADED_CACHE_TTL_SECONDS = 30.0
REDIS_WRITE_BACKOFFS = (0.1, 0.5, 2.0)

# Event-level aggregate priority for DispositionOutbox/Receipt statuses (ISSUE-093
# §3): the most attention-needing state wins whenever several outboxes disagree.
# CONFIRMED (fully done) is deliberately least-severe; PARTIAL (some confirmed,
# some not) ranks just above it since the cycle is not fully settled yet.
STATUS_AGGREGATE_PRIORITY: tuple[WritebackStatus, ...] = (
    WritebackStatus.CONFLICT,
    WritebackStatus.UNKNOWN,
    WritebackStatus.PENDING,
    WritebackStatus.SENDING,
    WritebackStatus.ACCEPTED,
    WritebackStatus.FAILED,
    WritebackStatus.PARTIAL,
    WritebackStatus.CONFIRMED,
)

# Event-level aggregate priority for per-Action WritebackReadiness: the worst
# (most blocking) reason present among applicable-required actions wins.
READINESS_AGGREGATE_PRIORITY: tuple[WritebackReadiness, ...] = (
    WritebackReadiness.PERMISSION_DENIED,
    WritebackReadiness.CONNECTOR_UNAVAILABLE,
    WritebackReadiness.CAPABILITY_UNSUPPORTED,
    WritebackReadiness.CAPABILITY_UNKNOWN,
    WritebackReadiness.NOT_CONFIGURED,
    WritebackReadiness.SOURCE_UNRESOLVED,
    WritebackReadiness.READY,
    WritebackReadiness.NOT_REQUIRED,
)


def _pick_by_priority(present: set[Any], priority: tuple[Any, ...]) -> Any | None:
    """Return the highest-priority (first-listed) member of ``priority`` present."""
    for candidate in priority:
        if candidate in present:
            return candidate
    return None


# EventContext Hash field names (excludes companion ``{key}__version`` keys).
CONTEXT_FIELD_NAMES: frozenset[str] = frozenset(EventContext.model_fields.keys())


@dataclass(frozen=True, slots=True)
class InitResult:
    redis_ok: bool
    version: int
    initialized: bool = True


@dataclass(frozen=True, slots=True)
class SetResult:
    redis_ok: bool
    version: int


def ctx_key(event_id: str) -> str:
    return f"{CTX_KEY_PREFIX}{event_id}"


def ctx_log_key(event_id: str) -> str:
    return f"{CTX_LOG_PREFIX}{event_id}"


def version_field(key: str) -> str:
    return f"{key}__version"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _journal_value(value: Any) -> Any:
    """JSONB-safe representation of an EventContext field value."""
    return _to_jsonable(value)


async def append_context_journal_in_session(
    session: AsyncSession,
    event_id: str,
    field_name: str,
    value: Any,
) -> int:
    """Append one EventContext journal entry inside an existing DB transaction."""
    if field_name not in CONTEXT_FIELD_NAMES:
        raise KeyError(f"unknown EventContext field: {field_name!r}")
    stored = _journal_value(value)
    result = await session.execute(
        text(
            "INSERT INTO event_context_field_version "
            "(event_id, field_name, current_version) "
            "VALUES (:event_id, :field_name, 1) "
            "ON CONFLICT (event_id, field_name) DO UPDATE "
            "SET current_version = event_context_field_version.current_version + 1 "
            "RETURNING current_version"
        ),
        {"event_id": event_id, "field_name": field_name},
    )
    new_version = int(result.one()[0])
    session.add(
        orm.EventContextJournal(
            event_id=event_id,
            field_name=field_name,
            value=stored if isinstance(stored, dict) else {"_scalar": stored},
            version=new_version,
        )
    )
    await session.flush()
    return new_version


def event_summary_from_security_event(row: orm.SecurityEvent) -> EventSummary:
    """Build the EventContext ``event`` field (EventSummary) from the ORM row."""
    policy = DispositionPolicy(row.disposition_policy)
    writeback_required = policy is DispositionPolicy.REQUIRED
    if not writeback_required:
        writeback_readiness = WritebackReadiness.NOT_REQUIRED
    elif not row.disposition_source_ref:
        writeback_readiness = WritebackReadiness.SOURCE_UNRESOLVED
    else:
        # Capability is not authoritative on security_event. Fail closed until
        # PolicyFilter evaluates the connector/adapter and writes Action readiness.
        writeback_readiness = WritebackReadiness.CAPABILITY_UNKNOWN
    return EventSummary(
        event_id=row.event_id,
        event_type=EventType(row.event_type),
        title=row.title,
        status=EventStatus(row.status),
        severity=Severity(row.severity),
        risk_score=row.risk_score,
        final_verdict=FinalVerdict(row.final_verdict),
        writeback_required=writeback_required,
        writeback_readiness=writeback_readiness,
        writeback_overall_status=None,
        pending_writeback_count=0,
        created_at=row.created_at,
        updated_at=row.updated_at,
        occurred_at=row.occurred_at,
        disposition_policy=policy,
        external_unsynced=bool(row.external_unsynced),
        escalated=bool(row.escalated),
    )


def event_summary_from_domain(event: SecurityEvent) -> EventSummary:
    """Build EventSummary from the public SecurityEvent model (non-ORM paths)."""
    writeback_required = event.disposition_policy is DispositionPolicy.REQUIRED
    if not writeback_required:
        writeback_readiness = WritebackReadiness.NOT_REQUIRED
    elif event.disposition_source_ref is None:
        writeback_readiness = WritebackReadiness.SOURCE_UNRESOLVED
    else:
        writeback_readiness = WritebackReadiness.CAPABILITY_UNKNOWN
    return EventSummary(
        event_id=event.event_id,
        event_type=event.event_type,
        title=event.title,
        status=event.status,
        severity=event.severity,
        risk_score=event.risk_score,
        final_verdict=event.final_verdict,
        writeback_required=writeback_required,
        writeback_readiness=writeback_readiness,
        writeback_overall_status=None,
        pending_writeback_count=0,
        created_at=event.created_at,
        updated_at=event.updated_at,
        occurred_at=event.occurred_at,
        disposition_policy=event.disposition_policy,
        external_unsynced=event.external_unsynced,
        escalated=event.escalated,
    )


def _default_context_dict() -> dict[str, Any]:
    """Field defaults without going through SecurityEvent-typed ``event`` dumps."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in EventContext.model_fields.items()
    }


def _context_as_dict(ctx: EventContext) -> dict[str, Any]:
    """Shallow field dict; preserves EventSummary-shaped ``event`` without warnings."""
    out: dict[str, Any] = {}
    for name in CONTEXT_FIELD_NAMES:
        out[name] = getattr(ctx, name)
    return out


def _assemble_event_context(raw: dict[str, Any]) -> EventContext:
    """Build EventContext, always validating (ISSUE-094 §2: no ``model_construct``
    bypass). ``event`` is typed as ``EventSummary | None`` so the EventSummary-shaped
    dict persisted by the journal/Redis/snapshot paths validates directly."""
    payload = {k: v for k, v in raw.items() if k in CONTEXT_FIELD_NAMES}
    base = _default_context_dict()
    base.update(payload)
    return EventContext.model_validate(base)


class EventContextStore:
    """Versioned EventContext store: PostgreSQL is authority; Redis is the hot cache."""

    def __init__(
        self,
        redis: RedisClient,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._redis = redis
        self._session_factory = session_factory
        self._degraded_cache: dict[str, EventContext] = {}
        self._degraded_cache_ts: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def init_context(self, event_id: str, event: EventSummary) -> InitResult:
        """Atomically initialize the ``event`` field once, healing Redis on repeats."""
        event_value = _journal_value(event)
        async with self._session_factory() as session:
            async with session.begin():
                inserted = await session.execute(
                    text(
                        "INSERT INTO event_context_field_version "
                        "(event_id, field_name, current_version) "
                        "VALUES (:event_id, 'event', 1) "
                        "ON CONFLICT (event_id, field_name) DO NOTHING "
                        "RETURNING current_version"
                    ),
                    {"event_id": event_id},
                )
                row = inserted.first()
                initialized = row is not None
                if row is not None:
                    version = int(row[0])
                    await self._insert_journal(session, event_id, "event", event_value, version)
                else:
                    existing = await session.scalar(
                        select(orm.EventContextFieldVersion.current_version).where(
                            orm.EventContextFieldVersion.event_id == event_id,
                            orm.EventContextFieldVersion.field_name == "event",
                        )
                    )
                    if existing is None:
                        raise RuntimeError(
                            "event context version disappeared during initialization"
                        )
                    version = int(existing)

        redis_ok = await self._redis_set_fields(
            event_id,
            {"event": event_value, version_field("event"): version},
            log_entry=(
                {
                    "op": "init_context",
                    "field_name": "event",
                    "version": version,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                if initialized
                else None
            ),
        )
        return InitResult(
            redis_ok=redis_ok,
            version=version,
            initialized=initialized,
        )

    async def get(self, event_id: str, key: str) -> Any:
        if key not in CONTEXT_FIELD_NAMES:
            raise KeyError(f"unknown EventContext field: {key!r}")

        if await self._redis.ping():
            self._clear_degraded_cache(event_id)
            client = self._redis.get_client()
            raw = await client.hget(ctx_key(event_id), key)
            if raw is not None:
                raw_version = await client.hget(ctx_key(event_id), version_field(key))
                redis_version = (
                    int(RedisClient.loads(raw_version)) if raw_version is not None else None
                )
                db_version = await self.get_field_version(event_id, key)
                if db_version is None or redis_version != db_version:
                    ctx = await self.rebuild_context(event_id)
                    return getattr(ctx, key)
                return RedisClient.loads(raw)
            ctx = await self.rebuild_context(event_id)
            return getattr(ctx, key)

        cached = self._get_degraded_if_fresh(event_id)
        if cached is not None:
            return getattr(cached, key)

        ctx = await self.rebuild_context(event_id)
        return getattr(ctx, key)

    async def set(
        self,
        event_id: str,
        key: str,
        value: Any,
        version: int | None = None,  # noqa: ARG002 — reserved; DB UPSERT is authority
    ) -> SetResult:
        if key not in CONTEXT_FIELD_NAMES:
            raise KeyError(f"unknown EventContext field: {key!r}")

        stored = _journal_value(value)
        async with self._session_factory() as session:
            async with session.begin():
                new_version = await self._upsert_version(session, event_id, key)
                await self._insert_journal(session, event_id, key, stored, new_version)

        redis_ok = await self._redis_set_fields(
            event_id,
            {key: stored, version_field(key): new_version},
            log_entry={
                "op": "set",
                "field_name": key,
                "version": new_version,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        # Keep degraded memory view coherent when Redis is down.
        if not redis_ok and event_id in self._degraded_cache:
            current = self._degraded_cache[event_id]
            updated = EventContext.model_validate({**_context_as_dict(current), key: value})
            self._degraded_cache[event_id] = updated
            self._degraded_cache_ts[event_id] = time.monotonic()

        return SetResult(redis_ok=redis_ok, version=new_version)

    async def get_full_context(self, event_id: str) -> EventContext:
        if await self._redis.ping():
            self._clear_degraded_cache(event_id)
            client = self._redis.get_client()
            raw_hash = await client.hgetall(ctx_key(event_id))
            if raw_hash:
                decoded = self._decode_hash(raw_hash)
                if any(k in CONTEXT_FIELD_NAMES for k in decoded):
                    db_versions = await self._load_current_field_versions(event_id)
                    if any(
                        decoded.get(version_field(field_name)) != db_version
                        for field_name, db_version in db_versions.items()
                    ):
                        return await self.rebuild_context(event_id)
                    return _assemble_event_context(decoded)
            return await self.rebuild_context(event_id)

        cached = self._get_degraded_if_fresh(event_id)
        if cached is not None:
            return cached
        return await self.rebuild_context(event_id)

    async def compare_and_set(
        self,
        event_id: str,
        key: str,
        expected_version: int,
        value: Any,
    ) -> bool:
        if key not in CONTEXT_FIELD_NAMES:
            raise KeyError(f"unknown EventContext field: {key!r}")

        stored = _journal_value(value)
        async with self._session_factory() as session:
            async with session.begin():
                if expected_version == 0:
                    result = await session.execute(
                        text(
                            "INSERT INTO event_context_field_version "
                            "(event_id, field_name, current_version) "
                            "VALUES (:event_id, :field_name, 1) "
                            "ON CONFLICT (event_id, field_name) DO NOTHING "
                            "RETURNING current_version"
                        ),
                        {"event_id": event_id, "field_name": key},
                    )
                else:
                    result = await session.execute(
                        text(
                            "UPDATE event_context_field_version "
                            "SET current_version = current_version + 1 "
                            "WHERE event_id = :event_id AND field_name = :field_name "
                            "AND current_version = :expected "
                            "RETURNING current_version"
                        ),
                        {
                            "event_id": event_id,
                            "field_name": key,
                            "expected": expected_version,
                        },
                    )
                row = result.first()
                if row is None:
                    return False
                new_version = int(row[0])
                await self._insert_journal(session, event_id, key, stored, new_version)

        await self._redis_set_fields(
            event_id,
            {key: stored, version_field(key): new_version},
            log_entry={
                "op": "compare_and_set",
                "field_name": key,
                "version": new_version,
                "expected_version": expected_version,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return True

    async def rebuild_context(self, event_id: str) -> EventContext:
        async with self._session_factory() as session:
            se = await session.get(orm.SecurityEvent, event_id)
            if se is None:
                raise KeyError(f"security_event not found: {event_id}")

            ctx: EventContext
            if EventStatus(se.status) is EventStatus.CLOSED and se.event_context_snapshot:
                ctx = _assemble_event_context(dict(se.event_context_snapshot))
            else:
                ctx = await self._rebuild_from_journal(session, event_id)

            # Always overlay authoritative mirrors from security_event.
            summary = event_summary_from_security_event(se)
            flags = list(se.degraded_flags or [])
            writeback = await self._merge_writeback_summary(session, se)
            merged = _context_as_dict(ctx)
            merged.update(
                {
                    "event": summary,
                    "degraded_flags": [str(f) for f in flags],
                    "replan_count": int(se.replan_count or 0),
                    "writeback_summary": writeback,
                }
            )
            ctx = EventContext.model_validate(merged)

            versions = await self._load_field_versions(session, event_id)

        redis_ok = await self._redis.ping()
        if redis_ok:
            self._clear_degraded_cache(event_id)
            mapping = self._context_to_redis_mapping(ctx, versions)
            await self._redis_set_fields(event_id, mapping, log_entry=None, expire=False)
        else:
            self._degraded_cache[event_id] = ctx
            self._degraded_cache_ts[event_id] = time.monotonic()

        return ctx

    async def delete_cached_context(self, event_id: str) -> bool:
        """Delete Redis/in-process cache for an event merged into another event."""
        self._clear_degraded_cache(event_id)
        if not await self._redis.ping():
            return False
        try:
            await self._redis.get_client().delete(
                ctx_key(event_id),
                ctx_log_key(event_id),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.warning(
                "delete_cached_context failed event_id=%s",
                event_id,
                exc_info=True,
            )
            return False

    async def set_closed_ttl(self, event_id: str) -> bool:
        """Apply 24h TTL to the context Hash (and change log). Returns redis_ok."""
        if not await self._redis.ping():
            return False
        client = self._redis.get_client()
        try:
            await client.expire(ctx_key(event_id), CLOSED_TTL_SECONDS)
            await client.expire(ctx_log_key(event_id), CLOSED_TTL_SECONDS)
            return True
        except Exception:  # noqa: BLE001
            logger.warning("set_closed_ttl failed event_id=%s", event_id, exc_info=True)
            return False

    async def refresh_closed_snapshot(self, event_id: str) -> EventContext:
        """Rebuild snapshot from journal + security_event mirrors; no Redis required."""
        async with self._session_factory() as session:
            async with session.begin():
                se = await session.get(orm.SecurityEvent, event_id)
                if se is None:
                    raise KeyError(f"security_event not found: {event_id}")

                ctx = await self._rebuild_from_journal(session, event_id)
                summary = event_summary_from_security_event(se)
                flags = list(se.degraded_flags or [])
                writeback = await self._merge_writeback_summary(session, se)
                merged = _context_as_dict(ctx)
                merged.update(
                    {
                        "event": summary,
                        "degraded_flags": [str(f) for f in flags],
                        "replan_count": int(se.replan_count or 0),
                        "writeback_summary": writeback,
                    }
                )
                ctx = EventContext.model_validate(merged)
                snapshot = {k: _to_jsonable(v) for k, v in _context_as_dict(ctx).items()}
                snapshot["event"] = summary.model_dump(mode="json")
                snapshot["writeback_summary"] = (
                    writeback.model_dump(mode="json") if writeback is not None else None
                )
                se.event_context_snapshot = snapshot
                await session.flush()

        return ctx

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _clear_degraded_cache(self, event_id: str) -> None:
        self._degraded_cache.pop(event_id, None)
        self._degraded_cache_ts.pop(event_id, None)

    def _get_degraded_if_fresh(self, event_id: str) -> EventContext | None:
        ts = self._degraded_cache_ts.get(event_id)
        cached = self._degraded_cache.get(event_id)
        if ts is None or cached is None:
            return None
        if time.monotonic() - ts > DEGRADED_CACHE_TTL_SECONDS:
            self._clear_degraded_cache(event_id)
            return None
        return cached

    @staticmethod
    async def _upsert_version(session: AsyncSession, event_id: str, field_name: str) -> int:
        result = await session.execute(
            text(
                "INSERT INTO event_context_field_version "
                "(event_id, field_name, current_version) "
                "VALUES (:event_id, :field_name, 1) "
                "ON CONFLICT (event_id, field_name) DO UPDATE "
                "SET current_version = event_context_field_version.current_version + 1 "
                "RETURNING current_version"
            ),
            {"event_id": event_id, "field_name": field_name},
        )
        row = result.one()
        return int(row[0])

    @staticmethod
    async def _insert_journal(
        session: AsyncSession,
        event_id: str,
        field_name: str,
        value: Any,
        version: int,
    ) -> None:
        session.add(
            orm.EventContextJournal(
                event_id=event_id,
                field_name=field_name,
                value=value if isinstance(value, dict) else {"_scalar": value},
                version=version,
            )
        )
        await session.flush()

    @staticmethod
    def _unwrap_journal_value(value: Any) -> Any:
        if isinstance(value, dict) and set(value.keys()) == {"_scalar"}:
            return value["_scalar"]
        return value

    async def _rebuild_from_journal(self, session: AsyncSession, event_id: str) -> EventContext:
        result = await session.execute(
            text(
                "SELECT DISTINCT ON (field_name) field_name, value "
                "FROM event_context_journal "
                "WHERE event_id = :event_id "
                "ORDER BY field_name, version DESC"
            ),
            {"event_id": event_id},
        )
        raw: dict[str, Any] = {}
        for field_name, value in result.all():
            if field_name in CONTEXT_FIELD_NAMES:
                raw[field_name] = self._unwrap_journal_value(value)
        return _assemble_event_context(raw)

    async def get_field_version(self, event_id: str, key: str) -> int | None:
        """Authoritative current version for a field, or None when unset.

        Reads ``event_context_field_version`` (the sole version source); callers
        must not treat the Redis ``{key}__version`` cache as authority.
        """
        async with self._session_factory() as session:
            row = await session.get(orm.EventContextFieldVersion, (event_id, key))
            return int(row.current_version) if row is not None else None

    async def get_versioned_field(self, event_id: str, key: str) -> tuple[Any, int]:
        """Read a field value and its authoritative version in one DB statement."""
        if key not in CONTEXT_FIELD_NAMES:
            raise KeyError(f"unknown EventContext field: {key!r}")
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT journal.value, version.current_version "
                    "FROM event_context_field_version AS version "
                    "JOIN event_context_journal AS journal "
                    "ON journal.event_id = version.event_id "
                    "AND journal.field_name = version.field_name "
                    "AND journal.version = version.current_version "
                    "WHERE version.event_id = :event_id "
                    "AND version.field_name = :field_name"
                ),
                {"event_id": event_id, "field_name": key},
            )
            row = result.first()
        if row is None:
            return None, 0
        return self._unwrap_journal_value(row[0]), int(row[1])

    async def _load_current_field_versions(self, event_id: str) -> dict[str, int]:
        async with self._session_factory() as session:
            return await self._load_field_versions(session, event_id)

    @staticmethod
    async def _load_field_versions(session: AsyncSession, event_id: str) -> dict[str, int]:
        rows = await session.execute(
            select(
                orm.EventContextFieldVersion.field_name,
                orm.EventContextFieldVersion.current_version,
            ).where(orm.EventContextFieldVersion.event_id == event_id)
        )
        return {str(name): int(ver) for name, ver in rows.all()}

    def _context_to_redis_mapping(
        self, ctx: EventContext, versions: dict[str, int]
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key in CONTEXT_FIELD_NAMES:
            mapping[key] = _to_jsonable(getattr(ctx, key))
            if key in versions:
                mapping[version_field(key)] = versions[key]
        return mapping

    def _decode_hash(self, raw_hash: dict[Any, Any]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for raw_key, raw_val in raw_hash.items():
            if isinstance(raw_key, (bytes, bytearray)):
                key = raw_key.decode("utf-8")
            else:
                key = str(raw_key)
            if key.endswith("__version"):
                ver = RedisClient.loads(raw_val)
                decoded[key] = int(ver) if not isinstance(ver, int) else ver
            else:
                decoded[key] = RedisClient.loads(raw_val)
        return decoded

    async def _redis_set_fields(
        self,
        event_id: str,
        fields: dict[str, Any],
        *,
        log_entry: dict[str, Any] | None,
        expire: bool = False,
    ) -> bool:
        """Write Hash fields with retry; append optional change-log entry."""
        key = ctx_key(event_id)
        encoded: dict[str | bytes, bytes] = {}
        for field, value in fields.items():
            encoded[field] = RedisClient.dumps(value)

        last_exc: Exception | None = None
        # One initial attempt plus up to len(backoffs) retries (0.1/0.5/2.0s).
        max_attempts = 1 + len(REDIS_WRITE_BACKOFFS)
        for attempt in range(max_attempts):
            if not await self._redis.ping():
                last_exc = RuntimeError("redis ping failed")
            else:
                try:
                    client = self._redis.get_client()
                    if encoded:
                        await client.hset(key, mapping=encoded)  # type: ignore[arg-type]
                    if log_entry is not None:
                        await client.rpush(ctx_log_key(event_id), RedisClient.dumps(log_entry))
                    if expire:
                        await client.expire(key, CLOSED_TTL_SECONDS)
                    return True
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
            if attempt + 1 < max_attempts:
                await asyncio.sleep(REDIS_WRITE_BACKOFFS[attempt])

        logger.warning(
            "Redis context write failed after retries event_id=%s error_type=%s",
            event_id,
            type(last_exc).__name__ if last_exc is not None else "unknown",
        )
        return False

    async def _merge_writeback_summary(
        self,
        session: AsyncSession,
        se: orm.SecurityEvent,
    ) -> WritebackSummary | None:
        """Recompute the event-level WritebackSummary from Action + outbox rows.

        Always derived fresh from persisted Action.writeback_* fields and
        DispositionOutbox/Receipt rows — never carried forward from a stale
        prior summary (no ``model_construct`` bypass, no "existing" fallback)
        so every rebuild path (Redis miss, journal rebuild, CLOSED snapshot
        refresh) converges on the same, unique correct readiness/status.
        """
        policy = DispositionPolicy(se.disposition_policy)

        actions = (
            await session.scalars(select(orm.Action).where(orm.Action.event_id == se.event_id))
        ).all()
        required_actions = [a for a in actions if a.writeback_required]
        applicable_actions = [a for a in required_actions if a.writeback_applicable]

        outboxes = (
            await session.scalars(
                select(orm.DispositionOutbox).where(orm.DispositionOutbox.event_id == se.event_id)
            )
        ).all()

        if not required_actions and not outboxes:
            if policy is DispositionPolicy.NOT_REQUIRED:
                aggregate_readiness = WritebackReadiness.NOT_REQUIRED
            else:
                # REQUIRED policy but nothing has been planned yet: never
                # invent READY from an empty action set — surface as unknown
                # until a real Action/outbox exists to evaluate.
                aggregate_readiness = WritebackReadiness.CAPABILITY_UNKNOWN
            return WritebackSummary(
                event_id=se.event_id,
                closure_cycle=0,
                disposition_policy=policy,
                aggregate_readiness=aggregate_readiness,
                external_unsynced=bool(se.external_unsynced),
                updated_at=datetime.now(UTC),
            )

        readiness_counts: Counter[WritebackReadiness] = Counter()
        blocked_action_ids: list[str] = []
        for action in applicable_actions:
            try:
                readiness = WritebackReadiness(action.writeback_readiness)
            except ValueError:
                readiness = WritebackReadiness.CAPABILITY_UNKNOWN
            readiness_counts[readiness] += 1
            if readiness is not WritebackReadiness.READY:
                blocked_action_ids.append(action.action_id)

        if applicable_actions:
            picked = _pick_by_priority(set(readiness_counts), READINESS_AGGREGATE_PRIORITY)
            assert picked is not None
            aggregate_readiness = cast(WritebackReadiness, picked)
        elif required_actions:
            # Required policy, but no action is (yet) applicable to a writable
            # source object — never invent READY from an empty applicable set.
            aggregate_readiness = WritebackReadiness.CAPABILITY_UNKNOWN
        else:
            aggregate_readiness = WritebackReadiness.NOT_REQUIRED

        writeback_ids = {o.writeback_id for o in outboxes}
        receipts_by_wb: dict[str, orm.DispositionReceipt] = {}
        if writeback_ids:
            receipt_rows = (
                await session.scalars(
                    select(orm.DispositionReceipt).where(
                        orm.DispositionReceipt.writeback_id.in_(writeback_ids)
                    )
                )
            ).all()
            for receipt in receipt_rows:
                prev = receipts_by_wb.get(receipt.writeback_id)
                if prev is None or receipt.sequence > prev.sequence:
                    receipts_by_wb[receipt.writeback_id] = receipt

        status_counts: Counter[WritebackStatus] = Counter()
        terminal_event_action_id: str | None = None
        terminal_event_writeback_id: str | None = None
        terminal_event_disposition: SourceDisposition | None = None
        terminal_event_confirmed = False
        closure_cycle = 0

        for outbox in outboxes:
            closure_cycle = max(closure_cycle, int(outbox.closure_cycle or 0))
            status_raw = outbox.latest_writeback_status
            latest_receipt = receipts_by_wb.get(outbox.writeback_id)
            if latest_receipt is not None:
                status_raw = latest_receipt.status
            if status_raw:
                try:
                    status = WritebackStatus(status_raw)
                except ValueError:
                    status = WritebackStatus.UNKNOWN
                status_counts[status] += 1

            if outbox.intent_kind == DispositionIntentKind.EVENT_STATUS_UPDATE.value:
                terminal_event_action_id = outbox.action_id
                terminal_event_writeback_id = outbox.writeback_id
                if (
                    latest_receipt is not None
                    and latest_receipt.status == WritebackStatus.CONFIRMED.value
                ):
                    terminal_event_confirmed = True
                payload = outbox.command_payload or {}
                disp = payload.get("disposition") or payload.get("source_disposition")
                if isinstance(disp, str):
                    try:
                        terminal_event_disposition = SourceDisposition(disp)
                    except ValueError:
                        terminal_event_disposition = None

        aggregate_status = (
            _pick_by_priority(set(status_counts), STATUS_AGGREGATE_PRIORITY)
            if status_counts
            else None
        )

        return WritebackSummary(
            event_id=se.event_id,
            closure_cycle=closure_cycle,
            disposition_policy=policy,
            required_action_count=len(required_actions),
            applicable_action_count=len(applicable_actions),
            blocked_action_ids=blocked_action_ids,
            readiness_counts=dict(readiness_counts),
            aggregate_readiness=aggregate_readiness,
            writeback_counts=dict(status_counts),
            aggregate_status=aggregate_status,
            terminal_event_action_id=terminal_event_action_id,
            terminal_event_writeback_id=terminal_event_writeback_id,
            terminal_event_disposition=terminal_event_disposition,
            terminal_event_confirmed=terminal_event_confirmed,
            external_unsynced=bool(se.external_unsynced),
            updated_at=datetime.now(UTC),
        )
