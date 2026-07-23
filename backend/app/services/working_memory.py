"""WorkingMemory + FIELD_OWNERSHIP (ISSUE-014 / intro §4.11)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.config import get_settings
from app.core.errors import GuardrailViolationError
from app.core.redis_client import RedisClient
from app.models.context import EventContext
from app.models.working_memory import MemoryAccessLog, ScratchpadEntry
from app.services.context_service import EventContextStore
from app.services.degraded_flag_service import DegradedFlagService

logger = logging.getLogger(__name__)

SCRATCHPAD_LIMIT = 200
WM_KEY_PREFIX = "shadowtrace:wm:"
WRITE_CAS_MAX_ATTEMPTS = 3

# --------------------------------------------------------------------------- #
# FIELD_OWNERSHIP — exact EventContext field → trusted writer identity
# --------------------------------------------------------------------------- #

FIELD_OWNERSHIP: dict[str, str] = {
    "event": "EventService",
    "source_snapshot": "EventService",
    "source_sync_state": "SourceIngester",
    "triage_result": "TriageAgent",
    "false_positive_match": "FalsePositiveMatcher",
    "evidence_output": "EvidenceAgent",
    "storyline": "StorylineService",
    "graph_output": "GraphAgent",
    "rag_output": "RAGAgent",
    "risk_assessment": "RiskAgent",
    "execution_plan": "PlannerAgent",
    "response_plan": "ResponseAgent",
    "approval_records": "ApprovalEngine",
    "disposition_only_intent": "WorkflowRuntimeService",
    "execution_substate": "WorkflowRuntimeService",
    "execution_summary": "ActionExecutionService",
    "execution_jobs": "ActionExecutionService",
    "verification_result": "VerifyAgent",
    "rollback_results": "RollbackService",
    "impact_assessments": "ImpactAssessmentService",
    "report": "ReportAgent",
    "memory_output": "MemoryAgent",
    "disposition_commands": "DispositionSyncService",
    "disposition_receipts": "DispositionSyncService",
    "writeback_summary": "DispositionSyncService",
    "state_history": "StateMachineService",
    "replan_count": "StateMachineService",
    "budget_usage": "BudgetService",
    "guard_violations": "OutputGuard",
    "convergence_state": "ConvergenceGuard",
    "quality_scores": "OutputQualityEvaluator",
    "scratchpad": "WorkingMemory",
    "degraded_flags": "DegradedFlagService",
    "triage_degraded": "TriageAgent",
    "graph_degraded": "GraphAgent",
}

# P0 RuleBasedFalsePositiveHook shares the FalsePositiveMatcher writer identity.
WRITER_ALIASES: dict[str, str] = {
    "RuleBasedFalsePositiveHook": "FalsePositiveMatcher",
}


def _validate_field_ownership() -> None:
    """Fail fast if ownership drifts from the EventContext schema (both directions)."""
    schema_fields = set(EventContext.model_fields.keys())
    owned_fields = set(FIELD_OWNERSHIP.keys())
    missing = schema_fields - owned_fields
    ghost = owned_fields - schema_fields
    if missing or ghost:
        raise RuntimeError(
            "FIELD_OWNERSHIP must exactly cover EventContext fields: "
            f"missing={sorted(missing)} ghost={sorted(ghost)}"
        )


_validate_field_ownership()


def wm_key(event_id: str) -> str:
    return f"{WM_KEY_PREFIX}{event_id}"


def normalize_writer(writer: str) -> str:
    """Map known aliases onto the canonical FIELD_OWNERSHIP identity."""
    return WRITER_ALIASES.get(writer, writer)


@dataclass(frozen=True, slots=True)
class WriterCapability:
    """Opaque writer identity issued and tracked by one WorkingMemory instance."""

    owner: str
    _nonce: object


@dataclass(frozen=True, slots=True)
class BoundWorkingMemory:
    """Agent-facing memory view bound to one non-self-reported writer identity."""

    _memory: WorkingMemory
    _capability: WriterCapability

    @property
    def writer_name(self) -> str:
        return self._capability.owner

    async def read(self, event_id: str, key: str) -> Any:
        return await self._memory.read(event_id, key, reader=self._capability)

    async def write(self, event_id: str, key: str, value: Any) -> None:
        await self._memory.write(event_id, key, value, writer=self._capability)

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        await self._memory.append_scratchpad(event_id, note, writer=self._capability)

    async def read_scratchpad(self, event_id: str) -> list[ScratchpadEntry]:
        return await self._memory.read_scratchpad(event_id, reader=self._capability)

    def for_writer(self, writer: str) -> BoundWorkingMemory:
        """Mint a new ``BoundWorkingMemory`` for *writer* from the same backing
        ``WorkingMemory``, preserving the single-instance invariants.
        """
        return self._memory.for_writer(writer)


class WorkingMemory:
    """Owner-gated EventContext access with scratchpad + access audit."""

    def __init__(
        self,
        store: EventContextStore,
        redis: RedisClient,
        *,
        degraded_flags: DegradedFlagService | None = None,
        wm_strict: bool | None = None,
    ) -> None:
        self._store = store
        self._redis = redis
        self._degraded_flags = degraded_flags
        self._wm_strict = get_settings().wm_strict if wm_strict is None else wm_strict
        self._access_logs: dict[str, list[MemoryAccessLog]] = {}
        self._redis_degrade_marked: set[str] = set()
        self._issued_capabilities: dict[WriterCapability, str] = {}

    def bind_degraded_flag_service(self, service: DegradedFlagService) -> None:
        """Wire DegradedFlagService after construction (breaks init cycles)."""
        self._degraded_flags = service

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def for_writer(self, writer: str) -> BoundWorkingMemory:
        """Bind a trusted composition-root identity to an agent-safe memory view."""
        canonical = normalize_writer(writer)
        if canonical not in set(FIELD_OWNERSHIP.values()):
            raise GuardrailViolationError(
                f"unknown working-memory writer identity: {writer!r}",
                error_code="working_memory_unauthorized_write",
                details={"writer": writer},
            )
        capability = WriterCapability(owner=canonical, _nonce=object())
        self._issued_capabilities[capability] = canonical
        return BoundWorkingMemory(self, capability)

    async def read(
        self,
        event_id: str,
        key: str,
        reader: WriterCapability,
    ) -> Any:
        reader_name = self._resolve_capability(reader)
        if key not in FIELD_OWNERSHIP:
            raise GuardrailViolationError(
                f"unregistered EventContext field: {key!r}",
                error_code="working_memory_unauthorized_write",
                details={"event_id": event_id, "key": key, "reader": reader_name},
            )
        value = await self._store.get(event_id, key)
        self._record_access(
            event_id,
            agent_name=reader_name,
            op="read",
            key=key,
            allowed=True,
        )
        return value

    async def write(
        self,
        event_id: str,
        key: str,
        value: Any,
        writer: WriterCapability,
    ) -> None:
        writer_name = self._capability_label(writer)
        if key not in FIELD_OWNERSHIP:
            self._record_access(
                event_id,
                agent_name=writer_name,
                op="write",
                key=key,
                allowed=False,
            )
            raise GuardrailViolationError(
                f"unregistered EventContext field: {key!r}",
                error_code="working_memory_unauthorized_write",
                details={"event_id": event_id, "key": key, "writer": writer_name},
            )

        owner = FIELD_OWNERSHIP[key]
        canonical = self._resolve_capability(writer)
        if canonical != owner:
            self._record_access(
                event_id,
                agent_name=canonical,
                op="write",
                key=key,
                allowed=False,
            )
            raise GuardrailViolationError(
                f"writer {canonical!r} is not owner of {key!r} (owner={owner!r})",
                error_code="working_memory_unauthorized_write",
                details={
                    "event_id": event_id,
                    "key": key,
                    "writer": canonical,
                    "owner": owner,
                },
            )

        await self._write_with_version_retry(event_id, key, value)
        self._record_access(event_id, agent_name=canonical, op="write", key=key, allowed=True)

    async def append_scratchpad(
        self,
        event_id: str,
        note: str,
        *,
        writer: WriterCapability,
    ) -> None:
        agent_name = self._resolve_capability(writer)
        entry = ScratchpadEntry(
            agent_name=agent_name,
            timestamp=datetime.now(UTC),
            note=note,
        )
        serialized = entry.model_dump(mode="json")

        def append_to(current: Any) -> list[Any]:
            entries = list(current) if isinstance(current, list) else []
            entries.append(serialized)
            return entries[-SCRATCHPAD_LIMIT:]

        entries = await self._write_with_version_retry(
            event_id,
            "scratchpad",
            transform=append_to,
        )
        self._record_access(
            event_id,
            agent_name=agent_name,
            op="write",
            key="scratchpad",
            allowed=True,
        )
        await self._mirror_wm_scratchpad(event_id, entries)

    async def read_scratchpad(
        self,
        event_id: str,
        *,
        reader: WriterCapability,
    ) -> list[ScratchpadEntry]:
        raw = await self.read(event_id, "scratchpad", reader=reader)
        if not raw:
            return []
        if not isinstance(raw, list):
            return []
        return [ScratchpadEntry.model_validate(item) for item in raw]

    async def get_access_log(self, event_id: str) -> list[MemoryAccessLog]:
        return list(self._access_logs.get(event_id, []))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_capability(self, capability: WriterCapability) -> str:
        try:
            return self._issued_capabilities[capability]
        except (KeyError, TypeError) as exc:
            raise GuardrailViolationError(
                "unrecognized working-memory writer capability",
                error_code="working_memory_unauthorized_write",
                details={"writer": self._capability_label(capability)},
            ) from exc

    @staticmethod
    def _capability_label(capability: object) -> str:
        if isinstance(capability, WriterCapability):
            return capability.owner
        return f"<invalid:{type(capability).__name__}>"

    def _record_access(
        self,
        event_id: str,
        *,
        agent_name: str,
        op: str,
        key: str,
        allowed: bool,
    ) -> None:
        log = MemoryAccessLog(
            timestamp=datetime.now(UTC),
            agent_name=agent_name,
            op=op,  # type: ignore[arg-type]
            key=key,
            allowed=allowed,
        )
        self._access_logs.setdefault(event_id, []).append(log)

    async def _write_with_version_retry(
        self,
        event_id: str,
        key: str,
        value: Any = None,
        *,
        transform: Callable[[Any], Any] | None = None,
    ) -> Any:
        """CAS a value, recomputing mutations from the latest DB value on conflict."""
        last_conflict = False
        for attempt in range(WRITE_CAS_MAX_ATTEMPTS):
            if transform is None:
                expected = await self._read_field_version(event_id, key)
            else:
                current, expected = await self._store.get_versioned_field(event_id, key)
                value = transform(current)
            ok = await self._store.compare_and_set(
                event_id,
                key,
                expected or 0,
                value,
            )
            if ok:
                # compare_and_set does not return redis_ok; probe after success.
                redis_ok = await self._redis.ping()
                await self._maybe_mark_redis_unavailable(event_id, redis_ok)
                return value
            last_conflict = True
            logger.info(
                "WorkingMemory CAS conflict event_id=%s key=%s attempt=%s",
                event_id,
                key,
                attempt + 1,
            )

        if last_conflict:
            raise GuardrailViolationError(
                f"version conflict writing {key!r} after {WRITE_CAS_MAX_ATTEMPTS} attempts",
                error_code="version_conflict",
                details={"event_id": event_id, "key": key},
            )

    async def _read_field_version(self, event_id: str, key: str) -> int | None:
        """Authoritative current version from the DB, not the Redis cache.

        The Redis ``{key}__version`` companion is only a cache and can lag the
        ``event_context_field_version`` table after a degraded (Redis-down) write;
        using it as CAS ``expected`` would spuriously fail a legitimate owner write.
        """
        return await self._store.get_field_version(event_id, key)

    async def _maybe_mark_redis_unavailable(self, event_id: str, redis_ok: bool) -> None:
        if redis_ok:
            return
        if event_id in self._redis_degrade_marked:
            return
        if self._degraded_flags is None:
            logger.warning(
                "redis_ok=false but DegradedFlagService not bound event_id=%s",
                event_id,
            )
            return
        if await self._degraded_flags.has_flag(event_id, "redis_context_unavailable"):
            self._redis_degrade_marked.add(event_id)
            return
        await self._degraded_flags.set_flag(
            event_id,
            "redis_context_unavailable",
            True,
            writer="WorkingMemory",
        )
        self._redis_degrade_marked.add(event_id)

    async def _mirror_wm_scratchpad(self, event_id: str, entries: list[Any]) -> None:
        """Best-effort mirror into ``shadowtrace:wm:{event_id}`` Hash."""
        if not await self._redis.ping():
            return
        try:
            client = self._redis.get_client()
            await client.hset(
                wm_key(event_id),
                "scratchpad",
                RedisClient.dumps(entries),
            )
        except Exception:  # noqa: BLE001 — draft mirror must not fail the write
            logger.warning(
                "failed to mirror scratchpad to wm key event_id=%s",
                event_id,
                exc_info=True,
            )
