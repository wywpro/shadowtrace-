"""EvidenceAgent: sequential 7-source evidence collection (ISSUE-033).

Concurrency upgrade is ISSUE-034. This module only implements the serial path.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.base import BaseAgent
from app.agents.evidence_parser import (
    TOOL_SOURCE_MAP,
    EvidenceParser,
    truncate_timestamp_to_second,
)
from app.db import models as orm
from app.models.agent_io import CollectionStatus, EvidenceAgentInput, EvidenceOutput
from app.models.entities import EntitySet
from app.models.enums import EvidenceSource
from app.models.evidence import Evidence, EvidenceGap
from app.models.tool_meta import ToolResult, ToolResultStatus
from app.services.evidence_projection import (
    EvidenceQueryScope,
    bind_evidence_query_scope,
)

logger = logging.getLogger(__name__)

# Fixed serial query order (ISSUE-033).
EVIDENCE_QUERY_ORDER: tuple[str, ...] = (
    "query_account_login",
    "query_edr_process",
    "query_file_access",
    "query_network_flow",
    "query_dns",
    "query_asset_info",
    "query_threat_intel",
)

# Default window when TriageResult has no explicit time range (ISSUE-005 has none).
DEFAULT_TIME_RANGE: dict[str, str] = {
    "start": "2024-06-15T08:00:00Z",
    "end": "2024-06-15T10:00:00Z",
}

_SUCCESS_STATUSES = frozenset(
    {
        ToolResultStatus.SUCCESS,
        ToolResultStatus.PARTIAL_SUCCESS,
        ToolResultStatus.ACCEPTED,
    }
)

_STATUS_PENALTY: dict[CollectionStatus, float] = {
    CollectionStatus.COMPLETED: 0.0,
    CollectionStatus.PARTIAL_DONE: 0.10,
    CollectionStatus.DEGRADED: 0.25,
    CollectionStatus.FAILED: 0.0,
}


class EvidenceRepository(Protocol):
    """Persistence port for Evidence rows (conflict on evidence_id)."""

    async def upsert_batch(self, evidence_list: list[Evidence]) -> None: ...

    async def list_by_event(self, event_id: str) -> list[Evidence]: ...


class InMemoryEvidenceRepository:
    """Test / degraded store: keep higher-confidence row on evidence_id conflict."""

    def __init__(self) -> None:
        self._rows: dict[str, Evidence] = {}

    async def upsert_batch(self, evidence_list: list[Evidence]) -> None:
        for item in evidence_list:
            existing = self._rows.get(item.evidence_id)
            if existing is None or item.confidence > existing.confidence:
                self._rows[item.evidence_id] = item

    async def list_by_event(self, event_id: str) -> list[Evidence]:
        return [row for row in self._rows.values() if row.event_id == event_id]


class SqlAlchemyEvidenceRepository:
    """Postgres upsert for ``evidence`` table (ISSUE-033 step 5)."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert_batch(self, evidence_list: list[Evidence]) -> None:
        if not evidence_list:
            return
        async with self._session_factory() as session:
            async with session.begin():
                for item in evidence_list:
                    values = {
                        "evidence_id": item.evidence_id,
                        "event_id": item.event_id,
                        "source": item.source.value
                        if isinstance(item.source, EvidenceSource)
                        else str(item.source),
                        "evidence_type": item.evidence_type,
                        "description": item.description,
                        "confidence": item.confidence,
                        "timestamp": item.timestamp,
                        "related_entities": list(item.related_entities),
                        "source_ref": (
                            item.source_ref.model_dump(mode="json")
                            if item.source_ref is not None
                            else None
                        ),
                        "raw_data": dict(item.raw_data),
                        "mitre_technique": item.mitre_technique,
                        "is_conflicting": item.is_conflicting,
                    }
                    stmt = pg_insert(orm.Evidence).values(**values)
                    excluded = stmt.excluded
                    # Keep the higher-confidence row on evidence_id conflict.
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["evidence_id"],
                        set_={
                            "event_id": excluded.event_id,
                            "source": excluded.source,
                            "evidence_type": excluded.evidence_type,
                            "description": excluded.description,
                            "confidence": excluded.confidence,
                            "timestamp": excluded.timestamp,
                            "related_entities": excluded.related_entities,
                            "source_ref": excluded.source_ref,
                            "raw_data": excluded.raw_data,
                            "mitre_technique": excluded.mitre_technique,
                            "is_conflicting": excluded.is_conflicting,
                        },
                        where=(orm.Evidence.confidence < excluded.confidence),
                    )
                    await session.execute(stmt)

    async def list_by_event(self, event_id: str) -> list[Evidence]:
        from sqlalchemy import select

        from app.models.source import SourceReference

        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(orm.Evidence).where(orm.Evidence.event_id == event_id)
                    )
                )
                .scalars()
                .all()
            )
            result: list[Evidence] = []
            for row in rows:
                source_ref = None
                if isinstance(row.source_ref, dict):
                    try:
                        source_ref = SourceReference.model_validate(row.source_ref)
                    except Exception:
                        source_ref = None
                result.append(
                    Evidence(
                        evidence_id=row.evidence_id,
                        event_id=row.event_id,
                        source=EvidenceSource(row.source),
                        evidence_type=row.evidence_type,
                        description=row.description,
                        confidence=row.confidence,
                        timestamp=row.timestamp,
                        related_entities=list(row.related_entities or []),
                        source_ref=source_ref,
                        raw_data=dict(row.raw_data or {}),
                        mitre_technique=row.mitre_technique,
                        is_conflicting=bool(row.is_conflicting),
                    )
                )
            return result


class EvidenceAgent(BaseAgent[EvidenceAgentInput, EvidenceOutput]):
    """Sequential EvidenceAgent — concurrency upgrade is ISSUE-034."""

    agent_name = "evidence_agent"

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        tool_executor: Any | None = None,
        working_memory: Any | None = None,
        budget_service: Any | None = None,
        output_guard: Any | None = None,
        trace_service: Any | None = None,
        audit_service: Any | None = None,
        event_bus: Any | None = None,
        event_service: Any | None = None,
        evidence_repository: EvidenceRepository | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        evidence_parser: EvidenceParser | None = None,
        default_time_range: dict[str, str] | None = None,
        query_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            tool_executor=tool_executor,
            working_memory=working_memory,
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
            audit_service=audit_service,
            event_bus=event_bus,
        )
        self.event_service = event_service
        if evidence_repository is not None:
            self.evidence_repository: EvidenceRepository | None = evidence_repository
        elif session_factory is not None:
            self.evidence_repository = SqlAlchemyEvidenceRepository(session_factory)
        else:
            self.evidence_repository = InMemoryEvidenceRepository()
        self.parser = evidence_parser or EvidenceParser()
        self.default_time_range = dict(default_time_range or DEFAULT_TIME_RANGE)
        self.query_timeout_s = query_timeout_s
        # Populated each run for agent_trace / acceptance checks.
        self.last_query_timings: list[dict[str, Any]] = []

    async def _run(self, input: EvidenceAgentInput) -> EvidenceOutput:
        if self.tool_executor is None:
            raise RuntimeError("EvidenceAgent requires tool_executor")

        self.last_query_timings = []
        time_range = self._resolve_time_range(input)
        collected: list[Evidence] = []
        success_sources: list[str] = []
        failed_sources: list[str] = []
        gaps: list[EvidenceGap] = []

        scope = await self._resolve_scope(input.event_id)

        for tool_name in EVIDENCE_QUERY_ORDER:
            source = TOOL_SOURCE_MAP[tool_name]
            source_value = source.value
            params = self._build_params(
                tool_name,
                input.triage_result.entities,
                time_range,
                ioc_list=list(input.triage_result.ioc_list),
            )
            if params is None:
                failed_sources.append(source_value)
                gaps.append(
                    EvidenceGap(
                        event_id=input.event_id,
                        missing_source=source,
                        reason="missing_entity",
                        detail={"tool_name": tool_name},
                    )
                )
                self.last_query_timings.append(
                    {
                        "tool_name": tool_name,
                        "source": source_value,
                        "status": "skipped_missing_entity",
                        "execution_time_ms": 0,
                    }
                )
                await self._note_timing(
                    input.event_id,
                    tool_name,
                    status="skipped_missing_entity",
                    execution_time_ms=0,
                )
                continue

            tool_result, timing_ms, call_error = await self._call_query(
                tool_name,
                params,
                input.event_id,
                scope=scope,
            )
            status_text = (
                tool_result.status.value
                if tool_result is not None
                else f"error:{call_error}"
            )
            self.last_query_timings.append(
                {
                    "tool_name": tool_name,
                    "source": source_value,
                    "status": status_text,
                    "execution_time_ms": timing_ms,
                }
            )
            await self._note_timing(
                input.event_id,
                tool_name,
                status=status_text,
                execution_time_ms=timing_ms,
            )

            if tool_result is None or tool_result.status not in _SUCCESS_STATUSES:
                failed_sources.append(source_value)
                gaps.append(
                    EvidenceGap(
                        event_id=input.event_id,
                        missing_source=source,
                        reason="tool_failed",
                        detail={
                            "tool_name": tool_name,
                            "execution_time_ms": timing_ms,
                            "error": call_error
                            or (tool_result.error_detail if tool_result else None),
                            "status": (
                                tool_result.status.value if tool_result is not None else None
                            ),
                        },
                    )
                )
                continue

            parsed = self.parser.parse(
                tool_name,
                tool_result,
                event_id=input.event_id,
            )
            if not parsed:
                # Tool succeeded but produced no usable evidence → gap, not hard fail.
                gaps.append(
                    EvidenceGap(
                        event_id=input.event_id,
                        missing_source=source,
                        reason="no_records",
                        detail={
                            "tool_name": tool_name,
                            "execution_time_ms": timing_ms,
                        },
                    )
                )
                continue

            collected.extend(parsed)
            if source_value not in success_sources:
                success_sources.append(source_value)

        evidence_list = self._dedup_and_sort(collected)
        collection_status = self._collection_status(len(success_sources))
        overall_confidence = self._overall_confidence(evidence_list, collection_status)

        output = EvidenceOutput(
            evidence_list=evidence_list,
            conflicts=[],
            gaps=gaps,
            success_sources=success_sources,
            failed_sources=failed_sources,
            overall_confidence=overall_confidence,
            collection_status=collection_status,
        )

        await self._persist_evidence(evidence_list)
        await self._write_context(input.event_id, output)
        return output

    async def _resolve_scope(self, event_id: str) -> EvidenceQueryScope | None:
        if self.event_service is None:
            return None
        try:
            return await self.event_service.get_evidence_query_scope(event_id)
        except Exception:
            logger.warning(
                "failed to resolve evidence query scope for event=%s",
                event_id,
                exc_info=True,
            )
            return None

    async def _call_query(
        self,
        tool_name: str,
        params: dict[str, Any],
        event_id: str,
        *,
        scope: EvidenceQueryScope | None,
    ) -> tuple[ToolResult | None, int, str | None]:
        assert self.tool_executor is not None
        try:
            if scope is not None:
                with bind_evidence_query_scope(scope):
                    result = await self.tool_executor.call(
                        tool_name,
                        params,
                        event_id,
                        timeout=self.query_timeout_s,
                        agent_name=self.agent_name,
                    )
            else:
                result = await self.tool_executor.call(
                    tool_name,
                    params,
                    event_id,
                    timeout=self.query_timeout_s,
                    agent_name=self.agent_name,
                )
            tool_result = (
                result if isinstance(result, ToolResult) else ToolResult.model_validate(result)
            )
            timing = int(tool_result.execution_time_ms or 0)
            return tool_result, timing, None
        except Exception as exc:
            logger.info(
                "evidence query failed tool=%s event=%s err=%s",
                tool_name,
                event_id,
                exc,
            )
            return None, 0, str(exc)

    async def _note_timing(
        self,
        event_id: str,
        tool_name: str,
        *,
        status: str,
        execution_time_ms: int,
    ) -> None:
        if self.working_memory is None:
            return
        note = (
            f"evidence_query tool={tool_name} status={status} "
            f"execution_time_ms={execution_time_ms}"
        )
        try:
            await self.working_memory.append_scratchpad(event_id, note)
        except Exception:
            logger.debug("scratchpad timing note failed", exc_info=True)

    async def _persist_evidence(self, evidence_list: list[Evidence]) -> None:
        if self.evidence_repository is None or not evidence_list:
            return
        try:
            await self.evidence_repository.upsert_batch(evidence_list)
        except Exception:
            logger.warning("evidence upsert failed; continuing without DB write", exc_info=True)

    async def _write_context(self, event_id: str, output: EvidenceOutput) -> None:
        if self.working_memory is None:
            return
        try:
            await self.working_memory.write(
                event_id,
                "evidence_output",
                output.model_dump(mode="json"),
            )
        except Exception:
            logger.warning(
                "failed to write evidence_output to working memory event=%s",
                event_id,
                exc_info=True,
            )

    def _resolve_time_range(self, _input: EvidenceAgentInput) -> dict[str, str]:
        return dict(self.default_time_range)

    def _build_params(
        self,
        tool_name: str,
        entities: EntitySet,
        time_range: dict[str, str],
        *,
        ioc_list: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Build tool params from triage entities — no hardcoded demo identities."""
        account = next((a.username for a in entities.accounts if a.username), None)
        host = next((h for h in entities.hosts if h.hostname or h.ip), None)
        hostname = host.hostname if host else None
        host_ip = host.ip if host else None
        internal_ip = next(
            (ip.address for ip in entities.ips if ip.address and ip.scope == "internal"),
            host_ip,
        )
        external_ip = next(
            (ip.address for ip in entities.ips if ip.address and ip.scope == "external"),
            None,
        )
        domain = next((d.fqdn for d in entities.domains if d.fqdn), None)
        iocs = [item for item in (ioc_list or []) if item]
        indicator = next(iter(iocs), None) or external_ip or domain

        if tool_name == "query_account_login":
            if not account:
                return None
            return {"account": account, "time_range": time_range}
        if tool_name == "query_edr_process":
            if not hostname:
                return None
            return {"host_id": hostname, "time_range": time_range}
        if tool_name == "query_file_access":
            if not account:
                return None
            return {"account": account, "time_range": time_range}
        if tool_name == "query_network_flow":
            if not internal_ip and not external_ip:
                return None
            params: dict[str, Any] = {"time_range": time_range}
            if internal_ip:
                params["src_ip"] = internal_ip
            elif external_ip:
                params["dst_ip"] = external_ip
            return params
        if tool_name == "query_dns":
            if not domain:
                return None
            return {"domain": domain, "time_range": time_range}
        if tool_name == "query_asset_info":
            if not host_ip and not hostname:
                return None
            params = {}
            if host_ip:
                params["ip"] = host_ip
            if hostname:
                params["hostname"] = hostname
            return params
        if tool_name == "query_threat_intel":
            if not indicator:
                return None
            return {"indicator": indicator}
        return None

    @staticmethod
    def _dedup_and_sort(items: list[Evidence]) -> list[Evidence]:
        best: dict[tuple[str, str, datetime | None], Evidence] = {}
        for item in items:
            ts = truncate_timestamp_to_second(item.timestamp)
            key = (
                item.source.value if isinstance(item.source, EvidenceSource) else str(item.source),
                item.evidence_type,
                ts,
            )
            existing = best.get(key)
            if existing is None or item.confidence > existing.confidence:
                if ts != item.timestamp:
                    item = item.model_copy(update={"timestamp": ts})
                best[key] = item

        def sort_key(ev: Evidence) -> tuple[int, datetime]:
            if ev.timestamp is None:
                return (1, datetime.min.replace(tzinfo=UTC))
            return (0, ev.timestamp)

        return sorted(best.values(), key=sort_key)

    @staticmethod
    def _collection_status(success_count: int) -> CollectionStatus:
        # 统一命名规则优先于「2 路失败」散文：按成功源数量判定。
        if success_count >= 5:
            return CollectionStatus.COMPLETED
        if success_count >= 3:
            return CollectionStatus.PARTIAL_DONE
        if success_count >= 1:
            return CollectionStatus.DEGRADED
        return CollectionStatus.FAILED

    @staticmethod
    def _overall_confidence(
        evidence_list: list[Evidence],
        collection_status: CollectionStatus,
    ) -> float:
        if not evidence_list:
            return 0.0
        avg_conf = sum(item.confidence for item in evidence_list) / len(evidence_list)
        unique_sources = {item.source for item in evidence_list}
        diversity = min(len(unique_sources), 5) / 5.0 * 0.15
        quantity = min(len(evidence_list), 6) / 6.0 * 0.1
        penalty = _STATUS_PENALTY[collection_status]
        score = avg_conf * 0.75 + diversity + quantity - penalty
        return min(1.0, max(0.0, score))
