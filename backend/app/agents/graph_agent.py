"""GraphAgent: entity-relationship graph from evidence (ISSUE-050).

Builds a PostgreSQL-backed graph from EvidenceOutput, computes centrality
and attack-path candidates, persists nodes/edges, and writes ``graph_output``
to EventContext via WorkingMemory.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.base import BaseAgent
from app.agents.graph_builder import GraphBuilder
from app.core.errors import ShadowTraceError
from app.db.orm.graph import GraphEdgeORM, GraphNodeORM
from app.models.agent_io import GraphAgentInput, GraphOutput
from app.models.evidence import Evidence

logger = logging.getLogger(__name__)

MAX_PATH_DEPTH = 6
MAX_ATTACK_PATHS = 3


class GraphAgent(BaseAgent[GraphAgentInput, GraphOutput]):
    """Transform evidence into an entity-relationship graph.

    Persists nodes and edges to PostgreSQL (graph_node / graph_edge tables),
    computes degree-based centrality (top 3), and discovers time-monotonic
    attack path candidates.  Graph construction failure records a degraded
    flag but does not block the investigation pipeline (降级策略).
    """

    agent_name: str = "graph_agent"

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
        session_factory: async_sessionmaker[AsyncSession] | None = None,
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
        self._session_factory = session_factory
        self.last_persist_error: str | None = None
        self.last_persist_ok: bool = False
        self.last_degraded_reason: str | None = None

    # ------------------------------------------------------------------ #
    # Public entry-point
    # ------------------------------------------------------------------ #

    async def _run(self, input: GraphAgentInput) -> GraphOutput:
        event_id = input.event_id
        evidence_list: list[Evidence] = input.evidence_output.evidence_list
        self.last_degraded_reason = None

        # 1. Build graph from evidence (pure in-memory transformation)
        try:
            nodes, edges = GraphBuilder.build(evidence_list)
        except Exception:
            logger.exception("GraphBuilder failed for event=%s", event_id)
            output = self._empty_degraded()
            await self._mark_degraded(event_id, reason="graph_builder_failed")
            await self._write_context(event_id, output)
            return output

        # 2. Compute centrality (top 3 entities by degree)
        central_entities = _compute_central_entities(nodes, edges)

        # 3. Compute attack-path candidates (time-monotonic, depth ≤ 6, max 3)
        attack_path_candidates = _find_attack_paths(nodes, edges)

        # 4. Build output
        output = GraphOutput(
            nodes=nodes,
            edges=edges,
            central_entities=central_entities,
            attack_path_candidates=attack_path_candidates,
        )

        # 5. Persist to PostgreSQL (best-effort; degrades on failure)
        await self._persist_graph(event_id, nodes, edges)
        if self.last_persist_error is not None:
            await self._mark_degraded(
                event_id,
                reason=f"graph_persist_failed: {self.last_persist_error}",
            )

        # 6. Write to EventContext via WorkingMemory
        await self._write_context(event_id, output)

        return output

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    async def _persist_graph(
        self,
        event_id: str,
        nodes: list[Any],
        edges: list[Any],
    ) -> None:
        """Upsert nodes and edges into PostgreSQL.  Best-effort: failure sets
        ``last_persist_error`` and logs the degraded state but does NOT raise."""
        if self._session_factory is None:
            self.last_persist_error = "no session_factory configured"
            logger.warning("GraphAgent persist skipped: %s", self.last_persist_error)
            return

        self.last_persist_error = None
        self.last_persist_ok = False
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await self._upsert_nodes(session, event_id, nodes)
                    await self._upsert_edges(session, event_id, edges)
            self.last_persist_ok = True
        except Exception as exc:
            self.last_persist_error = str(exc)
            logger.exception("GraphAgent persist failed for event=%s", event_id)

    @staticmethod
    async def _upsert_nodes(
        session: AsyncSession,
        event_id: str,
        nodes: list[Any],
    ) -> None:
        if not nodes:
            return
        rows = [
            {
                "node_id": n.node_id,
                "event_id": n.event_id,
                "entity_type": n.entity_type,
                "entity_value": n.entity_value,
                "properties": n.properties,
            }
            for n in nodes
        ]
        stmt = pg_insert(GraphNodeORM).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_graph_node_identity",
            set_={"properties": stmt.excluded.properties},
        )
        await session.execute(stmt)

    @staticmethod
    async def _upsert_edges(
        session: AsyncSession,
        event_id: str,
        edges: list[Any],
    ) -> None:
        if not edges:
            return
        rows = [
            {
                "edge_id": e.edge_id,
                "event_id": e.event_id,
                "source_node_id": e.source_node_id,
                "target_node_id": e.target_node_id,
                "relation_type": e.relation_type.value
                if hasattr(e.relation_type, "value")
                else str(e.relation_type),
                "evidence_id": e.evidence_id,
                "occurred_at": e.occurred_at,
            }
            for e in edges
        ]
        # Edge dedup: ON CONFLICT DO NOTHING (edge_id is PK, derived deterministically)
        stmt = (
            pg_insert(GraphEdgeORM)
            .values(rows)
            .on_conflict_do_nothing(
                index_elements=["edge_id"],
            )
        )
        await session.execute(stmt)

    # ------------------------------------------------------------------ #
    # WorkingMemory
    # ------------------------------------------------------------------ #

    async def _write_context(self, event_id: str, output: GraphOutput) -> None:
        if self.working_memory is None:
            return
        try:
            await self.working_memory.write(
                event_id,
                "graph_output",
                output.model_dump(mode="json"),
            )
        except Exception:
            logger.warning("GraphAgent WM write failed event=%s", event_id, exc_info=True)

    async def _mark_degraded(self, event_id: str, *, reason: str) -> None:
        """Best-effort degraded marker for graph build/persist failures."""
        self.last_degraded_reason = reason
        if self.working_memory is None:
            return
        try:
            await self.working_memory.write(
                event_id,
                "graph_degraded",
                {
                    "degraded": True,
                    "reason": reason,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except ShadowTraceError:
            logger.exception(
                "Failed to persist graph_degraded flag for event=%s",
                event_id,
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_degraded() -> GraphOutput:
        return GraphOutput(
            nodes=[],
            edges=[],
            central_entities=[],
            attack_path_candidates=[],
        )


# ====================================================================== #
# Centrality
# ====================================================================== #


def _compute_central_entities(
    nodes: list[Any],
    edges: list[Any],
    top_n: int = 3,
) -> list[str]:
    """Return the top-N entity_values ranked by degree (undirected).

    ``entity_value`` is used as the label so results read naturally
    (e.g. "zhangsan", "PC-FIN-023").
    """
    degree: dict[str, int] = defaultdict(int)
    node_value_by_id: dict[str, str] = {}

    for n in nodes:
        node_value_by_id[n.node_id] = n.entity_value

    for e in edges:
        src = e.source_node_id
        tgt = e.target_node_id
        src_label = node_value_by_id.get(src, src)
        tgt_label = node_value_by_id.get(tgt, tgt)
        degree[src_label] += 1
        degree[tgt_label] += 1

    # Fallback: include any node not incident to any edge
    for n in nodes:
        label = n.entity_value
        if label not in degree:
            degree[label] = 0

    ranked = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
    return [label for label, _ in ranked[:top_n]]


# ====================================================================== #
# Attack-path discovery
# ====================================================================== #


def _find_attack_paths(
    nodes: list[Any],
    edges: list[Any],
    max_depth: int = MAX_PATH_DEPTH,
    max_paths: int = MAX_ATTACK_PATHS,
) -> list[list[str]]:
    """Discover time-monotonic attack-path candidates via depth-limited DFS.

    Returns up to *max_paths* chains of ``node_id`` values.  A chain is
    considered valid only when its edges have monotonically non-decreasing
    ``occurred_at`` timestamps.
    """
    if not edges:
        return []

    # Build adjacency list: source → [(target, edge)]
    adj: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for e in edges:
        adj[e.source_node_id].append((e.target_node_id, e))

    # Sort outgoing edges by timestamp
    for src in adj:
        adj[src].sort(key=lambda item: _ts_or_min(item[1].occurred_at))

    paths: list[list[str]] = []

    # Start from every node
    for node in nodes:
        for path in _dfs_chain(node.node_id, adj, [], max_depth):
            if len(path) >= 2:  # at least one edge
                paths.append(path)
            if len(paths) >= max_paths * 3:  # collect extra then filter
                break

    # Deduplicate by canonical string representation; pick longest then earliest
    seen: set[str] = set()
    unique: list[list[str]] = []
    for p in sorted(paths, key=lambda p: (-len(p), str(p))):
        key = "|".join(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique[:max_paths]


def _dfs_chain(
    current: str,
    adj: dict[str, list[tuple[str, Any]]],
    visited: list[str],
    max_depth: int,
    last_ts: datetime | None = None,
) -> list[list[str]]:
    """Depth-limited DFS that enforces time monotonicity."""
    results: list[list[str]] = []

    if len(visited) >= max_depth:
        return results

    new_visited = visited + [current]
    results.append(list(new_visited))

    for neighbor, edge in adj.get(current, []):
        if neighbor in new_visited:
            continue
        edge_ts = edge.occurred_at
        if last_ts is not None and edge_ts is not None and edge_ts < last_ts:
            continue
        results.extend(_dfs_chain(neighbor, adj, new_visited, max_depth, edge_ts or last_ts))

    return results


def _ts_or_min(ts: datetime | None) -> datetime:
    return ts if ts is not None else datetime.min.replace(tzinfo=UTC)
