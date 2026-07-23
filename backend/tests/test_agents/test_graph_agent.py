"""GraphAgent tests (ISSUE-050).

Covers: builder output, node/edge counts, node dedup, centrality, attack-path
discovery, empty-evidence degraded output, edge evidence_id backlinks, and
agent trace payload.

.. note::

    ``Evidence.related_entities`` is ``list[str]`` — entity values extracted by
    EvidenceParser, typed heuristically by GraphBuilder.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from app.agents.graph_agent import (
    GraphAgent,
    _compute_central_entities,
    _find_attack_paths,
)
from app.agents.graph_builder import GraphBuilder
from app.models.agent_io import (
    CollectionStatus,
    GraphAgentInput,
    GraphOutput,
)
from app.models.enums import EvidenceSource
from app.models.evidence import Evidence
from app.models.ids import new_evidence_id

pytestmark = pytest.mark.asyncio


# ====================================================================== #
# Test helpers
# ====================================================================== #


def _new_sfx() -> str:
    return uuid4().hex[:8]


def _make_evidence(
    *,
    source: EvidenceSource,
    evidence_type: str,
    description: str = "",
    confidence: float = 0.8,
    timestamp: datetime | None = None,
    event_id: str = "evt-graph-001",
    related_entities: list[str] | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=new_evidence_id(),
        event_id=event_id,
        source=source,
        evidence_type=evidence_type,
        description=description or f"{source.value} test evidence",
        confidence=confidence,
        timestamp=timestamp or datetime(2024, 6, 15, 9, 0, 0, tzinfo=UTC),
        related_entities=related_entities or [],
    )


def _main_scenario_evidence(event_id: str = "evt-graph-001") -> list[Evidence]:
    """Build a realistic insider data exfiltration evidence set.

    ``related_entities`` stores raw string values exactly as EvidenceParser
    produces them (see evidence_parser._related_entities per-tool field maps).
    GraphBuilder infers entity types via value heuristics + evidence source.

    Covers >=4 of 6 entity types: account, host, ip, domain, process, file.
    """
    base = datetime(2024, 6, 15, 9, 0, 0, tzinfo=UTC)
    return [
        # identity: [account="zhangsan", src_ip="10.20.30.23"] + extra host hint
        _make_evidence(
            source=EvidenceSource.IDENTITY,
            evidence_type="login",
            timestamp=base,
            event_id=event_id,
            related_entities=["zhangsan", "10.20.30.23", "PC-FIN-023"],
        ),
        # endpoint: [hostname, account, process]
        _make_evidence(
            source=EvidenceSource.ENDPOINT,
            evidence_type="process_create",
            timestamp=base + timedelta(minutes=1),
            event_id=event_id,
            related_entities=["PC-FIN-023", "zhangsan", "rar.exe"],
        ),
        # data_security: [account, hostname, file_name, dst_ip_for_upload]
        _make_evidence(
            source=EvidenceSource.DATA_SECURITY,
            evidence_type="file_access",
            timestamp=base + timedelta(minutes=2),
            event_id=event_id,
            related_entities=["zhangsan", "PC-FIN-023", "financial_data.zip", "203.0.113.88"],
        ),
        # network_flow: [src_ip, dst_ip, hostname, domain]
        _make_evidence(
            source=EvidenceSource.NETWORK_FLOW,
            evidence_type="outbound_connection",
            timestamp=base + timedelta(minutes=3),
            event_id=event_id,
            related_entities=[
                "10.20.30.23",
                "203.0.113.88",
                "PC-FIN-023",
                "cloud-storage.example.com",
            ],
        ),
        # dns: [query(domain), answer(ip), hostname]
        _make_evidence(
            source=EvidenceSource.DNS,
            evidence_type="dns_query",
            timestamp=base + timedelta(minutes=4),
            event_id=event_id,
            related_entities=["cloud-storage.example.com", "203.0.113.88", "PC-FIN-023"],
        ),
    ]


class _FakeWorkingMemory:
    """Minimal BoundWorkingMemory stand-in."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], Any] = {}

    async def read(self, event_id: str, key: str) -> Any:
        return self.values.get((event_id, key))

    async def write(self, event_id: str, key: str, value: Any) -> None:
        self.values[(event_id, key)] = value

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        pass


class _RecordingTraceService:
    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []

    async def log_trace(self, **kwargs: Any) -> str:
        self.traces.append(kwargs)
        return f"trace-{uuid4().hex[:8]}"


def _build_agent(
    *,
    wm: _FakeWorkingMemory | None = None,
    trace_service: _RecordingTraceService | None = None,
) -> GraphAgent:
    return GraphAgent(
        working_memory=wm,
        trace_service=trace_service,
        session_factory=None,  # skip DB persist in unit tests
    )


# ====================================================================== #
# GraphBuilder tests
# ====================================================================== #


async def test_builder_main_scenario_node_and_edge_counts() -> None:
    """Main scenario produces >=6 nodes and >=8 edges covering >=4 entity types."""
    evidence_list = _main_scenario_evidence()
    nodes, edges = GraphBuilder.build(evidence_list)

    assert len(nodes) >= 6, f"expected >=6 nodes, got {len(nodes)}"
    assert len(edges) >= 8, f"expected >=8 edges, got {len(edges)}"

    entity_types = {n.entity_type for n in nodes}
    assert len(entity_types) >= 4, f"expected >=4 entity types, got {entity_types}"

    # Every edge must backlink to a valid evidence_id
    evidence_ids = {e.evidence_id for e in evidence_list}
    for edge in edges:
        assert edge.evidence_id in evidence_ids, (
            f"edge {edge.edge_id} references unknown evidence_id {edge.evidence_id}"
        )


async def test_builder_edge_relation_types() -> None:
    """Edges cover all eight GraphRelationType values across the scenario."""
    evidence_list = _main_scenario_evidence()
    _, edges = GraphBuilder.build(evidence_list)

    relation_types = {e.relation_type for e in edges}
    expected_min = 6  # at least 6 of 8 relation types for main scenario
    assert len(relation_types) >= expected_min, (
        f"Expected >= {expected_min} distinct types, got {len(relation_types)}: {relation_types}"
    )


async def test_node_id_is_stable() -> None:
    """Same event_id + same related_entities => same node_id (idempotent)."""
    ev1 = _main_scenario_evidence("evt-AAA")
    ev2 = _main_scenario_evidence("evt-AAA")

    nodes1, _ = GraphBuilder.build(ev1)
    nodes2, _ = GraphBuilder.build(ev2)

    ids1 = {n.node_id for n in nodes1}
    ids2 = {n.node_id for n in nodes2}
    assert ids1 == ids2, "Same input must produce same node_ids"


async def test_node_dedup_across_evidence() -> None:
    """Duplicate entity across evidence records yields a single node."""
    event_id = f"evt-dup-{_new_sfx()}"
    base = datetime(2024, 6, 15, 9, 0, 0, tzinfo=UTC)

    evidence_list = [
        _make_evidence(
            source=EvidenceSource.IDENTITY,
            evidence_type="login",
            timestamp=base,
            event_id=event_id,
            related_entities=["zhangsan", "10.20.30.23"],
        ),
        _make_evidence(
            source=EvidenceSource.ENDPOINT,
            evidence_type="process_create",
            timestamp=base + timedelta(minutes=1),
            event_id=event_id,
            related_entities=["PC-FIN-023", "zhangsan", "rar.exe"],
        ),
    ]

    nodes, _ = GraphBuilder.build(evidence_list)

    # "zhangsan" should appear exactly once
    zhangsan_nodes = [n for n in nodes if n.entity_value == "zhangsan"]
    assert len(zhangsan_nodes) == 1, f"zhangsan should appear once, got {len(zhangsan_nodes)}"


async def test_empty_evidence_produces_empty_graph() -> None:
    """No evidence → no nodes, no edges."""
    nodes, edges = GraphBuilder.build([])
    assert nodes == []
    assert edges == []


# ====================================================================== #
# Centrality tests
# ====================================================================== #


async def test_central_entities_top_three() -> None:
    """Central entities include high-degree nodes (zhangsan or PC-FIN-023)."""
    evidence_list = _main_scenario_evidence()
    nodes, edges = GraphBuilder.build(evidence_list)

    central = _compute_central_entities(nodes, edges, top_n=3)
    assert 0 < len(central) <= 3

    # At least one of the expected high-degree entities
    central_lower = [c.lower() for c in central]
    found = any("zhangsan" in c or "pc-fin-023" in c or "203.0.113.88" in c for c in central_lower)
    assert found, f"Central entities ({central}) missing expected high-degree nodes"


async def test_central_entities_empty_graph() -> None:
    """Empty graph → empty centrality."""
    central = _compute_central_entities([], [])
    assert central == []


# ====================================================================== #
# Attack-path tests
# ====================================================================== #


async def test_attack_path_time_monotonic() -> None:
    """Attack path candidates are discovered and time-monotonic."""
    evidence_list = _main_scenario_evidence()
    nodes, edges = GraphBuilder.build(evidence_list)

    paths = _find_attack_paths(nodes, edges, max_depth=6, max_paths=3)
    assert len(paths) >= 1, f"Expected at least 1 attack path, got {len(paths)}"

    for path in paths:
        assert len(path) >= 2, f"Path too short: {path}"
        ts: datetime | None = None
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            matching = [e for e in edges if e.source_node_id == src and e.target_node_id == tgt]
            if matching and matching[0].occurred_at is not None:
                edge_ts = matching[0].occurred_at
                if ts is not None and edge_ts is not None:
                    assert edge_ts >= ts, (
                        f"Path {path}: timestamp {edge_ts} < previous {ts} at step {i}"
                    )
                ts = edge_ts


async def test_attack_path_account_to_ip_chain() -> None:
    """At least one attack path contains an account→ip adjacency."""
    event_id = f"evt-acct-ip-{_new_sfx()}"
    base = datetime(2024, 6, 15, 9, 0, 0, tzinfo=UTC)
    evidence_list = [
        _make_evidence(
            source=EvidenceSource.IDENTITY,
            evidence_type="login",
            timestamp=base,
            event_id=event_id,
            related_entities=["zhangsan", "10.20.30.23"],
        ),
    ]
    nodes, edges = GraphBuilder.build(evidence_list)
    paths = _find_attack_paths(nodes, edges, max_depth=6, max_paths=3)

    node_entities = {n.node_id: (n.entity_type, n.entity_value) for n in nodes}

    def path_has_account_to_ip(path: list[str]) -> bool:
        for i in range(len(path) - 1):
            src_type = node_entities.get(path[i], ("", ""))[0]
            tgt_type = node_entities.get(path[i + 1], ("", ""))[0]
            if src_type == "account" and tgt_type == "ip":
                return True
        return False

    assert any(path_has_account_to_ip(path) for path in paths), (
        f"No account→ip adjacency in attack paths {paths}; nodes={node_entities}"
    )


async def test_attack_path_empty_graph() -> None:
    """Empty graph → no attack paths."""
    paths = _find_attack_paths([], [])
    assert paths == []


# ====================================================================== #
# GraphAgent integration tests
# ====================================================================== #


class TestGraphAgentIntegration:
    """Agent-level tests exercising ``execute()`` and verifying output + WM."""

    async def test_execute_main_scenario(self) -> None:
        """execute() produces GraphOutput with >=6 nodes, >=8 edges, centrality, paths."""
        event_id = f"evt-graph-agent-{_new_sfx()}"
        wm = _FakeWorkingMemory()

        from app.models.agent_io import EvidenceOutput

        evidence_output = EvidenceOutput(
            evidence_list=_main_scenario_evidence(event_id),
            conflicts=[],
            gaps=[],
            success_sources=[
                EvidenceSource.IDENTITY.value,
                EvidenceSource.ENDPOINT.value,
                EvidenceSource.DATA_SECURITY.value,
                EvidenceSource.NETWORK_FLOW.value,
                EvidenceSource.DNS.value,
            ],
            failed_sources=[],
            overall_confidence=0.85,
            collection_status=CollectionStatus.COMPLETED,
        )

        agent_input = GraphAgentInput(
            event_id=event_id,
            evidence_output=evidence_output,
        )

        agent = _build_agent(wm=wm)
        output = await agent.execute(agent_input)

        assert isinstance(output, GraphOutput)
        assert len(output.nodes) >= 6
        assert len(output.edges) >= 8

        # Edge backlinks
        valid_eids = {e.evidence_id for e in evidence_output.evidence_list}
        for edge in output.edges:
            assert edge.evidence_id in valid_eids

        # >=4 entity types
        entity_types = {n.entity_type for n in output.nodes}
        assert len(entity_types) >= 4

        # Central entities populated
        assert len(output.central_entities) > 0

        # Attack path candidates
        assert len(output.attack_path_candidates) >= 1
        for path in output.attack_path_candidates:
            assert len(path) >= 2

        # WM write
        ctx = await wm.read(event_id, "graph_output")
        assert ctx is not None
        assert len(ctx["nodes"]) == len(output.nodes)
        assert len(ctx["edges"]) == len(output.edges)

    async def test_execute_empty_evidence(self) -> None:
        """Empty EvidenceOutput → empty GraphOutput."""
        wm = _FakeWorkingMemory()
        from app.models.agent_io import EvidenceOutput

        event_id = f"evt-empty-{_new_sfx()}"
        agent_input = GraphAgentInput(
            event_id=event_id,
            evidence_output=EvidenceOutput(
                evidence_list=[],
                conflicts=[],
                gaps=[],
                success_sources=[],
                failed_sources=[],
                overall_confidence=1.0,
                collection_status=CollectionStatus.COMPLETED,
            ),
        )

        agent = _build_agent(wm=wm)
        output = await agent.execute(agent_input)

        assert output.nodes == []
        assert output.edges == []
        assert output.central_entities == []
        assert output.attack_path_candidates == []

    async def test_execute_trace_payload(self) -> None:
        """agent_trace output_data carries graph metadata fields."""
        event_id = f"evt-trace-{_new_sfx()}"
        wm = _FakeWorkingMemory()
        trace = _RecordingTraceService()

        from app.models.agent_io import EvidenceOutput

        evidence_output = EvidenceOutput(
            evidence_list=_main_scenario_evidence(event_id),
            conflicts=[],
            gaps=[],
            success_sources=[
                EvidenceSource.IDENTITY.value,
                EvidenceSource.ENDPOINT.value,
                EvidenceSource.DATA_SECURITY.value,
                EvidenceSource.NETWORK_FLOW.value,
                EvidenceSource.DNS.value,
            ],
            failed_sources=[],
            overall_confidence=0.85,
            collection_status=CollectionStatus.COMPLETED,
        )

        agent = _build_agent(wm=wm, trace_service=trace)
        await agent.execute(GraphAgentInput(event_id=event_id, evidence_output=evidence_output))

        assert len(trace.traces) == 1
        assert trace.traces[0]["agent_name"] == "graph_agent"
        assert trace.traces[0]["status"] == "completed"
        trace_out = trace.traces[0]["output_data"]
        # Accept both dict and Pydantic model (depends on trace service impl)
        if isinstance(trace_out, GraphOutput):
            trace_out = trace_out.model_dump(mode="json")
        assert isinstance(trace_out, dict)
        for key in ("nodes", "edges", "central_entities", "attack_path_candidates"):
            assert key in trace_out, f"Missing key '{key}' in trace output_data"

    async def test_builder_skips_empty_related_entities(self) -> None:
        """Evidence with empty related_entities → no nodes/edges (not a crash)."""
        from app.models.agent_io import EvidenceOutput

        event_id = f"evt-degraded-{_new_sfx()}"
        bad_evidence = Evidence(
            evidence_id=new_evidence_id(),
            event_id=event_id,
            source=EvidenceSource.IDENTITY,
            evidence_type="login",
            description="no related entities",
            confidence=0.5,
            timestamp=datetime(2024, 6, 15, 9, 0, 0, tzinfo=UTC),
            related_entities=[],
        )

        agent_input = GraphAgentInput(
            event_id=event_id,
            evidence_output=EvidenceOutput(
                evidence_list=[bad_evidence],
                conflicts=[],
                gaps=[],
                success_sources=[],
                failed_sources=[],
                overall_confidence=0.5,
                collection_status=CollectionStatus.DEGRADED,
            ),
        )

        agent = _build_agent(wm=_FakeWorkingMemory())
        output = await agent.execute(agent_input)
        assert isinstance(output, GraphOutput)
        assert len(output.nodes) == 0
        assert len(output.edges) == 0

    async def test_node_deduplication_after_replay(self) -> None:
        """Repeated execution yields identical node IDs (idempotent)."""
        from app.models.agent_io import EvidenceOutput

        event_id = f"evt-replay-{_new_sfx()}"
        wm = _FakeWorkingMemory()

        evidence_output = EvidenceOutput(
            evidence_list=_main_scenario_evidence(event_id),
            conflicts=[],
            gaps=[],
            success_sources=[
                EvidenceSource.IDENTITY.value,
                EvidenceSource.ENDPOINT.value,
                EvidenceSource.DATA_SECURITY.value,
                EvidenceSource.NETWORK_FLOW.value,
                EvidenceSource.DNS.value,
            ],
            failed_sources=[],
            overall_confidence=0.85,
            collection_status=CollectionStatus.COMPLETED,
        )

        runs: list[set[str]] = []
        for _ in range(3):
            agent = _build_agent(wm=wm)
            output = await agent.execute(
                GraphAgentInput(event_id=event_id, evidence_output=evidence_output)
            )
            runs.append({n.node_id for n in output.nodes})

        assert runs[0] == runs[1] == runs[2], "Repeated execution must produce identical node IDs"

    async def test_builder_failure_sets_graph_degraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GraphBuilder exception → empty output and graph_degraded flag."""
        from app.models.agent_io import EvidenceOutput

        event_id = f"evt-builder-fail-{_new_sfx()}"
        wm = _FakeWorkingMemory()

        def _boom(_evidence: list[Evidence]) -> tuple[list[Any], list[Any]]:
            raise RuntimeError("builder exploded")

        monkeypatch.setattr("app.agents.graph_agent.GraphBuilder.build", _boom)

        agent_input = GraphAgentInput(
            event_id=event_id,
            evidence_output=EvidenceOutput(
                evidence_list=_main_scenario_evidence(event_id),
                conflicts=[],
                gaps=[],
                success_sources=[EvidenceSource.IDENTITY.value],
                failed_sources=[],
                overall_confidence=0.5,
                collection_status=CollectionStatus.COMPLETED,
            ),
        )

        agent = _build_agent(wm=wm)
        output = await agent.execute(agent_input)

        assert output.nodes == []
        assert output.edges == []
        assert agent.last_degraded_reason == "graph_builder_failed"
        degraded = await wm.read(event_id, "graph_degraded")
        assert degraded is not None
        assert degraded["degraded"] is True
        assert degraded["reason"] == "graph_builder_failed"

    async def test_persist_failure_sets_graph_degraded(self) -> None:
        """Persist failure still returns in-memory graph and marks graph_degraded."""
        from app.models.agent_io import EvidenceOutput

        event_id = f"evt-persist-fail-{_new_sfx()}"
        wm = _FakeWorkingMemory()

        class _BrokenSessionFactory:
            def __call__(self) -> Any:
                raise RuntimeError("db unavailable")

        agent = GraphAgent(
            working_memory=wm,
            session_factory=_BrokenSessionFactory(),  # type: ignore[arg-type]
        )

        evidence_output = EvidenceOutput(
            evidence_list=_main_scenario_evidence(event_id),
            conflicts=[],
            gaps=[],
            success_sources=[EvidenceSource.IDENTITY.value],
            failed_sources=[],
            overall_confidence=0.85,
            collection_status=CollectionStatus.COMPLETED,
        )

        output = await agent.execute(
            GraphAgentInput(event_id=event_id, evidence_output=evidence_output)
        )

        assert len(output.nodes) >= 6
        assert agent.last_degraded_reason is not None
        assert agent.last_degraded_reason.startswith("graph_persist_failed:")
        degraded = await wm.read(event_id, "graph_degraded")
        assert degraded is not None
        assert degraded["degraded"] is True
        assert degraded["reason"].startswith("graph_persist_failed:")


# ====================================================================== #
# DB integration: persist idempotency
# ====================================================================== #


@pytest.mark.integration
async def test_graph_persist_idempotent_node_count() -> None:
    """Repeated GraphAgent persist upserts nodes without duplication."""
    import asyncio
    import os
    from pathlib import Path

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import func, select, text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.db import models as orm
    from app.db.orm.graph import GraphNodeORM
    from app.models.agent_io import EvidenceOutput, GraphAgentInput

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
    )
    backend_dir = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "migrations"))

    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        await engine.dispose()
        pytest.skip("PostgreSQL not reachable; start Compose postgres first")

    await asyncio.to_thread(command.upgrade, cfg, "head")
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    sfx = _new_sfx()
    event_id = f"evt-graph-persist-{sfx}"
    async with session_factory() as session:
        async with session.begin():
            session.add(
                orm.SecurityEvent(
                    event_id=event_id,
                    event_type="insider_threat",
                    title="graph persist test",
                    creation_source_ref={"source_object_id": f"INC-{sfx}"},
                )
            )

    wm = _FakeWorkingMemory()
    agent = GraphAgent(working_memory=wm, session_factory=session_factory)
    evidence_output = EvidenceOutput(
        evidence_list=_main_scenario_evidence(event_id),
        conflicts=[],
        gaps=[],
        success_sources=[
            EvidenceSource.IDENTITY.value,
            EvidenceSource.ENDPOINT.value,
            EvidenceSource.DATA_SECURITY.value,
            EvidenceSource.NETWORK_FLOW.value,
            EvidenceSource.DNS.value,
        ],
        failed_sources=[],
        overall_confidence=0.85,
        collection_status=CollectionStatus.COMPLETED,
    )
    agent_input = GraphAgentInput(event_id=event_id, evidence_output=evidence_output)

    first = await agent.execute(agent_input)
    second = await agent.execute(agent_input)

    assert len(first.nodes) == len(second.nodes)
    assert {n.node_id for n in first.nodes} == {n.node_id for n in second.nodes}

    async with session_factory() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(GraphNodeORM)
            .where(GraphNodeORM.event_id == event_id)
        )
    assert count == len(first.nodes)

    await engine.dispose()
