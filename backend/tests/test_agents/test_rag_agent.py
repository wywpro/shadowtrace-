"""Tests for RAGAgent, RAGQueryBuilder, and result assembly helpers (ISSUE-046)."""

from __future__ import annotations

import pydantic
import pytest

from app.agents.rag_agent import (
    RAGAgent,
    _aggregate_citations,
    _build_attack_techniques,
    _build_fp_similarity,
    _build_playbook_refs,
    _build_similar_cases,
)
from app.agents.rag_query_builder import RAGQueryBuilder
from app.core.errors import (
    DependencyUnavailableError,
    GuardrailViolationError,
    ShadowTraceError,
)
from app.models.agent_io import (
    CollectionStatus,
    EvidenceOutput,
    FpSimilarity,
    RAGAgentInput,
    RAGOutput,
    TriageResult,
)
from app.models.entities import EntitySet, HostEntity, IPEntity, ProcessEntity
from app.models.enums import EventType, EvidenceSource, Severity
from app.models.evidence import Evidence
from app.models.knowledge import RetrievalResult, RetrievedChunk
from app.models.workflow import FP_LOW_THRESHOLD

# --------------------------------------------------------------------------- #
# Mock helpers
# --------------------------------------------------------------------------- #


class _MockBoundWorkingMemory:
    """Minimal mock matching BoundWorkingMemory interface."""

    def __init__(self, writer_name: str = "RAGAgent") -> None:
        self.writer_name = writer_name
        self._store: dict[str, object] = {}
        self._memory = self

    def for_writer(self, writer: str) -> _MockBoundWorkingMemory:
        from app.services.working_memory import normalize_writer

        return _MockBoundWorkingMemory(writer_name=normalize_writer(writer))

    async def read(self, event_id: str, key: str) -> object:
        return self._store.get(key)

    async def write(self, event_id: str, key: str, value: object) -> None:
        self._store[key] = value

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        pass

    async def read_scratchpad(self, event_id: str) -> list:
        return []


class _FailingWriteMockWM:
    """Mock WM that raises on write for a specific key."""

    def __init__(
        self,
        writer_name: str = "RAGAgent",
        *,
        fail_key: str | None = None,
        fail_error: Exception | None = None,
    ) -> None:
        self.writer_name = writer_name
        self._store: dict[str, object] = {}
        self._fail_key = fail_key
        self._fail_error = fail_error or DependencyUnavailableError("wm unavailable")
        self._memory = self

    def for_writer(self, writer: str) -> _FailingWriteMockWM:
        from app.services.working_memory import normalize_writer

        return _FailingWriteMockWM(
            writer_name=normalize_writer(writer),
            fail_key=self._fail_key,
            fail_error=self._fail_error,
        )

    async def read(self, event_id: str, key: str) -> object:
        return self._store.get(key)

    async def write(self, event_id: str, key: str, value: object) -> None:
        if self._fail_key is not None and key == self._fail_key:
            raise self._fail_error
        self._store[key] = value

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        pass

    async def read_scratchpad(self, event_id: str) -> list:
        return []


class _MockPipeline:
    """Configurable mock pipeline whose retrieve() returns preset results per KB."""

    def __init__(
        self,
        results: dict[str, RetrievalResult | Exception] | None = None,
    ) -> None:
        self._results = results or {}
        self.calls: list[dict] = []

    async def retrieve(self, query: str, kb_names: list[str], top_k: int = 5) -> RetrievalResult:
        self.calls.append({"query": query, "kb_names": kb_names, "top_k": top_k})
        kb_name = kb_names[0] if kb_names else "unknown"
        if kb_name in self._results:
            item = self._results[kb_name]
            if isinstance(item, Exception):
                raise item
            return item
        return RetrievalResult(query=query)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def _make_triage_result(
    event_type: EventType = EventType.DATA_EXFILTRATION,
    severity: Severity = Severity.HIGH,
) -> TriageResult:
    entities = EntitySet(
        ips=[
            IPEntity(entity_id="ip-1", entity_type="ip", address="45.153.12.88", scope="external"),
            IPEntity(entity_id="ip-2", entity_type="ip", address="10.0.0.5", scope="internal"),
        ],
        hosts=[
            HostEntity(entity_id="host-1", entity_type="host", hostname="web-server-01"),
        ],
        processes=[
            ProcessEntity(entity_id="proc-1", entity_type="process", name="curl.exe"),
        ],
    )
    return TriageResult(
        event_type=event_type,
        severity=severity,
        need_investigation=True,
        entities=entities,
        ioc_list=["45.153.12.88", "malware-c2.example.com"],
        reasoning=(
            "Data exfiltration detected: 500MB upload to external IP 45.153.12.88 via curl.exe"
        ),
    )


def _make_input(
    event_id: str = "evt-001",
    event_type: EventType = EventType.DATA_EXFILTRATION,
    severity: Severity = Severity.HIGH,
    evidence_output: EvidenceOutput | None = None,
) -> RAGAgentInput:
    return RAGAgentInput(
        event_id=event_id,
        triage_result=_make_triage_result(event_type=event_type, severity=severity),
        evidence_output=evidence_output,
    )


def _make_chunk(
    chunk_id: str,
    kb_name: str,
    content: str,
    score: float = 0.85,
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        kb_name=kb_name,
        content=content,
        score=score,
        retrieval_method="reranked",
        metadata=metadata or {},
    )


def _make_knowledge_citation(
    citation_id: str,
    chunk_id: str,
    kb_name: str,
    quoted_text: str = "relevant excerpt",
    relevance_score: float = 0.85,
):
    """Create a knowledge.Citation (with the pattern constraint on citation_id)."""
    from app.models.knowledge import Citation as KnowledgeCitation

    return KnowledgeCitation(
        citation_id=citation_id,
        chunk_id=chunk_id,
        kb_name=kb_name,
        quoted_text=quoted_text,
        relevance_score=relevance_score,
    )


# --------------------------------------------------------------------------- #
# Attack KB test data
# --------------------------------------------------------------------------- #

_ATTACK_CHUNKS = [
    _make_chunk(
        "atk-001",
        "attack_kb",
        "Technique: Exfiltration Over Web Service\nID: T1567\nTactics: exfiltration\n...",
        score=0.92,
        metadata={
            "technique_id": "T1567",
            "technique_name": "Exfiltration Over Web Service",
            "tactics": ["exfiltration"],
            "description": "Adversaries may exfiltrate data over web services.",
            "detection": "Monitor for large outbound transfers.",
        },
    ),
    _make_chunk(
        "atk-002",
        "attack_kb",
        "Technique: Exfiltration Over Alternative Protocol\nID: T1048\nTactics: exfiltration\n...",
        score=0.78,
        metadata={
            "technique_id": "T1048",
            "technique_name": "Exfiltration Over Alternative Protocol",
            "tactics": ["exfiltration"],
            "description": "Adversaries may steal data by exfiltrating it over a different "
            "protocol.",
            "detection": "Monitor for unusual protocol usage.",
        },
    ),
    _make_chunk(
        "atk-003",
        "attack_kb",
        "Technique: Data Transfer Size Limits\nID: T1030\nTactics: exfiltration\n...",
        score=0.25,
        metadata={
            "technique_id": "T1030",
            "technique_name": "Data Transfer Size Limits",
            "tactics": ["exfiltration"],
        },
    ),
]

_ATTACK_CITATIONS = [
    _make_knowledge_citation(
        "cit-a1b2c3d4",
        "atk-001",
        "attack_kb",
        "exfiltrate data over web services",
        0.92,
    ),
    _make_knowledge_citation(
        "cit-e5f6a7b8",
        "atk-002",
        "attack_kb",
        "exfiltrating it over a different protocol",
        0.78,
    ),
]

_FP_CHUNKS = [
    _make_chunk(
        "fp-001",
        "fp_case_kb",
        "Pattern: Ops change window bulk login | ops_change_window_bulk_login | ...",
        score=0.88,
        metadata={
            "case_id": "case-fp00001",
            "pattern_summary": "Bulk login during scheduled ops change window",
            "alert_signature": "ops_change_window_bulk_login",
            "entity_pattern": "svc-* accounts",
            "fp_reason": "Scheduled maintenance window activity",
            "confirmed_by": "soc-analyst-1",
            "confirmed_at": "2026-01-15T10:00:00Z",
        },
    ),
]

_FP_CITATIONS = [
    _make_knowledge_citation(
        "cit-f0000001",
        "fp-001",
        "fp_case_kb",
        "ops change window bulk login",
        0.88,
    ),
]

_HISTORY_CHUNKS = [
    _make_chunk(
        "hist-001",
        "history_case_kb",
        "Case: Data exfiltration via Dropbox | key entities: 10.0.0.5; 45.153.12.88",
        score=0.82,
        metadata={
            "case_id": "case-h00001",
            "event_id": "evt-00001",
            "event_type": "data_exfiltration",
            "case_label": "true_positive",
            "summary": "Attacker exfiltrated 2GB of customer PII via Dropbox API.",
            "key_entities": "10.0.0.5; 45.153.12.88",
            "final_verdict": "true_positive",
            "risk_score": 85,
            "resolution": "Contained, endpoint isolated, credentials rotated.",
            "closed_at": "2026-02-20T08:00:00Z",
        },
    ),
    _make_chunk(
        "hist-002",
        "history_case_kb",
        "Case: FTP exfiltration after hours | key entities: 192.168.1.100; ftp.example.com",
        score=0.65,
        metadata={
            "case_id": "case-h00002",
            "event_id": "evt-00002",
            "event_type": "data_exfiltration",
            "case_label": "true_positive",
            "summary": "Sensitive documents uploaded to external FTP server.",
            "key_entities": "192.168.1.100; ftp.example.com",
            "final_verdict": "true_positive",
            "risk_score": 70,
            "resolution": "Firewall rule added, FTP blocked.",
            "closed_at": "2026-03-01T12:00:00Z",
        },
    ),
]

_HISTORY_CITATIONS = [
    _make_knowledge_citation(
        "cit-0a000001",
        "hist-001",
        "history_case_kb",
        "exfiltrated 2GB of customer PII",
        0.82,
    ),
    _make_knowledge_citation(
        "cit-0a000002",
        "hist-002",
        "history_case_kb",
        "uploaded to external FTP server",
        0.65,
    ),
]

_PLAYBOOK_CHUNKS = [
    _make_chunk(
        "pbk-001",
        "playbook_kb",
        "Playbook: Data Exfiltration Response\n"
        "Event Type: data_exfiltration\nMin Severity: high\n...",
        score=0.91,
        metadata={
            "playbook_id": "pb-exfil-001",
            "playbook_name": "Data Exfiltration Response",
            "event_type": "data_exfiltration",
            "min_severity": "high",
            "description": "Isolate affected hosts, block external IPs, initiate DLP scan.",
            "steps": [],
        },
    ),
    _make_chunk(
        "pbk-002",
        "playbook_kb",
        "Playbook: Generic Data Protection\n"
        "Event Type: data_exfiltration\nMin Severity: medium\n...",
        score=0.73,
        metadata={
            "playbook_id": "pb-data-prot-001",
            "playbook_name": "Generic Data Protection",
            "event_type": "data_exfiltration",
            "min_severity": "medium",
            "description": "Audit data access, review DLP policies.",
            "steps": [],
        },
    ),
]

_PLAYBOOK_CITATIONS = [
    _make_knowledge_citation(
        "cit-0b000001",
        "pbk-001",
        "playbook_kb",
        "Isolate affected hosts",
        0.91,
    ),
    _make_knowledge_citation(
        "cit-0b000002",
        "pbk-002",
        "playbook_kb",
        "Audit data access",
        0.73,
    ),
]


def _make_full_results() -> dict[str, RetrievalResult]:
    return {
        "attack_kb": RetrievalResult(
            query="",
            chunks=_ATTACK_CHUNKS,
            citations=_ATTACK_CITATIONS,
        ),
        "fp_case_kb": RetrievalResult(
            query="",
            chunks=_FP_CHUNKS,
            citations=_FP_CITATIONS,
        ),
        "history_case_kb": RetrievalResult(
            query="",
            chunks=_HISTORY_CHUNKS,
            citations=_HISTORY_CITATIONS,
        ),
        "playbook_kb": RetrievalResult(
            query="",
            chunks=_PLAYBOOK_CHUNKS,
            citations=_PLAYBOOK_CITATIONS,
        ),
    }


# --------------------------------------------------------------------------- #
# Tests: RAGQueryBuilder
# --------------------------------------------------------------------------- #


class TestRAGQueryBuilder:
    def test_builds_four_queries(self):
        triage = _make_triage_result()
        queries = RAGQueryBuilder.build_queries(triage)
        assert set(queries.keys()) == {
            "attack_kb",
            "fp_case_kb",
            "history_case_kb",
            "playbook_kb",
        }
        for q in queries.values():
            assert isinstance(q, str) and len(q) > 0

    def test_attack_query_includes_event_type(self):
        triage = _make_triage_result(EventType.DATA_EXFILTRATION)
        queries = RAGQueryBuilder.build_queries(triage)
        assert "data_exfiltration" in queries["attack_kb"]

    def test_attack_query_includes_evidence_behaviors(self):
        triage = _make_triage_result()
        evidence = EvidenceOutput(
            evidence_list=[
                Evidence(
                    evidence_id="ev-001",
                    event_id="evt-001",
                    source=EvidenceSource.NETWORK_FLOW,
                    evidence_type="network_connection",
                    description=(
                        "Outbound connection to rare external IP 45.153.12.88 on port 443"
                    ),
                    confidence=0.9,
                ),
            ],
            collection_status=CollectionStatus.COMPLETED,
        )
        queries = RAGQueryBuilder.build_queries(triage, evidence)
        assert "45.153.12.88" in queries["attack_kb"]

    def test_fp_query_includes_event_type_and_severity(self):
        triage = _make_triage_result(EventType.DATA_EXFILTRATION, Severity.HIGH)
        queries = RAGQueryBuilder.build_queries(triage)
        assert "data_exfiltration" in queries["fp_case_kb"].lower()
        assert "high" in queries["fp_case_kb"].lower()

    def test_history_query_includes_entities(self):
        triage = _make_triage_result()
        queries = RAGQueryBuilder.build_queries(triage)
        assert "45.153.12.88" in queries["history_case_kb"]

    def test_playbook_query_includes_event_type_and_severity(self):
        triage = _make_triage_result(EventType.DATA_EXFILTRATION, Severity.HIGH)
        queries = RAGQueryBuilder.build_queries(triage)
        assert "data_exfiltration" in queries["playbook_kb"]
        assert "high" in queries["playbook_kb"]


# --------------------------------------------------------------------------- #
# Tests: Result assembly helpers (pure functions)
# --------------------------------------------------------------------------- #


class TestBuildAttackTechniques:
    def test_extracts_techniques_above_threshold(self):
        result = RetrievalResult(
            query="",
            chunks=_ATTACK_CHUNKS,
            citations=_ATTACK_CITATIONS,
        )
        techniques = _build_attack_techniques(result)
        assert len(techniques) >= 2
        technique_ids = {t.technique_id for t in techniques}
        assert "T1567" in technique_ids or "T1048" in technique_ids

    def test_filters_below_03_threshold(self):
        result = RetrievalResult(
            query="",
            chunks=_ATTACK_CHUNKS,  # atk-003 has score 0.25
            citations=_ATTACK_CITATIONS,
        )
        techniques = _build_attack_techniques(result)
        technique_ids = {t.technique_id for t in techniques}
        assert "T1030" not in technique_ids

    def test_each_technique_has_citation_id(self):
        result = RetrievalResult(
            query="",
            chunks=_ATTACK_CHUNKS,
            citations=_ATTACK_CITATIONS,
        )
        techniques = _build_attack_techniques(result)
        for t in techniques:
            assert t.citation_id, f"Technique {t.technique_id} missing citation_id"

    def test_empty_result_returns_empty_list(self):
        assert _build_attack_techniques(None) == []
        assert _build_attack_techniques(RetrievalResult(query="")) == []

    def test_deduplicates_by_technique_id(self):
        meta = {"technique_id": "T1567", "technique_name": "X", "tactics": ["exfil"]}
        chunks = [
            _make_chunk("a-1", "attack_kb", "T1567 v1", score=0.9, metadata=meta),
            _make_chunk("a-2", "attack_kb", "T1567 v2", score=0.7, metadata=meta),
        ]
        citations = [
            _make_knowledge_citation("cit-11111111", "a-1", "attack_kb", "x", 0.9),
            _make_knowledge_citation("cit-22222222", "a-2", "attack_kb", "x", 0.7),
        ]
        result = RetrievalResult(query="", chunks=chunks, citations=citations)
        techniques = _build_attack_techniques(result)
        assert len(techniques) == 1
        assert techniques[0].match_confidence == 0.9


class TestBuildFpSimilarity:
    def test_extracts_high_score_match(self):
        result = RetrievalResult(
            query="",
            chunks=_FP_CHUNKS,
            citations=_FP_CITATIONS,
        )
        fp = _build_fp_similarity(result)
        assert fp.max_score >= FP_LOW_THRESHOLD
        assert fp.matched_case_id == "case-fp00001"
        assert fp.matched_pattern is not None

    def test_empty_result_returns_default(self):
        fp = _build_fp_similarity(None)
        assert fp.max_score == 0.0
        assert fp.matched_case_id is None
        assert fp.matched_pattern is None

    def test_empty_chunks_returns_default(self):
        fp = _build_fp_similarity(RetrievalResult(query=""))
        assert fp.max_score == 0.0

    def test_clips_score_to_0_1(self):
        chunk = _make_chunk(
            "fp-99",
            "fp_case_kb",
            "x",
            score=1.5,
            metadata={"case_id": "case-99", "pattern_summary": "test"},
        )
        result = RetrievalResult(query="", chunks=[chunk], citations=[])
        fp = _build_fp_similarity(result)
        assert 0.0 <= fp.max_score <= 1.0


class TestBuildSimilarCases:
    def test_extracts_case_summaries(self):
        result = RetrievalResult(
            query="",
            chunks=_HISTORY_CHUNKS,
            citations=_HISTORY_CITATIONS,
        )
        cases = _build_similar_cases(result)
        assert len(cases) == 2
        assert cases[0].case_id == "case-h00001"
        assert cases[0].event_type == EventType.DATA_EXFILTRATION
        assert cases[0].risk_score == 85

    def test_empty_result_returns_empty(self):
        assert _build_similar_cases(None) == []

    def test_handles_invalid_enum_values(self):
        chunk = _make_chunk(
            "hist-99",
            "history_case_kb",
            "test",
            score=0.5,
            metadata={
                "case_id": "case-99",
                "event_type": "invalid_type",
                "final_verdict": "invalid_verdict",
                "summary": "test",
                "risk_score": 50,
            },
        )
        result = RetrievalResult(query="", chunks=[chunk], citations=[])
        cases = _build_similar_cases(result)
        assert len(cases) == 1
        assert cases[0].event_type is None
        assert cases[0].final_verdict is None


class TestBuildPlaybookRefs:
    def test_extracts_playbook_ids(self):
        result = RetrievalResult(
            query="",
            chunks=_PLAYBOOK_CHUNKS,
            citations=_PLAYBOOK_CITATIONS,
        )
        refs = _build_playbook_refs(result)
        assert len(refs) == 2
        assert "pb-exfil-001" in refs
        assert "pb-data-prot-001" in refs

    def test_deduplicates_playbook_ids(self):
        chunks = [
            _make_chunk("p1", "playbook_kb", "v1", score=0.9, metadata={"playbook_id": "pb-001"}),
            _make_chunk("p2", "playbook_kb", "v2", score=0.7, metadata={"playbook_id": "pb-001"}),
        ]
        result = RetrievalResult(query="", chunks=chunks, citations=[])
        refs = _build_playbook_refs(result)
        assert refs == ["pb-001"]

    def test_empty_result_returns_empty(self):
        assert _build_playbook_refs(None) == []


class TestAggregateCitations:
    def test_aggregates_all_citations(self):
        results = {
            "attack_kb": RetrievalResult(query="", citations=_ATTACK_CITATIONS),
            "fp_case_kb": RetrievalResult(query="", citations=_FP_CITATIONS),
            "history_case_kb": RetrievalResult(query="", citations=_HISTORY_CITATIONS),
            "playbook_kb": RetrievalResult(query="", citations=_PLAYBOOK_CITATIONS),
        }
        aggregated = _aggregate_citations(results)
        expected_count = (
            len(_ATTACK_CITATIONS)
            + len(_FP_CITATIONS)
            + len(_HISTORY_CITATIONS)
            + len(_PLAYBOOK_CITATIONS)
        )
        assert len(aggregated) == expected_count

    def test_deduplicates_by_citation_id(self):
        results = {
            "attack_kb": RetrievalResult(query="", citations=_ATTACK_CITATIONS),
            "fp_case_kb": RetrievalResult(query="", citations=_ATTACK_CITATIONS),
        }
        aggregated = _aggregate_citations(results)
        assert len(aggregated) == len(_ATTACK_CITATIONS)

    def test_handles_none_results(self):
        results: dict = {"attack_kb": None, "fp_case_kb": None}
        assert _aggregate_citations(results) == []

    def test_clips_relevance_score(self):
        c = _make_knowledge_citation("cit-99999999", "chk-1", "test_kb", "x", relevance_score=1.5)
        results = {"test_kb": RetrievalResult(query="", citations=[c])}
        aggregated = _aggregate_citations(results)
        assert 0.0 <= aggregated[0].relevance_score <= 1.0


# --------------------------------------------------------------------------- #
# Tests: RAGAgent
# --------------------------------------------------------------------------- #


class TestRAGAgentBasic:
    @pytest.mark.asyncio
    async def test_main_scenario_returns_attack_techniques(self):
        """Main scenario: RAGOutput has >= 2 attack techniques each with citation_id."""
        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)

        assert isinstance(output, RAGOutput)
        assert len(output.attack_techniques) >= 2
        for t in output.attack_techniques:
            assert t.citation_id, f"Technique {t.technique_id} missing citation_id"
        assert output.degraded is False

    @pytest.mark.asyncio
    async def test_main_scenario_includes_t1567_or_t1048(self):
        """At least one of T1567 or T1048 must appear in attack_techniques."""
        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)

        technique_ids = {t.technique_id for t in output.attack_techniques}
        assert ("T1567" in technique_ids) or ("T1048" in technique_ids), (
            f"Expected T1567 or T1048, got {technique_ids}"
        )

    @pytest.mark.asyncio
    async def test_fp_scenario_high_similarity(self):
        """FP scenario: fp_similarity.max_score >= 0.7 with matched case."""
        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)

        assert output.fp_similarity.max_score >= FP_LOW_THRESHOLD, (
            f"Expected fp max_score >= {FP_LOW_THRESHOLD}, got {output.fp_similarity.max_score}"
        )
        assert output.fp_similarity.matched_case_id is not None

    @pytest.mark.asyncio
    async def test_similar_cases_and_playbook_refs(self):
        """RAGOutput contains similar_cases and playbook_refs."""
        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)

        assert len(output.similar_cases) >= 1
        assert len(output.playbook_refs) >= 1

    @pytest.mark.asyncio
    async def test_citations_aggregated(self):
        """RAGOutput citations are aggregated from all KBs."""
        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)

        assert len(output.citations) >= 4

    @pytest.mark.asyncio
    async def test_writes_rag_output_to_event_context(self):
        """rag_output is persisted to working memory."""
        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        await agent._run(input_)

        stored = await wm.read("evt-001", "rag_output")
        assert stored is not None
        assert isinstance(stored, dict)
        assert "attack_techniques" in stored


class TestRAGAgentDegraded:
    @pytest.mark.asyncio
    async def test_single_kb_failure_does_not_interrupt(self):
        """When one KB fails, the other three return results normally."""
        wm = _MockBoundWorkingMemory()
        full = _make_full_results()
        results = {
            "attack_kb": full["attack_kb"],
            "fp_case_kb": RuntimeError("FP KB unavailable"),
            "history_case_kb": full["history_case_kb"],
            "playbook_kb": full["playbook_kb"],
        }
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)

        # Attack techniques should still be present.
        assert len(output.attack_techniques) >= 2
        # FP similarity should be default (KB failed).
        assert output.fp_similarity.max_score == 0.0
        # Similar cases and playbook refs should be present.
        assert len(output.similar_cases) >= 1
        assert len(output.playbook_refs) >= 1
        # Not fully degraded (3 of 4 KBs succeeded).
        assert output.degraded is False

    @pytest.mark.asyncio
    async def test_all_kb_failure_degraded(self):
        """When all KBs fail, degraded=true with complete output structure."""
        wm = _MockBoundWorkingMemory()
        results: dict = {
            "attack_kb": RuntimeError("DB down"),
            "fp_case_kb": RuntimeError("DB down"),
            "history_case_kb": RuntimeError("DB down"),
            "playbook_kb": RuntimeError("DB down"),
        }
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)

        assert output.degraded is True
        assert output.attack_techniques == []
        assert output.fp_similarity.max_score == 0.0
        assert output.similar_cases == []
        assert output.playbook_refs == []
        assert output.citations == []

    @pytest.mark.asyncio
    async def test_no_pipeline_returns_degraded(self):
        """When no pipeline is provided, return degraded empty output."""
        wm = _MockBoundWorkingMemory()
        agent = RAGAgent(working_memory=wm, pipeline=None)

        input_ = _make_input()
        output = await agent._run(input_)

        assert output.degraded is True


class TestRAGAgentPersistence:
    @pytest.mark.asyncio
    async def test_transient_write_failure_marks_degraded(self):
        """When wm.write raises DependencyUnavailableError, output.degraded=True."""
        wm = _FailingWriteMockWM(
            writer_name="RAGAgent",
            fail_key="rag_output",
            fail_error=DependencyUnavailableError("Redis down"),
        )
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent._run(input_)
        assert output.degraded is True

    @pytest.mark.asyncio
    async def test_guardrail_violation_propagates(self):
        """GuardrailViolationError is propagated, not swallowed."""
        wm = _FailingWriteMockWM(
            writer_name="RAGAgent",
            fail_key="rag_output",
            fail_error=GuardrailViolationError(
                "ownership mismatch",
                error_code="working_memory_unauthorized_write",
                details={},
            ),
        )
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        with pytest.raises(GuardrailViolationError):
            await agent._run(input_)

    @pytest.mark.asyncio
    async def test_non_retryable_shadowtrace_error_raises(self):
        """Non-retryable ShadowTraceError propagates."""
        wm = _FailingWriteMockWM(
            writer_name="RAGAgent",
            fail_key="rag_output",
            fail_error=ShadowTraceError(
                "Schema mismatch",
                error_code="schema_error",
                retryable=False,
            ),
        )
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        with pytest.raises(ShadowTraceError) as exc_info:
            await agent._run(input_)
        assert exc_info.value.error_code == "schema_error"


class TestRAGAgentTrace:
    """Verify that execute() writes agent traces."""

    @pytest.mark.asyncio
    async def test_execute_writes_completed_trace(self):
        """When execute() succeeds, trace_service.log_trace is called with completed status."""
        from unittest.mock import AsyncMock, MagicMock

        trace_svc = MagicMock()
        trace_svc.log_trace = AsyncMock()

        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(
            working_memory=wm,
            pipeline=pipeline,
            trace_service=trace_svc,
        )

        input_ = _make_input()
        output = await agent.execute(input_)

        assert isinstance(output, RAGOutput)
        assert output.degraded is False
        trace_svc.log_trace.assert_called_once()
        call_kwargs = trace_svc.log_trace.call_args.kwargs
        assert call_kwargs["agent_name"] == "rag_agent"
        assert call_kwargs["status"] == "completed"
        assert call_kwargs["event_id"] == "evt-001"

    @pytest.mark.asyncio
    async def test_execute_writes_trace_after_pipeline_error(self):
        """When all KBs fail, execute() still records a completed trace."""
        from unittest.mock import AsyncMock, MagicMock

        trace_svc = MagicMock()
        trace_svc.log_trace = AsyncMock()

        wm = _MockBoundWorkingMemory()
        # All four KBs fail → degraded=true, agent completes normally.
        pipeline = _MockPipeline(
            results={
                "attack_kb": RuntimeError("DB crash"),
                "fp_case_kb": RuntimeError("DB crash"),
                "history_case_kb": RuntimeError("DB crash"),
                "playbook_kb": RuntimeError("DB crash"),
            }
        )

        agent = RAGAgent(
            working_memory=wm,
            pipeline=pipeline,
            trace_service=trace_svc,
        )

        input_ = _make_input()
        output = await agent.execute(input_)

        assert output.degraded is True
        trace_svc.log_trace.assert_called_once()
        call_kwargs = trace_svc.log_trace.call_args.kwargs
        assert call_kwargs["agent_name"] == "rag_agent"
        assert call_kwargs["status"] == "completed"

    @pytest.mark.asyncio
    async def test_agent_name_and_input_type_match(self):
        """RAGAgent.agent_name maps to RAGAgentInput in AGENT_INPUT_BY_NAME."""
        from app.models.agent_io import AGENT_INPUT_BY_NAME

        assert AGENT_INPUT_BY_NAME.get("rag_agent") is RAGAgentInput

    @pytest.mark.asyncio
    async def test_execute_without_trace_service_does_not_crash(self):
        """Agent works fine without trace_service injected."""
        wm = _MockBoundWorkingMemory()
        results = _make_full_results()
        pipeline = _MockPipeline(results=results)
        agent = RAGAgent(working_memory=wm, pipeline=pipeline)

        input_ = _make_input()
        output = await agent.execute(input_)
        assert isinstance(output, RAGOutput)
        assert output.degraded is False

    @pytest.mark.asyncio
    async def test_wrong_input_type_raises_typeerror(self):
        """execute() raises TypeError when input doesn't match agent name."""
        agent = RAGAgent()
        input_ = TriageResult(  # type: ignore[call-arg]
            event_type=EventType.DATA_EXFILTRATION,
            severity=Severity.HIGH,
            need_investigation=True,
        )
        with pytest.raises(TypeError, match="requires RAGAgentInput"):
            await agent.execute(input_)  # type: ignore[arg-type]


class TestRAGAgentInputValidation:
    def test_rag_agent_input_accepts_none_evidence(self):
        """RAGAgentInput should accept evidence_output=None."""
        input_ = RAGAgentInput(
            event_id="evt-001",
            triage_result=_make_triage_result(),
            evidence_output=None,
        )
        assert input_.evidence_output is None

    def test_rag_agent_input_extra_forbid(self):
        """RAGAgentInput rejects extra fields."""
        with pytest.raises(pydantic.ValidationError):
            RAGAgentInput(
                event_id="evt-001",
                triage_result=_make_triage_result(),
                unknown_field="should_reject",  # type: ignore[call-arg]
            )


class TestRAGOutputSchema:
    def test_default_rag_output_is_valid(self):
        output = RAGOutput()
        assert output.degraded is False
        assert output.attack_techniques == []
        assert output.fp_similarity.max_score == 0.0
        assert output.similar_cases == []
        assert output.playbook_refs == []
        assert output.citations == []

    def test_fp_similarity_score_bounds(self):
        with pytest.raises(pydantic.ValidationError):
            FpSimilarity(max_score=1.5)

    def test_attack_technique_confidence_bounds(self):
        with pytest.raises(pydantic.ValidationError):
            from app.models.agent_io import AttackTechniqueMatch

            AttackTechniqueMatch(
                technique_id="T1234",
                technique_name="Test",
                match_confidence=1.5,
                citation_id="cit-12345678",
            )
