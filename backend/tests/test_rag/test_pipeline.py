"""Tests for RAG hybrid retrieval pipeline (ISSUE-045).

Covers:
- RRF fusion math correctness (hand-computed rankings)
- MockReranker determinism and ordering
- CitationTracer excerpt generation and traceability
- Pipeline degradation: rewrite failure, rerank failure
- Full end-to-end deterministic results (integration, requires PostgreSQL)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.core.embedding.service import EmbeddingService
from app.core.errors import LLMError
from app.core.llm.base import InMemoryLLMCallAuditRecorder
from app.core.llm.mock_client import MockLLMClient
from app.models.knowledge import KnowledgeChunk, RetrievalResult, RetrievedChunk
from app.rag.citation_tracer import CitationTracer
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.pipeline import RetrievalPipeline
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import MockReranker, Reranker
from app.rag.rrf_fusion import rrf_fuse
from app.services.knowledge_store import KnowledgeStore

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return cfg


# ============================================================================
# Helpers
# ============================================================================


def _make_chunk(
    chunk_id: str, kb_name: str, content: str, score: float = 1.0, method: str = "vector"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, kb_name=kb_name, content=content, score=score, retrieval_method=method
    )


def _knowledge_chunk(chunk_id: str, kb_name: str, content: str) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=chunk_id, kb_name=kb_name, content=content)


# ============================================================================
# Pure function tests (no DB required)
# ============================================================================


class TestRRFFusion:
    """RRF math correctness with hand-computed rankings."""

    def test_empty_input_returns_empty(self) -> None:
        assert rrf_fuse([]) == []
        assert rrf_fuse([[]]) == []
        assert rrf_fuse([[], [], []]) == []

    def test_single_list_preserves_ordering(self) -> None:
        chunks = [
            _make_chunk("chk-a", "kb1", "first", score=0.9),
            _make_chunk("chk-b", "kb1", "second", score=0.7),
            _make_chunk("chk-c", "kb1", "third", score=0.5),
        ]
        result = rrf_fuse([chunks], k=60)
        assert len(result) == 3
        assert result[0].chunk_id == "chk-a"
        assert result[1].chunk_id == "chk-b"
        assert result[2].chunk_id == "chk-c"
        for c in result:
            assert c.retrieval_method == "hybrid"
            assert c.raw_rrf_score > 0

    def test_normalized_scores_in_0_1_range(self) -> None:
        chunks_a = [_make_chunk(f"chk-{i}", "kb1", f"doc{i}") for i in range(5)]
        chunks_b = [_make_chunk(f"chk-{i}", "kb1", f"doc{i}") for i in range(3)]
        result = rrf_fuse([chunks_a, chunks_b], k=60)
        for c in result:
            assert 0.0 <= c.score <= 1.0, f"score {c.score} out of [0,1]"
            assert c.raw_rrf_score > 0

    def test_hand_computed_rrf_math(self) -> None:
        """Verify RRF scores against manual calculation for two small lists."""
        list_a = [
            _make_chunk("chk-x", "kb1", "doc x"),
            _make_chunk("chk-y", "kb1", "doc y"),
        ]
        list_b = [
            _make_chunk("chk-y", "kb1", "doc y"),  # ranked 1st here
            _make_chunk("chk-z", "kb1", "doc z"),
        ]
        k = 60
        # chk-x: rank 1 in list_a -> 1/(60+1) = 1/61
        # chk-y: rank 2 in list_a + rank 1 in list_b -> 1/62 + 1/61
        # chk-z: rank 2 in list_b -> 1/62
        # N=2 lists, theoretical max = 2/61
        raw_x = 1.0 / (k + 1)
        raw_y = 1.0 / (k + 2) + 1.0 / (k + 1)
        raw_z = 1.0 / (k + 2)
        norm_factor = 2.0 / (k + 1)

        result = rrf_fuse([list_a, list_b], k=k)
        assert len(result) == 3

        by_id = {c.chunk_id: c for c in result}
        assert by_id["chk-y"].score == pytest.approx(raw_y / norm_factor)
        assert by_id["chk-y"].raw_rrf_score == pytest.approx(raw_y)
        assert by_id["chk-x"].score == pytest.approx(raw_x / norm_factor)
        assert by_id["chk-z"].score == pytest.approx(raw_z / norm_factor)

        # chk-y should rank first (highest combined score)
        assert result[0].chunk_id == "chk-y"

    def test_deduplicates_by_chunk_id_across_lists(self) -> None:
        """Same chunk in multiple lists should be deduplicated."""
        chunk = _make_chunk("chk-dup", "kb1", "duplicate content")
        result = rrf_fuse([[chunk], [chunk]], k=60)
        assert len(result) == 1
        assert result[0].chunk_id == "chk-dup"

    def test_cross_kb_same_chunk_id_kept_separate(self) -> None:
        """Same chunk_id in different KBs are distinct."""
        a = _make_chunk("chk-01", "attack_kb", "attack info")
        b = _make_chunk("chk-01", "playbook_kb", "playbook info")
        result = rrf_fuse([[a], [b]], k=60)
        assert len(result) == 2


class TestMockReranker:
    """Deterministic mock reranker tests."""

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self) -> None:
        reranker = MockReranker()
        result = await reranker.rerank("test query", [], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_scores_in_0_1_range(self) -> None:
        chunks = [
            _make_chunk(f"chk-{i}", "kb1", f"content number {i}", score=0.8 - i * 0.1)
            for i in range(3)
        ]
        reranker = MockReranker()
        result = await reranker.rerank("content number 1", chunks, top_k=3)
        assert len(result) == 3
        for c in result:
            assert 0.0 <= c.score <= 1.0, f"score {c.score} out of [0,1]"
            assert c.retrieval_method == "reranked"

    @pytest.mark.asyncio
    async def test_query_overlap_boosts_relevant_content(self) -> None:
        chunks = [
            _make_chunk("chk-a", "kb1", "ransomware encrypts all files on endpoints", score=0.8),
            _make_chunk("chk-b", "kb1", "phishing email with malicious link detected", score=0.8),
            _make_chunk(
                "chk-c", "kb1", "ransomware attack via phishing email on network", score=0.8
            ),
        ]
        reranker = MockReranker()
        result = await reranker.rerank("ransomware network", chunks, top_k=3)
        assert len(result) == 3
        # chk-c has both "ransomware" and "network" -> highest overlap
        # chk-a has "ransomware" only -> middle overlap
        # chk-b has neither -> lowest overlap
        assert result[0].chunk_id == "chk-c"
        assert result[-1].chunk_id == "chk-b"

    @pytest.mark.asyncio
    async def test_rerank_is_deterministic(self) -> None:
        chunks = [
            _make_chunk("chk-a", "kb1", "sql injection in login form", score=0.9),
            _make_chunk("chk-b", "kb1", "cross-site scripting in comment field", score=0.7),
            _make_chunk("chk-c", "kb1", "sql injection prevention best practices", score=0.8),
        ]
        reranker = MockReranker()
        result1 = await reranker.rerank("sql injection", chunks, top_k=3)
        result2 = await reranker.rerank("sql injection", chunks, top_k=3)
        assert [c.chunk_id for c in result1] == [c.chunk_id for c in result2]
        assert [c.score for c in result1] == [c.score for c in result2]

    @pytest.mark.asyncio
    async def test_respects_top_k(self) -> None:
        chunks = [_make_chunk(f"chk-{i}", "kb1", f"doc {i}", score=0.5) for i in range(10)]
        reranker = MockReranker()
        result = await reranker.rerank("doc 5", chunks, top_k=3)
        assert len(result) == 3


class TestCitationTracer:
    """Citation generation and traceability."""

    def test_generates_citation_for_each_chunk(self) -> None:
        chunks = [
            _make_chunk(
                "chk-a", "attack_kb", "spear phishing campaign targeting executives", score=0.95
            ),
            _make_chunk(
                "chk-b", "playbook_kb", "isolate compromised host from network", score=0.82
            ),
        ]
        citations = CitationTracer.generate("phishing attack", chunks)
        assert len(citations) == 2
        for cit in citations:
            assert cit.citation_id.startswith("cit-")
            assert len(cit.citation_id) == 12  # "cit-" + 8 hex
            assert cit.relevance_score > 0

    def test_citation_id_is_deterministic(self) -> None:
        chunks = [_make_chunk("chk-x", "kb1", "test content", score=0.9)]
        c1 = CitationTracer.generate("test", chunks)
        c2 = CitationTracer.generate("test", chunks)
        assert c1[0].citation_id == c2[0].citation_id

    def test_citation_id_uses_chunk_id_and_kb_name(self) -> None:
        """Different chunk_id or kb_name -> different citation_id."""
        a = [_make_chunk("chk-01", "attack_kb", "content", score=0.9)]
        b = [_make_chunk("chk-01", "playbook_kb", "content", score=0.9)]
        c = [_make_chunk("chk-02", "attack_kb", "content", score=0.9)]
        cit_a = CitationTracer.generate("test", a)[0]
        cit_b = CitationTracer.generate("test", b)[0]
        cit_c = CitationTracer.generate("test", c)[0]
        assert cit_a.citation_id != cit_b.citation_id
        assert cit_a.citation_id != cit_c.citation_id

    def test_quoted_text_found_in_original_content(self) -> None:
        content = (
            "The threat actor used a sophisticated spear-phishing campaign "
            "targeting C-level executives to deploy ransomware across the corporate network. "
            "Initial access was gained through a malicious Excel attachment."
        )
        chunks = [_make_chunk("chk-01", "attack_kb", content, score=0.9)]
        citations = CitationTracer.generate("ransomware spear-phishing", chunks)
        quoted = citations[0].quoted_text
        assert quoted in content
        assert len(quoted) <= 200

    def test_short_content_not_truncated(self) -> None:
        short = "brief alert: suspicious login"
        chunks = [_make_chunk("chk-s", "kb1", short, score=0.9)]
        citations = CitationTracer.generate("suspicious login", chunks)
        assert citations[0].quoted_text == short


# ============================================================================
# Pipeline degradation tests (no DB required)
# ============================================================================


class _FailingRewriter:
    async def rewrite(self, query: str) -> list[str]:
        raise LLMError("simulated LLM failure", error_code="llm_test_error")


class _BrokenLLM:
    async def chat(self, *args: object, **kwargs: object) -> object:
        raise LLMError("simulated LLM failure", error_code="llm_test_error")


class _FailingReranker:
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        raise RuntimeError("simulated reranker failure")


class _ConstantRetriever:
    def __init__(self, result_lists: list[list[RetrievedChunk]]) -> None:
        self._results = result_lists

    async def retrieve(
        self, queries: list[str], kb_names: list[str], top_k: int = 5
    ) -> list[list[RetrievedChunk]]:
        return self._results


class TestPipelineDegradation:
    """Degraded step handling per acceptance criteria."""

    @pytest.mark.asyncio
    async def test_rewrite_failure_falls_back_to_original_query(self) -> None:
        """rewrite failure -> degraded_steps + original query used."""
        chunks = [
            _make_chunk("chk-01", "attack_kb", "ransomware attack on corporate network", score=0.9),
            _make_chunk(
                "chk-02", "attack_kb", "phishing email detected and quarantined", score=0.7
            ),
        ]
        pipeline = RetrievalPipeline(
            rewriter=_FailingRewriter(),  # type: ignore[arg-type]
            retriever=_ConstantRetriever([chunks, chunks]),  # type: ignore[arg-type]
            reranker=MockReranker(),
        )
        result = await pipeline.retrieve("ransomware attack", ["attack_kb"], top_k=2)
        assert "query_rewriter" in result.degraded_steps
        assert result.query == "ransomware attack"
        assert result.rewritten_queries == ["ransomware attack"]
        assert len(result.chunks) > 0
        assert len(result.citations) > 0

    @pytest.mark.asyncio
    async def test_query_rewriter_llm_failure_records_degraded_step(self) -> None:
        """Real QueryRewriter surfaces LLM failures to degraded_steps via pipeline."""
        chunks = [_make_chunk("chk-01", "attack_kb", "ransomware on corporate network", score=0.9)]
        pipeline = RetrievalPipeline(
            rewriter=QueryRewriter(_BrokenLLM(), event_id="test", agent_name="test"),  # type: ignore[arg-type]
            retriever=_ConstantRetriever([chunks, chunks]),  # type: ignore[arg-type]
            reranker=MockReranker(),
        )
        result = await pipeline.retrieve("ransomware attack", ["attack_kb"], top_k=1)
        assert "query_rewriter" in result.degraded_steps
        assert result.rewritten_queries == ["ransomware attack"]
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_remote_rerank_mode_degrades_to_rrf_order(self) -> None:
        chunks = [
            _make_chunk("chk-a", "kb1", "first result", score=0.9),
            _make_chunk("chk-b", "kb1", "second result", score=0.7),
        ]
        pipeline = RetrievalPipeline(
            rewriter=QueryRewriter(
                MockLLMClient(audit_recorder=InMemoryLLMCallAuditRecorder()),
                event_id="test",
                agent_name="test",
            ),
            retriever=_ConstantRetriever([chunks, chunks]),  # type: ignore[arg-type]
            reranker=Reranker(Settings(rerank_mode="remote")),
        )
        result = await pipeline.retrieve("test query", ["kb1"], top_k=2)
        assert "reranker" in result.degraded_steps
        assert [c.chunk_id for c in result.chunks] == ["chk-a", "chk-b"]

    @pytest.mark.asyncio
    async def test_rerank_failure_uses_rrf_order(self) -> None:
        """rerank failure -> degraded_steps + RRF order preserved."""
        chunks = [
            _make_chunk("chk-a", "kb1", "first result", score=0.9),
            _make_chunk("chk-b", "kb1", "second result", score=0.7),
            _make_chunk("chk-c", "kb1", "third result", score=0.5),
        ]
        pipeline = RetrievalPipeline(
            rewriter=QueryRewriter(
                MockLLMClient(audit_recorder=InMemoryLLMCallAuditRecorder()),
                event_id="test",
                agent_name="test",
            ),
            retriever=_ConstantRetriever([chunks, chunks]),  # type: ignore[arg-type]
            reranker=_FailingReranker(),  # type: ignore[arg-type]
        )
        result = await pipeline.retrieve("test query", ["kb1"], top_k=3)
        assert "reranker" in result.degraded_steps
        assert len(result.chunks) == 3
        assert result.chunks[0].chunk_id == "chk-a"
        assert result.chunks[1].chunk_id == "chk-b"
        assert result.chunks[2].chunk_id == "chk-c"

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_empty_result(self) -> None:
        """When retrieval returns zero results, pipeline returns empty without error."""
        pipeline = RetrievalPipeline(
            rewriter=QueryRewriter(
                MockLLMClient(audit_recorder=InMemoryLLMCallAuditRecorder()),
                event_id="test",
                agent_name="test",
            ),
            retriever=_ConstantRetriever([]),  # type: ignore[arg-type]
            reranker=MockReranker(),
        )
        result = await pipeline.retrieve("empty", ["kb1"], top_k=5)
        assert result.chunks == []
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_both_rewrite_and_rerank_failure_recorded(self) -> None:
        """Both failures -> both in degraded_steps, pipeline still returns results."""
        chunks = [_make_chunk("chk-x", "kb1", "some content", score=0.9)]
        pipeline = RetrievalPipeline(
            rewriter=_FailingRewriter(),  # type: ignore[arg-type]
            retriever=_ConstantRetriever([chunks, chunks]),  # type: ignore[arg-type]
            reranker=_FailingReranker(),  # type: ignore[arg-type]
        )
        result = await pipeline.retrieve("test", ["kb1"], top_k=1)
        assert "query_rewriter" in result.degraded_steps
        assert "reranker" in result.degraded_steps
        assert len(result.chunks) == 1


# ============================================================================
# Full pipeline integration tests (requires PostgreSQL)
# ============================================================================


@pytest.fixture(scope="module")
def migrated() -> None:
    command.upgrade(_alembic_config(), "head")


@pytest_asyncio.fixture
async def session_factory(
    migrated: None,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def clean_knowledge(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async with session_factory() as session:
        await session.execute(text("DELETE FROM knowledge_chunk"))
        await session.commit()


@pytest_asyncio.fixture
def embed_service() -> EmbeddingService:
    return EmbeddingService(Settings(embedding_mode="mock"))


@pytest_asyncio.fixture
def knowledge_store(
    session_factory: async_sessionmaker[AsyncSession],
    embed_service: EmbeddingService,
) -> KnowledgeStore:
    return KnowledgeStore(session_factory, embed_service)


@pytest_asyncio.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(audit_recorder=InMemoryLLMCallAuditRecorder())


def _build_pipeline(
    store: KnowledgeStore,
    embed_service: EmbeddingService,
    llm: MockLLMClient,
) -> RetrievalPipeline:
    return RetrievalPipeline(
        rewriter=QueryRewriter(llm, event_id="test-event", agent_name="RAGAgent"),
        retriever=HybridRetriever(store, embed_service),
        reranker=MockReranker(),
    )


@pytest_asyncio.fixture
async def seeded_store(
    knowledge_store: KnowledgeStore,
    clean_knowledge: None,
) -> KnowledgeStore:
    """Seed 6 chunks across 2 KBs for integration tests."""
    attack_chunks = [
        _knowledge_chunk(
            "atk-00000001",
            "attack_kb",
            "Spear phishing campaign targeting C-level executives via malicious Excel attachments",
        ),
        _knowledge_chunk(
            "atk-00000002",
            "attack_kb",
            "Ransomware deployment using CVE-2024-1234 for lateral movement "
            "across domain controllers",
        ),
        _knowledge_chunk(
            "atk-00000003",
            "attack_kb",
            "Credential dumping with Mimikatz on compromised Windows workstations",
        ),
    ]
    playbook_chunks = [
        _knowledge_chunk(
            "plb-00000001",
            "playbook_kb",
            "Isolate compromised host from network immediately upon detection",
        ),
        _knowledge_chunk(
            "plb-00000002",
            "playbook_kb",
            "Reset all domain admin passwords and revoke active sessions",
        ),
        _knowledge_chunk(
            "plb-00000003",
            "playbook_kb",
            "Collect memory dump and disk image from affected endpoint for forensic analysis",
        ),
    ]
    await knowledge_store.upsert_chunks("attack_kb", attack_chunks)
    await knowledge_store.upsert_chunks("playbook_kb", playbook_chunks)
    return knowledge_store


class TestFullPipelineIntegration:
    """End-to-end pipeline tests requiring a running PostgreSQL."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_deterministic_results_same_query_twice(
        self,
        seeded_store: KnowledgeStore,
        embed_service: EmbeddingService,
        mock_llm: MockLLMClient,
    ) -> None:
        """Acceptance criterion: same query twice -> identical results in mock mode."""
        pipeline = _build_pipeline(seeded_store, embed_service, mock_llm)

        result1 = await pipeline.retrieve("phishing attack on executives", ["attack_kb"], top_k=3)
        result2 = await pipeline.retrieve("phishing attack on executives", ["attack_kb"], top_k=3)

        assert [c.chunk_id for c in result1.chunks] == [c.chunk_id for c in result2.chunks]
        assert [c.score for c in result1.chunks] == [c.score for c in result2.chunks]
        assert [c.citation_id for c in result1.citations] == [
            c.citation_id for c in result2.citations
        ]
        assert result1.query == result2.query

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_kb_retrieval(
        self,
        seeded_store: KnowledgeStore,
        embed_service: EmbeddingService,
        mock_llm: MockLLMClient,
    ) -> None:
        """Retrieval across multiple KBs returns results from all specified KBs."""
        pipeline = _build_pipeline(seeded_store, embed_service, mock_llm)

        result = await pipeline.retrieve(
            "incident response and threat detection",
            ["attack_kb", "playbook_kb"],
            top_k=5,
        )
        assert len(result.chunks) > 0
        kbs = {c.kb_name for c in result.chunks}
        assert kbs.issubset({"attack_kb", "playbook_kb"})

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_every_chunk_has_citation_with_quoted_text(
        self,
        seeded_store: KnowledgeStore,
        embed_service: EmbeddingService,
        mock_llm: MockLLMClient,
    ) -> None:
        """Acceptance criterion: each final chunk has a citation with quoted_text."""
        pipeline = _build_pipeline(seeded_store, embed_service, mock_llm)

        result = await pipeline.retrieve("ransomware lateral movement", ["attack_kb"], top_k=3)
        assert len(result.citations) == len(result.chunks)

        for cit in result.citations:
            assert cit.citation_id.startswith("cit-")
            assert len(cit.citation_id) == 12
            assert len(cit.quoted_text) > 0
            assert len(cit.quoted_text) <= 200

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_citation_quoted_text_traceable_to_chunk(
        self,
        seeded_store: KnowledgeStore,
        embed_service: EmbeddingService,
        mock_llm: MockLLMClient,
    ) -> None:
        """Acceptance criterion: each quoted_text can be traced back to its chunk."""
        pipeline = _build_pipeline(seeded_store, embed_service, mock_llm)

        result = await pipeline.retrieve("credential dumping mimikatz", ["attack_kb"], top_k=3)

        chunk_by_id = {c.chunk_id: c.content for c in result.chunks}
        for cit in result.citations:
            assert cit.chunk_id in chunk_by_id
            assert cit.quoted_text in chunk_by_id[cit.chunk_id]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_empty_kb_returns_empty_result(
        self,
        knowledge_store: KnowledgeStore,
        embed_service: EmbeddingService,
        mock_llm: MockLLMClient,
        clean_knowledge: None,
    ) -> None:
        """Querying an empty KB returns empty but valid RetrievalResult."""
        pipeline = _build_pipeline(knowledge_store, embed_service, mock_llm)

        result = await pipeline.retrieve("anything", ["attack_kb"], top_k=5)
        assert isinstance(result, RetrievalResult)
        assert result.chunks == []
        assert result.citations == []
        assert result.query == "anything"
