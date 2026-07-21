"""RetrievalPipeline: full RAG pipeline orchestrator (ISSUE-045)."""

from __future__ import annotations

import logging

from app.models.knowledge import RetrievalResult, RetrievedChunk
from app.rag.citation_tracer import CitationTracer
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import Reranker
from app.rag.rrf_fusion import rrf_fuse

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Wire query rewriting → hybrid retrieval → RRF fusion → rerank → citation.

    Each non-retrieval step that fails is recorded in ``degraded_steps`` and the
    pipeline continues with the best available intermediate results.  If all
    retrieval paths return empty, the pipeline returns an empty result.
    """

    def __init__(
        self,
        rewriter: QueryRewriter,
        retriever: HybridRetriever,
        reranker: Reranker,
    ) -> None:
        self._rewriter = rewriter
        self._retriever = retriever
        self._reranker = reranker

    async def retrieve(self, query: str, kb_names: list[str], top_k: int = 5) -> RetrievalResult:
        degraded: list[str] = []

        # Step 1: Query rewriting
        rewritten: list[str]
        try:
            rewritten = await self._rewriter.rewrite(query)
        except Exception as exc:
            logger.warning("Query rewriting failed: %s", exc)
            degraded.append("query_rewriter")
            rewritten = [query]

        # Step 2: Hybrid retrieval (per query, per kb, vector + keyword).
        # Per-path failures are swallowed inside HybridRetriever as empty lists;
        # an all-empty outcome is handled below without marking retrieval degraded.
        result_lists = await self._retriever.retrieve(rewritten, kb_names, top_k=top_k)

        # If all lists are empty, return empty
        if not any(result_lists):
            return RetrievalResult(
                query=query,
                rewritten_queries=rewritten,
                chunks=[],
                citations=[],
                degraded_steps=degraded,
            )

        # Step 3: RRF fusion
        fused: list[RetrievedChunk] = rrf_fuse(result_lists, k=60)

        # Step 4: Rerank
        reranked: list[RetrievedChunk]
        try:
            reranked = await self._reranker.rerank(query, fused, top_k)
        except Exception as exc:
            logger.warning("Reranking failed, using RRF order: %s", exc)
            degraded.append("reranker")
            reranked = fused[:top_k]
            # Ensure scores are 0-1 if using raw RRF results
            reranked = _ensure_normalized(reranked)

        # Step 5: Citations
        citations = CitationTracer.generate(query, reranked)

        return RetrievalResult(
            query=query,
            rewritten_queries=rewritten,
            chunks=reranked,
            citations=citations,
            degraded_steps=degraded,
        )


def _ensure_normalized(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-normalize chunk scores to [0, 1] when reranker was skipped."""
    if not chunks:
        return chunks
    scores = [c.score for c in chunks]
    max_s = max(scores)
    min_s = min(scores)
    rng = max_s - min_s if max_s != min_s else 1.0
    result: list[RetrievedChunk] = []
    for c in chunks:
        norm_score = (c.score - min_s) / rng if rng > 0 else 1.0
        result.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                kb_name=c.kb_name,
                content=c.content,
                metadata=c.metadata,
                score=norm_score,
                retrieval_method=c.retrieval_method,
                raw_rrf_score=c.raw_rrf_score,
            )
        )
    return result
