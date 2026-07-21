"""Reranker: score-based re-ranking with mock and remote backends (ISSUE-045)."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.models.knowledge import RetrievedChunk


class Reranker:
    """Re-rank retrieved chunks by relevance to the query.

    Dispatch on ``RERANK_MODE``:
      - ``mock``: deterministic score + query-overlap weighted re-rank
      - ``remote``: reserved for future cross-encoder endpoint
    """

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self._mode = cfg.rerank_mode.strip().lower()

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if self._mode == "mock":
            return _mock_rerank(query, chunks, top_k)
        raise NotImplementedError(
            f"RERANK_MODE={self._mode!r} is not implemented; use mock or skip reranking"
        )

    @property
    def mode(self) -> str:
        return self._mode


def _mock_rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Deterministic re-rank: weighted combination of original score and query overlap."""
    query_terms = set(query.lower().split())

    def _overlap(content: str) -> float:
        content_lower = content.lower()
        if not query_terms:
            return 0.0
        hits = sum(1 for t in query_terms if t in content_lower)
        return hits / len(query_terms)

    scored: list[tuple[float, RetrievedChunk]] = []
    for chunk in chunks:
        overlap = _overlap(chunk.content)
        new_score = 0.6 * chunk.score + 0.4 * overlap
        scored.append((new_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return []

    # Re-normalize scores to 0-1
    max_score = top[0][0]
    min_score = top[-1][0]
    score_range = max_score - min_score if max_score != min_score else 1.0

    result: list[RetrievedChunk] = []
    for score, chunk in top:
        normalized = (score - min_score) / score_range if score_range > 0 else 1.0
        result.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                kb_name=chunk.kb_name,
                content=chunk.content,
                metadata=chunk.metadata,
                score=normalized,
                retrieval_method="reranked",
                raw_rrf_score=chunk.raw_rrf_score,
            )
        )
    return result


class MockReranker(Reranker):
    """Explicit mock reranker for use when mode is known at construction time."""

    def __init__(self) -> None:
        self._mode = "mock"
