"""RRF (Reciprocal Rank Fusion) with dedup and 0-1 normalization (ISSUE-045)."""

from __future__ import annotations

from app.models.knowledge import RetrievedChunk


def rrf_fuse(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Fuse multiple ranked result lists via Reciprocal Rank Fusion.

    Each chunk's ``raw_rrf_score = sum(1 / (k + rank_i))`` across all lists
    where it appears (rank is 1-indexed).  The score is then normalized to
    [0, 1] by dividing by the theoretical maximum ``N / (k + 1)`` where *N*
    is the number of non-empty input lists.

    Returns chunks sorted by normalized score descending, with
    ``retrieval_method="hybrid"``.
    """
    effective_lists = [lst for lst in result_lists if lst]
    if not effective_lists:
        return []

    rrf: dict[str, tuple[RetrievedChunk, float]] = {}
    n = len(effective_lists)

    for lst in effective_lists:
        for rank, chunk in enumerate(lst, start=1):
            contribution = 1.0 / (k + rank)
            key = f"{chunk.kb_name}:{chunk.chunk_id}"
            if key in rrf:
                existing_chunk, existing_score = rrf[key]
                rrf[key] = (existing_chunk, existing_score + contribution)
            else:
                rrf[key] = (chunk, contribution)

    if not rrf:
        return []

    max_theoretical = n / (k + 1)

    fused: list[RetrievedChunk] = []
    for chunk, raw_score in rrf.values():
        normalized = raw_score / max_theoretical if max_theoretical > 0 else 0.0
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                kb_name=chunk.kb_name,
                content=chunk.content,
                metadata=chunk.metadata,
                score=min(normalized, 1.0),
                retrieval_method="hybrid",
                raw_rrf_score=raw_score,
            )
        )

    fused.sort(key=lambda c: c.score, reverse=True)
    return fused
