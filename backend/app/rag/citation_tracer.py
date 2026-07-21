"""CitationTracer: generate citations from retrieved chunks (ISSUE-045)."""

from __future__ import annotations

import hashlib
import re

from app.models.knowledge import Citation, RetrievedChunk


class CitationTracer:
    """Generate deterministic citations with quoted excerpts from final chunks."""

    @staticmethod
    def generate(query: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for chunk in chunks:
            citation_id = _make_citation_id(chunk.chunk_id, chunk.kb_name)
            quoted_text = _extract_quoted_text(query, chunk.content)
            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    kb_name=chunk.kb_name,
                    quoted_text=quoted_text,
                    relevance_score=chunk.score,
                )
            )
        return citations


def _make_citation_id(chunk_id: str, kb_name: str) -> str:
    raw = f"{chunk_id}:{kb_name}"
    hex_digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"cit-{hex_digest}"


def _extract_quoted_text(query: str, content: str, max_len: int = 200) -> str:
    """Extract the most query-relevant excerpt from content, up to max_len chars."""
    query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
    if not query_terms or not content:
        return content[:max_len]

    content_lower = content.lower()

    # Find the window with the most query term hits
    if len(content) <= max_len:
        return content

    best_start = 0
    best_score = 0
    step = max(1, max_len // 4)

    for start in range(0, len(content) - max_len + 1, step):
        window = content_lower[start : start + max_len]
        score = sum(1 for t in query_terms if t in window)
        if score > best_score:
            best_score = score
            best_start = start

    # Fine-tune: slide around the best position by smaller steps
    fine_start = max(0, best_start - step)
    fine_end = min(len(content) - max_len, best_start + step)
    for start in range(fine_start, fine_end + 1):
        window = content_lower[start : start + max_len]
        score = sum(1 for t in query_terms if t in window)
        if score > best_score:
            best_score = score
            best_start = start

    return content[best_start : best_start + max_len]
