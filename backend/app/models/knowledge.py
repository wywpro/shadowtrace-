"""Pydantic domain models for knowledge chunks and retrieval results (ISSUE-041, ISSUE-045)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    """A chunk of knowledge to be stored and embedded."""

    chunk_id: str = Field(..., description="chk-{8 hex}")
    kb_name: str = Field(..., description="attack_kb | fp_case_kb | history_case_kb | playbook_kb")
    content: str = Field(..., description="Plain-text chunk body")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class RetrievedChunk(BaseModel):
    """A chunk returned from vector or keyword search, or after RRF fusion."""

    chunk_id: str
    kb_name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(..., description="Normalized 0-1 relevance score (RRF or rerank)")
    retrieval_method: str = Field(..., description="'vector', 'keyword', 'hybrid', or 'reranked'")
    raw_rrf_score: float = Field(default=0.0, description="Raw RRF score before normalization")


class Citation(BaseModel):
    """A citation referencing a retrieved chunk (ISSUE-045)."""

    citation_id: str = Field(..., pattern=r"^cit-[0-9a-fA-F]{8}$", description="cit-{8 hex}")
    chunk_id: str
    kb_name: str
    quoted_text: str = Field(..., description="Relevant excerpt, max 200 chars")
    relevance_score: float


class RetrievalResult(BaseModel):
    """Complete result from the RAG retrieval pipeline (ISSUE-045)."""

    query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    degraded_steps: list[str] = Field(default_factory=list)
