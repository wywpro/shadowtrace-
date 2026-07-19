"""KnowledgeStore: pgvector-backed chunk upsert and similarity retrieval (ISSUE-041)."""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, bindparam, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.embedding.service import EmbeddingService
from app.db.orm.knowledge import KnowledgeChunkORM
from app.models.knowledge import KnowledgeChunk, RetrievedChunk


def _merge_hybrid_results(
    vector_hits: list[RetrievedChunk],
    keyword_hits: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Merge vector and keyword hits by chunk_id, keeping the best score."""
    merged: dict[str, RetrievedChunk] = {}
    for hit in vector_hits:
        merged[hit.chunk_id] = hit
    for hit in keyword_hits:
        existing = merged.get(hit.chunk_id)
        if existing is None:
            merged[hit.chunk_id] = hit
            continue
        merged[hit.chunk_id] = RetrievedChunk(
            chunk_id=existing.chunk_id,
            kb_name=existing.kb_name,
            content=existing.content,
            metadata=existing.metadata,
            score=max(existing.score, hit.score),
            retrieval_method="hybrid",
        )
    return sorted(merged.values(), key=lambda row: row.score, reverse=True)[:top_k]


class KnowledgeStore:
    """Persist knowledge chunks and serve vector / keyword search.

    Chunks are idempotent by *chunk_id* across upsert calls.  Vector search
    uses pgvector cosine distance (``<=>``); keyword search uses PostgreSQL
    full-text search with the ``simple`` configuration.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embed_service: EmbeddingService,
    ) -> None:
        self._session_factory = session_factory
        self._embed = embed_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upsert_chunks(self, kb_name: str, chunks: list[KnowledgeChunk]) -> None:
        """Insert or update *chunks* into *kb_name*, computing embeddings inline."""
        if not chunks:
            return
        # Validate kb_name consistency and build content list
        contents: list[str] = []
        for c in chunks:
            if c.kb_name != kb_name:
                raise ValueError(f"chunk {c.chunk_id} kb_name={c.kb_name} != {kb_name}")
            contents.append(c.content)
        vectors = await self._embed.embed_texts(contents)
        async with self._session_factory() as session:
            async with session.begin():
                for chunk, vec in zip(chunks, vectors, strict=True):
                    stmt = (
                        pg_insert(KnowledgeChunkORM)
                        .values(
                            chunk_id=chunk.chunk_id,
                            kb_name=kb_name,
                            content=chunk.content,
                            chunk_metadata=chunk.metadata,
                            embedding=vec,
                        )
                        .on_conflict_do_update(
                            index_elements=["chunk_id"],
                            set_={
                                "kb_name": kb_name,
                                "content": chunk.content,
                                "metadata": chunk.metadata,
                                "embedding": vec,
                            },
                        )
                    )
                    await session.execute(stmt)

    async def vector_search(
        self, kb_name: str, query_embedding: list[float], top_k: int = 10
    ) -> list[RetrievedChunk]:
        """Cosine-similarity search across vectors in *kb_name*."""
        sql = text(
            """
            SELECT chunk_id, kb_name, content, metadata,
                   1.0 - (embedding <=> :q) AS score
            FROM knowledge_chunk
            WHERE kb_name = :kb_name
            ORDER BY embedding <=> :q
            LIMIT :top_k
            """
        ).bindparams(
            bindparam("q", type_=Vector),
            bindparam("kb_name", type_=String),
            bindparam("top_k", type_=Integer),
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"kb_name": kb_name, "q": query_embedding, "top_k": top_k},
            )
            return [
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    kb_name=row.kb_name,
                    content=row.content,
                    metadata=row.metadata or {},
                    score=float(row.score),
                    retrieval_method="vector",
                )
                for row in result.fetchall()
            ]

    async def keyword_search(
        self, kb_name: str, query_text: str, top_k: int = 10
    ) -> list[RetrievedChunk]:
        """PostgreSQL full-text search across chunks in *kb_name*."""
        sql = text(
            """
            SELECT chunk_id, kb_name, content, metadata,
                   ts_rank(to_tsvector('simple', content),
                           plainto_tsquery('simple', :q)) AS score
            FROM knowledge_chunk
            WHERE kb_name = :kb_name
              AND to_tsvector('simple', content) @@ plainto_tsquery('simple', :q)
            ORDER BY score DESC
            LIMIT :top_k
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"kb_name": kb_name, "q": query_text, "top_k": top_k},
            )
            return [
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    kb_name=row.kb_name,
                    content=row.content,
                    metadata=row.metadata or {},
                    score=float(row.score),
                    retrieval_method="keyword",
                )
                for row in result.fetchall()
            ]

    async def hybrid_search(
        self,
        kb_name: str,
        query_text: str,
        *,
        keyword_query: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Vector search plus keyword fallback, merged by chunk_id."""
        query_vec = await self._embed.embed_query(query_text)
        vector_hits = await self.vector_search(kb_name, query_vec, top_k=top_k)
        keyword_hits = await self.keyword_search(
            kb_name,
            keyword_query if keyword_query is not None else query_text,
            top_k=top_k,
        )
        return _merge_hybrid_results(vector_hits, keyword_hits, top_k)

    async def count(self, kb_name: str) -> int:
        """Return the number of chunks stored in *kb_name*."""
        sql = text("SELECT COUNT(*) AS cnt FROM knowledge_chunk WHERE kb_name = :kb_name")
        async with self._session_factory() as session:
            result = await session.execute(sql, {"kb_name": kb_name})
            row = result.fetchone()
            return int(row.cnt) if row else 0
