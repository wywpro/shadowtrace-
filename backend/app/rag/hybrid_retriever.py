"""HybridRetriever: concurrent vector + keyword retrieval across KBs (ISSUE-045)."""

from __future__ import annotations

import asyncio

from app.core.embedding.service import EmbeddingService
from app.models.knowledge import RetrievedChunk
from app.services.knowledge_store import KnowledgeStore


class HybridRetriever:
    """For each query variant, run vector + keyword search concurrently across KBs.

    Each search path fetches ``top_k * 2`` candidates; the separate result lists
    are fed into RRF fusion downstream.
    """

    def __init__(self, store: KnowledgeStore, embed_service: EmbeddingService) -> None:
        self._store = store
        self._embed = embed_service

    async def retrieve(
        self, queries: list[str], kb_names: list[str], top_k: int = 5
    ) -> list[list[RetrievedChunk]]:
        """Return one result list per (query, kb, method) combination.

        Order: for each query, for each kb, vector then keyword.
        Total lists = len(queries) * len(kb_names) * 2.
        """
        fetch_k = top_k * 2

        async def _search(query: str, kb: str, method: str) -> list[RetrievedChunk]:
            if method == "vector":
                vec = await self._embed.embed_query(query)
                return await self._store.vector_search(kb, vec, top_k=fetch_k)
            return await self._store.keyword_search(kb, query, top_k=fetch_k)

        tasks: list[asyncio.Task[list[RetrievedChunk]]] = []
        for query in queries:
            for kb in kb_names:
                for method in ("vector", "keyword"):
                    tasks.append(asyncio.create_task(_search(query, kb, method)))

        results: list[list[RetrievedChunk]] = []
        for task in tasks:
            try:
                results.append(await task)
            except Exception:
                results.append([])
        return results
