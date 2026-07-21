"""AttackKBService: ATT&CK technique knowledge base operations (ISSUE-042)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.knowledge import KnowledgeChunk, RetrievedChunk
from app.services.knowledge_store import KnowledgeStore

KB_NAME = "attack_kb"

# P0 mock-stage aliases: MockEmbedder cannot cross-match Chinese queries to
# English technique text. Keyword leg uses expanded terms until remote
# embeddings land (GitHub issue #522).
_KEYWORD_QUERY_ALIASES: dict[str, str] = {
    "数据外泄": "exfiltration",
}


def _derive_chunk_id(technique_id: str, attack_version: str) -> str:
    raw = f"technique_id:{technique_id}:attack_version:{attack_version}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"atk-{digest}"


def _format_content(t: dict[str, Any]) -> str:
    tactics = ", ".join(t["tactics"])
    return (
        f"Technique: {t['technique_name']}\n"
        f"ID: {t['technique_id']}\n"
        f"Tactics: {tactics}\n"
        f"Description: {t['description']}\n"
        f"Detection: {t['detection']}"
    )


class AttackKBService:
    """Manage the ATT&CK technique knowledge base.

    Provides file-based loading with idempotent upsert, precise technique
    lookup by technique_id, and semantic search over technique descriptions.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._store = store
        self._session_factory = session_factory

    async def load_from_file(self, path: str | Path) -> int:
        """Load techniques from a JSON file and upsert into attack_kb.

        Returns the number of techniques loaded.  Repeated loads are
        idempotent — chunk_id is derived from technique_id + attack_version.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        attack_version: str = data["attack_version"]
        techniques: list[dict[str, Any]] = data["techniques"]

        chunks: list[KnowledgeChunk] = []
        for t in techniques:
            chunk_id = _derive_chunk_id(t["technique_id"], attack_version)
            content = _format_content(t)
            metadata: dict[str, Any] = {
                "technique_id": t["technique_id"],
                "technique_name": t["technique_name"],
                "tactics": t["tactics"],
                "description": t["description"],
                "detection": t["detection"],
                "attack_version": attack_version,
            }
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    kb_name=KB_NAME,
                    content=content,
                    metadata=metadata,
                )
            )

        await self._store.upsert_chunks(KB_NAME, chunks)
        return len(chunks)

    async def get_technique(self, technique_id: str) -> dict[str, Any] | None:
        """Look up a technique by its MITRE ATT&CK technique ID (e.g. T1078)."""
        sql = text(
            """
            SELECT metadata
            FROM knowledge_chunk
            WHERE kb_name = :kb_name
              AND metadata ->> 'technique_id' = :technique_id
            """
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"kb_name": KB_NAME, "technique_id": technique_id},
            )
            row = result.fetchone()
            return dict(row.metadata) if row else None

    async def search_techniques(
        self, query_text: str, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Hybrid vector + keyword search across ATT&CK technique descriptions."""
        stripped = query_text.strip()
        keyword_query = _KEYWORD_QUERY_ALIASES.get(stripped, stripped)
        return await self._store.hybrid_search(
            KB_NAME,
            query_text,
            keyword_query=keyword_query,
            top_k=top_k,
        )
