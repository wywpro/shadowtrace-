"""Tests for AttackKBService: load, get_technique, search, idempotency (ISSUE-042)."""

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
from app.services.attack_kb_service import AttackKBService, KB_NAME
from app.services.knowledge_store import KnowledgeStore

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
DATA_FILE = REPO_ROOT / "data" / "knowledge" / "attack_techniques.json"

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return cfg


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


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
def embed_service() -> EmbeddingService:
    return EmbeddingService(Settings(embedding_mode="mock"))


@pytest_asyncio.fixture
def store(
    session_factory: async_sessionmaker[AsyncSession],
    embed_service: EmbeddingService,
) -> KnowledgeStore:
    return KnowledgeStore(session_factory, embed_service)


@pytest_asyncio.fixture
def service(
    store: KnowledgeStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> AttackKBService:
    return AttackKBService(store, session_factory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _clean(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async with session_factory() as session:
        await session.execute(text("DELETE FROM knowledge_chunk"))
        await session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    @pytest.mark.asyncio
    async def test_loads_at_least_60_techniques(
        self,
        service: AttackKBService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        await _clean(session_factory)
        count = await service.load_from_file(DATA_FILE)
        assert count >= 60

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(
        self,
        service: AttackKBService,
        store: KnowledgeStore,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        await _clean(session_factory)
        first = await service.load_from_file(DATA_FILE)
        second = await service.load_from_file(DATA_FILE)
        assert first == second
        assert await store.count(KB_NAME) == first

    @pytest.mark.asyncio
    async def test_missing_file_raises(
        self,
        service: AttackKBService,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            await service.load_from_file("/nonexistent/path.json")


class TestGetTechnique:
    @pytest.mark.asyncio
    async def test_t1078_returns_full_entry(
        self,
        service: AttackKBService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        await _clean(session_factory)
        await service.load_from_file(DATA_FILE)

        result = await service.get_technique("T1078")
        assert result is not None
        assert result["technique_id"] == "T1078"
        assert result["technique_name"] == "Valid Accounts"
        assert "Defense Evasion" in result["tactics"]
        assert result["attack_version"] == "v15.1"
        assert len(result["description"]) > 0
        assert len(result["detection"]) > 0

    @pytest.mark.asyncio
    async def test_unknown_technique_returns_none(
        self,
        service: AttackKBService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        await _clean(session_factory)
        await service.load_from_file(DATA_FILE)

        result = await service.get_technique("T9999")
        assert result is None


class TestSearchTechniques:
    @pytest.mark.asyncio
    async def test_vector_search_ranks_exact_content_highest(
        self,
        service: AttackKBService,
        store: KnowledgeStore,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """With mock (deterministic) embeddings, same text → score near 1.0."""
        await _clean(session_factory)
        await service.load_from_file(DATA_FILE)

        # Retrieve the exact stored content for T1078 to guarantee mock-embedder match
        t1078 = await service.get_technique("T1078")
        assert t1078 is not None
        from app.services.attack_kb_service import _format_content

        query_text = _format_content(t1078)
        results = await service.search_techniques(query_text, top_k=3)
        assert len(results) >= 1
        assert results[0].retrieval_method in {"vector", "hybrid"}
        assert results[0].score > 0.9

    @pytest.mark.asyncio
    async def test_search_数据外泄_hits_exfiltration_via_hybrid(
        self,
        service: AttackKBService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """ISSUE-042: hybrid search maps 数据外泄 → exfiltration under mock embeddings."""
        await _clean(session_factory)
        await service.load_from_file(DATA_FILE)

        results = await service.search_techniques("数据外泄", top_k=5)
        assert len(results) >= 1
        assert any(
            "Exfiltration" in (r.metadata.get("tactics") or []) for r in results
        )
        assert any(r.retrieval_method in {"keyword", "hybrid"} for r in results)

    @pytest.mark.asyncio
    async def test_respects_top_k(
        self,
        service: AttackKBService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        await _clean(session_factory)
        await service.load_from_file(DATA_FILE)

        results = await service.search_techniques("lateral movement", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_empty_kb_returns_empty(
        self,
        service: AttackKBService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        await _clean(session_factory)
        results = await service.search_techniques("anything", top_k=5)
        assert results == []
