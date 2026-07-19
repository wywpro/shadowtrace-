"""Load ATT&CK techniques into the attack_kb knowledge base (ISSUE-042).

Usage::

    cd backend && python -m scripts.load_attack_kb

The script is idempotent — repeated runs produce the same chunk set.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Ensure the backend package root is on sys.path for ``from app.…`` imports.
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.config import Settings  # noqa: E402
from app.core.embedding.service import EmbeddingService  # noqa: E402
from app.services.attack_kb_service import AttackKBService  # noqa: E402
from app.services.knowledge_store import KnowledgeStore  # noqa: E402

REPO_ROOT = _BACKEND.parent
DATA_FILE = REPO_ROOT / "data" / "knowledge" / "attack_techniques.json"

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)


async def _main() -> None:
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        sys.exit(1)

    settings = Settings()
    engine = create_async_engine(DATABASE_URL)
    session_factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, autoflush=False
    )
    embed_service = EmbeddingService(settings)
    store = KnowledgeStore(session_factory, embed_service)
    service = AttackKBService(store, session_factory)

    try:
        count = await service.load_from_file(DATA_FILE)
        print(f"Loaded {count} ATT&CK techniques into attack_kb")
    finally:
        await embed_service.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
