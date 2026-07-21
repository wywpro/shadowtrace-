"""Load fp_cases.json and history_cases.json into KnowledgeStore (ISSUE-043).

Idempotent — safe to run multiple times.  Usage:

    cd backend && python -m scripts.load_case_kb
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.config import Settings  # noqa: E402
from app.core.embedding.service import EmbeddingService  # noqa: E402
from app.models.case import (  # noqa: E402
    FalsePositiveCase,
    HistoryCase,
    fp_case_metadata,
    fp_case_to_text,
    history_case_metadata,
    history_case_to_text,
    make_chunk_id,
)
from app.models.knowledge import KnowledgeChunk  # noqa: E402
from app.services.knowledge_store import KnowledgeStore  # noqa: E402

ROOT_DIR = _BACKEND.parent
DATA_DIR = ROOT_DIR / "data" / "knowledge"

FP_CASES_FILE = DATA_DIR / "fp_cases.json"
HISTORY_CASES_FILE = DATA_DIR / "history_cases.json"

FP_KB_NAME = "fp_case_kb"
HISTORY_KB_NAME = "history_case_kb"

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)


def _load_json(path: Path) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


async def _upsert_fp_cases(store: KnowledgeStore, cases: list[FalsePositiveCase]) -> int:
    chunks: list[KnowledgeChunk] = []
    for case in cases:
        content = fp_case_to_text(case)
        chunk_id = make_chunk_id(FP_KB_NAME, case.case_id)
        metadata = fp_case_metadata(case)
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                kb_name=FP_KB_NAME,
                content=content,
                metadata=metadata,
            )
        )
    await store.upsert_chunks(FP_KB_NAME, chunks)
    return len(chunks)


async def _upsert_history_cases(store: KnowledgeStore, cases: list[HistoryCase]) -> int:
    chunks: list[KnowledgeChunk] = []
    for case in cases:
        content = history_case_to_text(case)
        chunk_id = make_chunk_id(HISTORY_KB_NAME, case.case_id)
        metadata = history_case_metadata(case)
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                kb_name=HISTORY_KB_NAME,
                content=content,
                metadata=metadata,
            )
        )
    await store.upsert_chunks(HISTORY_KB_NAME, chunks)
    return len(chunks)


async def main() -> None:
    settings = Settings()
    engine = create_async_engine(
        DATABASE_URL,
        poolclass=NullPool,
    )
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    embed_service = EmbeddingService(settings)
    store = KnowledgeStore(session_factory, embed_service)

    try:
        # --- FP cases ---
        if FP_CASES_FILE.exists():
            raw = _load_json(FP_CASES_FILE)
            fp_cases = [FalsePositiveCase.model_validate(r) for r in raw]
            count = await _upsert_fp_cases(store, fp_cases)
            print(f"[load_case_kb] upserted {count} FP cases into {FP_KB_NAME}")
        else:
            print(f"[load_case_kb] WARNING: {FP_CASES_FILE} not found, skipping FP cases")

        # --- History cases ---
        if HISTORY_CASES_FILE.exists():
            raw = _load_json(HISTORY_CASES_FILE)
            history_cases = [HistoryCase.model_validate(r) for r in raw]
            count = await _upsert_history_cases(store, history_cases)
            print(f"[load_case_kb] upserted {count} history cases into {HISTORY_KB_NAME}")
        else:
            print(f"[load_case_kb] WARNING: {HISTORY_CASES_FILE} not found, skipping history cases")

        # --- Verify ---
        fp_total = await store.count(FP_KB_NAME)
        hist_total = await store.count(HISTORY_KB_NAME)
        print(f"[load_case_kb] fp_case_kb total chunks: {fp_total}")
        print(f"[load_case_kb] history_case_kb total chunks: {hist_total}")

        if fp_total < 10:
            print(
                f"[load_case_kb] ERROR: fp_case_kb has {fp_total} chunks, expected >= 10",
                file=sys.stderr,
            )
            sys.exit(1)
        if hist_total < 16:
            print(
                f"[load_case_kb] ERROR: history_case_kb has {hist_total} chunks, expected >= 16",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        await embed_service.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
