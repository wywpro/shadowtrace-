"""QueryRewriter: LLM-driven query rewriting for RAG retrieval (ISSUE-045)."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.core.errors import LLMError
from app.core.llm.base import BaseLLMClient, LLMMessage

logger = logging.getLogger(__name__)


class QueryRewriteError(Exception):
    """Rewrite failed; ``RetrievalPipeline`` should fall back to the original query."""


class QueryRewriteOutput(BaseModel):
    rewrites: list[str] = Field(default_factory=list, max_length=2)


class QueryRewriter:
    """Rewrite a query into up to 2 variants using LLM JSON mode.

    On failure (timeout, LLM error, invalid JSON) raises :class:`QueryRewriteError`
    so the pipeline can record ``query_rewriter`` in ``degraded_steps`` and fall
    back to the original query.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        event_id: str = "rag-pipeline",
        agent_name: str = "RAGAgent",
    ) -> None:
        self._llm = llm_client
        self._event_id = event_id
        self._agent_name = agent_name

    async def rewrite(self, query: str) -> list[str]:
        """Return original query plus up to 2 rewritten variants."""
        try:
            response = await self._llm.chat(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a search query rewriter. Generate up to 2 alternative "
                            "search queries that express the same information need using "
                            "different keywords, terminology, or perspectives."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=f"Rewrite this query for hybrid search: {query}",
                    ),
                ],
                event_id=self._event_id,
                agent_name=self._agent_name,
                prompt_key="query_rewrite",
                temperature=0.3,
                max_tokens=256,
                json_mode=True,
                response_model=QueryRewriteOutput,
                timeout=15.0,
            )
            parsed = response.parsed
            if isinstance(parsed, QueryRewriteOutput) and parsed.rewrites:
                rewrites = [r for r in parsed.rewrites if r.strip() and r.strip() != query.strip()]
                return [query, *rewrites[:2]]
            if isinstance(parsed, QueryRewriteOutput):
                return [query]
            raise QueryRewriteError(
                "query_rewrite returned no structured QueryRewriteOutput payload"
            )
        except QueryRewriteError:
            raise
        except (LLMError, ValueError, OSError) as exc:
            logger.warning("Query rewrite failed, using original query: %s", exc)
            raise QueryRewriteError(str(exc)) from exc
