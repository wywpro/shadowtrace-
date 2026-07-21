"""RAGAgent: knowledge-augmented retrieval across four KBs (ISSUE-046).

Concurrently queries attack_kb, fp_case_kb, history_case_kb, and playbook_kb
via RetrievalPipeline, assembles a structured RAGOutput, and persists it to
EventContext.rag_output.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from app.agents.base import BaseAgent
from app.agents.rag_query_builder import RAGQueryBuilder
from app.core.errors import (
    DependencyUnavailableError,
    GuardrailViolationError,
    ShadowTraceError,
)
from app.models.agent_io import (
    AttackTechniqueMatch,
    Citation,
    FpSimilarity,
    RAGAgentInput,
    RAGOutput,
    SimilarCaseSummary,
)
from app.models.enums import EventType, FinalVerdict
from app.models.knowledge import RetrievalResult

logger = logging.getLogger(__name__)

_KB_NAMES = ["attack_kb", "fp_case_kb", "history_case_kb", "playbook_kb"]
_TOP_K = 5


class RAGAgent(BaseAgent[RAGAgentInput, RAGOutput]):
    """Stage 6 Agent: concurrent RAG retrieval across four knowledge bases.

    Each KB is queried independently; a single KB failure does not interrupt
    the others. When all four KBs are unavailable the agent returns an empty
    RAGOutput with ``degraded=True``.
    """

    agent_name: str = "rag_agent"

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        tool_executor: Any | None = None,
        working_memory: Any | None = None,
        budget_service: Any | None = None,
        output_guard: Any | None = None,
        trace_service: Any | None = None,
        audit_service: Any | None = None,
        event_bus: Any | None = None,
        pipeline: Any | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            tool_executor=tool_executor,
            working_memory=working_memory,
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
            audit_service=audit_service,
            event_bus=event_bus,
        )
        self._pipeline = pipeline

    # ------------------------------------------------------------------ #
    # _run
    # ------------------------------------------------------------------ #

    async def _run(self, input: RAGAgentInput) -> RAGOutput:
        queries = RAGQueryBuilder.build_queries(input.triage_result, input.evidence_output)

        # Concurrent retrieval — one task per KB.
        if self._pipeline is None:
            output = RAGOutput(degraded=True)
            await self._write_rag_output(input, output)
            return output

        tasks: dict[str, asyncio.Task[RetrievalResult | None]] = {}
        for kb_name in _KB_NAMES:
            query = queries.get(kb_name, "")
            tasks[kb_name] = asyncio.create_task(self._retrieve_safe(kb_name, query, top_k=_TOP_K))

        results: dict[str, RetrievalResult | None] = {}
        for kb_name in _KB_NAMES:
            results[kb_name] = await tasks[kb_name]

        # Assemble output sections.
        attack_techniques = _build_attack_techniques(results.get("attack_kb"))
        fp_similarity = _build_fp_similarity(results.get("fp_case_kb"))
        similar_cases = _build_similar_cases(results.get("history_case_kb"))
        playbook_refs = _build_playbook_refs(results.get("playbook_kb"))
        citations = _aggregate_citations(results)

        all_failed = all(r is None for r in results.values())
        output = RAGOutput(
            attack_techniques=attack_techniques,
            fp_similarity=fp_similarity,
            similar_cases=similar_cases,
            playbook_refs=playbook_refs,
            citations=citations,
            degraded=all_failed,
        )

        # Persist to EventContext.
        await self._write_rag_output(input, output)

        return output

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _retrieve_safe(
        self, kb_name: str, query: str, top_k: int = 5
    ) -> RetrievalResult | None:
        """Call pipeline.retrieve, returning None on failure."""
        if self._pipeline is None:
            return None
        try:
            return await self._pipeline.retrieve(query, [kb_name], top_k=top_k)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed for kb=%s query=%.100s: %s",
                kb_name,
                query,
                exc,
            )
            return None

    async def _write_rag_output(self, input: RAGAgentInput, output: RAGOutput) -> None:
        """Persist ``rag_output`` to ``EventContext``."""
        wm = self.working_memory
        if wm is None:
            return
        try:
            await wm.write(
                input.event_id,
                "rag_output",
                output.model_dump(mode="json"),
            )
        except GuardrailViolationError:
            logger.exception(
                "GuardrailViolationError writing rag_output for event=%s",
                input.event_id,
            )
            raise
        except (DependencyUnavailableError, ConnectionError, TimeoutError):
            logger.warning(
                "Transient failure writing rag_output for event=%s",
                input.event_id,
                exc_info=True,
            )
            output.degraded = True
            await self._try_persist_degraded_flag(input.event_id)
        except ShadowTraceError as exc:
            if exc.retryable:
                logger.warning(
                    "Retryable error writing rag_output for event=%s: %s",
                    input.event_id,
                    exc.error_code,
                    exc_info=True,
                )
                output.degraded = True
                await self._try_persist_degraded_flag(input.event_id)
            else:
                raise

    async def _try_persist_degraded_flag(self, event_id: str) -> None:
        wm = self.working_memory
        if wm is None:
            return
        try:
            await wm.write(
                event_id,
                "degraded_flags",
                {
                    "degraded": True,
                    "reason": "rag_output persistence failed",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except ShadowTraceError:
            logger.exception(
                "Failed to persist degraded_flags for event=%s",
                event_id,
            )


# --------------------------------------------------------------------------- #
# Result assembly helpers
# --------------------------------------------------------------------------- #


def _build_attack_techniques(
    result: RetrievalResult | None,
) -> list[AttackTechniqueMatch]:
    """Extract attack technique matches from attack_kb retrieval result.

    Only techniques with a reranked score >= 0.3 are kept.  match_confidence
    is the chunk score clipped to [0, 1].
    """
    if result is None or not result.chunks:
        return []

    citation_by_chunk = {c.chunk_id: c.citation_id for c in result.citations}

    techniques: list[AttackTechniqueMatch] = []
    for chunk in result.chunks:
        score = max(0.0, min(1.0, chunk.score))
        if score < 0.3:
            continue
        meta = chunk.metadata
        technique_id = meta.get("technique_id", "")
        technique_name = meta.get("technique_name", "")
        tactics: list[str] = meta.get("tactics", [])
        if not isinstance(tactics, list):
            tactics = []
        citation_id = citation_by_chunk.get(chunk.chunk_id, "")
        techniques.append(
            AttackTechniqueMatch(
                technique_id=technique_id,
                technique_name=technique_name,
                tactics=tactics,
                match_confidence=score,
                citation_id=citation_id,
            )
        )

    # Sort by confidence descending, deduplicate by technique_id.
    seen: set[str] = set()
    deduped: list[AttackTechniqueMatch] = []
    for t in sorted(techniques, key=lambda x: x.match_confidence, reverse=True):
        if t.technique_id not in seen:
            seen.add(t.technique_id)
            deduped.append(t)
    return deduped


def _build_fp_similarity(result: RetrievalResult | None) -> FpSimilarity:
    """Compute false-positive similarity from fp_case_kb retrieval result."""
    if result is None or not result.chunks:
        return FpSimilarity(max_score=0.0)

    best = max(result.chunks, key=lambda c: c.score)
    max_score = max(0.0, min(1.0, best.score))
    meta = best.metadata
    return FpSimilarity(
        max_score=max_score,
        matched_case_id=meta.get("case_id"),
        matched_pattern=meta.get("pattern_summary"),
    )


def _build_similar_cases(
    result: RetrievalResult | None,
) -> list[SimilarCaseSummary]:
    """Extract similar case summaries from history_case_kb retrieval result."""
    if result is None or not result.chunks:
        return []

    cases: list[SimilarCaseSummary] = []
    for chunk in result.chunks:
        meta = chunk.metadata
        event_type_raw = meta.get("event_type")
        event_type: EventType | None = None
        if isinstance(event_type_raw, str):
            try:
                event_type = EventType(event_type_raw)
            except ValueError:
                pass

        verdict_raw = meta.get("final_verdict")
        final_verdict: FinalVerdict | None = None
        if isinstance(verdict_raw, str):
            try:
                final_verdict = FinalVerdict(verdict_raw)
            except ValueError:
                pass

        risk_score_raw = meta.get("risk_score")
        risk_score: int | None = None
        if isinstance(risk_score_raw, (int, float)):
            risk_score = max(0, min(100, int(risk_score_raw)))

        cases.append(
            SimilarCaseSummary(
                case_id=meta.get("case_id", ""),
                event_type=event_type,
                summary=meta.get("summary", ""),
                final_verdict=final_verdict,
                risk_score=risk_score,
                score=max(0.0, min(1.0, chunk.score)),
            )
        )

    return cases


def _build_playbook_refs(result: RetrievalResult | None) -> list[str]:
    """Extract playbook IDs from playbook_kb retrieval result."""
    if result is None or not result.chunks:
        return []

    seen: set[str] = set()
    refs: list[str] = []
    for chunk in result.chunks:
        pb_id = chunk.metadata.get("playbook_id", "")
        if pb_id and pb_id not in seen:
            seen.add(pb_id)
            refs.append(pb_id)
    return refs


def _aggregate_citations(
    results: dict[str, RetrievalResult | None],
) -> list[Citation]:
    """Collect and deduplicate citations across all four KB results."""
    seen: set[str] = set()
    aggregated: list[Citation] = []
    for result in results.values():
        if result is None:
            continue
        for c in result.citations:
            if c.citation_id in seen:
                continue
            seen.add(c.citation_id)
            aggregated.append(
                Citation(
                    citation_id=c.citation_id,
                    chunk_id=c.chunk_id,
                    kb_name=c.kb_name,
                    quoted_text=c.quoted_text,
                    relevance_score=max(0.0, min(1.0, c.relevance_score)),
                )
            )
    return aggregated
