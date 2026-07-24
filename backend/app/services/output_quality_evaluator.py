"""OutputQualityEvaluator: rule-based + optional LLM-judge quality scoring (ISSUE-065).

Evaluates key agent outputs (triage_result, evidence_output, risk_assessment,
report) across four weighted rule metrics, with an optional LLM-judge
calibration path gated by ``QUALITY_JUDGE_ENABLED``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from app.agents.prompts.quality_judge_prompt import build_quality_judge_messages
from app.models.agent_io import OutputQualityScore
from app.models.enums import QualityVerdict

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Fixed evaluation targets (ISSUE-065 spec)
# --------------------------------------------------------------------------- #

_EVALUATED_AGENTS: tuple[str, ...] = (
    "triage",
    "evidence",
    "risk",
    "report",
)

# Metric names as defined in the spec
_METRIC_NAMES: tuple[str, ...] = (
    "completeness",
    "grounding_ratio",
    "consistency",
    "specificity",
)

# Weights (must sum to 1.0)
_METRIC_WEIGHTS: dict[str, float] = {
    "completeness": 0.30,
    "grounding_ratio": 0.30,
    "consistency": 0.25,
    "specificity": 0.15,
}

# Thresholds
_PASS_THRESHOLD: float = 0.75
_WARN_THRESHOLD: float = 0.50
# score < 0.50 → fail


def _verdict_from_score(score: float) -> QualityVerdict:
    if score >= _PASS_THRESHOLD:
        return QualityVerdict.PASS
    if score >= _WARN_THRESHOLD:
        return QualityVerdict.WARN
    return QualityVerdict.FAIL


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class OutputQualityEvaluator:
    """Rule-based output quality evaluator with optional LLM-judge calibration.

    Usage::

        evaluator = OutputQualityEvaluator(llm_client=client)
        # Evaluate all four target agents
        scores = await evaluator.evaluate_all(event_context)
        # Or evaluate a single agent output
        score = await evaluator.evaluate("triage", triage_output, event_context)
    """

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        judge_enabled: bool = False,
        working_memory: Any | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._judge_enabled = judge_enabled and llm_client is not None
        if working_memory is not None:
            self._bound_wm = working_memory.for_writer("OutputQualityEvaluator")
        else:
            self._bound_wm = None
        self.last_degraded_reason: str | None = None

    # ------------------------------------------------------------------ #
    # evaluate_all
    # ------------------------------------------------------------------ #

    async def evaluate_all(self, event_context: dict[str, Any]) -> dict[str, OutputQualityScore]:
        """Evaluate all four target agent outputs from the event context.

        Returns a dict mapping agent_name → OutputQualityScore.  When
        ``working_memory`` was provided at construction time and ``event_id`` is
        present in *event_context*, results are also persisted to
        ``quality_scores``.
        """
        results: dict[str, OutputQualityScore] = {}
        for agent_name in _EVALUATED_AGENTS:
            output_key = _output_key_for_agent(agent_name)
            output = event_context.get(output_key)
            if output is None:
                logger.debug(
                    "Skipping quality eval for agent=%s — no output key=%s",
                    agent_name,
                    output_key,
                )
                continue
            try:
                results[agent_name] = await self.evaluate(agent_name, output, event_context)
            except Exception as exc:
                logger.warning(
                    "Quality eval failed for agent=%s: %s",
                    agent_name,
                    exc,
                )
                self.last_degraded_reason = f"eval_failed_{agent_name}: {exc}"
                # Fail-safe: pass with minimal score so the main pipeline
                # is never blocked by a broken evaluator.
                results[agent_name] = OutputQualityScore(
                    agent_name=agent_name,
                    score=0.75,
                    verdict=QualityVerdict.PASS,
                    metrics={m: 0.75 for m in _METRIC_NAMES},
                    reasons=[f"eval_error_defaulted: {exc}"],
                    evaluated_by="rule",
                )
        await self._persist_quality_scores(event_context, results)
        return results

    async def _persist_quality_scores(
        self,
        event_context: dict[str, Any],
        results: dict[str, OutputQualityScore],
    ) -> None:
        """Write agent_name → score dict to WorkingMemory when bound (ISSUE-065 §4)."""
        if self._bound_wm is None or not results:
            return
        event_id = event_context.get("event_id")
        if not event_id:
            return
        payload = {
            agent_name: score.model_dump(mode="json") for agent_name, score in results.items()
        }
        try:
            await self._bound_wm.write(str(event_id), "quality_scores", payload)
        except Exception:
            logger.warning(
                "Failed to persist quality_scores for event=%s",
                event_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # evaluate — single agent
    # ------------------------------------------------------------------ #

    async def evaluate(
        self,
        agent_name: str,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> OutputQualityScore:
        """Evaluate a single agent output.

        Returns ``OutputQualityScore`` with rule metrics.  When
        ``QUALITY_JUDGE_ENABLED`` is true and an LLM client is available,
        the rule score is averaged with the LLM judge's independent score.
        """
        # --- 1. Rule path ---
        output_dict = _to_dict(output)
        metrics = _compute_rule_metrics(agent_name, output_dict, context or {})
        weighted_score = _weighted_score(metrics)
        reasons = _build_reasons(metrics)

        verdict = _verdict_from_score(weighted_score)
        evaluated_by: Literal["rule", "llm"] = "rule"

        # --- 2. Optional LLM-judge calibration ---
        if self._judge_enabled and self._llm_client is not None:
            try:
                judge_score = await self._llm_judge(
                    agent_name=agent_name,
                    output_dict=output_dict,
                    context=context or {},
                )
                if judge_score is not None:
                    weighted_score = round((weighted_score + judge_score) / 2.0, 4)
                    verdict = _verdict_from_score(weighted_score)
                    evaluated_by = "llm"
                    reasons.append(
                        f"llm_judge_calibrated: rule={_weighted_score(metrics):.3f} "
                        f"judge={judge_score:.3f} final={weighted_score:.3f}"
                    )
            except Exception as exc:
                logger.warning(
                    "LLM judge failed for agent=%s, using rule-only score: %s",
                    agent_name,
                    exc,
                )
                self.last_degraded_reason = f"llm_judge_failed_{agent_name}: {exc}"

        return OutputQualityScore(
            agent_name=agent_name,
            score=weighted_score,
            verdict=verdict,
            metrics=metrics,
            reasons=reasons,
            evaluated_by=evaluated_by,
        )

    # ------------------------------------------------------------------ #
    # LLM judge
    # ------------------------------------------------------------------ #

    async def _llm_judge(
        self,
        *,
        agent_name: str,
        output_dict: dict[str, Any],
        context: dict[str, Any],
    ) -> float | None:
        """Call the LLM quality judge, returning its 0-1 score or None on failure."""
        if self._llm_client is None:
            return None

        messages = build_quality_judge_messages(
            agent_name=agent_name,
            output_summary=_summarize_for_judge(output_dict),
            context=context,
        )
        response = await self._llm_client.chat(
            messages,
            event_id=context.get("event_id", "unknown"),
            agent_name="quality_judge",
            prompt_key="quality_judge",
            json_mode=True,
        )

        try:
            llm_data: Any = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(llm_data, dict):
            return None

        raw_score = llm_data.get("score")
        if raw_score is None:
            return None

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return None

        return max(0.0, min(1.0, score))


# ====================================================================== #
# Rule metrics
# ====================================================================== #


def _output_key_for_agent(agent_name: str) -> str:
    """Map agent short name to event_context key."""
    _KEY_MAP: dict[str, str] = {
        "triage": "triage_result",
        "evidence": "evidence_output",
        "risk": "risk_assessment",
        "report": "report",
    }
    return _KEY_MAP.get(agent_name, f"{agent_name}_output")


def _to_dict(output: Any) -> dict[str, Any]:
    """Convert a Pydantic model or dict to a plain dict for inspection."""
    if isinstance(output, dict):
        return output
    if hasattr(output, "model_dump"):
        return output.model_dump(mode="json")  # type: ignore[no-any-return]
    if hasattr(output, "dict"):
        return output.dict()  # type: ignore[no-any-return]
    return {"value": str(output)}


# ---- completeness (0.30) ----


def _completeness(agent_name: str, output: dict[str, Any]) -> float:
    """Fraction of required fields that are non-null / non-empty."""
    required = _REQUIRED_FIELDS.get(agent_name, [])
    if not required:
        return 1.0
    filled = 0
    for field in required:
        val = output.get(field)
        if val is not None and val != "" and val != []:
            filled += 1
    return round(filled / len(required), 4)


_REQUIRED_FIELDS: dict[str, list[str]] = {
    "triage": [
        "event_type",
        "severity",
        "need_investigation",
        "reasoning",
    ],
    "evidence": [
        "evidence_list",
        "collection_status",
        "overall_confidence",
    ],
    "risk": [
        "risk_score",
        "severity",
        "confidence",
        "risk_factors",
        "scoring_mode",
    ],
    "report": [
        "title",
        "summary",
        "findings",
        "final_verdict",
    ],
}


# ---- grounding_ratio (0.30) ----


def _grounding_ratio(agent_name: str, output: dict[str, Any]) -> float:
    """Ratio of assertions that reference evidence / citations.

    For evidence: fraction of items with non-empty evidence_id.
    For risk: always 1.0 if factors are present (grounded by upstream data).
    For report: fraction of citations relative to total findings.
    """
    if agent_name == "evidence":
        ev_list = output.get("evidence_list")
        if not isinstance(ev_list, list) or len(ev_list) == 0:
            return 0.0
        grounded = sum(1 for e in ev_list if isinstance(e, dict) and e.get("evidence_id", "") != "")
        return round(grounded / len(ev_list), 4)

    if agent_name == "risk":
        factors = output.get("risk_factors")
        if isinstance(factors, list) and len(factors) > 0:
            # Each factor must have a non-empty reasoning
            reasoned = sum(
                1
                for f in factors
                if isinstance(f, dict) and str(f.get("reasoning", "")).strip() != ""
            )
            return round(reasoned / len(factors), 4)
        return 1.0  # No factors → no grounding gap to measure

    if agent_name == "report":
        findings = output.get("findings")
        if not isinstance(findings, list) or len(findings) == 0:
            return 0.0
        referenced = sum(
            1
            for f in findings
            if isinstance(f, dict) and (f.get("evidence_id") or f.get("citation"))
        )
        return round(referenced / len(findings), 4)

    # triage: check reasoning references entities or IOCs
    if agent_name == "triage":
        reasoning = str(output.get("reasoning", ""))
        iocs = output.get("ioc_list")
        entities = output.get("entities")
        has_entities = (isinstance(entities, dict) and len(entities) > 0) or (
            isinstance(iocs, list) and len(iocs) > 0
        )
        if has_entities and len(reasoning) > 20:
            return 1.0
        if len(reasoning) > 50:
            return 0.7
        return 0.3

    return 1.0


# ---- consistency (0.25) ----


def _consistency(agent_name: str, output: dict[str, Any]) -> float:
    """Cross-field consistency checks.

    - Triage: severity matches event_type and need_investigation.
    - Evidence: overall_confidence matches collection_status.
    - Risk: severity matches risk_score band.
    - Report: final_verdict matches risk severity direction.
    """
    if agent_name == "triage":
        score = 1.0
        sev = str(output.get("severity", "")).lower()
        evt = str(output.get("event_type", "")).lower()
        need = output.get("need_investigation")

        # High/critical severity should require investigation
        if sev in ("high", "critical") and need is False:
            score -= 0.3
        elif sev == "low" and need is True:
            score -= 0.1

        # Certain event types imply non-trivial severity
        _hi_events = {"host_compromise", "data_exfiltration", "lateral_movement"}
        if evt in _hi_events and sev == "low":
            score -= 0.2

        return max(0.0, score)

    if agent_name == "evidence":
        score = 1.0
        status = str(output.get("collection_status", "")).lower()
        confidence = float(output.get("overall_confidence", 0))

        if status == "failed" and confidence > 0.3:
            score -= 0.3
        if status in ("completed",) and confidence < 0.5:
            score -= 0.3

        return max(0.0, score)

    if agent_name == "risk":
        raw = output.get("risk_score")
        try:
            rs = int(raw) if raw is not None else 50
        except (TypeError, ValueError):
            rs = 50
        sev = str(output.get("severity", "")).lower()

        expected_sev_band = _severity_for_risk_score(rs)
        if sev != expected_sev_band:
            return 0.4  # Major inconsistency
        return 1.0

    if agent_name == "report":
        verdict = str(output.get("final_verdict", "")).lower()
        summary = str(output.get("summary", "")).lower()

        # If summary contains threat language but verdict is false_positive → inconsistency
        threat_words = ("confirmed", "threat", "attack", "malicious", "compromise")
        fp_words = ("false_positive", "false positive", "benign", "误报")

        has_threat = any(w in summary for w in threat_words)
        has_fp = any(w in summary for w in fp_words)
        is_fp_verdict = verdict in ("false_positive", "possible_false_positive")

        if has_threat and is_fp_verdict:
            return 0.5
        if has_fp and not is_fp_verdict:
            return 0.5
        return 1.0

    return 1.0


def _severity_for_risk_score(score: int) -> str:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


# ---- specificity (0.15) ----


def _specificity(agent_name: str, output: dict[str, Any]) -> float:
    """Penalize vague / boilerplate descriptions.

    Checks whether the output contains concrete entity references
    (IPs, hostnames, usernames, file paths) rather than generic phrasing.
    """
    import re

    text = _extract_text(agent_name, output)
    if not text:
        return 0.0

    # Count concrete indicators
    ip_count = len(re.findall(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", text))
    host_count = len(re.findall(r"\b[A-Z]{2,6}-\w+-\d{2,4}\b", text))
    file_count = len(re.findall(r"\b\w+\.(zip|exe|rar|7z|pdf|doc|xls|ps1|sh|bat)\b", text))
    user_count = len(re.findall(r"(?:user|account|username|账号)\S*\s+['\"]?\w+['\"]?", text))
    hash_count = len(re.findall(r"\b[a-f0-9]{32,64}\b", text))

    concrete_count = ip_count + host_count + file_count + user_count + hash_count

    # Vague boilerplate phrases
    vague_phrases = [
        "further investigation needed",
        "unable to determine",
        "insufficient data",
        "no actionable information",
        "需要进一步调查",
        "无法确定",
        "数据不足",
    ]
    vague_count = sum(1 for p in vague_phrases if p in text.lower())

    # Score: presence of concrete entities raises it; vagueness lowers it
    if concrete_count >= 5:
        return 1.0
    if concrete_count >= 2:
        base = 0.8
    elif concrete_count >= 1:
        base = 0.5
    else:
        base = 0.1

    return max(0.0, base - vague_count * 0.15)


def _extract_text(agent_name: str, output: dict[str, Any]) -> str:
    """Extract the main narrative text from an agent output."""
    if agent_name == "triage":
        return str(output.get("reasoning", ""))
    if agent_name == "evidence":
        pieces: list[str] = []
        for ev in output.get("evidence_list") or []:
            if isinstance(ev, dict):
                pieces.append(ev.get("description", ""))
                pieces.append(ev.get("evidence_type", ""))
        return " ".join(pieces)
    if agent_name == "risk":
        risk_pieces: list[str] = []
        for f in output.get("risk_factors") or []:
            if isinstance(f, dict):
                risk_pieces.append(f.get("reasoning", ""))
        return " ".join(risk_pieces)
    if agent_name == "report":
        return str(output.get("summary", "")) + " " + str(output.get("narrative", ""))
    return ""


# ---- composite ----


_METRIC_FUNCS: dict[str, Any] = {
    "completeness": _completeness,
    "grounding_ratio": _grounding_ratio,
    "consistency": _consistency,
    "specificity": _specificity,
}


def _compute_rule_metrics(
    agent_name: str,
    output: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, float]:
    """Compute all four rule metrics for an agent output."""
    metrics: dict[str, float] = {}
    for name in _METRIC_NAMES:
        try:
            metric_val = _METRIC_FUNCS[name](agent_name, output)
        except Exception:
            metric_val = 0.0
        metrics[name] = round(float(metric_val), 4)
    return metrics


def _weighted_score(metrics: dict[str, float]) -> float:
    """Compute the weighted score from metric sub-scores."""
    total = 0.0
    for name in _METRIC_NAMES:
        total += metrics.get(name, 0.0) * _METRIC_WEIGHTS.get(name, 0.0)
    return round(total, 4)


def _build_reasons(metrics: dict[str, float]) -> list[str]:
    """Build human-readable reason strings from metric scores."""
    reasons: list[str] = []
    for name in _METRIC_NAMES:
        val = metrics.get(name, 0.0)
        if val >= 0.8:
            reasons.append(f"{name}: {val:.2f} — good")
        elif val >= 0.5:
            reasons.append(f"{name}: {val:.2f} — fair")
        else:
            reasons.append(f"{name}: {val:.2f} — low")
    return reasons


def _summarize_for_judge(output: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Create a compact summary of agent output for the LLM judge.

    Truncates long fields so the judge prompt stays within budget.
    """
    if depth >= 5:  # Safety limit — prevent infinite recursion on circular refs
        return {"_truncated": "max_depth_reached"}

    summary: dict[str, Any] = {}
    for key, value in output.items():
        if isinstance(value, str):
            summary[key] = value[:500]
        elif isinstance(value, list):
            if len(value) > 5:
                summary[key] = [
                    (
                        _summarize_for_judge(item, depth=depth + 1)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value[:5]
                ]
                summary[f"{key}_truncated"] = f"{len(value) - 5} items omitted"
            else:
                summary[key] = value
        elif isinstance(value, dict):
            summary[key] = _summarize_for_judge(value, depth=depth + 1)
        else:
            summary[key] = value
    return summary
