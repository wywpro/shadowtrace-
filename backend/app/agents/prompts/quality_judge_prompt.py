"""Quality judge prompt builder (ISSUE-065).

Used by OutputQualityEvaluator when QUALITY_JUDGE_ENABLED=true and an LLM
client is available. The judge calibrates the rule-based score by providing
an independent assessment, and the final score is the mean of rule + judge.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.llm.base import LLMMessage

_QUALITY_JUDGE_SYSTEM = (
    "You are ShadowTrace QualityJudge. Rate the quality of one security "
    "agent output across four dimensions. Reply with JSON only. Do not "
    "include hidden chain-of-thought."
)

_FOUR_METRICS: tuple[str, ...] = (
    "completeness",
    "grounding_ratio",
    "consistency",
    "specificity",
)


def build_quality_judge_messages(
    *,
    agent_name: str,
    output_summary: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[LLMMessage]:
    """Build JSON-mode messages for LLM quality judging.

    The LLM is asked to score the same four metrics independently so its
    score can be averaged with the rule score.
    """
    payload: dict[str, Any] = {
        "agent_name": agent_name,
        "output": output_summary,
        "context": context or {},
        "metrics": list(_FOUR_METRICS),
        "response_schema": {
            "score": "0-1",
            "metrics": {
                "<metric_name>": "0-1",
            },
            "reasons": ["short string"],
        },
    }
    user = (
        "Score this agent output on four dimensions (0-1 per metric). "
        "Return JSON: "
        '{"score":0.85,"metrics":{"completeness":0.9,"grounding_ratio":0.8,'
        '"consistency":0.85,"specificity":0.85},'
        '"reasons":["one sentence per metric"]}\n'
        f"Details:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [
        LLMMessage(role="system", content=_QUALITY_JUDGE_SYSTEM),
        LLMMessage(role="user", content=user),
    ]
