"""Prompt templates for PlannerAgent (ISSUE-049)."""

from __future__ import annotations

import json

from app.agents.evidence_agent import EVIDENCE_QUERY_ORDER
from app.models.agent_io import ExecutionPlan, TriageResult

_CANONICAL_EVIDENCE_TOOLS = ", ".join(EVIDENCE_QUERY_ORDER)

# --------------------------------------------------------------------------- #
# Plan generation prompt
# --------------------------------------------------------------------------- #

PLAN_GENERATE_SYSTEM = f"""\
You are a security investigation planner. Given a triage result, produce a structured
investigation plan as JSON. The plan must include concrete steps, each assigned to a
specific agent and listing required tools by their canonical names.

Available agents and the tools they can use:

- evidence_agent: {_CANONICAL_EVIDENCE_TOOLS}

- risk_agent: (no tools — uses evidence output directly)

- response_agent: (no tools — generates disposition plan)

- rag_agent: (no tools — uses RetrievalPipeline; only if P1 RAG enabled)

- graph_agent: (no tools — uses evidence output directly)

Output a JSON object with these fields:
- plan_id: "pln-{{8 hex chars}}" (generate a random 8-char hex)
- event_id: the event_id from the input
- steps: list of {{ step_order, step_goal, assigned_agent, required_tools, success_criteria }}
- budget: {{ max_tool_calls: 30, max_llm_calls: 20, max_duration_s: 300 }}
- revision: 0
- revise_reason: null
- degraded: false

Rules:
1. Always include evidence_agent steps first to collect evidence.
2. Always include a risk_agent step for scoring.
3. Always include a response_agent step for disposition planning.
4. Only include rag_agent or graph_agent steps if the triage event_type strongly suggests
   ATT&CK mapping (e.g. malicious_process, lateral_movement) or entity relationship analysis.
5. Every step must have assigned_agent set to one of the valid agent names listed above.
6. Every tool in required_tools must be from the canonical evidence_agent list above.
7. Plan must have at least 4 steps.
"""

PLAN_GENERATE_USER = """\
Event ID: {event_id}

Triage result:
{triage_json}

Generate the investigation plan as JSON only (no extra text)."""


def build_plan_generate_messages(
    event_id: str,
    triage_result: TriageResult | None,
) -> list[dict[str, str]]:
    triage_json = json.dumps(
        triage_result.model_dump(mode="json") if triage_result else {},
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": PLAN_GENERATE_SYSTEM},
        {
            "role": "user",
            "content": PLAN_GENERATE_USER.format(
                event_id=event_id,
                triage_json=triage_json,
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Plan revise prompt
# --------------------------------------------------------------------------- #

PLAN_REVISE_SYSTEM = """\
You are a security investigation planner. A previous investigation plan failed or produced
insufficient results. Revise the plan to address the failure reason.

Output a JSON object with these fields:
- plan_id: same plan_id as the previous plan
- event_id: the event_id from the input
- steps: list of { step_order, step_goal, assigned_agent, required_tools, success_criteria }
- budget: same as previous or adjusted
- revision: previous.revision + 1
- revise_reason: the failure reason from the input
- degraded: false

Rules:
1. Keep steps that produced useful results; replace or augment those that failed.
2. Add more specific tools or broader evidence collection as needed.
3. The new plan must have steps that are NOT identical to the previous plan.
4. When assigned_agent is evidence_agent, required_tools must use canonical query tool names.
"""

PLAN_REVISE_USER = """\
Event ID: {event_id}

Failure reason: {failure_reason}

Previous plan:
{previous_plan_json}

Generate the revised plan as JSON only (no extra text)."""


def build_plan_revise_messages(
    event_id: str,
    failure_reason: str,
    previous_plan: ExecutionPlan,
) -> list[dict[str, str]]:
    previous_plan_json = json.dumps(
        previous_plan.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": PLAN_REVISE_SYSTEM},
        {
            "role": "user",
            "content": PLAN_REVISE_USER.format(
                event_id=event_id,
                failure_reason=failure_reason,
                previous_plan_json=previous_plan_json,
            ),
        },
    ]


__all__ = [
    "PLAN_GENERATE_SYSTEM",
    "PLAN_GENERATE_USER",
    "PLAN_REVISE_SYSTEM",
    "PLAN_REVISE_USER",
    "build_plan_generate_messages",
    "build_plan_revise_messages",
]
