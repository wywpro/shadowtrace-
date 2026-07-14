"""Shared execution path for evidence projection query tools."""

from __future__ import annotations

from typing import Any

from app.models.ids import new_call_id
from app.models.tool_meta import ToolMeta
from app.services.evidence_projection import (
    ProjectionSource,
    build_query_tool_result,
    confidence_for_query_data,
    get_evidence_projection,
    query_output_schema,
)
from app.tools.inputs import TOOL_INPUT_MODELS
from app.tools.specs import baseline_tool_index

_ENTITY_FIELDS: dict[ProjectionSource, tuple[str, ...]] = {
    "account_login": ("account",),
    "edr_process": ("host_id",),
    "file_access": ("account",),
    "network_flow": ("src_ip", "dst_ip"),
    "dns": ("domain",),
    "asset_info": ("ip", "hostname"),
    "vuln_info": ("ip", "hostname"),
    "threat_intel": ("indicator",),
    "history_cases": ("pattern_description",),
}


def query_tool_meta(tool_name: str) -> ToolMeta:
    """Use the ISSUE-006 canonical meta with the ISSUE-019 output contract."""
    return baseline_tool_index()[tool_name].model_copy(
        deep=True,
        update={"output_schema": query_output_schema()},
    )


async def execute_projected_query(
    tool_name: str,
    source: ProjectionSource,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Validate input, query the bound projection, and normalize ToolResult."""
    parsed = TOOL_INPUT_MODELS[tool_name].model_validate(params)
    entity = {
        field: value
        for field in _ENTITY_FIELDS[source]
        if (value := getattr(parsed, field, None)) is not None
    }
    parsed_time_range = getattr(parsed, "time_range", None)
    time_range = (
        (parsed_time_range.start, parsed_time_range.end) if parsed_time_range is not None else None
    )
    data = await get_evidence_projection().query(
        source,
        entity,
        time_range,
        cursor=None,
        limit=100,
    )
    return build_query_tool_result(
        call_id=new_call_id(),
        tool_name=tool_name,
        data=data,
        confidence=confidence_for_query_data(data),
    )
