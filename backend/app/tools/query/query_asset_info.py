"""Asset inventory evidence query."""

from __future__ import annotations

from typing import Any

from app.tools.query._common import execute_projected_query, query_tool_meta

TOOL_META = query_tool_meta("query_asset_info")


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    return await execute_projected_query(TOOL_META.tool_name, "asset_info", params)
