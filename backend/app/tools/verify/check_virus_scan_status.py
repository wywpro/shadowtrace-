from typing import Any

from app.tools.verify._common import execute_verification_tool, verification_tool_meta

TOOL_META = verification_tool_meta("check_virus_scan_status")


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    return await execute_verification_tool(TOOL_META.tool_name, params)
