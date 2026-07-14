"""Tool implementation contract shared by discovered ToolProvider modules."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast, runtime_checkable

from app.models.tool_meta import ToolMeta

ToolImplementation = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
_TOOL_META_ATTRIBUTE = "__shadowtrace_tool_meta__"


@runtime_checkable
class ToolImplementationProtocol(Protocol):
    """Callable shape exported as ``async execute(params: dict) -> dict``."""

    async def __call__(self, params: dict[str, Any]) -> dict[str, Any]: ...


def validate_tool_implementation(tool_impl: object) -> ToolImplementation:
    """Return a typed implementation or reject a non-async/non-callable object."""
    if not callable(tool_impl):
        raise TypeError("tool implementation must be callable")

    if not inspect.iscoroutinefunction(tool_impl):
        raise TypeError("tool implementation must be async")
    return cast(ToolImplementation, tool_impl)


def tool(tool_meta: ToolMeta) -> Callable[[ToolImplementation], ToolImplementation]:
    """Tag an async implementation with its canonical :class:`ToolMeta`."""

    def decorator(tool_impl: ToolImplementation) -> ToolImplementation:
        validated = validate_tool_implementation(tool_impl)
        setattr(validated, _TOOL_META_ATTRIBUTE, tool_meta)
        return validated

    return decorator


def get_declared_tool_meta(tool_impl: object) -> ToolMeta | None:
    """Read metadata attached by :func:`tool`, if the decorator was used."""
    value = getattr(tool_impl, _TOOL_META_ATTRIBUTE, None)
    return value if isinstance(value, ToolMeta) else None


# Explicit alias for callers that prefer a descriptive decorator name.
tool_implementation = tool


__all__ = [
    "ToolImplementation",
    "ToolImplementationProtocol",
    "get_declared_tool_meta",
    "tool",
    "tool_implementation",
    "validate_tool_implementation",
]
