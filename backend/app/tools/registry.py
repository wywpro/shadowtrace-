"""Runtime registry for canonical tools and Provider execution bindings."""

from __future__ import annotations

import importlib
import pkgutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import ModuleType
from typing import Any

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

from app.core.errors import ShadowTraceError
from app.models.enums import ErrorCategory, ExecutionOwner, ToolCategory
from app.models.execution import ActionExecutionJob
from app.models.tool_meta import (
    ExecutionChannel,
    ProviderToolBinding,
    RoutingKind,
    ToolMeta,
    WrongExecutionChannelError,
    ensure_tool_provider_executable,
)
from app.tools.base import (
    ToolImplementation,
    get_declared_tool_meta,
    validate_tool_implementation,
)
from app.tools.specs import BASELINE_TOOL_METAS

_DISCOVERY_PACKAGES = ("query", "response", "verify", "rollback")
_EXTERNAL_SCHEMA_REFS: dict[str, dict[str, Any]] = {
    "ActionExecutionJob": ActionExecutionJob.model_json_schema(),
}


class ToolAlreadyRegisteredError(ShadowTraceError):
    """A tool name or owner binding already exists in this registry."""

    status_code = 409
    default_error_code = "tool_already_registered"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


class ToolNotFoundError(ShadowTraceError):
    """A tool or matching Provider binding is unavailable."""

    status_code = 404
    default_error_code = "tool_not_found"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


class ToolValidationError(ShadowTraceError):
    """Tool metadata, implementation, binding, input, or output is invalid."""

    status_code = 422
    default_error_code = "tool_validation_error"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


@dataclass(slots=True)
class RegisteredTool:
    """A canonical tool plus the currently available Provider capabilities."""

    tool_meta: ToolMeta
    tool_impl: ToolImplementation | None = None
    bindings: list[ProviderToolBinding] = field(default_factory=list)
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    call_count: int = 0
    error_count: int = 0
    healthy: bool = True

    def require_tool_provider_impl(self) -> ToolImplementation:
        """Fail closed when a virtual/disposition-only meta reaches ToolExecutor."""
        ensure_tool_provider_executable(self.tool_meta)
        if self.tool_impl is None:
            raise WrongExecutionChannelError(
                self.tool_meta.tool_name,
                routing_kind=self.tool_meta.routing_kind,
            )
        return self.tool_impl

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Invoke the implementation while maintaining lightweight health stats."""
        implementation = self.require_tool_provider_impl()
        self.call_count += 1
        try:
            result = await implementation(params)
        except Exception:
            self.error_count += 1
            self.healthy = False
            raise
        self.healthy = True
        return result


class ToolRegistry:
    """Register canonical tools without assuming a fixed catalog size."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        tool_meta: ToolMeta,
        tool_impl: ToolImplementation | None = None,
    ) -> None:
        """Register one canonical tool and its optional ToolProvider implementation."""
        if tool_meta.tool_name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"tool {tool_meta.tool_name!r} is already registered",
                details={"tool_name": tool_meta.tool_name},
            )

        is_virtual = (
            tool_meta.routing_kind is RoutingKind.DISPOSITION_ONLY or not tool_meta.executable
        )
        if is_virtual and tool_impl is not None:
            raise ToolValidationError(
                f"virtual tool {tool_meta.tool_name!r} must not define an execute implementation",
                details={"tool_name": tool_meta.tool_name, "field": "tool_impl"},
            )
        if not is_virtual and tool_impl is None:
            raise ToolValidationError(
                f"executable tool {tool_meta.tool_name!r} requires an async implementation",
                details={"tool_name": tool_meta.tool_name, "field": "tool_impl"},
            )

        validated_impl: ToolImplementation | None = None
        if tool_impl is not None:
            try:
                validated_impl = validate_tool_implementation(tool_impl)
            except TypeError as exc:
                raise ToolValidationError(
                    f"invalid implementation for tool {tool_meta.tool_name!r}: {exc}",
                    details={"tool_name": tool_meta.tool_name, "reason": str(exc)},
                ) from exc
            declared_meta = get_declared_tool_meta(validated_impl)
            if declared_meta is not None and declared_meta != tool_meta:
                raise ToolValidationError(
                    f"decorated metadata does not match {tool_meta.tool_name!r}",
                    details={"tool_name": tool_meta.tool_name},
                )

        self._check_schema(tool_meta.tool_name, "input", tool_meta.input_schema)
        self._check_schema(tool_meta.tool_name, "output", tool_meta.output_schema)
        self._tools[tool_meta.tool_name] = RegisteredTool(
            tool_meta=tool_meta.model_copy(deep=True),
            tool_impl=validated_impl,
        )

    def register_binding(self, binding: ProviderToolBinding) -> None:
        """Attach the sole binding for a canonical ``(tool, execution_owner)`` pair."""
        registered = self.get_tool(binding.tool_name)
        self._validate_binding_route(registered.tool_meta, binding)
        if any(item.execution_owner is binding.execution_owner for item in registered.bindings):
            raise ToolAlreadyRegisteredError(
                f"tool {binding.tool_name!r} already has a binding for "
                f"execution_owner={binding.execution_owner.value}",
                details={
                    "tool_name": binding.tool_name,
                    "execution_owner": binding.execution_owner.value,
                },
            )
        registered.bindings.append(binding.model_copy(deep=True))

    def get_tool(self, tool_name: str) -> RegisteredTool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"tool {tool_name!r} is not registered",
                details={"tool_name": tool_name},
            ) from exc

    def list_tools(self, category: ToolCategory | str | None = None) -> list[ToolMeta]:
        """List a snapshot of available metadata, optionally filtered by category."""
        expected = category.value if isinstance(category, ToolCategory) else category
        return [
            registered.tool_meta.model_copy(deep=True)
            for registered in self._tools.values()
            if expected is None or registered.tool_meta.tool_category.value == expected
        ]

    def list_bindings(self, tool_name: str) -> list[ProviderToolBinding]:
        return [item.model_copy(deep=True) for item in self.get_tool(tool_name).bindings]

    def resolve_binding(
        self,
        tool_name: str,
        execution_owner: ExecutionOwner,
        required_capabilities: Sequence[str],
    ) -> ProviderToolBinding:
        """Freeze one owner binding whose capabilities cover all current requirements."""
        registered = self.get_tool(tool_name)
        required = set(registered.tool_meta.required_capabilities)
        required.update(required_capabilities)
        for binding in registered.bindings:
            if binding.execution_owner is not execution_owner:
                continue
            if required.issubset(binding.capabilities):
                return binding.model_copy(deep=True)
        raise ToolNotFoundError(
            f"no binding for tool {tool_name!r}, execution_owner={execution_owner.value}, "
            f"capabilities={sorted(required)!r}",
            details={
                "tool_name": tool_name,
                "execution_owner": execution_owner.value,
                "required_capabilities": sorted(required),
            },
        )

    def validate_input(self, tool_name: str, params: dict[str, Any]) -> None:
        self._validate_instance(tool_name, "input", params)

    def validate_output(self, tool_name: str, result: dict[str, Any]) -> None:
        self._validate_instance(tool_name, "output", result)

    def unregister(self, tool_name: str) -> None:
        self.get_tool(tool_name)
        del self._tools[tool_name]

    def get_tool_stats(self) -> dict[str, Any]:
        """Return registry/cardinality and execution health statistics."""
        by_category = Counter(
            registered.tool_meta.tool_category.value for registered in self._tools.values()
        )
        return {
            "total_tools": len(self._tools),
            "executable_tools": sum(item.tool_meta.executable for item in self._tools.values()),
            "virtual_tools": sum(not item.tool_meta.executable for item in self._tools.values()),
            "healthy_tools": sum(item.healthy for item in self._tools.values()),
            "unhealthy_tools": sum(not item.healthy for item in self._tools.values()),
            "total_bindings": sum(len(item.bindings) for item in self._tools.values()),
            "total_calls": sum(item.call_count for item in self._tools.values()),
            "total_errors": sum(item.error_count for item in self._tools.values()),
            "by_category": dict(sorted(by_category.items())),
        }

    def load_virtual_metas(self, metas: Sequence[ToolMeta] = BASELINE_TOOL_METAS) -> list[str]:
        """Load disposition-only catalog entries from specs, never from execute modules."""
        loaded: list[str] = []
        for meta in metas:
            if meta.routing_kind is not RoutingKind.DISPOSITION_ONLY:
                continue
            existing = self._tools.get(meta.tool_name)
            if existing is not None:
                if existing.tool_meta != meta or existing.tool_impl is not None:
                    raise ToolAlreadyRegisteredError(
                        f"conflicting virtual tool {meta.tool_name!r} is already registered",
                        details={"tool_name": meta.tool_name},
                    )
                continue
            self.register(meta)
            loaded.append(meta.tool_name)
        return loaded

    def auto_discover(
        self,
        base_package: str = "app.tools",
        *,
        include_virtual: bool = True,
    ) -> list[str]:
        """Import public tool modules from the four ToolProvider package locations."""
        discovered: list[str] = []
        for category in _DISCOVERY_PACKAGES:
            package_name = f"{base_package}.{category}"
            package = self._optional_package(package_name)
            if package is None:
                continue
            for module_info in pkgutil.iter_modules(package.__path__, f"{package_name}."):
                leaf_name = module_info.name.rsplit(".", 1)[-1]
                if module_info.ispkg or leaf_name.startswith("_"):
                    continue
                module = importlib.import_module(module_info.name)
                if not hasattr(module, "TOOL_META") and not hasattr(module, "execute"):
                    # Support modules (for example query/fixture_loader.py) are
                    # not tool implementations and must not enter the catalog.
                    continue
                meta, implementation = self._discovered_exports(module)
                existing = self._tools.get(meta.tool_name)
                if existing is not None:
                    if existing.tool_meta == meta and existing.tool_impl is implementation:
                        continue
                    raise ToolAlreadyRegisteredError(
                        f"discovered tool {meta.tool_name!r} conflicts with an existing tool",
                        details={"tool_name": meta.tool_name, "module": module.__name__},
                    )
                self.register(meta, implementation)
                discovered.append(meta.tool_name)
        if include_virtual:
            discovered.extend(self.load_virtual_metas())
        return discovered

    def _validate_instance(self, tool_name: str, direction: str, value: dict[str, Any]) -> None:
        registered = self.get_tool(tool_name)
        schema = (
            registered.tool_meta.input_schema
            if direction == "input"
            else registered.tool_meta.output_schema
        )
        if not schema:
            return
        validator = self._validator(tool_name, direction, schema)
        try:
            errors = list(validator.iter_errors(value))
        except Exception as exc:
            raise ToolValidationError(
                f"unable to validate {direction} for tool {tool_name!r}: {exc}",
                details={
                    "tool_name": tool_name,
                    "direction": direction,
                    "reason": str(exc),
                },
            ) from exc
        if not errors:
            return
        error = jsonschema_exceptions.best_match(errors)
        path = self._error_path(error)
        raise ToolValidationError(
            f"{direction} validation failed for tool {tool_name!r} at {path}: {error.message}",
            details={
                "tool_name": tool_name,
                "direction": direction,
                "path": path,
                "reason": error.message,
                "validator": error.validator,
            },
        )

    @classmethod
    def _check_schema(cls, tool_name: str, direction: str, schema: dict[str, Any]) -> None:
        if not schema:
            return
        try:
            validator_for(cls._resolved_schema(schema)).check_schema(cls._resolved_schema(schema))
        except jsonschema_exceptions.SchemaError as exc:
            path = cls._path_from_parts(exc.absolute_schema_path)
            raise ToolValidationError(
                f"invalid {direction} schema for tool {tool_name!r} at {path}: {exc.message}",
                details={
                    "tool_name": tool_name,
                    "direction": direction,
                    "path": path,
                    "reason": exc.message,
                },
            ) from exc

    @classmethod
    def _validator(cls, tool_name: str, direction: str, schema: dict[str, Any]) -> Validator:
        resolved = cls._resolved_schema(schema)
        validator_class = validator_for(resolved)
        try:
            validator_class.check_schema(resolved)
        except jsonschema_exceptions.SchemaError as exc:
            raise ToolValidationError(
                f"invalid {direction} schema for tool {tool_name!r}: {exc.message}",
                details={
                    "tool_name": tool_name,
                    "direction": direction,
                    "reason": exc.message,
                },
            ) from exc
        return validator_class(resolved)

    @staticmethod
    def _resolved_schema(schema: dict[str, Any]) -> dict[str, Any]:
        if set(schema) == {"$ref"}:
            ref = schema["$ref"]
            if isinstance(ref, str) and ref in _EXTERNAL_SCHEMA_REFS:
                return _EXTERNAL_SCHEMA_REFS[ref]
        return schema

    @staticmethod
    def _error_path(error: jsonschema_exceptions.ValidationError) -> str:
        parts = list(error.absolute_path)
        if error.validator == "required" and isinstance(error.instance, dict):
            missing = [field for field in error.validator_value if field not in error.instance]
            if missing:
                parts.append(missing[0])
        return ToolRegistry._path_from_parts(parts)

    @staticmethod
    def _path_from_parts(parts: Sequence[Any]) -> str:
        path = "$"
        for part in parts:
            path += f"[{part}]" if isinstance(part, int) else f".{part}"
        return path

    @staticmethod
    def _optional_package(package_name: str) -> ModuleType | None:
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError as exc:
            if exc.name is not None and (
                exc.name == package_name or package_name.startswith(f"{exc.name}.")
            ):
                return None
            raise
        if not hasattr(package, "__path__"):
            raise ToolValidationError(
                f"tool discovery target {package_name!r} is not a package",
                details={"package": package_name},
            )
        return package

    @staticmethod
    def _discovered_exports(module: ModuleType) -> tuple[ToolMeta, ToolImplementation]:
        meta = getattr(module, "TOOL_META", None)
        implementation = getattr(module, "execute", None)
        if not isinstance(meta, ToolMeta):
            raise ToolValidationError(
                f"discovered module {module.__name__!r} must export TOOL_META",
                details={"module": module.__name__, "field": "TOOL_META"},
            )
        if (
            meta.routing_kind is RoutingKind.DISPOSITION_ONLY
            or not meta.executable
            or implementation is None
        ):
            raise ToolValidationError(
                f"discovered module {module.__name__!r} must export an executable "
                "ToolProvider async execute",
                details={"module": module.__name__, "tool_name": meta.tool_name},
            )
        try:
            return meta, validate_tool_implementation(implementation)
        except TypeError as exc:
            raise ToolValidationError(
                f"invalid execute export in {module.__name__!r}: {exc}",
                details={"module": module.__name__, "reason": str(exc)},
            ) from exc

    @staticmethod
    def _validate_binding_route(meta: ToolMeta, binding: ProviderToolBinding) -> None:
        if meta.routing_kind is RoutingKind.TOOL_PROVIDER_ONLY:
            allowed = (
                binding.execution_owner is ExecutionOwner.DIRECT_TOOL
                and binding.execution_channel is ExecutionChannel.TOOL_PROVIDER
            )
        else:
            allowed = binding.execution_owner in meta.supported_execution_owners
        if not allowed:
            raise ToolValidationError(
                f"binding execution_owner={binding.execution_owner.value} is not supported "
                f"by tool {meta.tool_name!r}",
                details={
                    "tool_name": meta.tool_name,
                    "execution_owner": binding.execution_owner.value,
                    "routing_kind": meta.routing_kind.value,
                },
            )


def get_tool_registry() -> ToolRegistry:
    """FastAPI dependency returning the process registry singleton."""
    return tool_registry


tool_registry = ToolRegistry()
tool_registry.auto_discover()


__all__ = [
    "RegisteredTool",
    "ToolAlreadyRegisteredError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolValidationError",
    "get_tool_registry",
    "tool_registry",
]
