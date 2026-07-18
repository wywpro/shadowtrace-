"""Vendor-neutral live ToolProvider adapter contracts and mode routing."""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.sanitization import is_sensitive_key
from app.models.enums import CapabilityState, ExecutionJobStatus, ExecutionOwner
from app.models.execution import ActionExecutionJob
from app.models.tool_meta import (
    CapabilityManifest,
    ExecutionChannel,
    ProviderToolBinding,
    ToolMeta,
    ToolResult,
    ToolResultStatus,
)

if TYPE_CHECKING:
    from app.providers.tools.mock_provider import MockToolProvider
    from app.tools.registry import ToolRegistry


class ToolMode(StrEnum):
    MOCK = "mock"
    LIVE = "live"
    MIXED = "mixed"


class AdapterConfig(BaseModel):
    """Adapter configuration containing credential references, never values."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str
    auth_type: Literal["none", "bearer", "basic"] = "none"
    credential_ref: str = ""
    timeout_s: float = Field(default=30.0, gt=0)
    tls_verify: bool = True
    enabled: bool = False

    @field_validator("endpoint")
    @classmethod
    def _endpoint_must_not_embed_credentials(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint must not embed credentials")
        if any(is_sensitive_key(key) for key, _ in parse_qsl(parsed.query)):
            raise ValueError("endpoint must not carry credential query parameters")
        return value

    @field_validator("credential_ref")
    @classmethod
    def _credential_ref_is_an_environment_name(cls, value: str) -> str:
        if value and (not value.replace("_", "").isalnum() or value.upper() != value):
            raise ValueError("credential_ref must be an uppercase environment variable name")
        return value


class BaseToolAdapter(ABC):
    """One explicitly configured Provider implementation for one canonical tool."""

    name: str
    tool_meta: ToolMeta
    simulated: bool = False

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    @abstractmethod
    def capability_manifest(self) -> CapabilityManifest:
        """Declare status, lookup, and idempotent-execute support separately."""

    @abstractmethod
    async def execute(
        self,
        params: dict[str, Any],
        idempotency_key: str,
    ) -> ToolResult:
        """Execute one canonical tool operation."""

    async def get_job_status(self, provider_job_id: str) -> ToolResult:
        return self.unsupported_result(
            error_detail="provider job status query is unsupported",
            provider_code="status_query_unsupported",
        )

    async def lookup_by_idempotency(self, idempotency_key: str) -> ToolResult | None:
        return self.unsupported_result(
            error_detail="idempotency lookup is unsupported",
            provider_code="idempotency_lookup_unsupported",
        )

    @abstractmethod
    async def health_check(self) -> bool:
        """Return current submission readiness without producing side effects."""

    def validate_config(self) -> bool:
        if not self.config.enabled or not self.config.endpoint.strip():
            return False
        if self.config.auth_type == "none":
            return not self.config.credential_ref
        return bool(self.config.credential_ref)

    def unsupported_result(
        self,
        *,
        error_detail: str,
        provider_code: str,
    ) -> ToolResult:
        return ToolResult(
            call_id="call-adapter-unsupported",
            tool_name=self.tool_meta.tool_name,
            provider_name=self.name,
            status=ToolResultStatus.UNSUPPORTED,
            data={"simulated": self.simulated},
            provider_code=provider_code,
            error_detail=error_detail,
        )


def _job_status(status: ToolResultStatus) -> ExecutionJobStatus:
    return {
        ToolResultStatus.ACCEPTED: ExecutionJobStatus.QUEUED,
        ToolResultStatus.SUCCESS: ExecutionJobStatus.SUCCESS,
        ToolResultStatus.PARTIAL_SUCCESS: ExecutionJobStatus.PARTIAL_SUCCESS,
        ToolResultStatus.FAILED: ExecutionJobStatus.FAILED,
        ToolResultStatus.UNKNOWN: ExecutionJobStatus.UNKNOWN,
        ToolResultStatus.TIMEOUT: ExecutionJobStatus.TIMED_OUT,
    }.get(status, ExecutionJobStatus.FAILED)


def _adapter_implementation(adapter: BaseToolAdapter) -> Any:
    async def execute(params: dict[str, Any]) -> dict[str, Any]:
        from app.providers.tools.mock_provider import get_tool_execution_context

        context = get_tool_execution_context(adapter.tool_meta.tool_name, params)
        try:
            healthy = await adapter.health_check()
        except Exception:  # noqa: BLE001 - health failures must fail closed
            healthy = False
        if not healthy:
            result = adapter.unsupported_result(
                error_detail="provider is unavailable; action requires manual handling",
                provider_code="provider_unavailable",
            ).model_copy(update={"job_id": context.execution_job_id})
            return result.model_dump(mode="json")

        result = await adapter.execute(params, context.idempotency_key)
        result = result.model_copy(
            update={
                "provider_name": adapter.name,
                "job_id": result.job_id or context.execution_job_id,
                "data": {**result.data, "simulated": adapter.simulated},
                "raw_result": {**result.raw_result, "simulated": adapter.simulated},
            }
        )
        if adapter.tool_meta.output_schema == {"$ref": "ActionExecutionJob"} and result.status in {
            ToolResultStatus.ACCEPTED,
            ToolResultStatus.SUCCESS,
            ToolResultStatus.PARTIAL_SUCCESS,
        }:
            return ActionExecutionJob(
                job_id=context.execution_job_id or result.job_id or "",
                event_id=context.event_id,
                action_id=context.action_id,
                provider_name=adapter.name,
                idempotency_key=context.idempotency_key,
                provider_job_id=result.provider_job_id,
                status=(
                    ExecutionJobStatus.RUNNING
                    if adapter.tool_meta.async_mode and result.status is ToolResultStatus.SUCCESS
                    else _job_status(result.status)
                ),
                target_results=result.target_results,
                provider_code=result.provider_code,
                provider_message=result.provider_message,
                raw_result=result.raw_result,
            ).model_dump(mode="json")
        return result.model_dump(mode="json")

    return execute


async def _register_adapter(registry: ToolRegistry, adapter: BaseToolAdapter) -> str:
    from app.tools.registry import ToolValidationError

    if not adapter.config.enabled:
        raise ToolValidationError(
            f"adapter {adapter.name!r} is disabled",
            details={"provider_name": adapter.name},
        )
    if not adapter.validate_config():
        raise ToolValidationError(
            f"adapter {adapter.name!r} configuration is invalid",
            details={"provider_name": adapter.name},
        )
    manifest = adapter.capability_manifest()
    if manifest.provider_name != adapter.name:
        raise ToolValidationError(
            "adapter manifest provider_name does not match adapter name",
            details={
                "provider_name": adapter.name,
                "manifest_provider_name": manifest.provider_name,
            },
        )

    registry.register(
        adapter.tool_meta,
        _adapter_implementation(adapter),
    )
    registry.register_binding(
        ProviderToolBinding(
            tool_name=adapter.tool_meta.tool_name,
            provider_name=adapter.name,
            execution_owner=ExecutionOwner.DIRECT_TOOL,
            execution_channel=ExecutionChannel.TOOL_PROVIDER,
            capabilities=list(adapter.tool_meta.required_capabilities),
        )
    )
    try:
        healthy = await adapter.health_check()
    except Exception:  # noqa: BLE001 - registration remains audit-visible
        healthy = False
    registered = registry.get_tool(adapter.tool_meta.tool_name)
    registered.healthy = (
        manifest_is_executable(
            manifest,
            adapter.tool_meta.tool_name,
        )
        and healthy
    )
    registered.submission_ready = registered.healthy
    return adapter.tool_meta.tool_name


def discover_tool_adapter_classes(
    base_package: str = "app.tools.adapters",
) -> dict[str, type[BaseToolAdapter]]:
    """Discover in-tree Adapter classes; configuration still must be explicit."""

    from app.tools.registry import ToolValidationError

    package = importlib.import_module(base_package)
    discovered: dict[str, type[BaseToolAdapter]] = {}
    for module_info in pkgutil.iter_modules(package.__path__, f"{base_package}."):
        leaf_name = module_info.name.rsplit(".", 1)[-1]
        if module_info.ispkg or leaf_name.startswith("_") or leaf_name == "base":
            continue
        module = importlib.import_module(module_info.name)
        adapter_class = getattr(module, "ADAPTER_CLASS", None)
        if adapter_class is None:
            continue
        if not isinstance(adapter_class, type) or not issubclass(
            adapter_class,
            BaseToolAdapter,
        ):
            raise ToolValidationError(
                f"module {module.__name__!r} exports an invalid ADAPTER_CLASS",
                details={"module": module.__name__},
            )
        if adapter_class.name in discovered:
            raise ToolValidationError(
                f"duplicate discovered adapter Provider {adapter_class.name!r}",
                details={"provider_name": adapter_class.name},
            )
        discovered[adapter_class.name] = adapter_class
    return discovered


async def configure_tool_registry(
    registry: ToolRegistry,
    *,
    tool_mode: ToolMode | str,
    adapters: Sequence[BaseToolAdapter] = (),
    adapter_configs: Mapping[str, AdapterConfig] | None = None,
    mixed_routes: Mapping[str, str] | None = None,
    simulation_enabled: bool = True,
    allow_live_side_effects: bool = False,
    mock_provider: MockToolProvider | None = None,
) -> list[str]:
    """Populate an empty registry with strict mock/live/mixed Provider routes.

    ``live`` never discovers Mock implementations. ``mixed`` loads only tools
    named in its route table, so a missing route cannot silently fall back.
    """

    from app.providers.tools.mock_provider import MockToolProvider
    from app.tools.registry import ToolValidationError

    mode = ToolMode(tool_mode)
    configured_adapters = list(adapters)
    if adapter_configs:
        discovered_classes = discover_tool_adapter_classes()
        unknown_adapter_configs = sorted(set(adapter_configs) - set(discovered_classes))
        if unknown_adapter_configs:
            raise ToolValidationError(
                "configuration references an unknown ToolAdapter",
                details={"provider_names": unknown_adapter_configs},
            )
        configured_adapters.extend(
            discovered_classes[provider_name](config)
            for provider_name, config in adapter_configs.items()
        )

    adapter_by_name = {adapter.name: adapter for adapter in configured_adapters}
    if len(adapter_by_name) != len(configured_adapters):
        raise ToolValidationError("adapter provider names must be unique")

    if mode is ToolMode.MOCK:
        if not simulation_enabled:
            raise ToolValidationError("mock tool mode requires simulation_enabled=true")
        discovered = registry.auto_discover()
        (mock_provider or MockToolProvider()).register_bindings(registry)
        return discovered

    if mode is ToolMode.LIVE:
        simulated_providers = sorted(
            adapter.name for adapter in configured_adapters if adapter.simulated
        )
        if simulated_providers:
            raise ToolValidationError(
                "live tool mode forbids simulated Providers",
                details={"provider_names": simulated_providers},
            )
        if configured_adapters and not allow_live_side_effects:
            raise ToolValidationError(
                "live ToolProvider side effects are disabled",
                details={"allow_live_side_effects": False},
            )
        discovered = [await _register_adapter(registry, adapter) for adapter in configured_adapters]
        discovered.extend(registry.load_virtual_metas())
        return discovered

    routes = dict(mixed_routes or {})
    if not routes:
        raise ToolValidationError("mixed tool mode requires an explicit per-tool route table")

    mock_names = {
        tool_name
        for tool_name, provider_name in routes.items()
        if provider_name in {"mock", "mock_tool_provider"}
    }
    if mock_names and not simulation_enabled:
        raise ToolValidationError("mixed Mock routes require simulation_enabled=true")

    unknown_providers = sorted(
        {
            provider_name
            for provider_name in routes.values()
            if provider_name not in {"mock", "mock_tool_provider"}
            and provider_name not in adapter_by_name
        }
    )
    if unknown_providers:
        raise ToolValidationError(
            "mixed route references an unknown Provider",
            details={"provider_names": unknown_providers},
        )
    live_route_providers = sorted(
        {
            provider_name
            for provider_name in routes.values()
            if provider_name in adapter_by_name and not adapter_by_name[provider_name].simulated
        }
    )
    if live_route_providers and not allow_live_side_effects:
        raise ToolValidationError(
            "mixed live ToolProvider side effects are disabled",
            details={
                "allow_live_side_effects": False,
                "provider_names": live_route_providers,
            },
        )

    discovered = registry.auto_discover(
        include_virtual=True,
        include_tools=mock_names,
    )
    provider = mock_provider or MockToolProvider()
    for binding in provider.provider_bindings():
        if binding.tool_name not in mock_names:
            continue
        registry.register_binding(binding)

    for tool_name, provider_name in routes.items():
        if provider_name in {"mock", "mock_tool_provider"}:
            continue
        adapter = adapter_by_name[provider_name]
        if adapter.tool_meta.tool_name != tool_name:
            raise ToolValidationError(
                "mixed route tool does not match adapter tool",
                details={
                    "tool_name": tool_name,
                    "adapter_tool_name": adapter.tool_meta.tool_name,
                    "provider_name": provider_name,
                },
            )
        discovered.append(await _register_adapter(registry, adapter))
    return discovered


def manifest_is_executable(manifest: CapabilityManifest, tool_name: str) -> bool:
    """Return whether a manifest explicitly supports direct idempotent execution."""

    return (
        manifest.online
        and manifest.entity_response is CapabilityState.SUPPORTED
        and manifest.supports_idempotency
        and tool_name in manifest.allowed_operations
        and ExecutionChannel.TOOL_PROVIDER in manifest.allowed_execution_channels
    )


__all__ = [
    "AdapterConfig",
    "BaseToolAdapter",
    "ToolMode",
    "configure_tool_registry",
    "discover_tool_adapter_classes",
    "manifest_is_executable",
]
