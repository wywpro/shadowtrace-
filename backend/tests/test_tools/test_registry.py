"""ISSUE-018 ToolRegistry contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.enums import (
    ActionCategory,
    ActionLevel,
    DispositionIntentKind,
    ExecutionOwner,
    ToolCategory,
)
from app.models.tool_meta import (
    ExecutionChannel,
    ProviderToolBinding,
    RoutingKind,
    SideEffectLevel,
    ToolMeta,
    WrongExecutionChannelError,
)
from app.tools.base import get_declared_tool_meta, tool
from app.tools.registry import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistrationView,
    ToolRegistry,
    ToolUnavailableReason,
    ToolValidationError,
    get_tool_registry,
    tool_registry,
)
from app.tools.specs import baseline_tool_index


def _query_meta(name: str = "query_fixture") -> ToolMeta:
    return ToolMeta(
        tool_name=name,
        tool_category=ToolCategory.QUERY,
        routing_kind=RoutingKind.TOOL_PROVIDER_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "mode": {"type": "string", "enum": ["brief", "full"]},
            },
            "required": ["account", "mode"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"records": {"type": "array"}},
            "required": ["records"],
            "additionalProperties": False,
        },
    )


def _owner_routed_meta(name: str = "fixture_response") -> ToolMeta:
    return ToolMeta(
        tool_name=name,
        tool_category=ToolCategory.RESPONSE,
        action_category=ActionCategory.RESPONSE,
        routing_kind=RoutingKind.OWNER_ROUTED,
        supported_execution_owners=[
            ExecutionOwner.DIRECT_TOOL,
            ExecutionOwner.XDR_MANAGED,
        ],
        required_disposition_intent_by_owner={
            ExecutionOwner.DIRECT_TOOL: DispositionIntentKind.EXECUTION_RESULT_RECORD,
            ExecutionOwner.XDR_MANAGED: DispositionIntentKind.ENTITY_ACTION_SUBMIT,
        },
        required_capabilities=["entity_response"],
        side_effect_level=SideEffectLevel.MEDIUM,
        action_level=ActionLevel.L2,
    )


async def _execute(params: dict[str, object]) -> dict[str, object]:
    return {"records": [params]}


def test_register_lookup_list_stats_and_unregister_flow() -> None:
    registry = ToolRegistry()
    query_meta = _query_meta()
    registry.register(query_meta, _execute)

    registered = registry.get_tool(query_meta.tool_name)
    assert registered.tool_meta == query_meta
    assert registered.tool_impl is _execute
    assert registered.registered_at.tzinfo is not None
    assert registry.list_tools() == [query_meta]
    assert registry.list_available_tools() == [query_meta]
    assert registry.list_registered_tools() == [
        ToolRegistrationView(
            tool_meta=query_meta,
            bindings=(),
            healthy=True,
            available=True,
            unavailable_reasons=(),
        )
    ]
    assert registry.list_tools(ToolCategory.QUERY) == [query_meta]
    assert registry.list_tools("response") == []
    assert registry.get_tool_stats() == {
        "total_tools": 1,
        "executable_tools": 1,
        "virtual_tools": 0,
        "healthy_tools": 1,
        "unhealthy_tools": 0,
        "total_bindings": 0,
        "total_calls": 0,
        "total_errors": 0,
        "by_category": {"query": 1},
    }

    registry.unregister(query_meta.tool_name)
    with pytest.raises(ToolNotFoundError) as exc:
        registry.get_tool(query_meta.tool_name)
    assert exc.value.error_code == "tool_not_found"


def test_duplicate_tool_and_missing_implementation_are_rejected() -> None:
    registry = ToolRegistry()
    meta = _query_meta()
    registry.register(meta, _execute)
    with pytest.raises(ToolAlreadyRegisteredError) as duplicate:
        registry.register(meta, _execute)
    assert duplicate.value.error_code == "tool_already_registered"

    with pytest.raises(ToolValidationError, match="requires an async implementation"):
        ToolRegistry().register(_query_meta("missing_execute"))


def test_tool_decorator_requires_async_and_records_meta() -> None:
    meta = _query_meta()

    @tool(meta)
    async def decorated(params: dict[str, object]) -> dict[str, object]:
        return {"records": [params]}

    assert get_declared_tool_meta(decorated) == meta
    ToolRegistry().register(meta, decorated)

    with pytest.raises(TypeError, match="must be async"):

        @tool(meta)
        def invalid(params: dict[str, object]) -> dict[str, object]:
            return {"records": [params]}


@pytest.mark.parametrize(
    ("params", "expected_path", "expected_reason"),
    [
        ({"mode": "brief"}, "$.account", "required property"),
        ({"account": 7, "mode": "brief"}, "$.account", "not of type 'string'"),
        ({"account": "alice", "mode": "verbose"}, "$.mode", "is not one of"),
    ],
)
def test_input_validation_reports_path_and_reason(
    params: dict[str, object],
    expected_path: str,
    expected_reason: str,
) -> None:
    registry = ToolRegistry()
    registry.register(_query_meta(), _execute)

    with pytest.raises(ToolValidationError) as exc:
        registry.validate_input("query_fixture", params)
    assert exc.value.error_code == "tool_validation_error"
    assert exc.value.details["path"] == expected_path
    assert expected_reason in exc.value.details["reason"]
    assert expected_path in str(exc.value)


def test_output_validation_uses_declared_schema() -> None:
    registry = ToolRegistry()
    registry.register(_query_meta(), _execute)
    registry.validate_output("query_fixture", {"records": []})

    with pytest.raises(ToolValidationError) as exc:
        registry.validate_output("query_fixture", {"records": "not-a-list"})
    assert exc.value.details["path"] == "$.records"


def test_schema_validation_reason_does_not_echo_rejected_secret_value() -> None:
    registry = ToolRegistry()
    registry.register(_query_meta(), _execute)
    secret = "Bearer registry-validation-secret"

    with pytest.raises(ToolValidationError) as exc:
        registry.validate_input("query_fixture", {"account": [secret], "mode": "brief"})

    assert secret not in str(exc.value)
    assert secret not in str(exc.value.details)
    assert exc.value.details["reason"] == "value is not of type 'string'"


def test_owner_routed_tool_supports_exactly_one_binding_per_owner() -> None:
    registry = ToolRegistry()
    meta = _owner_routed_meta()
    registry.register(meta, _execute)
    direct = ProviderToolBinding(
        tool_name=meta.tool_name,
        provider_name="mock-tool-provider",
        execution_owner=ExecutionOwner.DIRECT_TOOL,
        execution_channel=ExecutionChannel.TOOL_PROVIDER,
        capabilities=["entity_response", "fast_path"],
    )
    xdr = ProviderToolBinding(
        tool_name=meta.tool_name,
        provider_name="mock-xdr",
        execution_owner=ExecutionOwner.XDR_MANAGED,
        execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
        capabilities=["entity_response", "status_lookup"],
    )
    registry.register_binding(direct)
    registry.register_binding(xdr)

    assert registry.list_bindings(meta.tool_name) == [direct, xdr]
    assert (
        registry.resolve_binding(
            meta.tool_name,
            ExecutionOwner.DIRECT_TOOL,
            ["fast_path"],
        )
        == direct
    )
    assert (
        registry.resolve_binding(
            meta.tool_name,
            ExecutionOwner.XDR_MANAGED,
            ["status_lookup"],
        )
        == xdr
    )

    conflict = direct.model_copy(update={"provider_name": "other-provider"})
    with pytest.raises(ToolAlreadyRegisteredError, match="execution_owner=direct_tool"):
        registry.register_binding(conflict)


def test_binding_resolution_merges_meta_and_runtime_capabilities() -> None:
    registry = ToolRegistry()
    meta = _owner_routed_meta()
    registry.register(meta, _execute)
    registry.register_binding(
        ProviderToolBinding(
            tool_name=meta.tool_name,
            provider_name="limited",
            execution_owner=ExecutionOwner.DIRECT_TOOL,
            execution_channel=ExecutionChannel.TOOL_PROVIDER,
            capabilities=["fast_path"],
        )
    )

    with pytest.raises(ToolNotFoundError) as exc:
        registry.resolve_binding(meta.tool_name, ExecutionOwner.DIRECT_TOOL, ["fast_path"])
    assert exc.value.details["required_capabilities"] == ["entity_response", "fast_path"]


def test_available_view_filters_missing_binding_and_unhealthy_tools() -> None:
    registry = ToolRegistry()
    meta = _owner_routed_meta()
    registry.register(meta, _execute)

    assert registry.list_available_tools() == []
    missing = registry.list_registered_tools()[0]
    assert missing.available is False
    assert missing.unavailable_reasons == (ToolUnavailableReason.BINDING_UNAVAILABLE,)

    registry.register_binding(
        ProviderToolBinding(
            tool_name=meta.tool_name,
            provider_name="mock-xdr",
            execution_owner=ExecutionOwner.XDR_MANAGED,
            execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
            capabilities=["entity_response"],
        )
    )
    assert registry.list_available_tools() == [meta]
    assert registry.list_available_tools(execution_owner=ExecutionOwner.XDR_MANAGED) == [meta]
    assert registry.list_available_tools(execution_owner=ExecutionOwner.DIRECT_TOOL) == []
    assert registry.list_available_tools(required_capabilities=["status_lookup"]) == []

    registry.get_tool(meta.tool_name).healthy = False
    assert registry.list_available_tools() == []
    unhealthy = registry.list_registered_tools()[0]
    assert unhealthy.available is False
    assert unhealthy.unavailable_reasons == (ToolUnavailableReason.UNHEALTHY,)
    assert unhealthy.bindings[0].provider_name == "mock-xdr"


def test_tool_provider_only_binding_rejects_xdr_owner() -> None:
    registry = ToolRegistry()
    meta = _query_meta()
    registry.register(meta, _execute)
    with pytest.raises(ToolValidationError, match="is not supported"):
        registry.register_binding(
            ProviderToolBinding(
                tool_name=meta.tool_name,
                provider_name="mock-xdr",
                execution_owner=ExecutionOwner.XDR_MANAGED,
                execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
            )
        )


@pytest.mark.asyncio
async def test_virtual_meta_is_listed_but_cannot_execute_via_tool_provider() -> None:
    registry = ToolRegistry()
    virtual = baseline_tool_index()["update_source_event_disposition"]
    registry.register(virtual)

    registered = registry.get_tool(virtual.tool_name)
    assert registered.tool_impl is None
    assert registry.list_tools(ToolCategory.RESPONSE) == [virtual]
    assert registry.list_available_tools(ToolCategory.RESPONSE) == []
    audit = registry.list_registered_tools(ToolCategory.RESPONSE)[0]
    assert audit.available is False
    assert audit.unavailable_reasons == (
        ToolUnavailableReason.VIRTUAL_META,
        ToolUnavailableReason.IMPLEMENTATION_MISSING,
    )
    with pytest.raises(WrongExecutionChannelError) as exc:
        await registered.execute({})
    assert exc.value.error_code == "wrong_execution_channel"
    assert registered.call_count == 0

    with pytest.raises(ToolValidationError, match="must not define"):
        ToolRegistry().register(virtual, _execute)


def test_auto_discovery_handles_empty_or_missing_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "empty_tool_provider"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    for category in ("query", "response", "verify", "rollback"):
        package = root / category
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ToolRegistry()
    assert registry.auto_discover("empty_tool_provider", include_virtual=False) == []
    assert registry.auto_discover("does_not_exist", include_virtual=False) == []
    assert registry.list_tools() == []


@pytest.mark.asyncio
async def test_auto_discovery_registers_public_tool_modules_and_skips_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "discovered_tool_provider"
    query = root / "query"
    query.mkdir(parents=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    (query / "__init__.py").write_text("", encoding="utf-8")
    (query / "fixture_loader.py").write_text("HELPER = True\n", encoding="utf-8")
    (query / "query_dynamic.py").write_text(
        "\n".join(
            [
                "from app.models.enums import ToolCategory",
                "from app.models.tool_meta import RoutingKind, ToolMeta",
                "TOOL_META = ToolMeta(",
                "    tool_name='query_dynamic',",
                "    tool_category=ToolCategory.QUERY,",
                "    routing_kind=RoutingKind.TOOL_PROVIDER_ONLY,",
                "    input_schema={'type': 'object'},",
                ")",
                "async def execute(params: dict) -> dict:",
                "    return {'echo': params}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ToolRegistry()
    assert registry.auto_discover(
        "discovered_tool_provider",
        include_virtual=False,
    ) == ["query_dynamic"]
    assert await registry.get_tool("query_dynamic").execute({"value": 1}) == {"echo": {"value": 1}}
    assert registry.get_tool_stats()["total_calls"] == 1


def test_module_singleton_and_dependency_are_stable() -> None:
    assert get_tool_registry() is tool_registry
    virtual_names = {
        entry.tool_meta.tool_name
        for entry in tool_registry.list_registered_tools()
        if not entry.tool_meta.executable
    }
    assert "update_source_event_disposition" in virtual_names
