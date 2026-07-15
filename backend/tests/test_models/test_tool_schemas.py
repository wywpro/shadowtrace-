"""Tool contract / CapabilityManifest / baseline catalog tests (ISSUE-006)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.action import Action
from app.models.disposition import (
    DispositionCommand,
    SetEventDispositionParams,
    SourceObjectLocator,
)
from app.models.enums import (
    ActionCategory,
    ActionLevel,
    CapabilityState,
    DispositionIntentKind,
    DispositionPolicy,
    ExecutionOwner,
    SourceDisposition,
    SourceObjectKind,
    ToolCategory,
    WritebackReadiness,
)
from app.models.tool_meta import (
    TERMINAL_DISPOSITION_TOOL,
    CapabilityBindingEntry,
    CapabilityManifest,
    ExecutionChannel,
    ProviderToolBinding,
    RoutingKind,
    SideEffectLevel,
    ToolMeta,
    ToolResult,
    ToolResultStatus,
    WrongExecutionChannelError,
    ensure_tool_provider_executable,
)
from app.tools.inputs import TOOL_INPUT_MODELS, QueryNetworkFlowInput, TimeRange
from app.tools.specs import (
    BASELINE_TOOL_METAS,
    BASELINE_TOOL_NAMES,
    RESPONSE_ROLLBACK_MAP,
    ROLLBACK_SOURCE_MAP,
    baseline_tool_index,
    export_baseline_tool_schemas,
    merge_provider_tools,
)

# Intro §4.5 required baseline names (sections 1–4). Total count is NOT a contract.
REQUIRED_BASELINE_NAMES = frozenset(
    {
        # query
        "query_account_login",
        "query_edr_process",
        "query_file_access",
        "query_network_flow",
        "query_dns",
        "query_asset_info",
        "query_vuln_info",
        "query_threat_intel",
        "query_history_cases",
        # response
        "block_ip",
        "block_domain",
        "isolate_host",
        "quarantine_file",
        "block_process",
        "scan_host_for_virus",
        "disable_account",
        "force_logout",
        "reset_password",
        "revoke_token",
        "create_ticket",
        "notify_security_team",
        "update_source_event_disposition",
        # verification
        "check_ip_block_status",
        "check_domain_block_status",
        "check_host_isolation_status",
        "check_file_quarantine_status",
        "check_process_block_status",
        "check_virus_scan_status",
        "check_account_status",
        "check_new_alerts",
        "check_traffic_drop",
        # rollback
        "unblock_ip",
        "unblock_domain",
        "restore_account",
        "cancel_host_isolation",
        "restore_file",
        "close_false_positive_ticket",
    }
)


def test_baseline_contains_all_required_tools_without_locking_count() -> None:
    missing = REQUIRED_BASELINE_NAMES - BASELINE_TOOL_NAMES
    assert not missing, f"missing baseline tools: {sorted(missing)}"
    # Providers may extend; the absolute count must not be treated as a contract.
    assert len(BASELINE_TOOL_METAS) >= len(REQUIRED_BASELINE_NAMES)


def test_baseline_tool_names_are_unique() -> None:
    names = [m.tool_name for m in BASELINE_TOOL_METAS]
    assert len(names) == len(set(names))


def test_query_tools_have_null_category_and_empty_owners() -> None:
    for meta in BASELINE_TOOL_METAS:
        if meta.tool_category is ToolCategory.QUERY:
            assert meta.action_category is None
            assert meta.supported_execution_owners == []
            assert meta.routing_kind is RoutingKind.TOOL_PROVIDER_ONLY


def test_verification_tools_have_verification_category_and_empty_owners() -> None:
    for meta in BASELINE_TOOL_METAS:
        if meta.tool_category is ToolCategory.VERIFICATION:
            assert meta.action_category is ActionCategory.VERIFICATION
            assert meta.supported_execution_owners == []
            assert meta.routing_kind is RoutingKind.TOOL_PROVIDER_ONLY


def test_response_action_levels_match_issue_guidance() -> None:
    idx = baseline_tool_index()
    assert idx["notify_security_team"].action_level is ActionLevel.L1
    assert idx["create_ticket"].action_level is ActionLevel.L1
    assert idx["block_ip"].action_level is ActionLevel.L2
    assert idx["block_domain"].action_level is ActionLevel.L2
    assert idx["isolate_host"].action_level is ActionLevel.L3
    assert idx["quarantine_file"].action_level is ActionLevel.L3
    assert idx["block_process"].action_level is ActionLevel.L3
    assert idx["disable_account"].action_level is ActionLevel.L3
    assert idx["reset_password"].action_level is ActionLevel.L4
    assert idx["revoke_token"].action_level is ActionLevel.L4


def test_rollback_mapping_is_bidirectional() -> None:
    idx = baseline_tool_index()
    for response_name, rollback_name in RESPONSE_ROLLBACK_MAP.items():
        response = idx[response_name]
        assert response.rollback_tool_name == rollback_name
        assert response.rollback_supported is True
        rollback = idx[rollback_name]
        assert rollback.tool_category is ToolCategory.ROLLBACK
        assert ROLLBACK_SOURCE_MAP[rollback_name] == response_name
    # Inverse agrees exactly.
    assert ROLLBACK_SOURCE_MAP == {v: k for k, v in RESPONSE_ROLLBACK_MAP.items()}


def test_non_mapped_response_tools_do_not_invent_rollback() -> None:
    idx = baseline_tool_index()
    for name in ("block_process", "scan_host_for_virus", "force_logout", "reset_password"):
        assert idx[name].rollback_supported is False
        assert idx[name].rollback_tool_name is None


def test_virtual_disposition_meta_is_not_tool_provider_executable() -> None:
    meta = baseline_tool_index()[TERMINAL_DISPOSITION_TOOL]
    assert meta.routing_kind is RoutingKind.DISPOSITION_ONLY
    assert meta.executable is False
    assert meta.async_mode is False
    assert meta.supported_execution_owners == [ExecutionOwner.XDR_MANAGED]
    assert (
        meta.required_disposition_intent_by_owner[ExecutionOwner.XDR_MANAGED]
        is DispositionIntentKind.EVENT_STATUS_UPDATE
    )
    with pytest.raises(WrongExecutionChannelError) as exc:
        ensure_tool_provider_executable(meta)
    assert exc.value.error_code == "wrong_execution_channel"


def test_owner_routed_response_maps_intents_correctly() -> None:
    meta = baseline_tool_index()["block_ip"]
    assert meta.routing_kind is RoutingKind.OWNER_ROUTED
    assert set(meta.supported_execution_owners) == {
        ExecutionOwner.XDR_MANAGED,
        ExecutionOwner.DIRECT_TOOL,
    }
    assert (
        meta.required_disposition_intent_by_owner[ExecutionOwner.XDR_MANAGED]
        is DispositionIntentKind.ENTITY_ACTION_SUBMIT
    )
    assert (
        meta.required_disposition_intent_by_owner[ExecutionOwner.DIRECT_TOOL]
        is DispositionIntentKind.EXECUTION_RESULT_RECORD
    )


def test_action_freezes_exactly_one_execution_owner() -> None:
    """A single Action may not dispatch DIRECT_TOOL and XDR_MANAGED together."""
    meta = baseline_tool_index()["block_ip"]
    # Meta may advertise both owners; Action freezes exactly one.
    assert len(meta.supported_execution_owners) == 2
    owner = meta.freeze_execution_owner(ExecutionOwner.DIRECT_TOOL)
    action = Action(
        action_id="act-00000001",
        event_id="evt-20260101-0a1b2c3d",
        plan_revision=1,
        action_fingerprint="fp-block",
        action_category=ActionCategory.RESPONSE,
        action_name="Block IP",
        tool_name="block_ip",
        action_level=ActionLevel.L2,
        execution_owner=owner,
        writeback_required=True,
        writeback_applicable=True,
        writeback_readiness=WritebackReadiness.READY,
        reason="contain",
    )
    assert action.execution_owner is ExecutionOwner.DIRECT_TOOL
    # No second owner field exists on Action — dual dispatch is structurally impossible.
    assert not hasattr(action, "execution_owners")
    # A different Action may freeze the other owner; never both on one Action.
    other = meta.freeze_execution_owner(ExecutionOwner.XDR_MANAGED)
    assert other is ExecutionOwner.XDR_MANAGED
    with pytest.raises(ValueError, match="not supported"):
        # Query/empty-owner tools reject any freeze attempt.
        baseline_tool_index()["query_dns"].freeze_execution_owner(ExecutionOwner.DIRECT_TOOL)


def test_direct_tool_must_not_map_to_entity_action_submit() -> None:
    with pytest.raises(ValidationError):
        ToolMeta(
            tool_name="block_ip",
            tool_category=ToolCategory.RESPONSE,
            action_category=ActionCategory.RESPONSE,
            routing_kind=RoutingKind.OWNER_ROUTED,
            supported_execution_owners=[ExecutionOwner.DIRECT_TOOL],
            required_disposition_intent_by_owner={
                ExecutionOwner.DIRECT_TOOL: DispositionIntentKind.ENTITY_ACTION_SUBMIT
            },
            side_effect_level=SideEffectLevel.MEDIUM,
            action_level=ActionLevel.L2,
        )


def test_provider_binding_channel_rules() -> None:
    ProviderToolBinding(
        tool_name="block_ip",
        provider_name="mock",
        execution_owner=ExecutionOwner.DIRECT_TOOL,
        execution_channel=ExecutionChannel.TOOL_PROVIDER,
    )
    ProviderToolBinding(
        tool_name="block_ip",
        provider_name="mock_xdr",
        execution_owner=ExecutionOwner.XDR_MANAGED,
        execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
    )
    with pytest.raises(ValidationError):
        ProviderToolBinding(
            tool_name="block_ip",
            provider_name="bad",
            execution_owner=ExecutionOwner.DIRECT_TOOL,
            execution_channel=ExecutionChannel.DISPOSITION_ADAPTER,
        )


def test_required_policy_keeps_writeback_obligation_when_capability_unknown() -> None:
    """Business required stays required; readiness is blocked by capability."""
    policy = DispositionPolicy.REQUIRED
    manifest = CapabilityManifest(
        provider_name="live_unverified",
        online=True,
        source_read=CapabilityState.SUPPORTED,
        event_disposition=CapabilityState.UNKNOWN,  # live default
        entity_response=CapabilityState.UNKNOWN,
    )
    # Obligation is policy-derived, never capability-derived.
    writeback_required = policy is DispositionPolicy.REQUIRED
    assert writeback_required is True
    readiness = WritebackReadiness(manifest.writeback_readiness_for_required())
    assert readiness is WritebackReadiness.CAPABILITY_UNKNOWN

    action = Action(
        action_id="act-00000002",
        event_id="evt-20260101-0a1b2c3d",
        plan_revision=1,
        action_fingerprint="fp-block-2",
        action_category=ActionCategory.RESPONSE,
        action_name="Block IP",
        tool_name="block_ip",
        action_level=ActionLevel.L2,
        execution_owner=ExecutionOwner.XDR_MANAGED,
        writeback_required=True,
        writeback_applicable=True,
        writeback_readiness=readiness,
        writeback_block_reason="capability_unknown",
        reason="still obligated",
    )
    assert action.writeback_required is True
    assert action.writeback_readiness is WritebackReadiness.CAPABILITY_UNKNOWN


def test_required_policy_blocks_on_unsupported_without_downgrading() -> None:
    manifest = CapabilityManifest(
        provider_name="live",
        online=True,
        event_disposition=CapabilityState.UNSUPPORTED,
    )
    readiness = WritebackReadiness(manifest.writeback_readiness_for_required())
    assert readiness is WritebackReadiness.CAPABILITY_UNSUPPORTED
    # Still required — never silently rewritten to not_required.
    assert DispositionPolicy.REQUIRED.value == "required"


def test_offline_connector_blocks_readiness_even_if_disposition_supported() -> None:
    """Online is a separate dimension from event_disposition capability."""
    manifest = CapabilityManifest(
        provider_name="offline",
        online=False,
        event_disposition=CapabilityState.SUPPORTED,
        entity_response=CapabilityState.SUPPORTED,
    )
    readiness = WritebackReadiness(manifest.writeback_readiness_for_required())
    assert readiness is WritebackReadiness.CONNECTOR_UNAVAILABLE


def test_sync_response_tools_do_not_claim_async_job_output() -> None:
    idx = baseline_tool_index()
    assert idx["notify_security_team"].async_mode is False
    assert idx["notify_security_team"].output_schema == {}
    assert idx["create_ticket"].async_mode is False
    assert idx["create_ticket"].output_schema == {}
    assert idx["block_ip"].async_mode is True
    assert idx["block_ip"].output_schema == {"$ref": "ActionExecutionJob"}


def test_capability_manifest_allows_source_kind_and_native_type() -> None:
    manifest = CapabilityManifest(
        provider_name="mock_xdr",
        online=True,
        source_read=CapabilityState.SUPPORTED,
        event_disposition=CapabilityState.SUPPORTED,
        entity_response=CapabilityState.SUPPORTED,
        allowed_intents=[DispositionIntentKind.EVENT_STATUS_UPDATE],
        allowed_operations=["set_event_disposition"],
        allowed_source_kinds=[SourceObjectKind.INCIDENT],
        allowed_native_source_object_types=["xdr_incident"],
        supports_status_query=True,
        supports_idempotency=True,
        supports_lookup_by_idempotency=True,
        bindings=[
            CapabilityBindingEntry(
                intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
                operation_code="set_event_disposition",
                source_kind=SourceObjectKind.INCIDENT,
                native_source_object_type="xdr_incident",
                state=CapabilityState.SUPPORTED,
            )
        ],
    )
    assert (
        manifest.allows(
            intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
            operation_code="set_event_disposition",
            source_kind=SourceObjectKind.INCIDENT,
            native_source_object_type="xdr_incident",
        )
        is CapabilityState.SUPPORTED
    )
    assert (
        manifest.allows(
            intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
            operation_code="set_event_disposition",
            source_kind=SourceObjectKind.ALERT,
        )
        is CapabilityState.UNSUPPORTED
    )
    assert (
        manifest.allows(
            intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
            operation_code="set_event_disposition",
            source_kind=SourceObjectKind.INCIDENT,
            native_source_object_type="other_type",
        )
        is CapabilityState.UNSUPPORTED
    )


@pytest.mark.parametrize(
    ("generic_state", "specific_state", "expected"),
    [
        (CapabilityState.SUPPORTED, CapabilityState.UNSUPPORTED, CapabilityState.UNSUPPORTED),
        (CapabilityState.UNSUPPORTED, CapabilityState.SUPPORTED, CapabilityState.SUPPORTED),
        (CapabilityState.UNKNOWN, CapabilityState.SUPPORTED, CapabilityState.SUPPORTED),
    ],
)
def test_capability_manifest_most_specific_binding_wins(
    generic_state: CapabilityState,
    specific_state: CapabilityState,
    expected: CapabilityState,
) -> None:
    """A specific (source_kind-scoped) binding must win over a generic one,
    in either direction — specific UNSUPPORTED is never overridden by generic
    SUPPORTED, and a specific SUPPORTED is not blocked by a generic non-support."""
    manifest = CapabilityManifest(
        provider_name="mixed",
        online=True,
        allowed_intents=[DispositionIntentKind.EVENT_STATUS_UPDATE],
        allowed_operations=["set_event_disposition"],
        bindings=[
            CapabilityBindingEntry(
                intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
                operation_code="set_event_disposition",
                state=generic_state,
            ),
            CapabilityBindingEntry(
                intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
                operation_code="set_event_disposition",
                source_kind=SourceObjectKind.INCIDENT,
                state=specific_state,
            ),
        ],
    )
    assert (
        manifest.allows(
            intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
            operation_code="set_event_disposition",
            source_kind=SourceObjectKind.INCIDENT,
        )
        is expected
    )
    # An unrelated source_kind never matches the specific binding — falls back
    # to the generic one.
    assert (
        manifest.allows(
            intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
            operation_code="set_event_disposition",
            source_kind=SourceObjectKind.ALERT,
        )
        is generic_state
    )


def test_capability_manifest_equal_specificity_conflict_fails_closed() -> None:
    """Two equally-specific bindings that disagree must resolve to UNSUPPORTED."""
    manifest = CapabilityManifest(
        provider_name="conflicting",
        online=True,
        allowed_intents=[DispositionIntentKind.EVENT_STATUS_UPDATE],
        allowed_operations=["set_event_disposition"],
        bindings=[
            CapabilityBindingEntry(
                intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
                operation_code="set_event_disposition",
                source_kind=SourceObjectKind.INCIDENT,
                state=CapabilityState.SUPPORTED,
            ),
            CapabilityBindingEntry(
                intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
                operation_code="set_event_disposition",
                source_kind=SourceObjectKind.INCIDENT,
                state=CapabilityState.UNSUPPORTED,
            ),
        ],
    )
    assert (
        manifest.allows(
            intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
            operation_code="set_event_disposition",
            source_kind=SourceObjectKind.INCIDENT,
        )
        is CapabilityState.UNSUPPORTED
    )


def test_side_effect_input_rejects_illegal_target_type() -> None:
    """target_type is a closed set — cross-wiring a tool's target kind must fail."""
    from app.tools.inputs import BlockIpInput, CheckStatusInput

    with pytest.raises(ValidationError):
        BlockIpInput(target_type="account", target="1.2.3.4")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BlockIpInput(target_type="not_a_real_type", target="1.2.3.4")  # type: ignore[arg-type]
    # Base class default is fine when unset; explicit override to a legal
    # literal for the specific subclass is rejected too (must stay "ip").
    assert BlockIpInput(target="1.2.3.4").target_type == "ip"
    # Generic verification input accepts any of the closed target kinds.
    CheckStatusInput(target_type="account", target="user-1")
    with pytest.raises(ValidationError):
        CheckStatusInput(target_type="bogus", target="user-1")  # type: ignore[arg-type]


def test_disposition_command_writeback_field_allowlist() -> None:
    """Outbound disposition envelope must not carry analysis / free-form fields."""
    cmd = DispositionCommand(
        disposition_id="disp-00000001",
        action_id="act-00000001",
        closure_cycle=1,
        intent_kind=DispositionIntentKind.EVENT_STATUS_UPDATE,
        source_locator=SourceObjectLocator(
            source_product="mock_xdr",
            source_tenant_id="t1",
            connector_id="conn-1",
            source_kind=SourceObjectKind.INCIDENT,
            source_object_id="INC-1",
        ),
        operation_code="set_event_disposition",
        operation_params=SetEventDispositionParams(target_disposition=SourceDisposition.CONTAINED),
        operator_id="system",
        idempotency_key="idem-1",
        execution_owner=ExecutionOwner.XDR_MANAGED,
    )
    payload = cmd.model_dump()
    forbidden = {"parameters", "reason", "prompt", "evidence", "report", "decision_trace"}
    assert forbidden.isdisjoint(payload.keys())
    with pytest.raises(ValidationError):
        DispositionCommand(**payload, reason="leaked analysis")  # type: ignore[call-arg]


def test_tool_result_status_and_job_fields() -> None:
    result = ToolResult(
        call_id="call-1",
        tool_name="block_ip",
        provider_name="mock",
        status=ToolResultStatus.ACCEPTED,
        job_id="job-1",
        provider_job_id="ext-99",
        confidence=0.9,
    )
    assert result.job_id == "job-1"
    assert result.provider_job_id == "ext-99"
    with pytest.raises(ValidationError):
        ToolResult(
            call_id="c",
            tool_name="block_ip",
            provider_name="mock",
            status="ok",  # type: ignore[arg-type]
        )


def test_query_input_primary_keys() -> None:
    tr = TimeRange(start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 2, tzinfo=UTC))
    TOOL_INPUT_MODELS["query_account_login"](account="svc", time_range=tr)
    with pytest.raises(ValidationError):
        QueryNetworkFlowInput(time_range=tr)  # missing src_ip and dst_ip
    QueryNetworkFlowInput(time_range=tr, src_ip="203.0.113.9")


def test_provider_may_extend_but_not_overwrite_different_schema() -> None:
    base = baseline_tool_index()
    extra = ToolMeta(
        tool_name="vendor_custom_query",
        tool_category=ToolCategory.QUERY,
        routing_kind=RoutingKind.TOOL_PROVIDER_ONLY,
        side_effect_level=SideEffectLevel.NONE,
        description="provider extension",
    )
    merged = merge_provider_tools(base, [extra])
    assert "vendor_custom_query" in merged
    assert len(merged) == len(base) + 1

    clash = ToolMeta(
        tool_name="block_ip",
        tool_category=ToolCategory.RESPONSE,
        action_category=ActionCategory.RESPONSE,
        routing_kind=RoutingKind.OWNER_ROUTED,
        supported_execution_owners=[ExecutionOwner.DIRECT_TOOL],
        required_disposition_intent_by_owner={
            ExecutionOwner.DIRECT_TOOL: DispositionIntentKind.EXECUTION_RESULT_RECORD
        },
        side_effect_level=SideEffectLevel.MEDIUM,
        action_level=ActionLevel.L2,
        input_schema={"type": "object", "properties": {"different": {"type": "string"}}},
    )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        merge_provider_tools(base, [clash])


def test_export_schemas_match_baseline_set(tmp_path: Path) -> None:
    written = export_baseline_tool_schemas(tmp_path)
    file_names = {p.stem for p in written}
    assert file_names == BASELINE_TOOL_NAMES
    # Async tools reference ActionExecutionJob.
    block = (tmp_path / "block_ip.json").read_text(encoding="utf-8")
    assert "ActionExecutionJob" in block
    virtual = (tmp_path / TERMINAL_DISPOSITION_TOOL).with_suffix(".json")
    doc = __import__("json").loads(virtual.read_text(encoding="utf-8"))
    assert doc["executable"] is False
    assert doc["routing_kind"] == "disposition_only"


def test_contracts_tools_directory_matches_baseline() -> None:
    """Committed export under contracts/schemas/tools/ stays in sync."""
    root = Path(__file__).resolve().parents[3] / "contracts" / "schemas" / "tools"
    assert root.is_dir(), "run export_baseline_tool_schemas to populate contracts/schemas/tools"
    on_disk = {p.stem for p in root.glob("*.json")}
    assert on_disk == BASELINE_TOOL_NAMES
