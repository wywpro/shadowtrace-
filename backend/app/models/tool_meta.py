"""Open tool contracts: ToolMeta, CapabilityManifest, ToolResult (ISSUE-006).

Mock and future live Providers share these internal schemas. Vendor-specific
fields may appear only in the Provider mapping layer and in ``raw_result`` —
never as frozen ``operation_code`` on ``ToolMeta``.

``writeback_required`` is a *business* obligation derived from event
``disposition_policy`` (and category rules). It must never be reverse-driven by
capability; readiness carries the technical blocking reason instead.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import (
    ActionCategory,
    ActionExecutionPhase,
    ActionLevel,
    CapabilityState,
    DispositionIntentKind,
    ExecutionOwner,
    SourceObjectKind,
    ToolCategory,
)
from app.models.execution import TargetExecutionResult

TERMINAL_DISPOSITION_TOOL = "update_source_event_disposition"


class RoutingKind(StrEnum):
    """How an Action for this tool is routed at execution time."""

    TOOL_PROVIDER_ONLY = "tool_provider_only"
    OWNER_ROUTED = "owner_routed"
    DISPOSITION_ONLY = "disposition_only"


class ExecutionChannel(StrEnum):
    """Concrete channel declared on a ProviderToolBinding."""

    TOOL_PROVIDER = "tool_provider"
    DISPOSITION_ADAPTER = "disposition_adapter"


class SideEffectLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolResultStatus(StrEnum):
    ACCEPTED = "accepted"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    UNKNOWN = "unknown"
    VALIDATION_ERROR = "validation_error"
    AUTH_ERROR = "auth_error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    REMOTE_ERROR = "remote_error"
    CIRCUIT_OPEN = "circuit_open"
    UNSUPPORTED = "unsupported"


class WrongExecutionChannelError(Exception):
    """Raised when ToolExecutor (or equivalent) is asked to run a non-executable meta."""

    error_code = "wrong_execution_channel"

    def __init__(self, tool_name: str, *, routing_kind: RoutingKind) -> None:
        self.tool_name = tool_name
        self.routing_kind = routing_kind
        super().__init__(
            f"tool {tool_name!r} routing_kind={routing_kind.value} is not "
            "executable via ToolProvider"
        )


class ToolMeta(BaseModel):
    """Canonical tool metadata. Provider-agnostic; no vendor operation_code."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_category: ToolCategory
    description: str = ""
    # Null for pure query tools (they never produce an Action).
    action_category: ActionCategory | None = None
    routing_kind: RoutingKind
    supported_execution_owners: list[ExecutionOwner] = Field(default_factory=list)
    # Owner -> required disposition intent when that owner is frozen on an Action.
    required_disposition_intent_by_owner: dict[ExecutionOwner, DispositionIntentKind] = Field(
        default_factory=dict
    )
    required_capabilities: list[str] = Field(default_factory=list)
    side_effect_level: SideEffectLevel = SideEffectLevel.NONE
    action_level: ActionLevel = ActionLevel.L0
    idempotency: bool = True
    async_mode: bool = False
    rollback_supported: bool = False
    rollback_tool_name: str | None = None
    default_timeout_s: float = 30.0
    execution_phase: ActionExecutionPhase = ActionExecutionPhase.IMMEDIATE
    activation_condition: str | None = None
    # False for disposition_only virtual metas (catalog/approval only).
    executable: bool = True
    target_types: list[str] = Field(default_factory=list)
    # JSON Schema fragments (optional; full per-tool schemas live under contracts/).
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _routing_and_owners_are_consistent(self) -> ToolMeta:
        owners = list(self.supported_execution_owners)
        intents = self.required_disposition_intent_by_owner

        if self.tool_category is ToolCategory.QUERY:
            if self.action_category is not None:
                raise ValueError("query tools must have action_category=null")
            if owners:
                raise ValueError("query tools must have empty supported_execution_owners")
            if self.routing_kind is not RoutingKind.TOOL_PROVIDER_ONLY:
                raise ValueError("query tools must use routing_kind=tool_provider_only")
            if self.side_effect_level is not SideEffectLevel.NONE:
                raise ValueError("query tools must have side_effect_level=none")

        if self.tool_category is ToolCategory.VERIFICATION:
            if self.action_category is not ActionCategory.VERIFICATION:
                raise ValueError("verification tools require action_category=verification")
            if owners:
                raise ValueError("verification tools must have empty supported_execution_owners")
            if self.routing_kind is not RoutingKind.TOOL_PROVIDER_ONLY:
                raise ValueError("verification tools must use routing_kind=tool_provider_only")

        if self.tool_name == TERMINAL_DISPOSITION_TOOL:
            if self.routing_kind is not RoutingKind.DISPOSITION_ONLY:
                raise ValueError(
                    f"{TERMINAL_DISPOSITION_TOOL} must be routing_kind=disposition_only"
                )
            if self.action_category is not ActionCategory.RESPONSE:
                raise ValueError(f"{TERMINAL_DISPOSITION_TOOL} must be action_category=response")
            if self.execution_phase is not ActionExecutionPhase.POST_VERIFY:
                raise ValueError(f"{TERMINAL_DISPOSITION_TOOL} must be execution_phase=POST_VERIFY")
            if self.activation_condition != "after_effect_resolution":
                raise ValueError(
                    f"{TERMINAL_DISPOSITION_TOOL} requires "
                    "activation_condition=after_effect_resolution"
                )
            if self.executable:
                raise ValueError(
                    f"{TERMINAL_DISPOSITION_TOOL} is virtual and must set executable=false"
                )
            if self.async_mode:
                raise ValueError(f"{TERMINAL_DISPOSITION_TOOL} must not declare async_mode execute")
            if owners != [ExecutionOwner.XDR_MANAGED]:
                raise ValueError(f"{TERMINAL_DISPOSITION_TOOL} supports only XDR_MANAGED")
            expected = DispositionIntentKind.EVENT_STATUS_UPDATE
            if intents.get(ExecutionOwner.XDR_MANAGED) is not expected:
                raise ValueError(
                    f"{TERMINAL_DISPOSITION_TOOL} requires XDR_MANAGED→EVENT_STATUS_UPDATE"
                )
            return self

        if self.tool_category in (ToolCategory.RESPONSE, ToolCategory.ROLLBACK):
            if self.routing_kind is not RoutingKind.OWNER_ROUTED:
                raise ValueError("side-effect response/rollback tools must use owner_routed")
            if self.action_category is None:
                raise ValueError("response/rollback tools require a non-null action_category")
            if not owners:
                raise ValueError(
                    "owner_routed tools must declare at least one supported_execution_owner"
                )
            # A single Action still freezes exactly one owner; meta may list both.
            if len(set(owners)) != len(owners):
                raise ValueError("supported_execution_owners must be unique")
            for owner in owners:
                intent = intents.get(owner)
                if intent is None:
                    raise ValueError(
                        f"required_disposition_intent_by_owner missing for {owner.value}"
                    )
                if owner is ExecutionOwner.XDR_MANAGED:
                    if self.tool_category is ToolCategory.RESPONSE and intent is not (
                        DispositionIntentKind.ENTITY_ACTION_SUBMIT
                    ):
                        raise ValueError("response XDR_MANAGED must map to ENTITY_ACTION_SUBMIT")
                    if self.tool_category is ToolCategory.ROLLBACK and intent is not (
                        DispositionIntentKind.COMPENSATION_RECORD
                    ):
                        raise ValueError("rollback XDR_MANAGED must map to COMPENSATION_RECORD")
                if owner is ExecutionOwner.DIRECT_TOOL:
                    if intent is not DispositionIntentKind.EXECUTION_RESULT_RECORD:
                        raise ValueError(
                            "DIRECT_TOOL must map to EXECUTION_RESULT_RECORD "
                            "(never ENTITY_ACTION_SUBMIT)"
                        )

        if self.routing_kind is RoutingKind.DISPOSITION_ONLY and self.executable:
            raise ValueError("disposition_only metas must set executable=false")

        return self

    def freeze_execution_owner(self, owner: ExecutionOwner) -> ExecutionOwner:
        """Validate and return the single owner frozen onto an Action.

        Forbidden to "dispatch both" — callers pass exactly one owner.
        """
        if owner not in self.supported_execution_owners:
            raise ValueError(f"execution_owner {owner.value} is not supported by {self.tool_name}")
        return owner


class ProviderToolBinding(BaseModel):
    """Concrete Provider channel binding for one (tool, owner) pair."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    provider_name: str
    execution_owner: ExecutionOwner
    execution_channel: ExecutionChannel
    capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _channel_matches_owner(self) -> ProviderToolBinding:
        if self.execution_owner is ExecutionOwner.DIRECT_TOOL:
            if self.execution_channel is not ExecutionChannel.TOOL_PROVIDER:
                raise ValueError("DIRECT_TOOL bindings must use execution_channel=tool_provider")
        if self.execution_owner is ExecutionOwner.XDR_MANAGED:
            # Entity actions / disposition updates go through the disposition adapter.
            if self.execution_channel is not ExecutionChannel.DISPOSITION_ADAPTER:
                raise ValueError(
                    "XDR_MANAGED bindings must use execution_channel=disposition_adapter"
                )
        return self


class CapabilityBindingEntry(BaseModel):
    """Per intent+operation(+source) capability probe result."""

    model_config = ConfigDict(extra="forbid")

    intent_kind: DispositionIntentKind
    operation_code: str
    source_kind: SourceObjectKind | None = None
    native_source_object_type: str | None = None
    state: CapabilityState = CapabilityState.UNKNOWN


class CapabilityManifest(BaseModel):
    """Provider/Adapter capability surface (internal ShadowTrace terms only).

    Online / readable / writable / executable are expressed separately so a
    readable-but-not-writable Adapter cannot be mistaken for a full loop.
    Live defaults every probe to UNKNOWN until verified.
    """

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    # Connectivity vs capability dimensions (must not be collapsed into one flag).
    online: bool = False
    source_read: CapabilityState = CapabilityState.UNKNOWN
    event_disposition: CapabilityState = CapabilityState.UNKNOWN
    entity_response: CapabilityState = CapabilityState.UNKNOWN
    allowed_intents: list[DispositionIntentKind] = Field(default_factory=list)
    allowed_operations: list[str] = Field(default_factory=list)
    allowed_target_types: list[str] = Field(default_factory=list)
    allowed_source_kinds: list[SourceObjectKind] = Field(default_factory=list)
    allowed_native_source_object_types: list[str] | None = None
    supports_status_query: bool = False
    supports_lookup_by_idempotency: bool = False
    supports_idempotency: bool = False
    supports_concurrency_control: bool = False
    supports_fencing: bool = False
    allowed_execution_channels: list[ExecutionChannel] = Field(default_factory=list)
    bindings: list[CapabilityBindingEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _execution_guarantees_are_consistent(self) -> CapabilityManifest:
        if self.supports_fencing and not self.supports_concurrency_control:
            raise ValueError("fencing requires concurrency control")
        if len(set(self.allowed_execution_channels)) != len(self.allowed_execution_channels):
            raise ValueError("allowed_execution_channels must be unique")
        return self

    def allows(
        self,
        *,
        intent_kind: DispositionIntentKind,
        operation_code: str,
        source_kind: SourceObjectKind | None = None,
        native_source_object_type: str | None = None,
    ) -> CapabilityState:
        """Resolve binding state for intent+operation(+optional source filters).

        Most-specific binding wins. Specificity is the count of
        ``(source_kind, native_source_object_type)`` the binding declares AND
        that also match the query; a binding that declares a dimension which
        does not match the query is excluded entirely (it never wins and can
        never "leak" UNKNOWN/SUPPORTED from an unrelated probe). When several
        bindings tie for the highest specificity but disagree on state, the
        result is UNSUPPORTED — a specific UNSUPPORTED binding must never be
        silently overridden by a more generic SUPPORTED one, and genuine
        conflicts must fail closed rather than optimistically picking SUPPORTED.
        """
        if intent_kind not in self.allowed_intents:
            return CapabilityState.UNSUPPORTED
        if operation_code not in self.allowed_operations:
            return CapabilityState.UNSUPPORTED
        if source_kind is not None and self.allowed_source_kinds:
            if source_kind not in self.allowed_source_kinds:
                return CapabilityState.UNSUPPORTED
        if (
            native_source_object_type is not None
            and self.allowed_native_source_object_types is not None
        ):
            if native_source_object_type not in self.allowed_native_source_object_types:
                return CapabilityState.UNSUPPORTED

        candidates = [
            b
            for b in self.bindings
            if b.intent_kind is intent_kind and b.operation_code == operation_code
        ]

        scored: list[tuple[int, CapabilityState]] = []
        for binding in candidates:
            specificity = 0
            if binding.source_kind is not None:
                if source_kind is None or binding.source_kind != source_kind:
                    continue
                specificity += 1
            if binding.native_source_object_type is not None:
                if (
                    native_source_object_type is None
                    or binding.native_source_object_type != native_source_object_type
                ):
                    continue
                specificity += 1
            scored.append((specificity, binding.state))

        if not scored:
            # Allowed lists passed but no probed binding → live-default UNKNOWN.
            return CapabilityState.UNKNOWN

        max_specificity = max(specificity for specificity, _ in scored)
        winning_states = {state for specificity, state in scored if specificity == max_specificity}
        if len(winning_states) == 1:
            return next(iter(winning_states))
        # Conflicting equally-specific bindings → fail closed.
        return CapabilityState.UNSUPPORTED

    def writeback_readiness_for_required(self) -> str:
        """Map connectivity + event_disposition probe to WritebackReadiness value.

        Returned as the WritebackReadiness *value* so callers that do not want
        to import the enum cycle can still assert. Online / readable / writable
        stay separate dimensions: an offline connector blocks with
        ``connector_unavailable`` even if a prior probe said SUPPORTED. Live
        UNKNOWN/UNSUPPORTED block readiness without rewriting writeback_required.
        """
        if not self.online:
            return "connector_unavailable"
        if self.event_disposition is CapabilityState.SUPPORTED:
            return "ready"
        if self.event_disposition is CapabilityState.UNSUPPORTED:
            return "capability_unsupported"
        return "capability_unknown"


class ToolResult(BaseModel):
    """Normalized tool call result. Provider codes stay local (never outbound)."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool_name: str
    provider_name: str
    status: ToolResultStatus
    # ShadowTrace-internal pre-persisted job id (async path).
    job_id: str | None = None
    provider_job_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    target_results: list[TargetExecutionResult] = Field(default_factory=list)
    provider_code: str | None = None
    provider_message: str | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)
    error_detail: str | None = None
    execution_time_ms: int | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


def ensure_tool_provider_executable(meta: ToolMeta) -> None:
    """Fail-closed guard used by ToolExecutor (ISSUE-021) and schema tests.

    Disposition-only virtual metas are catalog/approval only — never executed
    through the ToolProvider channel.
    """
    if meta.routing_kind is RoutingKind.DISPOSITION_ONLY or not meta.executable:
        raise WrongExecutionChannelError(meta.tool_name, routing_kind=meta.routing_kind)


def default_response_intents() -> dict[ExecutionOwner, DispositionIntentKind]:
    """Ordinary response tool: XDR_MANAGED→entity submit, DIRECT_TOOL→result record."""
    return {
        ExecutionOwner.XDR_MANAGED: DispositionIntentKind.ENTITY_ACTION_SUBMIT,
        ExecutionOwner.DIRECT_TOOL: DispositionIntentKind.EXECUTION_RESULT_RECORD,
    }


def default_rollback_intents() -> dict[ExecutionOwner, DispositionIntentKind]:
    return {
        ExecutionOwner.XDR_MANAGED: DispositionIntentKind.COMPENSATION_RECORD,
        ExecutionOwner.DIRECT_TOOL: DispositionIntentKind.EXECUTION_RESULT_RECORD,
    }
