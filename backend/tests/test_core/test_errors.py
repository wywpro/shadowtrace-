"""Structured error taxonomy tests (ISSUE-008)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.errors import (
    ERROR_CODE_REGISTRY,
    ApprovalRequiredError,
    BudgetExceededError,
    DependencyUnavailableError,
    EventNotFoundError,
    GuardrailViolationError,
    InternalError,
    InvalidStateTransitionError,
    InvalidVerdictStatusCombinationError,
    LLMError,
    ShadowTraceError,
    ToolExecutionError,
    ValidationError,
    classify_exception,
    is_retryable,
    register_error_code,
)
from app.core.llm.base import (
    LLMAuditError,
    LLMAuthError,
    LLMInvalidJSONError,
    LLMProviderError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
from app.models.enums import ErrorCategory, EventStatus
from app.models.tool_meta import RoutingKind, WrongExecutionChannelError

# Documented / in-tree codes that must appear in the registry (acceptance §2).
_REQUIRED_DOCUMENTED_CODES: frozenset[str] = frozenset(
    {
        "event_not_found",
        "invalid_state_transition",
        "invalid_verdict_status_combination",
        "tool_timeout",
        "llm_invalid_json",
        "budget_exceeded",
        "guardrail_failed",
        "working_memory_unauthorized_write",
        "permission_denied",
        "invalid_operation",
        "version_conflict",
        "rate_limited",
        "unknown_delivery",
        "writeback_pending",
        "writeback_failed",
        "writeback_conflict",
        "writeback_unsupported",
        "disposition_permission_denied",
        "approval_required",
        "not_found",
        "validation_error",
        "unauthorized",
        "forbidden",
        "internal_error",
        "investigation_in_progress",
        "wrong_execution_channel",
        "storyline_not_ready",
        "qa_unavailable",
        "auth_error",
        "remote_error",
        "http_5xx",
        "circuit_open",
        "unsupported",
        "capacity_limit_exceeded",
        "react_action_denied",
        "tool_not_found",
        "tool_already_registered",
        "tool_validation_error",
        "timeout",
        "dependency_unavailable",
        "tool_execution_error",
        "llm_error",
        "invalid_cursor",
        "unauthorized_field",
        "mock_validation_error",
        "idempotency_key_reuse",
        "adapter_not_found",
        "adapter_validation_error",
    }
)

_FIXED_SUBCLASSES: list[tuple[type[ShadowTraceError], ErrorCategory, bool]] = [
    (ValidationError, ErrorCategory.USER_INPUT, False),
    (InvalidStateTransitionError, ErrorCategory.PERMANENT, False),
    (InvalidVerdictStatusCombinationError, ErrorCategory.PERMANENT, False),
    (ToolExecutionError, ErrorCategory.TOOL, True),
    (LLMError, ErrorCategory.LLM, True),
    (BudgetExceededError, ErrorCategory.BUDGET, False),
    (GuardrailViolationError, ErrorCategory.GUARDRAIL, False),
    (DependencyUnavailableError, ErrorCategory.TRANSIENT, True),
    (InternalError, ErrorCategory.SYSTEM, False),
]


def test_error_category_has_eight_values() -> None:
    assert {m.value for m in ErrorCategory} == {
        "transient",
        "permanent",
        "user_input",
        "system",
        "llm",
        "tool",
        "budget",
        "guardrail",
    }


def test_nine_subclasses_category_and_retryable() -> None:
    permanent = [cls for cls, cat, _ in _FIXED_SUBCLASSES if cat is ErrorCategory.PERMANENT]
    assert len(_FIXED_SUBCLASSES) == 9
    assert len(permanent) == 2
    assert set(permanent) == {
        InvalidStateTransitionError,
        InvalidVerdictStatusCombinationError,
    }

    for cls, category, retryable in _FIXED_SUBCLASSES:
        if cls is InvalidStateTransitionError:
            exc = cls("bad edge", current=EventStatus.NEW, target=EventStatus.CLOSED)
        else:
            exc = cls("probe")
        assert exc.category is category
        assert exc.retryable is retryable
        assert is_retryable(exc) is retryable
        assert classify_exception(exc) is category


def test_issue004_errors_are_shadowtrace_subclasses() -> None:
    assert issubclass(EventNotFoundError, ShadowTraceError)
    assert issubclass(ApprovalRequiredError, ShadowTraceError)
    nf = EventNotFoundError("missing")
    assert nf.error_code == "event_not_found"
    assert nf.status_code == 404
    assert is_retryable(nf) is False
    ar = ApprovalRequiredError("need approval")
    assert ar.error_code == "approval_required"
    assert ar.status_code == 409
    assert is_retryable(ar) is False


def test_to_response_matches_unified_body() -> None:
    exc = ValidationError("bad field", details={"field": "event_id"})
    body = exc.to_response()
    assert set(body.keys()) == {"error_code", "error_message", "details"}
    assert body["error_code"] == "validation_error"
    assert body["error_message"] == "bad field"
    assert body["details"] == {"field": "event_id"}


def test_invalid_state_transition_details_include_current_target() -> None:
    exc = InvalidStateTransitionError(
        "nope",
        current=EventStatus.NEW,
        target=EventStatus.CLOSED,
    )
    assert exc.details["current"] == EventStatus.NEW.value
    assert exc.details["target"] == EventStatus.CLOSED.value
    assert exc.error_code == "invalid_state_transition"
    assert isinstance(exc, ShadowTraceError)


def test_classify_exception_stdlib_and_fallback() -> None:
    assert classify_exception(TimeoutError()) is ErrorCategory.TRANSIENT
    assert classify_exception(ConnectionError("down")) is ErrorCategory.TRANSIENT
    assert classify_exception(RuntimeError("boom")) is ErrorCategory.SYSTEM
    assert is_retryable(TimeoutError()) is True
    assert is_retryable(RuntimeError("boom")) is False


def test_classify_exception_registered_attribute() -> None:
    exc = WrongExecutionChannelError(
        "update_source_event_disposition",
        routing_kind=RoutingKind.DISPOSITION_ONLY,
    )
    assert classify_exception(exc) is ErrorCategory.PERMANENT
    assert is_retryable(exc) is False


def test_is_retryable_category_defaults() -> None:
    assert is_retryable(DependencyUnavailableError("redis")) is True
    assert is_retryable(ValidationError("x")) is False
    assert is_retryable(GuardrailViolationError("blocked")) is False


def test_writeback_retry_rules() -> None:
    """Step 8: writeback code classification / retry policy."""
    assert ERROR_CODE_REGISTRY["permission_denied"] is ErrorCategory.PERMANENT
    assert ERROR_CODE_REGISTRY["invalid_operation"] is ErrorCategory.PERMANENT
    assert ERROR_CODE_REGISTRY["version_conflict"] is ErrorCategory.PERMANENT
    assert ERROR_CODE_REGISTRY["rate_limited"] is ErrorCategory.TRANSIENT
    assert ERROR_CODE_REGISTRY["http_5xx"] is ErrorCategory.TRANSIENT
    assert ERROR_CODE_REGISTRY["remote_error"] is ErrorCategory.TRANSIENT
    assert ERROR_CODE_REGISTRY["unknown_delivery"] is ErrorCategory.TRANSIENT
    assert ERROR_CODE_REGISTRY["writeback_pending"] is ErrorCategory.PERMANENT

    # Non-auto-retry overlays
    assert is_retryable(ToolExecutionError("denied", error_code="permission_denied")) is False
    assert is_retryable(ToolExecutionError("cas", error_code="version_conflict")) is False
    assert is_retryable(DependencyUnavailableError("unk", error_code="unknown_delivery")) is False
    rl = ToolExecutionError("rl", error_code="rate_limited")
    assert rl.category is ErrorCategory.TRANSIENT
    assert is_retryable(rl) is True
    http5 = ToolExecutionError("5xx", error_code="http_5xx")
    assert http5.category is ErrorCategory.TRANSIENT
    assert is_retryable(http5) is True
    # writeback_failed is a domain failure — not a blind ToolExecutor retry signal.
    from app.core.errors import WritebackFailedError

    assert is_retryable(WritebackFailedError("failed")) is False
    wb = ToolExecutionError("wb", error_code="writeback_failed")
    assert wb.category is ErrorCategory.TOOL
    assert is_retryable(wb) is False


def test_registry_covers_documented_codes() -> None:
    missing = _REQUIRED_DOCUMENTED_CODES - set(ERROR_CODE_REGISTRY)
    assert not missing, f"unregistered documented codes: {sorted(missing)}"


def test_registry_covers_in_tree_error_code_literals() -> None:
    """Every ``error_code = "..."`` / handler literal in backend must be registered."""
    root = Path(__file__).resolve().parents[2] / "app"
    pattern = re.compile(r"""error_code\s*=\s*["']([a-z][a-z0-9_]+)["']""")
    found: set[str] = set()
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        found.update(pattern.findall(text))
    # Handler string literals for auth / validation / catch-all
    found.update({"unauthorized", "forbidden", "validation_error", "internal_error"})
    missing = found - set(ERROR_CODE_REGISTRY)
    assert not missing, f"in-tree error_code literals not registered: {sorted(missing)}"


def test_register_error_code() -> None:
    code = "test_only_ephemeral_code"
    try:
        register_error_code(code, ErrorCategory.TRANSIENT)
        assert ERROR_CODE_REGISTRY[code] is ErrorCategory.TRANSIENT
        register_error_code(code, ErrorCategory.SYSTEM)
        assert ERROR_CODE_REGISTRY[code] is ErrorCategory.SYSTEM
    finally:
        ERROR_CODE_REGISTRY.pop(code, None)


def test_register_error_code_rejects_bad_names() -> None:
    with pytest.raises(ValueError):
        register_error_code("NotSnake", ErrorCategory.SYSTEM)
    with pytest.raises(ValueError):
        register_error_code("", ErrorCategory.SYSTEM)


def test_llm_error_subclass_retry_rules() -> None:
    """ISSUE-027: auth/audit/invalid-json are non-retryable; timeout/rate-limited retryable."""
    # Non-retryable
    assert is_retryable(LLMAuthError("bad credentials")) is False
    assert is_retryable(LLMAuditError("audit down")) is False
    assert (
        is_retryable(LLMInvalidJSONError("bad json", invalid_content="x", validation_error="e"))
        is False
    )
    # Retryable
    assert is_retryable(LLMTimeoutError("timed out")) is True
    assert is_retryable(LLMRateLimitedError("rate limited")) is True
    assert is_retryable(LLMProviderError("provider error")) is True

    # Defense-in-depth: generic construction with these codes also non-retryable
    assert is_retryable(LLMError("x", error_code="llm_auth_error")) is False
    assert is_retryable(LLMError("x", error_code="llm_audit_error")) is False
    assert is_retryable(LLMError("x", error_code="llm_invalid_json")) is False


def test_guardrail_working_memory_code() -> None:
    exc = GuardrailViolationError(
        "non-owner write",
        error_code="working_memory_unauthorized_write",
    )
    assert exc.category is ErrorCategory.GUARDRAIL
    assert is_retryable(exc) is False
    assert ERROR_CODE_REGISTRY["working_memory_unauthorized_write"] is ErrorCategory.GUARDRAIL
