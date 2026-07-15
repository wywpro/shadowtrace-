"""Structured error taxonomy: ShadowTraceError, registry, and helpers (ISSUE-008).

All Agent / tool / service / API failures should surface as ``ShadowTraceError``
(or a registered ``error_code``) so callers can classify and decide retry safely.
"""

from __future__ import annotations

from typing import Any

from app.models.enums import ErrorCategory

# --------------------------------------------------------------------------- #
# Category → default retryability (简介 §4.9)
# transient and *partial* llm/tool are retryable; permanent / user_input /
# guardrail / budget / system are not.
# --------------------------------------------------------------------------- #

_CATEGORY_RETRYABLE_DEFAULT: dict[ErrorCategory, bool] = {
    ErrorCategory.TRANSIENT: True,
    ErrorCategory.PERMANENT: False,
    ErrorCategory.USER_INPUT: False,
    ErrorCategory.SYSTEM: False,
    ErrorCategory.LLM: True,
    ErrorCategory.TOOL: True,
    ErrorCategory.BUDGET: False,
    ErrorCategory.GUARDRAIL: False,
}

# Codes that must never be auto-retried even when their category default is True
# (writeback step 8: permission_denied / invalid_operation / version_conflict;
# unknown_delivery = verify-first; writeback_pending = state conflict).
_NON_AUTO_RETRY_CODES: frozenset[str] = frozenset(
    {
        "permission_denied",
        "invalid_operation",
        "version_conflict",
        "unknown_delivery",
        "writeback_pending",
        "writeback_failed",  # only Adapter-gated safe retry may re-enqueue
        "auth_error",
        "validation_error",
        "tool_validation_error",
        "unsupported",
        "capacity_limit_exceeded",
        "wrong_execution_channel",
        "llm_invalid_json",  # repair path is explicit, not blind retry
    }
)


def _category_retryable_default(category: ErrorCategory) -> bool:
    return _CATEGORY_RETRYABLE_DEFAULT[category]


def _retryable_for_code(error_code: str, category_default: bool) -> bool:
    if error_code in _NON_AUTO_RETRY_CODES:
        return False
    return category_default


# --------------------------------------------------------------------------- #
# Registry — every documented / in-tree error_code (snake_case noun phrases)
# --------------------------------------------------------------------------- #

ERROR_CODE_REGISTRY: dict[str, ErrorCategory] = {
    # API / auth / validation
    "event_not_found": ErrorCategory.USER_INPUT,
    "not_found": ErrorCategory.USER_INPUT,
    "approval_required": ErrorCategory.PERMANENT,
    "validation_error": ErrorCategory.USER_INPUT,
    "unauthorized": ErrorCategory.USER_INPUT,
    "forbidden": ErrorCategory.USER_INPUT,
    "internal_error": ErrorCategory.SYSTEM,
    # State machine
    "invalid_state_transition": ErrorCategory.PERMANENT,
    "invalid_verdict_status_combination": ErrorCategory.PERMANENT,
    # Tools / LLM / budget / guardrail
    "tool_timeout": ErrorCategory.TOOL,
    "timeout": ErrorCategory.TRANSIENT,
    "llm_invalid_json": ErrorCategory.LLM,
    "budget_exceeded": ErrorCategory.BUDGET,
    "guardrail_failed": ErrorCategory.GUARDRAIL,
    "working_memory_unauthorized_write": ErrorCategory.GUARDRAIL,
    "tool_not_found": ErrorCategory.USER_INPUT,
    "tool_already_registered": ErrorCategory.USER_INPUT,
    "tool_validation_error": ErrorCategory.USER_INPUT,
    "wrong_execution_channel": ErrorCategory.PERMANENT,
    "auth_error": ErrorCategory.TOOL,
    "rate_limited": ErrorCategory.TRANSIENT,
    "remote_error": ErrorCategory.TRANSIENT,
    "http_5xx": ErrorCategory.TRANSIENT,
    "circuit_open": ErrorCategory.TRANSIENT,
    "unsupported": ErrorCategory.PERMANENT,
    "capacity_limit_exceeded": ErrorCategory.TOOL,
    "react_action_denied": ErrorCategory.GUARDRAIL,
    # Writeback / disposition (step 8)
    "permission_denied": ErrorCategory.PERMANENT,
    "invalid_operation": ErrorCategory.PERMANENT,
    "version_conflict": ErrorCategory.PERMANENT,
    "unknown_delivery": ErrorCategory.TRANSIENT,  # verify-first; not auto-retry
    "writeback_pending": ErrorCategory.PERMANENT,  # state conflict, not system fault
    "writeback_failed": ErrorCategory.TOOL,
    "writeback_conflict": ErrorCategory.PERMANENT,
    "writeback_unsupported": ErrorCategory.PERMANENT,
    "disposition_permission_denied": ErrorCategory.USER_INPUT,
    # Product / API surface codes referenced in the plan
    "investigation_in_progress": ErrorCategory.PERMANENT,
    "storyline_not_ready": ErrorCategory.USER_INPUT,
    "qa_unavailable": ErrorCategory.TRANSIENT,
    # Generic dependency / domain defaults used by subclasses
    "dependency_unavailable": ErrorCategory.TRANSIENT,
    "tool_execution_error": ErrorCategory.TOOL,
    "llm_error": ErrorCategory.LLM,
    # Mock XDR (ISSUE-010) — fixture-only codes, not vendor facts
    "invalid_cursor": ErrorCategory.USER_INPUT,
    "unauthorized_field": ErrorCategory.USER_INPUT,
    "mock_validation_error": ErrorCategory.USER_INPUT,
    "idempotency_key_reuse": ErrorCategory.USER_INPUT,
    "disposition_id_reuse": ErrorCategory.USER_INPUT,
    # Adapters (ISSUE-012)
    "adapter_not_found": ErrorCategory.USER_INPUT,
    "adapter_validation_error": ErrorCategory.USER_INPUT,
    # Startup / runtime configuration (ISSUE-093 §5)
    "configuration_error": ErrorCategory.SYSTEM,
}


def register_error_code(code: str, category: ErrorCategory) -> None:
    """Register or update an ``error_code`` → ``ErrorCategory`` mapping."""
    if not code or not code.replace("_", "").isalnum() or code != code.lower():
        raise ValueError(f"error_code must be snake_case: {code!r}")
    ERROR_CODE_REGISTRY[code] = category


# --------------------------------------------------------------------------- #
# Base exception
# --------------------------------------------------------------------------- #


class ShadowTraceError(Exception):
    """Unified structured exception for ShadowTrace."""

    status_code: int = 500
    default_error_code: str = "internal_error"
    default_category: ErrorCategory = ErrorCategory.SYSTEM
    default_retryable: bool | None = None

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str | None = None,
        category: ErrorCategory | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.error_code = error_code or self.default_error_code
        # Prefer explicit category; otherwise align with the registry when the
        # code is known so ToolExecutionError(..., error_code="rate_limited")
        # classifies as transient, not the subclass default (tool).
        if category is not None:
            self.category = category
        elif self.error_code in ERROR_CODE_REGISTRY:
            self.category = ERROR_CODE_REGISTRY[self.error_code]
        else:
            self.category = self.default_category
        self.message = message or self.error_code.replace("_", " ")
        # API / workflow handlers historically read ``error_message``.
        self.error_message = self.message
        self.details = dict(details or {})

        if retryable is not None:
            self.retryable = retryable
        elif self.default_retryable is not None and error_code is None:
            # Subclass default applies only when the caller did not override the
            # code; overridden codes follow registry category + overlays.
            self.retryable = _retryable_for_code(self.error_code, self.default_retryable)
        else:
            cat_default = _category_retryable_default(self.category)
            self.retryable = _retryable_for_code(self.error_code, cat_default)

        super().__init__(self.message)

    def to_response(self) -> dict[str, Any]:
        """Serialize to the unified API error body (简介 §4.2)."""
        return {
            "error_code": self.error_code,
            "error_message": self.message,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Fixed subclasses (9) + ISSUE-004 API domain errors
# --------------------------------------------------------------------------- #


class ValidationError(ShadowTraceError):
    """Malformed / rejected user or request input."""

    status_code = 422
    default_error_code = "validation_error"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


class InvalidStateTransitionError(ShadowTraceError):
    """Illegal EventStatus / sub-state / job / outbox / writeback edge."""

    status_code = 400
    default_error_code = "invalid_state_transition"
    default_category = ErrorCategory.PERMANENT
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        current: Any | None = None,
        target: Any | None = None,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        category: ErrorCategory | None = None,
        retryable: bool | None = None,
    ) -> None:
        merged = {
            **(details or {}),
            **({"current": getattr(current, "value", current)} if current is not None else {}),
            **({"target": getattr(target, "value", target)} if target is not None else {}),
        }
        self.current = current
        self.target = target
        super().__init__(
            message,
            error_code=error_code,
            category=category,
            retryable=retryable,
            details=merged,
        )


class InvalidVerdictStatusCombinationError(ShadowTraceError):
    """``FinalVerdict`` incompatible with the current EventStatus / plan shape."""

    status_code = 400
    default_error_code = "invalid_verdict_status_combination"
    default_category = ErrorCategory.PERMANENT
    default_retryable = False


class ToolExecutionError(ShadowTraceError):
    """ToolProvider / ToolExecutor failure."""

    status_code = 502
    default_error_code = "tool_execution_error"
    default_category = ErrorCategory.TOOL
    default_retryable = True


class LLMError(ShadowTraceError):
    """LLMProvider failure (ISSUE-027 subclasses this further)."""

    status_code = 502
    default_error_code = "llm_error"
    default_category = ErrorCategory.LLM
    default_retryable = True


class BudgetExceededError(ShadowTraceError):
    """Token / cost budget exhausted."""

    status_code = 429
    default_error_code = "budget_exceeded"
    default_category = ErrorCategory.BUDGET
    default_retryable = False


class GuardrailViolationError(ShadowTraceError):
    """Policy / schema / ownership / sanitization guardrail blocked the op."""

    status_code = 403
    default_error_code = "guardrail_failed"
    default_category = ErrorCategory.GUARDRAIL
    default_retryable = False


class DependencyUnavailableError(ShadowTraceError):
    """Downstream dependency temporarily unavailable."""

    status_code = 503
    default_error_code = "dependency_unavailable"
    default_category = ErrorCategory.TRANSIENT
    default_retryable = True


class InternalError(ShadowTraceError):
    """Unexpected internal failure."""

    status_code = 500
    default_error_code = "internal_error"
    default_category = ErrorCategory.SYSTEM
    default_retryable = False


class EventNotFoundError(ShadowTraceError):
    """Security event id not found (ISSUE-004)."""

    status_code = 404
    default_error_code = "event_not_found"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


class ApprovalRequiredError(ShadowTraceError):
    """Action requires human approval before execution (ISSUE-004)."""

    status_code = 409
    default_error_code = "approval_required"
    default_category = ErrorCategory.PERMANENT
    default_retryable = False


# Writeback / disposition HTTP domain errors (ISSUE-004 codes; registered above).


class WritebackPendingError(ShadowTraceError):
    status_code = 409
    default_error_code = "writeback_pending"
    default_category = ErrorCategory.PERMANENT
    default_retryable = False


class WritebackFailedError(ShadowTraceError):
    status_code = 409
    default_error_code = "writeback_failed"
    default_category = ErrorCategory.TOOL
    # Blind auto-retry is forbidden; OUTBOX/Worker may re-enqueue only when the
    # Adapter explicitly allows a safe retry (see WRITEBACK_STATUS_TRANSITIONS).
    default_retryable = False


class WritebackConflictError(ShadowTraceError):
    status_code = 409
    default_error_code = "writeback_conflict"
    default_category = ErrorCategory.PERMANENT
    default_retryable = False


class WritebackUnsupportedError(ShadowTraceError):
    status_code = 422
    default_error_code = "writeback_unsupported"
    default_category = ErrorCategory.PERMANENT
    default_retryable = False


class DispositionPermissionDenied(ShadowTraceError):
    status_code = 403
    default_error_code = "disposition_permission_denied"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


class ResourceNotFoundError(ShadowTraceError):
    """Generic 404 for non-event resources (jobs, dispositions, etc.)."""

    status_code = 404
    default_error_code = "not_found"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


class AdapterNotFoundError(ShadowTraceError):
    """No adapter registered under the requested name (ISSUE-012)."""

    status_code = 404
    default_error_code = "adapter_not_found"
    default_category = ErrorCategory.USER_INPUT
    default_retryable = False


class ConfigurationError(ShadowTraceError):
    """Illegal runtime configuration (ISSUE-093 §5).

    Raised from ``Settings`` validation / application lifespan startup to
    fail-closed BEFORE serving traffic — e.g. ``app_env=production`` combined
    with any mock/simulation mode. Never retryable; the process must not start.
    """

    status_code = 500
    default_error_code = "configuration_error"
    default_category = ErrorCategory.SYSTEM
    default_retryable = False


# Backward-compat alias used by API modules that still say ``APIError``.
APIError = ShadowTraceError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Map an exception to an ``ErrorCategory``.

    Known ``ShadowTraceError`` → its ``category``.
    Registered ``error_code`` attribute → registry lookup.
    ``TimeoutError`` / ``ConnectionError`` → transient.
    Everything else → system.
    """
    if isinstance(exc, ShadowTraceError):
        return exc.category

    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and code in ERROR_CODE_REGISTRY:
        return ERROR_CODE_REGISTRY[code]

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ErrorCategory.TRANSIENT

    return ErrorCategory.SYSTEM


def is_retryable(exc: BaseException) -> bool:
    """Whether automatic retry is allowed for this exception."""
    if isinstance(exc, ShadowTraceError):
        return exc.retryable

    code = getattr(exc, "error_code", None)
    if isinstance(code, str):
        category = ERROR_CODE_REGISTRY.get(code, classify_exception(exc))
        return _retryable_for_code(code, _category_retryable_default(category))

    return _category_retryable_default(classify_exception(exc))
