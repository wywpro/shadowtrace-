"""Vendor-neutral LLM client contracts and factory (ISSUE-027)."""

from app.core.llm.base import (
    BaseLLMClient,
    LLMAuditError,
    LLMAuthError,
    LLMInvalidJSONError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitedError,
    LLMResponse,
    LLMTimeoutError,
)
from app.core.llm.factory import get_llm_client
from app.core.llm.mock_client import MockLLMClient

__all__ = [
    "BaseLLMClient",
    "LLMAuditError",
    "LLMAuthError",
    "LLMInvalidJSONError",
    "LLMMessage",
    "LLMProviderError",
    "LLMRateLimitedError",
    "LLMResponse",
    "LLMTimeoutError",
    "MockLLMClient",
    "get_llm_client",
]
