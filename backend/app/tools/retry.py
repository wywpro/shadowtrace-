"""Retry policy for ToolExecutor (ISSUE-024)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff policy for retryable tool failures."""

    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_multiplier: float = 2.0

    def delay_for_attempt(self, attempt: int) -> float:
        """Return sleep seconds before retry attempt ``attempt`` (1-based)."""

        if attempt <= 0:
            return 0.0
        return self.backoff_base * (self.backoff_multiplier ** (attempt - 1))


__all__ = ["RetryPolicy"]
