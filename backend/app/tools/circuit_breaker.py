"""Per-tool circuit breaker for ToolExecutor (ISSUE-024)."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Three-state breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self._clock = clock or time.monotonic
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_request(self) -> bool:
        """Return False when the breaker is open and recovery has not elapsed."""

        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.OPEN:
            opened_at = self._opened_at
            if opened_at is not None and (self._clock() - opened_at >= self.recovery_timeout_s):
                self._state = CircuitState.HALF_OPEN
                self._half_open_probe_in_flight = False
                return True
            return False
        # HALF_OPEN: allow a single probe request.
        if self._half_open_probe_in_flight:
            return False
        self._half_open_probe_in_flight = True
        return True

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
            self._half_open_probe_in_flight = False
            return
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
        self._half_open_probe_in_flight = False


class CircuitBreakerRegistry:
    """Maintain one breaker instance per tool name (process-local)."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, tool_name: str) -> CircuitBreaker:
        breaker = self._breakers.get(tool_name)
        if breaker is None:
            breaker = CircuitBreaker(
                failure_threshold=self._failure_threshold,
                recovery_timeout_s=self._recovery_timeout_s,
                clock=self._clock,
            )
            self._breakers[tool_name] = breaker
        return breaker


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
]
