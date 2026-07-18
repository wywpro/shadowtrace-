"""Shared LLM contracts, fallback orchestration, hooks, and audit (ISSUE-027)."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import LLMError, ShadowTraceError
from app.db import models as orm

logger = logging.getLogger(__name__)


class LLMMessage(BaseModel):
    """One vendor-neutral chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class LLMResponse(BaseModel):
    """Normalized response returned to every Agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    parsed: BaseModel | None = None
    model_name: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    fallback_level: int = Field(default=0, ge=0, le=2)
    degraded_reason: str | None = None


class LLMTimeoutError(LLMError):
    default_error_code = "llm_timeout"


class LLMAuthError(LLMError):
    default_error_code = "llm_auth_error"
    default_retryable = False


class LLMRateLimitedError(LLMError):
    default_error_code = "llm_rate_limited"


class LLMInvalidJSONError(LLMError):
    default_error_code = "llm_invalid_json"
    default_retryable = False

    def __init__(self, message: str, *, invalid_content: str, validation_error: str) -> None:
        self.invalid_content = invalid_content
        self.validation_error = validation_error
        super().__init__(
            message,
            details={"validation_error": validation_error},
        )


class LLMProviderError(LLMError):
    default_error_code = "llm_provider_error"


class LLMAuditError(LLMError):
    default_error_code = "llm_audit_error"
    default_retryable = False


@dataclass(frozen=True)
class ProviderResponse:
    """Internal normalized result from one actual provider request."""

    content: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMCallAudit(BaseModel):
    """Minimal audit payload; prompt text and credentials are intentionally absent."""

    event_id: str
    agent_name: str
    prompt_key: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int | None = None
    fallback_level: int = 0
    status: str


@runtime_checkable
class LLMCallAuditRecorder(Protocol):
    async def record(self, entry: LLMCallAudit) -> None: ...


class InMemoryLLMCallAuditRecorder:
    """Deterministic audit recorder for unit tests and local adapters."""

    def __init__(self) -> None:
        self.entries: list[LLMCallAudit] = []

    async def record(self, entry: LLMCallAudit) -> None:
        self.entries.append(entry)


class SQLAlchemyLLMCallAuditRecorder:
    """Persist each request attempt in its own short transaction."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record(self, entry: LLMCallAudit) -> None:
        async with self._session_factory() as session:
            session.add(orm.LLMCallLog(**entry.model_dump()))
            await session.commit()


@runtime_checkable
class ConvergenceGuardHook(Protocol):
    def record_step(self, event_id: str, step_type: str, signature: str) -> None: ...

    def should_stop(self, event_id: str) -> Any: ...


@runtime_checkable
class MessageBudgeterHook(Protocol):
    def fit(self, messages: list[LLMMessage], max_input_tokens: int) -> list[LLMMessage]: ...


@runtime_checkable
class BudgetMeterHook(Protocol):
    """ISSUE-029 BudgetService surface used by LLMClient."""

    async def check(self, event_id: str, agent_name: str) -> None: ...

    async def charge_llm(
        self,
        event_id: str,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Any: ...


BudgetCallback: TypeAlias = Callable[..., Awaitable[None] | None]

# CJK Unified Ideographs + common CJK punctuation / compatibility blocks.
_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf"
    r"\u4e00-\u9fff\uf900-\ufaff]"
)


def estimate_tokens(text: str) -> int:
    """Deterministic heuristic token estimate (ISSUE-031).

    CJK characters count as 1 token each; remaining characters count as
    ``ceil(n / 4)`` tokens. Empty text is 0.
    """

    if not text:
        return 0
    cjk = 0
    other = 0
    for char in text:
        if _CJK_RE.fullmatch(char):
            cjk += 1
        else:
            other += 1
    return cjk + (other + 3) // 4


def _fallback_level(model_index: int) -> int:
    return 0 if model_index == 0 else 1


def _plain_truncate(messages: Sequence[LLMMessage], max_chars: int) -> list[LLMMessage]:
    """Keep the first system message and newest context within a deterministic cap."""

    if max_chars <= 0:
        return []
    copied = [message.model_copy(deep=True) for message in messages]
    if sum(len(message.content) for message in copied) <= max_chars:
        return copied

    system = next((message for message in copied if message.role == "system"), None)
    remaining = max_chars - (len(system.content) if system else 0)
    if remaining < 0 and system is not None:
        return [system.model_copy(update={"content": system.content[:max_chars]})]

    tail: list[LLMMessage] = []
    for message in reversed(copied):
        if message is system:
            continue
        if remaining <= 0:
            break
        content = message.content
        if len(content) > remaining:
            content = content[-remaining:]
        tail.append(message.model_copy(update={"content": content}))
        remaining -= len(content)
    tail.reverse()
    return ([system] if system is not None else []) + tail


class BaseLLMClient(ABC):
    """Provider-independent chat flow with repair, fallback, guard, and audit."""

    def __init__(
        self,
        *,
        primary_model: str,
        fallback_models: Sequence[str] = (),
        timeout_seconds: float = 30.0,
        audit_recorder: LLMCallAuditRecorder | None = None,
        convergence_guard: ConvergenceGuardHook | None = None,
        budget_callback: BudgetCallback | None = None,
        budget_service: BudgetMeterHook | None = None,
        message_budgeter: MessageBudgeterHook | None = None,
        max_input_tokens: int = 16_000,
    ) -> None:
        if not primary_model.strip():
            raise ValueError("primary_model must not be empty")
        self.primary_model = primary_model.strip()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        deduplicated_fallbacks = dict.fromkeys(
            model.strip()
            for model in fallback_models
            if model.strip() and model.strip() != self.primary_model
        )
        self.fallback_models = tuple(deduplicated_fallbacks)
        self.timeout_seconds = timeout_seconds
        if audit_recorder is None:
            raise ValueError("audit_recorder is required")
        self.audit_recorder = audit_recorder
        self.convergence_guard = convergence_guard
        self.budget_callback = budget_callback
        self.budget_service = budget_service
        self.message_budgeter = message_budgeter
        self.max_input_tokens = max_input_tokens

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        event_id: str,
        agent_name: str,
        prompt_key: str,
        scenario_id: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        del scenario_id  # Used by MockLLMClient; never inferred from prompt content.
        chat_started = time.perf_counter()
        self._validate_context(event_id, agent_name, prompt_key, messages)
        prepared = self._fit_messages(messages)
        require_json = json_mode or response_model is not None
        last_error: LLMError | None = None

        for model_index, model_name in enumerate((self.primary_model, *self.fallback_models)):
            level = _fallback_level(model_index)
            try:
                raw, parsed = await self._attempt(
                    prepared,
                    model_name=model_name,
                    event_id=event_id,
                    agent_name=agent_name,
                    prompt_key=prompt_key,
                    fallback_level=level,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=require_json,
                    response_model=response_model,
                )
            except LLMInvalidJSONError as exc:
                last_error = exc
                try:
                    repaired, parsed = await self._repair_json(
                        prepared,
                        invalid_content=exc.invalid_content,
                        validation_error=exc.validation_error,
                        model_name=model_name,
                        event_id=event_id,
                        agent_name=agent_name,
                        prompt_key=prompt_key,
                        fallback_level=level,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_model=response_model,
                    )
                    raw = repaired
                except LLMError:
                    raise
            except LLMAuditError:
                raise
            except LLMError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                continue
            except ShadowTraceError:
                raise

            response = LLMResponse(
                content=raw.content,
                parsed=parsed,
                model_name=raw.model_name,
                prompt_tokens=raw.prompt_tokens,
                completion_tokens=raw.completion_tokens,
                total_tokens=raw.total_tokens or raw.prompt_tokens + raw.completion_tokens,
                latency_ms=max(0, round((time.perf_counter() - chat_started) * 1000)),
                fallback_level=level,
                degraded_reason=(
                    f"primary model unavailable: {type(last_error).__name__}" if level else None
                ),
            )
            return response

        if last_error is not None:
            raise last_error
        raise LLMProviderError("no LLM models are configured")

    @abstractmethod
    async def _request(
        self,
        messages: list[LLMMessage],
        *,
        model_name: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ProviderResponse:
        """Perform exactly one provider request."""

    async def aclose(self) -> None:
        """Release provider resources when the concrete client owns them."""

        return None

    def _fit_messages(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        if self.message_budgeter is not None:
            return self.message_budgeter.fit(list(messages), self.max_input_tokens)
        # Conservative 2 chars/token estimate: safe for CJK (~1–2) and English (~4).
        # Over-provision via message_budgeter when precise token counts are needed.
        return _plain_truncate(messages, self.max_input_tokens * 2)

    async def _attempt(
        self,
        messages: list[LLMMessage],
        *,
        model_name: str,
        event_id: str,
        agent_name: str,
        prompt_key: str,
        fallback_level: int,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        response_model: type[BaseModel] | None,
    ) -> tuple[ProviderResponse, BaseModel | None]:
        self._check_convergence(event_id, agent_name, prompt_key, model_name)
        started = time.perf_counter()
        raw: ProviderResponse | None = None
        status = "error"
        error: BaseException | None = None
        try:
            await self._check_budget(event_id=event_id, agent_name=agent_name)
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    raw = await self._request(
                        messages,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
            except TimeoutError as exc:
                raise LLMTimeoutError(
                    "LLM request timed out",
                    details={"model_name": model_name},
                ) from exc
        except LLMError as exc:
            status = exc.error_code
            error = exc
        except ShadowTraceError as exc:
            status = exc.error_code
            error = exc
        except Exception as exc:
            status = "llm_provider_error"
            error = LLMProviderError("unexpected LLM provider failure")
            error.__cause__ = exc

        parsed: BaseModel | None = None
        if error is None:
            assert raw is not None
            try:
                await self._charge_budget(raw, event_id=event_id, agent_name=agent_name)
                parsed = self._parse(raw.content, response_model) if json_mode else None
                status = "success"
            except ShadowTraceError as exc:
                status = exc.error_code
                error = exc
            except Exception as exc:
                status = "llm_provider_error"
                error = LLMProviderError("LLM post-processing failed")
                error.__cause__ = exc

        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        await self._record_audit(
            LLMCallAudit(
                event_id=event_id,
                agent_name=agent_name,
                prompt_key=prompt_key,
                model_name=model_name,
                prompt_tokens=raw.prompt_tokens if raw else 0,
                completion_tokens=raw.completion_tokens if raw else 0,
                total_tokens=(
                    raw.total_tokens or raw.prompt_tokens + raw.completion_tokens if raw else 0
                ),
                latency_ms=latency_ms,
                fallback_level=fallback_level,
                status=status,
            )
        )
        if error is not None:
            raise error
        assert raw is not None
        return raw, parsed

    async def _repair_json(
        self,
        messages: list[LLMMessage],
        *,
        invalid_content: str,
        validation_error: str,
        model_name: str,
        event_id: str,
        agent_name: str,
        prompt_key: str,
        fallback_level: int,
        temperature: float,
        max_tokens: int,
        response_model: type[BaseModel] | None,
    ) -> tuple[ProviderResponse, BaseModel | None]:
        schema = (
            response_model.model_json_schema() if response_model is not None else {"type": "object"}
        )
        repair = LLMMessage(
            role="user",
            content=(
                "Return corrected JSON only. The previous output was invalid.\n"
                f"Validation error: {validation_error}\n"
                f"Required schema: {json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
                f"Invalid output: {invalid_content}"
            ),
        )
        repaired_messages = [
            *messages,
            LLMMessage(role="assistant", content=invalid_content),
            repair,
        ]
        return await self._attempt(
            self._fit_messages(repaired_messages),
            model_name=model_name,
            event_id=event_id,
            agent_name=agent_name,
            prompt_key=prompt_key,
            fallback_level=fallback_level,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            response_model=response_model,
        )

    @staticmethod
    def _parse(content: str, response_model: type[BaseModel] | None) -> BaseModel | None:
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON must be an object")
            return response_model.model_validate(payload) if response_model is not None else None
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            validation_error = (
                json.dumps(exc.errors(include_input=False, include_url=False), ensure_ascii=False)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            raise LLMInvalidJSONError(
                "LLM returned invalid structured output",
                invalid_content=content,
                validation_error=validation_error,
            ) from exc

    def _check_convergence(
        self, event_id: str, agent_name: str, prompt_key: str, model_name: str
    ) -> None:
        guard = self.convergence_guard
        if guard is None:
            return
        signature = f"{agent_name}:{prompt_key}:{model_name}"
        guard.record_step(event_id, "llm_call", signature)
        decision = guard.should_stop(event_id)
        if bool(getattr(decision, "stop", False)):
            reason = str(getattr(decision, "reason", "convergence_guard"))
            raise LLMProviderError(
                "LLM request blocked by convergence guard",
                retryable=False,
                details={"reason": reason},
            )

    async def _check_budget(self, *, event_id: str, agent_name: str) -> None:
        if self.budget_service is None:
            return
        await self.budget_service.check(event_id, agent_name)

    async def _charge_budget(
        self, response: LLMResponse | ProviderResponse, *, event_id: str, agent_name: str
    ) -> None:
        if self.budget_service is not None:
            await self.budget_service.charge_llm(
                event_id,
                agent_name,
                response.model_name,
                response.prompt_tokens,
                response.completion_tokens,
            )
            return
        if self.budget_callback is None:
            return
        result = self.budget_callback(
            event_id=event_id,
            agent_name=agent_name,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        if inspect.isawaitable(result):
            await result

    async def _record_audit(self, entry: LLMCallAudit) -> None:
        try:
            await self.audit_recorder.record(entry)
        except Exception as exc:
            raise LLMAuditError(
                "failed to persist LLM call audit",
                details={"event_id": entry.event_id, "prompt_key": entry.prompt_key},
            ) from exc

    @staticmethod
    def _validate_context(
        event_id: str, agent_name: str, prompt_key: str, messages: list[LLMMessage]
    ) -> None:
        missing = [
            field
            for field, value in (
                ("event_id", event_id),
                ("agent_name", agent_name),
                ("prompt_key", prompt_key),
            )
            if not value.strip()
        ]
        if missing:
            raise ValueError(f"required LLM context is empty: {', '.join(missing)}")
        if not messages:
            raise ValueError("messages must not be empty")


def default_golden_root() -> Path:
    return Path(__file__).with_name("golden")


__all__ = [
    "BaseLLMClient",
    "BudgetCallback",
    "BudgetMeterHook",
    "ConvergenceGuardHook",
    "InMemoryLLMCallAuditRecorder",
    "LLMAuditError",
    "LLMAuthError",
    "LLMCallAudit",
    "LLMCallAuditRecorder",
    "LLMInvalidJSONError",
    "LLMMessage",
    "LLMProviderError",
    "LLMRateLimitedError",
    "LLMResponse",
    "LLMTimeoutError",
    "MessageBudgeterHook",
    "ProviderResponse",
    "SQLAlchemyLLMCallAuditRecorder",
    "default_golden_root",
    "estimate_tokens",
]
