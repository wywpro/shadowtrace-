"""Configuration-driven LLM client factory."""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings, get_settings
from app.core.llm.base import (
    BaseLLMClient,
    BudgetCallback,
    ConvergenceGuardHook,
    LLMCallAuditRecorder,
    MessageBudgeterHook,
    SQLAlchemyLLMCallAuditRecorder,
)
from app.core.llm.mock_client import MockLLMClient
from app.db.session import get_session_factory
from app.providers.llm.custom import CustomLLMClient
from app.providers.llm.openai_compatible import OpenAICompatibleLLMClient

CustomFactory = Callable[..., CustomLLMClient]


def _models(value: str) -> tuple[str, ...]:
    return tuple(model.strip() for model in value.split(",") if model.strip())


def get_llm_client(
    *,
    settings: Settings | None = None,
    audit_recorder: LLMCallAuditRecorder | None = None,
    convergence_guard: ConvergenceGuardHook | None = None,
    budget_callback: BudgetCallback | None = None,
    message_budgeter: MessageBudgeterHook | None = None,
    custom_factory: CustomFactory | None = None,
) -> BaseLLMClient:
    """Build the configured provider without any implicit Mock fallback."""

    config = settings or get_settings()
    recorder = audit_recorder or SQLAlchemyLLMCallAuditRecorder(get_session_factory())
    fallback_models = _models(config.llm_fallback_models)
    mode = config.llm_mode.strip().lower()
    if mode == "mock":
        return MockLLMClient(
            primary_model=config.llm_primary_model,
            fallback_models=fallback_models,
            timeout_seconds=config.llm_timeout_seconds,
            audit_recorder=recorder,
            convergence_guard=convergence_guard,
            budget_callback=budget_callback,
            message_budgeter=message_budgeter,
        )
    if mode == "openai_compatible":
        return OpenAICompatibleLLMClient(
            base_url=config.llm_api_base_url,
            api_key=config.llm_api_key,
            primary_model=config.llm_primary_model,
            fallback_models=fallback_models,
            timeout_seconds=config.llm_timeout_seconds,
            audit_recorder=recorder,
            convergence_guard=convergence_guard,
            budget_callback=budget_callback,
            message_budgeter=message_budgeter,
        )
    if mode == "custom":
        if custom_factory is None:
            raise ValueError("custom LLM mode requires custom_factory")
        return custom_factory(
            base_url=config.llm_api_base_url,
            api_key=config.llm_api_key,
            primary_model=config.llm_primary_model,
            fallback_models=fallback_models,
            timeout_seconds=config.llm_timeout_seconds,
            audit_recorder=recorder,
            convergence_guard=convergence_guard,
            budget_callback=budget_callback,
            message_budgeter=message_budgeter,
        )
    raise ValueError(f"unsupported LLM_MODE: {config.llm_mode!r}")


__all__ = ["CustomFactory", "get_llm_client"]
