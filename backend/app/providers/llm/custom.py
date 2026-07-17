"""Extension contract for non-OpenAI-compatible LLM providers."""

from __future__ import annotations

from abc import abstractmethod

from app.core.llm.base import BaseLLMClient, LLMMessage, ProviderResponse


class CustomLLMClient(BaseLLMClient):
    """Implement one request without leaking provider fields into Agent code."""

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
        """Map the custom protocol to :class:`ProviderResponse`."""


__all__ = ["CustomLLMClient"]
