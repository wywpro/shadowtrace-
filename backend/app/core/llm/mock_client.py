"""Deterministic, network-free MockLLM implementation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.errors import LLMError, ShadowTraceError
from app.core.llm.base import (
    BaseLLMClient,
    LLMCallAudit,
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    default_golden_root,
)


class MockLLMClient(BaseLLMClient):
    """Route golden responses by explicit prompt_key and scenario_id."""

    def __init__(self, *, golden_root: Path | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("primary_model", "mock-model")
        super().__init__(**kwargs)
        self.golden_root = (golden_root or default_golden_root()).resolve()

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
        del temperature, max_tokens
        self._validate_context(event_id, agent_name, prompt_key, messages)
        await self._check_convergence(event_id, agent_name, prompt_key, self.primary_model)
        started = time.perf_counter()
        response: LLMResponse | None = None
        status = "success"
        error: BaseException | None = None
        try:
            await self._check_budget(event_id=event_id, agent_name=agent_name)
            payload = self._load_golden(prompt_key, scenario_id)
            content_value = payload.get("content", payload)
            content = (
                content_value
                if isinstance(content_value, str)
                else json.dumps(content_value, ensure_ascii=False, sort_keys=True)
            )
            parsed = self._parse(content, response_model) if json_mode or response_model else None
            response = LLMResponse(
                content=content,
                parsed=parsed,
                model_name=str(payload.get("model_name") or self.primary_model),
                prompt_tokens=int(payload.get("prompt_tokens") or 0),
                completion_tokens=int(payload.get("completion_tokens") or 0),
                total_tokens=int(payload.get("total_tokens") or 0),
                latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
                fallback_level=2,
                degraded_reason=payload.get("degraded_reason"),
            )
            if response.total_tokens == 0:
                response.total_tokens = response.prompt_tokens + response.completion_tokens
            await self._charge_budget(response, event_id=event_id, agent_name=agent_name)
        except LLMError as exc:
            status = exc.error_code
            error = exc
        except ShadowTraceError as exc:
            status = exc.error_code
            error = exc
        except Exception as exc:
            status = "llm_provider_error"
            error = LLMProviderError("mock LLM response failed", retryable=False)
            error.__cause__ = exc

        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        await self._record_audit(
            LLMCallAudit(
                event_id=event_id,
                agent_name=agent_name,
                prompt_key=prompt_key,
                model_name=response.model_name if response else self.primary_model,
                prompt_tokens=response.prompt_tokens if response else 0,
                completion_tokens=response.completion_tokens if response else 0,
                total_tokens=response.total_tokens if response else 0,
                latency_ms=latency_ms,
                fallback_level=2,
                status=status,
            )
        )
        if error is not None:
            raise error
        assert response is not None
        response.latency_ms = latency_ms
        return response

    async def _request(
        self,
        messages: list[LLMMessage],
        *,
        model_name: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> Any:
        raise AssertionError("MockLLMClient.chat must never make provider requests")

    def _load_golden(self, prompt_key: str, scenario_id: str | None) -> dict[str, Any]:
        safe_prompt_key = self._safe_component(prompt_key, "prompt_key")
        prompt_dir = (self.golden_root / safe_prompt_key).resolve()
        if prompt_dir.parent != self.golden_root:
            raise ValueError("prompt_key escapes golden root")
        candidates = []
        if scenario_id:
            safe_scenario_id = self._safe_component(scenario_id, "scenario_id")
            candidates.append(prompt_dir / f"{safe_scenario_id}.json")
        candidates.append(prompt_dir / "default.json")
        for candidate in candidates:
            if candidate.is_file():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise LLMProviderError(
                        "mock golden response is unreadable",
                        retryable=False,
                        details={"prompt_key": prompt_key, "scenario_id": scenario_id},
                    ) from exc
                if not isinstance(data, dict):
                    raise ValueError(f"golden response must be an object: {candidate}")
                return data
        raise LLMProviderError(
            "mock golden response not found",
            retryable=False,
            details={"prompt_key": prompt_key, "scenario_id": scenario_id},
        )

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in value):
            raise ValueError(f"{field_name} must contain lowercase letters, digits, _ or -")
        return value


__all__ = ["MockLLMClient"]
