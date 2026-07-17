"""Vendor-neutral LLM provider tests (ISSUE-027)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import BudgetExceededError
from app.core.llm.base import (
    InMemoryLLMCallAuditRecorder,
    LLMAuditError,
    LLMInvalidJSONError,
    LLMMessage,
    LLMProviderError,
    LLMTimeoutError,
    SQLAlchemyLLMCallAuditRecorder,
)
from app.core.llm.factory import get_llm_client
from app.core.llm.mock_client import MockLLMClient
from app.db import models as orm
from app.providers.llm.openai_compatible import OpenAICompatibleLLMClient


class TriagePayload(BaseModel):
    event_type: str
    confidence: float


MESSAGES = [LLMMessage(role="user", content="Classify this event")]


def _response(content: str, *, model: str, prompt_tokens: int = 4) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 3,
            "total_tokens": prompt_tokens + 3,
        },
    }


def _client(
    http_client: httpx.AsyncClient,
    *,
    audit: InMemoryLLMCallAuditRecorder | None = None,
    primary_model: str = "primary-model",
    fallback_models: tuple[str, ...] = (),
    **kwargs: Any,
) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url="https://llm.example/v1",
        api_key="test-key",
        client=http_client,
        primary_model=primary_model,
        fallback_models=fallback_models,
        audit_recorder=audit,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_mock_mode_is_deterministic_and_uses_scenario_then_default(
    tmp_path: Path,
) -> None:
    golden = tmp_path / "triage_extract"
    golden.mkdir()
    (golden / "default.json").write_text(
        json.dumps({"content": {"event_type": "other", "confidence": 0.4}}),
        encoding="utf-8",
    )
    (golden / "scenario-a.json").write_text(
        json.dumps({"content": {"event_type": "account_anomaly", "confidence": 0.9}}),
        encoding="utf-8",
    )
    audit = InMemoryLLMCallAuditRecorder()
    client = MockLLMClient(golden_root=tmp_path, audit_recorder=audit)

    first = await client.chat(
        MESSAGES,
        event_id="evt-2026-mock",
        agent_name="TriageAgent",
        prompt_key="triage_extract",
        scenario_id="scenario-a",
        response_model=TriagePayload,
    )
    second = await client.chat(
        MESSAGES,
        event_id="evt-2026-mock",
        agent_name="TriageAgent",
        prompt_key="triage_extract",
        scenario_id="missing-scenario",
        response_model=TriagePayload,
    )

    assert first.parsed == TriagePayload(event_type="account_anomaly", confidence=0.9)
    assert second.parsed == TriagePayload(event_type="other", confidence=0.4)
    assert first.fallback_level == second.fallback_level == 2
    assert [entry.status for entry in audit.entries] == ["success", "success"]


@pytest.mark.asyncio
async def test_mock_mode_never_constructs_or_calls_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("mock mode attempted network access")

    monkeypatch.setattr(httpx, "AsyncClient", fail_network)
    settings = Settings(
        LLM_MODE="mock",
        LLM_PRIMARY_MODEL="mock-model",
        APP_ENV="development",
    )
    client = get_llm_client(
        settings=settings,
        audit_recorder=InMemoryLLMCallAuditRecorder(),
    )

    response = await client.chat(
        MESSAGES,
        event_id="evt-2026-no-network",
        agent_name="TriageAgent",
        prompt_key="triage_extract",
        response_model=TriagePayload,
    )
    assert response.model_name == "mock-model"


@pytest.mark.asyncio
async def test_json_mode_repairs_invalid_output_once_and_parses_model() -> None:
    calls: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(200, json=_response("not-json", model=payload["model"]))
        return httpx.Response(
            200,
            json=_response(
                '{"event_type":"host_compromise","confidence":0.87}',
                model=payload["model"],
            ),
        )

    audit = InMemoryLLMCallAuditRecorder()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        response = await _client(http_client, audit=audit).chat(
            MESSAGES,
            event_id="evt-2026-json",
            agent_name="TriageAgent",
            prompt_key="triage_extract",
            json_mode=True,
            response_model=TriagePayload,
        )

    assert response.parsed == TriagePayload(event_type="host_compromise", confidence=0.87)
    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "Return corrected JSON only" in calls[1]["messages"][-1]["content"]
    assert [entry.status for entry in audit.entries] == ["llm_invalid_json", "success"]


@pytest.mark.asyncio
async def test_json_mode_raises_after_exactly_one_failed_repair() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json=_response("still-invalid", model="primary-model"))

    audit = InMemoryLLMCallAuditRecorder()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(LLMInvalidJSONError):
            await _client(http_client, audit=audit).chat(
                MESSAGES,
                event_id="evt-2026-bad-json",
                agent_name="TriageAgent",
                prompt_key="triage_extract",
                response_model=TriagePayload,
            )

    assert attempts == 2
    assert [entry.status for entry in audit.entries] == [
        "llm_invalid_json",
        "llm_invalid_json",
    ]


@pytest.mark.asyncio
async def test_primary_timeout_falls_back_and_marks_level_one() -> None:
    audit = InMemoryLLMCallAuditRecorder()

    with respx.mock(base_url="https://llm.example/v1") as router:
        route = router.post("/chat/completions")
        route.side_effect = [
            httpx.ReadTimeout("primary timed out"),
            httpx.Response(
                200,
                json=_response("fallback answer", model="fallback-model", prompt_tokens=7),
            ),
        ]
        async with httpx.AsyncClient(base_url="https://llm.example/v1") as http_client:
            response = await _client(
                http_client,
                audit=audit,
                fallback_models=("fallback-model",),
            ).chat(
                MESSAGES,
                event_id="evt-2026-fallback",
                agent_name="RiskAgent",
                prompt_key="risk_score",
            )

    assert response.content == "fallback answer"
    assert response.model_name == "fallback-model"
    assert response.fallback_level == 1
    assert response.degraded_reason is not None
    assert [(entry.model_name, entry.status, entry.fallback_level) for entry in audit.entries] == [
        ("primary-model", "llm_timeout", 0),
        ("fallback-model", "success", 1),
    ]


@pytest.mark.asyncio
async def test_exhausted_real_models_raise_without_mock_fallback() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("unavailable", request=request)

    audit = InMemoryLLMCallAuditRecorder()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(LLMTimeoutError):
            await _client(
                http_client,
                audit=audit,
                fallback_models=("fallback-model",),
            ).chat(
                MESSAGES,
                event_id="evt-2026-exhausted",
                agent_name="ReportAgent",
                prompt_key="report_generate",
            )

    assert [entry.model_name for entry in audit.entries] == ["primary-model", "fallback-model"]
    assert all(entry.model_name != "mock-model" for entry in audit.entries)


@pytest.mark.asyncio
async def test_base_timeout_bounds_injected_or_custom_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(
            200,
            json=_response("late", model="primary-model"),
            request=request,
        )

    audit = InMemoryLLMCallAuditRecorder()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(LLMTimeoutError):
            await _client(
                http_client,
                audit=audit,
                timeout_seconds=0.01,
            ).chat(
                MESSAGES,
                event_id="evt-2026-base-timeout",
                agent_name="RiskAgent",
                prompt_key="risk_score",
            )

    assert [(entry.status, entry.fallback_level) for entry in audit.entries] == [("llm_timeout", 0)]


def test_fallback_chain_deduplicates_primary_and_repeated_models() -> None:
    client = OpenAICompatibleLLMClient(
        base_url="https://llm.example/v1",
        api_key="test-key",
        primary_model="primary-model",
        fallback_models=("primary-model", "fallback-model", "fallback-model"),
        audit_recorder=InMemoryLLMCallAuditRecorder(),
    )
    assert client.fallback_models == ("fallback-model",)


@pytest.mark.asyncio
async def test_versioned_base_url_is_preserved_with_injected_client() -> None:
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            json=_response("ok", model="primary-model"),
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        await _client(http_client, audit=InMemoryLLMCallAuditRecorder()).chat(
            MESSAGES,
            event_id="evt-2026-url",
            agent_name="TriageAgent",
            prompt_key="triage_extract",
        )

    assert requested_urls == ["https://llm.example/v1/chat/completions"]


@pytest.mark.asyncio
async def test_budget_exceeded_is_not_wrapped_or_retried() -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json=_response("ok", model="primary-model"),
            request=request,
        )

    async def charge(**kwargs: Any) -> None:
        del kwargs
        raise BudgetExceededError("budget exhausted")

    audit = InMemoryLLMCallAuditRecorder()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(BudgetExceededError):
            await _client(
                http_client,
                audit=audit,
                fallback_models=("fallback-model",),
                budget_callback=charge,
            ).chat(
                MESSAGES,
                event_id="evt-2026-budget",
                agent_name="RiskAgent",
                prompt_key="risk_score",
            )

    assert requests == 1
    assert [(entry.model_name, entry.status) for entry in audit.entries] == [
        ("primary-model", "budget_exceeded")
    ]


@pytest.mark.asyncio
async def test_audit_failure_prevents_unaudited_success() -> None:
    class BrokenAudit:
        async def record(self, entry: object) -> None:
            del entry
            raise RuntimeError("database unavailable")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response("ok", model="primary-model"),
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(LLMAuditError):
            await _client(http_client, audit=BrokenAudit()).chat(  # type: ignore[arg-type]
                MESSAGES,
                event_id="evt-2026-audit-down",
                agent_name="RiskAgent",
                prompt_key="risk_score",
            )


@pytest.mark.asyncio
async def test_guard_and_budget_hooks_run_for_each_actual_request() -> None:
    class Guard:
        def __init__(self) -> None:
            self.steps: list[tuple[str, str, str]] = []

        def record_step(self, event_id: str, step_type: str, signature: str) -> None:
            self.steps.append((event_id, step_type, signature))

        def should_stop(self, event_id: str) -> Any:
            return SimpleNamespace(stop=False, reason="none")

    charges: list[dict[str, Any]] = []

    async def charge(**kwargs: Any) -> None:
        charges.append(kwargs)

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "bad" if attempts == 1 else '{"event_type":"other","confidence":0.6}'
        return httpx.Response(200, json=_response(content, model="primary-model"))

    guard = Guard()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        await _client(
            http_client,
            audit=InMemoryLLMCallAuditRecorder(),
            convergence_guard=guard,
            budget_callback=charge,
        ).chat(
            MESSAGES,
            event_id="evt-2026-hooks",
            agent_name="TriageAgent",
            prompt_key="triage_extract",
            response_model=TriagePayload,
        )

    assert len(guard.steps) == 2
    assert all(step[1] == "llm_call" for step in guard.steps)
    assert len(charges) == 2
    assert sum(item["prompt_tokens"] for item in charges) == 8


@pytest.mark.asyncio
async def test_each_success_and_failure_attempt_is_persisted_as_orm_rows() -> None:
    class Session:
        def __init__(self, rows: list[orm.LLMCallLog]) -> None:
            self.rows = rows
            self.committed = False

        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def add(self, row: orm.LLMCallLog) -> None:
            self.rows.append(row)

        async def commit(self) -> None:
            self.committed = True

    rows: list[orm.LLMCallLog] = []
    sessions: list[Session] = []

    def session_factory() -> Session:
        session = Session(rows)
        sessions.append(session)
        return session

    event_id = "evt-2026-llm-audit"

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("primary timeout", request=request)
        return httpx.Response(200, json=_response("ok", model="fallback-model"))

    recorder = SQLAlchemyLLMCallAuditRecorder(session_factory)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        await _client(
            http_client,
            audit=recorder,
            fallback_models=("fallback-model",),
        ).chat(
            MESSAGES,
            event_id=event_id,
            agent_name="RiskAgent",
            prompt_key="risk_score",
        )

    assert [(row.prompt_key, row.status, row.fallback_level) for row in rows] == [
        ("risk_score", "llm_timeout", 0),
        ("risk_score", "success", 1),
    ]
    assert all(session.committed for session in sessions)


@pytest.mark.asyncio
async def test_unknown_mock_prompt_fails_explicitly() -> None:
    audit = InMemoryLLMCallAuditRecorder()
    client = MockLLMClient(audit_recorder=audit)
    with pytest.raises(LLMProviderError) as exc:
        await client.chat(
            MESSAGES,
            event_id="evt-2026-unknown",
            agent_name="TriageAgent",
            prompt_key="unknown_prompt",
        )
    assert exc.value.retryable is False
    assert [(entry.status, entry.fallback_level) for entry in audit.entries] == [
        ("llm_provider_error", 2)
    ]
