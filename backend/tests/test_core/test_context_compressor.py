"""PromptBudgeter / ContextCompressor tests (ISSUE-031)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.context_compressor import ContextCompressor, PromptBudgeter
from app.core.llm.base import LLMMessage, estimate_tokens
from app.models.enums import EvidenceSource
from app.models.evidence import Evidence


def _msg(role: str, content: str, *, name: str | None = None) -> LLMMessage:
    return LLMMessage(role=role, content=content, name=name)  # type: ignore[arg-type]


def _evidence(evidence_id: str, confidence: float, description: str, hour: int) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        event_id="evt-20260101-compress",
        source=EvidenceSource.ENDPOINT,
        evidence_type="process",
        description=description,
        confidence=confidence,
        timestamp=datetime(2026, 1, 1, hour, tzinfo=UTC),
    )


def test_estimate_tokens_deterministic_and_monotonic() -> None:
    sample = "hello 世界 shadowtrace"
    assert estimate_tokens(sample) == estimate_tokens(sample)
    assert estimate_tokens("") == 0
    assert estimate_tokens("中") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    shorter = "abc"
    longer = shorter + "def"
    assert estimate_tokens(longer) >= estimate_tokens(shorter)
    cjk = "研究" * 10
    mixed = cjk + "abcd"
    assert estimate_tokens(mixed) >= estimate_tokens(cjk)


def test_summarize_evidence_keeps_highest_confidence() -> None:
    compressor = ContextCompressor()
    items = [
        _evidence("evd-low", 0.2, "noise event", 1),
        _evidence("evd-high", 0.95, "critical beacon", 2),
        _evidence("evd-mid", 0.6, "suspicious login", 3),
        _evidence("evd-high-old", 0.95, "older critical", 0),
    ]
    summary = compressor.summarize_evidence(items, max_tokens=80)
    assert "evd-high" in summary
    # Equal confidence prefers newer timestamp.
    assert summary.index("evd-high") < summary.index("evd-high-old")
    assert "critical beacon" in summary
    # Tight budget should drop the lowest confidence item first.
    tight = compressor.summarize_evidence(items, max_tokens=45)
    assert "evd-high" in tight
    assert "evd-low" not in tight


def test_sliding_window_keeps_newest() -> None:
    compressor = ContextCompressor()
    history = ["h1", "h2", "h3", "h4", "h5"]
    assert compressor.sliding_window(history, 3) == ["h3", "h4", "h5"]
    assert compressor.sliding_window(history, 0) == []
    assert compressor.sliding_window(history, 10) == history


def test_prompt_budgeter_fit_respects_token_cap_and_keeps_system() -> None:
    budgeter = PromptBudgeter()
    system = _msg("system", "You are ShadowTrace triage assistant. Keep this system prompt.")
    history = [_msg("user", f"history turn {i} " + ("x" * 80)) for i in range(12)]
    goal = _msg("user", "Current goal: extract entities from the alert.")
    fitted = budgeter.fit([system, *history, goal], max_input_tokens=120)
    total = sum(estimate_tokens(message.content) for message in fitted)
    assert total <= 120
    assert fitted[0].role == "system"
    assert "ShadowTrace triage assistant" in fitted[0].content
    assert any(message.content.startswith("Current goal:") for message in fitted)
    assert budgeter.compressed is True
    assert "compressed=true" in fitted[0].content


def test_prompt_budgeter_compression_priority_history_before_goal() -> None:
    budgeter = PromptBudgeter()
    system = _msg("system", "SYSTEM_PROMPT_STABLE")
    old_history = _msg("assistant", "OLD_HISTORY_" + ("h" * 200))
    evidence = _msg(
        "user",
        Evidence(
            evidence_id="evd-1",
            event_id="evt-1",
            source=EvidenceSource.DNS,
            evidence_type="query",
            description="dns " + ("d" * 120),
            confidence=0.9,
        ).model_dump_json(),
        name="evidence",
    )
    raw = _msg("user", "RAW:" + ("r" * 200), name="raw_payload")
    goal = _msg("user", "GOAL_KEEP_ME")
    fitted = budgeter.fit(
        [system, old_history, evidence, raw, goal],
        max_input_tokens=90,
    )
    contents = [message.content for message in fitted]
    joined = "\n".join(contents)
    assert "SYSTEM_PROMPT_STABLE" in joined
    assert "GOAL_KEEP_ME" in joined
    # History should be dropped or shortened before the goal disappears.
    assert any(
        message.content == "GOAL_KEEP_ME" or message.content.startswith("GOAL_KEEP_ME")
        for message in fitted
    )
    assert sum(estimate_tokens(message.content) for message in fitted) <= 90


def test_prompt_budgeter_noop_when_under_budget() -> None:
    budgeter = PromptBudgeter()
    messages = [
        _msg("system", "short system"),
        _msg("user", "short goal"),
    ]
    fitted = budgeter.fit(messages, max_input_tokens=1000)
    assert budgeter.compressed is False
    assert [m.content for m in fitted] == ["short system", "short goal"]


def test_compress_context_trims_history_before_evidence_summary() -> None:
    compressor = ContextCompressor()
    context = {
        "event": {"event_id": "evt-priority"},
        "state_history": [{"i": i, "note": "n" * 30} for i in range(12)],
        "evidence_output": {
            "evidence_list": [
                _evidence("evd-top", 0.99, "top signal", 1).model_dump(mode="json"),
            ]
        },
    }
    compressed = compressor.compress_context(context, max_tokens=90)
    assert compressed["compressed"] is True
    assert len(compressed.get("state_history", [])) < 12
    assert "evd-top" in compressed.get("evidence_summary", "")


def test_compress_context_priority_and_flag() -> None:
    compressor = ContextCompressor()
    context = {
        "event": {"event_id": "evt-1", "title": "demo"},
        "evidence_output": {
            "evidence_list": [
                _evidence("evd-a", 0.4, "low " + ("a" * 40), 1).model_dump(mode="json"),
                _evidence("evd-b", 0.9, "high signal", 2).model_dump(mode="json"),
            ]
        },
        "state_history": [{"i": i, "note": "n" * 20} for i in range(20)],
        "source_snapshot": {"raw": "z" * 400},
        "risk_assessment": {"risk_score": 80},
    }
    compressed = compressor.compress_context(context, max_tokens=120)
    assert compressed["compressed"] is True
    assert compressed["event"]["event_id"] == "evt-1"
    assert "evidence_summary" in compressed
    assert "evd-b" in compressed["evidence_summary"]
    assert (
        estimate_tokens(
            __import__("json").dumps(compressed, ensure_ascii=False, default=str, sort_keys=True)
        )
        <= 120
    )


@pytest.mark.asyncio
async def test_llm_client_uses_prompt_budgeter_hook() -> None:
    from app.core.llm.base import InMemoryLLMCallAuditRecorder
    from app.core.llm.mock_client import MockLLMClient

    budgeter = PromptBudgeter()
    client = MockLLMClient(
        audit_recorder=InMemoryLLMCallAuditRecorder(),
        message_budgeter=budgeter,
        max_input_tokens=80,
        primary_model="mock-model",
    )
    # Missing golden will fail after fit; we only assert fit side-effect via budgeter.
    long_messages = [
        _msg("system", "sys"),
        *[_msg("user", "pad " + ("p" * 50)) for _ in range(10)],
        _msg("user", "goal"),
    ]
    fitted = client._fit_messages(long_messages)
    assert sum(estimate_tokens(m.content) for m in fitted) <= 80
    assert budgeter.compressed is True
