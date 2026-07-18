"""Prompt token budgeting and context compression (ISSUE-031)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from app.core.llm.base import LLMMessage, estimate_tokens
from app.models.evidence import Evidence

_COMPRESSED_MARKER = "compressed=true"

_PROTECTED_CONTEXT_KEYS = frozenset(
    {
        "event",
        "disposition_only_intent",
        "execution_substate",
    }
)
_EVIDENCE_CONTEXT_KEYS = frozenset({"evidence_output", "evidence_list", "evidence"})
_RAW_CONTEXT_KEYS = frozenset(
    {
        "source_snapshot",
        "scratchpad",
        "raw_alert_snapshot",
        "raw_data",
    }
)


def _message_tokens(message: LLMMessage) -> int:
    return estimate_tokens(message.content)


def _total_message_tokens(messages: Sequence[LLMMessage]) -> int:
    return sum(_message_tokens(message) for message in messages)


def _as_evidence(item: Any) -> Evidence | None:
    if isinstance(item, Evidence):
        return item
    if isinstance(item, Mapping):
        try:
            return Evidence.model_validate(item)
        except Exception:  # noqa: BLE001 — best-effort summary input
            return None
    return None


def _evidence_sort_key(item: Evidence) -> tuple[float, float]:
    ts = item.timestamp
    ts_score = ts.timestamp() if isinstance(ts, datetime) else float("-inf")
    return (float(item.confidence), ts_score)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        tokens = estimate_tokens(candidate)
        if tokens <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _is_evidence_message(message: LLMMessage) -> bool:
    name = (message.name or "").lower()
    if "evidence" in name:
        return True
    content = message.content.lstrip()
    if content.startswith("{") or content.startswith("["):
        lowered = content[:200].lower()
        return "evidence_id" in lowered or "evidence_list" in lowered or '"confidence"' in lowered
    return content.lower().startswith("evidence:")


def _is_raw_message(message: LLMMessage) -> bool:
    name = (message.name or "").lower()
    if any(token in name for token in ("raw", "snapshot", "payload", "scratchpad")):
        return True
    content = message.content.lstrip().lower()
    return content.startswith("raw:") or content.startswith("source_snapshot")


class ContextCompressor:
    """Rule-based evidence/history/context compression helpers."""

    def summarize_evidence(
        self,
        evidence_list: Sequence[Any],
        max_tokens: int,
    ) -> str:
        """Keep highest-confidence (then newest) evidence bullets within token budget."""

        if max_tokens <= 0:
            return ""
        parsed = [item for item in (_as_evidence(raw) for raw in evidence_list) if item is not None]
        parsed.sort(key=_evidence_sort_key, reverse=True)

        lines: list[str] = []
        used = 0
        for item in parsed:
            stamp = item.timestamp.isoformat() if item.timestamp is not None else "unknown-time"
            line = (
                f"- [{item.confidence:.2f}] {item.evidence_id} "
                f"({item.source.value}/{item.evidence_type} @ {stamp}): {item.description}"
            )
            cost = estimate_tokens(line if not lines else "\n" + line)
            if used + cost > max_tokens:
                remaining = max_tokens - used - estimate_tokens("\n- " if lines else "- ")
                if remaining <= 8:
                    break
                truncated = _truncate_text_to_tokens(
                    f"[{item.confidence:.2f}] {item.evidence_id}: {item.description}",
                    remaining,
                )
                if truncated:
                    lines.append(f"- {truncated}")
                break
            lines.append(line)
            used += cost
        return "\n".join(lines)

    def sliding_window(self, history: Sequence[Any], max_items: int) -> list[Any]:
        """Keep the newest ``max_items`` history entries."""

        if max_items <= 0:
            return []
        items = list(history)
        if len(items) <= max_items:
            return items
        return items[-max_items:]

    def compress_context(
        self,
        event_context: Mapping[str, Any] | Any,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Compress an EventContext-like mapping by history → evidence → raw priority."""

        if hasattr(event_context, "model_dump"):
            payload = event_context.model_dump(mode="python")
        else:
            payload = dict(event_context)

        if max_tokens <= 0:
            return {"compressed": True}

        result: dict[str, Any] = {"compressed": False}
        for key in _PROTECTED_CONTEXT_KEYS:
            if key in payload:
                result[key] = payload[key]

        def _budget_used() -> int:
            return estimate_tokens(
                json.dumps(result, ensure_ascii=False, default=str, sort_keys=True)
            )

        evidence_blob: list[Any] = []
        evidence_output = payload.get("evidence_output")
        if isinstance(evidence_output, Mapping):
            evidence_blob = list(evidence_output.get("evidence_list") or [])
        elif isinstance(payload.get("evidence_list"), list):
            evidence_blob = list(payload["evidence_list"])

        history = payload.get("state_history")
        if isinstance(history, list) and history:
            windowed = self.sliding_window(history, min(5, len(history)))
            while windowed:
                candidate = dict(result)
                candidate["state_history"] = windowed
                encoded = json.dumps(candidate, ensure_ascii=False, default=str, sort_keys=True)
                if estimate_tokens(encoded) <= max_tokens:
                    result["state_history"] = windowed
                    if len(windowed) < len(history):
                        result["compressed"] = True
                    break
                result["compressed"] = True
                windowed = self.sliding_window(windowed, len(windowed) - 1)

        remaining = max_tokens - _budget_used()
        if evidence_blob and remaining > 0:
            summary = self.summarize_evidence(evidence_blob, max(16, remaining // 2))
            if summary:
                result["evidence_summary"] = summary
                result["compressed"] = True

        for key, value in payload.items():
            if key in result or key in _PROTECTED_CONTEXT_KEYS:
                continue
            if key in _EVIDENCE_CONTEXT_KEYS or key in _RAW_CONTEXT_KEYS:
                continue
            if key in {"state_history", "scratchpad"}:
                continue
            candidate = dict(result)
            candidate[key] = value
            encoded = json.dumps(candidate, ensure_ascii=False, default=str, sort_keys=True)
            if estimate_tokens(encoded) <= max_tokens:
                result[key] = value
            else:
                result["compressed"] = True

        for key in _RAW_CONTEXT_KEYS:
            if key not in payload:
                continue
            raw_value = payload[key]
            text = (
                json.dumps(raw_value, ensure_ascii=False, default=str, sort_keys=True)
                if not isinstance(raw_value, str)
                else raw_value
            )
            remaining = max_tokens - _budget_used()
            if remaining <= 0:
                result["compressed"] = True
                break
            truncated = _truncate_text_to_tokens(text, remaining)
            if truncated != text:
                result["compressed"] = True
            if truncated:
                result[key] = truncated

        encoded = json.dumps(result, ensure_ascii=False, default=str, sort_keys=True)
        if estimate_tokens(encoded) > max_tokens:
            result = {
                "compressed": True,
                "event": result.get("event"),
                "evidence_summary": _truncate_text_to_tokens(
                    str(result.get("evidence_summary") or ""),
                    max(0, max_tokens // 2),
                ),
            }
            encoded = json.dumps(result, ensure_ascii=False, default=str, sort_keys=True)
            if estimate_tokens(encoded) > max_tokens:
                result = {
                    "compressed": True,
                    "event": _truncate_text_to_tokens(
                        json.dumps(result.get("event"), ensure_ascii=False, default=str),
                        max_tokens,
                    ),
                }
        return result


class PromptBudgeter:
    """Fit chat messages into a token budget with fixed compression priority.

    Priority (ISSUE-031): sliding-window history → summarize evidence → truncate
    raw payloads. System prompt and the current goal (latest user message) are
    preserved until a final hard truncate is unavoidable.
    """

    def __init__(self, compressor: ContextCompressor | None = None) -> None:
        self.compressor = compressor or ContextCompressor()
        self.compressed = False

    def fit(self, messages: list[LLMMessage], max_input_tokens: int) -> list[LLMMessage]:
        self.compressed = False
        if max_input_tokens <= 0:
            self.compressed = True
            return []

        fitted = [message.model_copy(deep=True) for message in messages]
        if _total_message_tokens(fitted) <= max_input_tokens:
            return fitted

        system_indexes = [idx for idx, msg in enumerate(fitted) if msg.role == "system"]
        goal_index = next(
            (idx for idx in range(len(fitted) - 1, -1, -1) if fitted[idx].role == "user"),
            len(fitted) - 1 if fitted else -1,
        )
        protected = set(system_indexes)
        if goal_index >= 0:
            protected.add(goal_index)

        # 1) Sliding-window trim of unprotected history (drop oldest first).
        history_indexes = [idx for idx in range(len(fitted)) if idx not in protected]
        if history_indexes:
            keep = len(history_indexes)
            while keep >= 0 and _total_message_tokens(fitted) > max_input_tokens:
                keep -= 1
                self.compressed = True
                retained = set(self.compressor.sliding_window(history_indexes, keep))
                fitted = [
                    msg for idx, msg in enumerate(fitted) if idx in protected or idx in retained
                ]
                system_indexes = [idx for idx, msg in enumerate(fitted) if msg.role == "system"]
                goal_index = next(
                    (idx for idx in range(len(fitted) - 1, -1, -1) if fitted[idx].role == "user"),
                    len(fitted) - 1 if fitted else -1,
                )
                protected = set(system_indexes)
                if goal_index >= 0:
                    protected.add(goal_index)
                history_indexes = [idx for idx in range(len(fitted)) if idx not in protected]

        if _total_message_tokens(fitted) <= max_input_tokens:
            return self._mark_if_needed(fitted)

        # 2) Summarize evidence-bearing messages.
        for idx, message in enumerate(list(fitted)):
            protected = {i for i, msg in enumerate(fitted) if msg.role == "system"}
            goal_index = next(
                (i for i in range(len(fitted) - 1, -1, -1) if fitted[i].role == "user"),
                -1,
            )
            if goal_index >= 0:
                protected.add(goal_index)
            if idx in protected or not _is_evidence_message(message):
                continue
            evidence_items = self._extract_evidence_items(message.content)
            if not evidence_items:
                continue
            over = _total_message_tokens(fitted) - max_input_tokens
            target = max(16, _message_tokens(message) - max(over, 0))
            summary = self.compressor.summarize_evidence(evidence_items, target)
            if summary and summary != message.content:
                fitted[idx] = message.model_copy(update={"content": "Evidence:\n" + summary})
                self.compressed = True
            if _total_message_tokens(fitted) <= max_input_tokens:
                return self._mark_if_needed(fitted)

        # 3) Truncate raw / remaining unprotected payloads from oldest to newest.
        protected = {i for i, msg in enumerate(fitted) if msg.role == "system"}
        goal_index = next(
            (i for i in range(len(fitted) - 1, -1, -1) if fitted[i].role == "user"),
            -1,
        )
        if goal_index >= 0:
            protected.add(goal_index)
        for idx, message in enumerate(fitted):
            if idx in protected:
                continue
            over = _total_message_tokens(fitted) - max_input_tokens
            if over <= 0:
                break
            keep_tokens = max(0, _message_tokens(message) - over)
            truncated = _truncate_text_to_tokens(message.content, keep_tokens)
            if truncated != message.content:
                fitted[idx] = message.model_copy(update={"content": truncated})
                self.compressed = True
            if _total_message_tokens(fitted) <= max_input_tokens:
                return self._mark_if_needed(fitted)

        # 4) Hard truncate goal then system if still over budget.
        system_indexes = [idx for idx, msg in enumerate(fitted) if msg.role == "system"]
        goal_index = next(
            (idx for idx in range(len(fitted) - 1, -1, -1) if fitted[idx].role == "user"),
            -1,
        )
        hard_order = []
        if goal_index >= 0:
            hard_order.append(goal_index)
        hard_order.extend(system_indexes)
        for idx in list(dict.fromkeys(hard_order)):
            over = _total_message_tokens(fitted) - max_input_tokens
            if over <= 0:
                break
            keep_tokens = max(0, _message_tokens(fitted[idx]) - over)
            truncated = _truncate_text_to_tokens(fitted[idx].content, keep_tokens)
            if truncated != fitted[idx].content:
                fitted[idx] = fitted[idx].model_copy(update={"content": truncated})
                self.compressed = True

        if _total_message_tokens(fitted) > max_input_tokens:
            self.compressed = True
            system_indexes = [idx for idx, msg in enumerate(fitted) if msg.role == "system"]
            if system_indexes:
                system = fitted[system_indexes[0]]
                fitted = [
                    system.model_copy(
                        update={
                            "content": _truncate_text_to_tokens(system.content, max_input_tokens)
                        }
                    )
                ]
            else:
                fitted = fitted[:1]
                if fitted:
                    fitted[0] = fitted[0].model_copy(
                        update={
                            "content": _truncate_text_to_tokens(fitted[0].content, max_input_tokens)
                        }
                    )
        return self._mark_if_needed(fitted)

    def _mark_if_needed(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        if not self.compressed or not messages:
            return messages
        for idx, message in enumerate(messages):
            if message.role != "system":
                continue
            if _COMPRESSED_MARKER in message.content:
                return messages
            marker = f"\n[{_COMPRESSED_MARKER}]"
            messages[idx] = message.model_copy(update={"content": message.content + marker})
            return messages
        first = messages[0]
        messages[0] = first.model_copy(update={"name": (first.name or "") + "|compressed=true"})
        return messages

    @staticmethod
    def _extract_evidence_items(content: str) -> list[Any]:
        text = content.strip()
        if text.lower().startswith("evidence:"):
            text = text.split(":", 1)[1].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            if isinstance(payload.get("evidence_list"), list):
                return list(payload["evidence_list"])
            if "evidence_id" in payload:
                return [payload]
        return []


__all__ = [
    "ContextCompressor",
    "PromptBudgeter",
]
