"""Shared secret redaction for external payloads, receipts, events, and logs."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEY_RE = re.compile(
    r"(?:password|passwd|pwd|secret|token|authorization|api[_-]?key|cookie|"
    r"credential|private[_-]?key|session[_-]?id|raw[_-]?(?:result|payload))",
    re.IGNORECASE,
)
_AUTH_SCHEME_RE = re.compile(
    r"(?P<prefix>\b(?:bearer|basic)\s+)(?P<secret>[A-Za-z0-9._~+/=-]{4,})",
    re.IGNORECASE,
)
_SENSITIVE_HEADER_RE = re.compile(
    r"(?P<prefix>\b(?:authorization|cookie)\b[\"']?\s*[:=]\s*)"
    r"(?P<value>[^,\r\n}]+)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>\b(?:password|passwd|pwd|secret|token|access[_-]?token|"
    r"refresh[_-]?token|api[_-]?key|credential|session[_-]?id)"
    r"\b[\"']?\s*[:=]\s*)"
    r"(?P<secret>\"[^\"]*\"|'[^']*'|[^\s,;&}]+)",
    re.IGNORECASE,
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"(?![A-Za-z0-9_-])"
)
_KNOWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}|xox[baprs]-[A-Za-z0-9-]{16,}|"
    r"AKIA[A-Z0-9]{16})(?![A-Za-z0-9])"
)
_URL_CREDENTIAL_RE = re.compile(
    r"(?P<scheme>\b[a-z][a-z0-9+.-]*://)[^/\s:@]+:[^@/\s]+@",
    re.IGNORECASE,
)


def is_sensitive_key(key: object) -> bool:
    """Return whether a mapping key is secret-bearing by policy."""

    return bool(_SENSITIVE_KEY_RE.search(str(key)))


def redact_sensitive_text(value: str, *, replacement: str = REDACTED) -> str:
    """Redact conservative credential value patterns from free-form text."""

    cleaned = _URL_CREDENTIAL_RE.sub(
        lambda match: f"{match.group('scheme')}{replacement}@",
        value,
    )
    cleaned = _SENSITIVE_HEADER_RE.sub(
        lambda match: f"{match.group('prefix')}{replacement}",
        cleaned,
    )
    cleaned = _AUTH_SCHEME_RE.sub(
        lambda match: f"{match.group('prefix')}{replacement}",
        cleaned,
    )
    cleaned = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('prefix')}{replacement}",
        cleaned,
    )
    cleaned = _JWT_RE.sub(replacement, cleaned)
    return _KNOWN_TOKEN_RE.sub(replacement, cleaned)


def sanitize_data(
    value: Any,
    *,
    replacement: str = REDACTED,
    max_depth: int = 32,
) -> Any:
    """Recursively redact secret keys and credential-shaped string values."""

    def _sanitize(item: Any, depth: int) -> Any:
        if depth > max_depth:
            return replacement
        if isinstance(item, Mapping):
            cleaned: dict[str, Any] = {}
            for key, nested in item.items():
                key_str = str(key)
                cleaned[key_str] = (
                    replacement if is_sensitive_key(key_str) else _sanitize(nested, depth + 1)
                )
            return cleaned
        if isinstance(item, list | tuple | set | frozenset):
            return [_sanitize(nested, depth + 1) for nested in item]
        if isinstance(item, str):
            return redact_sensitive_text(item, replacement=replacement)
        if isinstance(item, bytes):
            return redact_sensitive_text(
                item.decode("utf-8", errors="replace"),
                replacement=replacement,
            )
        return item

    return _sanitize(value, 0)


class RedactingFormatter(logging.Formatter):
    """Logging formatter that removes credential patterns from final output."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(super().format(record))


__all__ = [
    "REDACTED",
    "RedactingFormatter",
    "is_sensitive_key",
    "redact_sensitive_text",
    "sanitize_data",
]
