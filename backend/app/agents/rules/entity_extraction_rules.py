"""Deterministic regex-based entity extraction fallback (ISSUE-032).

Used by ``TriageAgent._extract_entities`` when the LLM path is unavailable,
times out, or returns unparseable output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# --------------------------------------------------------------------------- #
# Regex patterns (compiled once at import time)
# --------------------------------------------------------------------------- #

# IPv4 address (dotted decimal).
_IP_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
IP_PATTERN: re.Pattern[str] = _IP_PATTERN  # Public alias for reuse by triage_agent

# FQDN / domain name (requires at least one dot-separated label + valid TLD).
_DOMAIN_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}\b"
)

# Hostname: must start with a letter, contain letters/digits/hyphens, end with
# a digit OR contain a known Windows/Linux hostname indicator.  Tighter than the
# original `([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)` which matched any
# hyphenated English words.
_HOSTNAME_PATTERN: re.Pattern[str] = re.compile(
    r"\b"
    r"(?:"
    r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"  # must include at least one hyphen
    r"(?:\d|[A-Za-z]{2,})"  # ends with digit or ≥2 letters
    r"|"
    r"[A-Za-z]{2,}\d{1,4}"  # PC-FIN-023 style
    r"|"
    r"[A-Za-z][A-Za-z0-9_]*-(?:SRV|DC|DB|WEB|OPS|FIN|SQL|AD|FS|APP|JUMP|ADMIN|MAIL|PROXY|VPN|NODE|PRD|STG|DEV)"
    r"[A-Za-z0-9_-]*"  # known role suffixes
    r"[A-Za-z0-9_-]*"
    r")"
    r"\b"
)

# Account / username — quoted or bare.  Supports English prefixes (account, user,
# username) and Chinese prefixes (账号, 用户, 用户名) for mixed-language alerts.
_ACCOUNT_PATTERN: re.Pattern[str] = re.compile(
    r'(?:account|user|username|账号|用户|用户名)\s+["\']?([A-Za-z][A-Za-z0-9@._-]{1,63})["\']?',
    re.IGNORECASE,
)

# Process name with common Windows/Linux executable extensions.
_PROCESS_PATTERN: re.Pattern[str] = re.compile(
    r"\b([A-Za-z][A-Za-z0-9._-]{0,63}\.(?:exe|dll|sys|bat|cmd|ps1|vbs|py|sh|bin|run|out))\b"
)

# File name with common document/archive extensions — NOT executables.
_FILE_PATTERN: re.Pattern[str] = re.compile(
    r"\b([A-Za-z][A-Za-z0-9._-]{0,63}\.(?:zip|7z|rar|tar|gz|csv|doc|docx|xls|xlsx|pdf|txt|log|sql|db|bak|pst|ost|eml|msg|json|xml|yaml|yml|ini|cfg|conf|key|pem|crt|cer|p12|pfx|jpg|png|bmp|wav|mp3|mp4|avi|mov))(?:\b|(?=[\s,;\"'<>]|$))",
)


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EntityExtractionResult:
    """Regex fallback extraction output."""

    ips: list[str]
    domains: list[str]
    hostnames: list[str]
    accounts: list[str]
    processes: list[str]
    files: list[str]

    def is_empty(self) -> bool:
        return not any(
            (self.ips, self.domains, self.hostnames, self.accounts, self.processes, self.files)
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def extract_entities_regex(alert_text: str) -> EntityExtractionResult:
    """Extract entity strings from raw alert text using deterministic regex.

    Returns:
        ``EntityExtractionResult`` with de-duplicated, order-preserving lists.
        Caller is responsible for converting strings into typed entity models
        (``IPEntity``, ``HostEntity``, …) and assigning ``entity_id`` values.
    """
    ips = _unique(_IP_PATTERN.findall(alert_text))
    domains = _unique(_DOMAIN_PATTERN.findall(alert_text))
    # Exclude values that already matched as domains.
    hostnames = _unique(h for h in _HOSTNAME_PATTERN.findall(alert_text) if h not in domains)
    accounts = _unique(m.group(1) for m in _ACCOUNT_PATTERN.finditer(alert_text))
    processes = _unique(_PROCESS_PATTERN.findall(alert_text))
    files = _unique(_FILE_PATTERN.findall(alert_text))

    return EntityExtractionResult(
        ips=ips,
        domains=domains,
        hostnames=hostnames,
        accounts=accounts,
        processes=processes,
        files=files,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _unique(items: list[str]) -> list[str]:
    """Return order-preserving de-duplicated list."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


__all__ = [
    "EntityExtractionResult",
    "extract_entities_regex",
    "IP_PATTERN",
]
