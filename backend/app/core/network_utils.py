"""Shared network utilities: private-IP constants and helpers (ISSUE-032).

Extracted from triage_agent.py and entity_extraction_rules.py to remove the
duplicate definitions noted in the PR review.
"""

from __future__ import annotations

# RFC 1918 / loopback / link-local prefixes (sorted for readability).
_INTERNAL_NETS: tuple[str, ...] = (
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "192.168.",
    "127.",
    "0.",
)


def is_internal_ip(addr: str) -> bool:
    """Return True when *addr* starts with a known private / reserved prefix."""
    return any(addr.startswith(prefix) for prefix in _INTERNAL_NETS)
