"""TriageAgent — alert parsing, entity extraction, event typing, IOC list (ISSUE-032).

LLM primary path + regex fallback. Severity is assigned via deterministic
``SEVERITY_RULES``; ``need_investigation`` is ``True`` when severity >= medium.
Two hook lists (``pre_triage_hooks``, ``post_triage_hooks``) alias the base
``pre_hooks`` / ``post_hooks`` lists. The P0 default ``RuleBasedFalsePositiveHook``
writes ``EventContext.false_positive_match`` for stable fixture signatures via
its own ``BoundWorkingMemory`` bound to the ``RuleBasedFalsePositiveHook``
identity (aliased to ``FalsePositiveMatcher`` by ``WRITER_ALIASES``).
"""

from __future__ import annotations

import ipaddress
import logging
import re as _re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts.triage_prompt import TriageLLMResponse, build_triage_messages
from app.agents.rules.entity_extraction_rules import (
    IP_PATTERN,
    extract_entities_regex,
)
from app.core.errors import (
    DependencyUnavailableError,
    GuardrailViolationError,
    LLMError,
    ShadowTraceError,
)
from app.core.llm.base import LLMResponse
from app.core.network_utils import is_internal_ip
from app.models.agent_io import TriageAgentInput, TriageResult
from app.models.entities import (
    AccountEntity,
    DomainEntity,
    EntitySet,
    FileEntity,
    HostEntity,
    IPEntity,
    ProcessEntity,
)
from app.models.enums import EventType, Severity
from app.services.working_memory import BoundWorkingMemory

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# SEVERITY_RULES — deterministic severity based on event_type
# --------------------------------------------------------------------------- #

SEVERITY_RULES: dict[str, list[tuple[str, str]]] = {
    # data_exfiltration with an external IP present → HIGH (ISSUE-032 spec:
    # "数据外泄类加外部 IP 为 high").  Without an external IP (e.g. pure
    # internal server-to-server exfiltration) severity is MEDIUM — the check
    # is applied in _apply_severity_rules via _external_ip_in_text().
    "high": [
        ("event_type", "data_exfiltration"),
        ("event_type", "malicious_process"),
        ("event_type", "host_compromise"),
        ("event_type", "lateral_movement"),
    ],
    # data_exfiltration + lateral_movement co-occurrence → critical (checked
    # in _apply_severity_rules via alert_text word-boundary match on
    # "lateral" so words like "bilateral"/"collateral" are excluded).
    "critical": [
        ("event_type", "data_exfiltration"),
    ],
    "low": [
        ("event_type", "account_anomaly"),
    ],
}

# --------------------------------------------------------------------------- #
# IOC extraction helpers
# --------------------------------------------------------------------------- #

# IP_PATTERN is imported from entity_extraction_rules to avoid duplication.
_IOC_DOMAIN_PATTERN: _re.Pattern[str] = _re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}\b"
)
_IOC_HASH_PATTERN: _re.Pattern[str] = _re.compile(
    r"\b([a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b"
)
_IOC_URL_PATTERN: _re.Pattern[str] = _re.compile(r"https?://[^\s,;\"'<>]+")

# --------------------------------------------------------------------------- #
# FP signatures (P0 default, no vector DB dependency)
# --------------------------------------------------------------------------- #

_FP_SIGNATURES: dict[str, str] = {
    "account_anomaly_fp": "ops_change_window_bulk_login",
    "ops_change_window_bulk_login": "ops_change_window_bulk_login",
}


# --------------------------------------------------------------------------- #
# RuleBasedFalsePositiveHook
# --------------------------------------------------------------------------- #


@dataclass
class RuleBasedFalsePositiveHook:
    """Deterministic false-positive signature matcher (P0 default).

    Scans the source_snapshot for stable scenario/fixture signatures and
    writes ``EventContext.false_positive_match`` when a known FP pattern is
    detected. Does NOT depend on pgvector or any knowledge base.

    Uses its own ``BoundWorkingMemory`` bound to the ``RuleBasedFalsePositiveHook``
    writer identity (aliased to ``FalsePositiveMatcher`` via ``WRITER_ALIASES``),
    NOT the TriageAgent's memory — fixing the FIELD_OWNERSHIP violation noted
    in the PR review.
    """

    _wm: BoundWorkingMemory | None = None

    def __init__(self, working_memory: BoundWorkingMemory | None = None) -> None:
        """Args:
        working_memory: BoundWorkingMemory created via
            ``WorkingMemory.for_writer("RuleBasedFalsePositiveHook")``.
            If None the hook is a no-op.
        """
        self._wm = working_memory

    async def __call__(self, agent: BaseAgent, input: TriageAgentInput) -> None:  # type: ignore[override]
        wm = self._wm
        if wm is None:
            return

        # Read source_snapshot through the TriageAgent's own memory (read is
        # not ownership-gated — any bound identity can read any field).
        agent_wm = getattr(agent, "working_memory", None)
        if agent_wm is None:
            return

        snapshot = await agent_wm.read(input.event_id, "source_snapshot")
        if not isinstance(snapshot, dict):
            return

        scenario = snapshot.get("scenario", "")
        signature = snapshot.get("signature", "")
        fp_match: dict[str, Any] | None = None

        # ---------------------------------------------------------------- #
        # Check known FP signatures against scenario / signature fields
        # ---------------------------------------------------------------- #
        if scenario in _FP_SIGNATURES:
            fp_match = {
                "matched_rule": _FP_SIGNATURES[scenario],
                "scenario": scenario,
                "source": "RuleBasedFalsePositiveHook",
                "matched_at": datetime.now(UTC).isoformat(),
                "recommendation": "close_as_fp",
            }
        elif signature in _FP_SIGNATURES:
            fp_match = {
                "matched_rule": _FP_SIGNATURES[signature],
                "signature": signature,
                "source": "RuleBasedFalsePositiveHook",
                "matched_at": datetime.now(UTC).isoformat(),
                "recommendation": "close_as_fp",
            }

        if fp_match is None:
            return

        # Write through the hook's OWN BoundWorkingMemory (writer identity =
        # FalsePositiveMatcher, matching FIELD_OWNERSHIP).
        await wm.write(input.event_id, "false_positive_match", fp_match)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _apply_severity_rules(
    event_type: EventType,
    alert_text: str = "",
) -> tuple[Severity, bool]:
    """Assign severity and need_investigation via SEVERITY_RULES.

    Returns:
        (severity, need_investigation) — need_investigation is True when
        severity is medium or higher.
    """
    severity = Severity.LOW
    event_type_value = event_type.value if isinstance(event_type, EventType) else event_type

    # Check critical rules first — highest priority.
    for rule_key, rule_val in SEVERITY_RULES.get("critical", []):
        if rule_key == "event_type" and rule_val == event_type_value:
            # Critical: data_exfiltration with lateral movement co-occurrence.
            # Use word-boundary match to avoid false positives on "bilateral",
            # "collateral", etc.
            if alert_text and _re.search(r"\blateral\b", alert_text.lower()):
                severity = Severity.CRITICAL
                return severity, True

    # Check high rules.
    for rule_key, rule_val in SEVERITY_RULES.get("high", []):
        if rule_key == "event_type" and rule_val == event_type_value:
            # ISSUE-032 spec: data_exfiltration → HIGH only when an external
            # IP is present in the alert text.  Pure internal exfiltration
            # (no external IP) → MEDIUM.
            if event_type_value == "data_exfiltration":
                if alert_text and _external_ip_in_text(alert_text):
                    severity = Severity.HIGH
                    return severity, True
                severity = Severity.MEDIUM
                return severity, True
            severity = Severity.HIGH
            return severity, True

    # Check low rules.
    for rule_key, rule_val in SEVERITY_RULES.get("low", []):
        if rule_key == "event_type" and rule_val == event_type_value:
            # P0 simplification: all account_anomaly → LOW per ISSUE-032.
            # Upgrade to MEDIUM when alert_text suggests bulk/mass activity,
            # privilege escalation, or geo-anomaly (to be refined in ISSUE-078).
            if event_type_value == "account_anomaly" and alert_text:
                _at = alert_text.lower()
                if any(
                    kw in _at
                    for kw in (
                        "bulk",
                        "mass",
                        "privilege escalation",
                        "地域异常",
                        "geo-anomaly",
                        "impossible travel",
                        "brute force",
                        "password spray",
                    )
                ):
                    severity = Severity.MEDIUM
                    return severity, True
            severity = Severity.LOW
            return severity, False

    # Default for unlisted event types: medium.
    severity = Severity.MEDIUM
    return severity, True


def _external_ip_in_text(alert_text: str) -> bool:
    """Return True when *alert_text* contains at least one external (non-internal) IP.

    Used by ``_apply_severity_rules`` to decide whether a data_exfiltration
    event qualifies for HIGH severity per ISSUE-032.
    """
    for ip_match in IP_PATTERN.findall(alert_text):
        if not is_internal_ip(ip_match):
            return True
    return False


def _extract_iocs(
    alert_text: str,
    entities: EntitySet | None = None,
) -> list[str]:
    """Extract IoC strings from raw alert text and entity IPs.

    Only external (non-internal) IPs are included.
    """
    iocs: set[str] = set()

    # Extract from raw text.
    for ip in IP_PATTERN.findall(alert_text):
        if not is_internal_ip(ip):
            iocs.add(ip)
    for domain in _IOC_DOMAIN_PATTERN.findall(alert_text):
        iocs.add(domain)
    for hash_val in _IOC_HASH_PATTERN.findall(alert_text):
        iocs.add(hash_val)
    for url in _IOC_URL_PATTERN.findall(alert_text):
        iocs.add(url)

    # Include external IPs from entities.
    if entities is not None:
        for ip_entity in entities.ips:
            addr = ip_entity.address or ""
            if addr and not is_internal_ip(addr):
                iocs.add(addr)

    # Sort: IPs via ipaddress for natural ordering, everything else
    # lexicographically.  This avoids "8.8.8.8" < "10.0.0.1" in string sort.
    def _ioc_sort_key(value: str) -> tuple[int, object]:
        try:
            addr = ipaddress.ip_address(value)
            return (0, addr)  # IP addresses sorted numerically
        except ValueError:
            return (1, value)  # domains, hashes, URLs sorted lexicographically

    return sorted(iocs, key=_ioc_sort_key)


def _resolve_alert_type_from_snapshot(snapshot: dict[str, Any] | None) -> str | None:
    """Resolve ``alert_type`` from frozen ``source_snapshot``.

    Primary path: top-level ``alert_type`` on the normalized snapshot.
    File fallback only: read the compatible ``raw_alert_snapshot`` field when
    the normalized snapshot has no ``alert_type`` (ISSUE-032 unified naming).
    """
    if not isinstance(snapshot, dict):
        return None

    top_level = snapshot.get("alert_type")
    if top_level:
        return str(top_level)

    raw_snap = snapshot.get("raw_alert_snapshot")
    if not isinstance(raw_snap, dict):
        return None

    nested_type = raw_snap.get("alert_type")
    if nested_type:
        return str(nested_type)

    raw_payload = raw_snap.get("raw")
    if isinstance(raw_payload, dict):
        payload_type = raw_payload.get("alert_type")
        if payload_type:
            return str(payload_type)

    return None


def _map_event_type(
    raw_type: str | None,
    alert_text: str = "",
) -> EventType:
    """Map raw event_type string to EventType enum with fallback heuristics.

    When raw_type is None or unrecognized, keyword matching on alert_text is
    used as a best-effort fallback.
    """
    if raw_type:
        try:
            return EventType(raw_type.lower())
        except ValueError:
            pass

    # Fallback keyword matching.
    text = alert_text.lower()
    if "exfil" in text or "upload" in text:
        return EventType.DATA_EXFILTRATION
    if "login fail" in text or "failed to login" in text or "login attempt" in text:
        return EventType.ACCOUNT_ANOMALY
    if "process" in text or "executed" in text or "malware" in text:
        return EventType.MALICIOUS_PROCESS
    if "domain" in text or "dns" in text:
        return EventType.SUSPICIOUS_DOMAIN
    if "lateral" in text or "pivot" in text:
        return EventType.LATERAL_MOVEMENT
    if "host" in text or "compromise" in text or "infected" in text:
        return EventType.HOST_COMPROMISE
    if "insider" in text or "privilege" in text or _re.search(r"(?<!de-)escalation\b", text):
        return EventType.INSIDER_THREAT
    return EventType.OTHER


def _merge_hint_entities(
    llm_entities: EntitySet,
    hint_entities: EntitySet,
) -> EntitySet:
    """Merge hint entities into LLM entities, returning a NEW ``EntitySet``.

    Entities from ``hint_entities`` that do not already exist (by ``entity_id``)
    in ``llm_entities`` are appended.  The input objects are never mutated
    (fixing the immutability contract issue noted in the PR review).
    """
    merged = EntitySet()

    # Traverse all entity categories defined on EntitySet so new categories
    # are picked up automatically (instead of a hardcoded six-item tuple).
    for category in EntitySet.model_fields.keys():
        llm_list: list = getattr(llm_entities, category)
        hint_list: list = getattr(hint_entities, category)
        existing_ids = {e.entity_id for e in llm_list}
        combined = list(llm_list) + [e for e in hint_list if e.entity_id not in existing_ids]
        setattr(merged, category, combined)

    return merged


# --------------------------------------------------------------------------- #
# TriageAgent
# --------------------------------------------------------------------------- #


class TriageAgent(BaseAgent[TriageAgentInput, TriageResult]):
    """Stage 1 Agent: parse alert → entities, event_type, severity, IoCs.

    Primary path: LLM (JSON mode) → EntitySet.
    Fallback path: regex when LLM is unavailable or fails.
    """

    agent_name: str = "triage_agent"

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        tool_executor: Any | None = None,
        working_memory: BoundWorkingMemory | None = None,
        budget_service: Any | None = None,
        output_guard: Any | None = None,
        trace_service: Any | None = None,
        audit_service: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            tool_executor=tool_executor,
            working_memory=working_memory,
            budget_service=budget_service,
            output_guard=output_guard,
            trace_service=trace_service,
            audit_service=audit_service,
            event_bus=event_bus,
        )

        # Convenience aliases matching the Issue-032 naming convention.
        self.pre_triage_hooks = self.pre_hooks
        self.post_triage_hooks = self.post_hooks

        # Install the P0 RuleBasedFalsePositiveHook with its own writer identity.
        # The hook must write to EventContext.false_positive_match via the
        # "FalsePositiveMatcher" owner — using the TriageAgent's own BoundWorkingMemory
        # would fail FIELD_OWNERSHIP validation. We mint a second capability token
        # via BoundWorkingMemory.for_writer() (the public API).
        if working_memory is not None:
            fp_hook_memory = working_memory.for_writer("RuleBasedFalsePositiveHook")
            self.pre_triage_hooks.append(RuleBasedFalsePositiveHook(working_memory=fp_hook_memory))

    # ------------------------------------------------------------------ #
    # _run
    # ------------------------------------------------------------------ #

    async def _run(self, input: TriageAgentInput) -> TriageResult:
        """Execute the full triage pipeline."""
        degraded = False
        reasoning_parts: list[str] = []

        # 1. Map event type from source_snapshot (file fallback via raw_alert_snapshot).
        snapshot = await self._read_source_snapshot(input.event_id)
        raw_type = _resolve_alert_type_from_snapshot(snapshot)
        event_type = _map_event_type(raw_type, input.raw_event_summary)

        # 2. Entity extraction — LLM primary, regex fallback.
        entities, llm_degraded, llm_reasoning = await self._extract_entities(
            input.raw_event_summary, input.event_id
        )
        if llm_degraded:
            degraded = True
            reasoning_parts.append("Entity extraction degraded to regex fallback.")
        if llm_reasoning:
            reasoning_parts.append(llm_reasoning)

        # 3. Merge hint entities from input.
        if input.hint_entities:
            entities = _merge_hint_entities(entities, input.hint_entities)

        # 4. Severity + need_investigation.
        severity, need_investigation = _apply_severity_rules(
            event_type, alert_text=input.raw_event_summary
        )

        # 5. IOC extraction.
        ioc_list = _extract_iocs(input.raw_event_summary, entities)

        # 6. Build result.
        result = TriageResult(
            event_type=event_type,
            severity=severity,
            need_investigation=need_investigation,
            entities=entities,
            ioc_list=ioc_list,
            reasoning=" ".join(reasoning_parts) if reasoning_parts else "",
            degraded=degraded,
        )

        # 7. Persist to EventContext.
        await self._write_triage_result(input, result)

        return result

    # ------------------------------------------------------------------ #
    # Entity extraction (LLM primary → regex fallback)
    # ------------------------------------------------------------------ #

    async def _extract_entities(
        self, alert_text: str, event_id: str
    ) -> tuple[EntitySet, bool, str]:
        """Extract entities via LLM (JSON mode) with regex fallback.

        Returns:
            (entities, degraded, reasoning) — ``degraded`` is True when the
            LLM path was unavailable and regex was used instead.
        """
        if self.llm_client is None:
            # No LLM client configured → go straight to regex fallback.
            entities = await self._regex_fallback(alert_text)
            return entities, True, ""

        # Empty alert text → nothing to extract; skip LLM call.
        # ``build_triage_messages`` raises ``ValueError`` on empty input which
        # is not a ``ShadowTraceError`` and would escape the try/except below,
        # causing the agent to fail instead of degrading gracefully.
        if not alert_text.strip():
            return EntitySet(), False, ""

        try:
            messages = build_triage_messages(alert_text)
            response: LLMResponse = await self.llm_client.chat(
                messages,
                event_id=event_id,
                agent_name=self.agent_name,
                prompt_key="triage_extract",
                json_mode=True,
                response_model=TriageLLMResponse,
                temperature=0.3,
                max_tokens=4096,
                timeout=15.0,
            )

            if response.parsed is not None and isinstance(response.parsed, TriageLLMResponse):
                parsed: TriageLLMResponse = response.parsed
                # fallback_level > 0 means the LLM primary model was unavailable
                # and a fallback model succeeded.  This is NOT a degradation to
                # regex — the LLM path still produced a valid result.  degraded
                # is only True when the LLM path fails entirely and we fall back
                # to regex extraction.
                return parsed.entities, False, parsed.reasoning

            # Parsed successfully but unexpected type — use regex.
            entities = await self._regex_fallback(alert_text)
            return entities, True, ""

        except (TimeoutError, OSError) as exc:
            # LLM transport / network-layer timeout (asyncio.timeout(), socket
            # errors) — these are not ShadowTraceError subclasses and must be
            # caught separately so the regex fallback engages instead of the
            # outer exception handler marking the agent as ``failed``.
            logger.warning(
                "LLM transport/timeout error for event=%s: %s",
                event_id,
                exc,
                exc_info=True,
            )
            entities = await self._regex_fallback(alert_text)
            return entities, True, ""

        except ShadowTraceError as exc:
            # Known failure modes: timeout, auth, rate-limit, provider error,
            # invalid JSON → all degrade gracefully to regex.
            if isinstance(exc, LLMError):
                logger.warning(
                    "LLM entity extraction failed for event=%s: %s",
                    event_id,
                    exc,
                    exc_info=True,
                )
            else:
                logger.warning(
                    "ShadowTrace error during entity extraction for event=%s: %s",
                    event_id,
                    exc,
                    exc_info=True,
                )
            entities = await self._regex_fallback(alert_text)
            return entities, True, ""

    async def _regex_fallback(self, alert_text: str) -> EntitySet:
        """Run regex extraction and convert to ``EntitySet``."""
        raw = extract_entities_regex(alert_text)
        return EntitySet(
            accounts=[
                AccountEntity(
                    entity_id=f"acct-{i}",
                    entity_type="account",
                    username=a,
                )
                for i, a in enumerate(raw.accounts, 1)
            ],
            hosts=[
                HostEntity(
                    entity_id=f"host-{i}",
                    entity_type="host",
                    hostname=h,
                )
                for i, h in enumerate(raw.hostnames, 1)
            ],
            ips=[
                IPEntity(
                    entity_id=f"ip-{i}",
                    entity_type="ip",
                    address=ip,
                    scope="internal" if is_internal_ip(ip) else "external",
                )
                for i, ip in enumerate(raw.ips, 1)
            ],
            domains=[
                DomainEntity(
                    entity_id=f"dom-{i}",
                    entity_type="domain",
                    fqdn=d,
                )
                for i, d in enumerate(raw.domains, 1)
            ],
            processes=[
                ProcessEntity(
                    entity_id=f"proc-{i}",
                    entity_type="process",
                    name=p,
                )
                for i, p in enumerate(raw.processes, 1)
            ],
            files=[
                FileEntity(
                    entity_id=f"file-{i}",
                    entity_type="file",
                    name=f,
                )
                for i, f in enumerate(raw.files, 1)
            ],
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    async def _write_triage_result(self, input: TriageAgentInput, result: TriageResult) -> None:
        """Persist ``triage_result`` to ``EventContext``.

        GuardrailViolationError (FIELD_OWNERSHIP mismatch) is always
        propagated — it indicates a code defect that must be fixed.
        Transient I/O failures are logged AND reflected on the result
        (``degraded=True``, reasoning annotation).  A lightweight
        ``triage_degraded`` flag is written separately so that downstream
        agents / recovery logic can detect the persistence gap even if the
        full result was not durably stored.
        """
        wm = self.working_memory
        if wm is None:
            return
        try:
            await wm.write(
                input.event_id,
                "triage_result",
                result.model_dump(mode="json"),
            )
        except GuardrailViolationError:
            # FIELD_OWNERSHIP violation is a code defect — must propagate.
            logger.exception(
                "GuardrailViolationError writing triage_result for event=%s",
                input.event_id,
            )
            raise
        except (DependencyUnavailableError, ConnectionError, TimeoutError):
            # Transient I/O failure (Redis, DB) — mark degraded so the
            # caller / downstream agents know this result is not durable.
            logger.warning(
                "Transient failure writing triage_result to EventContext for event=%s",
                input.event_id,
                exc_info=True,
            )
            result.degraded = True
            if result.reasoning:
                result.reasoning += " "
            result.reasoning += "triage_result persistence failed: working memory unavailable"
            # Best-effort persistence of the degraded flag so recovery /
            # downstream agents can detect the gap.
            await self._try_persist_degraded_flag(input.event_id)
        except ShadowTraceError as exc:
            if exc.retryable:
                logger.warning(
                    "Retryable error writing triage_result for event=%s: %s",
                    input.event_id,
                    exc.error_code,
                    exc_info=True,
                )
                result.degraded = True
                if result.reasoning:
                    result.reasoning += " "
                result.reasoning += f"triage_result persistence failed: {exc.error_code}"
                await self._try_persist_degraded_flag(input.event_id)
            else:
                raise

    async def _try_persist_degraded_flag(self, event_id: str) -> None:
        """Best-effort write of a lightweight ``triage_degraded`` flag.

        Called when the main ``triage_result`` write fails transiently.
        If even this lightweight write fails the error is logged but never
        propagated — the in-memory ``result.degraded=True`` is the final
        fallback for the immediate caller.
        """
        wm = self.working_memory
        if wm is None:
            return
        try:
            await wm.write(
                event_id,
                "triage_degraded",
                {
                    "degraded": True,
                    "reason": "triage_result persistence failed",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except ShadowTraceError:
            # All expected working-memory failures (connection errors,
            # GuardrailViolationError, DependencyUnavailableError, etc.)
            # are ShadowTraceError subclasses.  Programming errors
            # (AttributeError, TypeError, …) are intentionally allowed to
            # propagate so they surface in tests / monitoring rather than
            # being silently masked by an overly broad ``except Exception``.
            logger.exception(
                "Failed to persist triage_degraded flag for event=%s",
                event_id,
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _read_source_snapshot(self, event_id: str) -> dict[str, Any] | None:
        """Read the ``source_snapshot`` field from working memory.

        Transient I/O failures return None so the agent can continue with
        fallback keyword matching.  ``GuardrailViolationError`` is propagated
        because it indicates a code defect (e.g. FIELD_OWNERSHIP mismatch)
        that must be surfaced, consistent with ``_write_triage_result``.
        """
        wm = self.working_memory
        if wm is None:
            return None
        try:
            value = await wm.read(event_id, "source_snapshot")
            return value if isinstance(value, dict) else None
        except GuardrailViolationError:
            logger.exception(
                "GuardrailViolationError reading source_snapshot for event=%s",
                event_id,
            )
            raise
        except (DependencyUnavailableError, ConnectionError, TimeoutError):
            return None


__all__ = [
    "RuleBasedFalsePositiveHook",
    "SEVERITY_RULES",
    "TriageAgent",
    "_apply_severity_rules",
    "_extract_iocs",
    "_external_ip_in_text",
    "_map_event_type",
    "_merge_hint_entities",
    "_resolve_alert_type_from_snapshot",
]
