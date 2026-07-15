"""Mock XDR in-memory state: clock, cursors, disposition lineage (ISSUE-010)."""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from app.mock_xdr.models import MockFailureProfile, MockXDRScenario, ScenarioTick, TickOperation
from app.models.disposition import DispositionCommand, DispositionReceipt, TargetWritebackResult
from app.models.enums import (
    TERMINAL_SOURCE_DISPOSITIONS,
    ConfirmationEvidence,
    ConnectorStatus,
    DispositionIntentKind,
    ExecutionJobStatus,
    SourceDisposition,
    TargetWritebackStatus,
    WritebackStatus,
)
from app.models.source import SourceConnector
from app.models.workflow import validate_job_status_transition

# Mock-local SourceDisposition edges (ISSUE-010 step 4).
SOURCE_DISPOSITION_TRANSITIONS: dict[SourceDisposition, set[SourceDisposition]] = {
    SourceDisposition.PENDING: {
        SourceDisposition.PROCESSING,
        SourceDisposition.CONTAINED,
        SourceDisposition.COMPLETED,
        SourceDisposition.SUSPENDED,
        SourceDisposition.IGNORED,
        SourceDisposition.UNKNOWN,
    },
    SourceDisposition.PROCESSING: {
        SourceDisposition.CONTAINED,
        SourceDisposition.COMPLETED,
        SourceDisposition.SUSPENDED,
        SourceDisposition.IGNORED,
        SourceDisposition.UNKNOWN,
    },
    SourceDisposition.UNKNOWN: {
        SourceDisposition.PROCESSING,
        SourceDisposition.CONTAINED,
        SourceDisposition.COMPLETED,
        SourceDisposition.SUSPENDED,
        SourceDisposition.IGNORED,
    },
    SourceDisposition.CONTAINED: set(),
    SourceDisposition.COMPLETED: set(),
    SourceDisposition.SUSPENDED: set(),
    SourceDisposition.IGNORED: set(),
}

# ConnectorStatus: ONLINE↔DEGRADED/OFFLINE/UNKNOWN; ONLINE requires health recovery.
CONNECTOR_STATUS_TRANSITIONS: dict[ConnectorStatus, set[ConnectorStatus]] = {
    ConnectorStatus.ONLINE: {
        ConnectorStatus.DEGRADED,
        ConnectorStatus.OFFLINE,
        ConnectorStatus.UNKNOWN,
    },
    ConnectorStatus.DEGRADED: {
        ConnectorStatus.ONLINE,  # only after health_ok
        ConnectorStatus.OFFLINE,
        ConnectorStatus.UNKNOWN,
    },
    ConnectorStatus.OFFLINE: {
        ConnectorStatus.ONLINE,  # only after health_ok
        ConnectorStatus.DEGRADED,
        ConnectorStatus.UNKNOWN,
    },
    ConnectorStatus.UNKNOWN: {
        ConnectorStatus.ONLINE,  # only after health_ok
        ConnectorStatus.DEGRADED,
        ConnectorStatus.OFFLINE,
    },
}

ObjectKindName = Literal["incident", "alert", "asset", "log", "connector"]

# Recursively forbidden analysis / report / prompt / evidence payload keys.
_FORBIDDEN_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "report",
        "report_body",
        "report_markdown",
        "decision_trace",
        "prompt",
        "system_prompt",
        "user_prompt",
        "evidence",
        "evidence_raw",
        "raw_evidence",
        "analysis",
        "analysis_text",
        "matched_case_id",
        "similarity_score",
        "llm_reasoning",
        "chain_of_thought",
    }
)


class MockValidationError(Exception):
    """Illegal Mock state transition or lineage violation (does not mutate state)."""

    def __init__(self, message: str, *, error_code: str = "mock_validation_error") -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class MockAuthError(Exception):
    def __init__(self, message: str = "unauthorized") -> None:
        self.error_code = "unauthorized"
        self.message = message
        super().__init__(message)


class MockConflictError(Exception):
    def __init__(self, message: str, *, error_code: str = "version_conflict") -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def idempotency_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge ``patch`` into ``base`` without clobbering nested dict keys."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def find_forbidden_analysis_keys(obj: Any, *, path: str = "$") -> list[str]:
    """Return dotted paths of forbidden analysis/report/prompt/evidence keys."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            child = f"{path}.{key}"
            if key_l in _FORBIDDEN_PAYLOAD_KEYS:
                hits.append(child)
            hits.extend(find_forbidden_analysis_keys(value, path=child))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(find_forbidden_analysis_keys(item, path=f"{path}[{i}]"))
    return hits


@dataclass
class StoredObject:
    kind: ObjectKindName
    object_id: str
    body: dict[str, Any]
    source_updated_at: datetime
    concurrency_token: str
    payload_hash: str
    schema_version: str = "1"
    deleted: bool = False


@dataclass
class CursorPage:
    cursor: str
    object_ids: list[str]
    frozen_items: list[dict[str, Any]]
    kind: ObjectKindName
    updated_after: datetime | None
    page_size: int


@dataclass
class DispositionAttempt:
    command: DispositionCommand
    writeback_id: str
    receipts: list[DispositionReceipt] = field(default_factory=list)
    superseded: bool = False
    active: bool = True
    provider_job_id: str | None = None
    source_record_id: str = ""
    command_payload_hash: str = ""

    @property
    def latest_status(self) -> WritebackStatus | None:
        if not self.receipts:
            return None
        return max(self.receipts, key=lambda r: r.sequence).status


@dataclass
class ProviderJob:
    provider_job_id: str
    disposition_id: str
    status: ExecutionJobStatus
    writeback_id: str
    created_at: datetime
    terminal_writeback_status: WritebackStatus | None = None


@dataclass
class MockXDRState:
    """In-memory Mock XDR environment (virtual clock + stores + writeback)."""

    scenario: MockXDRScenario | None = None
    failure_profile: MockFailureProfile = field(default_factory=MockFailureProfile)
    clock: datetime = field(default_factory=_utc_now)
    request_counter: int = 0
    objects: dict[tuple[ObjectKindName, str], StoredObject] = field(default_factory=dict)
    connectors: dict[str, SourceConnector] = field(default_factory=dict)
    connector_health_ok: dict[str, bool] = field(default_factory=dict)
    # cursor -> frozen page
    cursor_pages: dict[str, CursorPage] = field(default_factory=dict)
    pending_ticks: list[ScenarioTick] = field(default_factory=list)
    # kind -> watermark (last successfully committed cursor)
    watermarks: dict[ObjectKindName, str | None] = field(default_factory=dict)
    disposition_by_id: dict[str, DispositionAttempt] = field(default_factory=dict)
    disposition_by_idem_hash: dict[str, str] = field(default_factory=dict)
    jobs: dict[str, ProviderJob] = field(default_factory=dict)
    # (source_object_id, closure_cycle) -> active EVENT_STATUS_UPDATE disposition_id
    active_terminal_heads: dict[tuple[str, int], str] = field(default_factory=dict)
    writeback_seq: int = 0
    # Auth tokens (read vs write clients)
    read_token: str = "mock-read-token"
    write_token: str = "mock-write-token"
    # Captured inbound payloads for analysis-leak assertions
    captured_requests: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ seed

    def load_scenario(self, scenario: MockXDRScenario) -> None:
        self.reset()
        self.scenario = scenario
        self.failure_profile = scenario.failure_profile.model_copy(deep=True)
        self.clock = (
            scenario.base_time
            if scenario.base_time.tzinfo
            else scenario.base_time.replace(tzinfo=UTC)
        )
        for connector in scenario.connectors:
            self.connectors[connector.connector_id] = connector.model_copy(deep=True)
            self.connector_health_ok[connector.connector_id] = (
                connector.status is ConnectorStatus.ONLINE
            )
        for incident in scenario.incidents:
            self._put_model("incident", incident.reference.source_object_id, incident)
        for alert in scenario.alerts:
            self._put_model("alert", alert.reference.source_object_id, alert)
        for asset in scenario.assets:
            self._put_model("asset", asset.reference.source_object_id, asset)
        for log in scenario.logs:
            self._put_model("log", log.reference.source_object_id, log)
        self.pending_ticks = sorted(
            (tick.model_copy(deep=True) for tick in scenario.ticks),
            key=lambda tick: tick.offset_seconds,
        )
        self._apply_due_ticks()

    def reset(self) -> None:
        self.scenario = None
        self.failure_profile = MockFailureProfile()
        self.clock = _utc_now()
        self.request_counter = 0
        self.objects.clear()
        self.connectors.clear()
        self.connector_health_ok.clear()
        self.cursor_pages.clear()
        self.pending_ticks.clear()
        self.watermarks.clear()
        self.disposition_by_id.clear()
        self.disposition_by_idem_hash.clear()
        self.jobs.clear()
        self.active_terminal_heads.clear()
        self.writeback_seq = 0
        self.captured_requests.clear()

    def advance_clock(self, seconds: float) -> datetime:
        if seconds < 0:
            raise MockValidationError(
                "virtual clock cannot move backwards",
                error_code="invalid_operation",
            )
        self.clock = self.clock + timedelta(seconds=seconds)
        self._apply_due_ticks()
        return self.clock

    def _apply_due_ticks(self) -> None:
        if self.scenario is None:
            return
        base_time = (
            self.scenario.base_time
            if self.scenario.base_time.tzinfo
            else self.scenario.base_time.replace(tzinfo=UTC)
        )
        elapsed_seconds = (self.clock - base_time).total_seconds()
        applied_offsets: list[int] = []
        while self.pending_ticks and self.pending_ticks[0].offset_seconds <= elapsed_seconds:
            tick = self.pending_ticks.pop(0)
            self.apply_tick(tick)
            applied_offsets.append(tick.offset_seconds)

    # --------------------------------------------------------------- storage

    def _new_token(self) -> str:
        return secrets.token_hex(8)

    def _put_model(self, kind: ObjectKindName, object_id: str, model: Any) -> StoredObject:
        body = model.model_dump(mode="json")
        return self.upsert_object(
            kind,
            object_id,
            body,
            schema_version=getattr(getattr(model, "reference", None), "schema_version", "1")
            if kind != "connector"
            else getattr(model, "schema_version", "1"),
        )

    def upsert_object(
        self,
        kind: ObjectKindName,
        object_id: str,
        body: dict[str, Any],
        *,
        schema_version: str | None = None,
    ) -> StoredObject:
        key = (kind, object_id)
        existing = self.objects.get(key)
        token = self._new_token()
        updated_at = self.clock
        version = schema_version or (
            self.failure_profile.schema_version_override
            or (existing.schema_version if existing else "1")
        )
        digest = payload_hash(body)
        stored = StoredObject(
            kind=kind,
            object_id=object_id,
            body=body,
            source_updated_at=updated_at,
            concurrency_token=token,
            payload_hash=digest,
            schema_version=version,
            deleted=False,
        )
        self.objects[key] = stored
        # Mirror disposition onto body.reference when present
        ref = body.get("reference")
        if isinstance(ref, dict):
            ref["source_concurrency_token"] = token
            ref["source_updated_at"] = updated_at.isoformat()
            ref["raw_payload_hash"] = digest
            ref["schema_version"] = version
        return stored

    def delete_object(self, kind: ObjectKindName, object_id: str) -> None:
        key = (kind, object_id)
        existing = self.objects.get(key)
        if existing is None:
            return
        existing.deleted = True
        existing.source_updated_at = self.clock
        existing.concurrency_token = self._new_token()
        existing.payload_hash = payload_hash({"deleted": True, "id": object_id})

    def apply_tick(self, tick: ScenarioTick) -> None:
        kind = tick.object_type
        if kind not in ("incident", "alert", "asset", "log", "connector"):
            raise MockValidationError(f"unknown object_type {tick.object_type}")
        typed_kind: ObjectKindName = kind  # type: ignore[assignment]
        if tick.operation is TickOperation.DELETE:
            self.delete_object(typed_kind, tick.object_id)
            return
        if tick.operation is TickOperation.CONNECTOR_CHANGE:
            connector = self.connectors.get(tick.object_id)
            if connector is None:
                raise MockValidationError(f"unknown connector {tick.object_id}")
            patch = dict(tick.patch)
            health_ok = bool(patch.pop("health_ok", self.connector_health_ok.get(tick.object_id)))
            if "status" in patch:
                target = ConnectorStatus(patch["status"])
                self.transition_connector(tick.object_id, target, health_ok=health_ok)
            self.connector_health_ok[tick.object_id] = health_ok
            return
        if tick.operation is TickOperation.UPSERT:
            key = (typed_kind, tick.object_id)
            base = dict(self.objects[key].body) if key in self.objects else {}
            base = _deep_merge(base, tick.patch)
            self.upsert_object(typed_kind, tick.object_id, base)

    # ---------------------------------------------------- disposition edges

    def transition_source_disposition(
        self,
        kind: ObjectKindName,
        object_id: str,
        target: SourceDisposition,
        *,
        allow_unknown_recovery: bool = False,
    ) -> StoredObject:
        key = (kind, object_id)
        stored = self.objects.get(key)
        if stored is None or stored.deleted:
            raise MockValidationError(f"{kind}/{object_id} not found")
        ref = stored.body.get("reference") or {}
        current_raw = ref.get("source_disposition", SourceDisposition.UNKNOWN.value)
        current = SourceDisposition(current_raw)
        if current in TERMINAL_SOURCE_DISPOSITIONS:
            raise MockValidationError(
                f"terminal disposition {current.value} cannot transition",
                error_code="invalid_state_transition",
            )
        allowed = SOURCE_DISPOSITION_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise MockValidationError(
                f"illegal SourceDisposition {current.value}→{target.value}",
                error_code="invalid_state_transition",
            )
        if current is SourceDisposition.UNKNOWN and not allow_unknown_recovery:
            raise MockValidationError(
                "unknown→* requires authoritative poll/readback or test control",
                error_code="invalid_state_transition",
            )
        ref = dict(ref)
        ref["source_disposition"] = target.value
        ref["source_status_raw"] = target.value
        body = dict(stored.body)
        body["reference"] = ref
        return self.upsert_object(kind, object_id, body, schema_version=stored.schema_version)

    def transition_connector(
        self,
        connector_id: str,
        target: ConnectorStatus,
        *,
        health_ok: bool = False,
    ) -> SourceConnector:
        connector = self.connectors.get(connector_id)
        if connector is None:
            raise MockValidationError(f"unknown connector {connector_id}")
        current = connector.status
        allowed = CONNECTOR_STATUS_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise MockValidationError(
                f"illegal ConnectorStatus {current.value}→{target.value}",
                error_code="invalid_state_transition",
            )
        if target is ConnectorStatus.ONLINE and not health_ok:
            raise MockValidationError(
                "ConnectorStatus→ONLINE requires successful health check",
                error_code="invalid_state_transition",
            )
        updated = connector.model_copy(update={"status": target, "last_sync_at": self.clock})
        self.connectors[connector_id] = updated
        self.connector_health_ok[connector_id] = health_ok and target is ConnectorStatus.ONLINE
        return updated

    # ----------------------------------------------------------- pagination

    def _sorted_ids(
        self,
        kind: ObjectKindName,
        *,
        updated_after: datetime | None,
    ) -> list[str]:
        items: list[StoredObject] = [
            o for (k, _), o in self.objects.items() if k == kind and not o.deleted
        ]
        if updated_after is not None:
            items = [o for o in items if o.source_updated_at > updated_after]
        items.sort(key=lambda o: (o.source_updated_at, o.object_id))
        if self.failure_profile.out_of_order_updates and len(items) > 1:
            # Deterministic shuffle of the first page only for discovery tests;
            # cursor pages remain frozen once issued.
            items = list(reversed(items))
        return [o.object_id for o in items]

    def list_page(
        self,
        kind: ObjectKindName,
        *,
        page_size: int = 100,
        cursor: str | None = None,
        updated_after: datetime | None = None,
        commit_watermark: bool = False,
    ) -> dict[str, Any]:
        if cursor:
            page = self.cursor_pages.get(cursor)
            if page is None:
                raise MockValidationError("unknown cursor", error_code="invalid_cursor")
            # Idempotent retry: same cursor → same immutable payload snapshot.
            items = copy.deepcopy(page.frozen_items)
            next_cursor = None
            # Find continuation from watermark chain stored alongside
            next_key = f"{cursor}::next"
            next_cursor = (
                self.cursor_pages[next_key].cursor if next_key in self.cursor_pages else None
            )
            if commit_watermark:
                self.watermarks[kind] = cursor
            return {
                "items": items,
                "next_cursor": next_cursor,
                "page_size": page.page_size,
                "cursor": cursor,
            }

        ids = self._sorted_ids(kind, updated_after=updated_after)
        snapshot_fingerprint = "|".join(
            f"{oid}:{self.objects[(kind, oid)].payload_hash}" for oid in ids
        )
        pages: list[list[str]] = [
            ids[i : i + page_size] for i in range(0, max(len(ids), 1), page_size)
        ] or [[]]
        # Build stable cursors for every page
        cursors: list[str] = []
        for idx, chunk in enumerate(pages):
            after = updated_after.isoformat() if updated_after else ""
            material = (
                f"{kind}|{after}|{idx}|{page_size}|{self.failure_profile.seed}"
                f"|{snapshot_fingerprint}"
            )
            c = hashlib.sha256(material.encode()).hexdigest()[:24]
            cursors.append(c)
            frozen_items = [copy.deepcopy(self._public_object(kind, oid)) for oid in chunk]
            if self.failure_profile.duplicate_page and frozen_items:
                frozen_items.append(copy.deepcopy(frozen_items[0]))
            self.cursor_pages[c] = CursorPage(
                cursor=c,
                object_ids=list(chunk),
                frozen_items=frozen_items,
                kind=kind,
                updated_after=updated_after,
                page_size=page_size,
            )
        for idx in range(len(cursors) - 1):
            self.cursor_pages[f"{cursors[idx]}::next"] = self.cursor_pages[cursors[idx + 1]]

        first = cursors[0]
        items = copy.deepcopy(self.cursor_pages[first].frozen_items)
        next_cursor = cursors[1] if len(cursors) > 1 else None
        if commit_watermark:
            self.watermarks[kind] = first
        return {
            "items": items,
            "next_cursor": next_cursor,
            "page_size": page_size,
            "cursor": first,
        }

    def _public_object(self, kind: ObjectKindName, object_id: str) -> dict[str, Any]:
        stored = self.objects[(kind, object_id)]
        body = copy.deepcopy(stored.body)
        # Preserve external IDs as opaque strings; never rewrite.
        body["_mock"] = {
            "source_updated_at": stored.source_updated_at.isoformat(),
            "concurrency_token": stored.concurrency_token,
            "payload_hash": stored.payload_hash,
            "schema_version": stored.schema_version,
            "external_id": object_id,
        }
        for key in self.failure_profile.missing_fields:
            body.pop(key, None)
            if "reference" in body and isinstance(body["reference"], dict):
                body["reference"].pop(key, None)
        return body

    def get_object(self, kind: ObjectKindName, object_id: str) -> dict[str, Any]:
        key = (kind, object_id)
        stored = self.objects.get(key)
        if stored is None or stored.deleted:
            raise MockValidationError(f"{kind}/{object_id} not found", error_code="not_found")
        return self._public_object(kind, object_id)

    # --------------------------------------------------------- dispositions

    def _next_writeback_id(self) -> str:
        self.writeback_seq += 1
        return f"wbk-mock-{self.writeback_seq:08d}"

    def submit_disposition(self, command: DispositionCommand) -> DispositionReceipt:
        raw = command.model_dump(mode="json")
        self.captured_requests.append(raw)
        leaks = find_forbidden_analysis_keys(raw)
        if leaks:
            raise MockValidationError(
                f"forbidden analysis fields in disposition payload: {leaks}",
                error_code="unauthorized_field",
            )

        command_payload_hash = payload_hash(raw)
        idem_hash = idempotency_key_hash(command.idempotency_key)
        existing_id = self.disposition_by_idem_hash.get(idem_hash)
        if existing_id is not None:
            attempt = self.disposition_by_id[existing_id]
            # Same idempotency key MUST carry the same payload; reuse with a
            # different command is a caller bug (masks outbox regressions).
            prior_hash = attempt.command_payload_hash
            if prior_hash and prior_hash != command_payload_hash:
                raise MockValidationError(
                    "idempotency key reused with a different payload",
                    error_code="idempotency_key_reuse",
                )
            return attempt.receipts[-1].model_copy(deep=True)
        if command.disposition_id in self.disposition_by_id:
            raise MockValidationError(
                "disposition_id reused for a different attempt",
                error_code="disposition_id_reuse",
            )

        locator = command.source_locator
        kind_value = locator.source_kind.value
        if kind_value not in ("incident", "alert", "asset", "log"):
            raise MockValidationError(f"unsupported source_kind {kind_value}")
        kind: ObjectKindName = kind_value  # type: ignore[assignment]
        object_id = locator.source_object_id
        key = (kind, object_id)
        stored = self.objects.get(key)
        if stored is None or stored.deleted:
            raise MockValidationError(
                f"source object {object_id} not found", error_code="not_found"
            )

        # Token conflict injection / CAS
        if self.failure_profile.force_token_conflict or (
            command.source_concurrency_token is not None
            and command.source_concurrency_token != stored.concurrency_token
        ):
            writeback_id = self._next_writeback_id()
            receipt = DispositionReceipt(
                writeback_id=writeback_id,
                sequence=1,
                disposition_id=command.disposition_id,
                action_id=command.action_id,
                source_record_id=f"src-mock-{object_id}",
                status=WritebackStatus.CONFLICT,
                provider_code="version_conflict",
                provider_message="concurrency token mismatch",
                observed_at=self.clock,
                submitted_at=self.clock,
                target_results=[],
                raw_result={"mock": True},
                simulated=True,
            )
            attempt = DispositionAttempt(
                command=command.model_copy(deep=True),
                writeback_id=writeback_id,
                receipts=[receipt.model_copy(deep=True)],
                active=False,
                source_record_id=receipt.source_record_id,
                command_payload_hash=command_payload_hash,
            )
            self.disposition_by_id[command.disposition_id] = attempt
            self.disposition_by_idem_hash[idem_hash] = command.disposition_id
            return receipt

        # EVENT_STATUS_UPDATE lineage: one active head per (object, closure_cycle)
        if command.intent_kind is DispositionIntentKind.EVENT_STATUS_UPDATE:
            slot = (object_id, command.closure_cycle)
            current_head = self.active_terminal_heads.get(slot)
            if current_head is not None and current_head != command.disposition_id:
                head = self.disposition_by_id[current_head]
                if command.supersedes_disposition_id == current_head:
                    # Approved supersede: deactivate old head, keep history
                    head.active = False
                    head.superseded = True
                else:
                    # Parallel active head or unapproved different payload
                    if head.active:
                        raise MockValidationError(
                            "parallel active EVENT_STATUS_UPDATE head rejected",
                            error_code="invalid_operation",
                        )
                # Different terminal payload without supersedes also rejected when head confirmed
                if head.active and head.latest_status is WritebackStatus.CONFIRMED:
                    old_params = head.command.operation_params.model_dump(mode="json")
                    new_params = command.operation_params.model_dump(mode="json")
                    if old_params != new_params and not command.supersedes_disposition_id:
                        raise MockValidationError(
                            "different terminal payload requires supersedes_disposition_id",
                            error_code="invalid_operation",
                        )

        writeback_id = self._next_writeback_id()
        source_record_id = f"src-mock-{object_id}"

        # Partial target success injection
        target_results: list[TargetWritebackResult] = []
        status = WritebackStatus.ACCEPTED
        evidence: ConfirmationEvidence | None = None
        provider_job_id: str | None = None
        confirmed_at: datetime | None = None

        if self.failure_profile.force_partial_targets and len(command.target_results) >= 2:
            for i, t in enumerate(command.target_results):
                tw = TargetWritebackStatus.CONFIRMED if i == 0 else TargetWritebackStatus.FAILED
                target_results.append(
                    TargetWritebackResult(
                        canonical_target=t.canonical_target,
                        status=tw,
                        provider_code="partial" if tw is TargetWritebackStatus.FAILED else None,
                    )
                )
            status = WritebackStatus.PARTIAL
        elif self.failure_profile.async_disposition:
            status = WritebackStatus.ACCEPTED
            provider_job_id = f"pjob-{secrets.token_hex(6)}"
            self.jobs[provider_job_id] = ProviderJob(
                provider_job_id=provider_job_id,
                disposition_id=command.disposition_id,
                status=ExecutionJobStatus.QUEUED,
                writeback_id=writeback_id,
                created_at=self.clock,
                terminal_writeback_status=WritebackStatus.CONFIRMED,
            )
        else:
            # Sync path: do NOT self-confirm instantly from write alone, and do
            # NOT mutate the source object's disposition yet. The write is only
            # ACCEPTED; the authoritative source_disposition transition happens
            # in confirm_via_readback (provider truth), never on accept.
            status = WritebackStatus.ACCEPTED

        receipt = DispositionReceipt(
            writeback_id=writeback_id,
            sequence=1,
            disposition_id=command.disposition_id,
            action_id=command.action_id,
            source_record_id=source_record_id,
            status=status,
            confirmation_evidence=evidence,
            provider_job_id=provider_job_id,
            provider_record_id=f"prec-{secrets.token_hex(4)}"
            if status is not WritebackStatus.FAILED
            else None,
            observed_at=self.clock,
            submitted_at=self.clock,
            confirmed_at=confirmed_at,
            target_results=target_results,
            raw_result={"mock": True, "async": bool(provider_job_id)},
            simulated=True,
        )
        attempt = DispositionAttempt(
            command=command.model_copy(deep=True),
            writeback_id=writeback_id,
            receipts=[receipt.model_copy(deep=True)],
            active=True,
            provider_job_id=provider_job_id,
            source_record_id=source_record_id,
            command_payload_hash=command_payload_hash,
        )
        self.disposition_by_id[command.disposition_id] = attempt
        self.disposition_by_idem_hash[idem_hash] = command.disposition_id
        if command.intent_kind is DispositionIntentKind.EVENT_STATUS_UPDATE and attempt.active:
            self.active_terminal_heads[(object_id, command.closure_cycle)] = command.disposition_id
        return receipt

    def _apply_confirmed_source_disposition(self, command: DispositionCommand) -> None:
        """Move the source object's disposition to provider truth on confirm only.

        No-op unless this is an EVENT_STATUS_UPDATE carrying a target disposition.
        If the local edge table forbids the transition (e.g. already terminal), the
        provider-confirmed writeback still stands; we simply do not re-apply.
        """
        if command.intent_kind is not DispositionIntentKind.EVENT_STATUS_UPDATE:
            return
        target_disp = getattr(command.operation_params, "target_disposition", None)
        if target_disp is None:
            return
        locator = command.source_locator
        kind_value = locator.source_kind.value
        if kind_value not in ("incident", "alert", "asset", "log"):
            return
        kind: ObjectKindName = kind_value  # type: ignore[assignment]
        object_id = locator.source_object_id
        stored = self.objects.get((kind, object_id))
        if stored is None or stored.deleted:
            return
        ref = stored.body.get("reference") or {}
        current_raw = ref.get("source_disposition", SourceDisposition.UNKNOWN.value)
        if SourceDisposition(current_raw) == target_disp:
            return
        try:
            self.transition_source_disposition(
                kind,
                object_id,
                target_disp,
                allow_unknown_recovery=True,
            )
        except MockValidationError:
            # Provider confirmed the action; local model just can't re-apply the edge.
            return

    def confirm_via_readback(self, disposition_id: str) -> DispositionReceipt:
        """Authoritative readback confirmation (Mock P0 evidence=readback_verified)."""
        attempt = self.disposition_by_id.get(disposition_id)
        if attempt is None:
            raise MockValidationError("disposition not found", error_code="not_found")
        latest = attempt.receipts[-1]
        if latest.status is WritebackStatus.CONFIRMED:
            return latest
        if latest.status not in (
            WritebackStatus.ACCEPTED,
            WritebackStatus.PARTIAL,
            WritebackStatus.UNKNOWN,
            WritebackStatus.SENDING,
        ):
            raise MockValidationError(
                f"cannot confirm from status {latest.status.value}",
                error_code="invalid_state_transition",
            )
        locator = attempt.command.source_locator
        kind_value = locator.source_kind.value
        stored: StoredObject | None = None
        if kind_value in ("incident", "alert", "asset", "log"):
            kind: ObjectKindName = kind_value  # type: ignore[assignment]
            stored = self.objects.get((kind, locator.source_object_id))
        target_disposition = getattr(attempt.command.operation_params, "target_disposition", None)
        observed_disposition = None
        observed_token = None
        if stored is not None:
            observed_disposition = (stored.body.get("reference") or {}).get("source_disposition")
            observed_token = stored.concurrency_token
        seq = latest.sequence + 1
        target_matches = (
            target_disposition is not None and observed_disposition == target_disposition.value
        )
        token_changed = (
            attempt.command.source_concurrency_token is None
            or observed_token != attempt.command.source_concurrency_token
        )
        if target_matches and token_changed:
            receipt = latest.model_copy(
                update={
                    "sequence": seq,
                    "status": WritebackStatus.CONFIRMED,
                    "confirmation_evidence": ConfirmationEvidence.READBACK_VERIFIED,
                    "provider_code": None,
                    "provider_message": None,
                    "confirmed_at": self.clock,
                    "observed_at": self.clock,
                }
            )
        else:
            authoritative_changed = (
                attempt.command.source_concurrency_token is not None
                and observed_token is not None
                and observed_token != attempt.command.source_concurrency_token
            )
            receipt = latest.model_copy(
                update={
                    "sequence": seq,
                    "status": (
                        WritebackStatus.CONFLICT
                        if authoritative_changed
                        else WritebackStatus.UNKNOWN
                    ),
                    "confirmation_evidence": None,
                    "provider_code": (
                        "readback_mismatch" if authoritative_changed else "readback_not_yet_applied"
                    ),
                    "provider_message": (
                        "authoritative state changed to a different disposition"
                        if authoritative_changed
                        else "authoritative state has not proved the requested disposition"
                    ),
                    "confirmed_at": None,
                    "observed_at": self.clock,
                }
            )
        attempt.receipts.append(receipt.model_copy(deep=True))
        return receipt

    def lookup_by_idempotency(self, key_hash: str) -> DispositionReceipt | None:
        disp_id = self.disposition_by_idem_hash.get(key_hash)
        if disp_id is None:
            return None
        return self.disposition_by_id[disp_id].receipts[-1].model_copy(deep=True)

    def get_job(self, provider_job_id: str) -> ProviderJob:
        job = self.jobs.get(provider_job_id)
        if job is None:
            raise MockValidationError("job not found", error_code="not_found")
        return job

    def advance_job(
        self,
        provider_job_id: str,
        target: ExecutionJobStatus,
        *,
        provider_confirmed_terminal: bool = False,
    ) -> ProviderJob:
        from app.core.errors import InvalidStateTransitionError

        job = self.get_job(provider_job_id)
        try:
            validate_job_status_transition(
                job.status,
                target,
                provider_confirmed_terminal=provider_confirmed_terminal,
            )
        except InvalidStateTransitionError as exc:
            raise MockValidationError(
                str(exc),
                error_code="invalid_state_transition",
            ) from exc
        job.status = target
        if target in {
            ExecutionJobStatus.SUCCESS,
            ExecutionJobStatus.PARTIAL_SUCCESS,
            ExecutionJobStatus.FAILED,
            ExecutionJobStatus.TIMED_OUT,
            ExecutionJobStatus.CANCELLED,
        }:
            attempt = self.disposition_by_id[job.disposition_id]
            latest = attempt.receipts[-1]
            if target is ExecutionJobStatus.SUCCESS:
                confirmed = self.confirm_via_readback(job.disposition_id)
                job.terminal_writeback_status = confirmed.status
            else:
                wb_status = (
                    WritebackStatus.PARTIAL
                    if target is ExecutionJobStatus.PARTIAL_SUCCESS
                    else WritebackStatus.FAILED
                )
                receipt = latest.model_copy(
                    update={
                        "sequence": latest.sequence + 1,
                        "status": wb_status,
                        "observed_at": self.clock,
                        "confirmation_evidence": None,
                    }
                )
                attempt.receipts.append(receipt)
                job.terminal_writeback_status = wb_status
        return job

    def readback_source_disposition(self, kind: ObjectKindName, object_id: str) -> dict[str, Any]:
        stored = self.objects.get((kind, object_id))
        if stored is None or stored.deleted:
            raise MockValidationError("not found", error_code="not_found")
        ref = stored.body.get("reference") or {}
        return {
            "source_object_id": object_id,
            "source_kind": kind,
            "source_disposition": ref.get("source_disposition"),
            "source_status_raw": ref.get("source_status_raw"),
            "concurrency_token": stored.concurrency_token,
            "source_updated_at": stored.source_updated_at.isoformat(),
            "payload_hash": stored.payload_hash,
            "schema_version": stored.schema_version,
        }

    def required_events_missing_terminal_lineage(self) -> list[str]:
        """Return incident IDs marked required that lack an active terminal head."""
        missing: list[str] = []
        if self.scenario is None:
            return missing
        required = bool(self.scenario.expected_outcome.get("disposition_policy") == "required")
        if not required:
            # Also treat connectors with disposition_policy_default=required
            required_connectors = {
                c.connector_id
                for c in self.connectors.values()
                if c.disposition_policy_default is not None
                and c.disposition_policy_default.value == "required"
            }
        else:
            required_connectors = set(self.connectors)
        for (kind, oid), stored in self.objects.items():
            if kind != "incident" or stored.deleted:
                continue
            ref = stored.body.get("reference") or {}
            connector_id = ref.get("connector_id")
            if required or connector_id in required_connectors:
                cycles = {
                    slot[1]
                    for slot, disp_id in self.active_terminal_heads.items()
                    if slot[0] == oid
                }
                if not cycles:
                    missing.append(oid)
                else:
                    for cycle in cycles:
                        head_id = self.active_terminal_heads.get((oid, cycle))
                        if head_id is None:
                            missing.append(oid)
                            continue
                        attempt = self.disposition_by_id[head_id]
                        if (
                            not attempt.active
                            or attempt.latest_status is not WritebackStatus.CONFIRMED
                        ):
                            # Still missing confirmed terminal for this cycle
                            if attempt.latest_status is not WritebackStatus.CONFIRMED:
                                missing.append(f"{oid}@{cycle}")
        return missing
