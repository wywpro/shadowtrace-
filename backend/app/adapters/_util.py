"""Shared adapter helpers: credential checks, sanitization, parsing (ISSUE-012)."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.adapters.source.base import DataQualityRecorder
from app.core.errors import ValidationError as ShadowTraceValidationError
from app.core.sanitization import redact_sensitive_text, sanitize_data
from app.models.disposition import DispositionReceipt
from app.models.enums import SourceObjectKind
from app.models.source import (
    SourceAlert,
    SourceAsset,
    SourceConnector,
    SourceIncident,
    SourceLog,
)

_KIND_MODEL: dict[str, type[SourceIncident | SourceAlert | SourceAsset | SourceLog]] = {
    SourceObjectKind.INCIDENT.value: SourceIncident,
    SourceObjectKind.ALERT.value: SourceAlert,
    SourceObjectKind.ASSET.value: SourceAsset,
    SourceObjectKind.LOG.value: SourceLog,
    "incident": SourceIncident,
    "alert": SourceAlert,
    "asset": SourceAsset,
    "log": SourceLog,
}


def require_separated_credentials(*, read_token: str, write_token: str) -> None:
    """Mock (and preferred live) path: read and write credentials must differ."""
    if not read_token or not write_token:
        raise ValueError("read_token and write_token are required")
    if read_token == write_token:
        raise ValueError(
            "Mock XDR requires separated read/write credentials; identical tokens are rejected"
        )


def sanitize_raw_result(payload: dict[str, Any], *, max_bytes: int = 8_192) -> dict[str, Any]:
    """Redact secret keys/values and truncate large blobs for local receipt storage."""

    def _truncate(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): _truncate(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_truncate(item) for item in obj[:100]]
        if isinstance(obj, str) and len(obj) > 2_000:
            return obj[:2_000] + "…[truncated]"
        return obj

    cleaned = _truncate(sanitize_data(payload, replacement="***"))
    encoded = str(cleaned)
    truncated = len(encoded) > max_bytes
    if truncated:
        return {"truncated": True, "preview": encoded[:max_bytes]}
    return cleaned if isinstance(cleaned, dict) else {"value": cleaned}


def sanitize_disposition_receipt(receipt: DispositionReceipt) -> DispositionReceipt:
    """Return a receipt safe for persistence and internal event propagation."""

    raw_result = sanitize_raw_result(dict(receipt.raw_result))
    target_results = [
        target.model_copy(
            update={
                "provider_code": (
                    redact_sensitive_text(target.provider_code)
                    if target.provider_code is not None
                    else None
                ),
                "message_code": (
                    redact_sensitive_text(target.message_code)
                    if target.message_code is not None
                    else None
                ),
                "artifact_ref": (
                    redact_sensitive_text(target.artifact_ref)
                    if target.artifact_ref is not None
                    else None
                ),
            }
        )
        for target in receipt.target_results
    ]
    return receipt.model_copy(
        update={
            "provider_code": (
                redact_sensitive_text(receipt.provider_code)
                if receipt.provider_code is not None
                else None
            ),
            "provider_message": (
                redact_sensitive_text(receipt.provider_message)
                if receipt.provider_message is not None
                else None
            ),
            "target_results": target_results,
            "raw_result": raw_result,
            "truncated": receipt.truncated or raw_result.get("truncated") is True,
        }
    )


def parse_source_item(
    kind: str,
    body: dict[str, Any],
    *,
    quality: DataQualityRecorder | None = None,
) -> SourceIncident | SourceAlert | SourceAsset | SourceLog | None:
    """Parse a Mock/file payload into a Source* model; unknown fields stay in raw_payload."""
    model = _KIND_MODEL.get(kind)
    if model is None:
        if quality is not None:
            quality.record(
                stage="source_normalize",
                error_category="unknown_object_kind",
                detail={"kind": kind},
            )
        return None

    payload = dict(body)
    mock_meta = payload.pop("_mock", None)
    unknown_fields = sorted(set(payload) - set(model.model_fields))
    # Preserve opaque external identity; fold unknown extras into raw_payload.
    extras = {key: payload.pop(key) for key in unknown_fields}
    if extras:
        existing_raw = payload.get("raw_payload")
        raw_payload = dict(existing_raw) if isinstance(existing_raw, dict) else {}
        raw_payload.update(extras)
        payload["raw_payload"] = raw_payload
    try:
        item = model.model_validate(payload)
    except ValidationError as exc:
        if quality is not None:
            quality.record(
                stage="source_normalize",
                error_category="schema_validation",
                detail={
                    "kind": kind,
                    "errors": exc.errors(include_input=False, include_url=False),
                },
            )
        return None

    if mock_meta and isinstance(mock_meta, dict):
        # Concurrency / watermark metadata lives outside the immutable reference.
        raw = dict(item.raw_payload)
        raw.setdefault("_mock", mock_meta)
        item = item.model_copy(update={"raw_payload": raw})
        token = mock_meta.get("concurrency_token")
        if token and item.reference.source_concurrency_token is None:
            item = item.model_copy(
                update={
                    "reference": item.reference.model_copy(
                        update={"source_concurrency_token": token}
                    )
                }
            )
    return item


def parse_connector(body: dict[str, Any]) -> SourceConnector:
    return SourceConnector.model_validate(body)


def kind_to_path(kind: SourceObjectKind | str) -> str:
    value = kind.value if isinstance(kind, SourceObjectKind) else str(kind)
    plural = {
        "incident": "incidents",
        "alert": "alerts",
        "asset": "assets",
        "log": "logs",
    }.get(value)
    if plural is None:
        raise ShadowTraceValidationError(
            f"unsupported source kind {value!r}",
            error_code="adapter_validation_error",
            details={"kind": value},
        )
    return plural
