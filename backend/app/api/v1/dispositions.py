"""Disposition / writeback read + controlled-retry endpoints (ISSUE-059).

These endpoints only read or controllably re-enqueue the outbox; they never
construct disposition commands or bypass the ApprovalEngine.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter

from app.api.v1 import schemas as s
from app.api.v1.deps import DispositionSyncDep
from app.core.auth import (
    ROLE_ADMIN,
    ROLE_DISPOSITION_OPERATOR,
    CurrentPrincipal,
    Principal,
    require_roles,
)
from app.core.errors import EventNotFoundError
from app.models.enums import ConfirmationEvidence, WritebackStatus

router = APIRouter(tags=["dispositions"])


@router.get("/events/{event_id}/dispositions", response_model=s.DispositionListResponse)
async def list_event_dispositions(
    event_id: str,
    principal: CurrentPrincipal,
    sync: DispositionSyncDep,
) -> s.DispositionListResponse:
    items = await sync.list_event_dispositions(event_id)
    return s.DispositionListResponse(
        event_id=event_id,
        items=[
            s.DispositionResponse(disposition=command, writeback_status=status)
            for command, status in items
        ],
    )


@router.get("/dispositions/{disposition_id}", response_model=s.DispositionResponse)
async def get_disposition(
    disposition_id: str,
    principal: CurrentPrincipal,
    sync: DispositionSyncDep,
) -> s.DispositionResponse:
    # disposition_id maps to the outbox command's disposition_id field.
    raise EventNotFoundError(
        f"disposition {disposition_id} not found",
        details={"disposition_id": disposition_id},
    )


@router.get("/writebacks/{writeback_id}", response_model=s.WritebackResponse)
async def get_writeback(
    writeback_id: str,
    principal: CurrentPrincipal,
    sync: DispositionSyncDep,
) -> s.WritebackResponse:
    record, receipt = await sync.get_writeback(writeback_id)
    status = (
        WritebackStatus(record.latest_writeback_status)
        if record.latest_writeback_status
        else WritebackStatus.PENDING
    )
    confirmation = receipt.confirmation_evidence if receipt is not None else None
    return s.WritebackResponse(
        writeback_id=writeback_id,
        disposition_id=record.disposition_id,
        action_id=record.action_id,
        status=status,
        confirmation_evidence=confirmation,
        evidence_tier=(
            "strong"
            if confirmation is ConfirmationEvidence.MANUAL_CONFIRMED
            or confirmation is ConfirmationEvidence.READBACK_VERIFIED
            else None
        ),
        provider_code=receipt.provider_code if receipt is not None else None,
        message_code=receipt.provider_message if receipt is not None else None,
        target_results=(
            [item for item in receipt.target_results] if receipt is not None else []
        ),
    )


@router.post("/writebacks/{writeback_id}/retry", response_model=s.WritebackOperationResponse)
async def retry_writeback(
    writeback_id: str,
    principal: Annotated[Principal, require_roles(ROLE_DISPOSITION_OPERATOR)],
    sync: DispositionSyncDep,
) -> s.WritebackOperationResponse:
    status = await sync.retry_writeback(writeback_id, operator=principal.subject)
    await sync.process_ready_outboxes(limit=1)
    return s.WritebackOperationResponse(
        writeback_id=writeback_id,
        status=status,
        message="re-enqueued",
    )


@router.post("/writebacks/{writeback_id}/resolve", response_model=s.WritebackOperationResponse)
async def resolve_writeback(
    writeback_id: str,
    body: s.ResolveWritebackRequest,
    principal: Annotated[Principal, require_roles(ROLE_ADMIN)],
    sync: DispositionSyncDep,
) -> s.WritebackOperationResponse:
    status = await sync.resolve_writeback(
        writeback_id,
        body.resolution,
        principal=principal.subject,
        comment=body.comment,
        evidence_ref=body.evidence_ref,
    )
    return s.WritebackOperationResponse(
        writeback_id=writeback_id,
        status=status,
        message="resolved",
    )
