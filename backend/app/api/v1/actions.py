"""Action approval / adjudication endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter

from app.api.v1 import schemas as s
from app.api.v1.deps import ActionExecutionDep, ApprovalEngineDep
from app.core.auth import ROLE_ADMIN, ROLE_APPROVER, Principal, require_roles

router = APIRouter(tags=["actions"])


@router.post("/actions/{action_id}/approve", response_model=s.ActionOperationResponse)
async def approve_action(
    action_id: str,
    body: s.ActionApproveRequest,
    principal: Annotated[Principal, require_roles(ROLE_APPROVER)],
    engine: ApprovalEngineDep,
) -> s.ActionOperationResponse:
    await engine.approve(
        action_id,
        principal,
        body.comment,
        body.decision_id,
    )
    await engine.scan_timeouts()
    return s.ActionOperationResponse(
        action_id=action_id,
        status="approved",
        decision_id=body.decision_id,
        message="approved",
    )


@router.post("/actions/{action_id}/reject", response_model=s.ActionOperationResponse)
async def reject_action(
    action_id: str,
    body: s.ActionRejectRequest,
    principal: Annotated[Principal, require_roles(ROLE_APPROVER)],
    engine: ApprovalEngineDep,
) -> s.ActionOperationResponse:
    await engine.reject(
        action_id,
        principal,
        body.comment,
        body.decision_id,
    )
    await engine.scan_timeouts()
    return s.ActionOperationResponse(
        action_id=action_id,
        status="rejected",
        decision_id=body.decision_id,
        message="rejected",
    )


@router.post("/actions/{action_id}/resolve-unknown", response_model=s.ActionOperationResponse)
async def resolve_unknown_action(
    action_id: str,
    body: s.ResolveUnknownRequest,
    principal: Annotated[Principal, require_roles(ROLE_ADMIN)],
    execution: ActionExecutionDep,
) -> s.ActionOperationResponse:
    action = await execution.resolve_unknown(
        action_id,
        body.resolution,
        principal=principal.subject,
        comment=body.comment,
        evidence_ref=body.evidence_ref,
    )
    return s.ActionOperationResponse(
        action_id=action_id,
        status=action.status.value,
        message="resolved",
    )
