"""Source, connector and execution-job model tests (ISSUE-002)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.enums import (
    CapabilityState,
    ConnectorCapability,
    DispositionPolicy,
    ExecutionJobStatus,
    SourceObjectKind,
    TargetExecutionStatus,
)
from app.models.execution import ActionExecutionJob, TargetExecutionResult
from app.models.source import (
    SourceConnector,
    SourceIncident,
    SourceReference,
)


def _ref() -> SourceReference:
    return SourceReference(
        source_kind=SourceObjectKind.INCIDENT,
        source_product="mock_xdr",
        source_tenant_id="t1",
        connector_id="conn-1",
        source_object_id="INC-1",
    )


def test_source_incident_defaults_related_refs_empty() -> None:
    inc = SourceIncident(reference=_ref())
    # Relationships are nullable; never inferred, default empty when absent.
    assert inc.related_alert_refs == []
    assert inc.impacted_asset_refs == []


def test_connector_policy_default_is_none_when_unset() -> None:
    """Model must NOT fold an unset policy into NOT_REQUIRED (fail-closed for live)."""
    conn = SourceConnector(
        connector_id="conn-1",
        source_product="manual",
        display_name="Manual intake",
        capabilities={ConnectorCapability.QUERY: CapabilityState.SUPPORTED},
    )
    assert conn.disposition_policy_default is None
    # Secrets stored only as references.
    assert conn.read_credential_ref is None


def test_connector_manual_policy_can_set_explicit_not_required() -> None:
    conn = SourceConnector(
        connector_id="conn-2",
        source_product="manual",
        display_name="Manual intake",
        capabilities={ConnectorCapability.QUERY: CapabilityState.SUPPORTED},
        disposition_policy_default=DispositionPolicy.NOT_REQUIRED,
    )
    assert conn.disposition_policy_default is DispositionPolicy.NOT_REQUIRED


def test_connector_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SourceConnector(
            connector_id="c",
            source_product="p",
            display_name="d",
            secret="oops",  # type: ignore[call-arg]
        )


def test_execution_job_defaults_queued() -> None:
    job = ActionExecutionJob(
        job_id="job-1",
        event_id="evt-1",
        action_id="act-1",
        provider_name="mock",
        idempotency_key="idem-1",
    )
    assert job.status is ExecutionJobStatus.QUEUED
    assert job.attempt == 0
    assert job.provider_job_id is None


def test_execution_job_partial_target_results() -> None:
    job = ActionExecutionJob(
        job_id="job-2",
        event_id="evt-1",
        action_id="act-1",
        provider_name="mock",
        idempotency_key="idem-2",
        status=ExecutionJobStatus.PARTIAL_SUCCESS,
        target_results=[
            TargetExecutionResult(canonical_target="ip:1", status=TargetExecutionStatus.SUCCESS),
            TargetExecutionResult(canonical_target="ip:2", status=TargetExecutionStatus.FAILED),
        ],
    )
    assert job.status is ExecutionJobStatus.PARTIAL_SUCCESS
    assert len(job.target_results) == 2
