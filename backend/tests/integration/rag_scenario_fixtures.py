"""Shared fixtures for ISSUE-047 RAG integration / e2e_basic tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from app.models.agent_io import CollectionStatus, EvidenceOutput, TriageResult
from app.models.entities import (
    AccountEntity,
    DomainEntity,
    EntitySet,
    HostEntity,
    IPEntity,
)
from app.models.enums import EventStatus, EventType, EvidenceSource, FinalVerdict, Severity
from app.models.evidence import Evidence
from app.models.ids import new_evidence_id


class FakeWorkingMemory:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], Any] = {}

    async def read(self, event_id: str, key: str) -> Any:
        return self.values.get((event_id, key))

    async def write(self, event_id: str, key: str, value: Any) -> None:
        self.values[(event_id, key)] = value

    async def append_scratchpad(self, event_id: str, note: str) -> None:
        return None


class FakeEventService:
    """Tracks verdicts the way ``AnalysisOnlyPipeline`` reads them back."""

    def __init__(self) -> None:
        self.risk_updates: list[dict[str, Any]] = []
        self.verdicts: list[FinalVerdict] = []
        self.final_verdict_by_event: dict[str, FinalVerdict] = {}
        self.transitions: list[EventStatus] = []

    async def update_risk_fields(
        self,
        event_id: str,
        *,
        risk_score: int,
        severity: Severity,
        confidence: float,
        operator: str | None = None,
        factor_names: list[str] | None = None,
    ) -> None:
        self.risk_updates.append(
            {
                "event_id": event_id,
                "risk_score": risk_score,
                "severity": severity,
                "confidence": confidence,
            }
        )

    async def set_final_verdict(
        self,
        event_id: str,
        verdict: FinalVerdict,
        *,
        operator: str | None = None,
        context: Any = None,
    ) -> None:
        self.verdicts.append(verdict)
        self.final_verdict_by_event[event_id] = verdict

    async def get_event(self, event_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            event_id=event_id,
            final_verdict=self.final_verdict_by_event.get(event_id, FinalVerdict.NONE),
        )

    async def transition_status(
        self,
        event_id: str,
        target: EventStatus,
        *,
        context: Any = None,
        operator: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.transitions.append(target)


def make_evidence_item(
    *,
    source: EvidenceSource,
    evidence_type: str,
    confidence: float,
    event_id: str,
    description: str,
    raw: dict[str, Any],
    mitre: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=new_evidence_id(),
        event_id=event_id,
        source=source,
        evidence_type=evidence_type,
        description=description,
        confidence=confidence,
        timestamp=datetime(2024, 6, 15, 9, 0, tzinfo=UTC),
        raw_data=raw,
        mitre_technique=mitre,
        is_conflicting=False,
        related_entities=[],
    )


def main_triage() -> TriageResult:
    return TriageResult(
        event_type=EventType.DATA_EXFILTRATION,
        severity=Severity.HIGH,
        need_investigation=True,
        entities=EntitySet(
            accounts=[AccountEntity(entity_id="a1", username="zhangsan")],
            hosts=[
                HostEntity(
                    entity_id="h1",
                    hostname="PC-FIN-023",
                    ip="10.20.30.23",
                )
            ],
            ips=[
                IPEntity(entity_id="i1", address="10.20.30.23", scope="internal"),
                IPEntity(entity_id="i2", address="203.0.113.88", scope="external"),
            ],
            domains=[DomainEntity(entity_id="d1", fqdn="unknown-upload-example.com")],
        ),
        ioc_list=["203.0.113.88"],
        reasoning="insider exfiltration",
    )


def main_evidence(event_id: str) -> EvidenceOutput:
    items = [
        make_evidence_item(
            source=EvidenceSource.ENDPOINT,
            evidence_type="process_create",
            confidence=0.9,
            event_id=event_id,
            description="powershell archive",
            raw={
                "hostname": "PC-FIN-023",
                "account": "zhangsan",
                "process": "powershell.exe",
                "action": "process_create",
            },
            mitre="T1059.001",
        ),
        make_evidence_item(
            source=EvidenceSource.DATA_SECURITY,
            evidence_type="upload",
            confidence=0.88,
            event_id=event_id,
            description="upload finance_report.zip",
            raw={
                "action": "upload",
                "file_name": "finance_report.zip",
                "bytes": 52428800,
            },
            mitre="T1567.002",
        ),
        make_evidence_item(
            source=EvidenceSource.NETWORK_FLOW,
            evidence_type="network_flow",
            confidence=0.85,
            event_id=event_id,
            description="external upload traffic",
            raw={
                "src_ip": "10.20.30.23",
                "dst_ip": "203.0.113.88",
                "bytes_out": 52000000,
            },
            mitre="T1041",
        ),
        make_evidence_item(
            source=EvidenceSource.THREAT_INTEL,
            evidence_type="ip",
            confidence=0.91,
            event_id=event_id,
            description="ti hit",
            raw={
                "indicator": "203.0.113.88",
                "confidence": 0.91,
                "tags": ["exfil", "unknown_infra"],
            },
        ),
    ]
    return EvidenceOutput(
        evidence_list=items,
        success_sources=["endpoint", "data_security", "network_flow", "threat_intel"],
        failed_sources=[],
        overall_confidence=0.86,
        collection_status=CollectionStatus.COMPLETED,
    )
