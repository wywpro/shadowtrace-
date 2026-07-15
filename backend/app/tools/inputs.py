"""Per-tool input schemas for baseline tools (ISSUE-006).

Query tools share ``TimeRange``. Each tool's primary key parameter(s) match
intro §4.5 / ISSUE-006 统一命名. These models are exported to
``contracts/schemas/tools/{tool_name}.json``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The complete, closed set of tool target kinds. ``event`` is read-only and is
# used by check_new_alerts; response subclasses still narrow their own Literal
# so an event target can never reach a side-effect Provider.
TargetType = Literal["ip", "domain", "host", "file", "process", "account", "event"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimeRange(_Strict):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _ordered(self) -> TimeRange:
        if self.end < self.start:
            raise ValueError("time_range.end must be >= time_range.start")
        return self


# --------------------------------------------------------------------------- #
# Query inputs
# --------------------------------------------------------------------------- #


class QueryAccountLoginInput(_Strict):
    account: str
    time_range: TimeRange


class QueryEdrProcessInput(_Strict):
    host_id: str
    time_range: TimeRange


class QueryFileAccessInput(_Strict):
    account: str
    time_range: TimeRange


class QueryNetworkFlowInput(_Strict):
    time_range: TimeRange
    src_ip: str | None = None
    dst_ip: str | None = None

    @model_validator(mode="after")
    def _require_ip(self) -> QueryNetworkFlowInput:
        if not self.src_ip and not self.dst_ip:
            raise ValueError("query_network_flow requires src_ip or dst_ip")
        return self


class QueryDnsInput(_Strict):
    domain: str
    time_range: TimeRange


class QueryAssetInfoInput(_Strict):
    time_range: TimeRange | None = None
    ip: str | None = None
    hostname: str | None = None

    @model_validator(mode="after")
    def _require_key(self) -> QueryAssetInfoInput:
        if not self.ip and not self.hostname:
            raise ValueError("query_asset_info requires ip or hostname")
        return self


class QueryVulnInfoInput(_Strict):
    time_range: TimeRange | None = None
    ip: str | None = None
    hostname: str | None = None

    @model_validator(mode="after")
    def _require_key(self) -> QueryVulnInfoInput:
        if not self.ip and not self.hostname:
            raise ValueError("query_vuln_info requires ip or hostname")
        return self


class QueryThreatIntelInput(_Strict):
    indicator: str
    time_range: TimeRange | None = None


class QueryHistoryCasesInput(_Strict):
    pattern_description: str
    time_range: TimeRange | None = None


# --------------------------------------------------------------------------- #
# Response / verification / rollback inputs (minimal stable contracts)
# --------------------------------------------------------------------------- #


class TargetInput(_Strict):
    """Common target envelope for side-effect tools."""

    target_type: TargetType
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class BlockIpInput(TargetInput):
    target_type: Literal["ip"] = "ip"


class BlockDomainInput(TargetInput):
    target_type: Literal["domain"] = "domain"


class IsolateHostInput(TargetInput):
    target_type: Literal["host"] = "host"


class QuarantineFileInput(TargetInput):
    target_type: Literal["file"] = "file"


class BlockProcessInput(TargetInput):
    target_type: Literal["process"] = "process"


class ScanHostForVirusInput(TargetInput):
    target_type: Literal["host"] = "host"


class DisableAccountInput(TargetInput):
    target_type: Literal["account"] = "account"


class ForceLogoutInput(TargetInput):
    target_type: Literal["account"] = "account"


class ResetPasswordInput(TargetInput):
    target_type: Literal["account"] = "account"


class RevokeTokenInput(TargetInput):
    target_type: Literal["account"] = "account"


class CreateTicketInput(_Strict):
    title: str
    description: str = ""
    severity: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class NotifySecurityTeamInput(_Strict):
    message: str
    channels: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class UpdateSourceEventDispositionInput(_Strict):
    """Virtual disposition-only input (executed by DispositionAdapter, not ToolProvider)."""

    source_record_id: str | None = None
    # Actual controlled disposition is derived post-effect by EventDispositionService.
    approved_terminal_dispositions: list[str] = Field(default_factory=list)


class CheckStatusInput(TargetInput):
    """Verification tool input — same target shape as the paired response tool."""


class UnblockIpInput(BlockIpInput):
    pass


class UnblockDomainInput(BlockDomainInput):
    pass


class RestoreAccountInput(DisableAccountInput):
    pass


class CancelHostIsolationInput(IsolateHostInput):
    pass


class RestoreFileInput(QuarantineFileInput):
    pass


class CloseFalsePositiveTicketInput(_Strict):
    ticket_id: str
    reason: str = ""


# tool_name -> input model (used for schema export + validation).
TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "query_account_login": QueryAccountLoginInput,
    "query_edr_process": QueryEdrProcessInput,
    "query_file_access": QueryFileAccessInput,
    "query_network_flow": QueryNetworkFlowInput,
    "query_dns": QueryDnsInput,
    "query_asset_info": QueryAssetInfoInput,
    "query_vuln_info": QueryVulnInfoInput,
    "query_threat_intel": QueryThreatIntelInput,
    "query_history_cases": QueryHistoryCasesInput,
    "block_ip": BlockIpInput,
    "block_domain": BlockDomainInput,
    "isolate_host": IsolateHostInput,
    "quarantine_file": QuarantineFileInput,
    "block_process": BlockProcessInput,
    "scan_host_for_virus": ScanHostForVirusInput,
    "disable_account": DisableAccountInput,
    "force_logout": ForceLogoutInput,
    "reset_password": ResetPasswordInput,
    "revoke_token": RevokeTokenInput,
    "create_ticket": CreateTicketInput,
    "notify_security_team": NotifySecurityTeamInput,
    "update_source_event_disposition": UpdateSourceEventDispositionInput,
    "check_ip_block_status": CheckStatusInput,
    "check_domain_block_status": CheckStatusInput,
    "check_host_isolation_status": CheckStatusInput,
    "check_file_quarantine_status": CheckStatusInput,
    "check_process_block_status": CheckStatusInput,
    "check_virus_scan_status": CheckStatusInput,
    "check_account_status": CheckStatusInput,
    "check_new_alerts": CheckStatusInput,
    "check_traffic_drop": CheckStatusInput,
    "unblock_ip": UnblockIpInput,
    "unblock_domain": UnblockDomainInput,
    "restore_account": RestoreAccountInput,
    "cancel_host_isolation": CancelHostIsolationInput,
    "restore_file": RestoreFileInput,
    "close_false_positive_ticket": CloseFalsePositiveTicketInput,
}
