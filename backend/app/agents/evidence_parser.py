"""EvidenceParser: normalize query ToolResult rows into Evidence (ISSUE-033)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.models.enums import EvidenceSource
from app.models.evidence import Evidence
from app.models.ids import new_evidence_id
from app.models.source import SourceReference
from app.models.tool_meta import ToolResult

# Fixed tool_name → EvidenceSource mapping (ISSUE-033 统一命名).
TOOL_SOURCE_MAP: dict[str, EvidenceSource] = {
    "query_account_login": EvidenceSource.IDENTITY,
    "query_edr_process": EvidenceSource.ENDPOINT,
    "query_file_access": EvidenceSource.DATA_SECURITY,
    "query_network_flow": EvidenceSource.NETWORK_FLOW,
    "query_dns": EvidenceSource.DNS,
    "query_asset_info": EvidenceSource.ASSET,
    "query_threat_intel": EvidenceSource.THREAT_INTEL,
}


def truncate_timestamp_to_second(value: datetime | None) -> datetime | None:
    """Truncate timestamp precision to whole seconds for dedup / timeline sort."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.replace(microsecond=0)


def parse_timestamp(raw: Any) -> datetime | None:
    """Best-effort parse of fixture / ToolResult timestamp fields."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return truncate_timestamp_to_second(raw)
    if isinstance(raw, (int, float)):
        return truncate_timestamp_to_second(datetime.fromtimestamp(float(raw), tz=UTC))
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return truncate_timestamp_to_second(datetime.fromisoformat(text))
    except ValueError:
        return None


class EvidenceParser:
    """Convert one query tool result into a list of Evidence objects."""

    def parse(
        self,
        tool_name: str,
        tool_result: ToolResult | dict[str, Any],
        *,
        event_id: str,
    ) -> list[Evidence]:
        source = TOOL_SOURCE_MAP.get(tool_name)
        if source is None:
            raise ValueError(f"unsupported evidence query tool: {tool_name!r}")

        result = (
            tool_result
            if isinstance(tool_result, ToolResult)
            else ToolResult.model_validate(tool_result)
        )
        data = result.data if isinstance(result.data, dict) else {}
        records = data.get("records") or []
        if not isinstance(records, list) or not records:
            return []

        references = self._index_source_refs(data.get("source_references") or [])
        default_confidence = float(result.confidence) if result.confidence is not None else 0.7

        evidence_list: list[Evidence] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            evidence_list.append(
                self._record_to_evidence(
                    tool_name=tool_name,
                    source=source,
                    event_id=event_id,
                    record=record,
                    references=references,
                    default_confidence=default_confidence,
                )
            )
        return evidence_list

    def _record_to_evidence(
        self,
        *,
        tool_name: str,
        source: EvidenceSource,
        event_id: str,
        record: dict[str, Any],
        references: dict[str, SourceReference],
        default_confidence: float,
    ) -> Evidence:
        timestamp = parse_timestamp(record.get("logged_at") or record.get("timestamp"))
        evidence_type = self._evidence_type(tool_name, record)
        description = self._description(tool_name, record, timestamp)
        confidence = self._confidence(record, default_confidence)
        related = self._related_entities(tool_name, record)
        source_ref = self._match_source_ref(record, references)

        return Evidence(
            evidence_id=new_evidence_id(),
            event_id=event_id,
            source=source,
            evidence_type=evidence_type,
            description=description,
            confidence=confidence,
            timestamp=timestamp,
            related_entities=related,
            source_ref=source_ref,
            raw_data=dict(record),
            mitre_technique=None,
            is_conflicting=bool(record.get("is_conflict_seed")),
        )

    @staticmethod
    def _index_source_refs(raw_refs: list[Any]) -> dict[str, SourceReference]:
        indexed: dict[str, SourceReference] = {}
        for item in raw_refs:
            try:
                ref = (
                    item
                    if isinstance(item, SourceReference)
                    else SourceReference.model_validate(item)
                )
            except Exception:
                continue
            indexed[ref.source_object_id] = ref
        return indexed

    @staticmethod
    def _match_source_ref(
        record: dict[str, Any],
        references: dict[str, SourceReference],
    ) -> SourceReference | None:
        record_id = record.get("record_id")
        if record_id is None:
            return None
        return references.get(str(record_id))

    @staticmethod
    def _confidence(record: dict[str, Any], default: float) -> float:
        raw = record.get("confidence")
        if raw is None:
            return max(0.0, min(1.0, default))
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return max(0.0, min(1.0, default))

    @staticmethod
    def _evidence_type(tool_name: str, record: dict[str, Any]) -> str:
        for key in ("event_type", "action", "indicator_type", "qtype", "agent_status"):
            value = record.get(key)
            if value:
                return str(value)
        return {
            "query_account_login": "login",
            "query_edr_process": "process",
            "query_file_access": "file_access",
            "query_network_flow": "network_flow",
            "query_dns": "dns_query",
            "query_asset_info": "asset_info",
            "query_threat_intel": "threat_intel",
        }.get(tool_name, "unknown")

    @staticmethod
    def _related_entities(tool_name: str, record: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        field_map = {
            "query_account_login": ("account", "src_ip"),
            "query_edr_process": ("hostname", "account", "process"),
            "query_file_access": ("account", "hostname", "file_name"),
            "query_network_flow": ("src_ip", "dst_ip", "hostname", "domain"),
            "query_dns": ("query", "answer", "hostname"),
            "query_asset_info": ("hostname", "ip", "owner"),
            "query_threat_intel": ("indicator",),
        }
        for key in field_map.get(tool_name, ()):
            value = record.get(key)
            if value is not None and str(value).strip():
                candidates.append(str(value))
        seen: set[str] = set()
        ordered: list[str] = []
        for item in candidates:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    @staticmethod
    def _format_time(timestamp: datetime | None) -> str:
        if timestamp is None:
            return "未知时间"
        return timestamp.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _description(
        self,
        tool_name: str,
        record: dict[str, Any],
        timestamp: datetime | None,
    ) -> str:
        time_text = self._format_time(timestamp)
        if tool_name == "query_account_login":
            account = record.get("account") or "未知账号"
            ip = record.get("src_ip") or "未知IP"
            result = record.get("result")
            if result and result != "success":
                return f"账号 {account} 于 {time_text} 登录查询结果为 {result}（来源IP {ip}）"
            return f"账号 {account} 于 {time_text} 从 {ip} 登录"
        if tool_name == "query_edr_process":
            host = record.get("hostname") or "未知主机"
            process = record.get("process") or record.get("file_name") or "未知进程"
            action = record.get("action") or "process_event"
            return f"主机 {host} 于 {time_text} 发生 {action}：{process}"
        if tool_name == "query_file_access":
            account = record.get("account") or "未知账号"
            file_name = record.get("file_name") or "未知文件"
            action = record.get("action") or "access"
            return f"账号 {account} 于 {time_text} 对文件 {file_name} 执行 {action}"
        if tool_name == "query_network_flow":
            src_ip = record.get("src_ip") or "未知源IP"
            dst_ip = record.get("dst_ip") or "未知目的IP"
            dst_port = record.get("dst_port")
            port_text = f":{dst_port}" if dst_port is not None else ""
            return f"主机 {src_ip} 于 {time_text} 连接 {dst_ip}{port_text}"
        if tool_name == "query_dns":
            host = record.get("hostname") or "未知主机"
            query = record.get("query") or "未知域名"
            answer = record.get("answer")
            if answer:
                return f"主机 {host} 于 {time_text} 解析域名 {query} → {answer}"
            return f"主机 {host} 于 {time_text} 解析域名 {query}"
        if tool_name == "query_asset_info":
            host = record.get("hostname") or "未知主机"
            ip = record.get("ip") or "未知IP"
            status = record.get("agent_status") or "unknown"
            return f"资产 {host}（{ip}）Agent 状态为 {status}"
        if tool_name == "query_threat_intel":
            indicator = record.get("indicator") or "未知指标"
            conf = record.get("confidence")
            conf_text = f"{float(conf):.2f}" if conf is not None else "未知"
            return f"威胁情报命中指标 {indicator}，置信度 {conf_text}"
        return f"采集到来自 {tool_name} 的证据记录"
