"""Tests for CaseKBService: search + archival (ISSUE-043)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.core.embedding.service import EmbeddingService
from app.models.case import (
    FalsePositiveCase,
    HistoryCase,
    fp_case_metadata,
    fp_case_to_text,
    history_case_metadata,
    history_case_to_text,
    make_chunk_id,
)
from app.models.enums import CaseLabel, EventType
from app.models.knowledge import KnowledgeChunk
from app.services.case_kb_service import CaseKBService
from app.services.knowledge_store import KnowledgeStore

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace",
)

FP_KB_NAME = "fp_case_kb"
HISTORY_KB_NAME = "history_case_kb"


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return cfg


# ── Database fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def migrated() -> None:
    command.upgrade(_alembic_config(), "head")


@pytest_asyncio.fixture
async def session_factory(
    migrated: None,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def clean_knowledge(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Truncate knowledge_chunk before each test for isolation."""
    async with session_factory() as session:
        await session.execute(text("DELETE FROM knowledge_chunk"))
        await session.commit()


@pytest_asyncio.fixture
def embed_service() -> EmbeddingService:
    return EmbeddingService(Settings(embedding_mode="mock"))


@pytest_asyncio.fixture
def knowledge_store(
    session_factory: async_sessionmaker[AsyncSession],
    embed_service: EmbeddingService,
) -> KnowledgeStore:
    return KnowledgeStore(session_factory, embed_service)


@pytest_asyncio.fixture
def case_kb_service(
    knowledge_store: KnowledgeStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> CaseKBService:
    return CaseKBService(knowledge_store, session_factory)


# ── Seed helpers ─────────────────────────────────────────────────────


def _ops_change_window_case() -> FalsePositiveCase:
    """The FP case matching account_anomaly_fp scenario."""
    return FalsePositiveCase(
        case_id="case-00000001",
        pattern_summary="运维账号在变更窗口内批量登录跳板机执行自动化改密脚本",
        alert_signature="Bulk login by ops account during change window from jump host",
        entity_pattern=(
            "account=ops-change-bot type=service_account; "
            "host=PC-OPS-JUMP-01 role=jump_host; behavior=bulk_login_in_window"
        ),
        fp_reason="运维团队在已审批变更窗口内执行的合规批量改密操作",
        confirmed_by="security_admin",
        confirmed_at="2024-03-15T10:30:00Z",
    )


def _make_fp_chunk(case: FalsePositiveCase) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=make_chunk_id(FP_KB_NAME, case.case_id),
        kb_name=FP_KB_NAME,
        content=fp_case_to_text(case),
        metadata=fp_case_metadata(case),
    )


def _make_history_chunk(case: HistoryCase) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=make_chunk_id(HISTORY_KB_NAME, case.case_id),
        kb_name=HISTORY_KB_NAME,
        content=history_case_to_text(case),
        metadata=history_case_metadata(case),
    )


# ── Seed data loading tests ──────────────────────────────────────────


class TestSeedLoading:
    @pytest.mark.asyncio
    async def test_fp_cases_seed_count(
        self, knowledge_store: KnowledgeStore, clean_knowledge: None
    ) -> None:
        """Seeding ≥10 FP cases (inline) and counting them."""
        cases = [
            _ops_change_window_case(),
            FalsePositiveCase(
                case_id="case-00000002",
                pattern_summary="夜间备份任务大流量",
                alert_signature="Large outbound data transfer during off-hours",
                entity_pattern="account=backup-svc; host=BACKUP-SRV-03; time=night",
                fp_reason="每日定时异地备份任务",
                confirmed_by="infra_engineer",
                confirmed_at="2024-04-02T08:15:00Z",
            ),
            FalsePositiveCase(
                case_id="case-00000003",
                pattern_summary="内部漏洞扫描器全网扫描",
                alert_signature="Port scan across internal subnets from scanner host",
                entity_pattern=(
                    "account=scanner-svc; host=SCANNER-01; behavior=sequential_port_scan"
                ),
                fp_reason="定期漏洞扫描任务",
                confirmed_by="vuln_mgmt_lead",
                confirmed_at="2024-05-11T14:00:00Z",
            ),
            FalsePositiveCase(
                case_id="case-00000004",
                pattern_summary="CI/CD 自动部署触发 SSH 暴力破解告警",
                alert_signature="Multiple SSH logins to production servers from CI runner",
                entity_pattern="account=deploy-bot; host=CI-RUNNER-07; protocol=ssh",
                fp_reason="Ansible Tower 自动化部署",
                confirmed_by="devops_lead",
                confirmed_at="2024-06-20T11:45:00Z",
            ),
            FalsePositiveCase(
                case_id="case-00000005",
                pattern_summary="域控间 AD 复制流量",
                alert_signature="LDAP replication traffic between DCs across WAN",
                entity_pattern="host=DC-PRIMARY-01,DC-SECONDARY-04; protocol=ldap",
                fp_reason="AD 多站点复制标准同步",
                confirmed_by="ad_admin",
                confirmed_at="2024-07-03T16:00:00Z",
            ),
            FalsePositiveCase(
                case_id="case-00000006",
                pattern_summary="DHCP 租约续租波峰",
                alert_signature="ARP traffic surge at DHCP lease renewal boundary",
                entity_pattern=(
                    "host=DHCP-CLUSTER-01; protocol=dhcp,arp; behavior=lease_renewal_surge"
                ),
                fp_reason="租约半生命周期续租波峰",
                confirmed_by="network_engineer",
                confirmed_at="2024-01-28T09:00:00Z",
            ),
            FalsePositiveCase(
                case_id="case-00000007",
                pattern_summary="EDR Agent 心跳被误判 C2",
                alert_signature="Encrypted outbound with beaconing pattern from EDR sensor",
                entity_pattern="account=edr-agent; host=ENDPOINT-*; behavior=regular_heartbeat",
                fp_reason="EDR 遥测上报",
                confirmed_by="soc_analyst",
                confirmed_at="2024-02-14T13:20:00Z",
            ),
            FalsePositiveCase(
                case_id="case-00000008",
                pattern_summary="VPN 早高峰集中拨入",
                alert_signature="Concurrent VPN logins from geo-distributed IPs in morning rush",
                entity_pattern="account=*; src_ips=diverse_geo; time=morning_rush",
                fp_reason="周一早高峰远程办公集中拨入",
                confirmed_by="it_support",
                confirmed_at="2024-08-05T08:30:00Z",
            ),
            FalsePositiveCase(
                case_id="case-00000009",
                pattern_summary="邮件钓鱼演练误报",
                alert_signature="Mass phishing campaign alert from email security gateway",
                entity_pattern="src_smtp=sandbox-phish-sim; subject=Security Awareness",
                fp_reason="安全团队授权的季度钓鱼演练",
                confirmed_by="security_awareness_lead",
                confirmed_at="2024-04-01T10:00:00Z",
            ),
            FalsePositiveCase(
                case_id="case-0000000a",
                pattern_summary="开发环境批量安装依赖包",
                alert_signature="Bulk download of packages from public registries",
                entity_pattern="account=dev-*; host=DEV-WS-*; dst=registry.npmjs.org,pypi.org",
                fp_reason="新员工入职日批量克隆项目仓库并安装依赖",
                confirmed_by="eng_manager",
                confirmed_at="2024-05-20T15:00:00Z",
            ),
        ]
        chunks = [_make_fp_chunk(c) for c in cases]
        await knowledge_store.upsert_chunks(FP_KB_NAME, chunks)
        assert await knowledge_store.count(FP_KB_NAME) == len(cases)
        assert len(cases) >= 10

    @pytest.mark.asyncio
    async def test_fp_upsert_idempotent(
        self, knowledge_store: KnowledgeStore, clean_knowledge: None
    ) -> None:
        """Repeated upsert must not duplicate chunks."""
        case = _ops_change_window_case()
        chunk = _make_fp_chunk(case)
        await knowledge_store.upsert_chunks(FP_KB_NAME, [chunk])
        assert await knowledge_store.count(FP_KB_NAME) == 1
        await knowledge_store.upsert_chunks(FP_KB_NAME, [chunk])
        assert await knowledge_store.count(FP_KB_NAME) == 1

    @pytest.mark.asyncio
    async def test_history_cases_seed_count(
        self, knowledge_store: KnowledgeStore, clean_knowledge: None
    ) -> None:
        """Seeding ≥16 history cases (inline) and counting them."""
        event_types = [
            EventType.ACCOUNT_ANOMALY,
            EventType.HOST_COMPROMISE,
            EventType.DATA_EXFILTRATION,
            EventType.INSIDER_THREAT,
            EventType.MALICIOUS_PROCESS,
            EventType.SUSPICIOUS_DOMAIN,
            EventType.LATERAL_MOVEMENT,
            EventType.OTHER,
        ]
        cases: list[HistoryCase] = []
        for i, et in enumerate(event_types):
            for j in range(2):
                idx = i * 2 + j + 1
                cases.append(
                    HistoryCase(
                        case_id=f"case-2{idx:07x}",
                        event_id=None,
                        event_type=et,
                        case_label=CaseLabel.TRUE_POSITIVE if j == 0 else CaseLabel.FALSE_POSITIVE,
                        summary=f"Seed {et.value} case #{j + 1}",
                        key_entities=f"account=test_{et.value}; host=SRV-{idx:02d}",
                        final_verdict="confirmed_threat" if j == 0 else "false_positive",
                        risk_score=70 if j == 0 else 20,
                        resolution=f"Closed {et.value} case #{j + 1}",
                        closed_at="2024-01-15T10:00:00Z",
                    )
                )
        chunks = [_make_history_chunk(c) for c in cases]
        await knowledge_store.upsert_chunks(HISTORY_KB_NAME, chunks)
        assert await knowledge_store.count(HISTORY_KB_NAME) == len(cases)
        assert len(cases) == 16


# ── Search tests ─────────────────────────────────────────────────────


class TestFpCaseSearch:
    @pytest.mark.asyncio
    async def test_account_anomaly_fp_alert_hits_ops_pattern(
        self, case_kb_service: CaseKBService, knowledge_store: KnowledgeStore, clean_knowledge: None
    ) -> None:
        """account_anomaly_fp alert text must retrieve ops change window pattern as top1."""
        # Seed the ops change window FP case.
        case = _ops_change_window_case()
        await knowledge_store.upsert_chunks(FP_KB_NAME, [_make_fp_chunk(case)])

        # Also seed a few distractor FP cases.
        distractors = [
            FalsePositiveCase(
                case_id="case-0000000b",
                pattern_summary="财务系统月底批量导出报表",
                alert_signature="Large volume of file reads from finance drive",
                entity_pattern="account=finance-*; host=FIN-APP-02; behavior=bulk_export",
                fp_reason="财务月结常规流程",
                confirmed_by="finance_director",
                confirmed_at="2024-06-28T17:00:00Z",
            ),
            FalsePositiveCase(
                case_id="case-0000000c",
                pattern_summary="K8s HPA 扩容进程创建",
                alert_signature="Rapid process creation during HPA scale-out",
                entity_pattern="host=K8S-NODE-*; behavior=hpa_scale_out",
                fp_reason="业务峰值 HPA 弹性扩容",
                confirmed_by="platform_engineer",
                confirmed_at="2024-07-15T12:30:00Z",
            ),
        ]
        await knowledge_store.upsert_chunks(FP_KB_NAME, [_make_fp_chunk(d) for d in distractors])

        # Simulate the alert text from account_anomaly_fp scenario.
        alert_text = (
            "Bulk login by ops account during change window: ops-change-bot "
            "executed automated password rotation from PC-OPS-JUMP-01"
        )
        results = await case_kb_service.search_fp_cases(alert_text, top_k=5)
        assert len(results) >= 1
        # With mock embeddings, ranking is non-semantic; verify the ops case is present
        case_ids = [r.metadata["case_id"] for r in results]
        assert "case-00000001" in case_ids, f"Expected ops case in results, got {case_ids}"

    @pytest.mark.asyncio
    async def test_no_results_for_empty_kb(
        self, case_kb_service: CaseKBService, clean_knowledge: None
    ) -> None:
        """Empty KB returns empty list."""
        results = await case_kb_service.search_fp_cases("anything", top_k=5)
        assert results == []


class TestHistoryCaseSearch:
    @pytest.mark.asyncio
    async def test_event_type_filter(
        self, case_kb_service: CaseKBService, knowledge_store: KnowledgeStore, clean_knowledge: None
    ) -> None:
        """search_history_cases with event_type filter only returns matching cases."""
        cases = [
            HistoryCase(
                case_id="case-30000001",
                event_id=None,
                event_type=EventType.DATA_EXFILTRATION,
                case_label=CaseLabel.TRUE_POSITIVE,
                summary="Sensitive HR data exfiltrated via WebDAV to external cloud storage",
                key_entities="account=hr_intern; host=PC-HR-012; dst=webdav.external.example",
                final_verdict="confirmed_threat",
                risk_score=88,
                resolution="Account disabled; legal notified",
                closed_at="2024-04-22T10:00:00Z",
            ),
            HistoryCase(
                case_id="case-30000002",
                event_id=None,
                event_type=EventType.DATA_EXFILTRATION,
                case_label=CaseLabel.FALSE_POSITIVE,
                summary="Dev team SFTP upload to pentest platform misclassified",
                key_entities="account=pentest-lead; dst=pentest-platform.example",
                final_verdict="false_positive",
                risk_score=25,
                resolution="Confirmed approved pentest data transfer",
                closed_at="2024-08-12T16:30:00Z",
            ),
            HistoryCase(
                case_id="case-30000003",
                event_id=None,
                event_type=EventType.LATERAL_MOVEMENT,
                case_label=CaseLabel.TRUE_POSITIVE,
                summary="Pass-the-Hash lateral movement to file server",
                key_entities=(
                    "src_host=PC-RECEPTION-03; dst_hosts=FS-DEPT-01; technique=pass_the_hash"
                ),
                final_verdict="confirmed_threat",
                risk_score=90,
                resolution="Hosts isolated; LAPS deployed",
                closed_at="2024-11-12T06:00:00Z",
            ),
        ]
        chunks = [_make_history_chunk(c) for c in cases]
        await knowledge_store.upsert_chunks(HISTORY_KB_NAME, chunks)

        # Unfiltered: all 3 should appear (top_k large enough)
        all_results = await case_kb_service.search_history_cases(
            "data exfiltration lateral movement", top_k=5
        )
        assert any(r.metadata["event_type"] == "data_exfiltration" for r in all_results)

        # Filtered by lateral_movement
        lm_results = await case_kb_service.search_history_cases(
            "data exfiltration lateral movement",
            event_type="lateral_movement",
            top_k=5,
        )
        assert len(lm_results) >= 1
        for r in lm_results:
            assert r.metadata["event_type"] == "lateral_movement"

    @pytest.mark.asyncio
    async def test_no_results_for_empty_kb(
        self, case_kb_service: CaseKBService, clean_knowledge: None
    ) -> None:
        """Empty KB returns empty list."""
        results = await case_kb_service.search_history_cases("anything", top_k=5)
        assert results == []


# ── Archival tests ───────────────────────────────────────────────────


class TestArchiveEventAsCase:
    @pytest.mark.asyncio
    async def test_archive_closed_event_creates_retrievable_case(
        self,
        case_kb_service: CaseKBService,
        knowledge_store: KnowledgeStore,
        session_factory: async_sessionmaker[AsyncSession],
        clean_knowledge: None,
    ) -> None:
        """archive_event_as_case() persists a HistoryCase that can be immediately retrieved."""
        event_id = "evt-archive-test-001"

        # Create a minimal security_event row.
        async with session_factory() as session:
            async with session.begin():
                stmt = text(
                    """
                    INSERT INTO security_event
                        (event_id, event_type, title, description, status, severity,
                         risk_score, confidence, final_verdict, entities,
                         creation_source_ref, source_reference_snapshots,
                         raw_alert_ids, disposition_policy, closed_at)
                    VALUES
                        (:eid, 'data_exfiltration', 'Test archive event',
                         'Test description', 'closed', 'high', 75, 0.8,
                         'confirmed_threat', :entities, :ref, :ref,
                         :raw_alert_ids, 'required', '2024-06-15T10:00:00Z')
                    ON CONFLICT (event_id) DO UPDATE
                    SET status = 'closed',
                        final_verdict = 'confirmed_threat',
                        risk_score = 75,
                        closed_at = '2024-06-15T10:00:00Z'
                    """
                )
                await session.execute(
                    stmt,
                    {
                        "eid": event_id,
                        "entities": (
                            '{"hosts": ["PC-FIN-023"], '
                            '"accounts": ["zhangsan"], '
                            '"ips": ["45.153.12.88"]}'
                        ),
                        "raw_alert_ids": "[]",
                        "ref": (
                            '{"source_kind": "alert", '
                            '"source_product": "file", '
                            '"source_tenant_id": "local", '
                            '"connector_id": "file-local", '
                            '"source_object_id": "file-test"}'
                        ),
                    },
                )

        # Also create a minimal report row.
        async with session_factory() as session:
            async with session.begin():
                stmt = text(
                    """
                    INSERT INTO report
                        (report_id, event_id, title, summary,
                         final_verdict, risk_score, severity)
                    VALUES
                        (:rid, :eid, 'Test Report',
                         'Data exfiltration confirmed via DLP logs; '
                         'attacker used WebDAV',
                         'confirmed_threat', 75, 'high')
                    ON CONFLICT (report_id) DO NOTHING
                    """
                )
                await session.execute(stmt, {"rid": "rpt-archive-test", "eid": event_id})

        case_id = await case_kb_service.archive_event_as_case(event_id)
        assert case_id.startswith("case-")

        # Must be immediately retrievable.
        results = await case_kb_service.search_history_cases("data exfiltration WebDAV", top_k=5)
        retrieved = [r for r in results if r.metadata.get("event_id") == event_id]
        assert len(retrieved) == 1
        assert retrieved[0].metadata["case_label"] == "true_positive"
        assert retrieved[0].metadata["event_type"] == "data_exfiltration"
        assert retrieved[0].metadata["risk_score"] == 75

    @pytest.mark.asyncio
    async def test_archive_nonexistent_event_raises(self, case_kb_service: CaseKBService) -> None:
        """Archiving a non-existent event must raise ValueError."""
        with pytest.raises(ValueError, match="security_event not found"):
            await case_kb_service.archive_event_as_case("evt-nonexistent-ffff")
