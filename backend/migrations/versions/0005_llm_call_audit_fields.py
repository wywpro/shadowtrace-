"""Add prompt routing and status fields to LLM call audit rows.

Revision ID: 0005_llm_call_audit_fields
Revises: 0004_source_checkpoint_kind
Create Date: 2026-07-17 00:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_llm_call_audit_fields"
down_revision: str | None = "0004_source_checkpoint_kind"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_call_log",
        sa.Column("prompt_key", sa.String(), server_default="unknown", nullable=False),
    )
    op.add_column(
        "llm_call_log",
        sa.Column("status", sa.String(), server_default="legacy", nullable=False),
    )
    op.alter_column("llm_call_log", "prompt_key", server_default=None)
    op.alter_column("llm_call_log", "status", server_default=None)


def downgrade() -> None:
    op.drop_column("llm_call_log", "status")
    op.drop_column("llm_call_log", "prompt_key")
