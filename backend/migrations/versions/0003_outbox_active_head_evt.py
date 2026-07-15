"""Scope the disposition_outbox active-head uniqueness by event, not action.

Revision ID: 0003_outbox_active_head_evt
Revises: 0002_connector_policy_nullable
Create Date: 2026-07-15 00:00:00.000000+00:00

The original partial unique index on
``(action_id, closure_cycle, intent_kind, logical_slot)`` only prevented a
*single* Action from having two active (non-superseded) EVENT_STATUS_UPDATE
outbox heads. It did NOT prevent two *different* Actions from each claiming
an active terminal-disposition head for the same event/closure_cycle/slot —
e.g. a re-planned Action superseding the original could race with the
original still being "active", producing two live terminal heads for one
event. The uniqueness must be scoped by ``event_id`` (ISSUE-093 §4).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_outbox_active_head_evt"
down_revision: str | None = "0002_connector_policy_nullable"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "uq_disposition_outbox_event_status_active_head"
_ACTIVE_HEAD_WHERE = "superseded_by_disposition_id IS NULL AND intent_kind = 'event_status_update'"


def upgrade() -> None:
    op.drop_index(
        _INDEX_NAME,
        table_name="disposition_outbox",
        postgresql_where=sa.text(_ACTIVE_HEAD_WHERE),
    )
    op.create_index(
        _INDEX_NAME,
        "disposition_outbox",
        ["event_id", "closure_cycle", "intent_kind", "logical_slot"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_HEAD_WHERE),
    )


def downgrade() -> None:
    op.drop_index(
        _INDEX_NAME,
        table_name="disposition_outbox",
        postgresql_where=sa.text(_ACTIVE_HEAD_WHERE),
    )
    op.create_index(
        _INDEX_NAME,
        "disposition_outbox",
        ["action_id", "closure_cycle", "intent_kind", "logical_slot"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_HEAD_WHERE),
    )
