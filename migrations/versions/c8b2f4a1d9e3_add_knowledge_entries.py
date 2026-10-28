"""add knowledge_entries table

Revision ID: c8b2f4a1d9e3
Revises: e4eba9cfaa6f
Create Date: 2026-10-19 12:00:00.000000

Creates the standalone knowledge-entry store: one row per explicitly saved note,
owned by a workspace only (no peer/session/collection coupling), deduplicated by
a sha256 content hash and searchable through a pgvector HNSW index.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from migrations.utils import index_exists, table_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "c8b2f4a1d9e3"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
schema = settings.DB.SCHEMA


def upgrade() -> None:
    op.create_table(
        "knowledge_entries",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_name", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column(
            "tags",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_knowledge_entries"),
        sa.ForeignKeyConstraint(
            ["workspace_name"],
            [f"{schema}.workspaces.name"],
            name="fk_knowledge_entries_workspace_name_workspaces",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("length(id) = 21", name="id_length"),
        sa.CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        sa.CheckConstraint("length(title) <= 200", name="title_length"),
        sa.CheckConstraint("length(content) <= 4000", name="content_length"),
        sa.CheckConstraint("length(source) <= 500", name="source_length"),
        sa.UniqueConstraint(
            "workspace_name",
            "content_hash",
            name="uq_knowledge_entries_workspace_name_content_hash",
        ),
        schema=schema,
    )

    op.create_index(
        "ix_knowledge_entries_workspace_created_at",
        "knowledge_entries",
        ["workspace_name", "created_at"],
        schema=schema,
    )

    # HNSW cosine index -- same parameters as the existing document index.
    op.execute(
        f"""
        CREATE INDEX ix_knowledge_entries_embedding_hnsw
        ON {schema}.knowledge_entries
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not table_exists("knowledge_entries", inspector):
        return

    # Drop indexes defensively before the table itself.
    indexes_to_drop = [
        "ix_knowledge_entries_embedding_hnsw",
        "ix_knowledge_entries_workspace_created_at",
    ]

    for index_name in indexes_to_drop:
        if index_exists("knowledge_entries", index_name, inspector):
            op.drop_index(index_name, table_name="knowledge_entries", schema=schema)

    op.drop_table("knowledge_entries", schema=schema)
