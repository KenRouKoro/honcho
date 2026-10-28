"""CRUD tests for the standalone knowledge-entry store.

The embedding provider is the deterministic autouse mock from ``conftest``; no
test here talks to a real Honcho server or a real embedding API.
"""

import asyncio
import hashlib

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models
from src.config import settings
from src.embedding_client import EmbeddingTokenLimitError
from src.exceptions import (
    ConflictException,
    EmbeddingUnavailableError,
    ResourceNotFoundException,
    ValidationException,
)

DIM = settings.EMBEDDING.VECTOR_DIMENSIONS


def _unit(index: int) -> list[float]:
    """One-hot vector: the cosine distance between two distinct ones is 1.0."""
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def _hash_of(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


async def _create(
    db: AsyncSession,
    workspace_name: str,
    *,
    title: str = "Note",
    content: str = "note one",
    source: str = "doc/example.md",
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
) -> tuple[models.KnowledgeEntry, bool]:
    return await crud.create_entry(
        db,
        workspace_name,
        title=title,
        content=content,
        source=source,
        tags=tags if tags is not None else [],
        embedding=embedding if embedding is not None else _unit(0),
    )


async def _second_workspace(db: AsyncSession) -> models.Workspace:
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.commit()
    return workspace


class TestKnowledgeEntryModel:
    """The standalone table keeps the frozen shape."""

    def test_embedding_dimension(self) -> None:
        assert models.KnowledgeEntry.__table__.c.embedding.type.dim == 1536

    def test_constraint_names_match_migration(self) -> None:
        names = {
            constraint.name
            for constraint in models.KnowledgeEntry.__table__.constraints
        }
        assert "uq_knowledge_entries_workspace_name_content_hash" in names
        assert "ck_knowledge_entries_id_length" in names


class TestCreateEntry:
    """Saving an entry: hash, metadata, idempotency, isolation."""

    async def test_creates_entry_with_hash_and_metadata(self, db_session, sample_data):
        workspace, _ = sample_data

        entry, created = await _create(
            db_session,
            workspace.name,
            title="HNSW 参数",
            content="m=16, ef_construction=64",
            source="honcho/src/models.py",
            tags=["honcho", "pgvector"],
        )

        assert created is True
        assert entry.workspace_name == workspace.name
        assert entry.title == "HNSW 参数"
        assert entry.source == "honcho/src/models.py"
        assert entry.tags == ["honcho", "pgvector"]
        assert entry.content_hash == _hash_of("m=16, ef_construction=64")
        assert len(entry.id) == 21
        assert entry.created_at is not None
        assert len(entry.embedding) == DIM

    async def test_duplicate_returns_existing_without_rewriting_metadata(
        self, db_session, sample_data
    ):
        workspace, _ = sample_data
        first, _ = await _create(
            db_session, workspace.name, title="first title", content="same body"
        )

        second, created = await _create(
            db_session,
            workspace.name,
            title="second title",
            content="same body",
            source="another/source.md",
            tags=["new"],
        )

        assert created is False
        assert second.id == first.id
        assert second.title == "first title"
        assert second.source == "doc/example.md"
        assert second.tags == []

    async def test_content_hash_ignores_surrounding_space_and_nul(
        self, db_session, sample_data
    ):
        workspace, _ = sample_data
        first, _ = await _create(db_session, workspace.name, content="  body  ")

        second, created = await _create(db_session, workspace.name, content="body")

        assert created is False
        assert second.id == first.id
        assert first.content == "  body  "
        assert "\x00" not in second.content

    async def test_same_content_in_two_workspaces_is_two_rows(
        self, db_session, sample_data
    ):
        workspace, _ = sample_data
        other = await _second_workspace(db_session)

        first, created_first = await _create(db_session, workspace.name, content="body")
        second, created_second = await _create(db_session, other.name, content="body")

        assert created_first is True
        assert created_second is True
        assert first.id != second.id

    async def test_concurrent_duplicates_create_exactly_one_row(
        self, db_engine: AsyncEngine, sample_data
    ):
        workspace, _ = sample_data
        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

        async with (
            session_factory() as first_session,
            session_factory() as second_session,
        ):
            results = await asyncio.gather(
                _create(first_session, workspace.name, content="racy body"),
                _create(second_session, workspace.name, content="racy body"),
            )

        entries = [result[0] for result in results]
        flags = [result[1] for result in results]
        assert sorted(flags) == [False, True]
        assert entries[0].id == entries[1].id


class TestGetEntry:
    """Fetching one entry: workspace scoping and 404 semantics."""

    async def test_get_entry_returns_row(self, db_session, sample_data):
        workspace, _ = sample_data
        entry, _ = await _create(db_session, workspace.name)

        fetched = await crud.get_entry(db_session, workspace.name, entry.id)

        assert fetched.id == entry.id

    async def test_get_entry_missing_raises_404(self, db_session, sample_data):
        workspace, _ = sample_data

        with pytest.raises(ResourceNotFoundException) as excinfo:
            await crud.get_entry(db_session, workspace.name, "does-not-exist")

        assert "knowledge entry does-not-exist not found" in str(excinfo.value)

    async def test_other_workspace_cannot_read_entry(self, db_session, sample_data):
        workspace, _ = sample_data
        other = await _second_workspace(db_session)
        entry, _ = await _create(db_session, workspace.name)

        with pytest.raises(ResourceNotFoundException):
            await crud.get_entry(db_session, other.name, entry.id)


class TestListEntries:
    """Pagination ordering and workspace-scoped totals."""

    async def test_lists_newest_first_with_total(self, db_session, sample_data):
        workspace, _ = sample_data
        ids = [
            (await _create(db_session, workspace.name, content=f"body {index}"))[0].id
            for index in range(3)
        ]

        items, total = await crud.list_entries(
            db_session, workspace.name, limit=10, offset=0
        )

        assert total == 3
        assert [item.id for item in items] == list(reversed(ids))

    async def test_limit_and_offset_page_through_entries(self, db_session, sample_data):
        workspace, _ = sample_data
        ids = [
            (await _create(db_session, workspace.name, content=f"body {index}"))[0].id
            for index in range(3)
        ]

        page, total = await crud.list_entries(
            db_session, workspace.name, limit=2, offset=2
        )

        assert total == 3
        assert [item.id for item in page] == [ids[0]]

    async def test_total_is_workspace_scoped(self, db_session, sample_data):
        workspace, _ = sample_data
        other = await _second_workspace(db_session)
        await _create(db_session, workspace.name, content="mine")
        await _create(db_session, other.name, content="theirs")

        items, total = await crud.list_entries(
            db_session, workspace.name, limit=10, offset=0
        )

        assert total == 1
        assert [item.content for item in items] == ["mine"]


class TestQueryEntries:
    """Semantic search: ordering, distances, filtering and failure mapping."""

    async def test_orders_by_cosine_distance_with_real_distance(
        self, db_session, sample_data
    ):
        workspace, _ = sample_data
        near, _ = await _create(
            db_session, workspace.name, content="near", embedding=_unit(0)
        )
        far, _ = await _create(
            db_session, workspace.name, content="far", embedding=_unit(1)
        )

        hits = await crud.query_entries(
            db_session,
            workspace.name,
            query="near",
            top_k=5,
            max_distance=None,
            embedding=_unit(0),
        )

        assert [hit[0].id for hit in hits] == [near.id, far.id]
        assert hits[0][1] == pytest.approx(0.0, abs=1e-6)
        assert hits[1][1] == pytest.approx(1.0, abs=1e-6)

    async def test_max_distance_filters_hits(self, db_session, sample_data):
        workspace, _ = sample_data
        near, _ = await _create(
            db_session, workspace.name, content="near", embedding=_unit(0)
        )
        await _create(db_session, workspace.name, content="far", embedding=_unit(1))

        hits = await crud.query_entries(
            db_session,
            workspace.name,
            query="near",
            top_k=5,
            max_distance=0.5,
            embedding=_unit(0),
        )

        assert [hit[0].id for hit in hits] == [near.id]

    async def test_no_match_returns_empty_list(self, db_session, sample_data):
        workspace, _ = sample_data

        hits = await crud.query_entries(
            db_session,
            workspace.name,
            query="anything",
            top_k=5,
            max_distance=None,
            embedding=_unit(0),
        )

        assert hits == []

    async def test_top_k_zero_short_circuits_without_embedding(
        self, db_session, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        await _create(db_session, workspace.name)

        hits = await crud.query_entries(
            db_session, workspace.name, query="near", top_k=0, max_distance=None
        )

        assert hits == []
        mock_openai_embeddings["embed"].assert_not_awaited()

    async def test_embeds_query_when_embedding_omitted(
        self, db_session, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        await _create(db_session, workspace.name)

        await crud.query_entries(
            db_session,
            workspace.name,
            query="some question",
            top_k=5,
            max_distance=None,
        )

        mock_openai_embeddings["embed"].assert_awaited_once_with("some question")

    async def test_query_does_not_cross_workspaces(self, db_session, sample_data):
        workspace, _ = sample_data
        other = await _second_workspace(db_session)
        ours, _ = await _create(
            db_session, workspace.name, content="ours", embedding=_unit(0)
        )
        await _create(db_session, other.name, content="theirs", embedding=_unit(0))
        await _create(db_session, other.name, content="closer", embedding=_unit(1))

        hits = await crud.query_entries(
            db_session,
            workspace.name,
            query="ours",
            top_k=10,
            max_distance=None,
            embedding=_unit(0),
        )

        assert [hit[0].id for hit in hits] == [ours.id]

    async def test_token_limit_maps_to_validation_error(
        self, db_session, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        mock_openai_embeddings["embed"].side_effect = EmbeddingTokenLimitError(
            "too long"
        )

        with pytest.raises(ValidationException) as excinfo:
            await crud.query_entries(
                db_session, workspace.name, query="q", top_k=5, max_distance=None
            )

        assert "exceeds maximum token limit" in str(excinfo.value)

    async def test_provider_failure_maps_to_503(
        self, db_session, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        mock_openai_embeddings["embed"].side_effect = RuntimeError("connection reset")

        with pytest.raises(EmbeddingUnavailableError) as excinfo:
            await crud.query_entries(
                db_session, workspace.name, query="q", top_k=5, max_distance=None
            )

        assert excinfo.value.status_code == 503


class TestUpdateEntry:
    """Partial updates: re-embedding, no-ops and content conflicts."""

    async def test_title_change_reembeds(
        self, db_session, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        entry, _ = await _create(db_session, workspace.name, content="body")

        updated, reembedded = await crud.update_entry(
            db_session,
            workspace.name,
            entry.id,
            crud.KnowledgePatch(title="Renamed"),
        )

        assert reembedded is True
        assert updated.title == "Renamed"
        assert updated.content == "body"
        mock_openai_embeddings["embed"].assert_awaited_once_with(
            crud.embedding_input("Renamed", "body")
        )

    async def test_content_change_updates_hash_and_embedding(
        self, db_session, sample_data
    ):
        workspace, _ = sample_data
        entry, _ = await _create(db_session, workspace.name, content="body")
        original_embedding = list(entry.embedding)

        updated, reembedded = await crud.update_entry(
            db_session,
            workspace.name,
            entry.id,
            crud.KnowledgePatch(content="new body"),
        )

        assert reembedded is True
        assert updated.content == "new body"
        assert updated.content_hash == _hash_of("new body")
        assert list(updated.embedding) != original_embedding

    async def test_source_and_tags_do_not_reembed(
        self, db_session, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        entry, _ = await _create(db_session, workspace.name)

        updated, reembedded = await crud.update_entry(
            db_session,
            workspace.name,
            entry.id,
            crud.KnowledgePatch(source="other/source.md", tags=["a"]),
        )

        assert reembedded is False
        assert updated.source == "other/source.md"
        assert updated.tags == ["a"]
        mock_openai_embeddings["embed"].assert_not_awaited()

    async def test_unchanged_values_write_nothing(
        self, db_session, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        workspace_name = workspace.name
        entry, _ = await _create(db_session, workspace_name, content="body", tags=["a"])
        original_updated_at = entry.updated_at
        entry_id = entry.id

        _, reembedded = await crud.update_entry(
            db_session,
            workspace_name,
            entry_id,
            crud.KnowledgePatch(title="Note", content="body", tags=["a"]),
        )

        db_session.expire_all()
        reloaded = await crud.get_entry(db_session, workspace_name, entry_id)
        assert reembedded is False
        assert reloaded.updated_at == original_updated_at
        mock_openai_embeddings["embed"].assert_not_awaited()

    async def test_content_conflict_reports_existing_id(self, db_session, sample_data):
        workspace, _ = sample_data
        first, _ = await _create(db_session, workspace.name, content="alpha")
        second, _ = await _create(db_session, workspace.name, content="beta")

        with pytest.raises(ConflictException) as excinfo:
            await crud.update_entry(
                db_session,
                workspace.name,
                first.id,
                crud.KnowledgePatch(content="beta"),
            )

        assert f"content already exists: {second.id}" in str(excinfo.value)

    async def test_unknown_entry_raises_404(self, db_session, sample_data):
        workspace, _ = sample_data

        with pytest.raises(ResourceNotFoundException):
            await crud.update_entry(
                db_session, workspace.name, "missing", crud.KnowledgePatch(title="x")
            )


class TestDeleteEntry:
    """Hard deletion stays inside its workspace."""

    async def test_deletes_entry(self, db_session, sample_data):
        workspace, _ = sample_data
        entry, _ = await _create(db_session, workspace.name)

        await crud.delete_entry(db_session, workspace.name, entry.id)

        with pytest.raises(ResourceNotFoundException):
            await crud.get_entry(db_session, workspace.name, entry.id)

    async def test_delete_missing_entry_raises_404(self, db_session, sample_data):
        workspace, _ = sample_data

        with pytest.raises(ResourceNotFoundException) as excinfo:
            await crud.delete_entry(db_session, workspace.name, "missing")

        assert "knowledge entry missing not found" in str(excinfo.value)
