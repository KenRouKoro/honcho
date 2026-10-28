"""CRUD operations for the standalone knowledge entries store.

Knowledge entries live in their own table (``knowledge_entries``) and never
reuse ``Document``/``Collection``/``Conclusion`` rows: they are not observations,
they belong to no peer/session pair and they never feed representations, dreams
or derivations. See ``doc/knowledge_base_plan.md``.

The functions here commit their own writes, mirroring ``crud.document``: request
handlers are thin and never hold a transaction open past the call.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from nanoid import generate as generate_nanoid
from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src import models
from src.config import settings
from src.embedding_client import EmbeddingTokenLimitError, embedding_client
from src.exceptions import (
    ConflictException,
    EmbeddingUnavailableError,
    ResourceNotFoundException,
    ValidationException,
)
from src.telemetry.events import EmbeddingCallPurpose
from src.utils.sanitization import strip_nul
from src.utils.types import embedding_call_purpose


@dataclass(frozen=True)
class KnowledgePatch:
    """Partial-update carrier for ``PATCH /knowledge/{entry_id}``.

    ``None`` means "leave unchanged"; ``tags=[]`` clears the tag list.
    """

    title: str | None = None
    content: str | None = None
    source: str | None = None
    tags: list[str] | None = None


def normalize_content(content: str) -> str:
    """The normalisation used for hashing: NUL bytes removed, then stripped."""
    return strip_nul(content).strip()


def content_hash(content: str) -> str:
    """Lowercase hex sha256 of the normalised content."""
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def embedding_input(title: str, content: str) -> str:
    """Text handed to the embedding provider -- title first, then content."""
    return strip_nul(title).strip() + "\n" + strip_nul(content).strip()


async def embed_knowledge_text(
    text: str,
    workspace_name: str,
    *,
    field: str = "Query",
) -> list[float]:
    """Embed knowledge text, mapping provider failures onto HTTP semantics.

    Token-limit violations are a validation problem (422); every other provider
    or transport failure means the embedding service is unavailable (503). No
    silent fallback: callers never get a zero vector.
    """
    try:
        with embedding_call_purpose(
            EmbeddingCallPurpose.KNOWLEDGE.value,
            workspace_name=workspace_name,
            parent_category="api",
        ):
            return await embedding_client.embed(text)
    except EmbeddingTokenLimitError as exc:
        raise ValidationException(
            f"{field} exceeds maximum token limit of "
            f"{settings.EMBEDDING.MAX_INPUT_TOKENS}."
        ) from exc
    except Exception as exc:
        raise EmbeddingUnavailableError() from exc


async def ensure_workspace_exists(db: AsyncSession, workspace_name: str) -> None:
    """Raise 404 if the workspace does not exist (this router never creates one)."""
    existing = await db.scalar(
        select(models.Workspace.name).where(models.Workspace.name == workspace_name)
    )
    if existing is None:
        raise ResourceNotFoundException(f"Workspace {workspace_name} not found")


async def get_entry(
    db: AsyncSession, workspace_name: str, entry_id: str
) -> models.KnowledgeEntry:
    """Fetch one entry inside a workspace, or raise 404."""
    entry = await db.scalar(
        select(models.KnowledgeEntry).where(
            models.KnowledgeEntry.workspace_name == workspace_name,
            models.KnowledgeEntry.id == entry_id,
        )
    )
    if entry is None:
        raise ResourceNotFoundException(f"knowledge entry {entry_id} not found")
    return entry


async def get_entry_by_content_hash(
    db: AsyncSession, workspace_name: str, hexdigest: str
) -> models.KnowledgeEntry | None:
    """Return the entry already holding this content hash, if any."""
    return await db.scalar(
        select(models.KnowledgeEntry).where(
            models.KnowledgeEntry.workspace_name == workspace_name,
            models.KnowledgeEntry.content_hash == hexdigest,
        )
    )


async def create_entry(
    db: AsyncSession,
    workspace_name: str,
    *,
    title: str,
    content: str,
    source: str,
    tags: Sequence[str],
    embedding: list[float],
) -> tuple[models.KnowledgeEntry, bool]:
    """Insert an entry; idempotent on ``(workspace_name, content_hash)``.

    Concurrency is handled by ``INSERT ... ON CONFLICT DO NOTHING`` -- never by
    "select then insert", which turns two simultaneous identical requests into a
    500. The returned flag is ``True`` only for the request that created the row.

    Returns:
        ``(entry, created)``; on a duplicate the existing entry is returned
        unchanged (its metadata is NOT overwritten -- use ``update_entry``).
    """
    content_hash_value = content_hash(content)
    insert_stmt = (
        pg_insert(models.KnowledgeEntry)
        .values(
            id=generate_nanoid(),
            workspace_name=workspace_name,
            title=strip_nul(title).strip(),
            content=strip_nul(content),
            source=strip_nul(source).strip(),
            tags=list(tags),
            content_hash=content_hash_value,
            embedding=embedding,
        )
        .on_conflict_do_nothing(
            index_elements=[
                models.KnowledgeEntry.workspace_name,
                models.KnowledgeEntry.content_hash,
            ]
        )
        .returning(models.KnowledgeEntry.id)
    )

    # The row we return may be the one a concurrent request just deleted, so a
    # miss on the follow-up SELECT is retried once instead of failing.
    for attempt in range(2):
        inserted_id = await db.scalar(insert_stmt)
        if inserted_id is not None:
            entry = await db.get(models.KnowledgeEntry, inserted_id)
            await db.commit()
            if entry is None:  # pragma: no cover - deleted inside our own txn
                raise ResourceNotFoundException(
                    f"knowledge entry {inserted_id} not found"
                )
            return entry, True

        existing = await get_entry_by_content_hash(
            db, workspace_name, content_hash_value
        )
        if existing is not None:
            return existing, False

        if attempt == 0:
            continue

    raise ConflictException(
        f"content already exists: {content_hash_value}",
    )


async def list_entries(
    db: AsyncSession,
    workspace_name: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[models.KnowledgeEntry], int]:
    """List entries newest first; ``total`` is the workspace-wide count."""
    items = list(
        (
            await db.scalars(
                select(models.KnowledgeEntry)
                .where(models.KnowledgeEntry.workspace_name == workspace_name)
                .order_by(
                    models.KnowledgeEntry.created_at.desc(),
                    models.KnowledgeEntry.id.desc(),
                )
                .limit(limit)
                .offset(offset)
            )
        ).all()
    )
    total = await db.scalar(
        select(func.count())
        .select_from(models.KnowledgeEntry)
        .where(models.KnowledgeEntry.workspace_name == workspace_name)
    )
    return items, int(total or 0)


async def query_entries(
    db: AsyncSession,
    workspace_name: str,
    *,
    query: str,
    top_k: int,
    max_distance: float | None,
    embedding: list[float] | None = None,
) -> list[tuple[models.KnowledgeEntry, float]]:
    """Cosine-similarity search over a workspace's entries.

    Ordered by ascending cosine distance (ties broken by ``id``), optionally
    filtered by ``max_distance``. An empty result is a normal empty list, never
    an error. ``top_k <= 0`` short-circuits without touching the provider.
    """
    if top_k <= 0:
        return []

    if embedding is None:
        embedding = await embed_knowledge_text(query, workspace_name, field="Query")

    distance = models.KnowledgeEntry.embedding.cosine_distance(embedding)
    stmt = select(models.KnowledgeEntry, distance.label("distance")).where(
        models.KnowledgeEntry.workspace_name == workspace_name
    )
    if max_distance is not None:
        stmt = stmt.where(distance <= max_distance)
    stmt = stmt.order_by(distance.asc(), models.KnowledgeEntry.id.asc()).limit(top_k)

    rows = (await db.execute(stmt)).all()
    return [(row[0], float(row[1])) for row in rows]


async def update_entry(
    db: AsyncSession,
    workspace_name: str,
    entry_id: str,
    patch: KnowledgePatch,
) -> tuple[models.KnowledgeEntry, bool]:
    """Apply a partial update to one entry.

    * ``title``/``content`` actually changing means the embedding input changed,
      so the entry is re-embedded (``reembedded=True``).
    * ``source``/``tags`` changed alone do not touch the vector.
    * A patch whose values all equal the stored ones writes nothing at all --
      no UPDATE, no ``updated_at`` bump.

    Returns:
        ``(entry, reembedded)``.

    Raises:
        ResourceNotFoundException: unknown entry in this workspace.
        ConflictException: the new content hash belongs to another entry.
    """
    entry = await get_entry(db, workspace_name, entry_id)

    values: dict[str, Any] = {}
    if patch.title is not None and patch.title != entry.title:
        values["title"] = patch.title
    if patch.content is not None:
        new_hash = content_hash(patch.content)
        if new_hash != entry.content_hash:
            values["content"] = strip_nul(patch.content)
            values["content_hash"] = new_hash
    if patch.source is not None and patch.source != entry.source:
        values["source"] = patch.source
    if patch.tags is not None and list(patch.tags) != list(entry.tags):
        values["tags"] = list(patch.tags)

    if not values:
        return entry, False

    # A content change that collides with a *different* row is a 409 before any
    # write happens; the UPDATE below re-checks it for the concurrent case.
    if "content_hash" in values:
        conflicting_id = await db.scalar(
            select(models.KnowledgeEntry.id).where(
                models.KnowledgeEntry.workspace_name == workspace_name,
                models.KnowledgeEntry.content_hash == values["content_hash"],
                models.KnowledgeEntry.id != entry_id,
            )
        )
        if conflicting_id is not None:
            raise ConflictException(f"content already exists: {conflicting_id}")

    reembedded = "title" in values or "content" in values
    if reembedded:
        values["embedding"] = await embed_knowledge_text(
            embedding_input(
                values.get("title", entry.title),
                values.get("content", entry.content),
            ),
            workspace_name,
            field="Content",
        )

    values["updated_at"] = func.now()
    try:
        await db.execute(
            update(models.KnowledgeEntry)
            .where(
                models.KnowledgeEntry.workspace_name == workspace_name,
                models.KnowledgeEntry.id == entry_id,
            )
            .values(**values)
        )
        await db.commit()
    except IntegrityError as exc:
        # Concurrent writer won the unique constraint: report the same 409 shape
        # instead of leaking an unexpected 500.
        await db.rollback()
        existing_id: str | None = None
        if "content_hash" in values:
            existing_id = await db.scalar(
                select(models.KnowledgeEntry.id).where(
                    models.KnowledgeEntry.workspace_name == workspace_name,
                    models.KnowledgeEntry.content_hash == values["content_hash"],
                )
            )
        if existing_id is None:
            raise
        raise ConflictException(f"content already exists: {existing_id}") from exc

    await db.refresh(entry)
    return entry, reembedded


async def delete_entry(db: AsyncSession, workspace_name: str, entry_id: str) -> None:
    """Hard-delete one entry; 404 if it is not in this workspace."""
    result = await db.execute(
        delete(models.KnowledgeEntry).where(
            models.KnowledgeEntry.workspace_name == workspace_name,
            models.KnowledgeEntry.id == entry_id,
        )
    )
    rowcount = cast(CursorResult[Any], result).rowcount
    if rowcount == 0:
        await db.rollback()
        raise ResourceNotFoundException(f"knowledge entry {entry_id} not found")
    await db.commit()
