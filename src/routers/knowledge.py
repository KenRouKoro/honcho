"""HTTP routes for the standalone knowledge-entry store.

``/v3/workspaces/{workspace_id}/knowledge`` -- explicitly saved notes with
source metadata plus semantic search. These routes never touch peers, sessions,
collections or conclusions; they only read/write ``knowledge_entries``.

Failure semantics are deliberate: an empty search result is a 200 with
``hits: []``, while an unreachable embedding provider is a 503 (never a silent
empty list), and a token-limit violation is a 422.
"""

import logging

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db, read_db
from src.exceptions import ValidationException
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


@router.post(
    "",
    response_model=schemas.KnowledgeCreateResult,
    status_code=201,
)
async def create_entry(
    response: Response,
    workspace_id: str = Path(...),
    body: schemas.KnowledgeEntryCreate = Body(
        ...,
        description="Knowledge entry to save",
    ),
    db: AsyncSession = db,
) -> schemas.KnowledgeCreateResult:
    """Save one knowledge entry explicitly.

    Returns 201 when this request created the entry and 200 when an entry with
    the same normalised content already existed in the workspace (in which case
    the stored metadata is returned untouched -- use PATCH to change it).
    Embedding is only requested when a new row is actually going to be written.
    """
    await crud.ensure_workspace_exists(db, workspace_id)

    existing = await crud.get_entry_by_content_hash(
        db, workspace_id, crud.content_hash(body.content)
    )
    if existing is not None:
        response.status_code = 200
        return schemas.KnowledgeCreateResult(
            created=False, entry=schemas.KnowledgeEntry.model_validate(existing)
        )

    embedding = await crud.embed_knowledge_text(
        crud.embedding_input(body.title, body.content),
        workspace_id,
        field="Content",
    )
    entry, created = await crud.create_entry(
        db,
        workspace_id,
        title=body.title,
        content=body.content,
        source=body.source,
        tags=body.tags,
        embedding=embedding,
    )
    if not created:
        response.status_code = 200
    logger.debug(
        "knowledge entry %s saved in workspace %s (created=%s)",
        entry.id,
        workspace_id,
        created,
    )
    return schemas.KnowledgeCreateResult(
        created=created, entry=schemas.KnowledgeEntry.model_validate(entry)
    )


@router.get(
    "",
    response_model=schemas.KnowledgePage,
)
async def list_entries(
    workspace_id: str = Path(...),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    read_db: AsyncSession = read_db,
) -> schemas.KnowledgePage:
    """List a workspace's entries, newest first (``created_at DESC, id DESC``)."""
    await crud.ensure_workspace_exists(read_db, workspace_id)
    items, total = await crud.list_entries(
        read_db, workspace_id, limit=limit, offset=offset
    )
    return schemas.KnowledgePage(
        items=[schemas.KnowledgeEntry.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/query",
    response_model=schemas.KnowledgeQueryResult,
)
async def query_entries(
    workspace_id: str = Path(...),
    body: schemas.KnowledgeQuery = Body(
        ...,
        description="Semantic query over saved knowledge entries",
    ),
    read_db: AsyncSession = read_db,
) -> schemas.KnowledgeQueryResult:
    """Semantic search over saved entries; every hit carries its actual distance.

    No match is a normal 200 with ``hits: []``; provider failure is a 503.
    """
    await crud.ensure_workspace_exists(read_db, workspace_id)
    hits = await crud.query_entries(
        read_db,
        workspace_id,
        query=body.query,
        top_k=body.top_k,
        max_distance=body.max_distance,
    )
    return schemas.KnowledgeQueryResult(
        query=body.query,
        hits=[
            schemas.KnowledgeHit(
                entry=schemas.KnowledgeEntry.model_validate(entry),
                distance=distance,
            )
            for entry, distance in hits
        ],
    )


@router.get(
    "/{entry_id}",
    response_model=schemas.KnowledgeEntryResult,
)
async def get_entry(
    workspace_id: str = Path(...),
    entry_id: str = Path(...),
    read_db: AsyncSession = read_db,
) -> schemas.KnowledgeEntryResult:
    """Read one entry by id (404 outside the workspace)."""
    await crud.ensure_workspace_exists(read_db, workspace_id)
    entry = await crud.get_entry(read_db, workspace_id, entry_id)
    return schemas.KnowledgeEntryResult(
        entry=schemas.KnowledgeEntry.model_validate(entry)
    )


@router.patch(
    "/{entry_id}",
    response_model=schemas.KnowledgeUpdateResult,
)
async def update_entry(
    workspace_id: str = Path(...),
    entry_id: str = Path(...),
    body: schemas.KnowledgeEntryUpdate = Body(
        ...,
        description="Partial update; omitted fields are left unchanged",
    ),
    db: AsyncSession = db,
) -> schemas.KnowledgeUpdateResult:
    """Partially update one entry.

    ``reembedded`` reports whether the vector was recomputed (title or content
    changed). Values that already match are a no-op: nothing is written and
    ``updated_at`` is not bumped.
    """
    if not body.model_fields_set:
        raise ValidationException("no fields to update")
    await crud.ensure_workspace_exists(db, workspace_id)
    patch = crud.KnowledgePatch(
        title=body.title,
        content=body.content,
        source=body.source,
        tags=body.tags,
    )
    entry, reembedded = await crud.update_entry(db, workspace_id, entry_id, patch)
    return schemas.KnowledgeUpdateResult(
        entry=schemas.KnowledgeEntry.model_validate(entry),
        reembedded=reembedded,
    )


@router.delete(
    "/{entry_id}",
    status_code=204,
)
async def delete_entry(
    workspace_id: str = Path(...),
    entry_id: str = Path(...),
    db: AsyncSession = db,
) -> Response:
    """Hard-delete one entry (404 if it is not in this workspace)."""
    await crud.ensure_workspace_exists(db, workspace_id)
    await crud.delete_entry(db, workspace_id, entry_id)
    return Response(status_code=204)
