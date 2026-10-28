"""Pydantic schemas for the standalone knowledge-entry API.

Kept in their own module instead of ``api.py`` on purpose: knowledge entries are
a self-contained resource (see ``doc/knowledge_base_plan.md``) whose validation
rules -- normalisation, tag hygiene, query bounds -- have nothing to do with the
peer/document schemas. ``src.schemas`` re-exports everything below.

Client-supplied ``id``/``content_hash``/timestamps are never declared here, so
they are ignored on input (``extra="ignore"``) and the server stays the single
source of truth for them.
"""

from datetime import datetime
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.utils.sanitization import strip_nul

TITLE_MAX_LENGTH: int = 200
CONTENT_MAX_LENGTH: int = 4000
SOURCE_MAX_LENGTH: int = 500
QUERY_MAX_LENGTH: int = 1000
MAX_TAGS: int = 10
TAG_MAX_LENGTH: int = 32


def _normalize_required_text(value: Any, *, field: str, max_length: int) -> Any:
    """Strip NUL bytes and surrounding whitespace, enforcing 1..``max_length``.

    Runs as a ``mode="before"`` validator so the length bounds are checked
    against the stripped value (``"   "`` is empty, ``" a "`` is one character).
    Non-string input is passed through so pydantic reports the type error.
    """
    if not isinstance(value, str):
        return value
    normalized = strip_nul(value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    return normalized


def _normalize_tags(value: Any) -> Any:
    """Strip each tag, drop repeats (order preserved), reject empties/overlong."""
    if not isinstance(value, list):
        return value
    normalized: list[str] = []
    for tag in value:
        if not isinstance(tag, str):
            return value  # let pydantic report the item type error
        cleaned = strip_nul(tag).strip()
        if not cleaned:
            raise ValueError("tags must not contain empty strings")
        if len(cleaned) > TAG_MAX_LENGTH:
            raise ValueError(f"each tag must be at most {TAG_MAX_LENGTH} characters")
        if cleaned not in normalized:
            normalized.append(cleaned)
    if len(normalized) > MAX_TAGS:
        raise ValueError(f"at most {MAX_TAGS} tags are allowed")
    return normalized


class KnowledgeEntryCreate(BaseModel):
    """``POST /knowledge`` request body."""

    title: str
    content: str
    source: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_title(cls, value: Any) -> Any:
        return _normalize_required_text(
            value, field="title", max_length=TITLE_MAX_LENGTH
        )

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: Any) -> Any:
        return _normalize_required_text(
            value, field="content", max_length=CONTENT_MAX_LENGTH
        )

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: Any) -> Any:
        return _normalize_required_text(
            value, field="source", max_length=SOURCE_MAX_LENGTH
        )

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> Any:
        return _normalize_tags(value)


class KnowledgeEntryUpdate(BaseModel):
    """``PATCH /knowledge/{entry_id}`` request body -- partial update.

    ``None`` means "not given" only for omitted fields: an explicit ``null`` is
    rejected (it is a 422, not "leave unchanged"). Use ``model_fields_set`` to
    tell the two apart -- never ``value is None``.
    """

    title: str | None = None
    content: str | None = None
    source: str | None = None
    tags: list[str] | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_title(cls, value: Any) -> Any:
        if value is None:
            return value
        return _normalize_required_text(
            value, field="title", max_length=TITLE_MAX_LENGTH
        )

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: Any) -> Any:
        if value is None:
            return value
        return _normalize_required_text(
            value, field="content", max_length=CONTENT_MAX_LENGTH
        )

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: Any) -> Any:
        if value is None:
            return value
        return _normalize_required_text(
            value, field="source", max_length=SOURCE_MAX_LENGTH
        )

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> Any:
        return _normalize_tags(value)

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> "KnowledgeEntryUpdate":
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(
                    f"{name} must not be null; omit it to leave it unchanged"
                )
        return self


class KnowledgeEntry(BaseModel):
    """A stored knowledge entry."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    title: str
    content: str
    source: str
    tags: list[str]
    content_hash: str
    created_at: datetime
    updated_at: datetime


class KnowledgeHit(BaseModel):
    """One semantic-search hit: the entry plus its actual cosine distance."""

    entry: KnowledgeEntry
    distance: float


class KnowledgeQuery(BaseModel):
    """``POST /knowledge/query`` request body."""

    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    max_distance: Annotated[float | None, Field(default=None, ge=0.0, le=2.0)] = None

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value: Any) -> Any:
        return _normalize_required_text(
            value, field="query", max_length=QUERY_MAX_LENGTH
        )


class KnowledgeQueryResult(BaseModel):
    """``POST /knowledge/query`` response body."""

    query: str
    hits: list[KnowledgeHit]


class KnowledgeCreateResult(BaseModel):
    """``POST /knowledge`` response body (201 created, 200 already existed)."""

    created: bool
    entry: KnowledgeEntry


class KnowledgeUpdateResult(BaseModel):
    """``PATCH /knowledge/{entry_id}`` response body."""

    entry: KnowledgeEntry
    reembedded: bool


class KnowledgeEntryResult(BaseModel):
    """``GET /knowledge/{entry_id}`` response body."""

    entry: KnowledgeEntry


class KnowledgePage(BaseModel):
    """``GET /knowledge`` response body -- offset/limit pagination."""

    items: list[KnowledgeEntry]
    total: int
    limit: int
    offset: int
