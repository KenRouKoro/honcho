"""Pydantic schemas for API request/response validation.

These schemas are consumed by the FastAPI routers and define the public
API contract.
"""

import datetime
import ipaddress
import re
from typing import Annotated, Any, Literal, Self, cast
from urllib.parse import urlparse

import tiktoken
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from src.config import ReasoningLevel, settings
from src.schemas.configuration import (
    DreamType,
    MessageConfiguration,
    SessionConfiguration,
    SessionPeerConfig,
    WorkspaceConfiguration,
)
from src.utils.sanitization import NulStripped, strip_nul
from src.utils.scopes import (
    SCOPE_PEER_PREFIX,
    is_scope_peer_name,
    scope_name_from_peer,
)
from src.utils.types import DocumentLevel

# ---------------------------------------------------------------------------
# Metadata validation helpers
# ---------------------------------------------------------------------------

RESOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"

_METADATA_MAX_KEYS = 100
_METADATA_MAX_DEPTH = 5


def _check_metadata_limits(
    data: dict[str, Any],
    *,
    _current_depth: int = 1,
) -> None:
    """Validate metadata dict doesn't exceed key count or nesting depth limits."""
    if _current_depth > _METADATA_MAX_DEPTH:
        raise ValueError(
            f"Metadata nesting exceeds maximum depth of {_METADATA_MAX_DEPTH}"
        )
    if _current_depth == 1 and len(data) > _METADATA_MAX_KEYS:
        raise ValueError(
            f"Metadata exceeds maximum of {_METADATA_MAX_KEYS} top-level keys"
        )
    for v in data.values():
        if isinstance(v, dict):
            _check_metadata_limits(
                cast(dict[str, Any], v), _current_depth=_current_depth + 1
            )


def _validate_metadata(v: Any) -> Any:
    """Validate and sanitize a metadata dict: enforce limits and strip NUL bytes."""
    if not isinstance(v, dict):
        return v
    data = cast(dict[str, Any], v)
    _check_metadata_limits(data)
    return strip_nul(data)


_SanitizedMetadata = Annotated[dict[str, Any], BeforeValidator(_validate_metadata)]

# Scope names are stored as peer names with the reserved prefix prepended, so
# they must leave room for the prefix within the 512-character peer name limit.
_SCOPE_NAME_MAX_LENGTH = 512 - len(SCOPE_PEER_PREFIX)


def _validate_scope_name(name: str) -> str:
    """Validate an unprefixed scope name."""
    if not 1 <= len(name) <= _SCOPE_NAME_MAX_LENGTH:
        raise ValueError(
            f"Scope name must be between 1 and {_SCOPE_NAME_MAX_LENGTH} characters"
        )
    # Checked before the charset pattern: the reserved prefix is itself outside
    # RESOURCE_NAME_PATTERN, so the pattern would otherwise reject a
    # double-prefixed name first and report the charset instead of the real
    # mistake.
    if name.startswith(SCOPE_PEER_PREFIX):
        raise ValueError(
            "Scope name must not start with the reserved prefix "
            + f"'{SCOPE_PEER_PREFIX}' (scope names are unprefixed)"
        )
    if not re.fullmatch(RESOURCE_NAME_PATTERN, name):
        raise ValueError(f"Scope name must match pattern {RESOURCE_NAME_PATTERN}")
    return name


_ScopeName = Annotated[str, AfterValidator(_validate_scope_name)]

# The `scope` read option (chat / representation): one scope name, or a bounded
# list of them. The length cap sits on the list member so it bounds the *list* —
# a single name is already bounded by `_validate_scope_name`, and a union-level
# `max_length` would cap that name's characters instead. The upper bound matches
# `SessionCreate.scopes`; the lower one rejects `[]`, which would otherwise
# resolve to an empty allowlist and silently recall nothing.
_ScopeOption = (
    _ScopeName | Annotated[list[_ScopeName], Field(min_length=1, max_length=100)]
)


# ---------------------------------------------------------------------------
# Workspace schemas
# ---------------------------------------------------------------------------


class WorkspaceBase(BaseModel):
    pass


class WorkspaceCreate(WorkspaceBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=512, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: _SanitizedMetadata = {}
    configuration: WorkspaceConfiguration = Field(
        default_factory=WorkspaceConfiguration
    )

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class WorkspaceGet(WorkspaceBase):
    filters: dict[str, Any] | None = None


class WorkspaceUpdate(WorkspaceBase):
    metadata: _SanitizedMetadata | None = None
    configuration: WorkspaceConfiguration | None = None


class Workspace(WorkspaceBase):
    name: str = Field(serialization_alias="id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


# ---------------------------------------------------------------------------
# Peer schemas
# ---------------------------------------------------------------------------


class PeerBase(BaseModel):
    pass


class PeerSpec(PeerBase):
    """Peer identity plus optional updates, for callers that already have a name.

    ``PeerCreate`` narrows ``name`` with ``pattern=RESOURCE_NAME_PATTERN`` because it
    validates a *new, user-supplied* peer id at the API boundary. crud paths reach
    ``get_or_create_peers`` with names that already exist — a path param, a message
    author, an existing row — including pre-``d429de0e5338`` legacy names containing
    '.' and every ``scope.``-prefixed peer name. Re-validating those turns a lookup
    into a raw pydantic ValidationError, i.e. an HTTP 500.

    Carries **no** constraints at all, deliberately. Length limits here were the
    same trap as the charset pattern: request-bound peer names (message authors,
    session peer-map keys) have no length bound of their own, so an empty or
    over-long name reached ``PeerSpec(...)`` and raised internally — again a 500.
    Every rule for a *new* name lives in ``crud.peer._validate_new_peer_names``,
    which runs on the insert path only.
    """

    name: str
    metadata: _SanitizedMetadata | None = None
    configuration: dict[str, Any] | None = None


class PeerCreate(PeerSpec):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=512, pattern=RESOURCE_NAME_PATTERN),
    ]

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class PeerGet(PeerBase):
    filters: dict[str, Any] | None = None
    kind: Literal["scope", "all"] | None = Field(
        default=None,
        description=(
            "Which kinds of peers to list. Omitted (default): regular peers only "
            "(scope peers are excluded). 'scope': scope peers only. 'all': every peer."
        ),
    )


class PeerUpdate(PeerBase):
    metadata: _SanitizedMetadata | None = None
    configuration: dict[str, Any] | None = None


class Peer(PeerBase):
    name: str = Field(serialization_alias="id")
    workspace_name: str = Field(serialization_alias="workspace_id")
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class PeerRepresentationGet(BaseModel):
    session_id: str | None = Field(
        None, description="Optional session ID within which to scope the representation"
    )
    filters: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional filters to scope the representation. This endpoint "
            "supports only the 'session_id' key: a session id, a list of "
            'session ids, or {"in": [...]}. When session_id is also set, it '
            "must be included in the allowlist."
        ),
    )
    scope: _ScopeOption | None = Field(
        None,
        description=(
            "Optional (unprefixed) scope name(s) to confine the representation. "
            "A single scope reads the scope's own representation of the target "
            "peer, formed only from the scope's member sessions. A list of "
            "scopes restricts the representation to conclusions from the union "
            "of the scopes' member sessions (explicit allowlist, fail-closed: "
            "an empty union yields an empty representation). Mutually "
            "exclusive with `filters` and `session_id`. Requires a workspace- "
            "or admin-level key."
        ),
    )
    target: str | None = Field(
        None,
        description="Optional peer ID to get the representation for, from the perspective of this peer",
    )
    search_query: str | None = Field(
        None,
        description="Optional input to curate the representation around semantic search results",
    )
    search_top_k: int | None = Field(
        None,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Number of semantic-search-retrieved conclusions to include in the representation",
    )
    search_max_distance: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Only used if `search_query` is provided. Maximum distance to search for semantically relevant conclusions",
    )
    include_most_frequent: bool | None = Field(
        default=None,
        description="Only used if `search_query` is provided. Whether to include the most frequent conclusions in the representation",
    )
    max_conclusions: int | None = Field(
        default=25,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Maximum number of conclusions to include in the representation",
    )


class RepresentationResponse(BaseModel):
    representation: str


class PeerCardResponse(BaseModel):
    peer_card: list[str] | None = Field(
        None, description="The peer card content, or None if not found"
    )


class PeerCardSet(BaseModel):
    peer_card: list[str] = Field(..., description="The peer card content to set")

    @field_validator("peer_card", mode="before")
    @classmethod
    def sanitize_peer_card(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [
                strip_nul(item) if isinstance(item, str) else item
                for item in cast(list[Any], v)
            ]
        return v


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=settings.MAX_MESSAGE_SIZE)]
    peer_name: str = Field(alias="peer_id")
    metadata: _SanitizedMetadata | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None

    _encoded_message: list[int] = PrivateAttr(default=[])

    @field_validator("content", mode="after")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return strip_nul(v)

    @property
    def encoded_message(self) -> list[int]:
        return self._encoded_message

    @model_validator(mode="after")
    def validate_and_set_token_count(self) -> Self:
        encoding = tiktoken.get_encoding("o200k_base")
        encoded_message = encoding.encode(self.content)

        self._encoded_message = encoded_message
        return self


class MessageGet(MessageBase):
    filters: dict[str, Any] | None = None


class MessageUpdate(MessageBase):
    metadata: _SanitizedMetadata | None = None


class Message(MessageBase):
    public_id: str = Field(serialization_alias="id")
    content: str
    peer_name: str = Field(serialization_alias="peer_id")
    session_name: str = Field(serialization_alias="session_id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    created_at: datetime.datetime
    workspace_name: str = Field(serialization_alias="workspace_id")
    token_count: int

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class MessageBatchCreate(BaseModel):
    """Schema for batch message creation with a max of 100 messages"""

    messages: list[MessageCreate] = Field(..., min_length=1, max_length=100)


class MessageUploadCreate(BaseModel):
    """Schema for message creation from file uploads"""

    peer_id: str = Field(..., description="ID of the peer creating the message")
    metadata: _SanitizedMetadata | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


# ---------------------------------------------------------------------------
# Session schemas
# ---------------------------------------------------------------------------


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=512, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: _SanitizedMetadata | None = None
    peer_names: dict[str, SessionPeerConfig] | None = Field(default=None, alias="peers")
    configuration: SessionConfiguration | None = None
    scopes: list[str] | None = Field(
        default=None,
        max_length=100,
        description=(
            "Optional list of (unprefixed) scope names to add this session to. "
            "Each scope is created if it does not exist yet. If the session "
            "already has messages, its existing documents are backfilled into "
            "the scope asynchronously."
        ),
    )

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for scope_name in v:
            _validate_scope_name(scope_name)
        return v

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class SessionGet(SessionBase):
    filters: dict[str, Any] | None = None


class SessionUpdate(SessionBase):
    metadata: _SanitizedMetadata | None = None
    configuration: SessionConfiguration | None = None


class Session(SessionBase):
    name: str = Field(serialization_alias="id")
    is_active: bool
    workspace_name: str = Field(serialization_alias="workspace_id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class Summary(BaseModel):
    content: str = Field(description="The summary text")
    message_id: int = Field(
        description="The internal ID of the message that this summary covers up to",
        exclude=True,
    )
    message_public_id: str = Field(
        description="The public ID of the message that this summary covers up to",
        serialization_alias="message_id",
    )
    summary_type: str = Field(description="The type of summary (short or long)")
    created_at: str = Field(
        description="The timestamp of when the summary was created (ISO format)"
    )
    token_count: int = Field(description="The number of tokens in the summary text")


class PeerProfile(BaseModel):
    """长期印象画像——把某个 peer 的**全部 explicit 观察**压成的稳定文本。

    这不是 ``peer_representation`` 的替代品，而是与它**并列**的一层：

    - ``peer_representation``：每轮按 ``search_query`` 现算的选择性子集，**每轮都变**
    - ``peer_profile``：**全局 per user**、由 dream 收尾阶段压缩出来的稳定画像，**每天最多变几次**

    稳定 = 可以进提示词的静态前缀、可以被 prompt cache 复用（见
    doc/memory_layer_decision.md §18.4 的半静态带）。所以读取方应该把画像放在**前缀**，
    表示放在**尾部**。

    存储位置：``Peer.internal_metadata`` 里 label = ``"profile"`` 的值
    （与 peer card 的 ``"peer_card"`` / ``"{observed}_peer_card"`` 并列）。**JSONB 列，无需迁移。**
    与 peer card 的分工：card 是**稳定身份标记**（IDENTITY/ATTRIBUTE/RELATIONSHIP/INSTRUCTION，
    ≤40 条）；profile 是**长期叙事**（生活、处境、关注、关系）——正是 card 刻意排除的那类内容。

    ``observation_count`` / ``oldest`` / ``newest`` / ``truncated`` 是**刻意保留的来源元信息**：
    读取方据此渲染出"这是一层压缩结果、原料多少条、覆盖到何时、有没有漏掉更早的"，
    让模型知道可以再用检索补细节——把未知的未知变成已知的未知（§17 的 P3）。
    """

    text: str = Field(description="压缩后的画像正文（固定分节的 markdown）")
    observation_count: int = Field(
        default=0,
        description=(
            "本次压缩覆盖的 explicit 观察条数。"
            "截断续压时含上一版画像间接覆盖的部分（累积值，是下界而非精确计数）。"
        ),
    )
    oldest: str = Field(default="", description="覆盖范围的最早日期（ISO date）")
    newest: str = Field(default="", description="覆盖范围的最新日期（ISO date）")
    compressed_at: str = Field(default="", description="压缩时刻（ISO 8601，UTC）")
    truncated: bool = Field(
        default=False,
        description=(
            "原料条数超过 ``PROFILE.MAX_INPUT_OBSERVATIONS``，本次只覆盖了最近一段。"
            "读取方据此知道还有更早的原料没被压进来，可以主动去检索。"
        ),
    )


class SessionProfile(BaseModel):
    """场地描述——把"这是什么地方"压成一份稳定文本。

    与 ``Summary`` 的分工（doc/memory_layer_decision.md §23.1、§23.2）：

    - ``Summary``：**发生过什么**，每 20/60 条消息重建一次，内容持续漂移，是流水
    - ``SessionProfile``：**这是什么地方**（氛围、规矩、话题域、黑话、成员构成），
      低频刷新，**稳定**，可以进提示词的静态前缀

    存储位置：``Session.internal_metadata`` 里 label = ``"session_profile"``。
    **JSONB 列，无需迁移。**

    ``observation_count`` / ``member_count`` / ``speaker_count`` /
    ``lurker_count`` 等是**代码算出来的**元信息——数字不交给模型编（§25.3）。
    ``lurker_count`` 尤其重要：**从未发言的人不会在 Honcho 里留下任何对象**，
    只有客户端名册才知道他们存在（§23.4）。
    """

    text: str = Field(description="描述正文（固定分节的 markdown）")
    observation_count: int = Field(
        default=0, description="参与压缩的 explicit 观察条数"
    )
    sample_messages: int = Field(default=0, description="该 session 当时的消息总数")
    speaker_count: int = Field(
        default=0, description="说过话的 peer 数（Honcho 唯一知道的）"
    )
    member_count: int | None = Field(
        default=None,
        description="名册总人数，来自客户端的 Session.metadata；未知时 None",
    )
    lurker_count: int | None = Field(
        default=None,
        description="从未发言的人数 = member_count − speaker_count；名册未知时 None",
    )
    oldest: str = Field(default="", description="覆盖范围的最早日期（ISO date）")
    newest: str = Field(default="", description="覆盖范围的最新日期（ISO date）")
    compressed_at: str = Field(default="", description="压缩时刻（ISO 8601，UTC）")
    truncated: bool = Field(
        default=False, description="原料超过 MAX_INPUT_OBSERVATIONS，本次只覆盖最近一段"
    )


class SessionSpeaker(BaseModel):
    """One participant of a session, as the session itself sees them.

    ``peer_name`` stays the stable key; ``display_name`` is the **group card** — the
    name used inside *this* group, which is per-session by nature and changes often
    (doc/memory_layer_decision.md §27).
    """

    peer_name: str = Field(description="Stable peer id")
    display_name: str | None = Field(
        default=None, description="The group card observed for this session, if any"
    )
    message_count: int = Field(default=0, description="Messages sent in this session")
    previous_names: list[str] = Field(
        default_factory=list,
        description=(
            "Group cards this member used **before** the current one, most recent first. "
            "Observations written under an older name stay in memory forever, so this is "
            "what lets a reader recognise 老白 and 小白 as the same person (§28.1)."
        ),
    )
    title: str | None = Field(
        default=None,
        description="Group-specific honorific (platform MemberInfo.title), when the client supplied it",
    )
    role: str | None = Field(
        default=None, description="owner / admin / member, when the client supplied it"
    )


class SessionStats(BaseModel):
    """零 LLM 的场地统计——每次请求实时算，永远新鲜。

    与 ``SessionProfile`` 的关系：这里全是**代码算出来的数字**，
    那边是**模型写出来的叙事**。数字永远不该让模型去数（§25.3）。

    分层依据（§23.5）：``regular`` / ``occasional`` 按消息数分档，
    ``lurker`` 只能由客户端名册给出——Honcho 里没有他们的任何痕迹。
    """

    group_name: str | None = Field(default=None, description="群名/场地名（名册）")
    member_count: int | None = Field(default=None, description="名册总人数")
    max_member_count: int | None = Field(default=None, description="人数上限")
    owner: str | None = Field(default=None, description="群主 peer 名（名册）")
    admins: list[str] = Field(
        default_factory=list, description="管理员 peer 名（名册）"
    )
    admin_details: list[SessionSpeaker] = Field(
        default_factory=list,
        description=(
            "``admins`` 的富化形式（同序、同长度）：带上群名片、头衔与角色。"
            "名册给的是裸 id，而描述里要写的是称呼——只给 id 等于让模型放弃引用这些人。"
        ),
    )
    roster_synced_at: str | None = Field(
        default=None, description="客户端上次拉名册的时刻（ISO 8601）；未知时 None"
    )
    roster_age_seconds: int | None = Field(
        default=None, description="名册年龄（秒）；无同步时刻时 None"
    )
    roster_stale: bool | None = Field(
        default=None,
        description=(
            "名册是否已超过 ``ROSTER_MAX_AGE_HOURS``。None = 客户端没给同步时刻，"
            "即**未知**，而不是「新鲜」——两者在提示词里说法不同（§28.3）"
        ),
    )
    message_count: int = Field(default=0, description="该 session 的消息总数")
    speaker_count: int = Field(default=0, description="说过话的人数")
    regular_count: int = Field(default=0, description="常驻（消息数达门槛）")
    occasional_count: int = Field(default=0, description="偶尔出现（说过话但未达门槛）")
    lurker_count: int | None = Field(
        default=None, description="纯潜水（从未发言）；名册未知时 None"
    )
    regular_min_messages: int = Field(default=0, description="判定「常驻」的消息数门槛")
    top_speakers: list[SessionSpeaker] = Field(
        default_factory=list,
        description=(
            "Highest-volume participants with their group cards, so a caller can render "
            "names instead of opaque peer ids. Bounded — a sample, not the roster."
        ),
    )
    duplicate_display_names: list[str] = Field(
        default_factory=list,
        description=(
            "Group cards used by more than one member **in this session**. A name is not "
            "an identity; when this is non-empty the reader must fall back to the id to "
            "tell two people apart (§28.5). Bounded by the roster, usually empty."
        ),
    )


class SessionContext(SessionBase):
    name: str = Field(serialization_alias="id")
    messages: list[Message]
    summary: Summary | None = Field(
        default=None, description="The summary if available"
    )
    peer_representation: str | None = Field(
        default=None,
        description="A curated subset of a peer representation, if context is requested from a specific perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card, if context is requested from a specific perspective",
    )
    peer_profile: PeerProfile | None = Field(
        default=None,
        description=(
            "The compressed long-term profile of the target peer, if one has been built. "
            "Stable across turns (unlike peer_representation), so it belongs in the prompt's static prefix."
        ),
    )
    session_profile: SessionProfile | None = Field(
        default=None,
        description=(
            "The compressed description of *this session* — what this place is, its norms, topics "
            "and membership. Stable across turns, unlike the summary. Follows the same access rule as "
            "`summary`: it describes only this session, so an allowlist or scope does not restrict it."
        ),
    )
    session_stats: SessionStats | None = Field(
        default=None,
        description=(
            "Live, LLM-free membership and volume statistics for this session. Cheap to compute and "
            "always fresh; the numbers here are what `session_profile` was written from."
        ),
    )

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class PeerContext(BaseModel):
    """Context for a peer, including representation and peer card."""

    peer_id: str = Field(description="The ID of the peer")
    target_id: str = Field(description="The ID of the target peer being observed")
    representation: str | None = Field(
        default=None,
        description="A curated subset of the representation of the target peer from the observer's perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card for the target peer from the observer's perspective",
    )


class SessionSummaries(SessionBase):
    name: str = Field(serialization_alias="id")
    short_summary: Summary | None = Field(
        default=None, description="The short summary if available"
    )
    long_summary: Summary | None = Field(
        default=None, description="The long summary if available"
    )

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


# ---------------------------------------------------------------------------
# Scope schemas
# ---------------------------------------------------------------------------


class ScopeCreate(BaseModel):
    """Schema for creating (or getting) a scope by its unprefixed name."""

    name: Annotated[str, Field(alias="id", min_length=1)]
    metadata: _SanitizedMetadata | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_scope_name(v)

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class Scope(BaseModel):
    """Scope response — external view of the peer backing a scope.

    The ``id`` is the unprefixed scope name; the reserved peer-name prefix is
    an internal implementation detail and never surfaces here.
    """

    name: str = Field(serialization_alias="id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    created_at: datetime.datetime

    @field_validator("name", mode="after")
    @classmethod
    def strip_scope_prefix(cls, v: str) -> str:
        # Constructed from Peer ORM rows whose names carry the prefix; accept
        # already-unprefixed names too so manual construction works.
        return scope_name_from_peer(v) if is_scope_peer_name(v) else v

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class ScopeSessionsAdd(BaseModel):
    """Schema for adding sessions to a scope."""

    session_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="IDs of existing sessions to add to the scope",
    )


class ScopeStatus(BaseModel):
    """Per-session backfill/reconciliation job status for a scope.

    ``backfill_status`` maps each session that has had a backfill enqueued to
    its current job state: ``{state, updated_at[, docs_copied]}`` where
    ``state`` is ``pending``/``completed``/``failed`` and ``docs_copied`` is
    present once a backfill completes.
    """

    backfill_status: dict[str, dict[str, Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conclusion schemas
# ---------------------------------------------------------------------------


class ConclusionGet(BaseModel):
    """Schema for listing conclusions with optional filters."""

    filters: dict[str, Any] | None = None


class Conclusion(BaseModel):
    """Conclusion response - external view of a document."""

    id: str
    content: str
    observer: str = Field(
        description="The peer who made the conclusion",
        serialization_alias="observer_id",
    )
    observed: str = Field(
        description="The peer the conclusion is about",
        serialization_alias="observed_id",
    )
    session_name: str | None = Field(default=None, serialization_alias="session_id")
    level: DocumentLevel = Field(
        default="explicit",
        description=(
            "Reasoning level of the conclusion: 'explicit' (directly extracted "
            "from messages) or 'deductive'/'inductive'/'contradiction' (derived "
            "during dreaming)."
        ),
    )
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True,
        populate_by_name=True,
    )


class ConclusionQuery(BaseModel):
    """Query parameters for semantic search of conclusions."""

    query: str = Field(..., description="Semantic search query")
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    distance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum cosine distance threshold for results",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Additional filters to apply",
    )


class ConclusionCreate(BaseModel):
    """Schema for creating a single conclusion."""

    content: Annotated[str, Field(min_length=1, max_length=65535)]
    observer_id: str = Field(..., description="The peer making the conclusion")
    observed_id: str = Field(..., description="The peer the conclusion is about")
    session_id: str | None = Field(
        default=None,
        description="A session ID to store the conclusion in, if specified",
    )

    _token_count: int = PrivateAttr(default=0)

    @field_validator("content", mode="after")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return strip_nul(v)

    @model_validator(mode="after")
    def validate_token_count(self) -> Self:
        """Validate that content doesn't exceed embedding token limit."""
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(self.content)
        self._token_count = len(tokens)

        if self._token_count > settings.EMBEDDING.MAX_INPUT_TOKENS:
            raise ValueError(
                "Content exceeds maximum embedding token limit of "
                + f"{settings.EMBEDDING.MAX_INPUT_TOKENS} "
                + f"(got {self._token_count} tokens)"
            )
        return self


class ConclusionBatchCreate(BaseModel):
    """Schema for batch conclusion creation with a max of 100 conclusions."""

    conclusions: list[ConclusionCreate] = Field(
        ...,
        min_length=1,
        max_length=100,
        validation_alias=AliasChoices("conclusions", "observations"),
    )


# ---------------------------------------------------------------------------
# Search schemas
# ---------------------------------------------------------------------------


class MessageSearchOptions(BaseModel):
    query: Annotated[str, Field(..., description="Search query"), NulStripped]
    filters: dict[str, Any] | None = Field(
        default=None, description="Filters to scope the search"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )


class WorkspaceMessageSearchOptions(MessageSearchOptions):
    """Workspace-level message search options, extended with `scope`."""

    scope: str | None = Field(
        default=None,
        description=(
            "Optional (unprefixed) scope name restricting search to the "
            "scope's member sessions. A scope with no member sessions returns "
            "no results. Mutually exclusive with a 'session_id' key in "
            "`filters`."
        ),
    )


# ---------------------------------------------------------------------------
# Dialectic schemas
# ---------------------------------------------------------------------------


class DialecticOptions(BaseModel):
    session_id: str | None = Field(
        None, description="ID of the session to scope the representation to"
    )
    filters: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional filters to scope recall. This endpoint supports only the "
            "'session_id' key: a session id, a list of session ids, or "
            '{"in": [...]}. Recall (conclusions and messages) is restricted to '
            "the allowlist; unsupported keys are rejected. When session_id is "
            "also set, it must be included in the allowlist."
        ),
    )
    scope: _ScopeOption | None = Field(
        None,
        description=(
            "Optional (unprefixed) scope name(s) to confine recall. A single "
            "scope answers from the scope's own representation of the target "
            "peer: conclusion recall is confined to what the scope observed "
            "and message recall to the scope's member sessions. A list of "
            "scopes restricts recall to the union of the scopes' member "
            "sessions (explicit allowlist, fail-closed: an empty union "
            "recalls nothing). Mutually exclusive with `filters` and "
            "`session_id`. Requires a workspace- or admin-level key."
        ),
    )
    target: str | None = Field(
        None,
        description="Optional peer to get the representation for, from the perspective of this peer",
    )
    query: Annotated[
        str,
        Field(min_length=1, max_length=10000, description="Dialectic API Prompt"),
        NulStripped,
    ]
    stream: bool = False
    reasoning_level: ReasoningLevel = Field(
        default="low",
        description="Level of reasoning to apply: minimal, low, medium, high, or max",
    )
    response_format: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional JSON Schema (root type 'object') the response must conform"
            " to. When provided, `content` is a JSON string matching this schema."
            " Only a conservative subset of JSON Schema is supported; unsupported"
            "  schemas are rejected with 422. Constraint keywords (minItems, "
            " maxLength, ...) are hints to the model, not enforced server-side."
        ),
    )


class WorkspaceChatOptions(BaseModel):
    """Options for workspace-level chat (no anchor peer; see DialecticOptions)."""

    session_id: str | None = Field(
        None, description="Optional session to scope message tools to"
    )
    query: Annotated[
        str,
        Field(min_length=1, max_length=10000, description="Workspace chat prompt"),
        NulStripped,
    ]
    stream: bool = False
    reasoning_level: ReasoningLevel = Field(
        default="low",
        description="Level of reasoning to apply: minimal, low, medium, high, or max",
    )
    response_format: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional JSON Schema (root type 'object') the response must conform"
            " to. When provided, `content` is a JSON string matching this schema."
        ),
    )
    scope: _ScopeOption | None = Field(
        None,
        description=(
            "Optional (unprefixed) scope name(s) restricting recall to the "
            "union of the scopes' member sessions (explicit allowlist, "
            "fail-closed: an empty union recalls nothing). Mutually exclusive "
            "with `session_id`. Requires a workspace- or admin-level key."
        ),
    )


class DialecticResponse(BaseModel):
    content: str | None


class DialecticStreamDelta(BaseModel):
    """Delta object for streaming dialectic responses."""

    content: str | None = None
    # Future fields can be added here:
    # premises: str | None = None
    # tokens: int | None = None
    # analytics: dict[str, Any] | None = None


class DialecticStreamChunk(BaseModel):
    """Chunk in a streaming dialectic response."""

    delta: DialecticStreamDelta
    done: bool = False


# ---------------------------------------------------------------------------
# Queue status schemas
# ---------------------------------------------------------------------------


class SessionQueueStatus(BaseModel):
    """Status for a specific session within the processing queue."""

    session_id: str | None = Field(
        default=None,
        description="Session ID if filtered by session",
    )
    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(description="Completed work units")
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")


class QueueStatus(BaseModel):
    """Aggregated processing queue status.

    Tracks user-facing task types only: representation, summary, and dream.
    Internal infrastructure tasks (reconciler, webhook, deletion) are excluded.

    Note: completed_work_units reflects items since the last periodic queue
    cleanup, not lifetime totals.
    """

    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(
        description="Completed work units (since last periodic cleanup)"
    )
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")
    sessions: dict[str, SessionQueueStatus] | None = Field(
        default=None,
        description="Per-session status when not filtered by session",
    )


# ---------------------------------------------------------------------------
# Dream scheduling schemas
# ---------------------------------------------------------------------------


class ScheduleDreamRequest(BaseModel):
    observer: str = Field(..., description="Observer peer name")
    observed: str | None = Field(
        None, description="Observed peer name (defaults to observer if not specified)"
    )
    dream_type: DreamType = Field(..., description="Type of dream to schedule")
    session_id: str | None = Field(
        None, description="Session ID to scope the dream to if specified"
    )
    rebuild: bool = Field(
        False,
        description=(
            "card_refresh dreams only: rebuild the peer card solely from "
            "observations currently in the collection, without injecting the "
            "existing card (use after removals)"
        ),
    )


# ---------------------------------------------------------------------------
# Webhook schemas
# ---------------------------------------------------------------------------


class WebhookEndpointBase(BaseModel):
    pass


class WebhookEndpointCreate(WebhookEndpointBase):
    url: str

    @field_validator("url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
        parsed = urlparse(v)

        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")

        # Only allow HTTP/HTTPS
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("Only HTTP and HTTPS URLs are allowed")

        # Block private/internal addresses
        if parsed.hostname:
            try:
                ip_address = ipaddress.ip_address(parsed.hostname)
            except ValueError:  # Not an IP literal — a hostname, leave it alone
                ip_address = None
            if ip_address is not None and (
                ip_address.is_private
                or ip_address.is_loopback
                or ip_address.is_link_local
                or ip_address.is_reserved
                or ip_address.is_multicast
                or ip_address.is_unspecified
            ):
                raise ValueError("Private/internal IP addresses are not allowed")

        return v


class WebhookEndpoint(WebhookEndpointBase):
    id: str
    workspace_name: str | None = Field(serialization_alias="workspace_id")
    url: str
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore
