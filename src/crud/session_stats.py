"""场地统计：**零 LLM** 的成员与会话体量数字。

设计要点（doc/memory_layer_decision.md §23.4、§23.5、§25.3）：

1. **数字由代码算，不交给模型数。** 模型的任务是写"氛围/规矩/黑话"，
   不是数人头——数人头它既不准也不必要。
2. **潜水者只能来自客户端名册。** Honcho 的 peer 是**发言驱动**创建的
   （`deriver/enqueue.py` 只在收到消息时写 peer），所以从未发言的人
   在这里**没有任何对象**。`lurker_count = member_count − speaker_count`，
   名册缺失时返回 None 而不是假装是 0。
3. **分档按消息数**（平台事实、不可伪造），不按观察条数。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.crud.display_name import (
    get_session_member_facts,
    short_peer_id,
)

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> int | None:
    """名册字段来自客户端 JSONB，类型不保证。不接受 bool（它是 int 的子类）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast(list[Any], value)
    return [item for item in items if isinstance(item, str) and item.strip()]


def _as_str_map(value: Any) -> dict[str, str]:
    """``{peer: text}`` from client JSONB, dropping anything malformed.

    The roster is a *declaration*: a wrong type must degrade to "not supplied" rather
    than crash the context path, and it must never be silently coerced into a string.
    """
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, item in cast(dict[Any, Any], value).items():
        if isinstance(key, str) and isinstance(item, str) and item.strip():
            out[key] = item.strip()
    return out


def _roster_age_seconds(synced_at: Any) -> tuple[str | None, int | None]:
    """``(iso, age_seconds)`` from the client's roster timestamp.

    A naive timestamp is read as UTC, matching how every other timestamp in this
    codebase is interpreted. An unparseable one is reported as unknown, never as 0 —
    "we don't know when this roster was taken" and "it was taken just now" lead to
    different sentences in the prompt (§28.3).
    """
    if not isinstance(synced_at, str) or not synced_at.strip():
        return None, None
    raw = synced_at.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Unparseable roster_synced_at: %r", synced_at)
        return None, None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - parsed).total_seconds()
    return raw, max(0, int(age))


async def _find_duplicate_display_names(
    db: AsyncSession, workspace_name: str, session_name: str
) -> list[str]:
    """Group cards shared by more than one member of this session.

    A name is not an identity: two people can both be called 「摸鱼」. The naming rule
    already carries the stable id, but a reader has to *know* when the id is load
    bearing, otherwise it will happily merge two people into one (§28.5).

    Done as an aggregate in SQL: this must not pull a whole roster into memory just to
    notice a collision, and the column is already indexed by the (ws, session) prefix.
    """
    name_expr = models.SessionPeer.internal_metadata.op("->>")("display_name")
    rows = (
        await db.execute(
            select(name_expr.label("name"))
            .where(
                models.SessionPeer.workspace_name == workspace_name,
                models.SessionPeer.session_name == session_name,
                name_expr.is_not(None),
            )
            .group_by(name_expr)
            .having(func.count() > 1)
        )
    ).all()
    return sorted(
        name.strip() for (name,) in rows if isinstance(name, str) and name.strip()
    )


async def get_session_stats(
    db: AsyncSession,
    workspace_name: str,
    *,
    session_name: str,
) -> schemas.SessionStats | None:
    """算出一个 session 的成员与体量统计。session 不存在时返回 None。

    只有三条查询：名册来自 ``Session.metadata``（客户端写的，零成本），
    发言分布来自 ``messages`` 的一次 GROUP BY，重名来自 ``session_peers`` 的一次
    GROUP BY（只返回碰撞项）。
    """
    session = await _get_session(db, workspace_name, session_name)
    if session is None:
        return None

    # 每条消息的发送者与条数。"常驻"按消息数分档——平台事实，比观察条数稳。
    rows = (
        await db.execute(
            select(models.Message.peer_name, func.count().label("n"))
            .where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
            .group_by(models.Message.peer_name)
        )
    ).all()

    threshold = settings.SESSION_PROFILE.REGULAR_MIN_MESSAGES
    message_count = sum(n for _, n in rows)
    regular = sum(1 for _, n in rows if n >= threshold)
    occasional = len(rows) - regular

    metadata: dict[str, Any] = session.h_metadata or {}
    cfg = settings.SESSION_PROFILE
    by_name = {peer: n for peer, n in rows}

    owner_raw = metadata.get(cfg.OWNER_METADATA_KEY)
    owner = (
        owner_raw.strip() if isinstance(owner_raw, str) and owner_raw.strip() else None
    )
    admins = _as_str_list(metadata.get(cfg.ADMINS_METADATA_KEY))
    # 平台事实里同一件事有两个来源：客户端"声明"的 role/title（名册），以及
    # Honcho 自己从消息里观察到的群名片。前者权威，后者兜底。
    declared_roles = _as_str_map(metadata.get(cfg.MEMBER_ROLES_METADATA_KEY))
    declared_titles = _as_str_map(metadata.get(cfg.MEMBER_TITLES_METADATA_KEY))

    # Bounded sample of the loudest participants, with their group cards. This is what
    # lets a session description (and any caller) say "小白" instead of
    # "user_123456789012" without shipping a 157-entry roster. §27.
    top_n = settings.SESSION_PROFILE.TOP_SPEAKERS
    loudest = sorted(rows, key=lambda r: (-r[1], r[0]))[:top_n] if top_n > 0 else []

    # One fetch covering both the sample and the admins — the latter need names too,
    # otherwise the roster arrives as a list of opaque ids the model cannot reuse.
    wanted = sorted(
        {peer for peer, _ in loudest} | set(admins) | ({owner} if owner else set())
    )
    facts = await get_session_member_facts(
        db, workspace_name, session_name=session_name, peers=wanted
    )
    duplicate_names = await _find_duplicate_display_names(
        db, workspace_name, session_name
    )

    def _speaker(peer: str, count: int) -> schemas.SessionSpeaker:
        fact = facts.get(peer)
        return schemas.SessionSpeaker(
            peer_name=peer,
            display_name=fact.display_name if fact else None,
            message_count=count,
            previous_names=fact.previous_names if fact else [],
            title=(declared_titles.get(peer) or (fact.title if fact else None)),
            role=(declared_roles.get(peer) or (fact.role if fact else None)),
        )

    top_speakers = [_speaker(peer, n) for peer, n in loudest]
    admin_speakers = [_speaker(peer, by_name.get(peer, 0)) for peer in admins]

    member_count = _as_int(metadata.get(cfg.MEMBER_COUNT_METADATA_KEY))
    # 名册可能过时（有人退群/新入群）。潜水数不该为负，也不该超出名册总数。
    lurker_count = (
        max(0, member_count - len(rows)) if member_count is not None else None
    )

    synced_at, roster_age = _roster_age_seconds(
        metadata.get(cfg.ROSTER_SYNCED_AT_METADATA_KEY)
    )
    roster_stale = (
        roster_age > cfg.ROSTER_MAX_AGE_HOURS * 3600 if roster_age is not None else None
    )

    group_name_raw = metadata.get(cfg.GROUP_NAME_METADATA_KEY)

    return schemas.SessionStats(
        group_name=group_name_raw if isinstance(group_name_raw, str) else None,
        member_count=member_count,
        max_member_count=_as_int(metadata.get(cfg.MAX_MEMBER_COUNT_METADATA_KEY)),
        owner=owner,
        admins=admins,
        admin_details=admin_speakers,
        roster_synced_at=synced_at,
        roster_age_seconds=roster_age,
        roster_stale=roster_stale,
        message_count=message_count,
        speaker_count=len(rows),
        regular_count=regular,
        occasional_count=occasional,
        lurker_count=lurker_count,
        regular_min_messages=threshold,
        top_speakers=top_speakers,
        duplicate_display_names=duplicate_names,
    )


def _speaker_label(speaker: schemas.SessionSpeaker) -> str:
    """``小白(123456789012)`` / ``小白(123456789012，曾用名：老白)`` / 裸 id。"""
    if speaker.display_name is None:
        return f"{speaker.peer_name}（本群名未知）"
    label = f"{speaker.display_name}({short_peer_id(speaker.peer_name)}"
    if speaker.previous_names:
        label += "，曾用名：" + "、".join(speaker.previous_names)
    label += ")"
    if speaker.title:
        label += f"·{speaker.title}"
    return label


def _age_text(seconds: int) -> str:
    if seconds < 3600:
        return f"{seconds // 60} 分钟前"
    if seconds < 86400:
        return f"{seconds // 3600} 小时前"
    return f"{seconds // 86400} 天前"


def stats_block(stats: schemas.SessionStats) -> str:
    """把统计渲染成提示词里可逐字引用的块。

    **代码算好的数字直接给模型看，并要求它原样引用**——防止模型自己估人数。
    """
    lines = ["<session_stats>"]
    if stats.group_name:
        lines.append(f"场地名称：{stats.group_name}")
    lines.append(f"消息总数：{stats.message_count}")
    lines.append(f"说过话的人：{stats.speaker_count}")
    lines.append(f"  - 常驻（≥{stats.regular_min_messages} 条）：{stats.regular_count}")
    lines.append(f"  - 偶尔出现：{stats.occasional_count}")
    if stats.member_count is None:
        lines.append("名册总人数：（客户端未提供）")
        lines.append("纯潜水人数：未知——**不要编造**，也不要假装群里只有上面这些人")
    else:
        lines.append(f"名册总人数：{stats.member_count}")
        lines.append(f"  - 纯潜水（从未发言）：{stats.lurker_count}")
        if stats.max_member_count is not None:
            lines.append(f"  - 人数上限：{stats.max_member_count}")
        # §28.3: a roster is a snapshot. An outdated count is a lower bound, not a fact.
        if stats.roster_age_seconds is None:
            lines.append("  - 名册同步时间：未知")
        elif stats.roster_stale:
            lines.append(
                f"  - ⚠️ 名册同步于 {_age_text(stats.roster_age_seconds)}"
                f"（超过 {settings.SESSION_PROFILE.ROSTER_MAX_AGE_HOURS} 小时）"
                "——人数可能已经变了，只能当参考，不要当成现状"
            )
        else:
            lines.append(f"  - 名册同步于 {_age_text(stats.roster_age_seconds)}")
    admin_speakers = stats.admin_details
    if admin_speakers:
        rendered = []
        for speaker in admin_speakers:
            label = _speaker_label(speaker)
            if speaker.role and speaker.role not in label:
                label += f"（{speaker.role}）"
            rendered.append(label)
        lines.append(f"管理员：{', '.join(rendered)}")
    if stats.top_speakers:
        lines.append(
            "发言最多的人（名字(稳定id)：条数）"
            + "——**提到这些人时用这个名字，不要用 id**："
        )
        for speaker in stats.top_speakers:
            lines.append(f"  - {_speaker_label(speaker)}：{speaker.message_count} 条")
    if stats.duplicate_display_names:
        # §28.5: the id in the label is normally decorative; here it is the only thing
        # that separates two people. Say so, or the reader will merge them.
        names = "、".join(f"「{name}」" for name in stats.duplicate_display_names)
        lines.append(
            f"⚠️ 本群有不止一个人用相同的群名片：{names}"
            "——**名字在这里不足以区分人，提到时必须带上括号里的 id**"
        )
    lines.append("</session_stats>")
    return "\n".join(lines)


async def _get_session(
    db: AsyncSession, workspace_name: str, session_name: str
) -> models.Session | None:
    from src import crud, exceptions

    try:
        return await crud.get_session(
            db, workspace_name=workspace_name, session_name=session_name
        )
    except exceptions.ResourceNotFoundException:
        return None
