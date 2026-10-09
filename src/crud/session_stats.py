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
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.crud.display_name import get_session_display_names, short_peer_id

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
    return [item for item in items if isinstance(item, str)]


async def get_session_stats(
    db: AsyncSession,
    workspace_name: str,
    *,
    session_name: str,
) -> schemas.SessionStats | None:
    """算出一个 session 的成员与体量统计。session 不存在时返回 None。

    只有两条查询：名册来自 ``Session.metadata``（客户端写的，零成本），
    发言分布来自 ``messages`` 的一次 GROUP BY。
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

    # Bounded sample of the loudest participants, with their group cards. This is what
    # lets a session description (and any caller) say "小白" instead of
    # "user_123456789012" without shipping a 157-entry roster. §27.
    top_n = settings.SESSION_PROFILE.TOP_SPEAKERS
    loudest = sorted(rows, key=lambda r: (-r[1], r[0]))[:top_n] if top_n > 0 else []
    names = await get_session_display_names(
        db,
        workspace_name,
        session_name=session_name,
        peers=[peer for peer, _ in loudest],
    )
    top_speakers = [
        schemas.SessionSpeaker(
            peer_name=peer, display_name=names.get(peer), message_count=n
        )
        for peer, n in loudest
    ]

    metadata: dict[str, Any] = session.h_metadata or {}
    cfg = settings.SESSION_PROFILE
    member_count = _as_int(metadata.get(cfg.MEMBER_COUNT_METADATA_KEY))
    # 名册可能过时（有人退群/新入群）。潜水数不该为负，也不该超出名册总数。
    lurker_count = (
        max(0, member_count - len(rows)) if member_count is not None else None
    )

    group_name_raw = metadata.get(cfg.GROUP_NAME_METADATA_KEY)

    return schemas.SessionStats(
        group_name=group_name_raw if isinstance(group_name_raw, str) else None,
        member_count=member_count,
        max_member_count=_as_int(metadata.get(cfg.MAX_MEMBER_COUNT_METADATA_KEY)),
        admins=_as_str_list(metadata.get(cfg.ADMINS_METADATA_KEY)),
        message_count=message_count,
        speaker_count=len(rows),
        regular_count=regular,
        occasional_count=occasional,
        lurker_count=lurker_count,
        regular_min_messages=threshold,
        top_speakers=top_speakers,
    )


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
    if stats.admins:
        lines.append(f"管理员：{', '.join(stats.admins)}")
    if stats.top_speakers:
        lines.append(
            "发言最多的人（名字(稳定id)：条数）"
            + "——**提到这些人时用这个名字，不要用 id**："
        )
        for speaker in stats.top_speakers:
            label = f"{speaker.display_name}({short_peer_id(speaker.peer_name)})"
            if speaker.display_name is None:
                label = f"{speaker.peer_name}（本群名未知）"
            lines.append(f"  - {label}：{speaker.message_count} 条")
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
