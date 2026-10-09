"""场地描述（session profile）的读写。

与 peer card / peer profile 的对照（doc/memory_layer_decision.md §23.1）：

| | peer | session |
|---|---|---|
| 稳定标记 | ``peer_card`` | （无） |
| 长期叙事 | ``peer_profile`` | **``session_profile``** |
| 时间线 | （无） | ``summaries`` |

存储位置：``Session.internal_metadata`` 里 label ``"session_profile"``
——与 ``"summaries"`` 同一个 JSONB 列，**无需迁移**。
"""

from __future__ import annotations

import logging
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.cache.client import safe_cache_delete
from src.crud.session import session_cache_key

logger = logging.getLogger(__name__)

SESSION_PROFILE_LABEL = "session_profile"


async def get_session_profile(
    db: AsyncSession,
    workspace_name: str,
    *,
    session_name: str,
) -> schemas.SessionProfile | None:
    """读取某个 session 的场地描述。

    Args:
        db: 数据库会话
        workspace_name: 工作区名
        session_name: 会话名

    Returns:
        描述，不存在或存量数据形态不符时返回 None。

    Raises:
        ResourceNotFoundException: session 不存在（与 ``get_both_summaries`` 一致，
            调用方按需捕获）。
    """
    session = await _get_session(db, workspace_name, session_name)
    if session is None:
        return None
    raw = (session.internal_metadata or {}).get(SESSION_PROFILE_LABEL)
    if not isinstance(raw, dict):
        return None
    try:
        return schemas.SessionProfile.model_validate(raw)
    except Exception:
        # 存量/手改数据形态不符时不该让读取路径炸掉——描述只是增强信息。
        logger.warning(
            "Malformed session profile for %s/%s; ignoring",
            workspace_name,
            session_name,
        )
        return None


async def set_session_profile(
    db: AsyncSession,
    workspace_name: str,
    profile: schemas.SessionProfile,
    *,
    session_name: str,
) -> None:
    """写入（整体替换）某个 session 的场地描述。

    Args:
        db: 数据库会话
        workspace_name: 工作区名
        profile: 新的描述
        session_name: 会话名

    Raises:
        ResourceNotFoundException: session 不存在。
    """
    stmt = (
        update(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
        .values(
            internal_metadata=models.Session.internal_metadata.op("||")(
                {SESSION_PROFILE_LABEL: profile.model_dump(mode="json")}
            )
        )
    )
    result = cast(CursorResult[Any], await db.execute(stmt))
    if result.rowcount == 0:
        raise exceptions.ResourceNotFoundException(
            f"Session {session_name} not found in workspace {workspace_name}"
        )
    await db.commit()

    # 读穿缓存：session 被缓存了，描述跟着一起失效。
    await safe_cache_delete(session_cache_key(workspace_name, session_name))


async def _get_session(
    db: AsyncSession, workspace_name: str, session_name: str
) -> models.Session | None:
    from src import crud

    try:
        return await crud.get_session(
            db, workspace_name=workspace_name, session_name=session_name
        )
    except exceptions.ResourceNotFoundException:
        return None
