from __future__ import annotations

import logging
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.cache.client import safe_cache_delete
from src.crud.peer import (
    get_or_create_peers,
    get_peer,
    peer_cache_key,
    reject_scope_observed,
)

logger = logging.getLogger(__name__)


async def get_peer_card(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
) -> list[str] | None:
    """
    Get peer card from internal_metadata.

    The peer card is returned for the observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observed: Peer name of the peer described in the peer card
        observer: Peer name of the observer

    Returns:
        The peer's card text if present, otherwise None.

    Raises:
        ResourceNotFoundException: If the peer does not exist.
    """
    peer = await get_peer(db, workspace_name, observer)
    return cast(
        list[str] | None,
        peer.internal_metadata.get(
            construct_peer_card_label(observer=observer, observed=observed)
        ),
    )


async def set_peer_card(
    db: AsyncSession,
    workspace_name: str,
    peer_card: list[str],
    *,
    observer: str,
    observed: str,
) -> None:
    """
    Set peer card for a peer.

    If observer_name is provided, the peer card is set for the observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_card: List of strings to set as the peer card
        observed: Peer name of the peer described in the peer card
        observer: Peer name of the observer

    """
    # A scope may be the card's *observer* — the Dreamer writes (scope, observed)
    # cards — but never its subject. Authoritative here rather than only in the
    # route, so the Dreamer and agent-tool paths are covered too, and in the same
    # transaction as the JSONB write below.
    #
    # A *missing* reserved name is refused as well: only the observer is resolved
    # below, so a card keyed on `scope.future` would otherwise persist while that
    # peer does not exist and retroactively describe the scope once created.
    await reject_scope_observed(
        db,
        workspace_name,
        [observed],
        action="No peer card is ever formed about a scope.",
    )

    # Ensure the peer exists (get-or-create)
    peers_result = await get_or_create_peers(
        db, workspace_name, [schemas.PeerSpec(name=observer)]
    )

    stmt = (
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == observer)
        .values(
            internal_metadata=models.Peer.internal_metadata.op("||")(
                {
                    construct_peer_card_label(
                        observer=observer, observed=observed
                    ): peer_card
                }
            )
        )
    )
    result = cast(CursorResult[Any], await db.execute(stmt))
    if result.rowcount == 0:
        raise exceptions.ResourceNotFoundException(
            f"Peer {observer} not found in workspace {workspace_name}"
        )
    await db.commit()
    await peers_result.post_commit()

    # Invalidate cache - read-through pattern
    cache_key = peer_cache_key(workspace_name, observer)
    await safe_cache_delete(cache_key)


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"


# ---------------------------------------------------------------------------
# Profile（长期印象压缩产物）
# ---------------------------------------------------------------------------
#
# 与 peer card 的两个关键差别（见 doc/memory_layer_decision.md §18.6）：
#
# 1. **不带 observer 维度**。card 是 per (observer, observed) 的（同一 subject 在不同
#    observer 眼里可以有不同的卡）；profile 是**全局 per user**——因为 deriver 对同一
#    observed 写出的观察内容与 observer 无关（§16.4），按 observer 分别压缩只是把同一份
#    原料压 N 遍。所以 label 就是裸的 ``"profile"``，写在 **observed 自己的**
#    ``internal_metadata`` 上。
# 2. **只存一条**，不累积：每次压缩**整体替换**上一次的结果。

PROFILE_LABEL = "profile"


async def get_peer_profile(
    db: AsyncSession,
    workspace_name: str,
    *,
    peer_name: str,
) -> schemas.PeerProfile | None:
    """读取某个 peer 的长期印象画像。

    Args:
        db: 数据库会话
        workspace_name: 工作区名
        peer_name: 画像所属的 peer（即被观察者）

    Returns:
        画像，不存在或存量数据形态不符时返回 None。

    Raises:
        ResourceNotFoundException: peer 不存在。
    """
    peer = await get_peer(db, workspace_name, peer_name)
    raw = peer.internal_metadata.get(PROFILE_LABEL)
    if not isinstance(raw, dict):
        return None
    try:
        return schemas.PeerProfile.model_validate(raw)
    except Exception:
        # 存量/手改数据形态不符时不该让读取路径炸掉——画像只是增强信息。
        logger.warning(
            "Malformed peer profile for %s in workspace %s; ignoring",
            peer_name,
            workspace_name,
        )
        return None


async def set_peer_profile(
    db: AsyncSession,
    workspace_name: str,
    profile: schemas.PeerProfile,
    *,
    peer_name: str,
) -> None:
    """写入（整体替换）某个 peer 的长期印象画像。

    Args:
        db: 数据库会话
        workspace_name: 工作区名
        profile: 新的画像
        peer_name: 画像所属的 peer（即被观察者）

    Raises:
        ResourceNotFoundException: peer 不存在。
    """
    # 与 set_peer_card 同一个理由：scope 是"访问范围"的虚拟 peer，不该有画像。
    await reject_scope_observed(
        db,
        workspace_name,
        [peer_name],
        action="No profile is ever formed about a scope.",
    )

    stmt = (
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == peer_name)
        .values(
            internal_metadata=models.Peer.internal_metadata.op("||")(
                {PROFILE_LABEL: profile.model_dump(mode="json")}
            )
        )
    )
    result = cast(CursorResult[Any], await db.execute(stmt))
    if result.rowcount == 0:
        raise exceptions.ResourceNotFoundException(
            f"Peer {peer_name} not found in workspace {workspace_name}"
        )
    await db.commit()

    # 读穿缓存：peer 被缓存了，画像跟着一起失效。
    await safe_cache_delete(peer_cache_key(workspace_name, peer_name))
