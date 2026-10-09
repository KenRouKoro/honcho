"""Per-session display names (群名片 / group card).

Why this module exists (doc/memory_layer_decision.md §27):

A QQ participant has at least **two** names, and they behave differently:

- the **QQ nickname** — global to the account, changes rarely;
- the **group card** (群名片) — *per group*, set by the participant or an admin,
  and frequently changed. On the reference deployment the bot itself had five
  different cards in a single group, and the global nickname disagreed with the card
  on 5 of 7 (peer, session) pairs inspected.

The bridge already ships the group card with every message
(``metadata.user_name``, present on 41,587 of 42,518 messages in the busiest group),
so the data has always been available — it just had nowhere to live. Writing it to
``Peer.h_metadata['name']`` would make one group's card overwrite another's, which is
why the client deliberately only ever writes the *global* nickname there.

``session_peers.internal_metadata`` is the correct home: it is keyed by
``(workspace, session, peer)``, it is an internal column (not the public
``configuration`` contract), and it already existed unused — 733 of 733 rows were empty.
**No migration.**
"""

from __future__ import annotations

import logging
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src import models

logger = logging.getLogger(__name__)

# Keys inside `session_peers.internal_metadata`.
DISPLAY_NAME_KEY = "display_name"  # 群名片：这个群里他被怎么称呼
NICKNAME_KEY = "nickname"  # QQ 昵称：全局名，作为名片的快照留底
TITLE_KEY = "title"  # 专属头衔（平台 MemberInfo.title）
ROLE_KEY = "role"  # owner / admin / member
SEEN_AT_KEY = "display_name_seen_at"  # 最近一次观察到该名片的时刻（ISO 8601）


async def get_session_display_names(
    db: AsyncSession,
    workspace_name: str,
    *,
    session_name: str,
    peers: list[str] | None = None,
) -> dict[str, str]:
    """Return ``{peer_name: display_name}`` for a session.

    Args:
        db: database session
        workspace_name: workspace
        session_name: session
        peers: restrict to these peers (``None`` = every peer with a known name)

    Returns:
        Only peers whose card is known; absent peers are simply missing, never blank.
    """
    stmt = select(
        models.SessionPeer.peer_name, models.SessionPeer.internal_metadata
    ).where(
        models.SessionPeer.workspace_name == workspace_name,
        models.SessionPeer.session_name == session_name,
    )
    if peers is not None:
        if not peers:
            return {}
        stmt = stmt.where(models.SessionPeer.peer_name.in_(peers))

    rows = (await db.execute(stmt)).all()
    out: dict[str, str] = {}
    for peer_name, metadata in rows:
        meta = cast(dict[str, Any], metadata or {})
        value = meta.get(DISPLAY_NAME_KEY)
        if isinstance(value, str) and value.strip():
            out[peer_name] = value.strip()
    return out


async def record_session_display_names(
    db: AsyncSession,
    workspace_name: str,
    *,
    session_name: str,
    names: dict[str, str],
    seen_at: str,
    extra: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Store the group cards we just observed. Returns how many rows changed.

    Only writes when the value actually differs or is new, so the steady-state cost of
    calling this on every deriver batch is one SELECT plus a no-op.

    ``extra`` allows attaching platform facts that arrive with a roster sync rather
    than with a message (``title`` / ``role``), keyed by peer name.
    """
    cleaned = {
        peer: name.strip() for peer, name in names.items() if name and name.strip()
    }
    if not cleaned:
        return 0

    existing = (
        await db.execute(
            select(
                models.SessionPeer.peer_name, models.SessionPeer.internal_metadata
            ).where(
                models.SessionPeer.workspace_name == workspace_name,
                models.SessionPeer.session_name == session_name,
                models.SessionPeer.peer_name.in_(list(cleaned)),
            )
        )
    ).all()
    known: dict[str, dict[str, Any]] = {
        peer: dict(metadata or {}) for peer, metadata in existing
    }

    written = 0
    for peer, name in cleaned.items():
        current = known.get(peer)
        patch: dict[str, Any] = {DISPLAY_NAME_KEY: name, SEEN_AT_KEY: seen_at}
        for key, value in (extra or {}).get(peer, {}).items():
            if value:
                patch[key] = value

        # Update-only: membership rows are created by the session/message paths, and a
        # membership row that exists but carries a name we already have is a no-op.
        # (On the reference deployment every speaker already had a row: 157/157.)
        if current is not None and all(current.get(k) == v for k, v in patch.items()):
            continue

        result = cast(
            CursorResult[Any],
            await db.execute(
                update(models.SessionPeer)
                .where(
                    models.SessionPeer.workspace_name == workspace_name,
                    models.SessionPeer.session_name == session_name,
                    models.SessionPeer.peer_name == peer,
                )
                .values(
                    internal_metadata=models.SessionPeer.internal_metadata.op("||")(
                        patch
                    )
                )
            ),
        )
        if result.rowcount:
            written += 1
        else:
            logger.debug(
                "No session_peers row for %s/%s/%s; display name not stored",
                workspace_name,
                session_name,
                peer,
            )

    if written:
        await db.commit()
    return written


def short_peer_id(peer_name: str) -> str:
    """``user_123456789012`` → ``123456789012``.

    Presentation only: identity labels read ``小白(123456789012)`` rather than
    ``小白(user_123456789012)``. Peer names are otherwise untouched — they stay the
    stable, unique key everywhere in the data model.
    """
    if peer_name.startswith("user_"):
        return peer_name[len("user_") :]
    return peer_name
