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
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings

logger = logging.getLogger(__name__)

# Keys inside `session_peers.internal_metadata`.
DISPLAY_NAME_KEY = "display_name"  # 群名片：这个群里他被怎么称呼
NICKNAME_KEY = "nickname"  # QQ 昵称：全局名，作为名片的快照留底
TITLE_KEY = "title"  # 专属头衔（平台 MemberInfo.title）
ROLE_KEY = "role"  # owner / admin / member
SEEN_AT_KEY = "display_name_seen_at"  # 最近一次观察到该名片的时刻（ISO 8601）
# 有界的曾用名列表，最近的在前：`[{"name": "老白", "last_seen": "...Z"}, …]`
# 观察是**按当时的名字**写的，所以旧名字仍然活在较早的观察里；不记历史，
# 模型就没有任何线索把"老白"和"小白"认成同一个人（§28.1）。
DISPLAY_NAME_HISTORY_KEY = "display_name_history"


@dataclass
class SessionMemberFacts:
    """Everything known about a member **in one session**.

    ``display_name`` / ``previous_names`` are observed (Honcho writes them from the
    group card that rides along with every message); ``title`` / ``role`` can only be
    *declared* by the client's roster and are surfaced for the caller's benefit — they
    are not invented here.
    """

    peer_name: str
    display_name: str | None = None
    previous_names: list[str] = field(default_factory=list)
    title: str | None = None
    role: str | None = None
    seen_at: str | None = None


def _clean_history(value: Any) -> list[dict[str, str]]:
    """Validate a stored history list, dropping anything malformed.

    Defensive on purpose: this value is JSONB written by older/other code paths, and a
    corrupt entry must degrade to "no history" rather than crash a prompt build.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in cast(list[Any], value):
        if not isinstance(item, dict):
            continue
        entry = cast(dict[str, Any], item)
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        last_seen = entry.get("last_seen")
        out.append(
            {
                "name": name.strip(),
                "last_seen": last_seen if isinstance(last_seen, str) else "",
            }
        )
    return out


def _text(meta: dict[str, Any], key: str) -> str | None:
    value = meta.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


async def get_session_member_facts(
    db: AsyncSession,
    workspace_name: str,
    *,
    session_name: str,
    peers: list[str] | None = None,
) -> dict[str, SessionMemberFacts]:
    """Return ``{peer_name: SessionMemberFacts}`` for a session.

    Args:
        db: database session
        workspace_name: workspace
        session_name: session
        peers: restrict to these peers (``None`` = every member row with a card)

    Returns:
        Only peers with at least one known fact; absent peers are missing, never blank.
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
    out: dict[str, SessionMemberFacts] = {}
    for peer_name, metadata in rows:
        meta = cast(dict[str, Any], metadata or {})

        raw_name = meta.get(DISPLAY_NAME_KEY)
        display_name = raw_name.strip() if isinstance(raw_name, str) else None
        if display_name == "":
            display_name = None

        history = _clean_history(meta.get(DISPLAY_NAME_HISTORY_KEY))
        previous_names: list[str] = []
        for entry in history:
            name = entry["name"]
            if name == display_name or name in previous_names:
                continue
            previous_names.append(name)

        title = _text(meta, TITLE_KEY)
        role = _text(meta, ROLE_KEY)
        seen_at = _text(meta, SEEN_AT_KEY)

        if (
            display_name is None
            and not previous_names
            and title is None
            and role is None
        ):
            continue
        out[peer_name] = SessionMemberFacts(
            peer_name=peer_name,
            display_name=display_name,
            previous_names=previous_names,
            title=title,
            role=role,
            seen_at=seen_at,
        )
    return out


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
    facts = await get_session_member_facts(
        db, workspace_name, session_name=session_name, peers=peers
    )
    return {
        peer: fact.display_name
        for peer, fact in facts.items()
        if fact.display_name is not None
    }


def _next_history(
    *, current_name: str | None, current_seen_at: str | None, stored: Any, seen_at: str
) -> list[dict[str, str]]:
    """Bounded, most-recent-first list of names this member used before ``seen_at``."""
    limit = settings.SESSION_PROFILE.DISPLAY_NAME_HISTORY_LIMIT
    if limit <= 0:
        return []
    history = _clean_history(stored)
    if current_name:
        # The name we are about to replace starts its life as a *previous* name. Its
        # `last_seen` is when we last observed it, which is exactly what tells the model
        # which observations were written under it.
        history = [entry for entry in history if entry["name"] != current_name]
        history.insert(
            0,
            {"name": current_name, "last_seen": current_seen_at or seen_at},
        )
    return history[:limit]


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

        # §28.1: a rename must not erase the name the earlier observations were written
        # under. Keep a bounded, most-recent-first history so "老白" and "小白" can still
        # be recognised as one person. Only pay for it when the name actually changed.
        if current is not None:
            previous_name = current.get(DISPLAY_NAME_KEY)
            if isinstance(previous_name, str) and previous_name.strip() != name:
                patch[DISPLAY_NAME_HISTORY_KEY] = _next_history(
                    current_name=previous_name.strip(),
                    current_seen_at=(
                        current.get(SEEN_AT_KEY)
                        if isinstance(current.get(SEEN_AT_KEY), str)
                        else None
                    ),
                    stored=current.get(DISPLAY_NAME_HISTORY_KEY),
                    seen_at=seen_at,
                )

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
