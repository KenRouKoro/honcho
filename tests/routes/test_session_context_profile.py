"""`peer_profile` on the session context route.

The profile sits *beside* `peer_representation`, and the two differ in exactly the
way the budget depends on: the representation is recomputed per request, the profile
is one stored row per peer. That also makes it cross-session by construction, so the
same fail-closed rule the peer card gets under an allowlist/scope has to apply.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.models import Peer, Workspace


async def _seed_profile(
    db_session: AsyncSession, workspace: Workspace, peer: Peer
) -> schemas.PeerProfile:
    profile = schemas.PeerProfile(
        text="## 生活与处境\n- 在准备考试",
        observation_count=12,
        oldest="2025-06-01",
        newest="2026-01-02",
        compressed_at="2026-01-03T00:00:00Z",
        truncated=False,
    )
    await crud.set_peer_profile(
        db_session, workspace.name, profile, peer_name=peer.name
    )
    return profile


def _context_url(
    workspace: Workspace, session_id: str, observer: Peer, *, extra: str = ""
) -> str:
    return (
        f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
        f"?summary=true&tokens=4000&peer_target={observer.name}{extra}"
    )


async def test_profile_is_returned_alongside_the_representation(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
) -> None:
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    await _seed_profile(db_session, workspace, peer)

    data = client.get(_context_url(workspace, session_id, peer)).json()

    assert data["peer_profile"] is not None
    assert data["peer_profile"]["text"].startswith("## 生活与处境")
    # The source metadata rides along so the reader knows how much is behind it.
    assert data["peer_profile"]["observation_count"] == 12
    assert data["peer_profile"]["oldest"] == "2025-06-01"
    assert data["peer_profile"]["truncated"] is False
    # Neither field replaces the other.
    assert "peer_representation" in data


async def test_profile_absent_when_none_is_stored(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
) -> None:
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )

    data = client.get(_context_url(workspace, session_id, peer)).json()

    assert data["peer_profile"] is None


async def test_profile_is_dropped_under_an_allowlist(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
) -> None:
    """One profile row covers every session the peer appeared in, so serving it
    under `limit_to_session` would hand back what the allowlist excludes."""
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    await _seed_profile(db_session, workspace, peer)

    data = client.get(
        _context_url(workspace, session_id, peer, extra="&limit_to_session=true")
    ).json()

    assert data["peer_profile"] is None
    assert data["peer_card"] is None, "same rule the card already follows"


@pytest.mark.parametrize("include_observer", [False, True])
async def test_profile_only_appears_on_the_observer_path(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
    include_observer: bool,
) -> None:
    """The profile is keyed by the observed peer, so the session-only path (no
    `peer_target`, hence no observer perspective) never carries one — mirroring
    `peer_representation` and `peer_card`."""
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    await _seed_profile(db_session, workspace, peer)

    if include_observer:
        url = _context_url(workspace, session_id, peer)
    else:
        url = (
            f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
            "?summary=true&tokens=4000"
        )

    data = client.get(url).json()
    if include_observer:
        assert data["peer_profile"] is not None
    else:
        assert data["peer_profile"] is None
