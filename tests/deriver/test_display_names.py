"""Identity: how a participant's *group card* reaches observations and stats.

A QQ participant has two names that behave differently — the global nickname and the
per-group card (群名片) — and until now only the global one was stored anywhere, on the
peer, while the card rode along on every message and was never read. Observations
therefore named their subject `user_123456789012` while mentioned others appeared by name,
because those come from QQ's at/reply rendering inside the message body.

These tests pin the four moving parts (doc/memory_layer_decision.md §27):

- the card is extracted from message metadata and stored per ``(session, peer)``
- the global nickname is never overwritten by a card
- the deriver prompt names the subject ``name(id)`` and the messages show the same
- stats surface a bounded name sample so callers stop printing ids
"""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.deriver.deriver import observed_display_names, speaker_label
from src.deriver.prompts import minimal_deriver_prompt


def _message(peer: str, user_name: str | None) -> models.Message:
    metadata = {"user_name": user_name} if user_name is not None else {}
    return models.Message(
        session_name="s",
        peer_name=peer,
        content="hi",
        workspace_name="w",
        seq_in_session=1,
        token_count=2,
        h_metadata=metadata,
    )


class TestSpeakerLabel:
    def test_uses_card_and_short_id(self):
        assert speaker_label("user_123456789012", "小白") == "小白(123456789012)"

    def test_falls_back_to_bare_peer_id(self):
        """A session that never supplied a card must degrade to the old behaviour,
        not to something ambiguous like `(123456789012)`."""
        assert speaker_label("user_123456789012", None) == "user_123456789012"
        assert speaker_label("user_123456789012", "   ") == "user_123456789012"

    def test_non_user_peers_keep_their_name(self):
        assert speaker_label("foxbot", "祥凤") == "祥凤(foxbot)"
        assert crud.short_peer_id("alice") == "alice"


class TestObservedDisplayNames:
    def test_collects_latest_non_empty_per_speaker(self):
        names = observed_display_names(
            [
                _message("user_1", "旧名字"),
                _message("user_2", ""),
                _message("user_1", "新名字"),
                _message("user_3", None),
            ]
        )
        # Last write wins: a card changed mid-batch resolves to the newest.
        assert names == {"user_1": "新名字"}

    def test_empty_for_a_session_without_cards(self):
        assert observed_display_names([_message("user_1", None)]) == {}


class TestDeriverPromptNaming:
    def test_subject_is_named_name_then_id(self):
        prompt = minimal_deriver_prompt(
            peer_id="user_123456789012", messages="<m/>", display_name="小白"
        )
        assert "小白(123456789012)" in prompt
        assert "Display name in this session: 小白" in prompt
        assert "Never the bare id, never the bare name" in prompt
        # The old blanket rule must be gone when a card is known.
        assert "Use the exact peer id from `Target peer:`" not in prompt

    def test_falls_back_to_the_old_rule_without_a_card(self):
        prompt = minimal_deriver_prompt(peer_id="user_123456789012", messages="<m/>")
        assert (
            "Use the exact peer id from `Target peer:` in final observations" in prompt
        )
        assert "Display name in this session" not in prompt

    def test_other_participants_reuse_their_labels(self):
        prompt = minimal_deriver_prompt(
            peer_id="user_1", messages="<m/>", display_name="甲"
        )
        assert "reuse those labels verbatim" in prompt


@pytest_asyncio.fixture
async def named_session(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> models.Session:
    workspace, _ = sample_data
    session = models.Session(name="g1", workspace_name=workspace.name)
    db_session.add(session)
    for peer_name in ("user_111", "user_222"):
        db_session.add(models.Peer(name=peer_name, workspace_name=workspace.name))
    await db_session.flush()
    for peer_name in ("user_111", "user_222"):
        db_session.add(
            models.SessionPeer(
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer_name,
                configuration={},
                internal_metadata={},
            )
        )
    await db_session.commit()
    await db_session.refresh(session)
    return session


class TestDisplayNameStorage:
    async def test_round_trip(
        self, db_session: AsyncSession, named_session: models.Session
    ):
        written = await crud.record_session_display_names(
            db_session,
            named_session.workspace_name,
            session_name=named_session.name,
            names={"user_111": "小白", "user_222": "  阿黄  "},
            seen_at="2026-10-10T00:00:00Z",
        )
        assert written == 2

        names = await crud.get_session_display_names(
            db_session, named_session.workspace_name, session_name=named_session.name
        )
        assert names == {"user_111": "小白", "user_222": "阿黄"}

    async def test_rewriting_the_same_names_writes_nothing(
        self, db_session: AsyncSession, named_session: models.Session
    ):
        first = await crud.record_session_display_names(
            db_session,
            named_session.workspace_name,
            session_name=named_session.name,
            names={"user_111": "小白"},
            seen_at="2026-10-10T00:00:00Z",
        )
        second = await crud.record_session_display_names(
            db_session,
            named_session.workspace_name,
            session_name=named_session.name,
            names={"user_111": "小白"},
            seen_at="2026-10-10T00:00:00Z",
        )
        assert (first, second) == (1, 0)

    async def test_a_changed_card_overwrites_only_the_session_row(
        self, db_session: AsyncSession, named_session: models.Session
    ):
        """The whole reason this lives on `session_peers` and not on the peer: one
        group's card must not clobber another's."""
        workspace_name = named_session.workspace_name
        await crud.record_session_display_names(
            db_session,
            workspace_name,
            session_name=named_session.name,
            names={"user_111": "第一个群的名字"},
            seen_at="2026-10-10T00:00:00Z",
        )
        await crud.record_session_display_names(
            db_session,
            workspace_name,
            session_name=named_session.name,
            names={"user_111": "改过的名字"},
            seen_at="2026-10-11T00:00:00Z",
        )

        names = await crud.get_session_display_names(
            db_session, workspace_name, session_name=named_session.name
        )
        assert names == {"user_111": "改过的名字"}

        peer = (
            await db_session.execute(
                select(models.Peer).where(
                    models.Peer.workspace_name == workspace_name,
                    models.Peer.name == "user_111",
                )
            )
        ).scalar_one()
        # The global nickname is untouched — that is the client's field, and cards must
        # not write into it.
        assert "display_name" not in (peer.h_metadata or {})
        assert "display_name" not in (peer.internal_metadata or {})

    async def test_missing_membership_row_is_skipped_not_created(
        self, db_session: AsyncSession, named_session: models.Session
    ):
        written = await crud.record_session_display_names(
            db_session,
            named_session.workspace_name,
            session_name=named_session.name,
            names={"user_999": "没有成员行"},
            seen_at="2026-10-10T00:00:00Z",
        )
        assert written == 0
        names = await crud.get_session_display_names(
            db_session, named_session.workspace_name, session_name=named_session.name
        )
        assert "user_999" not in names

    async def test_peers_filter(
        self, db_session: AsyncSession, named_session: models.Session
    ):
        await crud.record_session_display_names(
            db_session,
            named_session.workspace_name,
            session_name=named_session.name,
            names={"user_111": "小白", "user_222": "阿黄"},
            seen_at="2026-10-10T00:00:00Z",
        )
        only = await crud.get_session_display_names(
            db_session,
            named_session.workspace_name,
            session_name=named_session.name,
            peers=["user_222"],
        )
        assert only == {"user_222": "阿黄"}
        assert (
            await crud.get_session_display_names(
                db_session,
                named_session.workspace_name,
                session_name=named_session.name,
                peers=[],
            )
            == {}
        )


class TestStatsSurfacesNames:
    async def test_top_speakers_carry_the_group_card(
        self, db_session: AsyncSession, named_session: models.Session
    ):
        workspace_name = named_session.workspace_name
        for i in range(3):
            db_session.add(
                models.Message(
                    session_name=named_session.name,
                    peer_name="user_111",
                    content="hi",
                    workspace_name=workspace_name,
                    seq_in_session=i + 1,
                    token_count=2,
                )
            )
        db_session.add(
            models.Message(
                session_name=named_session.name,
                peer_name="user_222",
                content="hi",
                workspace_name=workspace_name,
                seq_in_session=4,
                token_count=2,
            )
        )
        await db_session.commit()
        await crud.record_session_display_names(
            db_session,
            workspace_name,
            session_name=named_session.name,
            names={"user_111": "小白", "user_222": "阿黄"},
            seen_at="2026-10-10T00:00:00Z",
        )

        stats = await crud.get_session_stats(
            db_session, workspace_name, session_name=named_session.name
        )
        assert stats is not None
        assert [
            (s.peer_name, s.display_name, s.message_count) for s in stats.top_speakers
        ] == [
            ("user_111", "小白", 3),
            ("user_222", "阿黄", 1),
        ]

        block = crud.stats_block(stats)
        assert "小白(111)：3 条" in block
        assert "阿黄(222)：1 条" in block

    async def test_speaker_without_a_card_says_so(
        self, db_session: AsyncSession, named_session: models.Session
    ):
        db_session.add(
            models.Message(
                session_name=named_session.name,
                peer_name="user_222",
                content="hi",
                workspace_name=named_session.workspace_name,
                seq_in_session=1,
                token_count=2,
            )
        )
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, named_session.workspace_name, session_name=named_session.name
        )
        assert stats is not None
        assert stats.top_speakers[0].display_name is None
        assert "（本群名未知）" in crud.stats_block(stats)
