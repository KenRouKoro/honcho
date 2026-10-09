"""场地描述（session profile）与零 LLM 统计。

设计依据 doc/memory_layer_decision.md §23、§25。四个要被守住的点：

- **数字由代码算**：成员/潜水/常驻不交给模型（§25.3）
- **潜水者只能来自名册**：Honcho 的 peer 是发言驱动建的，没发言的人在这里不存在；
  名册缺失时是 ``None`` 而不是 0（§23.4）
- **触发按 session 节流**，不挂 dream（一个群 N 个成员会让描述被写 N 遍，§25.2）
- **失败不外溢**：没原料、空输出、provider 报错都不能毁掉已有描述
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.dreamer.session_profile import (
    SESSION_PROFILE_SECTIONS,
    refresh_session_profile_if_needed,
    session_profile_prompt,
)

WS_KEY = "workspace"


def _llm_response(text: str = "## 这是什么地方\n- 一个大学群") -> MagicMock:
    response = MagicMock()
    response.content = text
    response.finish_reasons = ["stop"]
    return response


@pytest_asyncio.fixture
async def seeded_session(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> models.Session:
    """A session with 3 speakers at very different volumes, plus a name roster."""
    workspace, _peer = sample_data
    session = models.Session(
        name="group_1",
        workspace_name=workspace.name,
        h_metadata={
            "group_name": "武汉纺织大学宿舍群",
            "member_count": 10,
            "max_member_count": 500,
            "admins": ["user_admin"],
        },
    )
    db_session.add(session)
    # `messages` carries composite FKs to both `peers` and `sessions`, and SQLAlchemy
    # cannot infer an ordering for a composite FK target — create them explicitly.
    for name in ("user_regular", "user_occ1", "user_occ2", "user_admin", "foxbot"):
        db_session.add(models.Peer(name=name, workspace_name=workspace.name))
    await db_session.flush()

    # regular: 60 messages. occasional: 3 and 1 messages.
    plan = [("user_regular", 60), ("user_occ1", 3), ("user_occ2", 1)]
    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    seq = 0
    for name, count in plan:
        for _ in range(count):
            seq += 1
            db_session.add(
                models.Message(
                    session_name=session.name,
                    peer_name=name,
                    content="hi",
                    workspace_name=workspace.name,
                    seq_in_session=seq,
                    token_count=2,
                    created_at=base,
                )
            )
    await db_session.commit()
    await db_session.refresh(session)
    return session


async def _add_observations(
    db_session: AsyncSession,
    workspace_name: str,
    session_name: str,
    contents: list[str],
    *,
    observer: str = "user_regular",
    observed: str = "user_regular",
) -> None:
    # `documents` has a composite FK to `collections`, so the collection must exist.
    existing = (
        await db_session.execute(
            select(models.Collection).where(
                models.Collection.workspace_name == workspace_name,
                models.Collection.observer == observer,
                models.Collection.observed == observed,
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        db_session.add(
            models.Collection(
                observer=observer,
                observed=observed,
                workspace_name=workspace_name,
                internal_metadata={},
            )
        )
        await db_session.flush()

    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    for i, content in enumerate(contents):
        db_session.add(
            models.Document(
                content=content,
                level="explicit",
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                session_name=session_name,
                created_at=base + datetime.timedelta(days=i),
            )
        )
    await db_session.commit()


async def _add_display_names(
    db_session: AsyncSession,
    session: models.Session,
    names: dict[str, str],
) -> None:
    """Give members a group card, creating their membership row if it is missing.

    Tests are the only place that may create membership rows directly — the production
    writer is deliberately update-only (see `crud.record_session_display_names`).
    """
    for peer_name in names:
        existing = (
            await db_session.execute(
                select(models.SessionPeer).where(
                    models.SessionPeer.workspace_name == session.workspace_name,
                    models.SessionPeer.session_name == session.name,
                    models.SessionPeer.peer_name == peer_name,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            db_session.add(
                models.SessionPeer(
                    workspace_name=session.workspace_name,
                    session_name=session.name,
                    peer_name=peer_name,
                    configuration={},
                    internal_metadata={},
                )
            )
    await db_session.flush()
    await crud.record_session_display_names(
        db_session,
        session.workspace_name,
        session_name=session.name,
        names=names,
        seen_at="2026-10-10T00:00:00Z",
    )


class TestSessionStats:
    async def test_three_tiers_and_lurkers(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.group_name == "武汉纺织大学宿舍群"
        assert stats.message_count == 64
        assert stats.speaker_count == 3
        # 60 >= 50 counts as regular; 3 and 1 do not.
        assert stats.regular_count == 1
        assert stats.occasional_count == 2
        # 10 roster members - 3 who ever spoke = 7 who never did.
        assert stats.lurker_count == 7
        assert stats.admins == ["user_admin"]
        assert stats.regular_min_messages == 50

    async def test_lurkers_are_none_without_a_roster(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        """Honcho cannot know about people who never spoke, so guessing 0 would be
        a lie the prompt would then repeat."""
        seeded_session.h_metadata = {"group_name": "无名单群"}
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.member_count is None
        assert stats.lurker_count is None
        block = crud.stats_block(stats)
        assert "未知" in block
        assert "不要编造" in block

    async def test_stale_roster_never_yields_negative_lurkers(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        seeded_session.h_metadata = {"member_count": 2}  # roster smaller than speakers
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.lurker_count == 0

    async def test_missing_session_returns_none(self, db_session: AsyncSession):
        stats = await crud.get_session_stats(
            db_session, "no_such_workspace", session_name="nope"
        )
        assert stats is None


class TestRosterFacts:
    """`title` / `role` can only come from the platform's member list (§28.2).

    Honcho cannot call `get_group_member_list`, so these arrive with the client's roster
    and live in `Session.metadata` alongside `member_count`. What matters here is that
    they reach the *prompt* attached to a **name** — a roster of bare ids is something a
    description cannot quote, which is why the enriched form exists.
    """

    async def test_owner_roles_and_titles_reach_the_block(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        seeded_session.h_metadata = {
            "group_name": "武汉纺织大学宿舍群",
            "member_count": 10,
            "admins": ["user_admin", "user_regular"],
            "owner": "user_admin",
            "member_roles": {"user_admin": "owner", "user_regular": "admin"},
            "member_titles": {"user_regular": "守舍人"},
            "roster_synced_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        await db_session.commit()
        await _add_display_names(
            db_session,
            seeded_session,
            {"user_admin": "老王", "user_regular": "小白"},
        )

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.owner == "user_admin"
        assert [s.peer_name for s in stats.admin_details] == [
            "user_admin",
            "user_regular",
        ]
        assert [s.role for s in stats.admin_details] == ["owner", "admin"]

        block = crud.stats_block(stats)
        assert "老王(admin)（owner）" in block
        assert "小白(regular)·守舍人" in block

    async def test_title_falls_back_to_the_observed_fact(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        """A roster is optional; a `title` recorded on the membership row still counts."""
        row = (
            await db_session.execute(
                select(models.SessionPeer).where(
                    models.SessionPeer.workspace_name == seeded_session.workspace_name,
                    models.SessionPeer.session_name == seeded_session.name,
                    models.SessionPeer.peer_name == "user_regular",
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = models.SessionPeer(
                workspace_name=seeded_session.workspace_name,
                session_name=seeded_session.name,
                peer_name="user_regular",
                configuration={},
                internal_metadata={},
            )
            db_session.add(row)
        row.internal_metadata = {"display_name": "小白", "title": "守舍人"}
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.top_speakers[0].title == "守舍人"
        assert "·守舍人" in crud.stats_block(stats)

    async def test_malformed_roster_maps_are_ignored(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        seeded_session.h_metadata = {
            "member_count": 10,
            "owner": 12345,
            "member_roles": ["not", "a", "map"],
            "member_titles": {"user_regular": 42, "user_occ1": "  "},
        }
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.owner is None
        assert all(s.role is None for s in stats.top_speakers)
        assert all(s.title is None for s in stats.top_speakers)


class TestDuplicateDisplayNames:
    """A name is not an identity (§28.5).

    Two members can both be called 「摸鱼」. The label carries the stable id precisely so
    this is recoverable — but only if the reader is *told* that the id is load bearing.
    """

    async def test_two_members_sharing_a_card_are_flagged(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await _add_display_names(
            db_session,
            seeded_session,
            {
                "user_regular": "摸鱼",
                "user_occ1": "摸鱼",
                "user_occ2": "阿黄",
            },
        )

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.duplicate_display_names == ["摸鱼"]

        block = crud.stats_block(stats)
        assert "「摸鱼」" in block
        assert "提到时必须带上括号里的 id" in block

    async def test_unique_cards_produce_no_warning(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await _add_display_names(
            db_session,
            seeded_session,
            {"user_regular": "小白", "user_occ1": "阿黄"},
        )

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.duplicate_display_names == []
        assert "相同的群名片" not in crud.stats_block(stats)


class TestRosterStaleness:
    """A roster is a snapshot, not a fact (§28.3).

    `lurker_count` is `member_count − speaker_count`, so an outdated roster silently
    turns a lower bound into a number the description will assert. Age is therefore part
    of the payload, and "unknown" is a third state — not a synonym for "fresh".
    """

    async def test_fresh_roster_is_reported_as_fresh(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        synced = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=3)
        seeded_session.h_metadata = {
            "member_count": 10,
            "roster_synced_at": synced.isoformat(),
        }
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.roster_stale is False
        assert stats.roster_age_seconds is not None
        assert 2 * 3600 < stats.roster_age_seconds < 4 * 3600
        block = crud.stats_block(stats)
        assert "名册同步于 3 小时前" in block
        assert "⚠️" not in block

    async def test_stale_roster_is_marked_as_a_reference_only(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        synced = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=9)
        seeded_session.h_metadata = {
            "member_count": 10,
            "roster_synced_at": synced.isoformat(),
        }
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.roster_stale is True
        block = crud.stats_block(stats)
        assert "⚠️ 名册同步于 9 天前" in block
        assert "不要当成现状" in block

    async def test_missing_sync_time_is_unknown_not_fresh(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        seeded_session.h_metadata = {"member_count": 10}
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.roster_age_seconds is None
        assert stats.roster_stale is None
        assert "名册同步时间：未知" in crud.stats_block(stats)

    async def test_unparseable_sync_time_is_unknown_not_zero(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        seeded_session.h_metadata = {
            "member_count": 10,
            "roster_synced_at": "昨天下午",
        }
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.roster_synced_at is None
        assert stats.roster_age_seconds is None
        assert "名册同步时间：未知" in crud.stats_block(stats)

    async def test_naive_timestamp_is_read_as_utc(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        naive = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        seeded_session.h_metadata = {
            "member_count": 10,
            "roster_synced_at": naive.isoformat(),
        }
        await db_session.commit()

        stats = await crud.get_session_stats(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stats is not None
        assert stats.roster_age_seconds is not None
        assert stats.roster_age_seconds < 60


class TestSessionProfileStorage:
    async def test_round_trip(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        profile = schemas.SessionProfile(
            text="## 这是什么地方\n- 大学宿舍群",
            observation_count=5,
            speaker_count=3,
            member_count=10,
            lurker_count=7,
            oldest="2026-01-01",
            newest="2026-01-05",
            compressed_at="2026-01-06T00:00:00Z",
        )
        await crud.set_session_profile(
            db_session,
            seeded_session.workspace_name,
            profile,
            session_name=seeded_session.name,
        )

        stored = await crud.get_session_profile(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stored is not None
        assert stored.text == profile.text
        assert stored.lurker_count == 7
        # Coexists with summaries in the same JSONB column.
        await db_session.refresh(seeded_session)
        assert set(seeded_session.internal_metadata) == {"session_profile"}

    async def test_malformed_stored_value_is_ignored(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        seeded_session.internal_metadata = {"session_profile": "not a dict"}
        await db_session.commit()

        stored = await crud.get_session_profile(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stored is None


class TestRefreshSessionProfile:
    async def test_throttled_until_the_interval_elapses(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await _add_observations(
            db_session, seeded_session.workspace_name, seeded_session.name, ["a", "b"]
        )

        with patch(
            "src.dreamer.session_profile.honcho_llm_call", new=AsyncMock()
        ) as mock_call:
            result = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 199
            )

        assert result is None
        mock_call.assert_not_awaited()

    async def test_writes_profile_with_code_computed_numbers(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await _add_observations(
            db_session,
            seeded_session.workspace_name,
            seeded_session.name,
            ["likes tea", "has a cat"],
        )

        with patch(
            "src.dreamer.session_profile.honcho_llm_call",
            new=AsyncMock(return_value=_llm_response()),
        ) as mock_call:
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert profile.text.startswith("## 这是什么地方")
        assert profile.observation_count == 2
        # Numbers come from code, not from the model's completion.
        assert profile.speaker_count == 3
        assert profile.member_count == 10
        assert profile.lurker_count == 7
        assert profile.sample_messages == 64

        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert "<session_stats>" in prompt
        assert "纯潜水（从未发言）：7" in prompt
        assert "今天是 " in prompt

    async def test_dedupes_observations_duplicated_across_observers(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        """The deriver writes the same observation into every observer's collection
        (§16.4), so a session-scoped read must collapse those copies."""
        await _add_observations(
            db_session,
            seeded_session.workspace_name,
            seeded_session.name,
            ["only once"],
            observer="user_regular",
            observed="user_regular",
        )
        await _add_observations(
            db_session,
            seeded_session.workspace_name,
            seeded_session.name,
            ["only once"],
            observer="foxbot",
            observed="user_regular",
        )

        with patch(
            "src.dreamer.session_profile.honcho_llm_call",
            new=AsyncMock(return_value=_llm_response()),
        ) as mock_call:
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert profile.observation_count == 1
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert prompt.count("only once") == 1

    async def test_no_observations_skips_without_calling_the_llm(
        self, seeded_session: models.Session
    ):
        with patch(
            "src.dreamer.session_profile.honcho_llm_call", new=AsyncMock()
        ) as mock_call:
            result = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert result is None
        mock_call.assert_not_awaited()

    async def test_empty_completion_does_not_clobber(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await _add_observations(
            db_session, seeded_session.workspace_name, seeded_session.name, ["a"]
        )
        await crud.set_session_profile(
            db_session,
            seeded_session.workspace_name,
            schemas.SessionProfile(text="## 旧的\n- 有内容"),
            session_name=seeded_session.name,
        )

        with patch(
            "src.dreamer.session_profile.honcho_llm_call",
            new=AsyncMock(return_value=_llm_response(text="   ")),
        ):
            result = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert result is None
        stored = await crud.get_session_profile(
            db_session, seeded_session.workspace_name, session_name=seeded_session.name
        )
        assert stored is not None and stored.text == "## 旧的\n- 有内容"

    async def test_provider_error_is_contained(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await _add_observations(
            db_session, seeded_session.workspace_name, seeded_session.name, ["a"]
        )

        with patch(
            "src.dreamer.session_profile.honcho_llm_call",
            new=AsyncMock(side_effect=RuntimeError("provider exploded")),
        ):
            result = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert result is None

    async def test_truncated_window_carries_previous_forward(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await _add_observations(
            db_session,
            seeded_session.workspace_name,
            seeded_session.name,
            ["a", "b", "c", "d"],
        )
        await crud.set_session_profile(
            db_session,
            seeded_session.workspace_name,
            schemas.SessionProfile(
                text="## 旧\n- 很早的事",
                observation_count=10,
                oldest="2025-06-01",
            ),
            session_name=seeded_session.name,
        )

        with (
            patch(
                "src.dreamer.session_profile.settings.SESSION_PROFILE.MAX_INPUT_OBSERVATIONS",
                2,
            ),
            patch(
                "src.dreamer.session_profile.honcho_llm_call",
                new=AsyncMock(return_value=_llm_response()),
            ) as mock_call,
        ):
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert profile.truncated is True
        assert profile.oldest == "2025-06-01"
        assert profile.observation_count == 12
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert "<previous_profile>" in prompt
        assert "很早的事" in prompt

    async def test_disabled_short_circuits(self, seeded_session: models.Session):
        with (
            patch(
                "src.dreamer.session_profile.settings.SESSION_PROFILE.ENABLED", False
            ),
            patch(
                "src.dreamer.session_profile.honcho_llm_call", new=AsyncMock()
            ) as mock_call,
        ):
            result = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert result is None
        mock_call.assert_not_awaited()


class TestFirstBuildSampling:
    """The first build has no previous version to carry history forward (§28.6).

    A 42k-message group under a 400-observation guard, sampled from the newest window
    only, yields a "place description" covering roughly the last two days. So the first
    build samples the **whole history evenly** instead; from the second build on, the
    newest window plus the previous version is the better trade.
    """

    async def _seed_history(
        self, db_session: AsyncSession, session: models.Session, count: int
    ) -> None:
        await _add_observations(
            db_session,
            session.workspace_name,
            session.name,
            [f"observation-{i:03d}" for i in range(count)],
        )

    async def test_first_build_reaches_back_to_the_oldest_observation(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await self._seed_history(db_session, seeded_session, 40)

        with (
            patch(
                "src.dreamer.session_profile.settings.SESSION_PROFILE.MAX_INPUT_OBSERVATIONS",
                5,
            ),
            patch(
                "src.dreamer.session_profile.honcho_llm_call",
                new=AsyncMock(return_value=_llm_response()),
            ) as mock_call,
        ):
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        # Both ends of the history are in the sample...
        assert "observation-000" in prompt
        assert "observation-039" in prompt
        # ...and it is not just the newest window (which would be 035..039).
        assert "observation-035" not in prompt

    async def test_first_build_tells_the_model_it_is_a_sample(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await self._seed_history(db_session, seeded_session, 40)

        with (
            patch(
                "src.dreamer.session_profile.settings.SESSION_PROFILE.MAX_INPUT_OBSERVATIONS",
                5,
            ),
            patch(
                "src.dreamer.session_profile.honcho_llm_call",
                new=AsyncMock(return_value=_llm_response()),
            ) as mock_call,
        ):
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        # Sampled material was still not the whole history, so the caveat stands.
        assert profile.truncated is True
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert "整段历史等距抽样" in prompt
        # It must also say what *not* to conclude from a sample.
        assert "把某一条观察当成" in prompt
        # And it must not claim completeness — that is the failure mode being fixed.
        assert "这次给你的是**全量**原料" not in prompt

    async def test_second_build_uses_the_newest_window_and_carries_forward(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await self._seed_history(db_session, seeded_session, 40)
        await crud.set_session_profile(
            db_session,
            seeded_session.workspace_name,
            schemas.SessionProfile(text="## 旧\n- 最早的事", oldest="2025-01-01"),
            session_name=seeded_session.name,
        )

        with (
            patch(
                "src.dreamer.session_profile.settings.SESSION_PROFILE.MAX_INPUT_OBSERVATIONS",
                5,
            ),
            patch(
                "src.dreamer.session_profile.honcho_llm_call",
                new=AsyncMock(return_value=_llm_response()),
            ) as mock_call,
        ):
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        # Now recency density is worth more than span: the newest observations are in,
        # the oldest are not, and the older material survives through the profile.
        assert "observation-039" in prompt
        assert "observation-000" not in prompt
        assert "<previous_profile>" in prompt
        assert "最早的事" in prompt
        assert "整段历史等距抽样" not in prompt
        assert profile.oldest == "2025-01-01"

    async def test_small_history_is_not_called_a_sample(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        """Below the guard the material *is* the whole history; saying otherwise would
        make the model hedge about coverage it actually has."""
        await self._seed_history(db_session, seeded_session, 3)

        with patch(
            "src.dreamer.session_profile.honcho_llm_call",
            new=AsyncMock(return_value=_llm_response()),
        ) as mock_call:
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert profile.truncated is False
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert "全量" in prompt
        assert "整段历史等距抽样" not in prompt

    async def test_sampling_never_exceeds_the_guard(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        await self._seed_history(db_session, seeded_session, 60)

        with (
            patch(
                "src.dreamer.session_profile.settings.SESSION_PROFILE.MAX_INPUT_OBSERVATIONS",
                7,
            ),
            patch(
                "src.dreamer.session_profile.honcho_llm_call",
                new=AsyncMock(return_value=_llm_response()),
            ) as mock_call,
        ):
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert profile.observation_count == 7
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert 'count="7"' in prompt
        assert prompt.count("observation-0") == 7

    async def test_sampling_dedupes_across_observers(
        self, db_session: AsyncSession, seeded_session: models.Session
    ):
        """Every observer's collection holds a copy, so a sample must not spend half its
        budget on duplicates."""
        for observer, observed in (
            ("user_regular", "user_regular"),
            ("foxbot", "user_regular"),
        ):
            await _add_observations(
                db_session,
                seeded_session.workspace_name,
                seeded_session.name,
                [f"only-{i}" for i in range(10)],
                observer=observer,
                observed=observed,
            )

        with (
            patch(
                "src.dreamer.session_profile.settings.SESSION_PROFILE.MAX_INPUT_OBSERVATIONS",
                4,
            ),
            patch(
                "src.dreamer.session_profile.honcho_llm_call",
                new=AsyncMock(return_value=_llm_response()),
            ) as mock_call,
        ):
            profile = await refresh_session_profile_if_needed(
                seeded_session.workspace_name, seeded_session.name, 200
            )

        assert profile is not None
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert prompt.count("only-0") == 1
        assert profile.observation_count == 4


class TestSessionProfilePrompt:
    def test_fixed_sections_and_stats_block(self):
        prompt = session_profile_prompt(
            observations_text="[2026-01-01] x",
            observation_count=1,
            stats_text="<session_stats>\n名册总人数：10\n</session_stats>",
            max_tokens=1000,
            today="2026-01-02",
        )
        for name, _ in SESSION_PROFILE_SECTIONS:
            assert f"## {name}" in prompt
        assert "<session_stats>" in prompt
        assert "今天是 2026-01-02" in prompt
        # The prompt must forbid inventing a member count when the roster is absent.
        assert "不要假装群里只有出现过的人" in prompt

    def test_interpolated_blocks_are_flush_left(self):
        prompt = session_profile_prompt(
            observations_text="[2026-01-01] x",
            observation_count=1,
            stats_text="<session_stats>\n</session_stats>",
            max_tokens=1000,
            today="2026-01-02",
            previous_profile_text="## 旧\n- y",
        )
        assert prompt.startswith("你在为一个")
        assert "\n<previous_profile>\n## 旧\n- y\n</previous_profile>\n" in prompt
        assert f"\n## {SESSION_PROFILE_SECTIONS[0][0]}\n" in prompt


class TestContextRouteExposure:
    """`session_profile` / `session_stats` describe only this session, so unlike the
    peer card and peer profile they must survive an allowlist."""

    async def test_stats_and_profile_are_returned(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
        db_session: AsyncSession,
    ):
        workspace, peer = sample_data
        session_id = "group_route"
        client.post(
            f"/v3/workspaces/{workspace.name}/sessions",
            json={
                "id": session_id,
                "peers": {peer.name: {}},
                "metadata": {"member_count": 42, "group_name": "路由测试群"},
            },
        )
        await crud.set_session_profile(
            db_session,
            workspace.name,
            schemas.SessionProfile(text="## 这是什么地方\n- 测试", lurker_count=41),
            session_name=session_id,
        )

        url = (
            f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
            + f"?summary=false&peer_target={peer.name}&limit_to_session=true"
        )
        data = client.get(url).json()

        assert data["session_profile"]["text"].startswith("## 这是什么地方")
        assert data["session_stats"]["member_count"] == 42
        # Nobody has sent a message in this session yet, so every roster member is a
        # lurker. That is the honest answer — and the one Honcho could never derive
        # from its own data, since none of those 42 people has a peer row.
        assert data["session_stats"]["speaker_count"] == 0
        assert data["session_stats"]["lurker_count"] == 42
        assert data["session_profile"]["lurker_count"] == 41
        # The peer-scoped fields are still dropped under the allowlist.
        assert data["peer_card"] is None
        assert data["peer_profile"] is None
