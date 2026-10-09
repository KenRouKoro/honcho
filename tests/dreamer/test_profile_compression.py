"""Long-term profile compression (``src/dreamer/profile.py``).

Covers the four properties the design actually rests on
(doc/memory_layer_decision.md §18–§19):

- input is the **whole explicit set**, not a per-query slice, and never the
  deductive/inductive layers
- the output is stored on the **observed peer's own** metadata — global per user,
  no observer dimension (so one compression serves every perspective)
- a **truncated** window carries the previous profile forward, because each run
  *replaces* the stored text rather than appending to it
- failure is contained: no observations, an empty completion, or a provider error
  must never clobber a good existing profile
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.representation import RepresentationManager
from src.dreamer.profile import PROFILE_SECTIONS, compress_profile, profile_prompt


@pytest_asyncio.fixture
async def profile_collection(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> models.Collection:
    workspace, peer = sample_data
    collection = models.Collection(
        observer=peer.name,
        observed=peer.name,
        workspace_name=workspace.name,
        internal_metadata={},
    )
    db_session.add(collection)
    await db_session.commit()
    await db_session.refresh(collection)
    return collection


async def _add_documents(
    db_session: AsyncSession,
    collection: models.Collection,
    contents: list[str],
    *,
    level: str = "explicit",
) -> None:
    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    for i, content in enumerate(contents):
        db_session.add(
            models.Document(
                content=content,
                level=level,
                workspace_name=collection.workspace_name,
                observer=collection.observer,
                observed=collection.observed,
                created_at=base + datetime.timedelta(days=i),
            )
        )
    await db_session.commit()


def _llm_response(text: str = "## 生活与处境\n- 在准备考试") -> MagicMock:
    response = MagicMock()
    response.content = text
    response.finish_reasons = ["stop"]
    return response


class TestGetAllExplicit:
    """The fetch helper is the load-bearing part of "stable input" (§19.3)."""

    async def test_returns_ascending_and_excludes_derived_levels(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        await _add_documents(
            db_session, profile_collection, ["first", "second", "third"]
        )
        await _add_documents(
            db_session, profile_collection, ["deduced"], level="deductive"
        )
        await _add_documents(
            db_session, profile_collection, ["induced"], level="inductive"
        )

        manager = RepresentationManager(
            profile_collection.workspace_name,
            observer=profile_collection.observer,
            observed=profile_collection.observed,
        )
        documents, truncated = await manager.get_all_explicit(db_session, 100)

        assert [d.content for d in documents] == ["first", "second", "third"]
        assert truncated is False

    async def test_truncation_keeps_the_newest_not_the_oldest(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        await _add_documents(db_session, profile_collection, ["a", "b", "c", "d"])

        manager = RepresentationManager(
            profile_collection.workspace_name,
            observer=profile_collection.observer,
            observed=profile_collection.observed,
        )
        documents, truncated = await manager.get_all_explicit(db_session, 2)

        assert truncated is True
        # Newest two, still in ascending order for the prompt.
        assert [d.content for d in documents] == ["c", "d"]


class TestCompressProfile:
    async def test_writes_profile_to_observed_peer(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        await _add_documents(db_session, profile_collection, ["likes tea", "has a cat"])

        with patch(
            "src.dreamer.profile.honcho_llm_call",
            new=AsyncMock(return_value=_llm_response()),
        ) as mock_call:
            profile = await compress_profile(
                profile_collection.workspace_name,
                profile_collection.observer,
                profile_collection.observed,
                run_id="run-1",
            )

        assert profile is not None
        assert profile.text.startswith("## 生活与处境")
        assert profile.observation_count == 2
        assert profile.oldest == "2026-01-01"
        assert profile.newest == "2026-01-02"
        assert profile.truncated is False
        assert profile.compressed_at
        mock_call.assert_awaited_once()

        stored = await _read_stored_profile(db_session, profile_collection)
        assert stored is not None
        assert stored.text == profile.text
        assert stored.observation_count == 2

    async def test_no_observations_skips_without_calling_llm(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        with patch("src.dreamer.profile.honcho_llm_call", new=AsyncMock()) as mock_call:
            profile = await compress_profile(
                profile_collection.workspace_name,
                profile_collection.observer,
                profile_collection.observed,
            )

        assert profile is None
        mock_call.assert_not_awaited()
        assert await _read_stored_profile(db_session, profile_collection) is None

    async def test_empty_completion_does_not_clobber_existing_profile(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        await _add_documents(db_session, profile_collection, ["likes tea"])
        await _seed_profile(db_session, profile_collection, text="## 旧的\n- 有内容")

        with patch(
            "src.dreamer.profile.honcho_llm_call",
            new=AsyncMock(return_value=_llm_response(text="   ")),
        ):
            profile = await compress_profile(
                profile_collection.workspace_name,
                profile_collection.observer,
                profile_collection.observed,
            )

        assert profile is None
        stored = await _read_stored_profile(db_session, profile_collection)
        assert stored is not None
        assert stored.text == "## 旧的\n- 有内容"

    async def test_provider_error_is_contained(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        await _add_documents(db_session, profile_collection, ["likes tea"])

        with patch(
            "src.dreamer.profile.honcho_llm_call",
            new=AsyncMock(side_effect=RuntimeError("provider exploded")),
        ):
            profile = await compress_profile(
                profile_collection.workspace_name,
                profile_collection.observer,
                profile_collection.observed,
            )

        assert profile is None, "a dream enhancement must never raise"

    async def test_truncated_window_carries_previous_profile_forward(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        """Every run replaces the stored text, so the older span lives only in the
        previous profile. Dropping it would lose that span permanently."""
        await _add_documents(db_session, profile_collection, ["a", "b", "c", "d"])
        await _seed_profile(
            db_session,
            profile_collection,
            text="## 生活与处境\n- 很早就存在的事",
            observation_count=10,
            oldest="2025-06-01",
        )

        with (
            patch("src.dreamer.profile.settings.PROFILE.MAX_INPUT_OBSERVATIONS", 2),
            patch(
                "src.dreamer.profile.honcho_llm_call",
                new=AsyncMock(return_value=_llm_response()),
            ) as mock_call,
        ):
            profile = await compress_profile(
                profile_collection.workspace_name,
                profile_collection.observer,
                profile_collection.observed,
            )

        assert profile is not None
        assert profile.truncated is True
        # Coverage starts at the carried profile's oldest, not at the window.
        assert profile.oldest == "2025-06-01"
        assert profile.observation_count == 12, "10 carried + 2 in the window"

        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert "<previous_profile>" in prompt
        assert "很早就存在的事" in prompt
        assert "[2026-01-03]" in prompt and "[2026-01-04]" in prompt

    async def test_untruncated_run_does_not_feed_previous_profile(
        self, db_session: AsyncSession, profile_collection: models.Collection
    ):
        """Guard against drift: a full-set run must rewrite from scratch."""
        await _add_documents(db_session, profile_collection, ["a", "b"])
        await _seed_profile(
            db_session,
            profile_collection,
            text="## 旧\n- 不该出现",
            observation_count=7,
        )

        with patch(
            "src.dreamer.profile.honcho_llm_call",
            new=AsyncMock(return_value=_llm_response()),
        ) as mock_call:
            profile = await compress_profile(
                profile_collection.workspace_name,
                profile_collection.observer,
                profile_collection.observed,
            )

        assert profile is not None
        assert profile.observation_count == 2, "full recompression resets the count"
        assert mock_call.await_args is not None
        prompt = mock_call.await_args.kwargs["prompt"]
        assert "<previous_profile>" not in prompt
        assert "不该出现" not in prompt


class TestProfilePrompt:
    def test_uses_the_fixed_section_set(self):
        prompt = profile_prompt(
            "x", observation_count=1, max_tokens=1000, today="2026-01-05"
        )
        for name, _ in PROFILE_SECTIONS:
            assert f"## {name}" in prompt

    def test_observation_dates_have_a_now_anchor(self):
        """Observation lines carry dates but no clock. Without `today` the model
        cannot tell a stale 下周考试 from a live one, and 待跟进 accumulates
        resolved threads forever."""
        prompt = profile_prompt(
            "[2026-01-01] x", observation_count=1, max_tokens=1000, today="2026-03-09"
        )
        assert "今天是 2026-03-09" in prompt
        assert "用「今天」判断时效" in prompt

    def test_interpolated_blocks_are_flush_left(self):
        """`cleandoc` must run before interpolation — see the comment in profile_prompt.

        Otherwise the interpolated multi-line blocks (section spec, observations,
        the carried profile) drag the common indent to zero and every template line
        keeps its source indentation.
        """
        prompt = profile_prompt(
            "[2026-01-01] x",
            observation_count=1,
            max_tokens=1000,
            today="2026-01-05",
            previous_profile_text="## 旧\n- y",
        )
        assert prompt.startswith("你是长期记忆的压缩器。")
        assert "\n<previous_profile>\n## 旧\n- y\n</previous_profile>\n" in prompt
        assert f"\n## {PROFILE_SECTIONS[0][0]}\n" in prompt
        assert "\n[2026-01-01] x\n" in prompt


async def _load_peer(
    db_session: AsyncSession, collection: models.Collection
) -> models.Peer:
    result = await db_session.execute(
        select(models.Peer).where(
            models.Peer.name == collection.observed,
            models.Peer.workspace_name == collection.workspace_name,
        )
    )
    peer = result.scalar_one()
    await db_session.refresh(peer)
    return peer


async def _read_stored_profile(
    db_session: AsyncSession, collection: models.Collection
) -> schemas.PeerProfile | None:
    peer = await _load_peer(db_session, collection)
    raw = peer.internal_metadata.get("profile")
    if raw is None:
        return None
    return schemas.PeerProfile.model_validate(raw)


async def _seed_profile(
    db_session: AsyncSession,
    collection: models.Collection,
    *,
    text: str,
    observation_count: int = 1,
    oldest: str = "2026-01-01",
) -> None:
    peer = await _load_peer(db_session, collection)
    peer.internal_metadata = {
        **peer.internal_metadata,
        "profile": {
            "text": text,
            "observation_count": observation_count,
            "oldest": oldest,
            "newest": "2026-01-01",
            "compressed_at": "2026-01-01T00:00:00Z",
            "truncated": False,
        },
    }
    await db_session.commit()
