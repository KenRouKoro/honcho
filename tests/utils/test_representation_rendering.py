"""How observations are rendered into prompts (doc/memory_layer_decision.md §22.3, §26).

Two defects this pins down:

- timestamps were printed as **unmarked UTC** while the observation body usually repeats
  the moment in the author's own zone (measured: they disagreed by exactly the zone
  offset on 92/100 lines), and cost 30.5% of the explicit content in tokens;
- `session_name` existed on every observation and was **never rendered**, so a
  cross-session representation — one peer appeared in 6 groups — presented facts with
  no provenance at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from src.config import settings
from src.utils.representation import (
    ExplicitObservation,
    InductiveObservation,
    Representation,
    format_timestamp,
    render_observation,
    time_note,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _explicit(
    session: str | None = "g1",
    *,
    when: datetime | None = None,
    content: str = "likes tea",
    obs_id: str = "abc",
) -> ExplicitObservation:
    return ExplicitObservation(
        id=obs_id,
        # Naive on purpose: this is what `_safe_datetime_from_metadata` hands over.
        created_at=when or datetime(2026, 9, 30, 3, 33),
        message_ids=[],
        session_name=session,
        content=content,
    )


class TestTimestampFormat:
    def test_naive_timestamps_are_read_as_utc_then_converted(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The old rendering printed the naive UTC value directly, so a line said
        03:33 while its own text said 11:33."""
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "Asia/Shanghai")
        assert format_timestamp(datetime(2026, 9, 30, 3, 33)) == "09-30 11:33"

    def test_aware_timestamps_are_converted_not_relabelled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "Asia/Shanghai")
        assert (
            format_timestamp(datetime(2026, 9, 30, 3, 33, tzinfo=UTC)) == "09-30 11:33"
        )

    def test_compact_drops_the_year_and_seconds(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "UTC")
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "compact")
        assert format_timestamp(datetime(2026, 9, 30, 3, 33, 44)) == "09-30 03:33"

    @pytest.mark.parametrize(
        ("fmt", "expected"),
        [("full", "2026-09-30 03:33"), ("date", "2026-09-30"), ("none", "")],
    )
    def test_other_formats(
        self, monkeypatch: pytest.MonkeyPatch, fmt: str, expected: str
    ):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "UTC")
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", fmt)
        assert format_timestamp(datetime(2026, 9, 30, 3, 33)) == expected

    def test_bad_timezone_is_rejected_at_config_time(self):
        from src.config import RepresentationSettings

        with pytest.raises(Exception):
            RepresentationSettings(TIMEZONE="Not/AZone")

    def test_time_note_names_the_zone_once(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "Asia/Shanghai")
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "compact")
        assert time_note() == " — times in Asia/Shanghai"

    def test_no_time_note_when_timestamps_are_off(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "none")
        assert time_note() == ""


class TestSessionProvenance:
    def test_single_session_gets_no_labels(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.REPRESENTATION, "INCLUDE_SESSION_PROVENANCE", True)
        rep = Representation(explicit=[_explicit("g1"), _explicit("g1")])
        assert rep.needs_session_labels() is False
        md = rep.format_as_markdown()
        assert "[g1]" not in md

    def test_multiple_sessions_get_labels(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.REPRESENTATION, "INCLUDE_SESSION_PROVENANCE", True)
        # Pin the zone: the assertions below spell out a literal timestamp, and a
        # developer's .env legitimately sets a real one.
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "UTC")
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "compact")
        rep = Representation(explicit=[_explicit("g1"), _explicit("g2")])
        assert rep.needs_session_labels() is True
        md = rep.format_as_markdown()
        assert "[09-30 03:33 · g1]" in md
        assert "[09-30 03:33 · g2]" in md

    def test_provenance_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            settings.REPRESENTATION, "INCLUDE_SESSION_PROVENANCE", False
        )
        rep = Representation(explicit=[_explicit("g1"), _explicit("g2")])
        assert rep.needs_session_labels() is False
        assert "[g1]" not in rep.format_as_markdown()

    def test_labels_appear_in_every_display_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.REPRESENTATION, "INCLUDE_SESSION_PROVENANCE", True)
        rep = Representation(
            explicit=[_explicit("g1"), _explicit("g2")],
            inductive=[
                InductiveObservation(
                    id="ind1",
                    created_at=datetime(2026, 9, 30, 3, 33),
                    message_ids=[],
                    session_name="g1",
                    conclusion="prefers tea",
                    sources=["likes tea"],
                )
            ],
        )
        assert "· g1]" in str(rep)
        assert "· g1]" in rep.str_with_ids()
        assert "· g1]" in rep.format_as_markdown()

    def test_missing_session_name_does_not_produce_an_empty_field(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.REPRESENTATION, "INCLUDE_SESSION_PROVENANCE", True)
        rep = Representation(explicit=[_explicit(None), _explicit("g2")])
        line = rep.format_as_markdown()
        assert "· ]" not in line


class TestRenderObservation:
    def test_field_order_is_id_then_time_then_session(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "UTC")
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "compact")
        assert (
            render_observation(
                created_at=datetime(2026, 9, 30, 3, 33),
                body="x",
                session_name="g1",
                include_id=True,
                observation_id="abc",
                include_session=True,
            )
            == "[id:abc · 09-30 03:33 · g1] x"
        )

    def test_no_fields_means_no_brackets(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "none")
        assert render_observation(created_at=datetime(2026, 9, 30), body="x") == "x"


class TestMarkdownShape:
    def test_timestamp_note_is_stated_once_per_section(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "Asia/Shanghai")
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "compact")
        rep = Representation(explicit=[_explicit("g1")])
        md = rep.format_as_markdown()
        assert md.startswith("## Explicit Observations — times in Asia/Shanghai")
        # Once, not per line.
        assert md.count("times in Asia/Shanghai") == 1

    def test_local_prefix_matches_the_local_time_written_in_the_body(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The whole point of §22.3(c): the two must not disagree by the zone offset."""
        monkeypatch.setattr(settings.REPRESENTATION, "TIMEZONE", "Asia/Shanghai")
        obs = _explicit(
            "g1",
            when=datetime(2026, 9, 30, 3, 33),
            content="在 2026-09-30 周三 11:33 说了什么",
        )
        rendered = str(obs)
        assert rendered.startswith("[09-30 11:33]")
        assert "11:33" in obs.content
        # The old rendering would have said 03:33 here.
        assert "03:33" not in rendered

    def test_include_ids_still_works_for_derived_levels(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.REPRESENTATION, "TIMESTAMP_FORMAT", "compact")
        monkeypatch.setattr(
            settings.REPRESENTATION, "INCLUDE_SESSION_PROVENANCE", False
        )
        rep = Representation(
            inductive=[
                InductiveObservation(
                    id="ind1",
                    created_at=datetime(2026, 9, 30, 3, 33),
                    message_ids=[],
                    session_name="g1",
                    conclusion="prefers tea",
                    sources=["likes tea"],
                )
            ]
        )
        md = rep.format_as_markdown(include_ids=True)
        assert "id:ind1" in md
