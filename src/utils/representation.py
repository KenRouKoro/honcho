from collections.abc import Sequence
from datetime import UTC, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

from src import models
from src.config import settings
from src.utils.formatting import parse_datetime_iso

# Conclusion levels whose `session_name` stamp is trustworthy enough to scope on.
#
# Explicit conclusions come from the deriver over a single session's message
# batch, so their stamp is authoritative. Deductive/inductive conclusions are
# produced by the dreamer, which reads across *all* sessions (its discovery
# tools default to session_only=False) but stamps its output with one session —
# whichever holds the most recent explicit conclusion, see
# dreamer/dream_scheduler.py. Serving those under a session allowlist would leak
# conclusions synthesized from sessions outside it.
#
# ponytail: whole-level exclusion rather than per-conclusion provenance. The
# reasoning trees already link each conclusion to its premises, so the real fix
# is an authoritative source-session set per conclusion; until that exists this
# fails closed. Tracked in DEV-2201.
ALLOWLIST_SAFE_LEVELS = ("explicit",)


def allowlist_safe_levels(levels: list[str] | None) -> list[str]:
    """Narrow a level filter to those safe to serve under a session allowlist.

    Returns the intersection with :data:`ALLOWLIST_SAFE_LEVELS`; ``None`` means
    "no level filter requested" and yields the full safe set. An empty result
    means the caller asked only for levels we can't scope, and should receive
    nothing rather than unscoped conclusions.
    """
    if levels is None:
        return list(ALLOWLIST_SAFE_LEVELS)
    return [level for level in levels if level in ALLOWLIST_SAFE_LEVELS]


def _strip_microseconds_and_timezone(timestamp: datetime) -> datetime:
    """
    Remove microseconds and timezone info from a datetime for stable string formatting.
    """
    return timestamp.replace(microsecond=0, tzinfo=None)


# `compact` drops the year and the seconds. Measured on the reference deployment's
# real representation (87 explicit observations, ~4,000 content tokens): the old
# `[YYYY-MM-DD HH:MM:SS]` UTC prefix cost 1,218 tokens (30.5% of the content), while
# `[MM-DD HH:MM]` in the participants' own zone costs 783 (19.6%) — a 36% cut that
# keeps time-of-day, which is what "the group is up at 4am" reasoning actually needs.
_TIMESTAMP_FORMATS: dict[str, str] = {
    "compact": "%m-%d %H:%M",
    "full": "%Y-%m-%d %H:%M",
    "date": "%Y-%m-%d",
}


def format_timestamp(timestamp: datetime) -> str:
    """Render an observation timestamp for prompts (no brackets), per settings.

    Observation timestamps reach us **naive** — `_safe_datetime_from_metadata` strips
    the tzinfo — so a naive value is read as UTC and then converted. Getting this wrong
    is the bug this replaces: the old rendering printed UTC unmarked while the
    observation body usually repeated the same moment in the author's own zone, so the
    two disagreed by the zone offset (8 hours here).
    """
    if settings.REPRESENTATION.TIMESTAMP_FORMAT == "none":
        return ""
    aware = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)
    try:
        local = aware.astimezone(ZoneInfo(settings.REPRESENTATION.TIMEZONE))
    except Exception:  # pragma: no cover - TIMEZONE is validated at startup
        local = aware.astimezone(timezone.utc)
    fmt = _TIMESTAMP_FORMATS.get(
        settings.REPRESENTATION.TIMESTAMP_FORMAT, _TIMESTAMP_FORMATS["compact"]
    )
    return local.strftime(fmt)


def time_note() -> str:
    """A one-off note stating which zone observation times are in.

    Cheaper than a per-line zone marker: measured 1 token once, versus ~2 tokens on
    every one of 87 lines.
    """
    if settings.REPRESENTATION.TIMESTAMP_FORMAT == "none":
        return ""
    return f" — times in {settings.REPRESENTATION.TIMEZONE}"


def render_observation(
    *,
    created_at: datetime,
    body: str,
    session_name: str | None = None,
    include_id: bool = False,
    observation_id: str = "",
    include_session: bool = False,
) -> str:
    """Shared line renderer for all four observation levels.

    ``[MM-DD HH:MM] [group_1] body`` — the session appears only when asked for, and the
    caller asks only when the representation actually spans several sessions.
    """
    fields: list[str] = []
    if include_id and observation_id:
        fields.append(f"id:{observation_id}")
    formatted_time = format_timestamp(created_at)
    if formatted_time:
        fields.append(formatted_time)
    if include_session and session_name:
        fields.append(session_name)
    prefix = f"[{' · '.join(fields)}] " if fields else ""
    return f"{prefix}{body}"


def flatten_message_ids(
    message_ids: list[int] | list[list[int]] | list[tuple[int, int]],
) -> list[int]:
    """
    Flatten message_ids that may be in old tuple format or nested list format.

    This handles backwards compatibility with the old schema where message_ids
    was list[tuple[int, int]] representing ranges, and the new schema where
    it's list[int] representing individual message IDs.

    Args:
        message_ids: Either a flat list of ints, nested list, or list of tuples

    Returns:
        A flat list of unique message IDs, sorted

    Examples:
        [1, 2, 3] -> [1, 2, 3]
        [[1, 2], [3, 4]] -> [1, 2, 3, 4]
        [(105, 105)] -> [105]
        [[105, 105]] -> [105]
    """
    result: list[int] = []
    for item in message_ids:
        if isinstance(item, (list | tuple)):
            # Nested list or tuple - flatten it
            result.extend(item)
        else:
            # Already flat
            result.append(item)
    # Remove duplicates and sort
    return sorted(set(result))


class ObservationMetadata(BaseModel):
    id: str = Field(default="", description="Document ID for this observation")
    created_at: datetime
    message_ids: list[int]
    session_name: str | None = None


class ExplicitObservationBase(BaseModel):
    content: str = Field(description="The explicit observation")


class DeductiveObservationBase(BaseModel):
    source_ids: list[str] = Field(
        description="Document IDs of premise observations for tree traversal",
        default_factory=list,
    )
    premises: list[str] = Field(
        description="Human-readable premise text for display",
        default_factory=list,
    )
    conclusion: str = Field(description="The deductive conclusion")


class InductiveObservationBase(BaseModel):
    """Base model for inductive observations - patterns, generalizations, and personality insights."""

    source_ids: list[str] = Field(
        description="Document IDs of source observations for tree traversal",
        default_factory=list,
    )
    sources: list[str] = Field(
        description="Human-readable source text for display",
        default_factory=list,
    )
    pattern_type: str = Field(
        description="Type of pattern: 'preference', 'behavior', 'personality', 'tendency', 'correlation'",
        default="pattern",
    )
    conclusion: str = Field(description="The inductive generalization or pattern")
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', 'low'",
        default="medium",
    )


class ContradictionObservationBase(BaseModel):
    """Base model for contradiction observations - when user has made conflicting statements."""

    source_ids: list[str] = Field(
        description="Document IDs of the contradicting observations",
        default_factory=list,
    )
    sources: list[str] = Field(
        description="Human-readable text of the contradicting statements",
        default_factory=list,
    )
    content: str = Field(description="Description of the contradiction")


class PromptRepresentation(BaseModel):
    """
    The representation format that is used when getting structured output from an LLM.
    """

    explicit: list[ExplicitObservationBase] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog named Rover']",
        default_factory=list,
    )

    @field_validator("explicit", mode="before")
    @classmethod
    def convert_none_to_empty_list(cls, v: Any) -> Any:
        """Convert None to empty list - handles LLMs returning null instead of []."""
        if v is None:
            return []
        return v


class ExplicitObservation(ExplicitObservationBase, ObservationMetadata):
    """Explicit observation with content and metadata."""

    def render(self, *, include_id: bool = False, include_session: bool = False) -> str:
        return render_observation(
            created_at=self.created_at,
            body=self.content,
            session_name=self.session_name,
            include_id=include_id,
            observation_id=self.id,
            include_session=include_session,
        )

    def __str__(self) -> str:
        return self.render()

    def str_with_id(self) -> str:
        """Format with ID prefix for use by agents that need to reference observations."""
        return self.render(include_id=True)

    def __hash__(self) -> int:
        """
        Make ExplicitObservation hashable for use in sets.
        """
        return hash((self.content, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for ExplicitObservation objects.
        Two observations are equal if all their fields match.
        """
        if not isinstance(other, ExplicitObservation):
            return False
        return (
            self.content == other.content
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class DeductiveObservation(DeductiveObservationBase, ObservationMetadata):
    """Deductive observation with multiple premises and one conclusion, plus metadata."""

    def render(self, *, include_id: bool = False, include_session: bool = False) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        head = render_observation(
            created_at=self.created_at,
            body=self.conclusion,
            session_name=self.session_name,
            include_id=include_id,
            observation_id=self.id,
            include_session=include_session,
        )
        return f"{head}\n{premises_text}"

    def __str__(self) -> str:
        return self.render()

    def str_with_id(self) -> str:
        """Format with ID prefix for use by agents that need to reference observations."""
        return self.render(include_id=True)

    def str_no_timestamps(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{self.conclusion}\n{premises_text}"

    def __hash__(self) -> int:
        """
        Make DeductiveObservation hashable for use in sets. NOTE: premises are not included in the hash.
        """
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for DeductiveObservation objects.
        Two observations are equal if all their fields match -- NOTE: premises are not included in the equality check.
        """
        if not isinstance(other, DeductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class InductiveObservation(InductiveObservationBase, ObservationMetadata):
    """Inductive observation with sources, pattern type, and confidence, plus metadata."""

    def render(self, *, include_id: bool = False, include_session: bool = False) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        head = render_observation(
            created_at=self.created_at,
            body=f"[{self.confidence}] {self.conclusion}",
            session_name=self.session_name,
            include_id=include_id,
            observation_id=self.id,
            include_session=include_session,
        )
        return f"{head}{sources_text}"

    def __str__(self) -> str:
        return self.render()

    def str_with_id(self) -> str:
        """Format with ID prefix for use by agents that need to reference observations."""
        return self.render(include_id=True)

    def str_no_timestamps(self) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"[{self.confidence}] {self.conclusion}{sources_text}"

    def __hash__(self) -> int:
        """Make InductiveObservation hashable for use in sets."""
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """Define equality for InductiveObservation objects."""
        if not isinstance(other, InductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class ContradictionObservation(ContradictionObservationBase, ObservationMetadata):
    """Contradiction observation - notes when user has made conflicting statements, plus metadata."""

    def render(self, *, include_id: bool = False, include_session: bool = False) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        head = render_observation(
            created_at=self.created_at,
            body=f"CONTRADICTION: {self.content}",
            session_name=self.session_name,
            include_id=include_id,
            observation_id=self.id,
            include_session=include_session,
        )
        return f"{head}{sources_text}"

    def __str__(self) -> str:
        return self.render()

    def str_with_id(self) -> str:
        """Format with ID prefix for use by agents that need to reference observations."""
        return self.render(include_id=True)

    def str_no_timestamps(self) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"CONTRADICTION: {self.content}{sources_text}"

    def __hash__(self) -> int:
        """Make ContradictionObservation hashable for use in sets."""
        return hash((self.content, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """Define equality for ContradictionObservation objects."""
        if not isinstance(other, ContradictionObservation):
            return False
        return (
            self.content == other.content
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class Representation(BaseModel):
    """
    A Representation is a traversable and diffable map of observations.
    At the base, we have a list of explicit observations, derived from a peer's messages.

    From there, deductive observations can be made by establishing logical relationships between explicit observations.

    In the future, we can add more levels of reasoning on top of these.

    All of a peer's observations are stored as documents in a collection. These documents can be queried in various ways
    to produce this Representation object.

    Additionally, a "working representation" is a version of this data structure representing the most recent observations
    within a single session.

    A representation can have a maximum number of observations, which is applied individually to each level of reasoning.
    If a maximum is set, observations are added and removed in FIFO order.
    """

    explicit: list[ExplicitObservation] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog']",
        default_factory=list,
    )
    deductive: list[DeductiveObservation] = Field(
        description="Conclusions that MUST be true given explicit facts and premises - strict logical necessities. Each deduction should have premises and a single conclusion.",
        default_factory=list,
    )
    inductive: list[InductiveObservation] = Field(
        description="Patterns, generalizations, and personality insights inferred from multiple observations. Higher-level reasoning created by the Dreamer agent.",
        default_factory=list,
    )
    contradiction: list[ContradictionObservation] = Field(
        description="Conflicting statements made by the user that need clarification. The dialectic agent should surface these when relevant.",
        default_factory=list,
    )

    def is_empty(self) -> bool:
        """
        Check if the representation is empty.
        """
        return (
            len(self.explicit) == 0
            and len(self.deductive) == 0
            and len(self.inductive) == 0
            and len(self.contradiction) == 0
        )

    def len(self) -> int:
        """
        Return the total number of observations in the representation.
        """
        return (
            len(self.explicit)
            + len(self.deductive)
            + len(self.inductive)
            + len(self.contradiction)
        )

    def diff_representation(self, other: "Representation") -> "Representation":
        """
        Given this and another representation, return a new representation with only observations that are unique to the other.
        Note that this only removes literal duplicates, not semantically equivalent ones.
        """
        diff = Representation()
        diff.explicit = [o for o in other.explicit if o not in self.explicit]
        diff.deductive = [o for o in other.deductive if o not in self.deductive]
        diff.inductive = [o for o in other.inductive if o not in self.inductive]
        diff.contradiction = [
            o for o in other.contradiction if o not in self.contradiction
        ]
        return diff

    def merge_representation(
        self, other: "Representation", max_observations: int | None = None
    ):
        """
        Merge another representation object into this one.
        This will automatically deduplicate explicit, deductive, inductive, and contradiction observations.
        This *preserves order* of observations so that they retain FIFO order.

        NOTE: observations with the *same* timestamp will not have order preserved.
        That's fine though, because they are from the same timestamp...
        """
        # removing duplicates by going list->set->list
        self.explicit = list(set(self.explicit + other.explicit))
        self.deductive = list(set(self.deductive + other.deductive))
        self.inductive = list(set(self.inductive + other.inductive))
        self.contradiction = list(set(self.contradiction + other.contradiction))
        # sort by created_at
        self.explicit.sort(key=lambda x: x.created_at)
        self.deductive.sort(key=lambda x: x.created_at)
        self.inductive.sort(key=lambda x: x.created_at)
        self.contradiction.sort(key=lambda x: x.created_at)

        if max_observations:
            self.explicit = self.explicit[-max_observations:]
            self.deductive = self.deductive[-max_observations:]
            self.inductive = self.inductive[-max_observations:]
            self.contradiction = self.contradiction[-max_observations:]

    def _sessions_spanned(self) -> int:
        """How many distinct sessions contributed to this representation."""
        return len(
            {
                obs.session_name
                for obs in (
                    self.explicit + self.deductive + self.inductive + self.contradiction
                )
                if obs.session_name
            }
        )

    def needs_session_labels(self) -> bool:
        """Whether lines should carry their session of origin.

        Only when the representation actually mixes sessions, and only when enabled: a
        single-session view (the common case, and the only one `limit_to_session=true`
        produces) gains nothing from a label it would repeat on every line. Measured cost
        when it does apply: ~5 tokens per explicit line, i.e. ~11% of the content — paid
        only for the case where provenance is the difference between a usable and a
        misleading fact set.
        """
        return (
            settings.REPRESENTATION.INCLUDE_SESSION_PROVENANCE
            and self._sessions_spanned() > 1
        )

    def __str__(self) -> str:
        """
        Format representation into a clean, readable string for LLM prompts.
        NOTE: timestamps are rendered by `format_timestamp` (configured zone + format).

        Returns:
            Formatted string with clear sections and bullet points including temporal metadata
            Example:
            EXPLICIT:
            1. [01-01 12:00] The user has a dog named Rover
            2. [01-01 12:01] The user's dog is 5 years old
            3. [01-01 12:05] The user is 25 years old
            DEDUCTIVE:
            1. [01-01 12:01] Rover is 5 years old
                - The user has a dog named Rover
                - The user's dog is 5 years old

        """

        parts: list[str] = []
        include_session = self.needs_session_labels()

        parts.append("EXPLICIT:\n")
        for i, observation in enumerate(self.explicit, 1):
            parts.append(f"{i}. {observation.render(include_session=include_session)}")
        parts.append("")

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(f"{i}. {observation.render(include_session=include_session)}")
        parts.append("")

        parts.append("INDUCTIVE:\n")
        for i, observation in enumerate(self.inductive, 1):
            parts.append(f"{i}. {observation.render(include_session=include_session)}")
        parts.append("")

        parts.append("CONTRADICTION:\n")
        for i, observation in enumerate(self.contradiction, 1):
            parts.append(f"{i}. {observation.render(include_session=include_session)}")
        parts.append("")

        return "\n".join(parts)

    def str_with_ids(self) -> str:
        """
        Format representation with observation IDs for agents that need to reference/delete observations.

        Returns:
            Formatted string with IDs included
            Example:
            EXPLICIT:
            1. [id:abc123 · 01-01 12:00] The user has a dog named Rover
            2. [id:def456 · 01-01 12:01] The user's dog is 5 years old
            DEDUCTIVE:
            1. [id:ghi789 · 01-01 12:01] Rover is 5 years old
                - The user has a dog named Rover
                - The user's dog is 5 years old
            INDUCTIVE:
            1. [id:jkl012 · 01-01 12:05] [high] User tends to be methodical
                - id:abc123
                - id:def456
        """
        parts: list[str] = []
        include_session = self.needs_session_labels()

        parts.append("EXPLICIT:\n")
        for i, observation in enumerate(self.explicit, 1):
            parts.append(
                f"{i}. {observation.render(include_id=True, include_session=include_session)}"
            )
        parts.append("")

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(
                f"{i}. {observation.render(include_id=True, include_session=include_session)}"
            )
        parts.append("")

        parts.append("INDUCTIVE:\n")
        for i, observation in enumerate(self.inductive, 1):
            parts.append(
                f"{i}. {observation.render(include_id=True, include_session=include_session)}"
            )
        parts.append("")

        parts.append("CONTRADICTION:\n")
        for i, observation in enumerate(self.contradiction, 1):
            parts.append(
                f"{i}. {observation.render(include_id=True, include_session=include_session)}"
            )
        parts.append("")

        return "\n".join(parts)

    def str_no_timestamps(self) -> str:
        """
        Format representation into a clean, readable string for LLM prompts... but without timestamps.

        Returns:
            Formatted string with clear sections and bullet points including temporal metadata
            Example:
            EXPLICIT:
            1. The user has a dog named Rover
            2. The user's dog is 5 years old
            3. The user is 25 years old
            DEDUCTIVE:
            1. Rover is 5 years old
                - The user has a dog named Rover
                - The user's dog is 5 years old
            INDUCTIVE:
            1. [high] User tends to be methodical
                - id:abc123
                - id:def456

        """
        parts: list[str] = []

        parts.append("EXPLICIT:\n")
        for i, observation in enumerate(self.explicit, 1):
            parts.append(f"{i}. {observation.content}")
        parts.append("")

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(f"{i}. {observation.str_no_timestamps()}")
        parts.append("")

        parts.append("INDUCTIVE:\n")
        for i, observation in enumerate(self.inductive, 1):
            parts.append(f"{i}. {observation.str_no_timestamps()}")
        parts.append("")

        parts.append("CONTRADICTION:\n")
        for i, observation in enumerate(self.contradiction, 1):
            parts.append(f"{i}. {observation.str_no_timestamps()}")
        parts.append("")

        return "\n".join(parts)

    def format_as_markdown(self, include_ids: bool = False) -> str:
        """
        Format a Representation object as markdown.

        Timestamps come from `format_timestamp` (zone and shape from
        `settings.REPRESENTATION`). Session-of-origin labels appear only when the
        representation actually spans more than one session — a single-session view
        gains nothing from a label it would repeat on every line.

        Args:
            include_ids: If True, include observation IDs for use with get_reasoning_chain

        Returns:
            Formatted markdown string
        """

        parts: list[str] = []
        include_session = self.needs_session_labels()
        note = time_note()

        # Add explicit observations
        if self.explicit:
            parts.append(f"## Explicit Observations{note}\n")
            for obs in self.explicit:
                # Don't need IDs for explicit as these are the lowest level of reasoning.
                parts.append(f"{obs.render(include_session=include_session)}")
            parts.append("")

        # Add deductive observations
        if self.deductive:
            parts.append(f"## Deductive Observations{note}\n")
            for obs in self.deductive:
                head = render_observation(
                    created_at=obs.created_at,
                    body=obs.conclusion,
                    session_name=obs.session_name,
                    include_id=include_ids,
                    observation_id=obs.id,
                    include_session=include_session,
                )
                parts.append(head)
                if obs.premises:
                    parts.append("   Premises:")
                    for premise in obs.premises:
                        parts.append(f"   - {premise}")
                parts.append("")
            parts.append("")

        # Add inductive observations
        if self.inductive:
            parts.append(f"## Inductive Observations{note}\n")
            for obs in self.inductive:
                fields: list[str] = []
                if include_ids and obs.id:
                    fields.append(f"id:{obs.id}")
                formatted_time = format_timestamp(obs.created_at)
                if formatted_time:
                    fields.append(formatted_time)
                if include_session and obs.session_name:
                    fields.append(obs.session_name)
                prefix = f"[{' · '.join(fields)}] " if fields else ""
                parts.append(
                    f"{prefix} **Pattern** [{obs.confidence}]: {obs.conclusion}"
                )
                if obs.pattern_type:
                    parts.append(f"   **Type**: {obs.pattern_type}")
                if obs.sources:
                    parts.append("   **Sources**:")
                    for source in obs.sources[:5]:
                        parts.append(f"   - {source}")
                    if len(obs.sources) > 5:
                        parts.append(f"   - ... and {len(obs.sources) - 5} more")
                parts.append("")
            parts.append("")

        # Add contradiction observations
        if self.contradiction:
            parts.append(f"## Contradictions{note}\n")
            for obs in self.contradiction:
                fields = []
                if include_ids and obs.id:
                    fields.append(f"id:{obs.id}")
                formatted_time = format_timestamp(obs.created_at)
                if formatted_time:
                    fields.append(formatted_time)
                if include_session and obs.session_name:
                    fields.append(obs.session_name)
                prefix = f"[{' · '.join(fields)}] " if fields else ""
                parts.append(f"{prefix} **CONTRADICTION**: {obs.content}")
                if obs.sources:
                    parts.append("   **Conflicting statements**:")
                    for source in obs.sources:
                        parts.append(f"   - {source}")
                parts.append("")
            parts.append("")

        return "\n".join(parts)

    @classmethod
    def from_documents(cls, documents: Sequence[models.Document]) -> "Representation":
        return cls(
            explicit=[
                ExplicitObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    content=doc.content,
                    message_ids=flatten_message_ids(
                        doc.internal_metadata.get("message_ids", [])
                    ),
                    session_name=doc.session_name,
                )
                for doc in documents
                if doc.level == "explicit"
            ],
            deductive=[
                DeductiveObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=flatten_message_ids(
                        doc.internal_metadata.get("message_ids", [])
                    ),
                    session_name=doc.session_name,
                    # Support both top-level and metadata locations for backward compatibility
                    source_ids=doc.source_ids
                    or doc.internal_metadata.get("premise_ids", []),
                    premises=doc.internal_metadata.get("premises", []),
                )
                for doc in documents
                if doc.level == "deductive"
            ],
            inductive=[
                InductiveObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", []),
                    session_name=doc.session_name,
                    # Support both top-level and metadata locations for backward compatibility
                    source_ids=doc.source_ids
                    or doc.internal_metadata.get("source_ids", []),
                    sources=doc.internal_metadata.get("sources", []),
                    pattern_type=doc.internal_metadata.get("pattern_type", "pattern"),
                    confidence=doc.internal_metadata.get("confidence", "medium"),
                )
                for doc in documents
                if doc.level == "inductive"
            ],
            contradiction=[
                ContradictionObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    content=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", []),
                    session_name=doc.session_name,
                    # Support both top-level and metadata locations for backward compatibility
                    source_ids=doc.source_ids
                    or doc.internal_metadata.get("source_ids", []),
                    sources=doc.internal_metadata.get("sources", []),
                )
                for doc in documents
                if doc.level == "contradiction"
            ],
        )

    @classmethod
    def from_prompt_representation(
        cls,
        prompt_representation: "PromptRepresentation",
        message_ids: list[int],
        session_name: str,
        created_at: datetime,
    ) -> "Representation":
        """Convert PromptRepresentation to Representation."""
        return cls(
            explicit=[
                ExplicitObservation(
                    content=e.content,
                    created_at=created_at,
                    message_ids=message_ids,
                    session_name=session_name,
                )
                for e in prompt_representation.explicit
            ],
            deductive=[],
            inductive=[],
        )


def _safe_datetime_from_metadata(
    internal_metadata: dict[str, Any], fallback_datetime: datetime
) -> datetime:
    message_created_at = internal_metadata.get("message_created_at")
    if message_created_at is None:
        return _strip_microseconds_and_timezone(fallback_datetime)

    if isinstance(message_created_at, str):
        try:
            return _strip_microseconds_and_timezone(
                parse_datetime_iso(message_created_at)
            )
        except ValueError:
            return _strip_microseconds_and_timezone(fallback_datetime)

    if isinstance(message_created_at, datetime):
        return _strip_microseconds_and_timezone(message_created_at)
    return _strip_microseconds_and_timezone(fallback_datetime)
