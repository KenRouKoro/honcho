from typing import Any, cast

from src.dreamer.specialists import DeductionSpecialist
from src.utils.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    TOOLS,
)


def _observation_items_schema(tool_key: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        TOOLS[tool_key]["input_schema"]["properties"]["observations"]["items"],
    )


def test_generic_create_observations_schema_has_level_specific_requirements() -> None:
    items = _observation_items_schema("create_observations")

    assert items["additionalProperties"] is False

    level_requirements = {
        condition["if"]["properties"]["level"]["const"]: condition["then"]["required"]
        for condition in cast(list[dict[str, Any]], items["allOf"])
    }

    assert level_requirements["deductive"] == ["source_ids", "premises"]
    assert level_requirements["inductive"] == [
        "source_ids",
        "sources",
        "pattern_type",
        "confidence",
    ]
    assert level_requirements["contradiction"] == ["source_ids", "sources"]


def test_deductive_specialist_tool_requires_evidence_fields() -> None:
    items = _observation_items_schema("create_observations_deductive")

    assert TOOLS["create_observations_deductive"]["name"] == (
        "create_observations_deductive"
    )
    assert items["required"] == ["content", "source_ids", "premises"]
    assert items["properties"]["source_ids"]["minItems"] == 1
    assert items["properties"]["premises"]["minItems"] == 1


def test_inductive_specialist_tool_requires_pattern_fields() -> None:
    items = _observation_items_schema("create_observations_inductive")

    assert TOOLS["create_observations_inductive"]["name"] == (
        "create_observations_inductive"
    )
    assert items["required"] == [
        "content",
        "source_ids",
        "sources",
        "pattern_type",
        "confidence",
    ]
    assert items["properties"]["source_ids"]["minItems"] == 2
    assert items["properties"]["sources"]["minItems"] == 2


def test_contradiction_specialist_tool_requires_two_sides() -> None:
    """`contradiction` used to be a dead level: zero rows workspace-wide, no write path.

    The deduction prompt had always asked for contradictions, but its only creation
    tool hard-coded `level="deductive"`. This tool is that missing path, and it must
    demand *both* sides of the conflict — a contradiction with one source is just an
    assertion (doc/memory_layer_decision.md §28.7).
    """
    items = _observation_items_schema("create_observations_contradiction")

    assert TOOLS["create_observations_contradiction"]["name"] == (
        "create_observations_contradiction"
    )
    assert items["required"] == ["content", "source_ids", "sources"]
    assert items["properties"]["source_ids"]["minItems"] == 2
    assert items["properties"]["sources"]["minItems"] == 2
    assert items["additionalProperties"] is False
    # No `level` field: the level is the tool, so it cannot be got wrong.
    assert "level" not in items["properties"]


def test_dreamer_specialists_use_level_specific_creation_tools() -> None:
    deduction_tool_names = {tool["name"] for tool in DEDUCTION_SPECIALIST_TOOLS}
    induction_tool_names = {tool["name"] for tool in INDUCTION_SPECIALIST_TOOLS}

    assert "create_observations_deductive" in deduction_tool_names
    assert "create_observations_contradiction" in deduction_tool_names
    assert "create_observations_inductive" in induction_tool_names
    assert "create_observations" not in deduction_tool_names
    assert "create_observations" not in induction_tool_names
    # Conflicts are judged against what is already known, so they belong to the
    # deduction phase — not to induction, which looks for patterns across many facts.
    assert "create_observations_contradiction" not in induction_tool_names


def test_contradiction_handler_forces_its_level() -> None:
    """A tool whose name encodes the level must not let the model choose it."""
    from src.utils import agent_tools

    handlers = agent_tools._TOOL_HANDLERS  # pyright: ignore[reportPrivateUsage]
    assert handlers["create_observations_contradiction"] is (
        agent_tools._handle_create_observations_contradiction  # pyright: ignore[reportPrivateUsage]
    )


def test_deduction_prompt_points_at_the_contradiction_tool() -> None:
    prompt = DeductionSpecialist().build_system_prompt("bob")
    assert "create_observations_contradiction" in prompt
    # ...and keeps the update/conflict distinction, so the level does not fill up with
    # "the value changed" rows, which are knowledge updates instead.
    assert "knowledge update" in prompt.lower()
