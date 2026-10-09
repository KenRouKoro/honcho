"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def _normalized_custom_instructions(custom_instructions: str | None) -> str | None:
    """Return stripped custom instructions, if any."""
    if custom_instructions is None:
        return None

    normalized = custom_instructions.strip()
    return normalized or None


def _custom_instructions_section(custom_instructions: str | None) -> str:
    """Render optional custom instructions for the deriver prompt."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return ""

    return c(
        f"""
        CUSTOM INSTRUCTIONS:
        These instructions apply to the target peer identified below.
        {normalized_custom_instructions}
        """
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    custom_instructions: str | None = None,
    display_name: str | None = None,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: All messages in the range (interleaving messages and new turns combined).
        custom_instructions: Optional per-peer guidance.
        display_name: The peer's name **in this session** (the group card). When given,
            observations name the subject ``name(id)`` instead of the bare id — see the
            RULES below and doc/memory_layer_decision.md §27.

    Returns:
        Formatted prompt string for observation extraction.
    """
    custom_instructions_section = _custom_instructions_section(custom_instructions)
    subject_rule = _subject_naming_rule(peer_id, display_name)
    return c(
        f"""
Analyze messages to extract **explicit atomic facts** about the target peer.

[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
{subject_rule}
- Properly attribute observations to the correct subject: if it is about the target peer, use the target peer's label as the subject. If the target peer is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.
- Extract ALL observations from the target peer's messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

<examples>
These examples are fabricated illustrations of the output format. Never emit a conclusion for which content comes from these examples. Every conclusion must be supported by the <messages> block only.

EXAMPLES (using `alice` as the target peer id):
- EXPLICIT: "I just turned 25" → "alice is 25 years old"
- EXPLICIT: "I took my dog for a walk in NYC" → "alice has a dog", "alice walked her dog in NYC"
- EXPLICIT: "I've lived in NYC for six years" → "alice lives in NYC", "alice has lived in NYC for six years"
</examples>

{custom_instructions_section}

Target peer:
{peer_id}
{_display_name_line(display_name)}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


def _display_name_line(display_name: str | None) -> str:
    if not display_name:
        return ""
    return f"Display name in this session: {display_name}"


def _subject_naming_rule(peer_id: str, display_name: str | None) -> str:
    """How the target peer must be named inside observation text.

    Names are readable but not unique and they change — on the reference deployment the
    bot had five different cards in one group, and the global nickname disagreed with
    the card on 5 of 7 (peer, session) pairs. Ids are unique but unreadable, and the
    subject of every observation is the one entity the model has to reason *about*.
    So: name first, id in parentheses, always. See doc/memory_layer_decision.md §27.
    """
    short_id = peer_id[len("user_") :] if peer_id.startswith("user_") else peer_id
    if not display_name:
        return (
            "- Use the exact peer id from `Target peer:` in final observations, not the"
            ' phrase "the target peer".'
        )
    return (
        f"- Name the target peer exactly `{display_name}({short_id})` in every observation"
        f" — the display name as used in this session, then the stable id in parentheses."
        f" Never the bare id, never the bare name.\n"
        f"- Other participants are already labelled `name(id)` in the messages;"
        f" reuse those labels verbatim when you mention them.\n"
        f"- The parenthesised id is the only stable identifier: display names change and"
        f" can collide between people."
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate the static minimal deriver prompt without custom instructions."""
    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=None,
    )
    return estimate_tokens(prompt)


def estimate_deriver_prompt_tokens(custom_instructions: str | None) -> int:
    """Estimate minimal deriver prompt tokens, including custom instructions if present."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return estimate_minimal_deriver_prompt_tokens()

    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=normalized_custom_instructions,
    )
    return estimate_tokens(prompt)
