"""
Chapter generation. For each chapter, build a prompt with:
  - The book-level arc (so chapters connect)
  - The month digests covering this chapter's date range
  - The week digests for the same range
  - 20-30 verbatim highlight messages

Then write the chapter, then verify it.
"""

from datetime import date
from typing import NamedTuple

from .. import llm
from ..models import Message
from . import highlights


class Chapter(NamedTuple):
    index: int
    title: str
    when: str
    body: str
    pull_quote: str
    pull_quote_author: str
    illustration_prompt: str  # used by image gen


CHAPTER_PROMPT = """You are writing one chapter of a book about a personal
chat relationship. The book is honest, warm, third-person, biographer-style.

CONTEXT:
{arc_context}

THIS CHAPTER COVERS: {start_date} to {end_date}

MONTH-LEVEL CONTEXT for this period:
{month_summaries}

ACTUAL MESSAGES from this period (use these as your source — do NOT invent
events or dialogue beyond what's here):
{highlight_block}

WRITE THE CHAPTER. Format your response EXACTLY like this, with these
exact section headers:

TITLE: [3-6 word title]
WHEN: [a poetic location/time line, e.g. "Late nights, March 2024"]
BODY: [200-350 words of third-person narrative prose. Use specific details
from the messages. Do not invent events. Refer to the people by their actual
names.]
QUOTE: [pick ONE actual message from the highlights that captures the
chapter's mood. Quote it verbatim, no quotation marks needed. The quote
will be displayed prominently — choose something meaningful but APPROPRIATE
for public display. Avoid profanity, slurs, or explicit content. Skip
generic phrases like "good night" or "love you" — pick something specific
to this chapter.]
QUOTE_BY: [name of the person who sent the quoted message]
ILLUSTRATION: [a 15-word visual description for an illustrator. Describe a
SCENE, not the people. e.g. "A small apartment kitchen at night, two
half-empty mugs on the counter."]"""


def _format_highlights(msgs: list[Message]) -> str:
    return "\n".join(
        f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}] {m.sender}: {m.text}"
        for m in msgs
    )


def _pick_fallback_quote(messages: list[Message]) -> tuple[str, str]:
    """Pick a clean fallback pull-quote from raw messages if the LLM's
    pick gets rejected. Prefers medium-length, non-generic messages."""
    from . import content_filter
    candidates = [
        m for m in messages
        if m.kind == MessageKind.TEXT
        and 15 <= len(m.text) <= 140
        and content_filter.safe_for_display(m.text)
    ]
    if not candidates:
        return "", ""
    # Pick from the middle of the chapter — that's usually the emotional core
    chosen = candidates[len(candidates) // 2]
    return chosen.text, chosen.sender


def _parse_chapter_response(
    text: str, index: int, fallback_when: str,
    chapter_messages: list[Message] | None = None,
) -> Chapter:
    """Parse the structured response. Lenient — fill defaults if the LLM
    misses a section. Validates the pull-quote for user-facing display."""
    from . import content_filter

    sections: dict[str, str] = {}
    current_key = None
    buffer: list[str] = []

    for line in text.splitlines():
        m = None
        for key in ["TITLE", "WHEN", "BODY", "QUOTE_BY", "QUOTE", "ILLUSTRATION"]:
            prefix = f"{key}:"
            if line.strip().upper().startswith(prefix):
                if current_key:
                    sections[current_key] = "\n".join(buffer).strip()
                current_key = key
                buffer = [line.split(":", 1)[1].strip()]
                m = key
                break
        if m is None and current_key:
            buffer.append(line)

    if current_key:
        sections[current_key] = "\n".join(buffer).strip()

    # Validate the pull-quote. If it has profanity or is generic dialogue,
    # try to replace with a clean fallback rather than display it.
    quote = sections.get("QUOTE", "")
    quote_by = sections.get("QUOTE_BY", "")
    if quote and not content_filter.safe_for_display(quote):
        # Try fallback from the chapter's own messages
        if chapter_messages:
            fallback_q, fallback_by = _pick_fallback_quote(chapter_messages)
            quote, quote_by = fallback_q, fallback_by
        else:
            quote, quote_by = "", ""

    return Chapter(
        index=index,
        title=sections.get("TITLE", f"Chapter {index}"),
        when=sections.get("WHEN", fallback_when),
        body=sections.get("BODY", "(Chapter generation incomplete.)"),
        pull_quote=quote,
        pull_quote_author=quote_by,
        illustration_prompt=sections.get("ILLUSTRATION", "two friends in a cozy room"),
    )


async def generate_chapter(
    index: int,
    start_date: date,
    end_date: date,
    chapter_messages: list[Message],
    month_summaries: dict[date, str],
    arc_context: str,
) -> Chapter:
    # Pick highlights from this chapter's messages
    chapter_highlights = highlights.select_highlights(chapter_messages, n=25)
    if not chapter_highlights:
        chapter_highlights = chapter_messages[:25]

    # Find month summaries that overlap this chapter
    relevant_months = [
        f"{m.strftime('%B %Y')}: {summary}"
        for m, summary in sorted(month_summaries.items())
        if start_date <= m.replace(day=28) and m <= end_date
        if summary
    ]
    months_block = "\n\n".join(relevant_months) if relevant_months else "(no month summaries)"

    fallback_when = f"{start_date.strftime('%B %Y')} – {end_date.strftime('%B %Y')}"

    prompt = CHAPTER_PROMPT.format(
        arc_context=arc_context,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        month_summaries=months_block,
        highlight_block=_format_highlights(chapter_highlights),
    )

    try:
        response = await llm.complete(
            [
                {"role": "system", "content": (
                    "You are a careful, warm biographer. You write in third "
                    "person. You never invent events. You only narrate what "
                    "the source messages support."
                )},
                {"role": "user", "content": prompt},
            ],
            model_size="strong",
            temperature=0.5,
        )
    except llm.LLMError as e:
        return Chapter(
            index=index,
            title=f"Chapter {index}",
            when=fallback_when,
            body=f"(Could not generate chapter: {e})",
            pull_quote="",
            pull_quote_author="",
            illustration_prompt="two friends in a cozy room",
        )

    return _parse_chapter_response(response, index, fallback_when, chapter_messages)
