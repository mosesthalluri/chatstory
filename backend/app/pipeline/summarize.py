"""
Hierarchical summarization. Day → week → month → arc.

This is what makes huge chats tractable. Each layer compresses the
previous one, so the final chapter prompt sees structured context
(month digests + week digests + raw highlights) instead of 10,000
unfiltered messages.

All summaries are kept short. The point is structure, not prose.
"""

import asyncio
from datetime import date

from .. import llm
from ..models import Message
from . import chunker


# ----------------------------------------------------------------------
# Layer 1: Day digests
# ----------------------------------------------------------------------

DAY_PROMPT = """You're summarizing one day of a personal chat conversation.
Write a 60-100 word factual summary covering: what topics were discussed,
the emotional tone, anything notable.

RULES:
- Use only what's in the messages. Do not invent events.
- Refer to people by their names as written.
- No flowery language. Plain factual prose.
- Lines prefixed with [SHARED MEDIA] are NOT things the sender wrote.
  They are reels, songs, or videos the sender SHARED. The text after
  [SHARED MEDIA] is the caption of that shared content, written by
  someone else. Do not quote it as if the sender said it.

Messages from {date}:
{transcript}

Summary:"""


def _format_message_for_llm(m: Message) -> str:
    """Format a message line for LLM consumption, tagging media shares
    so the model doesn't quote shared captions as if the sender wrote them.
    """
    from ..models import MessageKind
    time_str = m.timestamp.strftime('%H:%M')
    text = m.text[:200]
    if m.kind == MessageKind.MEDIA_PLACEHOLDER:
        return f"{time_str} {m.sender}: [SHARED MEDIA] {text}"
    return f"{time_str} {m.sender}: {text}"


async def summarize_day(date_obj: date, messages: list[Message]) -> str:
    if len(messages) < 5:
        return ""  # too short to be meaningful

    # Cap transcript to ~400 messages to keep prompt reasonable
    sample = messages[:400]
    transcript = "\n".join(_format_message_for_llm(m) for m in sample)

    try:
        return await llm.complete(
            [
                {"role": "system", "content": "You are a careful biographer who never invents facts."},
                {"role": "user", "content": DAY_PROMPT.format(date=date_obj.isoformat(), transcript=transcript)},
            ],
            model_size="fast",
            temperature=0.2,
        )
    except llm.LLMError as e:
        return f"(Day summary failed: {e})"


async def summarize_all_days(
    messages: list[Message],
    on_progress=None,
    max_concurrent: int = 4,
) -> dict[date, str]:
    """Summarize every active day. Runs with bounded concurrency to avoid
    rate limits."""
    by_day = chunker.by_day(messages)
    active = {d: msgs for d, msgs in by_day.items() if len(msgs) >= 5}

    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[date, str] = {}
    completed = 0
    total = len(active)

    async def worker(d: date, msgs: list[Message]):
        nonlocal completed
        async with semaphore:
            results[d] = await summarize_day(d, msgs)
            completed += 1
            if on_progress:
                on_progress(completed, total)

    await asyncio.gather(*(worker(d, m) for d, m in active.items()))
    return results


# ----------------------------------------------------------------------
# Layer 2 & 3: Week and Month rollups
# ----------------------------------------------------------------------

ROLLUP_PROMPT = """You're rolling up several daily summaries into one
{period} summary. Write 100-200 words covering the major themes, tone
shifts, and notable moments. Stay factual.

Daily summaries for this {period}:
{summaries}

{period} summary:"""


async def rollup(period: str, summaries_text: str) -> str:
    try:
        return await llm.complete(
            [
                {"role": "system", "content": "You are a careful biographer who never invents facts."},
                {"role": "user", "content": ROLLUP_PROMPT.format(period=period, summaries=summaries_text)},
            ],
            model_size="fast",
            temperature=0.3,
        )
    except llm.LLMError as e:
        return f"(Rollup failed: {e})"


async def summarize_weeks(day_summaries: dict[date, str], messages: list[Message]) -> dict[date, str]:
    weekly_groups = chunker.by_week(messages)
    out: dict[date, str] = {}

    for week_start, week_msgs in weekly_groups.items():
        relevant_days = sorted(
            d for d in day_summaries
            if week_start <= d < week_start.fromordinal(week_start.toordinal() + 7)
        )
        if not relevant_days:
            continue
        block = "\n\n".join(
            f"{d.isoformat()}: {day_summaries[d]}"
            for d in relevant_days if day_summaries[d]
        )
        if not block.strip():
            continue
        out[week_start] = await rollup("week", block)
    return out


async def summarize_months(week_summaries: dict[date, str], messages: list[Message]) -> dict[date, str]:
    monthly_groups = chunker.by_month(messages)
    out: dict[date, str] = {}

    for month_start, _ in monthly_groups.items():
        # Find weeks that overlap with this month
        next_month_ord = month_start.toordinal() + 31
        relevant_weeks = sorted(
            w for w in week_summaries
            if month_start.toordinal() - 6 <= w.toordinal() < next_month_ord
        )
        if not relevant_weeks:
            continue
        block = "\n\n".join(
            f"Week of {w.isoformat()}: {week_summaries[w]}"
            for w in relevant_weeks if week_summaries[w]
        )
        if not block.strip():
            continue
        out[month_start] = await rollup("month", block)
    return out


# ----------------------------------------------------------------------
# Layer 4: Book-level arc
# ----------------------------------------------------------------------

ARC_PROMPT = """You're identifying the major narrative arcs in a chat
relationship spanning {span_days} days. Below are month-by-month summaries.

Identify 4-8 distinct arcs or phases the relationship went through. For each,
give a one-line title and a 2-3 sentence description.

Month summaries:
{summaries}

Output format:
ARC 1: [title]
[description]

ARC 2: [title]
[description]
..."""


async def identify_arc(month_summaries: dict[date, str], span_days: int) -> str:
    if not month_summaries:
        return "No data."

    summaries_text = "\n\n".join(
        f"{m.strftime('%B %Y')}: {s}"
        for m, s in sorted(month_summaries.items())
    )

    try:
        return await llm.complete(
            [
                {"role": "system", "content": "You are a perceptive biographer identifying narrative threads."},
                {"role": "user", "content": ARC_PROMPT.format(span_days=span_days, summaries=summaries_text)},
            ],
            model_size="strong",
            temperature=0.4,
        )
    except llm.LLMError as e:
        return f"(Arc identification failed: {e})"
