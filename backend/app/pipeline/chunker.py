"""
Chunker. Splits the message stream into manageable time windows.

The chunker is the gatekeeper for token cost. Get this wrong and you
either melt the LLM (chunks too big) or lose narrative cohesion (chunks
too small).
"""

from collections import defaultdict
from datetime import date

from ..models import Message


def by_day(messages: list[Message]) -> dict[date, list[Message]]:
    """Group messages by calendar date."""
    out: dict[date, list[Message]] = defaultdict(list)
    for m in messages:
        out[m.timestamp.date()].append(m)
    return dict(out)


def by_week(messages: list[Message]) -> dict[date, list[Message]]:
    """Group by ISO week start (Monday)."""
    out: dict[date, list[Message]] = defaultdict(list)
    for m in messages:
        # ISO week starts Monday; .isocalendar() gives (year, week, weekday)
        d = m.timestamp.date()
        monday = d.fromordinal(d.toordinal() - d.weekday())
        out[monday].append(m)
    return dict(out)


def by_month(messages: list[Message]) -> dict[date, list[Message]]:
    """Group by first-of-month date."""
    out: dict[date, list[Message]] = defaultdict(list)
    for m in messages:
        first = m.timestamp.date().replace(day=1)
        out[first].append(m)
    return dict(out)


def into_chapters(
    messages: list[Message],
    n_chapters: int,
) -> list[tuple[date, date, list[Message]]]:
    """Split the full message list into N roughly-equal chapters by time.

    Returns: [(start_date, end_date, messages), ...] in chronological order.

    We split by time, not by message count. This keeps each chapter
    feeling like a coherent "period" of the relationship rather than
    arbitrary slices.
    """
    if not messages:
        return []
    first = messages[0].timestamp
    last = messages[-1].timestamp
    span = (last - first).total_seconds()
    chunk_secs = span / n_chapters

    chapters: list[tuple[date, date, list[Message]]] = []
    for i in range(n_chapters):
        chunk_start = first.timestamp() + i * chunk_secs
        chunk_end = first.timestamp() + (i + 1) * chunk_secs
        chunk_msgs = [
            m for m in messages
            if chunk_start <= m.timestamp.timestamp() < chunk_end
        ]
        # Last chapter is inclusive of the final message
        if i == n_chapters - 1:
            chunk_msgs = [m for m in messages if m.timestamp.timestamp() >= chunk_start]
        if chunk_msgs:
            chapters.append((
                chunk_msgs[0].timestamp.date(),
                chunk_msgs[-1].timestamp.date(),
                chunk_msgs,
            ))
    return chapters


def suggest_chapter_count(
    messages: list[Message],
    min_chapters: int = 3,
    max_chapters: int = 12,
) -> int:
    """Pick a sensible number of chapters based on the chat's actual
    length and density. Better than forcing one global default.

    Heuristic:
      - One chapter per ~45 days of span
      - One chapter per ~60 active days, whichever is larger
      - Clamped to [min_chapters, max_chapters]
      - Single-day chats get 1 chapter; nothing forces 8 mini-slices
        of a single conversation.

    What you actually get:
      - 1 day:    1 chapter (a single conversation isn't a multi-chapter arc)
      - 1 week:   3 chapters
      - 1 month:  3 chapters
      - 3 months: 3 chapters
      - 6 months: 4 chapters
      - 1 year:   ~8 chapters (~6 weeks each)
      - 2 years:  ~12 chapters (~2 months each — feels like a memoir)
      - 5+ years: 12 chapters max (one per ~5 months — story-paced, not exhaustive)
    """
    if not messages:
        return min_chapters

    first = messages[0].timestamp.date()
    last = messages[-1].timestamp.date()
    span_days = (last - first).days

    if span_days < 2:
        return 1

    active_days = len({m.timestamp.date() for m in messages})

    by_time = max(1, round(span_days / 45))
    by_activity = max(1, round(active_days / 60))

    suggested = max(by_time, by_activity)
    return max(min_chapters, min(max_chapters, suggested))
