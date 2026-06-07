"""Time-aware session detection for emotionally coherent chat scenes."""

from __future__ import annotations

from datetime import date, timedelta

from ..models import ConversationSession, Message


def _should_split(previous: Message, current: Message) -> bool:
    gap = current.timestamp - previous.timestamp
    crosses_day = current.timestamp.date() != previous.timestamp.date()
    sleep_gap = previous.timestamp.hour >= 21 and current.timestamp.hour <= 8
    if gap >= timedelta(hours=6):
        return True
    if sleep_gap and gap >= timedelta(hours=3):
        return True
    if crosses_day and gap >= timedelta(minutes=45):
        return True
    return gap >= timedelta(minutes=90)


def detect_sessions(messages: list[Message]) -> list[ConversationSession]:
    """Split chronological messages on inactivity, overnight, and day gaps."""
    if not messages:
        return []

    groups: list[list[Message]] = []
    current: list[Message] = [messages[0]]
    for message in messages[1:]:
        if _should_split(current[-1], message):
            groups.append(current)
            current = []
        current.append(message)
    groups.append(current)

    return [
        ConversationSession(
            start_time=group[0].timestamp,
            end_time=group[-1].timestamp,
            messages=group,
            participants=sorted({message.sender for message in group}),
            duration=group[-1].timestamp - group[0].timestamp,
        )
        for group in groups
    ]


def sessions_into_chapters(
    sessions: list[ConversationSession],
    chapter_count: int,
) -> list[tuple[date, date, list[Message]]]:
    """Group complete selected sessions into chapter inputs without splitting scenes."""
    if not sessions:
        return []
    count = max(1, min(chapter_count, len(sessions)))
    groups: list[list[ConversationSession]] = [[] for _ in range(count)]
    for index, session in enumerate(sessions):
        group_index = min(index * count // len(sessions), count - 1)
        groups[group_index].append(session)

    chapters = []
    for group in groups:
        messages = [message for session in group for message in session.messages]
        if messages:
            chapters.append((messages[0].timestamp.date(), messages[-1].timestamp.date(), messages))
    return chapters
