"""
Highlight extraction. Picks the most narratively-significant messages
from a chapter's worth of chat. Deterministic — no LLM.

The chapter generator gets these as evidence. Better selection here =
better chapters. This is the single biggest non-prompt quality lever.
"""

import re
from collections import Counter
from datetime import timedelta
from typing import Iterable

from ..models import Message, MessageKind


# Phrases that often indicate emotionally significant moments.
# Multilingual-friendly: keep this short and let it grow with feedback.
SIGNIFICANCE_KEYWORDS = [
    "love", "miss", "sorry", "thank", "promise", "forever", "always",
    "scared", "happy", "sad", "cry", "smile", "remember", "first time",
    "i don't", "we should", "we could", "next time", "last time",
]


def _score_message(msg: Message, all_msgs: list[Message], idx: int) -> float:
    """Higher score = more likely to be a highlight."""
    score = 0.0

    # Length: medium-length messages often carry the meaning
    length = len(msg.text)
    if 30 <= length <= 200:
        score += 1.0
    elif 200 < length <= 500:
        score += 0.5

    # Emotionally significant keywords
    lower = msg.text.lower()
    for kw in SIGNIFICANCE_KEYWORDS:
        if kw in lower:
            score += 0.5

    # Late-night messages (10pm-3am) carry weight
    hour = msg.timestamp.hour
    if hour >= 22 or hour <= 3:
        score += 0.4

    # First message after a long silence (6+ hours) — conversation starter
    if idx > 0:
        gap = (msg.timestamp - all_msgs[idx - 1].timestamp).total_seconds()
        if gap > 6 * 3600:
            score += 0.6

    # Heavy emoji use — emotional moments
    emoji_count = sum(1 for c in msg.text if ord(c) > 0x2600)
    if emoji_count >= 2:
        score += 0.3

    return score


def _conversation_clusters(messages: list[Message], max_gap_minutes: int = 30) -> list[list[Message]]:
    """Group messages into conversation bursts. A burst ends when there's
    a gap of `max_gap_minutes` or more."""
    if not messages:
        return []
    bursts: list[list[Message]] = [[messages[0]]]
    for m in messages[1:]:
        gap = (m.timestamp - bursts[-1][-1].timestamp)
        if gap > timedelta(minutes=max_gap_minutes):
            bursts.append([m])
        else:
            bursts[-1].append(m)
    return bursts


def select_highlights(messages: list[Message], n: int = 25) -> list[Message]:
    """Pick the top ~N most narratively-significant messages from a list.

    Strategy: score every message, then select with a diversity bias so
    we don't pick 25 messages from one hot afternoon.
    """
    if len(messages) <= n:
        return [m for m in messages if m.kind == MessageKind.TEXT]

    # Score each text message
    scored: list[tuple[float, int, Message]] = []
    for i, m in enumerate(messages):
        if m.kind != MessageKind.TEXT:
            continue
        scored.append((_score_message(m, messages, i), i, m))

    # Sort by score descending, but enforce time-diversity:
    # No two highlights within 30 minutes of each other.
    scored.sort(key=lambda x: -x[0])
    selected: list[Message] = []
    selected_times = []
    for score, idx, msg in scored:
        too_close = any(
            abs((msg.timestamp - t).total_seconds()) < 1800
            for t in selected_times
        )
        if too_close:
            continue
        selected.append(msg)
        selected_times.append(msg.timestamp)
        if len(selected) >= n:
            break

    # Sort selected by time so the LLM sees them in order
    selected.sort(key=lambda m: m.timestamp)
    return selected
