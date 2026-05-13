"""
Highlight extraction. Picks the most narratively-significant TEXT messages
from a chapter's worth of chat. Deterministic — no LLM.

Critical design rules:

  1. MEDIA_PLACEHOLDER messages (reel shares, song shares, etc.) are NEVER
     in highlights. They're context, not narrative source. The chapter
     should narrate the actual conversation between two people, not what
     reels they shared.

  2. For compact chapters (under ~200 text messages), we pass ALL text messages
     instead of selecting a subset. With minute-resolution timestamps and
     bursty texting, individual messages are often very short — selecting
     "top 25 by length" produces meaningless highlights and starves the
     LLM of context. Better to give it the whole conversation.

  3. Length thresholds account for Hindi-English code-switched chat where
     real, meaningful messages are often 5-25 chars ("Mr Rabbit",
     "Mere sath didi hai", "4%", "Babu plz"). The old 30-char floor
     filtered all of these out.
"""

import re
from datetime import timedelta

from ..models import Message, MessageKind


# Phrases that often indicate emotionally significant moments. Multilingual.
SIGNIFICANCE_KEYWORDS = [
    # English
    "love", "miss", "sorry", "thank", "promise", "forever", "always",
    "scared", "happy", "sad", "cry", "smile", "remember", "first time",
    "i don't", "we should", "we could", "next time", "last time",
    # Hindi/Urdu transliteration — common in Indian chats
    "pyaar", "pyar", "yaad", "maafi", "sorry",
    "ro rahi", "ro raha", "roo rha", "rona", "hurt",
    "khush", "udaas", "dukhi", "dukhe",
    "dilo", "dil se", "saath", "sath",
    "didi", "bhaiya", "mummy", "papa", "nani",
    "soo jana", "so jana", "nind",
    "samjho", "samjh", "batao",
]


def _score_text_message(msg: Message, all_msgs: list[Message], idx: int) -> float:
    """Score a TEXT message for highlight selection. Higher = more
    likely to surface. Tuned for short, bursty bilingual chat."""
    score = 0.0
    length = len(msg.text.strip())

    # Length: tuned for code-switched chat where 5-30 chars is the norm.
    # We still prefer messages that contain SOME content over single-word
    # responses, but the threshold is much lower than before.
    if length < 3:
        score -= 1.0  # "Hi", "K" etc. — probably noise
    elif length < 10:
        score += 0.2  # short but plausible
    elif length < 40:
        score += 0.6  # ideal short-message zone for chat
    elif length < 120:
        score += 1.0  # substantive message
    elif length < 300:
        score += 0.8  # long; useful but may be rambling
    else:
        score += 0.3  # very long — less common, often quotes/forwards

    # Emotionally significant keywords (English + Hindi)
    lower = msg.text.lower()
    keyword_hits = sum(1 for kw in SIGNIFICANCE_KEYWORDS if kw in lower)
    score += min(keyword_hits * 0.5, 1.5)  # cap so one keyword-heavy msg doesn't dominate

    # Late-night messages (10pm-3am) carry weight in relationship chats
    hour = msg.timestamp.hour
    if hour >= 22 or hour <= 3:
        score += 0.4

    # First message after a long silence (6+ hours) — conversation starter
    if idx > 0:
        gap = (msg.timestamp - all_msgs[idx - 1].timestamp).total_seconds()
        if gap > 6 * 3600:
            score += 0.6

    # Question marks tend to mark turning points
    if "?" in msg.text:
        score += 0.2

    # Heavy emoji use — emotional moments
    emoji_count = sum(1 for c in msg.text if ord(c) > 0x2600)
    if emoji_count >= 2:
        score += 0.3

    return score


def select_highlights(
    messages: list[Message],
    n: int = 25,
    full_conversation_threshold: int = 200,
) -> list[Message]:
    """Pick narrative-source messages from a chapter's worth of chat.

    Returns ONLY TEXT messages — media-shares are excluded entirely from
    highlights. If the chapter has no more than `full_conversation_threshold`
    text messages, returns all of them (sorted chronologically) instead
    of selecting a subset. This is critical for short chats where
    individual messages are short but the conversation as a whole is
    rich. It also covers dense one-night chats where 80+ short turns can
    all fit comfortably in the LLM prompt.

    For longer chats, scores all text messages and picks the top `n`
    with adaptive time-diversity. The diversity window shrinks for dense
    chapters so one important half-hour thread is not reduced to one or
    two lines.
    """
    # CRITICAL: exclude media-shares. They are context only, never
    # narrative source. The chapter is about the conversation, not
    # the reels.
    text_msgs = [m for m in messages if m.kind == MessageKind.TEXT]

    if not text_msgs:
        return []

    # For short chats, just pass everything. The LLM benefits from full
    # conversation flow more than from "highlights" that are arbitrarily
    # short snippets.
    if len(text_msgs) <= full_conversation_threshold:
        return text_msgs

    # Long chat: score and select with diversity
    scored: list[tuple[float, int, Message]] = []
    for i, m in enumerate(text_msgs):
        scored.append((_score_text_message(m, text_msgs, i), i, m))

    scored.sort(key=lambda x: -x[0])
    selected: list[Message] = []
    selected_times = []
    span_seconds = max(
        (text_msgs[-1].timestamp - text_msgs[0].timestamp).total_seconds(),
        60,
    )
    diversity_window = min(15 * 60, max(60, span_seconds / max(n * 2, 1)))
    for score, idx, msg in scored:
        too_close = any(
            abs((msg.timestamp - t).total_seconds()) < diversity_window
            for t in selected_times
        )
        if too_close:
            continue
        selected.append(msg)
        selected_times.append(msg.timestamp)
        if len(selected) >= n:
            break

    selected.sort(key=lambda m: m.timestamp)
    return selected
