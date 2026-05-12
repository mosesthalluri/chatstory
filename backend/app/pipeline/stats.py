"""
Stats engine. Pure Python over the message list. No LLM, no
hallucinations possible. This powers the Wrapped page and feeds the
chapter generator with structural info.
"""

import re
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any

import emoji as emoji_lib

from ..models import Message, MessageKind, ParsedChat


# Common English stopwords. We strip these before computing top words /
# inside jokes. Not exhaustive — just the highest-frequency junk.
STOPWORDS = set("""
a an the and or but if then else when so for nor on at by to of in
i me my mine you your yours we us our he him his she her hers it its
they them their this that these those is am are was were be been being
have has had do does did will would could should can may might must
not no yes ok okay yeah yea yep nope yo hi hey lol lmao haha hehe
just like get got go going gonna want need think know say said get
got u ur n r y idk btw tbh omg pls plz so much really very also
""".split())


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, return word list."""
    text = text.lower()
    # Replace emoji with space so they don't merge with adjacent words
    text = emoji_lib.replace_emoji(text, replace=" ")
    return re.findall(r"[a-z']+", text)


def _extract_emojis(text: str) -> list[str]:
    return [e["emoji"] for e in emoji_lib.emoji_list(text)]


def compute_stats(parsed: ParsedChat) -> dict[str, Any]:
    """Compute the full stats dict. Used both in the Wrapped page and as
    structural input to the chapter generator."""
    msgs = parsed.messages
    if not msgs:
        return {"error": "no messages"}

    senders = parsed.senders or sorted({m.sender for m in msgs})

    first_ts = msgs[0].timestamp
    last_ts = msgs[-1].timestamp
    days_span = max((last_ts - first_ts).days, 1)

    # Per-sender counts and lengths
    per_sender_count = Counter(m.sender for m in msgs)
    per_sender_chars = defaultdict(int)
    for m in msgs:
        per_sender_chars[m.sender] += len(m.text)

    # Hour and day-of-week distributions
    hour_dist = Counter(m.timestamp.hour for m in msgs)
    dow_dist = Counter(m.timestamp.strftime("%A") for m in msgs)
    most_active_hour = max(hour_dist, key=hour_dist.get)
    most_active_dow = max(dow_dist, key=dow_dist.get)

    # Days active (any message that day)
    active_dates = {m.timestamp.date() for m in msgs}
    days_active = len(active_dates)

    # Emoji frequency
    all_emojis = []
    per_sender_emojis = defaultdict(list)
    for m in msgs:
        if m.kind != MessageKind.TEXT:
            continue
        es = _extract_emojis(m.text)
        all_emojis.extend(es)
        per_sender_emojis[m.sender].extend(es)
    top_emojis = Counter(all_emojis).most_common(10)

    # Top words and phrases (inside-joke detection)
    word_counter: Counter = Counter()
    bigram_counter: Counter = Counter()
    for m in msgs:
        if m.kind != MessageKind.TEXT:
            continue
        tokens = [t for t in _tokenize(m.text) if t not in STOPWORDS and len(t) > 2]
        word_counter.update(tokens)
        for a, b in zip(tokens, tokens[1:]):
            bigram_counter[f"{a} {b}"] += 1

    # Inside jokes: real ones must clear THREE filters, not just frequency:
    #   1. Appears 3+ times (frequency signal)
    #   2. Not a generic dialogue phrase ("good night", "love you", etc.)
    #   3. Not inappropriate (profanity/slurs — Wrapped page is screenshottable)
    #   4. Contains at least one "distinctive" token (not all top-50 common words)
    # We'd rather show NO inside joke than show a wrong one.
    from . import content_filter
    common_top_tokens = {w for w, _ in word_counter.most_common(50)}
    inside_joke_candidates = [
        (phrase, count) for phrase, count in bigram_counter.most_common(200)
        if count >= 3
        and content_filter.safe_for_display(phrase)
        and content_filter.has_distinctive_token(phrase, common_top_tokens)
    ]
    inside_jokes = inside_joke_candidates[:5]

    # Top words — also filter inappropriate, in case shown anywhere
    top_words = [
        (w, c) for w, c in word_counter.most_common(30)
        if not content_filter.contains_inappropriate(w)
    ][:15]

    # Response time distribution (gap between sender switches, capped at 24h)
    response_times: list[int] = []  # in seconds
    for prev, cur in zip(msgs, msgs[1:]):
        if cur.sender != prev.sender:
            gap = (cur.timestamp - prev.timestamp).total_seconds()
            if 0 < gap < 86400:
                response_times.append(int(gap))
    median_response = (
        sorted(response_times)[len(response_times) // 2]
        if response_times else 0
    )

    # Conversation initiation: who sends the first message after a gap of 6+ hours
    initiations = Counter()
    for prev, cur in zip(msgs, msgs[1:]):
        gap = (cur.timestamp - prev.timestamp).total_seconds()
        if gap > 6 * 3600:
            initiations[cur.sender] += 1

    # Identify "narrative moments" — days with abnormal activity
    daily_counts = Counter(m.timestamp.date() for m in msgs)
    avg_daily = sum(daily_counts.values()) / max(len(daily_counts), 1)
    narrative_days = sorted(
        [(d, c) for d, c in daily_counts.items() if c > avg_daily * 2.5],
        key=lambda x: -x[1],
    )[:20]

    # User-facing message count = raw count before consecutive merging.
    # Users expect "messages exchanged" to mean individual sends, not
    # merged conversational turns. We keep `turns` available separately
    # for downstream code that needs the merged number.
    raw_count = parsed.raw_message_count or len(msgs)

    return {
        "total_messages": raw_count,
        "turns": len(msgs),
        "days_span": days_span,
        "days_active": days_active,
        "first_message_date": first_ts.isoformat(),
        "last_message_date": last_ts.isoformat(),
        "senders": senders,
        "messages_per_sender": dict(per_sender_count),
        "chars_per_sender": dict(per_sender_chars),
        "most_active_hour": most_active_hour,
        "most_active_day_of_week": most_active_dow,
        "hour_distribution": dict(hour_dist),
        "top_emojis": top_emojis,
        "emojis_per_sender": {
            s: Counter(es).most_common(5) for s, es in per_sender_emojis.items()
        },
        "top_words": top_words,
        "inside_jokes": inside_jokes,
        "median_response_seconds": median_response,
        "initiations": dict(initiations),
        "narrative_days": [(d.isoformat(), c) for d, c in narrative_days],
    }
