"""Lightweight analytics built from scored conversation sessions."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from statistics import median
import re
from typing import Any

from ...models import ConversationSession, Memory, Message
from ..scoring.lexicons import NEGATIVE_SIGNALS, POSITIVE_SIGNALS


def _session_period(session: ConversationSession) -> str:
    return session.start_time.strftime("%Y-%m")


def relationship_timeline(
    sessions: list[ConversationSession],
    memories: list[Memory] | None = None,
) -> list[dict[str, Any]]:
    """Describe changing relationship phases using observable behavior."""
    grouped: dict[str, list[ConversationSession]] = defaultdict(list)
    memory_groups: dict[str, list[Memory]] = defaultdict(list)
    for session in sessions:
        grouped[_session_period(session)].append(session)
    for memory in memories or []:
        memory_groups[memory.start_time.strftime("%Y-%m")].append(memory)

    out: list[dict[str, Any]] = []
    previous_count = 0
    previous_phase = ""
    for index, period in enumerate(sorted(grouped)):
        group = grouped[period]
        period_memories = memory_groups[period]
        positives = sum(
            session.signal_counts.get(label, 0) for session in group for label in POSITIVE_SIGNALS
        )
        negatives = sum(
            session.signal_counts.get(label, 0) for session in group for label in NEGATIVE_SIGNALS
        )
        average_score = sum(session.score for session in group) / len(group)
        memory_types = Counter(memory.memory_type for memory in period_memories)
        reconnects = memory_types["reconnection"]
        support = memory_types["support"] + memory_types["comfort"] + memory_types["reassurance"]
        conflict = memory_types["conflict"]
        if conflict and conflict >= support:
            phase = "conflict periods"
            title = "Conflict became explicit"
            why = "Arguments or distance language outweighed repair in the remembered moments from this period."
        elif reconnects:
            phase = "reconciliation"
            title = "They came back after silence"
            why = "Conversation resumed after a substantial pause, with emotionally weighted replies instead of remaining distant."
        elif index > 0 and previous_count and len(group) < previous_count * 0.5:
            phase = "drifting apart"
            title = "Conversation thinned out"
            why = "The number of conversational scenes dropped sharply compared with the preceding period."
        elif support >= 2 or positives >= 3:
            phase = "attachment growth"
            title = "Reassurance became a habit"
            why = "Support and comfort repeatedly appeared as replies to vulnerable or emotionally open messages."
        else:
            phase = "getting comfortable"
            title = "Everyday rhythm formed"
            why = "The conversation settled into recurring contact before stronger emotional patterns emerged."
        if phase == previous_phase and period_memories:
            strongest = period_memories[0]
            title = f"{title}: {strongest.memory_type}"
            why = strongest.summary
        out.append({
            "period": period,
            "phase": phase,
            "title": title,
            "why": why,
            "tone": phase,
            "sessions": len(group),
            "emotional_score": round(average_score, 2),
            "positive_signals": positives,
            "negative_signals": negatives,
            "memory_types": dict(memory_types),
        })
        previous_count = len(group)
        previous_phase = phase
    return out


def emotional_trend(sessions: list[ConversationSession]) -> dict[str, Any]:
    points = [
        {
            "start_time": session.start_time.isoformat(),
            "score": round(session.score, 2),
            "sentiment": round(session.sentiment, 2),
            "messages": len(session.messages),
        }
        for session in sessions
    ]
    if len(sessions) < 2:
        direction = "insufficient_history"
    else:
        midpoint = max(len(sessions) // 2, 1)
        early = sum(session.score for session in sessions[:midpoint]) / midpoint
        recent = sum(session.score for session in sessions[midpoint:]) / max(len(sessions) - midpoint, 1)
        direction = "deepening" if recent > early * 1.15 else "cooling" if recent < early * 0.85 else "steady"
    peak = max(sessions, key=lambda session: session.score, default=None)
    return {
        "direction": direction,
        "points": points,
        "peak_session": peak.to_dict() if peak else None,
    }


def silence_drift(sessions: list[ConversationSession]) -> dict[str, Any]:
    gaps: list[tuple[timedelta, ConversationSession, ConversationSession]] = [
        (current.start_time - previous.end_time, previous, current)
        for previous, current in zip(sessions, sessions[1:])
    ]
    ranked = sorted(gaps, key=lambda item: item[0], reverse=True)
    longest = [
        {
            "from": previous.end_time.isoformat(),
            "to": current.start_time.isoformat(),
            "hours": round(gap.total_seconds() / 3600, 1),
        }
        for gap, previous, current in ranked[:5]
    ]
    withdrawal = [
        entry for entry in longest if entry["hours"] >= 72
    ]
    reconnections = [
        {
            "after_hours": round(gap.total_seconds() / 3600, 1),
            "at": current.start_time.isoformat(),
            "score": round(current.score, 2),
        }
        for gap, _, current in gaps
        if gap >= timedelta(days=2) and current.score >= 4
    ]
    return {
        "longest_silences": longest,
        "withdrawal_periods": withdrawal,
        "reconnection_moments": reconnections,
        "communication_drop_periods": withdrawal,
    }


def communication_rhythm(
    messages: list[Message],
    sessions: list[ConversationSession],
) -> dict[str, Any]:
    initiations = Counter(session.messages[0].sender for session in sessions if session.messages)
    reply_seconds: dict[str, list[float]] = defaultdict(list)
    sender_counts = Counter(message.sender for message in messages)
    for previous, current in zip(messages, messages[1:]):
        if previous.sender != current.sender:
            gap = (current.timestamp - previous.timestamp).total_seconds()
            if 0 <= gap <= 24 * 3600:
                reply_seconds[current.sender].append(gap)
    medians = {
        sender: int(median(times)) for sender, times in reply_seconds.items() if times
    }
    density = [
        {
            "start_time": session.start_time.isoformat(),
            "messages_per_hour": round(
                len(session.messages) / max(session.duration.total_seconds() / 3600, 0.25),
                2,
            ),
        }
        for session in sessions
    ]
    counts = list(sender_counts.values())
    imbalance = 0.0 if not counts else round((max(counts) - min(counts)) / max(sum(counts), 1), 3)
    return {
        "initiations": dict(initiations),
        "top_initiator": initiations.most_common(1)[0][0] if initiations else None,
        "median_reply_seconds": medians,
        "response_consistency_seconds": {
            sender: int(max(times) - min(times)) for sender, times in reply_seconds.items() if len(times) > 1
        },
        "conversation_imbalance": imbalance,
        "message_density": density,
    }


def shared_language_evolution(
    messages: list[Message],
    senders: list[str],
    phrases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Represent only weighted recurring language, already stripped of metadata."""
    recurring = []
    for phrase in phrases or []:
        recurring.append({
            **phrase,
            "word": phrase["phrase"],
            "first_used": next(
                (
                    message.timestamp.isoformat() for message in messages
                    if phrase["phrase"] in (message.normalized_text or message.text.casefold())
                ),
                "",
            ),
            "last_used": next(
                (
                    message.timestamp.isoformat() for message in reversed(messages)
                    if phrase["phrase"] in (message.normalized_text or message.text.casefold())
                ),
                "",
            ),
        })
    affectionate = [
        item for item in recurring
        if any(term in item["phrase"] for term in {"baby", "babu", "jaan", "cutie", "dear"})
    ]
    return {
        "recurring_phrases": recurring[:20],
        "nicknames": affectionate[:10],
        "inside_joke_candidates": [
            item for item in recurring if item["phrase_type"] == "relationship_specific"
        ][:8],
        "evolving_shared_slang": recurring[:12],
    }
