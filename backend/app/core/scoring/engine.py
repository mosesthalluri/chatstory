"""Explainable session scoring that runs before any model prompt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import re

from ...models import ConversationSession, MessageKind
from .lexicons import FILLER_TERMS, LOGISTICS_TERMS, NEGATIVE_SIGNALS, POSITIVE_SIGNALS


@dataclass(frozen=True)
class ScoringConfig:
    positive_weight: float = 3.0
    negative_weight: float = 3.5
    caps_boost: float = 1.0
    punctuation_boost: float = 0.75
    paragraph_boost: float = 1.5
    rapid_exchange_boost: float = 1.25
    late_night_boost: float = 1.0
    filler_penalty: float = 0.75
    logistics_penalty: float = 0.6
    low_information_penalty: float = 0.45
    minimum_selected_score: float = 4.0
    max_selected_sessions: int = 24


def _text(session: ConversationSession) -> list[str]:
    return [
        (message.normalized_text or message.text.casefold()).strip()
        for message in session.messages
        if message.kind == MessageKind.TEXT
    ]


def _hits(text: str, lexicon: set[str]) -> int:
    return sum(1 for term in lexicon if term in text)


def score_session(
    session: ConversationSession,
    config: ScoringConfig | None = None,
) -> ConversationSession:
    cfg = config or ScoringConfig()
    texts = _text(session)
    combined = " ".join(texts)
    signals: dict[str, int] = {}
    positive_hits = 0
    negative_hits = 0
    for label, terms in POSITIVE_SIGNALS.items():
        count = _hits(combined, terms)
        if count:
            signals[label] = count
            positive_hits += count
    for label, terms in NEGATIVE_SIGNALS.items():
        count = _hits(combined, terms)
        if count:
            signals[label] = count
            negative_hits += count

    boosts = 0.0
    original_text = " ".join(
        message.text for message in session.messages if message.kind == MessageKind.TEXT
    )
    if re.search(r"\b[A-Z]{4,}\b", original_text):
        boosts += cfg.caps_boost
        signals["all_caps"] = 1
    if re.search(r"[!?]{2,}", original_text):
        boosts += cfg.punctuation_boost
        signals["repeated_punctuation"] = 1
    if any(len(text) >= 160 for text in texts):
        boosts += cfg.paragraph_boost
        signals["long_emotional_paragraph"] = 1
    if len(session.messages) >= 6 and session.duration <= timedelta(minutes=12):
        boosts += cfg.rapid_exchange_boost
        signals["rapid_exchange"] = 1
    if any(message.timestamp.hour >= 22 or message.timestamp.hour <= 3 for message in session.messages):
        boosts += cfg.late_night_boost
        signals["late_night"] = 1

    filler_count = sum(1 for text in texts if text in FILLER_TERMS)
    logistics_count = sum(_hits(text, LOGISTICS_TERMS) for text in texts)
    low_information_count = sum(1 for text in texts if len(text.split()) <= 2)
    penalties = (
        filler_count * cfg.filler_penalty
        + logistics_count * cfg.logistics_penalty
        + low_information_count * cfg.low_information_penalty
    )
    if filler_count:
        signals["filler"] = filler_count
    if logistics_count:
        signals["logistics"] = logistics_count

    session.sentiment = positive_hits * cfg.positive_weight - negative_hits * cfg.negative_weight
    session.score = max(
        0.0,
        positive_hits * cfg.positive_weight
        + negative_hits * cfg.negative_weight
        + boosts
        - penalties,
    )
    session.signal_counts = signals
    return session


def score_sessions(
    sessions: list[ConversationSession],
    config: ScoringConfig | None = None,
) -> list[ConversationSession]:
    return [score_session(session, config) for session in sessions]


def select_meaningful_sessions(
    sessions: list[ConversationSession],
    config: ScoringConfig | None = None,
) -> list[ConversationSession]:
    """Choose LLM evidence while preserving chronological story order."""
    cfg = config or ScoringConfig()
    meaningful = [
        session for session in sessions
        if session.score >= cfg.minimum_selected_score
        and any(
            label in session.signal_counts
            for label in set(POSITIVE_SIGNALS) | set(NEGATIVE_SIGNALS)
        )
    ]
    chosen = sorted(meaningful, key=lambda session: session.score, reverse=True)[:cfg.max_selected_sessions]
    return sorted(chosen, key=lambda session: session.start_time)
