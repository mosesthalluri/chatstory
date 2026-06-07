"""Weighted phrase extraction prioritizing human meaning over raw frequency."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ...models import ConversationSession, Message
from ..scoring.lexicons import NEGATIVE_SIGNALS, POSITIVE_SIGNALS
from .noise import is_semantic_message, semantic_tokens, suppress_noise


EMOTIONAL_PHRASES = set().union(*POSITIVE_SIGNALS.values(), *NEGATIVE_SIGNALS.values())
UNIVERSAL_RITUALS = {
    "love you", "miss you", "thank you", "take care", "good night",
    "good morning", "do not worry", "i am here for you", "sorry",
    "proud of you", "do not leave",
}
GENERIC_RELATIONAL_TOKENS = {
    "love", "miss", "sorry", "proud", "promise", "forever", "care", "here",
    "leave", "you", "your", "of", "for", "with", "always", "thank",
}
EDGE_STOPWORDS = {"i", "you", "your", "of", "for", "to", "the", "and", "am", "are"}


def extract_semantic_phrases(
    messages: list[Message],
    sessions: list[ConversationSession],
    *,
    min_count: int = 2,
) -> list[dict[str, Any]]:
    session_by_message = {
        id(message): session for session in sessions for message in session.messages
    }
    counts: Counter[str] = Counter()
    phrase_messages: dict[str, list[Message]] = defaultdict(list)
    all_token_counts: Counter[str] = Counter()
    for message in messages:
        if not is_semantic_message(message):
            continue
        cleaned = suppress_noise(message.normalized_text or message.text)
        tokens = semantic_tokens(cleaned)
        all_token_counts.update(set(tokens))
        candidates: set[str] = set()
        session = session_by_message.get(id(message))
        if session and session.score >= 4:
            candidates.update(
                token for token in tokens
                if len(token) >= 4
                and token not in GENERIC_RELATIONAL_TOKENS
                and token not in EDGE_STOPWORDS
            )
        for size in (2, 3):
            candidates.update(
                " ".join(tokens[index:index + size])
                for index in range(len(tokens) - size + 1)
                if tokens[index] not in EDGE_STOPWORDS
                and tokens[index + size - 1] not in EDGE_STOPWORDS
            )
        candidates.update(term for term in EMOTIONAL_PHRASES if term in cleaned)
        for phrase in candidates:
            if len(phrase) < 5:
                continue
            counts[phrase] += 1
            phrase_messages[phrase].append(message)

    results: list[dict[str, Any]] = []
    for phrase, count in counts.items():
        occurrences = phrase_messages[phrase]
        if count < min_count:
            continue
        senders = sorted({message.sender for message in occurrences})
        distinct_sessions = {id(session_by_message[id(message)]) for message in occurrences if id(message) in session_by_message}
        tokens = phrase.split()
        emotional_intensity = 2.5 if phrase in EMOTIONAL_PHRASES else sum(
            1.0 for term in EMOTIONAL_PHRASES if term in phrase
        )
        uniqueness = sum(1.0 / max(all_token_counts[token], 1) for token in tokens) * 2
        reciprocity = 2.0 if len(senders) > 1 else 0.0
        recurrence_quality = min(len(distinct_sessions), 4) * 1.2
        session_relevance = sum(session_by_message[id(message)].score for message in occurrences if id(message) in session_by_message)
        session_relevance = min(session_relevance / max(count, 1) / 4, 3.0)
        has_distinctive_token = any(token not in GENERIC_RELATIONAL_TOKENS for token in tokens)
        nickname_specificity = 2.5 if len(tokens) == 1 and has_distinctive_token else 0.0
        score = (
            emotional_intensity + uniqueness + reciprocity
            + recurrence_quality + session_relevance + nickname_specificity
        )
        phrase_type = (
            "emotional_ritual"
            if any(ritual in phrase for ritual in UNIVERSAL_RITUALS) or not has_distinctive_token
            else "relationship_specific"
        )
        results.append({
            "phrase": phrase,
            "count": count,
            "quote": occurrences[0].text[:200],
            "senders": senders,
            "score": round(score, 2),
            "phrase_type": phrase_type,
            "emotional_intensity": round(emotional_intensity, 2),
            "reciprocity": round(reciprocity, 2),
            "recurrence_quality": round(recurrence_quality, 2),
            "nickname_specificity": round(nickname_specificity, 2),
        })
    ranked = sorted(results, key=lambda item: (-item["score"], -item["count"], item["phrase"]))
    deduped: list[dict[str, Any]] = []
    specific_token_sets: list[set[str]] = []
    for item in ranked:
        if item["phrase_type"] == "relationship_specific":
            tokens = set(item["phrase"].split())
            if any(
                len(tokens & existing) / max(min(len(tokens), len(existing)), 1) >= 0.66
                for existing in specific_token_sets
            ):
                continue
            specific_token_sets.append(tokens)
        deduped.append(item)
    return deduped[:24]
