"""Weighted ranking for grounded product evidence."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from ...models import Memory, Message
from ..scoring.lexicons import NEGATIVE_SIGNALS, POSITIVE_SIGNALS
from .noise import is_semantic_message, semantic_tokens, suppress_noise


EMOTIONAL_TERMS = set().union(*POSITIVE_SIGNALS.values(), *NEGATIVE_SIGNALS.values())


def _term_hits(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)


def rank_evidence_messages(
    messages: Iterable[Message],
    anchor_terms: set[str],
    memories: list[Memory] | None = None,
) -> list[tuple[float, Message]]:
    """Rank real messages by relevance, emotion, specificity, and memory context."""
    memory_ids = {
        id(message): memory.emotional_weight
        for memory in (memories or [])
        for message in memory.evidence_messages
    }
    clean_anchors = {suppress_noise(term) for term in anchor_terms if suppress_noise(term)}
    scored: list[tuple[float, Message]] = []
    for message in messages:
        if not is_semantic_message(message):
            continue
        cleaned = suppress_noise(message.normalized_text or message.text)
        tokens = semantic_tokens(cleaned)
        overlap = sum(1 for anchor in clean_anchors if anchor in cleaned)
        if clean_anchors and overlap == 0:
            continue
        emotional = _term_hits(cleaned, EMOTIONAL_TERMS)
        uniqueness = len(set(tokens)) / max(len(tokens), 1)
        context = min(memory_ids.get(id(message), 0.0) / 5.0, 3.0)
        length_quality = 1.0 if 4 <= len(tokens) <= 30 else 0.25
        score = overlap * 3.0 + emotional * 2.0 + uniqueness + context + length_quality
        scored.append((score, message))
    return sorted(scored, key=lambda item: (-item[0], item[1].timestamp))
