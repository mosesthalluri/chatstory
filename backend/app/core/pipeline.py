"""One deterministic intelligence entrypoint shared by all products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import ConversationSession, Memory, Message
from .memories import extract_memories
from .analytics import (
    communication_rhythm,
    emotional_trend,
    relationship_timeline,
    shared_language_evolution,
    silence_drift,
)
from .normalization import normalize_messages
from .scoring import ScoringConfig, score_sessions, select_meaningful_sessions
from .semantics import extract_semantic_phrases
from .sessions import detect_sessions


@dataclass
class IntelligenceResult:
    sessions: list[ConversationSession]
    selected_sessions: list[ConversationSession]
    memories: list[Memory]
    semantic_phrases: list[dict[str, Any]]
    analytics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "session_count": len(self.sessions),
            "selected_session_count": len(self.selected_sessions),
            "selected_sessions": [session.to_dict() for session in self.selected_sessions],
            "memories": [memory.to_dict() for memory in self.memories],
            "semantic_phrases": self.semantic_phrases,
            **self.analytics,
        }


def build_intelligence(
    messages: list[Message],
    senders: list[str] | None = None,
    config: ScoringConfig | None = None,
) -> IntelligenceResult:
    """Normalize, segment, score, filter, and analyze without an LLM call."""
    normalized = normalize_messages(messages)
    sessions = score_sessions(detect_sessions(normalized), config)
    selected = select_meaningful_sessions(sessions, config)
    semantic_phrases = extract_semantic_phrases(normalized, sessions)
    memories = extract_memories(selected, normalized)
    known_senders = senders or sorted({message.sender for message in normalized})
    analytics = {
        "relationship_timeline": relationship_timeline(sessions, memories),
        "emotional_trend": emotional_trend(sessions),
        "silence_drift": silence_drift(sessions),
        "communication_rhythm": communication_rhythm(normalized, sessions),
        "shared_language_evolution": shared_language_evolution(normalized, known_senders, semantic_phrases),
    }
    return IntelligenceResult(
        sessions=sessions,
        selected_sessions=selected,
        memories=memories,
        semantic_phrases=semantic_phrases,
        analytics=analytics,
    )
