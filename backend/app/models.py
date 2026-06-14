"""
Core data models. Everything in the codebase that handles messages
uses these types — never raw dicts or strings.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class MessageKind(str, Enum):
    """What kind of content this message contains.

    We use this rather than guessing from text content because each
    parser knows the answer authoritatively.
    """
    TEXT = "text"               # normal text message
    MEDIA_PLACEHOLDER = "media" # photo/video/voice we can't read
    SYSTEM = "system"           # "X added Y to the group" — usually filtered out
    REACTION = "reaction"       # ❤️ on someone's message — usually filtered out
    DELETED = "deleted"         # "this message was deleted"


@dataclass
class Message:
    """A single normalized message. The atomic unit of the pipeline.

    Once parsing is done, the rest of the codebase only sees lists of
    these. Adding a new chat platform = writing a new parser that
    produces these.
    """
    sender: str
    timestamp: datetime
    text: str
    kind: MessageKind = MessageKind.TEXT

    # Optional fields — useful when present, fine when absent
    reply_to_id: Optional[str] = None
    message_id: Optional[str] = None
    normalized_text: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["kind"] = self.kind.value
        return d


@dataclass
class ConversationSession:
    """A coherent conversational scene separated by meaningful pauses."""
    start_time: datetime
    end_time: datetime
    messages: list[Message]
    participants: list[str]
    duration: timedelta
    score: float = 0.0
    signal_counts: dict[str, int] = field(default_factory=dict)
    sentiment: float = 0.0

    def to_dict(self, *, include_messages: bool = False) -> dict:
        data = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "participants": self.participants,
            "duration_seconds": int(self.duration.total_seconds()),
            "message_count": len(self.messages),
            "score": round(self.score, 2),
            "signal_counts": self.signal_counts,
            "sentiment": round(self.sentiment, 2),
        }
        if include_messages:
            data["messages"] = [message.to_dict() for message in self.messages]
        return data


@dataclass
class Memory:
    """A grounded relationship memory extracted from a meaningful scene."""
    summary: str
    emotional_weight: float
    evidence_messages: list[Message]
    participants: list[str]
    themes: list[str]
    start_time: datetime
    end_time: datetime
    memory_type: str
    evidence_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "emotional_weight": round(self.emotional_weight, 2),
            "evidence_messages": [message.to_dict() for message in self.evidence_messages],
            "participants": self.participants,
            "themes": self.themes,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "memory_type": self.memory_type,
            "evidence_score": round(self.evidence_score, 2),
        }


@dataclass
class MemoryMoment(Memory):
    """A remembered interaction backed by explicit message evidence."""


@dataclass
class EmotionalMoment(MemoryMoment):
    """A memory whose evidence includes explicit emotional language."""


@dataclass
class ParsedChat:
    """The complete parsed result. What every parser returns."""
    messages: list[Message]
    detected_format: str
    senders: list[str] = field(default_factory=list)
    parser_warnings: list[str] = field(default_factory=list)
    # Raw count before consecutive-turn merging. Stats engine uses this
    # for the user-facing "messages exchanged" number — users expect
    # "messages" to mean individual sends, not merged conversational turns.
    raw_message_count: int = 0

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def date_range(self) -> tuple[datetime, datetime]:
        if not self.messages:
            raise ValueError("No messages")
        return self.messages[0].timestamp, self.messages[-1].timestamp


@dataclass
class JobStatus:
    """Status of a book-generation job. Persisted to disk as JSON."""
    job_id: str
    state: str  # queued, processing, parsing, analyzing, writing, rendering,
                 # generating_wrapped, generating_gifts, generating_story, done, failed
    progress: int  # 0-100
    message: str  # human-readable status
    created_at: datetime
    updated_at: datetime
    user_email: Optional[str] = None
    product: Optional[str] = None  # chat-wrapped, gift-engine, chatstory
    error: Optional[str] = None
    preview_pdf: Optional[str] = None
    full_pdf: Optional[str] = None
    paid: bool = False
    stats: Optional[dict] = None  # dropped here when computed, for the UI
    # Normalization artifacts — what the parser actually extracted from
    # the user's raw input. Made available immediately after parsing so
    # users can verify before paying for chapter generation.
    normalized_txt: Optional[str] = None
    normalized_json: Optional[str] = None
    # Phase tracking for detailed progress display
    phases: Optional[list[dict]] = None  # [{"name": "...", "status": "done|in_progress|pending", "progress": 0-100}]
    # ChatStory smart flow: user-selected date window (YYYY-MM-DD) and the
    # tier price computed from the selected message volume.
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    price: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d
