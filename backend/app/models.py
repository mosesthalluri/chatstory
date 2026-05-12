"""
Core data models. Everything in the codebase that handles messages
uses these types — never raw dicts or strings.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["kind"] = self.kind.value
        return d


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
    state: str  # "queued", "parsing", "analyzing", "writing", "rendering", "done", "error"
    progress: int  # 0-100
    message: str  # human-readable status
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    preview_pdf: Optional[str] = None
    full_pdf: Optional[str] = None
    paid: bool = False
    stats: Optional[dict] = None  # dropped here when computed, for the UI

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d
