"""
WhatsApp .txt export parser.

WhatsApp exports look like one of these (depending on phone OS / locale):

    15/01/25, 21:43 - Aarav: hey
    [15/01/25, 21:43:00] Aarav: hey
    1/15/25, 9:43 PM - Aarav: hey
    15.1.25, 21:43 - Aarav: hey

This parser handles all of them. Multi-line messages are joined onto the
previous message until a new timestamp line appears.

System messages and media placeholders are detected and tagged so the
downstream pipeline can filter them.
"""

import re
import zipfile
from datetime import datetime
from pathlib import Path

from ..models import Message, MessageKind, ParsedChat


# Regex matches the start of a message line: "[date, time] sender: text"
# or "date, time - sender: text". Supports multiple date formats.
LINE_RE = re.compile(
    r"^\[?"
    r"(?P<date>\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})"
    r",?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)"
    r"\]?\s*[-–—]?\s*"
    r"(?P<rest>.+)$"
)

# What "this message was deleted" looks like in different locales
DELETED_PATTERNS = [
    "this message was deleted",
    "you deleted this message",
    "<media omitted>",  # also media — but caught below
]

# Media placeholders seen across WhatsApp exports
MEDIA_PATTERNS = [
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "voice call",
    "missed voice call",
    "video call",
    "missed video call",
    "(file attached)",
]

# Common system messages — all start with one party doing something
# without it being a chat message. We only flag the obvious ones; rest
# fall through and get stats but can be filtered by the noise filter.
SYSTEM_PATTERNS = [
    "messages and calls are end-to-end encrypted",
    "you created this group",
    "added you",
    "changed the subject",
    "changed this group's icon",
    "changed the group description",
    "left the group",
]


def _try_parse_datetime(date_str: str, time_str: str) -> datetime | None:
    """Try every reasonable date format until one works."""
    candidates = [
        # day first
        "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%d.%m.%y %H:%M", "%d.%m.%Y %H:%M",
        "%d-%m-%y %H:%M", "%d-%m-%Y %H:%M",
        # month first (US)
        "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M",
        # 12-hour
        "%d/%m/%y %I:%M %p", "%m/%d/%y %I:%M %p",
        "%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p",
    ]
    combined = f"{date_str} {time_str}".strip()
    for fmt in candidates:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def _classify(text: str) -> MessageKind:
    """Decide what kind of message this is based on content."""
    lower = text.lower()
    if any(p in lower for p in DELETED_PATTERNS):
        return MessageKind.DELETED
    if any(p in lower for p in MEDIA_PATTERNS):
        return MessageKind.MEDIA_PLACEHOLDER
    if any(p in lower for p in SYSTEM_PATTERNS):
        return MessageKind.SYSTEM
    return MessageKind.TEXT


def _read_text(path: Path) -> str:
    """Read the WhatsApp chat text, handling encoding gracefully.

    If it's a ZIP, pick the real chat .txt (iOS `_chat.txt` OR Android
    `WhatsApp Chat with <name>.txt`) by name + content, ignoring any bundled
    .vcf contacts, media, or other files.
    """
    if zipfile.is_zipfile(path):
        from . import zip_utils
        member = zip_utils.find_whatsapp_txt(path)
        if not member:
            raise ValueError(
                "This ZIP doesn't contain a WhatsApp chat .txt — it may be a "
                "folder of media/contacts. Re-export the chat and upload the "
                "exported file.")
        return zip_utils.read_member_text(path, member)

    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def parse(path: Path) -> ParsedChat:
    """Parse a WhatsApp .txt or .zip export."""
    text = _read_text(path)
    lines = text.splitlines()

    messages: list[Message] = []
    warnings: list[str] = []
    senders_seen: set[str] = set()

    current_msg: Message | None = None
    bad_date_count = 0

    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip()
        if not line:
            continue

        match = LINE_RE.match(line)
        if not match:
            # Continuation of the previous message — append to its text
            if current_msg is not None:
                current_msg.text += "\n" + line
            continue

        # Save the previous message before starting a new one
        if current_msg is not None:
            messages.append(current_msg)

        date_str = match.group("date")
        time_str = match.group("time")
        rest = match.group("rest")

        dt = _try_parse_datetime(date_str, time_str)
        if dt is None:
            bad_date_count += 1
            current_msg = None
            continue

        # rest is either "Sender: text" or just "text" (system message)
        if ":" in rest:
            sender, text_content = rest.split(":", 1)
            sender = sender.strip()
            text_content = text_content.strip()
        else:
            sender = "system"
            text_content = rest.strip()

        senders_seen.add(sender)
        current_msg = Message(
            sender=sender,
            timestamp=dt,
            text=text_content,
            kind=_classify(text_content) if sender != "system" else MessageKind.SYSTEM,
        )

    # Don't forget the last message
    if current_msg is not None:
        messages.append(current_msg)

    if bad_date_count:
        warnings.append(f"Could not parse date on {bad_date_count} lines")
    if not messages:
        warnings.append("No messages found — check if this is really a WhatsApp export")

    # Drop the "system" pseudo-sender from the senders list
    real_senders = sorted(s for s in senders_seen if s != "system")

    return ParsedChat(
        messages=messages,
        detected_format="whatsapp",
        senders=real_senders,
        parser_warnings=warnings,
    )
