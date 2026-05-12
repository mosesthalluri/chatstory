"""
Telegram JSON export parser.

Telegram Desktop exports look like:

    {
      "name": "Aarav",
      "type": "personal_chat",
      "messages": [
        {
          "id": 1,
          "type": "message",
          "date": "2024-03-15T21:43:00",
          "from": "Aarav",
          "text": "hey",
        },
        {
          "id": 2,
          "type": "service",
          "action": "phone_call",
          "duration_seconds": 124
        }
      ]
    }

Multi-part text appears as a list of strings or dicts:
    "text": [
      "Check this out: ",
      {"type": "link", "text": "https://example.com"}
    ]
"""

import json
from datetime import datetime
from pathlib import Path

from ..models import Message, MessageKind, ParsedChat


def _extract_text(text_field) -> str:
    """Telegram's `text` is sometimes a string, sometimes a list of mixed
    strings and dicts. Flatten it to plain text.
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def parse(path: Path) -> ParsedChat:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    messages: list[Message] = []
    senders: set[str] = set()
    warnings: list[str] = []

    raw_messages = data.get("messages", [])

    for raw in raw_messages:
        msg_type = raw.get("type", "message")
        date_str = raw.get("date") or raw.get("date_unixtime")
        if not date_str:
            continue

        try:
            if isinstance(date_str, str):
                ts = datetime.fromisoformat(date_str)
            else:
                ts = datetime.fromtimestamp(int(date_str))
        except (ValueError, TypeError):
            continue

        sender = raw.get("from") or raw.get("actor") or "unknown"
        senders.add(sender)

        if msg_type == "service":
            action = raw.get("action", "system event")
            kind = MessageKind.SYSTEM
            if action == "phone_call":
                duration = raw.get("duration_seconds", 0)
                text = f"[phone call - {duration}s]"
                kind = MessageKind.MEDIA_PLACEHOLDER
            else:
                text = f"[{action}]"
        else:
            text = _extract_text(raw.get("text", ""))
            if not text:
                # Could be media-only
                if "photo" in raw:
                    text = "[photo]"
                    kind = MessageKind.MEDIA_PLACEHOLDER
                elif "voice_message" in raw or "audio_file" in raw:
                    text = "[voice message]"
                    kind = MessageKind.MEDIA_PLACEHOLDER
                elif "video_file" in raw or "video_message" in raw:
                    text = "[video]"
                    kind = MessageKind.MEDIA_PLACEHOLDER
                elif "sticker_emoji" in raw:
                    text = raw["sticker_emoji"]
                    kind = MessageKind.TEXT
                elif "file" in raw:
                    text = "[file]"
                    kind = MessageKind.MEDIA_PLACEHOLDER
                else:
                    continue
            else:
                kind = MessageKind.TEXT

        messages.append(Message(
            sender=sender,
            timestamp=ts,
            text=text,
            kind=kind,
            message_id=str(raw.get("id", "")) or None,
        ))

    return ParsedChat(
        messages=messages,
        detected_format="telegram",
        senders=sorted(senders),
        parser_warnings=warnings,
    )
