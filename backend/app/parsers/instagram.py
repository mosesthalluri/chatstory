"""
Instagram JSON export parser.

Instagram's data download produces JSON files like:

    {
      "participants": [{"name": "alice"}, {"name": "bob"}],
      "messages": [
        {
          "sender_name": "alice",
          "timestamp_ms": 1675123456789,
          "content": "hey",
        },
        {
          "sender_name": "bob",
          "timestamp_ms": 1675123478901,
          "share": {"link": "https://instagram.com/reel/..."}
        },
        {
          "sender_name": "alice",
          "timestamp_ms": 1675123500000,
          "reactions": [{"reaction": "❤", "actor": "bob"}]
        }
      ]
    }

For long conversations, Instagram splits across message_1.json,
message_2.json, etc. inside a ZIP. We handle both.

Important quirks:
  - Strings are mojibake-encoded (UTF-8 bytes interpreted as Latin-1).
    We have to undo this. Without it, "héllo" becomes "hÃ©llo".
  - Reactions are separate messages — usually filtered out.
  - Shared posts/reels appear with `share` key, not `content`.
"""

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

from ..models import Message, MessageKind, ParsedChat


def _fix_mojibake(text: str) -> str:
    """Instagram exports text as UTF-8 bytes interpreted as Latin-1.
    To fix: re-encode as Latin-1, then decode as UTF-8.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _extract_text(msg_json: dict) -> tuple[str, MessageKind]:
    """Get the message body and figure out its kind."""
    # Reactions are stored as their own messages with `reactions` key
    if "reactions" in msg_json and "content" not in msg_json:
        return "", MessageKind.REACTION

    if "content" in msg_json:
        content = _fix_mojibake(msg_json["content"])
        # Instagram uses these strings for system/deleted events
        if "unsent a message" in content.lower():
            return content, MessageKind.DELETED
        if content.startswith("Liked a message") or content.endswith("liked a message"):
            return "", MessageKind.REACTION
        return content, MessageKind.TEXT

    # Shared content: posts, reels, stories
    if "share" in msg_json:
        share = msg_json["share"]
        if isinstance(share, dict):
            if "share_text" in share:
                return f"[shared a post: {_fix_mojibake(share['share_text'])}]", MessageKind.MEDIA_PLACEHOLDER
            return "[shared a post]", MessageKind.MEDIA_PLACEHOLDER

    if "photos" in msg_json:
        return "[photo]", MessageKind.MEDIA_PLACEHOLDER
    if "videos" in msg_json:
        return "[video]", MessageKind.MEDIA_PLACEHOLDER
    if "audio_files" in msg_json:
        return "[voice message]", MessageKind.MEDIA_PLACEHOLDER
    if "gifs" in msg_json:
        return "[gif]", MessageKind.MEDIA_PLACEHOLDER

    if msg_json.get("call_duration") is not None:
        duration = msg_json["call_duration"]
        return f"[video call - {duration}s]", MessageKind.MEDIA_PLACEHOLDER

    return "", MessageKind.SYSTEM


def _parse_one_json(data: dict, messages: list[Message], senders: set[str]) -> None:
    """Parse one JSON file's messages into the messages list."""
    raw_messages = data.get("messages", [])

    for raw in raw_messages:
        sender = _fix_mojibake(raw.get("sender_name", "unknown"))
        ts_ms = raw.get("timestamp_ms")
        if ts_ms is None:
            continue
        try:
            ts = datetime.fromtimestamp(ts_ms / 1000)
        except (ValueError, OSError, OverflowError):
            continue

        text, kind = _extract_text(raw)

        # Skip pure reactions — they're noise we can't show as messages
        if kind == MessageKind.REACTION and not text:
            continue

        senders.add(sender)
        messages.append(Message(
            sender=sender,
            timestamp=ts,
            text=text,
            kind=kind,
        ))


def parse(path: Path) -> ParsedChat:
    """Parse Instagram JSON export. Path can be a .json file or .zip."""
    messages: list[Message] = []
    senders: set[str] = set()
    warnings: list[str] = []

    if zipfile.is_zipfile(path):
        from . import zip_utils
        json_files = zip_utils.find_instagram_jsons(path)
        if not json_files:
            raise ValueError(
                "This ZIP doesn't contain Instagram message JSONs (message_1.json …). "
                "Upload the message JSON, or the ZIP that contains it.")
        for name in json_files:
            try:
                data = json.loads(zip_utils.read_member_bytes(path, name))
                _parse_one_json(data, messages, senders)
            except json.JSONDecodeError as e:
                warnings.append(f"Could not parse {name}: {e}")
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        _parse_one_json(data, messages, senders)

    # Instagram returns messages newest-first. We want oldest-first.
    messages.sort(key=lambda m: m.timestamp)

    return ParsedChat(
        messages=messages,
        detected_format="instagram",
        senders=sorted(senders),
        parser_warnings=warnings,
    )
