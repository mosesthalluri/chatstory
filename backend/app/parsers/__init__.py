"""
Parser orchestrator and noise filter.

Public API:
    result = parse_chat(path)  # returns ParsedChat with cleaned messages
"""

from datetime import timedelta
from pathlib import Path

from ..models import Message, MessageKind, ParsedChat
from .detect import detect_format
from . import whatsapp, instagram, telegram, generic


PARSERS = {
    "whatsapp_txt": whatsapp,
    "whatsapp_zip": whatsapp,
    "instagram_json": instagram,
    "instagram_zip": instagram,
    "telegram_json": telegram,
    "generic_txt": generic,
}


def _filter_noise(messages: list[Message]) -> list[Message]:
    """Drop pure reactions, system events, deletions. Keep media
    placeholders since they're useful narrative context."""
    return [
        m for m in messages
        if m.kind in (MessageKind.TEXT, MessageKind.MEDIA_PLACEHOLDER)
        and m.text.strip()
    ]


def _sort_chronologically(messages: list[Message]) -> tuple[list[Message], bool]:
    """Return messages oldest-first while preserving sensible tie order.

    Some exports, especially manually cleaned Instagram text dumps, arrive
    newest-first. The rest of the pipeline assumes chronological order for
    response-time stats, chapter ranges, and consecutive-turn merging. If the
    input mostly moves backward in time, reverse it first so messages with the
    same timestamp keep their likely conversational order after sorting.
    """
    if len(messages) < 2:
        return messages, False

    increases = 0
    decreases = 0
    for prev, cur in zip(messages, messages[1:]):
        if cur.timestamp > prev.timestamp:
            increases += 1
        elif cur.timestamp < prev.timestamp:
            decreases += 1

    working = list(reversed(messages)) if decreases > increases else list(messages)
    ordered = [
        msg for _, msg in sorted(
            enumerate(working),
            key=lambda item: (item[1].timestamp, item[0]),
        )
    ]
    changed = any(a is not b for a, b in zip(messages, ordered))
    return ordered, changed


def _merge_consecutive(messages: list[Message], gap_seconds: int = 30) -> list[Message]:
    """Merge messages from the same sender within `gap_seconds` of each
    other. Cuts message count for fast typers without losing content."""
    if not messages:
        return messages

    merged = [messages[0]]
    for msg in messages[1:]:
        last = merged[-1]
        if (
            msg.sender == last.sender
            and (msg.timestamp - last.timestamp) <= timedelta(seconds=gap_seconds)
            and msg.kind == last.kind == MessageKind.TEXT
        ):
            last.text = last.text + " " + msg.text
            last.timestamp = msg.timestamp
        else:
            merged.append(msg)
    return merged


def parse_chat(path: Path) -> ParsedChat:
    """Parse a chat file. Returns a normalized, cleaned ParsedChat."""
    fmt = detect_format(path)
    if fmt == "unknown":
        raise ValueError(
            "Could not detect chat format. Supported: WhatsApp .txt or .zip, "
            "Instagram JSON, Telegram JSON, or plain text with "
            "'Sender: message' lines."
        )

    parser = PARSERS.get(fmt)
    if parser is None:
        raise ValueError(f"No parser registered for format: {fmt}")

    result = parser.parse(path)
    if not result.messages:
        raise ValueError(
            f"Detected '{fmt}' but extracted no messages. "
            f"Warnings: {result.parser_warnings}"
        )

    original_count = len(result.messages)
    ordered, reordered = _sort_chronologically(result.messages)
    cleaned = _filter_noise(ordered)
    pre_merge_count = len(cleaned)
    cleaned = _merge_consecutive(cleaned)

    parser_warnings = list(result.parser_warnings)
    if reordered:
        parser_warnings.append("Reordered messages chronologically")

    return ParsedChat(
        messages=cleaned,
        detected_format=result.detected_format,
        senders=result.senders,
        raw_message_count=pre_merge_count,
        parser_warnings=parser_warnings + [
            f"Filtered out {original_count - pre_merge_count} noise messages",
            f"After merging consecutive turns: {len(cleaned)} messages",
        ],
    )
