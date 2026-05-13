"""
Input normalizer. Converts any supported input format into our
canonical intermediate representation:

  [YYYY-MM-DD HH:MM:SS] sender: text

This is what the downstream pipeline expects. The normalizer is the
ONLY trust boundary between messy real-world export files and our
clean, predictable pipeline.

Why a separate visible normalizer step:
  - Users upload anything: WhatsApp .txt, Instagram JSON, manually
    cleaned dumps, PDF exports, even chat screenshots reformatted as
    text. Each has its own noise pattern.
  - When parsing goes wrong (and it WILL on real exports), the user
    needs to see what was extracted before paying for LLM generation.
  - The normalized text is downloadable, inspectable, and reusable.
    A user can normalize once, then re-run the pipeline cheaply.
  - For developers debugging a parser issue, comparing the input
    vs the normalized output isolates the problem fast.

Public API:
    summary, normalized_text = normalize(input_path)
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from ..models import Message, MessageKind, ParsedChat
from ..parsers import parse_chat


class NormalizationSummary(NamedTuple):
    """Compact stats about what came out of normalization. Suitable
    for showing to the user as a sanity check."""
    detected_format: str
    total_raw_messages: int  # before filtering
    text_messages: int       # actual typed messages
    media_messages: int      # reels, attachments, etc.
    filtered_out: int        # reactions, system events, deletions
    senders: list[str]
    senders_count: dict[str, int]
    date_range: tuple[str, str]  # (first, last) ISO dates
    days_span: int
    days_active: int
    parser_warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "detected_format": self.detected_format,
            "total_raw_messages": self.total_raw_messages,
            "text_messages": self.text_messages,
            "media_messages": self.media_messages,
            "filtered_out": self.filtered_out,
            "senders": self.senders,
            "senders_count": self.senders_count,
            "date_range": list(self.date_range),
            "days_span": self.days_span,
            "days_active": self.days_active,
            "parser_warnings": self.parser_warnings,
        }


def _format_message_line(m: Message) -> str:
    """Format a single message in our canonical intermediate format."""
    ts = m.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    if m.kind == MessageKind.MEDIA_PLACEHOLDER:
        # Make media-share status visible in the normalized output too
        return f"[{ts}] {m.sender}: [MEDIA] {m.text}"
    return f"[{ts}] {m.sender}: {m.text}"


def _build_summary(parsed: ParsedChat) -> NormalizationSummary:
    """Build the diagnostic summary from a ParsedChat."""
    messages = parsed.messages
    text_msgs = [m for m in messages if m.kind == MessageKind.TEXT]
    media_msgs = [m for m in messages if m.kind == MessageKind.MEDIA_PLACEHOLDER]

    raw_count = parsed.raw_message_count or len(messages)
    filtered_out = raw_count - len(messages)

    sender_counter = Counter(m.sender for m in messages)

    if messages:
        first = messages[0].timestamp
        last = messages[-1].timestamp
        days_span = max((last.date() - first.date()).days + 1, 1)
        days_active = len({m.timestamp.date() for m in messages})
        date_range = (first.date().isoformat(), last.date().isoformat())
    else:
        days_span = 0
        days_active = 0
        date_range = ("", "")

    return NormalizationSummary(
        detected_format=parsed.detected_format,
        total_raw_messages=raw_count,
        text_messages=len(text_msgs),
        media_messages=len(media_msgs),
        filtered_out=max(filtered_out, 0),
        senders=parsed.senders,
        senders_count=dict(sender_counter),
        date_range=date_range,
        days_span=days_span,
        days_active=days_active,
        parser_warnings=parsed.parser_warnings,
    )


def normalize(input_path: Path) -> tuple[NormalizationSummary, str]:
    """Normalize any supported input to (summary, canonical_text).

    Raises ValueError if the format can't be detected at all.
    """
    parsed = parse_chat(input_path)
    summary = _build_summary(parsed)
    canonical_text = "\n".join(_format_message_line(m) for m in parsed.messages)
    return summary, canonical_text


def write_normalized_outputs(
    input_path: Path, output_dir: Path
) -> tuple[NormalizationSummary, Path, Path]:
    """Normalize and persist outputs to disk.

    Writes two files into output_dir:
      - normalized.txt — canonical-format message list
      - summary.json   — diagnostic summary

    Returns: (summary, normalized_txt_path, summary_json_path).
    Use this in the upload pipeline so users can download the
    normalized representation before paying for chapter generation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, canonical_text = normalize(input_path)

    txt_path = output_dir / "normalized.txt"
    txt_path.write_text(canonical_text, encoding="utf-8")

    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary, txt_path, json_path
