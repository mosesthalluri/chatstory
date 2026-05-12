"""
Generic text parser. Last resort for unrecognized text formats and also
the workhorse for manually-cleaned exports.

Real-world cleaned exports (especially from Instagram) are messier than
they look. Common problems we handle:

  1. Timestamp lines have timezone suffixes: "[2025-08-08 04:29:00 IST]".
     `strptime` doesn't grok "IST"/"UTC"/"GMT" suffixes — we strip them.

  2. Lines get glued together with no separator. An Instagram reel share
     ends with "#hashtag1#hashtag2_https://insta..." and the very next
     character is the next message's "[2025-..." timestamp. We split
     these in a pre-pass before the line-by-line parser sees them.

  3. Sender names contain emojis ("✨Kristy_honey✨"). The regex allows
     any non-space, non-bracket, non-colon character in the sender.

  4. Noise types that should be filtered, not used:
     - "Liked a message" / "Reacted ❤ to your message"  (reactions)
     - "Mr sent an attachment." (often followed by reel share garbage)
     - URL-only lines (reel shares, post shares)
     - Hashtag-only lines (often used as filler between media shares)
     - Standalone "." lines (often a separator between media shares)

Strategy is intentionally lenient because this is the fallback. Better to
over-include garbage than to reject a valid file. The downstream stats
engine sees MessageKind tags and ignores noise.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

from ..models import Message, MessageKind, ParsedChat


# ---------------------------------------------------------------------------
# Line patterns. Order matters — try the most specific first.
# ---------------------------------------------------------------------------

# Allow any non-space, non-bracket, non-colon, non-newline char in sender.
# This includes emojis (✨), Unicode, dots, underscores, etc. Bounded
# length so a runaway sentence can't be mistaken for a sender.
_SENDER_CHARS = r"[^\s:\[\]<>\n][^:\[\]<>\n]{0,48}[^\s:\[\]<>\n]|[^\s:\[\]<>\n]"

PATTERNS = [
    # "[2025-08-08 04:29:00 IST] ✨Kristy_honey✨: Hmm"
    # This is the Instagram-cleaned format we see in real exports.
    re.compile(
        rf"^\[(?P<ts>[^\]]+)\]\s+(?P<sender>{_SENDER_CHARS})\s*:\s*(?P<text>.*)$"
    ),
    # "Alice 😀 [2024-03-15 21:43]: hey there"
    re.compile(
        rf"^(?P<sender>{_SENDER_CHARS})\s*\[(?P<ts>[^\]]+)\]\s*:\s*(?P<text>.*)$"
    ),
    # "2024-03-15 21:43 - Alice 😀: hey"
    re.compile(
        rf"^(?P<ts>\d{{1,4}}[-/.]\d{{1,2}}[-/.]\d{{1,4}}[ T]\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\s*[AaPp][Mm])?)\s*[-–—]\s*(?P<sender>{_SENDER_CHARS})\s*:\s*(?P<text>.*)$"
    ),
    # "21:43 - Alice 😀: hey"  (time only, no date — inherit prev date)
    re.compile(
        rf"^(?P<ts>\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\s*[AaPp][Mm])?)\s*[-–—]\s*(?P<sender>{_SENDER_CHARS})\s*:\s*(?P<text>.*)$"
    ),
    # "Alice 😀: hey there"  (no timestamp — synthesise)
    re.compile(rf"^(?P<sender>{_SENDER_CHARS}):\s+(?P<text>.+)$"),
]


# A bracketed timestamp anywhere in a line. Used to split glued lines.
# Catches "[2025-08-08 04:29:00 IST]" / "[2025-08-08 04:29 IST]" /
# "[2025-08-08 04:29]" etc. Requires year-month-day-hour-minute at minimum.
RE_BRACKET_TS = re.compile(
    r"\[\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\s*[A-Za-z]{2,5})?\]"
)


# Timezone suffixes we'll strip before strptime. Add more if you hit them.
TZ_SUFFIXES = (
    "IST", "UTC", "GMT", "EST", "EDT", "PST", "PDT", "CST", "CDT",
    "MST", "MDT", "BST", "CET", "CEST", "JST", "AEST", "AEDT",
    "PT", "ET", "CT", "MT",
)
_RE_TZ = re.compile(
    r"\s+(?:" + "|".join(TZ_SUFFIXES) + r")\b\s*$"
)


# Things that look like messages but should be tagged as noise.
RE_REACTION = re.compile(
    r"^\s*(?:Liked a message|Reacted .{1,5} to|Disliked a message)",
    re.IGNORECASE,
)
RE_ATTACHMENT_PREFIX = re.compile(
    r"^\s*(?:\S+\s+)?sent an attachment\.?\s*",
    re.IGNORECASE,
)
RE_URL_ONLY = re.compile(
    r"^\s*(?:https?://\S+|[#@]\S+|[#@]\S+\s+)+\s*$"
)
# Matches lines that are mostly a URL with at most a short attribution
# prefix (up to 30 chars, may contain spaces). Catches things like:
#   "Follow jjd.joydip__https://www.instagram.com/reel/..."
#   "Clips - @vaani_iitbrahgiirrhttps://..."
#   "Source https://..."
# Uses `.` (any char) rather than `\S` so spaces in the prefix don't break it.
RE_URL_WITH_SHORT_PREFIX = re.compile(
    r"^\s*.{0,30}https?://\S+\s*$"
)
RE_HASHTAGS_ONLY = re.compile(r"^\s*(?:#\S+\s*)+\s*$")
RE_DOTS_ONLY = re.compile(r"^\s*[.•·\-_]{1,5}\s*$")


# Timestamp formats. Listed from most to least specific.
TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M",
    "%d/%m/%y %H:%M", "%d-%m-%y %H:%M", "%m/%d/%y %H:%M",
    "%Y-%m-%d %I:%M %p", "%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p",
    "%d/%m/%y %I:%M %p", "%m/%d/%y %I:%M %p",
    "%H:%M:%S", "%H:%M", "%I:%M %p",
]


def _strip_tz(ts_str: str) -> str:
    return _RE_TZ.sub("", ts_str).strip()


def _try_parse_ts(s: str, prev_date: datetime | None = None) -> datetime | None:
    """Strip timezone suffix, normalize AM/PM, try every known format.
    If only a time-of-day matches and we have a previous date, attach it."""
    s = _strip_tz(s.strip())
    s = re.sub(r"\b([AaPp])([Mm])\b", lambda m: m.group().upper(), s)
    s = re.sub(r"(\d)\s*(AM|PM)\b", r"\1 \2", s)

    for fmt in TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year == 1900 and prev_date:
                dt = dt.replace(
                    year=prev_date.year, month=prev_date.month, day=prev_date.day
                )
            return dt
        except ValueError:
            continue
    return None


def _split_joined_lines(text: str) -> str:
    """Insert newlines wherever a bracketed timestamp appears mid-line.
    Handles the Instagram-export pattern where reel-share garbage runs
    directly into the next message's [timestamp]."""
    return RE_BRACKET_TS.sub(lambda m: "\n" + m.group(0), text)


def _classify_noise(text: str) -> MessageKind | None:
    """Return a MessageKind tag if the text is noise, else None."""
    if RE_REACTION.match(text):
        return MessageKind.REACTION
    stripped = RE_ATTACHMENT_PREFIX.sub("", text).strip()
    if not stripped:
        return MessageKind.MEDIA_PLACEHOLDER
    if RE_URL_ONLY.match(stripped) or RE_HASHTAGS_ONLY.match(stripped):
        return MessageKind.MEDIA_PLACEHOLDER
    if RE_URL_WITH_SHORT_PREFIX.match(stripped):
        return MessageKind.MEDIA_PLACEHOLDER
    if RE_DOTS_ONLY.match(stripped):
        return MessageKind.MEDIA_PLACEHOLDER
    return None


def _clean_text(text: str) -> str:
    """Strip 'X sent an attachment.' prefix and surrounding hashtag/URL
    spam from real message text. Returns cleaned text or empty if
    everything was noise."""
    text = RE_ATTACHMENT_PREFIX.sub("", text).strip()
    parts = text.split("\n")
    kept = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if RE_DOTS_ONLY.match(p):
            continue
        if RE_HASHTAGS_ONLY.match(p):
            continue
        if RE_URL_ONLY.match(p):
            continue
        if RE_URL_WITH_SHORT_PREFIX.match(p):
            continue
        kept.append(p)
    return " ".join(kept).strip()


def parse(path: Path) -> ParsedChat:
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    # Pre-pass: split lines that got glued together.
    text = _split_joined_lines(text)

    lines = text.splitlines()
    messages: list[Message] = []
    senders: set[str] = set()
    warnings: list[str] = []

    line_count = 0
    matched_count = 0
    continuation_count = 0
    used_synthetic_ts = 0
    noise_filtered = 0

    fake_ts = datetime(2024, 1, 1, 9, 0, 0)
    fake_step = timedelta(minutes=1)
    current_msg: Message | None = None
    last_real_date: datetime | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue
        line_count += 1

        matched = False
        for pat in PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            groups = m.groupdict()
            sender = groups["sender"].strip()
            text_content = groups.get("text", "").strip()
            ts_str = groups.get("ts")

            if not sender or sender.count(" ") > 4:
                continue
            if re.fullmatch(r"[\d\s\-/:.TPMam]+", sender):
                continue

            ts = _try_parse_ts(ts_str, prev_date=last_real_date) if ts_str else None
            if ts is None:
                ts = fake_ts
                fake_ts += fake_step
                used_synthetic_ts += 1
            else:
                last_real_date = ts

            # Classify and clean text
            kind = _classify_noise(text_content) or MessageKind.TEXT
            if kind == MessageKind.TEXT:
                cleaned = _clean_text(text_content)
                if not cleaned:
                    kind = MessageKind.MEDIA_PLACEHOLDER
                text_content = cleaned
            if kind != MessageKind.TEXT:
                noise_filtered += 1

            if current_msg is not None:
                messages.append(current_msg)
            senders.add(sender)
            current_msg = Message(
                sender=sender, timestamp=ts, text=text_content, kind=kind,
            )
            matched = True
            matched_count += 1
            break

        if not matched:
            if current_msg is not None and current_msg.kind == MessageKind.TEXT:
                # Only append to text messages. Don't pollute media placeholders.
                cleaned = _clean_text(line)
                if cleaned:
                    current_msg.text += " " + cleaned
                continuation_count += 1

    if current_msg is not None:
        messages.append(current_msg)

    # Post-pass: strip trailing sender-name suffixes from message bodies.
    # Instagram's export sometimes glues the recipient's name onto the end
    # of replied-to messages (e.g. "Okay 🥀❤✨Kristy_honey✨" where the
    # text was just "Okay 🥀❤" and ✨Kristy_honey✨ is reply-target context).
    # Without this pass, the recipient's emoji frame leaks into stats as
    # the "most used emoji."
    if len(senders) >= 2:
        # Try longest names first so we strip the full suffix, not a part.
        sender_list = sorted(senders, key=len, reverse=True)
        for msg in messages:
            if msg.kind != MessageKind.TEXT:
                continue
            for sender_name in sender_list:
                if sender_name == msg.sender:
                    continue  # don't strip own name
                # Only strip if the name is at the very end and not the
                # entire message body
                if msg.text.endswith(sender_name) and len(msg.text) > len(sender_name):
                    msg.text = msg.text[:-len(sender_name)].rstrip()
                    break

    # Quality warnings
    if line_count > 0:
        match_rate = matched_count / line_count
        if match_rate < 0.4:
            warnings.append(
                f"Only {matched_count}/{line_count} lines ({match_rate:.0%}) "
                f"matched a chat-line pattern. Your file format may not be "
                f"recognized. Expected format example: "
                f"'[2025-08-08 04:29:00] Alice: hello'"
            )
        if used_synthetic_ts > matched_count * 0.5 and matched_count > 0:
            warnings.append(
                f"{used_synthetic_ts}/{matched_count} messages had no parseable "
                f"timestamp and got synthetic dates. Days-span will be wrong. "
                f"Check timestamp format — supported examples: "
                f"'YYYY-MM-DD HH:MM:SS' (timezone suffix like 'IST' is fine)."
            )
        if noise_filtered > 0:
            warnings.append(
                f"Filtered {noise_filtered} noise messages "
                f"(reactions, attachments, URL/hashtag-only lines)."
            )

    if not messages:
        warnings.append("Couldn't extract any messages — using fallback")
        if text.strip():
            messages.append(Message(
                sender="narrator", timestamp=datetime(2024, 1, 1),
                text=text[:50000], kind=MessageKind.TEXT,
            ))
            senders.add("narrator")

    return ParsedChat(
        messages=messages,
        detected_format="generic_text",
        senders=sorted(senders),
        parser_warnings=warnings,
    )
