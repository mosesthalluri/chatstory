"""
Robust chat-file discovery inside ZIP exports.

WhatsApp/Instagram/Telegram exports are often shipped as ZIPs that ALSO contain
unrelated files — shared contacts (.vcf), images, videos, PDFs, etc. Picking the
wrong member (or, worse, feeding the raw ZIP bytes to the text parser) produced
nonsense like "the whole conversation happened in one day".

This module finds the *actual* chat file regardless of its name (iOS uses
`_chat.txt`; Android uses `WhatsApp Chat with <name>.txt`) by combining a name
hint with a content sniff, and ignores everything else in the archive.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

# A WhatsApp message line: "15/01/25, 21:43 - ", "[15/01/25, 21:43:00] ", etc.
_WA_LINE = re.compile(
    r"(?m)^\[?\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4},?\s+\d{1,2}:\d{2}"
)


def _real_members(z: zipfile.ZipFile) -> list[str]:
    """Files in the archive, excluding directories and macOS resource forks."""
    out = []
    for info in z.infolist():
        name = info.filename
        if info.is_dir():
            continue
        if "__MACOSX" in name or name.rsplit("/", 1)[-1].startswith("._"):
            continue
        out.append(name)
    return out


def _decode(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def find_whatsapp_txt(path: Path) -> str | None:
    """Return the member name of the WhatsApp chat .txt inside the ZIP, or None.

    Scores every .txt by name hint + how many lines look like WhatsApp
    messages, so the right file wins even when the archive is full of media and
    .vcf contacts."""
    try:
        with zipfile.ZipFile(path) as z:
            txts = [n for n in _real_members(z) if n.lower().endswith(".txt")]
            best: str | None = None
            best_score = 0
            for n in txts:
                low = n.lower().rsplit("/", 1)[-1]
                score = 0
                if low == "_chat.txt" or low.endswith("_chat.txt"):
                    score += 50
                if low.startswith("whatsapp chat") or "whatsapp" in low:
                    score += 30
                try:
                    sample = _decode(z.read(n)[:65536])
                    score += len(_WA_LINE.findall(sample))
                except Exception:
                    pass
                if score > best_score:
                    best_score, best = score, n
            # Require real evidence this is a chat (a few timestamped lines or a
            # strong name hint), so we never pick a random .txt.
            return best if best_score >= 3 else None
    except (zipfile.BadZipFile, OSError):
        return None


def _json_members(z: zipfile.ZipFile) -> list[str]:
    return [n for n in _real_members(z) if n.lower().endswith(".json")]


def _looks_like(z: zipfile.ZipFile, name: str, needs: tuple[str, ...]) -> bool:
    """Cheap shape check: does this JSON member contain the given top-level keys?"""
    try:
        head = _decode(z.read(name)[:8192])
        return all(f'"{k}"' in head for k in needs)
    except Exception:
        return False


def find_instagram_jsons(path: Path) -> list[str]:
    """Instagram message JSONs (message_1.json, message_2.json, …), in order.
    Falls back to any JSON that has a 'messages' array."""
    try:
        with zipfile.ZipFile(path) as z:
            numbered = [n for n in _real_members(z) if re.search(r"message_\d+\.json$", n)]
            if numbered:
                return sorted(
                    numbered,
                    key=lambda n: int(re.search(r"message_(\d+)\.json", n).group(1)),
                )
            return [n for n in _json_members(z)
                    if _looks_like(z, n, ("messages", "participants"))]
    except (zipfile.BadZipFile, OSError):
        return []


def find_telegram_json(path: Path) -> str | None:
    """Telegram Desktop export JSON inside a ZIP (usually result.json)."""
    try:
        with zipfile.ZipFile(path) as z:
            for n in _real_members(z):
                if n.lower().rsplit("/", 1)[-1] == "result.json":
                    return n
            for n in _json_members(z):
                if _looks_like(z, n, ("messages",)) and (
                    _looks_like(z, n, ("name",)) or _looks_like(z, n, ("type",))
                ):
                    return n
    except (zipfile.BadZipFile, OSError):
        return None
    return None


def read_member_text(path: Path, member: str) -> str:
    with zipfile.ZipFile(path) as z:
        return _decode(z.read(member))


def read_member_bytes(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as z:
        return z.read(member)
