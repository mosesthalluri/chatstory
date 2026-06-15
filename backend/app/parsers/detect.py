"""
Format detection. Given a file path, decide which parser to use.

The strategy is "sniff and route":
  1. Look at the file extension and a sample of content.
  2. Match against known signatures.
  3. Return the parser name (string), or "unknown" if nothing matches.

Adding a new format = adding a new signature here + a new parser file.
"""

import json
import re
import zipfile
from pathlib import Path


# Signature: (parser_name, predicate_function)
# Listed in priority order — first match wins.

def _is_zip(path: Path) -> bool:
    return zipfile.is_zipfile(path)


def _read_sample(path: Path, max_bytes: int = 8192) -> str:
    """Read a small sample for sniffing. Handles binary gracefully."""
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes)
        # Try UTF-8 first, then fall back to latin-1 which never fails
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="replace")
    except Exception:
        return ""


def _looks_like_whatsapp_txt(sample: str) -> bool:
    """WhatsApp .txt exports start with timestamped lines.

    Examples of formats we accept:
        15/01/25, 21:43 - Aarav: hey
        [15/01/25, 21:43:00] Aarav: hey
        15.1.25, 21:43 - Aarav: hey
        1/15/25, 9:43 PM - Aarav: hey
    """
    patterns = [
        r"^\[?\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4},?\s+\d{1,2}:\d{2}",
        r"^\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}",
    ]
    lines = sample.splitlines()[:20]
    matches = sum(1 for line in lines for p in patterns if re.match(p, line.strip()))
    return matches >= 3


def _looks_like_instagram_json(sample: str, path: Path) -> bool:
    """Instagram JSON has a 'messages' array and 'participants' field."""
    try:
        # Sample is partial; for JSON we need to actually parse, or check shape
        if not sample.lstrip().startswith(("{", "[")):
            return False
        # Try parsing the whole file (most Instagram message files are <50MB
        # per conversation; large enough to try)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return "messages" in data and "participants" in data
        return False
    except (json.JSONDecodeError, OSError, MemoryError):
        return False


def _looks_like_telegram_json(sample: str, path: Path) -> bool:
    """Telegram JSON has 'name' (chat name) and 'messages' at the top level."""
    try:
        if not sample.lstrip().startswith("{"):
            return False
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return "messages" in data and ("name" in data or "type" in data)
        return False
    except (json.JSONDecodeError, OSError, MemoryError):
        return False


def _is_zip_with_chat(path: Path) -> tuple[bool, str | None]:
    """Some platforms ship as ZIPs (WhatsApp, Instagram, Telegram), often
    bundled with unrelated files (.vcf contacts, media, PDFs). We find the real
    chat member by name + content sniff, ignoring everything else.

    Returns (True, suggested_format) if we recognize the contents,
    otherwise (False, None).
    """
    if not _is_zip(path):
        return False, None
    try:
        from . import zip_utils
        if zip_utils.find_whatsapp_txt(path):
            return True, "whatsapp_zip"
        if zip_utils.find_instagram_jsons(path):
            return True, "instagram_zip"
        if zip_utils.find_telegram_json(path):
            return True, "telegram_zip"
    except zipfile.BadZipFile:
        return False, None
    return False, None


def detect_format(path: Path) -> str:
    """Return one of: 'whatsapp_txt', 'whatsapp_zip', 'instagram_json',
    'instagram_zip', 'telegram_json', 'generic_txt', 'unknown'."""

    is_zip, zip_format = _is_zip_with_chat(path)
    if is_zip and zip_format:
        return zip_format

    # A ZIP we couldn't find a chat inside: do NOT fall through to the text
    # parser (it would read the raw ZIP bytes as a chat and emit garbage —
    # e.g. a single message dated today, looking like a "1-day" conversation).
    if _is_zip(path):
        return "unknown"

    sample = _read_sample(path)
    if not sample:
        return "unknown"

    # JSON formats — check before WhatsApp because JSON files don't match
    # WhatsApp's regex but we want to be sure about JSON shape
    suffix = path.suffix.lower()
    if suffix == ".json" or sample.lstrip().startswith(("{", "[")):
        if _looks_like_instagram_json(sample, path):
            return "instagram_json"
        if _looks_like_telegram_json(sample, path):
            return "telegram_json"

    # WhatsApp text
    if _looks_like_whatsapp_txt(sample):
        return "whatsapp_txt"

    # Plain text fallback
    if suffix in (".txt", "") or sample.isprintable() or "\n" in sample:
        return "generic_txt"

    return "unknown"
