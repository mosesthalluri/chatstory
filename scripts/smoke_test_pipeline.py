"""Extended smoke test: parse + chunker + highlights + stats.
Confirms the deterministic pipeline (no LLM calls) actually runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Stub settings module before any imports that touch it
import types
stub = types.ModuleType("app.settings")
root = Path(__file__).resolve().parent.parent / "backend"
stub.BACKEND_ROOT = root
stub.PROJECT_ROOT = root.parent
stub.STORAGE_ROOT = root / "storage"
stub.UPLOADS_DIR = root / "storage" / "uploads"
stub.JOBS_DIR = root / "storage" / "jobs"
stub.OUTPUT_DIR = root / "storage" / "output"
stub.TEMPLATES_DIR = root.parent / "frontend" / "templates"
stub.STATIC_DIR = root.parent / "frontend" / "static"
class _S:
    USE_OLLAMA = False
    GROQ_API_KEY = ""
    GEMINI_API_KEY = ""
    CHAPTERS_PER_BOOK = 4
    PREVIEW_CHAPTERS = 1
stub.settings = _S()
sys.modules["app.settings"] = stub

# Try importing emoji — required for stats. If missing, skip stats test.
try:
    import emoji  # noqa
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    print("⚠ emoji library not installed — skipping stats test")

from app.parsers.whatsapp import parse as parse_whatsapp
from app.pipeline import chunker, highlights

sample = root / "storage" / "sample_chat.txt"
parsed = parse_whatsapp(sample)
print(f"✓ Parsed {parsed.message_count} messages\n")

# Chunker
days = chunker.by_day(parsed.messages)
weeks = chunker.by_week(parsed.messages)
months = chunker.by_month(parsed.messages)
chapters = chunker.into_chapters(parsed.messages, n_chapters=4)

print(f"✓ Chunked into:")
print(f"    {len(days)} active days")
print(f"    {len(weeks)} weeks")
print(f"    {len(months)} months")
print(f"    {len(chapters)} chapters")
print()

print("Chapter breakdown:")
for i, (start, end, msgs) in enumerate(chapters, 1):
    print(f"  Chapter {i}: {start} → {end} ({len(msgs)} messages)")
print()

# Highlights
hi = highlights.select_highlights(parsed.messages, n=5)
print(f"✓ Selected {len(hi)} highlights:")
for m in hi:
    text = m.text if len(m.text) < 60 else m.text[:57] + "..."
    print(f"    [{m.timestamp.date()}] {m.sender}: {text}")
print()

# Stats — only if emoji is installed
if HAS_EMOJI:
    from app.pipeline import stats
    s = stats.compute_stats(parsed)
    print("✓ Stats engine ran:")
    print(f"    Messages: {s['total_messages']}")
    print(f"    Days span: {s['days_span']}")
    print(f"    Days active: {s['days_active']}")
    print(f"    Most active hour: {s['most_active_hour']}:00")
    print(f"    Per-sender counts: {s['messages_per_sender']}")
    print(f"    Top emojis: {s['top_emojis'][:3]}")
    if s.get('inside_jokes'):
        print(f"    Inside jokes: {s['inside_jokes']}")

print("\n✓ EXTENDED SMOKE TEST PASSED")
