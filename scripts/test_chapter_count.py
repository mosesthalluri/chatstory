"""Show what auto chapter-count picks for different chat shapes.

Run: python scripts/test_chapter_count.py
"""
import sys, types
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.modules["httpx"] = types.ModuleType("httpx")

from app.models import Message, MessageKind
from app.pipeline.chunker import suggest_chapter_count


def make_messages(start: datetime, days: int, msgs_per_active_day: int, active_ratio: float = 1.0):
    """Generate fake messages: 'days' calendar span, with `active_ratio`
    of those days actually having messages."""
    msgs = []
    end = start + timedelta(days=days)
    cur = start
    while cur < end:
        # Skip some days based on active_ratio
        if (cur - start).days % max(1, int(1/active_ratio)) == 0:
            for i in range(msgs_per_active_day):
                msgs.append(Message(
                    sender="alice", timestamp=cur + timedelta(minutes=i),
                    text="hi", kind=MessageKind.TEXT,
                ))
        cur += timedelta(days=1)
    return msgs


scenarios = [
    ("Single-day intense convo (200 msgs in 1 day)",
     make_messages(datetime(2025, 8, 1), 1, 200)),
    ("One week of daily texts",
     make_messages(datetime(2025, 8, 1), 7, 50)),
    ("One month of mostly-daily chat",
     make_messages(datetime(2025, 8, 1), 30, 80)),
    ("Three months of daily chat",
     make_messages(datetime(2025, 6, 1), 90, 80)),
    ("Six months of daily chat",
     make_messages(datetime(2025, 3, 1), 180, 80)),
    ("One year of mostly-daily chat",
     make_messages(datetime(2024, 9, 1), 365, 80, active_ratio=0.85)),
    ("Two years (LDR-style, ~700 active days)",
     make_messages(datetime(2023, 9, 1), 730, 60, active_ratio=0.90)),
    ("Two-and-a-half years (your real file shape)",
     make_messages(datetime(2023, 3, 1), 900, 100, active_ratio=0.85)),
    ("Five years of chat",
     make_messages(datetime(2020, 8, 1), 1825, 30, active_ratio=0.70)),
    ("Sparse chat (every other day for a year)",
     make_messages(datetime(2024, 9, 1), 365, 30, active_ratio=0.50)),
]

print(f"{'Scenario':<55}{'Span':<10}{'Active':<10}{'Suggested'}")
print("-" * 90)
for name, msgs in scenarios:
    if not msgs:
        continue
    span = (msgs[-1].timestamp.date() - msgs[0].timestamp.date()).days + 1
    active = len({m.timestamp.date() for m in msgs})
    suggested = suggest_chapter_count(msgs)
    print(f"{name:<55}{span:<10}{active:<10}{suggested}")
print()
print("AUTO mode (CHAPTERS_PER_BOOK=0) would use the 'Suggested' column.")
