"""Regression test for dense one-night chats.

The user's real file is only one hour of conversation, but after merging it
still has 80+ text turns. The old highlight selector treated that as "long"
and then applied a fixed 15-minute diversity window, leaving only two lines
for the chapter prompt. That made the generated story miss the central thread:
Proton asking for a picture, Kristy refusing because didi was there, her phone
was low, and she did not feel like sending it in that situation.

Run: python scripts/test_dense_photo_thread.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.models import MessageKind
from app.parsers import parse_chat
from app.pipeline import highlights
from app.pipeline.chapter_gen import _rough_english_hint, _split_for_commentary


INPUT = Path(r"C:\Projects\txtchat\test.txt")

parsed = parse_chat(INPUT)
text_messages = [m for m in parsed.messages if m.kind == MessageKind.TEXT]
picks = highlights.select_highlights(parsed.messages, n=40)
commentary_chunks = _split_for_commentary(parsed.messages)

print(f"Parsed turns:      {len(parsed.messages)}")
print(f"Text turns:        {len(text_messages)}")
print(f"Highlight picks:   {len(picks)}")
print(f"Commentary chunks: {len(commentary_chunks)}")
print()

joined = "\n".join(m.text for m in picks)
required = [
    "Ek pic bhejne",
    "Bhejo",
    "Mere sath didi hai",
    "Aur iss situation main bhejne ka dil nhi",
    "Mera phone 7% h",
    "Photo bhej deti",
    "Haa toh bhejo",
]

for phrase in required:
    found = phrase.lower() in joined.lower()
    print(f"{phrase:<45} {'OK' if found else 'MISSING'}")
    assert found, f"Expected highlight context to include: {phrase}"

assert len(picks) == len(text_messages), (
    "Dense compact chat should pass all text turns to the chapter prompt"
)
assert len(commentary_chunks) >= 5, (
    "Dense chat should be narrated in several small commentary windows"
)
assert all(len([m for m in chunk if m.kind == MessageKind.TEXT]) <= 16
           for chunk in commentary_chunks), (
    "Each commentary prompt should stay small for local Ollama"
)
assert "do not send" in _rough_english_hint("Thik hai Mt bhejoo")
assert "send" not in _rough_english_hint("Thik hai Mt bhejoo").split("; ")
assert "send" in _rough_english_hint("Hnn Bhejo Plz")

print()
print("✓ DENSE PHOTO THREAD TEST PASSED")
