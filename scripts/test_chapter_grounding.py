"""Test the grounded chapter generation helpers using realistic data
from the user's actual review (didi, Mr Rabbit, Mr Patel, Himanshu).

Run: python scripts/test_chapter_grounding.py
"""
import sys, types
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.modules["httpx"] = types.ModuleType("httpx")

# Stub pydantic_settings
pyds = types.ModuleType("pydantic_settings")
pyds.BaseSettings = type("BaseSettings", (), {})
pyds.SettingsConfigDict = lambda **kw: {}
sys.modules["pydantic_settings"] = pyds

from app.models import Message, MessageKind
from app.pipeline.chapter_gen import (
    _extract_entities, _format_highlights,
    _build_date_range_text, _body_word_target,
    _validate_quote_is_real_message,
)


# Recreate the user's chat sample
def make_msg(ts_str: str, sender: str, text: str, kind=MessageKind.TEXT):
    return Message(
        sender=sender,
        timestamp=datetime.fromisoformat(ts_str),
        text=text,
        kind=kind,
    )

messages = [
    make_msg("2025-08-08T03:39:00", "✨Kristy_honey✨", "Love you too❤"),
    make_msg("2025-08-08T03:36:00", "✨Kristy_honey✨", "Zidd nhi kro"),
    make_msg("2025-08-08T03:36:00", "✨Kristy_honey✨", "Baby please"),
    make_msg("2025-08-08T03:36:00", "Mr proton", "Baby please bolo"),
    make_msg("2025-08-08T03:36:00", "Mr proton", "Himanshu mt bolo"),
    make_msg("2025-08-08T03:36:00", "✨Kristy_honey✨", "Himanshu please"),
    make_msg("2025-08-08T03:25:00", "✨Kristy_honey✨", "Mr patel"),
    make_msg("2025-08-08T03:24:00", "✨Kristy_honey✨", "Mr rabbit"),
    make_msg("2025-08-08T03:17:00", "✨Kristy_honey✨", "Mere sath didi hai"),
    make_msg("2025-08-08T03:17:00", "✨Kristy_honey✨", "Woh soyi bhi nhi h shyd"),
    make_msg("2025-08-08T03:13:00", "✨Kristy_honey✨", "Aur didi bhi dekh legi"),
    make_msg("2025-08-08T03:19:00", "✨Kristy_honey✨", "Mera phone 7% h"),
    make_msg("2025-08-08T03:28:00", "✨Kristy_honey✨", "4%"),
    make_msg("2025-08-08T04:29:00", "Mr proton",
             "Meri tarah tu aahey bharey tu bhi kisi sai pyaar karey",
             MessageKind.MEDIA_PLACEHOLDER),
    make_msg("2025-08-08T04:28:00", "Mr proton",
             "Sari Sari Raat Soye Na Hum",
             MessageKind.MEDIA_PLACEHOLDER),
]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------
print("=== Entity extraction ===")
entities = _extract_entities(messages, sender_names={"Mr proton", "✨Kristy_honey✨"})
print(f"Nicknames detected: {entities['nicknames']}")
print(f"Other people: {entities['other_people']}")
print(f"Proper nouns: {entities['proper_nouns']}")
print()

# The user's review said the book should have caught Himanshu, didi, Mr Rabbit, Mr Patel
assert "didi (elder sister)" in entities["other_people"], \
    f"Should detect 'didi' as elder sister, got {entities['other_people']}"
print("✓ 'didi' correctly identified as elder sister")
assert "Mr Rabbit" in entities["nicknames"] or "Mr Patel" in entities["nicknames"], \
    f"Should detect at least one Mr-X nickname, got {entities['nicknames']}"
print(f"✓ Nicknames captured: {entities['nicknames']}")
assert "baby" in entities["nicknames"], "Should detect 'baby' endearment"
print("✓ 'baby' endearment captured")
assert "Himanshu" in entities["proper_nouns"], \
    f"Should detect 'Himanshu' as proper noun, got {entities['proper_nouns']}"
print("✓ 'Himanshu' detected as proper noun")
print()


# ---------------------------------------------------------------------------
# Highlight formatting — media-share tagging
# ---------------------------------------------------------------------------
print("=== Media-share tagging in highlights ===")
formatted = _format_highlights(messages[-2:])  # last 2 are reel shares
print(formatted)
print()
assert "[SHARED MEDIA — caption follows, NOT their own words]" in formatted, \
    "Media messages must be tagged distinctly"
print("✓ Media messages clearly tagged for the LLM")
print()


# ---------------------------------------------------------------------------
# Date range builder
# ---------------------------------------------------------------------------
print("=== Date range text ===")
single = _build_date_range_text(date(2025, 8, 8), date(2025, 8, 8))
print(f"Single day:   {single!r}")
assert "single day" in single and "2025-08-08" in single
assert "no other dates" in single
print("✓ Single-day phrasing prevents LLM from expanding range")

multi = _build_date_range_text(date(2025, 8, 4), date(2025, 8, 8))
print(f"5-day range:  {multi!r}")
assert "5 days" in multi

long_range = _build_date_range_text(date(2024, 1, 1), date(2025, 12, 31))
print(f"Long range:   {long_range!r}")
print()


# ---------------------------------------------------------------------------
# Body word target — adaptive length
# ---------------------------------------------------------------------------
print("=== Adaptive body length ===")
for n in [5, 15, 30, 100, 300, 1000]:
    print(f"  {n} messages → {_body_word_target(n)} words")
assert "60-120" in _body_word_target(5)
assert "280-380" in _body_word_target(500)
print("✓ Body length scales with material")
print()


# ---------------------------------------------------------------------------
# Pull-quote validation
# ---------------------------------------------------------------------------
print("=== Pull-quote validation ===")
# A real quote should validate
assert _validate_quote_is_real_message("Mere sath didi hai", messages), \
    "Real text message should validate"
print("✓ Real text message validates as pull-quote source")

# A media-share caption should NOT validate (only TEXT messages count)
assert not _validate_quote_is_real_message(
    "Meri tarah tu aahey bharey tu bhi kisi sai pyaar karey", messages
), "Media-share caption must NOT validate"
print("✓ Media-share caption rejected (won't appear as pull-quote)")

# Invented text should not validate
assert not _validate_quote_is_real_message(
    "She felt the weight of unspoken words", messages
), "Invented prose must not validate"
print("✓ Invented prose rejected")
print()

print("✓ CHAPTER GROUNDING TEST PASSED")
