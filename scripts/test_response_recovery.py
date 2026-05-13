"""Test the two bugs the user hit after v0.2.0 was applied:
  A. Reel captions misclassified as TEXT (passed pull-quote validator)
  B. 'Chapter generation incomplete' because LLM used **markdown** headers

Run: python scripts/test_response_recovery.py
"""
import sys, types, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.modules["httpx"] = types.ModuleType("httpx")
pyds = types.ModuleType("pydantic_settings")
pyds.BaseSettings = type("BaseSettings", (), {})
pyds.SettingsConfigDict = lambda **kw: {}
sys.modules["pydantic_settings"] = pyds

from app.models import MessageKind
from app.parsers.generic import parse as parse_generic
from app.pipeline.chapter_gen import (
    _strip_markdown, _parse_chapter_response,
    _smart_body_fallback, _validate_quote_is_real_message,
)


# ============================================================================
# BUG A: Reel captions must classify as MEDIA_PLACEHOLDER, not TEXT
# ============================================================================
print("=== BUG A: Attachment messages with captions ===\n")

sample = """[2025-08-08 04:29:00 IST] Mr proton: Mr sent an attachment.Meri tarah tu aahey bharey tu bhi kisi sai pyaar karey 🛐

[2025-08-08 04:29:00 IST] Mr proton: Mr sent an attachment.I don't deserve this, You look perfect

[2025-08-08 03:17:00 IST] ✨Kristy_honey✨: Mere sath didi hai

[2025-08-08 03:13:00 IST] ✨Kristy_honey✨: Aur didi bhi dekh legi
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
    f.write(sample)
    path = Path(f.name)

parsed = parse_generic(path)
text_msgs = [m for m in parsed.messages if m.kind == MessageKind.TEXT]
media_msgs = [m for m in parsed.messages if m.kind == MessageKind.MEDIA_PLACEHOLDER]

print(f"TEXT messages ({len(text_msgs)}):")
for m in text_msgs:
    print(f"  • {m.sender}: {m.text}")

print(f"\nMEDIA messages ({len(media_msgs)}):")
for m in media_msgs:
    print(f"  • {m.sender}: {m.text[:60]}")

# Both reel-caption messages must be MEDIA, neither TEXT
assert len(media_msgs) == 2, f"Both reel shares should be MEDIA, got {len(media_msgs)}"
assert any("Meri tarah" in m.text for m in media_msgs), \
    "Reel caption 'Meri tarah...' must be in media bucket"
assert any("I don't deserve" in m.text for m in media_msgs), \
    "Reel caption 'I don't deserve...' must be in media bucket"
assert all("Meri tarah" not in m.text for m in text_msgs), \
    "Reel caption must NOT appear in TEXT bucket"
print("\n✓ BUG A FIXED: reel captions are MEDIA, not TEXT")
print()

# Pull-quote validator should now reject the reel caption
quote = "Meri tarah tu aahey bharey tu bhi kisi sai pyaar karey"
is_valid = _validate_quote_is_real_message(quote, parsed.messages)
assert not is_valid, "Validator MUST reject reel-caption quote now"
print("✓ Validator now rejects reel caption as pull-quote source")

# And rejects the concatenated multi-reel quote
combo_quote = "meri tarah tu aahey bharey tu bhi kisi sai pyar karey i dont deserve this, you look perfect"
is_valid = _validate_quote_is_real_message(combo_quote, parsed.messages)
assert not is_valid, "Validator MUST reject combined reel-caption quote"
print("✓ Validator rejects combined reel-caption quote")

# Real text message should still validate
real = "Mere sath didi hai"
is_valid = _validate_quote_is_real_message(real, parsed.messages)
assert is_valid, "Real TEXT message must still validate"
print("✓ Real text message validates correctly")

path.unlink()
print()


# ============================================================================
# BUG B: Parser must handle markdown-formatted LLM responses
# ============================================================================
print("=== BUG B: Markdown-formatted LLM response ===\n")

# Realistic failed response — Llama 3.3 70B style markdown formatting
markdown_response = """**TITLE:** Late August Whispers

**WHEN:** August 8, 2025

**BODY:**

In the early hours of August 8th, Kristy_honey and Mr proton had a tense conversation. Her sister was in the room beside her, and her phone battery was dwindling to 4%. She wanted to sleep. He wanted to keep talking. She refused to send a photo not because she didn't want to, but because her sister might see. He kept calling her by playful names — Mr Rabbit, Mr Patel — but the mood was fraying.

**QUOTE:** Mere sath didi hai

**QUOTE_BY:** ✨Kristy_honey✨

**ILLUSTRATION:** A bedroom at 3am with a phone glowing dim on the bedside.
"""

stripped = _strip_markdown(markdown_response)
print("After markdown stripping (first 200 chars):")
print(stripped[:200])
print()

assert "**" not in stripped, "Asterisks should be gone"
assert "TITLE: Late August Whispers" in stripped
assert "BODY:" in stripped and not "**BODY:**" in stripped
print("✓ Markdown decorations stripped")

# Now parse the response — body should be recovered, not the fallback string
chapter = _parse_chapter_response(
    markdown_response, index=1, fallback_when="August 2025",
    chapter_messages=parsed.messages if 'parsed' in dir() else None,
)
print(f"\nParsed chapter:")
print(f"  Title:    {chapter.title}")
print(f"  When:     {chapter.when}")
print(f"  Body:     {chapter.body[:80]}...")
print(f"  Quote:    {chapter.pull_quote}")
print(f"  By:       {chapter.pull_quote_author}")
print()

assert "Late August Whispers" in chapter.title, f"Title not parsed: {chapter.title}"
assert "incomplete" not in chapter.body.lower(), \
    f"Body should be recovered, got: {chapter.body[:100]}"
assert len(chapter.body) > 100, f"Body should be the real prose, got: {chapter.body}"
print("✓ BUG B FIXED: markdown headers parse, BODY not stuck on fallback string")
print()


# ============================================================================
# Smart body fallback — when even header detection fails
# ============================================================================
print("=== Smart body fallback ===\n")

# Worst case: LLM ignores format entirely, just writes prose
freeform_response = """Title: Late August Whispers

This is a chapter about Kristy_honey and Mr proton's late-night conversation on August 8th, 2025. They talked into the early hours about her sister being in the room, her phone running low, and his desire to keep her on the call. The chapter captures a single conversation but reveals the quiet tensions of long-distance closeness in a household where privacy is shared with family.

The quote should be: Mere sath didi hai
That was said by Kristy_honey.
"""

recovered = _smart_body_fallback(freeform_response, {})
print(f"Recovered body: {recovered[:100]}...")
assert recovered and len(recovered) > 100
assert "Kristy_honey" in recovered
print("✓ Smart fallback finds prose paragraphs even without BODY: header")
print()

print("✓ RESPONSE-RECOVERY TEST PASSED")
