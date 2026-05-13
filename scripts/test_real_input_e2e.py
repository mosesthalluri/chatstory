"""End-to-end sanity check using the user's actual test.txt input.

Verifies the full chain:
  1. Generic parser handles their format correctly
  2. Reel-shares classified as MEDIA, not TEXT
  3. Normalizer produces clean canonical output
  4. Highlight selector picks real conversation, not media

Run: python scripts/test_real_input_e2e.py
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
from app.parsers import parse_chat
from app.pipeline import highlights
from app.pipeline.normalizer import normalize


# The user's actual test.txt content (key section)
USER_INPUT = """Clips - @vaani_iitbrahgiirrhttps://www.instagram.com/reel/DL9_OpISo_o/

[2025-08-08 04:29:00 IST] Mr proton: Mr sent an attachment.Meri tarah tu aahey bharey tu bhi kisi sai pyaar karey 🛐
.
.
#song #trending

[2025-08-08 04:29:00 IST] Mr proton: Mr sent an attachment.I don't deserve this, You look perfect 
.
#sad

[2025-08-08 04:28:00 IST] Mr proton: Mr sent an attachment.Sari Sari Raat Soye Na Hum 🌙

[2025-08-08 03:42:00 IST] ✨Kristy_honey✨: Hmm

[2025-08-08 03:42:00 IST] Mr proton: Jao mujhe toh yehi rehne hai

[2025-08-08 03:41:00 IST] ✨Kristy_honey✨: Good night

[2025-08-08 03:41:00 IST] ✨Kristy_honey✨: I'm going

[2025-08-08 03:39:00 IST] Mr proton: Liked a message

[2025-08-08 03:36:00 IST] ✨Kristy_honey✨: Zidd nhi kro

[2025-08-08 03:36:00 IST] Mr proton: Baby please bolo

[2025-08-08 03:36:00 IST] Mr proton: Himanshu mt bolo

[2025-08-08 03:36:00 IST] ✨Kristy_honey✨: Himanshu please

[2025-08-08 03:25:00 IST] ✨Kristy_honey✨: Mr patel

[2025-08-08 03:24:00 IST] ✨Kristy_honey✨: Mr rabbit

[2025-08-08 03:19:00 IST] ✨Kristy_honey✨: Mera phone 7% h

[2025-08-08 03:28:00 IST] ✨Kristy_honey✨: 4%

[2025-08-08 03:17:00 IST] ✨Kristy_honey✨: Mere sath didi hai

[2025-08-08 03:17:00 IST] ✨Kristy_honey✨: Woh soyi bhi nhi h shyd

[2025-08-08 03:13:00 IST] ✨Kristy_honey✨: Aur didi bhi dekh legi

[2025-08-08 03:16:00 IST] ✨Kristy_honey✨: Babu plz

[2025-08-08 03:11:00 IST] ✨Kristy_honey✨: No means no
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
    f.write(USER_INPUT)
    path = Path(f.name)


# ============================================================================
# STAGE 1: Parser correctness
# ============================================================================
print("=== Stage 1: Parser output ===\n")

parsed = parse_chat(path)
text_msgs = [m for m in parsed.messages if m.kind == MessageKind.TEXT]
media_msgs = [m for m in parsed.messages if m.kind == MessageKind.MEDIA_PLACEHOLDER]

print(f"Total parsed: {len(parsed.messages)}")
print(f"  TEXT:  {len(text_msgs)}")
print(f"  MEDIA: {len(media_msgs)}")
print(f"Senders: {parsed.senders}")
print()

assert len(text_msgs) >= 5, f"Should have substantial text messages, got {len(text_msgs)}"
assert len(media_msgs) >= 2, f"Should have at least 2 media-shares, got {len(media_msgs)}"
assert "Mr proton" in parsed.senders
assert "✨Kristy_honey✨" in parsed.senders

# Critical: no reel caption should be in TEXT messages
for m in text_msgs:
    assert "Meri tarah tu aahey" not in m.text
    assert "Sari Sari Raat" not in m.text
print("✓ Reel captions correctly classified as MEDIA, not TEXT\n")


# ============================================================================
# STAGE 2: Normalizer output
# ============================================================================
print("=== Stage 2: Normalizer ===\n")

summary, canonical = normalize(path)
print(f"Detected format:     {summary.detected_format}")
print(f"Total messages:      {summary.total_raw_messages}")
print(f"  Text:              {summary.text_messages}")
print(f"  Media:             {summary.media_messages}")
print(f"  Filtered out:      {summary.filtered_out}")
print(f"Days active/span:    {summary.days_active}/{summary.days_span}")
print(f"Date range:          {summary.date_range}")
print(f"Per sender:          {summary.senders_count}")
print()

print("First 5 lines of canonical output:")
for line in canonical.split("\n")[:5]:
    print(f"  {line}")
print()

assert "[MEDIA]" in canonical, "Canonical text should mark media-shares"
assert summary.filtered_out >= 1, "At least the reaction should be filtered"
print("✓ Normalizer produces clean canonical output with media tagging\n")


# ============================================================================
# STAGE 3: Highlight selection
# ============================================================================
print("=== Stage 3: Highlight selection (the big bug fix) ===\n")

picks = highlights.select_highlights(parsed.messages, n=40)
text_picks = [p for p in picks if p.kind == MessageKind.TEXT]
media_picks = [p for p in picks if p.kind == MessageKind.MEDIA_PLACEHOLDER]

print(f"Highlights selected: {len(picks)}")
print(f"  TEXT:  {len(text_picks)}")
print(f"  MEDIA: {len(media_picks)}")
print()
print("First 8 highlights (chronological):")
for p in picks[:8]:
    print(f"  [{p.timestamp.strftime('%H:%M')}] {p.sender}: {p.text[:60]}")
print()

# THE critical assertion: media-shares MUST NOT appear in highlights
assert media_picks == [], \
    f"Media-shares must be excluded from highlights, but found: {media_picks}"
print("✓ Media-shares correctly excluded from highlights")

# Real conversation messages should be in the picks
important = ["Mere sath didi hai", "Mr rabbit", "Mr patel", "Mera phone 7%",
             "Babu plz", "Himanshu", "didi"]
matched = 0
for keyword in important:
    if any(keyword.lower() in p.text.lower() for p in text_picks):
        matched += 1
print(f"✓ {matched}/{len(important)} key narrative messages reached highlights")
assert matched >= 4, f"Most narrative messages should reach highlights, only got {matched}"
print()


# ============================================================================
# STAGE 4: What the LLM will see (the "fix the chapter" check)
# ============================================================================
print("=== Stage 4: Sample of what the LLM will receive ===\n")
print("(This is the input the chapter generator will use.")
print(" Should be conversation-heavy, not media-heavy.)\n")
for p in picks[:12]:
    tag = "[MEDIA] " if p.kind == MessageKind.MEDIA_PLACEHOLDER else ""
    print(f"  [{p.timestamp.strftime('%H:%M')}] {p.sender}: {tag}{p.text[:80]}")
print()

path.unlink()
print("✓ REAL-INPUT END-TO-END TEST PASSED")
print()
print("Expected change: the next chapter should narrate the actual")
print("conversation (didi present, photo refused, phone dying, nicknames)")
print("instead of just describing media-shares.")
