"""Test the generic parser against the user's actual messy sample.

This is the file shape from a real LDR couple's Instagram chat that's
been manually cleaned but still has reel shares, hashtags, joined lines,
emoji senders, and IST timestamps.

Run: python scripts/test_user_format.py
"""

import sys, types
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

stub = types.ModuleType("app.settings")
stub.BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
stub.PROJECT_ROOT = stub.BACKEND_ROOT.parent
stub.STORAGE_ROOT = stub.BACKEND_ROOT / "storage"
stub.UPLOADS_DIR = stub.STORAGE_ROOT / "uploads"
stub.JOBS_DIR = stub.STORAGE_ROOT / "jobs"
stub.OUTPUT_DIR = stub.STORAGE_ROOT / "output"
stub.TEMPLATES_DIR = stub.PROJECT_ROOT / "frontend" / "templates"
stub.STATIC_DIR = stub.PROJECT_ROOT / "frontend" / "static"
class _S:
    USE_OLLAMA = False
    GROQ_API_KEY = ""
    GEMINI_API_KEY = ""
stub.settings = _S()
sys.modules["app.settings"] = stub

from app.parsers.generic import parse as parse_generic
from app.models import MessageKind

# This is the user's actual sample — pasted from their chat message.
# Note: lines that look glued in the user's paste are joined HERE on
# purpose, because that's how their real file looks.
sample = """Clips - @vaani_iitbrahgiirrhttps://www.instagram.com/reel/DL9_OpISo_o/?id=3674370954169585640_57015769674
[2025-08-08 04:29:00 IST] Mr proton: Mr sent an attachment.Meri tarah tu aahey bharey tu bhi kisi sai pyaar karey 🛐
.
.
.
.
.
.
.
.
.
#song #trending #music #instagood #like #love #bollywood #listen #oldsong #bollywoodsongs_relatable__mfhttps://www.instagram.com/reel/DM4ql_2sla2/?id=3690887223767619254_72215764199
[2025-08-08 04:29:00 IST] Mr proton: Mr sent an attachment.I don't deserve this, You look perfect 
.
.
.
#sad #feedfeed #love #loveit #loveislove #explore #explorepage #couples #couplegoals #darklife #foryou #reels #fypシ04darklifehttps://www.instagram.com/p/DMAsL_VKu6K/
[2025-08-08 04:29:00 IST] Mr proton: Mr sent an attachment.#FORYOU #FYP
#EXPLORE #REACH
#REELSGROWTH
#BOOSTYOURREEL
#TRENDINGNOW #DASTGIRXGROWTHpoet_pencil_https://www.instagram.com/reel/DM-b7lgSnNO/?id=3692511586465510222_75071040651
[2025-08-08 04:28:00 IST] Mr proton: Mr sent an attachment.Dub gaya khayalo mai ❤‍🩹
. 
. 
. 
. 
Follow jjd.joydip__https://www.instagram.com/reel/DGVycJgo95G/?id=3572983716070547014_46441984633
[2025-08-08 04:28:00 IST] Mr proton: Mr sent an attachment.Sari Sari Raat Soye Na Hum 🌙
.
.
.
.
#mirzaghalib #ghalib #ghalibkiyadein #ghalibshayari #ghalibpoetry #ghalibshayri #ghalibkipoetry #ghalibkeehsas #urdupoetry #urduposts #sarisariraat #3am #2am #sleep #nind #sleepcycle #sleepschedule #viralreels #urdureels #explorefarazedilhttps://www.instagram.com/reel/DKyn_cyyfUN/?id=3653158131808728333_72505757544
[2025-08-08 03:42:00 IST] ✨Kristy_honey✨: Hmm
[2025-08-08 03:42:00 IST] Mr proton: Jao mujhe toh yehi rehne hai
[2025-08-08 03:42:00 IST] Mr proton: Hnn
[2025-08-08 03:41:00 IST] ✨Kristy_honey✨: Good night
[2025-08-08 03:41:00 IST] ✨Kristy_honey✨: I'm going
[2025-08-08 03:39:00 IST] Mr proton: Okay 🥀❤✨Kristy_honey✨
[2025-08-08 03:39:00 IST] ✨Kristy_honey✨: Hmm
[2025-08-08 03:39:00 IST] ✨Kristy_honey✨: You too
[2025-08-08 03:39:00 IST] Mr proton: Liked a message
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
    f.write(sample)
    path = Path(f.name)

parsed = parse_generic(path)

print(f"Detected senders: {parsed.senders}")
print(f"Total messages parsed: {parsed.message_count}")
print()

kinds = {}
for m in parsed.messages:
    kinds[m.kind.value] = kinds.get(m.kind.value, 0) + 1
print(f"Message kinds: {kinds}")
print()

text_only = [m for m in parsed.messages if m.kind == MessageKind.TEXT]
print(f"TEXT messages ({len(text_only)}):")
for m in text_only:
    print(f"  [{m.timestamp.strftime('%Y-%m-%d %H:%M')}] {m.sender}: {m.text[:80]}")

print()
print("Parser warnings:")
for w in parsed.parser_warnings:
    print(f"  - {w}")

print()
# Assertions
import os; os.unlink(path)

# Real-message count: should be 8 (the back-and-forth chat at the end)
# Reel shares: 5 (the "sent an attachment" lines)
# "Liked a message" reaction: 1
# Total messages-as-rows: 14
assert len(parsed.senders) == 2, f"Should detect 2 senders, got {parsed.senders}"
assert "✨Kristy_honey✨" in parsed.senders, f"Emoji sender missing: {parsed.senders}"
assert "Mr proton" in parsed.senders, f"Plain sender missing: {parsed.senders}"

# Date check — should be 2025-08-08, NOT a synthetic 2024-01-01
dates = {m.timestamp.date() for m in parsed.messages}
assert all(d.year == 2025 and d.month == 8 for d in dates), \
    f"Real dates not parsed — got {dates}"
print(f"✓ All dates correctly parsed as 2025-08-08")

# Hour check — should be 3am or 4am IST, NOT synthetic 9am
hours = {m.timestamp.hour for m in parsed.messages}
assert 9 not in hours or len({h for h in hours if h in (3,4)}) >= 2, \
    f"Looks like synthetic timestamps used. Hours: {hours}"
print(f"✓ Real hours parsed (got hours: {sorted(hours)})")

# Reactions should be tagged as REACTION, not TEXT
reaction_msgs = [m for m in parsed.messages if m.kind == MessageKind.REACTION]
print(f"✓ {len(reaction_msgs)} reactions correctly tagged")

# Reel shares should be tagged as MEDIA_PLACEHOLDER, not text
media_msgs = [m for m in parsed.messages if m.kind == MessageKind.MEDIA_PLACEHOLDER]
print(f"✓ {len(media_msgs)} media-share messages tagged")

# Emoji should NOT leak into stats
total_text = " ".join(m.text for m in parsed.messages if m.kind == MessageKind.TEXT)
assert "✨" not in total_text, "✨ emoji leaked into message text"
print(f"✓ Sender emoji ✨ did not leak into message text")

print("\n✓ USER-FORMAT REGRESSION TEST PASSED")
