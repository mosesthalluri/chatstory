"""Test the clipart matcher with realistic illustration prompts.

Run: python scripts/test_clipart_matcher.py
"""

import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Stub the dependencies that aren't installed in this sandbox
sys.modules["httpx"] = types.ModuleType("httpx")
sys.modules["pydantic_settings"] = types.ModuleType("pydantic_settings")

# Real settings module would import pydantic_settings — stub a minimal one
class FakeSettings:
    USE_GEMINI_IMAGES = False
    GEMINI_API_KEY = ""
    IMAGE_STYLE = "flat vector"
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    STATIC_DIR = PROJECT_ROOT / "frontend" / "static"
    TEMPLATES_DIR = PROJECT_ROOT / "frontend" / "templates"

stub_settings = types.ModuleType("app.settings")
stub_settings.settings = FakeSettings()
stub_settings.PROJECT_ROOT = FakeSettings.PROJECT_ROOT
stub_settings.STATIC_DIR = FakeSettings.STATIC_DIR
stub_settings.TEMPLATES_DIR = FakeSettings.TEMPLATES_DIR
stub_settings.BACKEND_ROOT = FakeSettings.PROJECT_ROOT / "backend"
stub_settings.STORAGE_ROOT = stub_settings.BACKEND_ROOT / "storage"
stub_settings.UPLOADS_DIR = stub_settings.STORAGE_ROOT / "uploads"
stub_settings.JOBS_DIR = stub_settings.STORAGE_ROOT / "jobs"
stub_settings.OUTPUT_DIR = stub_settings.STORAGE_ROOT / "output"
sys.modules["app.settings"] = stub_settings

from app.services.image_gen import pick_clipart_for_prompt

# Realistic illustration prompts that an LLM might write for chapters
# of an LDR couple's story.
test_prompts = [
    ("A small apartment kitchen at night, two half-empty mugs on the counter.",
     "04_coffee.svg"),
    ("A phone screen glowing in the dark, an unread message notification visible.",
     "07_phone.svg"),
    ("A paper airplane gliding across an open sky above scattered clouds.",
     "05_paperplane.svg"),
    ("Two birds flying side by side toward the horizon at dawn.",
     "10_birds.svg"),
    ("A bedroom at 3am, a glowing screen casts shadows on the wall.",
     "03_night.svg"),
    ("A calendar with one date circled in red, anticipation written all over it.",
     "06_calendar.svg"),
    ("Rain hits a window pane, a single figure watches from inside.",
     "09_rain.svg"),
    ("A globe sits on a desk, two pins stuck into different continents.",
     "12_globe.svg"),
    ("Two chat bubbles in conversation, the first message of many.",
     "01_chat_bubbles.svg"),
    ("A heart-shaped balloon floats up into a soft pastel sky.",
     "02_heart.svg"),
    ("Morning sunlight streams through a kitchen window onto a steaming mug.",
     # ambiguous — sun OR coffee could match
     None),
    ("A bookshelf covered in framed photographs and worn paperbacks.",
     "11_bookshelf.svg"),
]

print(f"{'Prompt':<70}{'Picked':<25}{'Score':<6}{'Expected'}")
print("-" * 130)
correct = 0
for prompt, expected in test_prompts:
    path, fname, score = pick_clipart_for_prompt(prompt)
    truncated = prompt[:67] + "…" if len(prompt) > 68 else prompt
    expected_str = expected or "(ambiguous)"
    match = "✓" if expected is None or fname == expected else "✗"
    if expected is None or fname == expected:
        correct += 1
    print(f"{truncated:<70}{fname:<25}{score:<6}{match} {expected_str}")

print()
print(f"Match rate: {correct}/{len(test_prompts)} ({100*correct//len(test_prompts)}%)")

# Diversification test — same prompt, different avoid sets
print()
print("Diversification test (8 chapters, all generic prompts):")
generic = "A scene that captures the feeling of this chapter."
used = set()
for i in range(1, 9):
    path, fname, score = pick_clipart_for_prompt(generic, avoid=used)
    used.add(fname)
    print(f"  Chapter {i}: {fname} (score {score})")
unique = len(used)
print(f"  Unique cliparts used: {unique}/8")
assert unique == 8, f"Diversification failed: only {unique} unique cliparts"

print("\n✓ CLIPART MATCHER TEST PASSED")
