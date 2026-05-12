"""Test the content filter behaviour against realistic bigram candidates.

Run: python scripts/test_content_filter.py
"""
import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.modules["httpx"] = types.ModuleType("httpx")

from app.pipeline.content_filter import (
    is_generic_phrase, contains_inappropriate,
    safe_for_display, has_distinctive_token,
)

# Realistic inputs from real chat-mining: bigrams that legitimately
# occur a lot in close-relationship chats. The question: which are
# genuine inside-joke candidates and which are noise?
test_cases = [
    # (phrase, expected_safe_for_display, reason)
    ("good night",       False, "universal farewell"),
    ("good morning",     False, "universal greeting"),
    ("love you",         False, "universal affection"),
    ("miss you",         False, "universal affection"),
    ("baby please",      False, "generic entreaty"),
    ("kya kar",          False, "generic Hindi 'what doing'"),
    ("thik hai",         False, "generic Hindi 'is okay'"),
    ("mr proton",        True,  "actual nickname — keep"),
    ("mr rabbit",        True,  "playful nickname variation"),
    ("mr patel",         True,  "playful nickname variation"),
    ("babu monkey",      True,  "made-up couple-nickname"),
    ("kristy honey",     True,  "username-as-nickname"),
    ("dub gaya",         True,  "distinctive Hindi phrase (sunk in thoughts)"),
    ("system design",    True,  "specific technical reference"),
    ("system thinking",  True,  "specific reference"),
    # Inappropriate filtering
    ("fucking hell",     False, "profanity"),
    ("you bitch",        False, "slur"),
    ("madarchod yaar",   False, "Hindi profanity"),
    ("randi rona",       False, "Hindi profanity"),
    ("get fucked",       False, "profanity"),
    # Genuinely benign + distinctive (should pass)
    ("purple monkey",    True,  "absurd combination — likely joke"),
    ("rabbit hole",      True,  "uncommon phrase"),
    ("nani ghar",        True,  "specific reference (grandmother's house)"),
    # Edge cases
    ("",                 False, "empty"),
    ("   ",              False, "whitespace only"),
    ("a b",              True,  "borderline — safe_for_display only checks block lists"),
]

print(f"{'Phrase':<24}{'Result':<10}{'Expected':<12}{'Reason'}")
print("-" * 90)
correct = 0
total = 0
for phrase, expected, reason in test_cases:
    result = safe_for_display(phrase)
    ok = result == expected
    correct += ok
    total += 1
    mark = "✓" if ok else "✗"
    print(f"{phrase[:23]:<24}{str(result):<10}{str(expected):<12}{mark} {reason}")

print()
print(f"Pass rate: {correct}/{total} ({100*correct//total}%)")
print()

# Distinctiveness test — needs the "common tokens" context
common_tokens = {
    "good", "morning", "night", "love", "miss", "please", "baby",
    "okay", "hmm", "you", "tum", "kya", "kar", "hai", "thik",
}
distinctive_tests = [
    ("good night",   False),  # both in common
    ("love you",     False),  # both in common
    ("kya kar",      False),  # both in common
    ("mr proton",    True),   # "proton" not in common
    ("babu monkey",  True),   # both rare
    ("good monkey",  True),   # one rare
]
print("Distinctive-token check (top common tokens = greetings, polite filler):")
for phrase, expected in distinctive_tests:
    result = has_distinctive_token(phrase, common_tokens)
    mark = "✓" if result == expected else "✗"
    print(f"  {phrase:<20} → distinctive={result}, expected {expected} {mark}")

print()
print("✓ CONTENT FILTER TEST COMPLETE")
