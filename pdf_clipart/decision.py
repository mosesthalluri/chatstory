"""
Clipart decision + prompt construction.

Lightweight, rule-based (no model needed) so the "should this page get
clipart?" decision is fast and deterministic:

  1. Skip sparse pages (too few words).
  2. Skip number-heavy pages (tables, invoices, math).
  3. Require at least one recognizable *visual theme* keyword, otherwise
     there's nothing concrete to illustrate.

The prompt for the image model is then built from the page's strongest
visual-theme keywords plus the configured style suffix.
"""

from __future__ import annotations

import re
from collections import Counter

from .config import Config


# A compact lexicon of concrete, illustratable themes. Each canonical theme
# maps to trigger words that may appear in the page text. Kept broad but
# visual — abstract words (e.g. "strategy") are intentionally excluded
# because they don't yield good clipart.
VISUAL_THEMES: dict[str, set[str]] = {
    "coffee cup": {"coffee", "espresso", "latte", "cafe", "caffeine", "tea", "chai"},
    "book": {"book", "novel", "reading", "library", "chapter", "story", "author"},
    "travel airplane": {"travel", "flight", "airplane", "trip", "vacation", "journey", "airport"},
    "mountains": {"mountain", "hiking", "trek", "summit", "valley", "trail"},
    "beach": {"beach", "ocean", "sea", "wave", "sand", "coast", "island"},
    "music note": {"music", "song", "playlist", "guitar", "piano", "concert", "melody"},
    "heart": {"love", "heart", "romance", "valentine", "affection", "couple"},
    "sun": {"summer", "sunny", "morning", "sunshine", "bright", "warm"},
    "moon and stars": {"night", "moon", "stars", "sleep", "dream", "midnight"},
    "rain cloud": {"rain", "storm", "cloudy", "weather", "monsoon", "drizzle"},
    "tree plant": {"tree", "plant", "garden", "forest", "nature", "leaf", "flower", "green"},
    "computer laptop": {"computer", "laptop", "code", "software", "programming", "developer", "tech"},
    "phone": {"phone", "call", "mobile", "message", "text", "notification"},
    "food plate": {"food", "recipe", "cooking", "dinner", "lunch", "meal", "kitchen", "pizza", "cake"},
    "gift box": {"gift", "present", "birthday", "celebration", "surprise", "party"},
    "lightbulb idea": {"idea", "innovation", "creative", "brainstorm", "invention", "insight"},
    "money coin": {"money", "savings", "budget", "finance", "salary", "investment"},
    "graduation cap": {"school", "college", "exam", "study", "student", "graduation", "degree", "education"},
    "clock": {"time", "schedule", "deadline", "clock", "hour", "calendar", "appointment"},
    "rocket": {"rocket", "launch", "startup", "space", "growth", "mission"},
    "camera": {"photo", "camera", "picture", "photography", "selfie", "snapshot"},
    "dog": {"dog", "puppy", "pet"},
    "cat": {"cat", "kitten", "kitty"},
    "car": {"car", "drive", "road", "vehicle", "traffic", "commute"},
    "house": {"home", "house", "apartment", "room", "family", "household"},
    "trophy": {"win", "award", "trophy", "champion", "victory", "achievement", "success"},
    "umbrella": {"umbrella", "shelter", "protection", "safety"},
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _numeric_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    digits = sum(c.isdigit() for c in chars)
    return digits / len(chars)


def theme_hits(text: str) -> Counter:
    """Count how strongly each visual theme is represented in the text."""
    toks = set(_tokens(text))
    hits: Counter = Counter()
    for theme, triggers in VISUAL_THEMES.items():
        overlap = len(toks & triggers)
        if overlap:
            hits[theme] = overlap
    return hits


def should_generate(text: str, config: Config) -> tuple[bool, str, Counter]:
    """Decide whether a page warrants clipart.

    Returns (decision, reason, theme_hits). `reason` is a short human string
    suitable for logging.
    """
    words = _tokens(text)
    if len(words) < config.min_words:
        return False, f"sparse ({len(words)} words < {config.min_words})", Counter()

    ratio = _numeric_ratio(text)
    if ratio > config.max_numeric_ratio:
        return False, f"numeric-heavy ({ratio:.0%} digits)", Counter()

    hits = theme_hits(text)
    if config.require_visual_theme and sum(hits.values()) < config.min_theme_score:
        return False, "no clear visual theme", hits

    if hits:
        top = ", ".join(t for t, _ in hits.most_common(2))
        return True, f"theme: {top}", hits
    # require_visual_theme is False and nothing matched — fall back to
    # generic content keywords.
    return True, "generic content", hits


def build_prompts(text: str, hits: Counter, config: Config) -> list[str]:
    """Build up to `max_cliparts_per_page` image prompts for a page.

    Prefers distinct visual themes; if there aren't enough, derives extra
    subjects from the most frequent content words.
    """
    subjects: list[str] = [theme for theme, _ in hits.most_common(config.max_cliparts_per_page)]

    if len(subjects) < config.max_cliparts_per_page:
        # Top frequent, reasonably long content words as fallback subjects.
        common = Counter(
            w for w in _tokens(text)
            if len(w) > 4 and w not in _COMMON_WORDS
        )
        for word, _ in common.most_common():
            if len(subjects) >= config.max_cliparts_per_page:
                break
            if word not in subjects:
                subjects.append(word)

    prompts = [f"{subject}, {config.image_style}" for subject in subjects]
    return prompts[: config.max_cliparts_per_page]


# Frequent non-visual words to ignore when picking fallback subjects.
_COMMON_WORDS = set("""
about above after again against because before being below between during
further having other should their there these those through under until
while would could about which whose where there their them then than that
this with your you're we'll they're into more most some such only also very
""".split())
