"""
Mood detection — gives each page a visual theme so the result looks
intentional (cute / dark / romantic / calm / etc.) instead of plain white
with stamped clipart.

Rule-based keyword matching maps a page's text to a Mood, which carries:
  * a soft full-page background tint colour,
  * an accent colour for the top/bottom decorative bands,
  * opacities tuned so page text stays readable,
  * a prompt modifier injected into the clipart prompt so the generated
    art matches the mood.

Colours are (r, g, b) in 0-255; placement converts to PyMuPDF's 0-1 floats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Mood:
    name: str
    bg: tuple[int, int, int]          # full-page tint
    accent: tuple[int, int, int]      # top/bottom bands
    tint_opacity: float               # how strong the page wash is
    band_opacity: float               # how strong the margin bands are
    prompt_modifier: str              # injected into the clipart prompt


# Ordered most-specific → most-generic. First mood with enough hits wins.
_MOODS: list[tuple[str, set[str], Mood]] = [
    ("scary", {
        "scary", "horror", "ghost", "haunt", "haunted", "fear", "afraid",
        "blood", "death", "dead", "dark", "shadow", "creepy", "nightmare",
        "monster", "evil", "grave", "whisper", "whispering", "silence",
    }, Mood("scary", bg=(28, 28, 36), accent=(70, 40, 70),
            tint_opacity=0.16, band_opacity=0.55,
            prompt_modifier="dark, moody, mysterious, muted shadows, eerie")),

    ("romantic", {
        "love", "loved", "heart", "hearts", "kiss", "romance", "romantic",
        "valentine", "beloved", "darling", "forever", "together", "embrace",
    }, Mood("romantic", bg=(255, 228, 235), accent=(219, 112, 147),
            tint_opacity=0.14, band_opacity=0.5,
            prompt_modifier="romantic, soft pink and rose, tender, gentle hearts")),

    ("sad", {
        "sad", "cry", "cried", "tears", "lonely", "alone", "grief", "miss",
        "missing", "goodbye", "loss", "lost", "sorrow", "ache", "empty",
    }, Mood("sad", bg=(224, 230, 242), accent=(120, 140, 180),
            tint_opacity=0.14, band_opacity=0.5,
            prompt_modifier="wistful, soft blue, gentle melancholy, quiet")),

    ("calm", {
        "garden", "flower", "flowers", "tree", "trees", "forest", "leaf",
        "leaves", "breeze", "calm", "peace", "peaceful", "serene", "nature",
        "river", "meadow", "morning", "quiet", "whispering", "petals", "bloom",
    }, Mood("calm", bg=(226, 242, 230), accent=(122, 158, 122),
            tint_opacity=0.16, band_opacity=0.5,
            prompt_modifier="calm, soft green, botanical, serene, airy")),

    ("happy", {
        "happy", "joy", "joyful", "celebrate", "celebration", "party", "laugh",
        "laughter", "fun", "smile", "sunshine", "bright", "win", "victory",
    }, Mood("happy", bg=(255, 247, 220), accent=(240, 196, 80),
            tint_opacity=0.16, band_opacity=0.5,
            prompt_modifier="bright, cheerful, sunny, playful, warm")),
]

# Default when nothing specific matches.
_CUTE = Mood("cute", bg=(255, 245, 235), accent=(232, 168, 124),
             tint_opacity=0.14, band_opacity=0.5,
             prompt_modifier="cute, soft pastel, adorable, friendly")

_WORD_RE = re.compile(r"[a-zA-Z']+")


def detect_mood(text: str) -> Mood:
    """Pick the page's mood from its text. Falls back to a cute pastel theme."""
    toks = {t.lower() for t in _WORD_RE.findall(text or "")}
    best: tuple[int, Mood] | None = None
    for _name, triggers, mood in _MOODS:
        hits = len(toks & triggers)
        if hits and (best is None or hits > best[0]):
            best = (hits, mood)
    return best[1] if best else _CUTE
