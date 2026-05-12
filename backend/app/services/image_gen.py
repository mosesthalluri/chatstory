"""
Image picker. Default mode: pick the best-matching clipart from a curated
set of pre-made SVGs in frontend/static/cliparts/. Optionally upgrades to
Gemini AI image generation when GEMINI_API_KEY is set AND
USE_GEMINI_IMAGES=true.

Rationale: the Gemini free tier (500/day, ~10 RPM) is too easy to exhaust
during testing and produces inconsistent quality. A small set of curated
SVG cliparts gives us:
  - Zero API cost
  - Zero failure modes
  - Consistent visual identity
  - Instant generation
  - Bundle-size friendly (~50KB for 12 cliparts)

How the matcher works:
  1. Each clipart has a tag set (love, distance, night, morning, etc.)
  2. The chapter's illustration_prompt (written by the LLM) is tokenized
     and scored against every clipart's tag set
  3. Highest-scoring clipart wins; ties broken by chapter index modulo
     count so different chapters in one book get visual variety
  4. If multiple chapters score the same clipart highest, the second one
     gets its second-best so the book doesn't repeat
"""

import asyncio
import base64
import re
import shutil
from pathlib import Path
from typing import Optional

import httpx

from ..settings import settings


# ---------------------------------------------------------------------------
# Clipart catalogue
# ---------------------------------------------------------------------------
#
# Each entry maps a clipart file to a set of tag keywords. Tags should be
# lowercase, single-word or short phrase, and cover the visual content plus
# the emotional/situational themes the clipart represents.
#
# When adding a new clipart, add (a) the SVG file to frontend/static/cliparts/
# and (b) an entry below. Keep tags broad enough that 5-10 prompts can
# reasonably match them.

CLIPART_CATALOG: list[tuple[str, set[str]]] = [
    # filename, tags
    ("01_chat_bubbles.svg", {
        "chat", "message", "messages", "talking", "conversation", "text",
        "texting", "reply", "beginning", "first", "intro", "meet", "hello",
        "bubble", "bubbles", "online",
    }),
    ("02_heart.svg", {
        "love", "heart", "hearts", "romance", "romantic", "affection",
        "valentine", "feeling", "feelings", "care", "caring", "anniversary",
        "confession", "warm", "sweet", "tender",
    }),
    ("03_night.svg", {
        "night", "midnight", "late", "moon", "stars", "dark", "sleep",
        "sleepy", "dream", "dreams", "bedtime", "insomnia", "3am",
        "evening", "lullaby", "quiet",
    }),
    ("04_coffee.svg", {
        "coffee", "morning", "tea", "cup", "mug", "kitchen", "breakfast",
        "cozy", "cosy", "warmth", "routine", "habit", "comfort",
        "domestic", "daily",
    }),
    ("05_paperplane.svg", {
        "plane", "airplane", "travel", "trip", "distance", "far", "away",
        "long-distance", "ldr", "sending", "letter", "miles", "across",
        "fly", "flying", "journey", "leaving", "goodbye",
    }),
    ("06_calendar.svg", {
        "calendar", "date", "dates", "schedule", "anniversary", "milestone",
        "waiting", "months", "weeks", "memory", "memories", "remember",
        "remembering", "time", "passing", "year", "years",
    }),
    ("07_phone.svg", {
        "phone", "mobile", "call", "calling", "notification", "ping",
        "ring", "buzz", "screen", "missed", "unread", "voice", "video",
        "facetime", "device",
    }),
    ("08_sun.svg", {
        "sun", "sunny", "morning", "bright", "happy", "joy", "warm",
        "summer", "spring", "fresh", "new", "beginning", "start", "hope",
        "hopeful", "smile", "light",
    }),
    ("09_rain.svg", {
        "rain", "rainy", "sad", "sadness", "cry", "crying", "tears",
        "miss", "missing", "lonely", "alone", "gloomy", "melancholy",
        "hard", "difficult", "storm", "tough", "fight", "argument",
    }),
    ("10_birds.svg", {
        "birds", "bird", "flying", "freedom", "together", "pair",
        "couple", "two", "side", "journey", "growth", "growing",
        "side-by-side", "flock", "wings",
    }),
    ("11_bookshelf.svg", {
        "book", "books", "bookshelf", "library", "memory", "memories",
        "photo", "photos", "picture", "history", "story", "stories",
        "growth", "shared", "home", "room", "shelf",
    }),
    ("12_globe.svg", {
        "globe", "world", "map", "earth", "international", "country",
        "countries", "distance", "long-distance", "ldr", "across",
        "continents", "ocean", "miles", "remote", "wifi", "internet",
        "connection", "online",
    }),
]


def _clipart_path(filename: str) -> Path:
    """Resolve a clipart file. Falls back to backend STATIC if frontend
    isn't where we expect."""
    candidates = [
        settings.PROJECT_ROOT / "frontend" / "static" / "cliparts" / filename,
        settings.STATIC_DIR / "cliparts" / filename,
    ] if hasattr(settings, "PROJECT_ROOT") else []
    # Fallback resolution if settings doesn't have PROJECT_ROOT
    if not candidates:
        from ..settings import TEMPLATES_DIR
        candidates = [TEMPLATES_DIR.parent / "static" / "cliparts" / filename]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # return first even if missing — caller checks .exists()


# A minimal embedded SVG used only if the cliparts directory is missing
# entirely (e.g. someone deleted it). Keeps the pipeline from crashing.
HARDCODED_PLACEHOLDER_SVG = """<svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg">
  <rect width="320" height="200" fill="#F5E6D8"/>
  <ellipse cx="160" cy="110" rx="135" ry="70" fill="#FAF6F0"/>
  <path d="M 60,70 Q 60,55 75,55 L 130,55 Q 145,55 145,70 L 145,95 Q 145,110 130,110 L 105,110 L 95,123 L 98,110 L 75,110 Q 60,110 60,95 Z" fill="#E8A87C"/>
  <path d="M 155,90 Q 155,75 170,75 L 230,75 Q 245,75 245,90 L 245,115 Q 245,130 230,130 L 195,130 L 185,143 L 188,130 L 170,130 Q 155,130 155,115 Z" fill="#7B9EA8"/>
</svg>"""


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

_STOPWORDS_FOR_MATCH = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be",
    "scene", "image", "illustration", "picture", "depicting", "showing",
    "this", "that", "their", "his", "her", "them", "they", "two", "one",
}


def _tokenize_prompt(prompt: str) -> set[str]:
    """Extract content words from the illustration prompt."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", prompt.lower())
    return {w for w in words if w not in _STOPWORDS_FOR_MATCH and len(w) > 2}


def _score_clipart(prompt_tokens: set[str], tags: set[str]) -> int:
    """Count how many of the prompt's content words match a clipart's tags.
    Also gives a small bonus for prefix matches so 'memories' matches
    'memory' tag etc."""
    score = 0
    for tag in tags:
        if tag in prompt_tokens:
            score += 3
            continue
        # Prefix match either direction (length 4+) — catches plurals,
        # gerunds, common stems
        for tok in prompt_tokens:
            if len(tag) >= 4 and len(tok) >= 4:
                if tag.startswith(tok[:5]) or tok.startswith(tag[:5]):
                    score += 1
                    break
    return score


def pick_clipart_for_prompt(
    prompt: str,
    avoid: Optional[set[str]] = None,
) -> tuple[Path, str, int]:
    """Pick the best-matching clipart for an illustration prompt.

    Returns (path, filename, score). If nothing scores > 0, rotates
    through unused cliparts deterministically based on the avoid set,
    so books with generic prompts still get visual variety.

    `avoid` is a set of filenames already used in the book; lets the
    caller diversify so the same clipart doesn't appear twice unless
    it really is the best match by a wide margin.
    """
    avoid = avoid or set()
    tokens = _tokenize_prompt(prompt)

    scored = [
        (fname, _score_clipart(tokens, tags))
        for fname, tags in CLIPART_CATALOG
    ]
    # Sort by score descending, then by catalog order for stability
    scored.sort(key=lambda x: (-x[1], x[0]))

    top_fname, top_score = scored[0]

    # Case A: top match is fresh (not in avoid) and has a real signal.
    # Use it.
    if top_fname not in avoid and top_score > 0:
        return _clipart_path(top_fname), top_fname, top_score

    # Case B: top match is taken. Look for a strong-enough alternative
    # that hasn't been used. "Strong enough" = within 3 points of top.
    if top_fname in avoid and top_score > 0:
        for fname, score in scored[1:]:
            if fname not in avoid and score >= max(top_score - 3, 1):
                return _clipart_path(fname), fname, score

    # Case C: nothing scored a real signal (generic prompt) OR top is
    # taken with no good alternative. Pick the first unused clipart in
    # catalog order so the book still gets variety.
    for fname, _tags in CLIPART_CATALOG:
        if fname not in avoid:
            return _clipart_path(fname), fname, 0

    # Case D: every clipart has been used. Allow repetition, pick best.
    return _clipart_path(top_fname), top_fname, top_score


# ---------------------------------------------------------------------------
# Stats tracker (kept for backwards compat with orchestrator)
# ---------------------------------------------------------------------------

class ImageGenStats:
    def __init__(self):
        self.succeeded = 0
        self.fell_back = 0
        self.fallback_reasons: dict[str, int] = {}
        self.clipart_picks = 0
        self.ai_picks = 0

    def record_success(self, source: str = "ai"):
        self.succeeded += 1
        if source == "clipart":
            self.clipart_picks += 1
        else:
            self.ai_picks += 1

    def record_fallback(self, reason: str):
        self.fell_back += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def summary(self) -> str:
        parts = []
        if self.clipart_picks:
            parts.append(f"{self.clipart_picks} from clipart library")
        if self.ai_picks:
            parts.append(f"{self.ai_picks} AI-generated")
        if self.fell_back:
            reasons = ", ".join(f"{n}× {r}" for r, n in self.fallback_reasons.items())
            parts.append(f"{self.fell_back} fallback ({reasons})")
        return "; ".join(parts) if parts else "no images generated"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_image(
    prompt: str,
    output_path: Path,
    style: str | None = None,
    stats: Optional[ImageGenStats] = None,
    avoid: Optional[set[str]] = None,
) -> Path:
    """Pick or generate an image for a chapter.

    Decision tree:
      1. If USE_GEMINI_IMAGES is true AND GEMINI_API_KEY is set →
         try AI generation, fall back to clipart on failure.
      2. Otherwise → pick best clipart immediately. No API calls.
      3. If clipart directory is missing → use embedded placeholder SVG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = style or settings.IMAGE_STYLE

    use_gemini = (
        getattr(settings, "USE_GEMINI_IMAGES", False)
        and bool(settings.GEMINI_API_KEY)
    )

    if use_gemini:
        ai_path = await _try_gemini(prompt, output_path, style)
        if ai_path is not None:
            if stats: stats.record_success(source="ai")
            return ai_path
        # Fall through to clipart on AI failure
        if stats: stats.record_fallback("ai_unavailable")

    # Clipart pick (the default fast path)
    src, fname, score = pick_clipart_for_prompt(prompt, avoid=avoid)
    target_svg = output_path.with_suffix(".svg")

    if src.exists():
        shutil.copyfile(src, target_svg)
        if stats and not use_gemini:
            stats.record_success(source="clipart")
        return target_svg

    # Last resort
    target_svg.write_text(HARDCODED_PLACEHOLDER_SVG, encoding="utf-8")
    if stats: stats.record_fallback("clipart_missing")
    return target_svg


async def generate_images_for_chapters(
    chapter_prompts: list[tuple[int, str]],
    output_dir: Path,
    max_concurrent: int = 4,
) -> tuple[dict[int, Path], ImageGenStats]:
    """Generate images for all chapters. With clipart mode (the default)
    this is effectively instant — no network calls. With Gemini mode it
    respects concurrency and rate limits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    stats = ImageGenStats()
    used_cliparts: set[str] = set()

    # For clipart mode we can run sequentially — diversification needs
    # to know what's been used. Still fast (no network).
    use_gemini = (
        getattr(settings, "USE_GEMINI_IMAGES", False)
        and bool(settings.GEMINI_API_KEY)
    )

    if use_gemini:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def worker(idx: int, prompt: str):
            async with semaphore:
                target = output_dir / f"chapter_{idx}"
                results[idx] = await generate_image(prompt, target, stats=stats)

        await asyncio.gather(*(worker(i, p) for i, p in chapter_prompts))
    else:
        # Sequential clipart picking with diversification
        for idx, prompt in chapter_prompts:
            target = output_dir / f"chapter_{idx}"
            path = await generate_image(prompt, target, stats=stats, avoid=used_cliparts)
            results[idx] = path
            used_cliparts.add(path.name)

    return results, stats


# ---------------------------------------------------------------------------
# Gemini path (optional upgrade)
# ---------------------------------------------------------------------------

async def _try_gemini(prompt: str, output_path: Path, style: str) -> Optional[Path]:
    """Try one Gemini call. Returns the PNG path on success, None on
    any failure (caller falls back to clipart). No retries here — we'd
    rather get a fast clipart than wait through backoff."""
    full_prompt = f"{style}. {prompt}"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-image:generateContent?key={settings.GEMINI_API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                return None
            data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                image_bytes = base64.b64decode(inline["data"])
                png_path = output_path.with_suffix(".png")
                png_path.write_bytes(image_bytes)
                return png_path
        return None
    except (httpx.HTTPError, KeyError, ValueError, IndexError):
        return None
