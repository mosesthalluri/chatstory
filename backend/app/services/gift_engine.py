"""
Gift Engine — evidence-grounded recommendations (no LLM).
"""

import asyncio
import json
import math
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..models import Message
from ..core import build_intelligence
from ..core.semantics import is_semantic_message, suppress_noise
from ..parsers import parse_chat
from ..pipeline import nlp_insights as nlp
from ..pipeline import content_filter
from ..settings import OUTPUT_DIR, TEMPLATES_DIR, settings
from . import jobs, pdf_render
from jinja2 import Environment, FileSystemLoader, select_autoescape

_templates = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)

# Each category has STRONG keywords (distinctive — a real signal) and WEAK
# keywords (common — supporting only). A gift is only generated when there's
# at least one STRONG match, and the supporting quote is guaranteed to be a
# message that actually contains a strong keyword. This is what stops "sleep"
# matching inside "sleepy", or a sarcastic "God is good" powering a faith gift.
PATTERNS = {
    "spiritual": {"strong": ["bible", "prayer", "pray", "praying", "church", "jesus",
                             "worship", "sermon", "verse", "devotional", "scripture",
                             "pastor", "blessed", "amen", "faith"],
                  "weak": ["god", "lord", "christ", "grace", "blessing"]},
    "hobbies": {"strong": ["painting", "sketching", "photography", "gardening", "baking",
                          "journaling", "crochet", "knitting", "pottery", "calligraphy"],
                "weak": ["paint", "sketch", "reading", "plant", "yoga", "dance", "cook", "bake"]},
    "music": {"strong": ["spotify", "playlist", "concert", "album", "guitar", "piano",
                        "singing", "band"],
              "weak": ["song", "music", "singer", "lyrics", "melody"]},
    "travel": {"strong": ["trip", "flight", "goa", "vacation", "passport", "roadtrip",
                         "itinerary", "getaway"],
               "weak": ["travel", "hotel", "journey", "outing", "beach", "mountain"]},
    "stress": {"strong": ["stressed", "anxious", "overwhelmed", "burnout", "deadline",
                        "panic", "exhausted", "breakdown"],
               "weak": ["tired", "stress", "exam", "workload", "busy", "pressure"]},
    "food": {"strong": ["biryani", "pizza", "chocolate", "swiggy", "zomato", "restaurant",
                       "dessert", "brownie", "shawarma"],
             "weak": ["coffee", "tea", "chai", "burger", "cake", "dinner", "food", "snack"]},
    "gaming": {"strong": ["valorant", "minecraft", "fortnite", "playstation", "xbox",
                        "steam", "ps5", "bgmi", "genshin"],
               "weak": ["game", "gaming", "gamer"]},
    "routines": {"strong": ["commute", "workout", "schedule", "treadmill", "skincare"],
                 "weak": ["morning", "night", "sleep", "walk", "office", "class", "routine"]},
    "support": {"strong": ["you got this", "i'm here for you", "im here for you",
                         "here for you", "proud of you", "you'll be okay", "you will be okay",
                         "i believe in you", "don't worry i", "im always here"],
                "weak": ["take care", "feel better", "i'm here", "im here"]},
}

URL_RE = re.compile(r"https?://\S+", re.I)


def _wb(words: list[str]):
    if not words:
        return None
    parts = sorted({w.strip() for w in words if w.strip()}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in parts) + r")\b", re.I)


_STRONG_RE = {name: _wb(cfg["strong"]) for name, cfg in PATTERNS.items()}
_WEAK_RE = {name: _wb(cfg["weak"]) for name, cfg in PATTERNS.items()}

# How each category leans across emotional gift types (0-100). Lets the UI
# show "Sentimental 95 · Funny 40 · Faith 90" so users can choose by feel.
TYPE_PROFILE: dict[str, dict[str, int]] = {
    "spiritual": {"faith": 95, "sentimental": 80, "practical": 40, "funny": 10, "romantic": 10},
    "support":   {"sentimental": 95, "practical": 50, "romantic": 40, "funny": 20, "faith": 20},
    "stress":    {"practical": 85, "sentimental": 60, "funny": 20, "romantic": 15, "faith": 10},
    "food":      {"sentimental": 70, "practical": 75, "funny": 40, "romantic": 40, "faith": 5},
    "music":     {"sentimental": 75, "practical": 55, "funny": 40, "romantic": 45, "faith": 5},
    "travel":    {"practical": 70, "sentimental": 65, "romantic": 55, "funny": 30, "faith": 5},
    "hobbies":   {"practical": 80, "sentimental": 55, "funny": 40, "romantic": 20, "faith": 5},
    "gaming":    {"funny": 80, "practical": 70, "sentimental": 40, "romantic": 15, "faith": 5},
    "routines":  {"practical": 85, "sentimental": 55, "funny": 25, "romantic": 20, "faith": 5},
    "inside_joke": {"funny": 90, "sentimental": 80, "practical": 40, "romantic": 30, "faith": 10},
}
_DEFAULT_TYPES = {"sentimental": 60, "practical": 60, "funny": 40, "romantic": 30, "faith": 10}

# Themed bundles — people think in themes, not isolated products.
BUNDLE_DEFS = [
    ("Faith & Comfort Pack", "For the spiritual heart-to-hearts", {"spiritual", "support"}),
    ("Late-Night Comfort Bundle", "For the 2am check-ins and tired weeks", {"stress", "routines"}),
    ("Inside-Joke Box", "Only makes sense if you've read the chat", {"inside_joke"}),
    ("Shared Cravings & Trips", "The food and travel you already talk about", {"food", "travel"}),
    ("Hobby & Play Kit", "Fuel for what they actually love doing", {"hobbies", "gaming", "music"}),
]

# Common words / phrases that are NOT real inside jokes.
_WEAK_PHRASES = {
    "alone", "scared", "thanks", "okay", "good", "sorry", "read", "morning",
    "night", "love", "miss", "yeah", "nice", "fine", "tomorrow", "together",
    "insecure", "at work", "at home", "good morning", "good night", "love you",
    "miss you", "right now", "i think", "i know", "i guess", "you know",
    "i mean", "of course", "by the way", "let me", "i was", "i am", "you are",
}


def _confidence(strong_refs: int, evidence_score: float) -> int:
    """Non-saturating confidence so cards vary instead of all reading 99%."""
    base = 55 + 25 * math.tanh(strong_refs / 6.0) + min(evidence_score, 4.0) * 2
    return int(max(50, min(95, round(base))))


def _best_quote(examples: list[dict], used: set) -> tuple[str, str, float]:
    """Pick the strongest human supporting quote that ACTUALLY contains a
    strong keyword (every example here is already strong-grounded), is a
    sensible length, is safe to display, and hasn't been used on another card."""
    best, best_score = None, -1.0
    for ex in examples:
        text = (ex.get("text") or "").strip()
        if not text or text in used:
            continue
        if not (12 <= len(text) <= 200):
            continue
        if not content_filter.safe_for_display(text):
            continue
        length_score = 1.0 - abs(110 - len(text)) / 150.0
        score = ex.get("strong_count", 1) * 1.5 + max(length_score, 0)
        if score > best_score:
            best, best_score = ex, score
    if not best:
        return "", "", 0.0
    used.add(best["text"].strip())
    return best["sender"], best["text"][:200], round(best_score, 1)


def _scan(messages: list[Message]) -> dict[str, Any]:
    signals = {name: {"score": 0.0, "strong_refs": 0, "weak_refs": 0,
                      "examples": [], "keywords": Counter()} for name in PATTERNS}
    links = []
    sender_support = defaultdict(int)

    for msg in messages:
        if not is_semantic_message(msg):
            continue
        raw = msg.text or ""
        for url in URL_RE.findall(raw):
            if any(host in url.lower() for host in ("spotify", "music", "youtu", "soundcloud")):
                links.append(url.rstrip(").,"))
        # Skip forwards / news / link dumps as gift EVIDENCE.
        if URL_RE.search(raw) or len(raw) > 240:
            continue
        text = suppress_noise(msg.normalized_text or raw)
        for name in PATTERNS:
            srx, wrx = _STRONG_RE[name], _WEAK_RE[name]
            strong = [s.lower() for s in (srx.findall(text) if srx else [])]
            weak = [w.lower() for w in (wrx.findall(text) if wrx else [])]
            if not strong and not weak:
                continue
            signals[name]["score"] += len(strong) * 3 + len(weak)
            signals[name]["strong_refs"] += len(strong)
            signals[name]["weak_refs"] += len(weak)
            signals[name]["keywords"].update(strong or weak)
            # Only STRONG-grounded messages become quote candidates, so the
            # supporting quote always actually contains a real signal word.
            if strong and len(signals[name]["examples"]) < 16:
                signals[name]["examples"].append({
                    "sender": msg.sender, "text": raw[:240],
                    "matched": strong, "strong_count": len(strong),
                })
            if name == "support" and strong:
                sender_support[msg.sender] += 1

    return {"signals": signals, "music_links": links[:20], "support_by_sender": dict(sender_support)}


def _gift(
    *,
    category: str,
    budget: str,
    title: str,
    reason: str,
    quote: str,
    quote_sender: str,
    gift_type: str,
    evidence_score: float = 0.0,
) -> dict[str, Any]:
    return {
        "category": category,
        "budget": budget,
        "title": title,
        "reason": reason,
        "quote": quote,
        "quote_sender": quote_sender,
        "gift_type": gift_type,
        "evidence_score": round(evidence_score, 1),
    }


def _ideas_for(category: str, signal: dict[str, Any], used: set) -> list[dict[str, Any]]:
    # Require a STRONG signal — no gift on weak/common words alone.
    if signal["strong_refs"] < 1 or not signal["examples"]:
        return []
    kws = [k for k, _ in signal["keywords"].most_common(5)]
    examples = signal["examples"]
    ideas: list[dict[str, Any]] = []

    def q(min_sc: float = 0.0):
        sender, text, score = _best_quote(examples, used)
        return sender, text, score

    if category == "spiritual":
        anchor = kws[0] if kws else "faith"
        title = f"Bible annotation tabs + journal because {anchor} shows up in your spiritual check-ins"
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="spiritual",
            title=title,
            reason=(
                f"Your chat references {anchor} in everyday faith moments — "
                "this gift mirrors that rhythm instead of generic devotion merch."
            ),
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "food":
        anchor = kws[0] if kws else "chai"
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="emotional",
            title=f"A late-night {anchor} thermos + handwritten playlist card",
            reason=(
                f"Because {anchor} appears when you comfort each other — "
                "turn that ritual into something they can hold."
            ),
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "stress":
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="late_night_comfort",
            title="Chai + weighted-eye-mask kit for exam/deadline weeks",
            reason=(
                "Stress words cluster around tired check-ins — "
                "match the gift to how you already soothe each other in text."
            ),
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "support":
        sender, quote, sc = q(2.5)
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="memory",
            title="Framed screenshot of the most supportive line in your chat",
            reason="Support is already your language — preserve the exact words, not a generic card.",
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "music":
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="digital",
            title="QR keychain opening a playlist built from songs you both named",
            reason="Music is already embedded in your thread — the gift should point to real shared tracks.",
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "hobbies":
        hobby = kws[0] if kws else "their hobby"
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="inside_reference",
            title=f"Meesho upgrade kit for their {hobby} corner",
            reason=f"You talk about {hobby} with real enthusiasm — fund the next step, not a random craft box.",
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "travel":
        place = kws[0] if kws else "that place"
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="medium_budget", gift_type="shared_experience",
            title=f"A day-trip plan toward {place} with a printed map of chat plans",
            reason=f"Travel mentions ({place}) are specific — anchor the experience to words you already used.",
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "routines":
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="routine",
            title="Commute comfort pouch (snack, thermos sleeve, tiny checklist)",
            reason="Routine talk shows how they survive hard days — support the rhythm they already named.",
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    elif category == "gaming":
        game = kws[0] if kws else "co-op"
        sender, quote, sc = q()
        ideas.append(_gift(
            category=category, budget="low_budget", gift_type="digital",
            title=f"Wallet card for your next {game} night",
            reason="Gaming is a shared language here — fund the session you already plan in chat.",
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))

    # Keep only gifts that ended up with a real supporting quote.
    return [i for i in ideas if i.get("quote")][:2]


def _inside_joke_gifts(phrases: list[dict], messages: list[Message], used: set) -> list[dict[str, Any]]:
    gifts = []
    candidates = [item for item in phrases if item["phrase_type"] == "relationship_specific"]
    for phrase in candidates:
        if len(gifts) >= 2:
            break
        ph = phrase["phrase"].strip().lower()
        # A real inside joke is a recurring MULTI-WORD phrase — never a single
        # common word like "together"/"tomorrow"/"insecure".
        if len(ph.split()) < 2 or ph in _WEAK_PHRASES:
            continue
        # The supporting message must contain the phrase, be human, and short.
        examples = [
            {"sender": m.sender, "text": (m.text or "")[:200], "strong_count": 2}
            for m in messages
            if ph in suppress_noise(m.normalized_text or m.text).lower()
            and not URL_RE.search(m.text or "") and 12 <= len(m.text or "") <= 200
        ]
        sender, quote, sc = _best_quote(examples, used)
        if not quote:
            continue
        gifts.append(_gift(
            category="inside_joke", budget="low_budget", gift_type="inside_joke",
            title=f"Custom mug / sticker pack referencing “{phrase['phrase']}”",
            reason=(
                f"“{phrase['phrase']}” keeps coming back in your chat — a shared "
                "phrase only the two of you would recognize."
            ),
            quote=quote, quote_sender=sender, evidence_score=sc,
        ))
    return gifts


def _build_bundles(all_ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bundles = []
    for name, blurb, cats in BUNDLE_DEFS:
        items = [g for g in all_ideas if g["category"] in cats]
        if items:
            bundles.append({
                "name": name, "blurb": blurb,
                "confidence": max(g.get("confidence", 0) for g in items),
                "items": [{"title": g["title"], "category": g["category"],
                           "confidence": g.get("confidence", 0)} for g in items[:4]],
            })
    return bundles


def compute_gifts(messages: list[Message], detected_format: str, senders: list[str]) -> dict[str, Any]:
    intelligence = build_intelligence(messages, senders)
    scanned = _scan(messages)
    signals = scanned["signals"]
    phrases = intelligence.semantic_phrases

    # References = STRONG keyword hits only (the meaningful, distinctive ones),
    # so "188 references" inflated by casual words can't happen.
    references = {name: data["strong_refs"] for name, data in signals.items()}
    top_keyword = {name: (data["keywords"].most_common(1)[0][0] if data["keywords"] else "")
                   for name, data in signals.items()}

    used_quotes: set = set()
    all_ideas: list[dict[str, Any]] = []
    for category, signal in signals.items():
        all_ideas.extend(_ideas_for(category, signal, used_quotes))
    all_ideas.extend(_inside_joke_gifts(phrases, messages, used_quotes))
    all_ideas = nlp.dedupe_gifts(all_ideas)

    # Annotate each gift with a confidence score, the data behind it, and an
    # emotional type breakdown so the card feels data-driven, not generic.
    for idea in all_ideas:
        refs = references.get(idea["category"], 0)
        idea["references"] = refs
        idea["top_keyword"] = top_keyword.get(idea["category"], "")
        idea["confidence"] = _confidence(refs, idea.get("evidence_score", 0))
        idea["types"] = TYPE_PROFILE.get(idea["category"], _DEFAULT_TYPES)

    all_ideas.sort(key=lambda g: (-g.get("confidence", 0), -g.get("evidence_score", 0)))
    bundles = _build_bundles(all_ideas)

    grouped = defaultdict(list)
    for idea in all_ideas:
        grouped[idea["budget"]].append(idea)

    return {
        "detected_format": detected_format,
        "senders": senders,
        "signals": {
            name: {
                "score": data["score"],
                "top_keywords": data["keywords"].most_common(8),
                # Strip the raw Message object (carries a datetime) — keep only
                # JSON-safe fields, else saving the job / dumping JSON crashes
                # with "object of type datetime is not JSON serializable".
                "examples": [
                    {"sender": e["sender"], "text": e["text"], "matched": e["matched"]}
                    for e in data["examples"][:3]
                ],
            }
            for name, data in signals.items()
        },
        "music_links": scanned["music_links"],
        "support_patterns": scanned["support_by_sender"],
        "bundles": bundles,
        "confidence_by_category": {
            name: _confidence(data["strong_refs"], data["score"])
            for name, data in signals.items() if data["strong_refs"] > 0
        },
        "relationship_intelligence": intelligence.summary(),
        "teasers": [
            "Every gift is tied to a chat quote that passed evidence matching.",
            "Inside-joke gifts only appear when a phrase repeats with a real message behind it.",
            "Unlock to see full WHY lines and India-realistic ideas.",
        ],
        "suggestions": {
            "low_budget": grouped["low_budget"][:8],
            "medium_budget": grouped["medium_budget"][:6],
            "premium": grouped["premium"][:4],
            "emotional_gifts": [g for g in all_ideas if g["gift_type"] in {"emotional", "memory", "spiritual", "late_night_comfort"}][:6],
            "experiences": [g for g in all_ideas if g["gift_type"] in {"shared_experience", "experience"}][:6],
            "hobby_gifts": [g for g in all_ideas if g["category"] in {"hobbies", "gaming", "music"}][:6],
            "stress_relief": [g for g in all_ideas if g["category"] == "stress" or g["gift_type"] == "late_night_comfort"][:4],
            "inside_jokes": [g for g in all_ideas if g["gift_type"] == "inside_joke"][:4],
        },
    }


async def _render_gift_pdf(job_id: str, gifts: dict[str, Any]) -> str:
    picks = []
    for key in ("inside_jokes", "emotional_gifts", "stress_relief", "low_budget"):
        picks.extend((gifts.get("suggestions") or {}).get(key, [])[:4])
    picks = nlp.dedupe_gifts(picks)[:10]
    html = _templates.get_template("gift_pdf.html").render(gifts=gifts, picks=picks)
    pdf_path = OUTPUT_DIR / job_id / "gift_engine.pdf"
    await pdf_render.render_html_to_pdf(html, pdf_path)
    return str(pdf_path.relative_to(OUTPUT_DIR.parent))


async def run_gift_engine_pipeline(job_id: str, upload_path: Path) -> None:
    try:
        phases = [
            {"name": "Parse export", "status": "in_progress", "progress": 25},
            {"name": "Match evidence", "status": "pending", "progress": 0},
            {"name": "Render gift PDF", "status": "pending", "progress": 0},
        ]
        jobs.update(job_id, state="parsing", progress=12, message="Reading your conversations…", phases=phases)
        parsed = await asyncio.to_thread(parse_chat, upload_path)
        if parsed.message_count > settings.MAX_MESSAGES:
            raise ValueError(f"Too many messages ({parsed.message_count}). Maximum is {settings.MAX_MESSAGES}.")

        phases[0] = {"name": "Parse export", "status": "done", "progress": 100}
        phases[1] = {"name": "Match evidence", "status": "in_progress", "progress": 55}
        jobs.update(job_id, state="generating_gifts", progress=58, message="Matching gifts to real quotes…", phases=phases)
        gifts = await asyncio.to_thread(
            compute_gifts, parsed.messages, parsed.detected_format, parsed.senders
        )
        gifts["parser_warnings"] = parsed.parser_warnings
        # Guarantee the whole structure is JSON-safe (no datetime/Message
        # objects) before it's stored on the job or written to disk.
        gifts = json.loads(json.dumps(gifts, default=str))

        phases[1] = {"name": "Match evidence", "status": "done", "progress": 100}
        phases[2] = {"name": "Render gift PDF", "status": "in_progress", "progress": 75}
        jobs.update(job_id, state="generating_gifts", progress=82, message="Writing gift cards…", stats=gifts, phases=phases)
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gift_engine.json").write_text(
            json.dumps(gifts, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        jobs.update(job_id, state="rendering", progress=92, message="Rendering gift keepsake PDF…", stats=gifts, phases=phases)
        pdf_rel = await _render_gift_pdf(job_id, gifts)

        phases[2] = {"name": "Render gift PDF", "status": "done", "progress": 100}
        jobs.update(
            job_id,
            state="done",
            progress=100,
            message="Gift recommendations ready",
            stats=gifts,
            preview_pdf=pdf_rel,
            full_pdf=pdf_rel,
            phases=phases,
        )
    except Exception as exc:
        jobs.update(job_id, state="failed", progress=100, message="Gift Engine failed", error=f"{type(exc).__name__}: {exc}")
        print(f"Gift Engine failed for job {job_id}:\n{traceback.format_exc()}")
