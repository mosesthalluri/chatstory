"""
Rule-based Gift Engine.

Uses parsed chat messages and keyword/link signals to produce categorized
gift ideas. No LLM calls.
"""

import json
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..models import Message, MessageKind
from ..parsers import parse_chat
from ..settings import OUTPUT_DIR, TEMPLATES_DIR, settings
from . import jobs, pdf_render
from jinja2 import Environment, FileSystemLoader, select_autoescape

_templates = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


PATTERNS = {
    "hobbies": {
        "keywords": ["paint", "sketch", "book", "read", "photography", "plant", "gym", "yoga", "dance", "cook", "bake"],
        "ideas": {
            "low_budget": ["A curated hobby kit from Amazon/Meesho", "A personalized notebook with their recurring phrase"],
            "medium_budget": ["A local workshop/class pass", "Upgraded tools for their favorite hobby"],
            "premium": ["A full hobby upgrade box", "A weekend retreat around that interest"],
        },
    },
    "music": {
        "keywords": ["spotify", "playlist", "song", "album", "concert", "guitar", "piano", "music", "singer"],
        "ideas": {
            "low_budget": ["A Spotify-code keychain/card", "A custom phone wallpaper of shared songs"],
            "medium_budget": ["BookMyShow concert/movie-music night", "Quality earbuds case and music journal"],
            "premium": ["Premium headphones", "A live music getaway"],
        },
    },
    "travel": {
        "keywords": ["trip", "flight", "hotel", "beach", "mountain", "travel", "vacation", "passport", "roadtrip"],
        "ideas": {
            "low_budget": ["A mini travel pouch", "A framed map print of a place they mention"],
            "medium_budget": ["A planned day trip nearby", "Packing cubes and a travel comfort set"],
            "premium": ["A surprise weekend stay", "Flight/train fund or experience voucher"],
        },
    },
    "stress": {
        "keywords": ["tired", "stress", "stressed", "anxious", "overwhelmed", "exam", "deadline", "workload", "burnout"],
        "ideas": {
            "low_budget": ["A comfort snack box with a handwritten reset note", "A calming chai/tea and sleep kit"],
            "medium_budget": ["Urban Company spa/massage voucher", "Weighted blanket or sunrise lamp"],
            "premium": ["Wellness staycation", "Therapy or coaching support fund"],
        },
    },
    "food": {
        "keywords": ["coffee", "tea", "pizza", "sushi", "burger", "cake", "chocolate", "restaurant", "dinner", "food"],
        "ideas": {
            "low_budget": ["Their favorite Indian snack box", "A coffee date kit"],
            "medium_budget": ["Dinner at a restaurant they mentioned", "A cooking class together"],
            "premium": ["Chef's table experience", "A premium coffee/dessert subscription"],
        },
    },
    "gaming": {
        "keywords": ["game", "gaming", "xbox", "playstation", "ps5", "steam", "valorant", "minecraft", "fortnite"],
        "ideas": {
            "low_budget": ["In-game currency or a small Steam/PlayStation gift card", "A cable organizer for their setup"],
            "medium_budget": ["Controller accessories or desk lighting", "A co-op game night bundle"],
            "premium": ["Gaming chair upgrade", "Console or monitor fund"],
        },
    },
    "routines": {
        "keywords": ["morning", "night", "sleep", "walk", "office", "class", "commute", "routine", "workout"],
        "ideas": {
            "low_budget": ["A routine-friendly daily planner", "A small commute comfort kit"],
            "medium_budget": ["Smart mug or desk organizer", "Fitness or habit-tracking accessory"],
            "premium": ["Smartwatch", "A home routine upgrade bundle"],
        },
    },
    "support": {
        "keywords": ["proud", "you got this", "i'm here", "miss you", "love you", "take care", "feel better"],
        "ideas": {
            "low_budget": ["A memory jar of supportive messages", "A framed chat quote"],
            "medium_budget": ["A personalized care package", "A photo book of shared moments"],
            "premium": ["A meaningful jewelry or keepsake piece", "A planned experience around their love language"],
        },
    },
}

URL_RE = re.compile(r"https?://\S+", re.I)


def _scan(messages: list[Message]) -> dict[str, Any]:
    signals = {
        name: {"score": 0, "examples": [], "keywords": Counter()}
        for name in PATTERNS
    }
    links = []
    sender_support = defaultdict(int)

    for msg in messages:
        if msg.kind != MessageKind.TEXT:
            continue
        text = msg.text.lower()
        for url in URL_RE.findall(msg.text):
            if any(host in url.lower() for host in ("spotify", "music", "youtu", "soundcloud")):
                links.append(url.rstrip(").,"))
        for name, cfg in PATTERNS.items():
            matched = [kw for kw in cfg["keywords"] if kw in text]
            if not matched:
                continue
            signals[name]["score"] += len(matched)
            signals[name]["keywords"].update(matched)
            if len(signals[name]["examples"]) < 5:
                signals[name]["examples"].append({"sender": msg.sender, "text": msg.text[:160]})
            if name == "support":
                sender_support[msg.sender] += 1

    return {"signals": signals, "music_links": links[:20], "support_by_sender": dict(sender_support)}


def _top_keywords(signal: dict[str, Any], limit: int = 3) -> list[str]:
    return [k for k, _ in signal["keywords"].most_common(limit)]


def _example(signal: dict[str, Any]) -> tuple[str, str]:
    if not signal["examples"]:
        return "", ""
    ex = signal["examples"][0]
    return ex.get("sender", ""), ex.get("text", "")[:180]


def _build_gift(
    *,
    category: str,
    budget: str,
    title: str,
    reason: str,
    quote: str,
    quote_sender: str,
    gift_type: str,
) -> dict[str, str]:
    return {
        "category": category,
        "budget": budget,
        "title": title,
        "reason": reason,
        "quote": quote,
        "quote_sender": quote_sender,
        "gift_type": gift_type,
    }


def _ideas_for(category: str, signal: dict[str, Any]) -> list[dict[str, str]]:
    if signal["score"] <= 0:
        return []
    kws = _top_keywords(signal)
    kw_label = ", ".join(kws) if kws else category.replace("_", " ")
    sender, quote = _example(signal)
    ideas: list[dict[str, str]] = []

    if category == "food":
        anchor = kws[0] if kws else "chai"
        ideas.append(_build_gift(
            category=category, budget="low_budget", gift_type="emotional",
            title=f"A handwritten {anchor}-and-snack care box from a local bakery",
            reason=(
                f"Because {anchor} keeps showing up when you two talk about comfort and small rituals — "
                f"this turns a chat habit into something tangible."
            ),
            quote=quote, quote_sender=sender,
        ))
        ideas.append(_build_gift(
            category=category, budget="medium_budget", gift_type="experience",
            title=f"Swiggy/Zomato night for their favourite {anchor} order",
            reason=f"They've mentioned {kw_label} enough that a shared food night will feel personal, not generic.",
            quote=quote, quote_sender=sender,
        ))
    elif category == "stress":
        ideas.append(_build_gift(
            category=category, budget="low_budget", gift_type="late_night_comfort",
            title="A calming chai-and-playlist care kit for heavy weeks",
            reason=(
                "Stress signals cluster around exams, deadlines, and tired check-ins — "
                "a low-cost night kit matches how they already seek comfort in chat."
            ),
            quote=quote, quote_sender=sender,
        ))
        ideas.append(_build_gift(
            category=category, budget="medium_budget", gift_type="emotional",
            title="Urban Company at-home reset (30–45 min) with a note from you",
            reason=f"Words like {kw_label} repeat when they're overwhelmed — practical relief plus emotional presence.",
            quote=quote, quote_sender=sender,
        ))
    elif category == "support":
        ideas.append(_build_gift(
            category=category, budget="low_budget", gift_type="memory",
            title="A framed print of one supportive line from your chat",
            reason="Support messages are a pattern here — preserving the exact words matters more than a generic card.",
            quote=quote, quote_sender=sender,
        ))
        ideas.append(_build_gift(
            category=category, budget="medium_budget", gift_type="digital",
            title="A private Notion/Google Doc ‘open when you need me’ letter set",
            reason="Your chat already functions as emotional backup — this digitizes that role beautifully.",
            quote=quote, quote_sender=sender,
        ))
    elif category == "music":
        ideas.append(_build_gift(
            category=category, budget="low_budget", gift_type="digital",
            title="A QR keychain that opens a playlist built from songs you both named",
            reason=f"Music links and words ({kw_label}) appear often — tie the gift to songs already in the thread.",
            quote=quote, quote_sender=sender,
        ))
        ideas.append(_build_gift(
            category=category, budget="medium_budget", gift_type="experience",
            title="BookMyShow + dinner near a venue you mentioned",
            reason="Shared listening is already part of your bond — extend it to a real-world night out.",
            quote=quote, quote_sender=sender,
        ))
    elif category == "hobbies":
        hobby = kws[0] if kws else "their hobby"
        ideas.append(_build_gift(
            category=category, budget="low_budget", gift_type="inside_reference",
            title=f"A Meesho/Amazon mini-kit to upgrade their {hobby} corner",
            reason=f"They talk about {kw_label} with enthusiasm — small tools beat a vague ‘hobby kit’.",
            quote=quote, quote_sender=sender,
        ))
    elif category == "travel":
        place = kws[0] if kws else "that trip"
        ideas.append(_build_gift(
            category=category, budget="medium_budget", gift_type="shared_experience",
            title=f"A planned day trip toward {place} with a printed chat map of plans you made",
            reason=f"Travel chatter about {kw_label} is specific — anchor the gift to the place you already discuss.",
            quote=quote, quote_sender=sender,
        ))
    elif category == "routines":
        ideas.append(_build_gift(
            category=category, budget="low_budget", gift_type="routine",
            title="A commute/morning pouch (thermos sleeve, snack, handwritten checklist)",
            reason=f"Routine words ({kw_label}) show how they structure hard days — support the rhythm, not random stuff.",
            quote=quote, quote_sender=sender,
        ))
    elif category == "gaming":
        game = kws[0] if kws else "co-op games"
        ideas.append(_build_gift(
            category=category, budget="low_budget", gift_type="digital",
            title=f"Steam/PlayStation wallet card for {game} night",
            reason=f"Gaming is a shared language here ({kw_label}) — fund the next session you already plan in chat.",
            quote=quote, quote_sender=sender,
        ))
    else:
        for budget, labels in PATTERNS[category]["ideas"].items():
            for label in labels[:1]:
                ideas.append(_build_gift(
                    category=category, budget=budget, gift_type="contextual",
                    title=label,
                    reason=f"Suggested from recurring signals: {kw_label}.",
                    quote=quote, quote_sender=sender,
                ))
    return ideas[:3]


def compute_gifts(messages: list[Message], detected_format: str, senders: list[str]) -> dict[str, Any]:
    scanned = _scan(messages)
    signals = scanned["signals"]
    all_ideas = []
    for category, signal in signals.items():
        all_ideas.extend(_ideas_for(category, signal))

    grouped = defaultdict(list)
    for idea in all_ideas:
        grouped[idea["budget"]].append(idea)

    emotional = [idea for idea in all_ideas if idea["category"] == "support"][:4]
    experiences = [idea for idea in all_ideas if idea["category"] in {"travel", "music", "food", "hobbies"} and "voucher" not in idea["title"].lower()][:6]
    hobby = [idea for idea in all_ideas if idea["category"] in {"hobbies", "gaming", "music"}][:6]
    stress = [idea for idea in all_ideas if idea["category"] == "stress"][:4]

    return {
        "detected_format": detected_format,
        "senders": senders,
        "signals": {
            name: {
                "score": data["score"],
                "top_keywords": data["keywords"].most_common(8),
                "examples": data["examples"],
            }
            for name, data in signals.items()
        },
        "music_links": scanned["music_links"],
        "support_patterns": scanned["support_by_sender"],
        "teasers": [
            "At least one gift is tied to a real supporting quote from your chat.",
            "Late-night comfort and chai/tea rituals may unlock a specific low-cost idea.",
            "Premium cards hide the full WHY and the exact message we used as evidence.",
        ],
        "suggestions": {
            "low_budget": grouped["low_budget"][:8],
            "medium_budget": grouped["medium_budget"][:8],
            "premium": grouped["premium"][:8],
            "emotional_gifts": emotional,
            "experiences": experiences,
            "hobby_gifts": hobby,
            "stress_relief": stress,
        },
    }


async def _render_gift_pdf(job_id: str, gifts: dict[str, Any]) -> str:
    picks = []
    for key in ("emotional_gifts", "stress_relief", "experiences", "low_budget"):
        picks.extend((gifts.get("suggestions") or {}).get(key, [])[:3])
    html = _templates.get_template("gift_pdf.html").render(gifts=gifts, picks=picks[:12])
    pdf_path = OUTPUT_DIR / job_id / "gift_engine.pdf"
    await pdf_render.render_html_to_pdf(html, pdf_path)
    return str(pdf_path.relative_to(OUTPUT_DIR.parent))


async def run_gift_engine_pipeline(job_id: str, upload_path: Path) -> None:
    try:
        phases = [
            {"name": "Parse export", "status": "in_progress", "progress": 25},
            {"name": "Extract signals", "status": "pending", "progress": 0},
            {"name": "Generate ideas", "status": "pending", "progress": 0},
        ]
        jobs.update(job_id, state="parsing", progress=12, message="Reading chat export...", phases=phases)
        parsed = parse_chat(upload_path)
        if parsed.message_count > settings.MAX_MESSAGES:
            raise ValueError(f"Too many messages ({parsed.message_count}). Maximum is {settings.MAX_MESSAGES}.")

        phases[0] = {"name": "Parse export", "status": "done", "progress": 100}
        phases[1] = {"name": "Extract signals", "status": "in_progress", "progress": 55}
        jobs.update(job_id, state="generating_gifts", progress=58, message="Finding meaningful gift signals...", phases=phases)
        gifts = compute_gifts(parsed.messages, parsed.detected_format, parsed.senders)
        gifts["parser_warnings"] = parsed.parser_warnings

        phases[1] = {"name": "Extract signals", "status": "done", "progress": 100}
        phases[2] = {"name": "Generate ideas", "status": "in_progress", "progress": 75}
        jobs.update(job_id, state="generating_gifts", progress=82, message="Organizing gift recommendations...", stats=gifts, phases=phases)
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gift_engine.json").write_text(
            json.dumps(gifts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        phases[2] = {"name": "Generate ideas", "status": "in_progress", "progress": 90}
        jobs.update(job_id, state="rendering", progress=92, message="Rendering gift PDF...", stats=gifts, phases=phases)
        pdf_rel = await _render_gift_pdf(job_id, gifts)

        phases[2] = {"name": "Generate ideas", "status": "done", "progress": 100}
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
