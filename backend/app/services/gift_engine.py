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
from ..settings import OUTPUT_DIR, settings
from . import jobs


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


def _ideas_for(category: str, signal: dict[str, Any]) -> list[dict[str, str]]:
    if signal["score"] <= 0:
        return []
    ideas = []
    quote = signal["examples"][0]["text"] if signal["examples"] else ""
    sender = signal["examples"][0]["sender"] if signal["examples"] else ""
    for budget, labels in PATTERNS[category]["ideas"].items():
        for label in labels:
            ideas.append({
                "category": category,
                "budget": budget,
                "title": label,
                "reason": f"Suggested because {', '.join(signal['keywords'].keys()) or category} appears repeatedly in the chat.",
                "quote": quote,
                "quote_sender": sender,
            })
    return ideas


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
            "We found one gift idea tied to emotional support patterns.",
            "Some recommendations are locked because they use direct supporting quotes.",
            "Premium ideas connect routines, stress, and shared experiences.",
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
        jobs.update(job_id, state="analyzing", progress=58, message="Finding meaningful gift signals...", phases=phases)
        gifts = compute_gifts(parsed.messages, parsed.detected_format, parsed.senders)
        gifts["parser_warnings"] = parsed.parser_warnings

        phases[1] = {"name": "Extract signals", "status": "done", "progress": 100}
        phases[2] = {"name": "Generate ideas", "status": "in_progress", "progress": 75}
        jobs.update(job_id, state="analyzing", progress=82, message="Organizing gift recommendations...", stats=gifts, phases=phases)
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gift_engine.json").write_text(
            json.dumps(gifts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        phases[2] = {"name": "Generate ideas", "status": "done", "progress": 100}
        jobs.update(job_id, state="done", progress=100, message="Gift recommendations ready", stats=gifts, phases=phases)
    except Exception as exc:
        jobs.update(job_id, state="error", progress=100, message="Gift Engine failed", error=f"{type(exc).__name__}: {exc}")
        print(f"Gift Engine failed for job {job_id}:\n{traceback.format_exc()}")
