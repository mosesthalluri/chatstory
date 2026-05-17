"""
Rule-based Chat Wrapped processor.

Pipeline:
  Chat export -> parser -> analytics -> JSON artifact
No LLM calls and no full-chat AI pass.
"""

import json
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import emoji as emoji_lib

from ..models import Message, MessageKind
from ..parsers import parse_chat
from ..pipeline.stats import STOPWORDS
from ..settings import OUTPUT_DIR, TEMPLATES_DIR, settings
from . import jobs, pdf_render
from jinja2 import Environment, FileSystemLoader, select_autoescape


SESSION_GAP_SECONDS = 30 * 60
STARTER_GAP_SECONDS = 6 * 60 * 60
NIGHT_HOURS = {0, 1, 2, 3, 4, 5}
POSITIVE_WORDS = {"love", "happy", "proud", "excited", "grateful", "thanks", "miss", "fun", "cute", "best"}
STRESS_WORDS = {"stress", "stressed", "tired", "anxious", "sad", "angry", "deadline", "exam", "overwhelmed", "sorry"}
EXTRA_STOPWORDS = set("""
what with why who whom whose where there here from into onto about after before
because thing things something anything everything today tomorrow yesterday
actually literally maybe probably already still even ever never always
""".split())
GENERIC_PHRASES = {"good morning", "good night", "love you", "miss you", "thank you", "how are"}

_templates = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


def _words(text: str) -> list[str]:
    text = emoji_lib.replace_emoji(text.lower(), replace=" ")
    return [
        word for word in re.findall(r"[a-z][a-z']{2,}", text)
        if word not in STOPWORDS and word not in EXTRA_STOPWORDS
    ]


def _emojis(text: str) -> list[str]:
    return [item["emoji"] for item in emoji_lib.emoji_list(text)]


def _longest_session(messages: list[Message]) -> dict[str, Any]:
    sessions: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages:
        if not current:
            current = [msg]
            continue
        gap = (msg.timestamp - current[-1].timestamp).total_seconds()
        if gap <= SESSION_GAP_SECONDS:
            current.append(msg)
        else:
            sessions.append(current)
            current = [msg]
    if current:
        sessions.append(current)

    longest = max(sessions, key=lambda group: (len(group), group[-1].timestamp - group[0].timestamp))
    duration = int((longest[-1].timestamp - longest[0].timestamp).total_seconds())
    return {
        "start": longest[0].timestamp.isoformat(),
        "end": longest[-1].timestamp.isoformat(),
        "duration_seconds": duration,
        "messages": len(longest),
        "top_sender": Counter(m.sender for m in longest).most_common(1)[0][0],
    }


def _shared_vocabulary(messages: list[Message]) -> list[dict[str, Any]]:
    per_sender: dict[str, Counter] = defaultdict(Counter)
    for msg in messages:
        if msg.kind == MessageKind.TEXT:
            per_sender[msg.sender].update(_words(msg.text))

    shared: list[dict[str, Any]] = []
    for word in set().union(*(counter.keys() for counter in per_sender.values())):
        users = {sender: count for sender, counter in per_sender.items() if (count := counter[word]) > 0}
        if len(users) >= 2:
            shared.append({"word": word, "count": sum(users.values()), "senders": users})

    return sorted(shared, key=lambda item: item["count"], reverse=True)[:50]


def _inside_jokes(messages: list[Message]) -> list[dict[str, Any]]:
    phrases = Counter()
    examples: dict[str, str] = {}
    for msg in messages:
        tokens = _words(msg.text) if msg.kind == MessageKind.TEXT else []
        for n in (2, 3):
            for parts in zip(*(tokens[i:] for i in range(n))):
                phrase = " ".join(parts)
                if phrase not in GENERIC_PHRASES:
                    phrases[phrase] += 1
                    examples.setdefault(phrase, msg.text[:140])
    return [
        {"phrase": phrase, "count": count, "quote": examples.get(phrase, "")}
        for phrase, count in phrases.most_common(8)
        if count >= 2
    ]


def _nicknames(messages: list[Message], senders: list[str]) -> list[dict[str, Any]]:
    names = {s.lower().split()[0] for s in senders if s}
    found = Counter()
    for msg in messages:
        text = msg.text.lower()
        for word in re.findall(r"\b[a-z]{3,14}\b", text):
            if word.endswith(("u", "y", "ie")) and word not in names and word not in STOPWORDS:
                found[word] += 1
    return [{"name": name, "count": count} for name, count in found.most_common(8) if count >= 2]


def _emotional_insights(messages: list[Message]) -> dict[str, Any]:
    monthly = defaultdict(lambda: {"positive": 0, "stress": 0, "messages": 0})
    support = Counter()
    for msg in messages:
        if msg.kind != MessageKind.TEXT:
            continue
        words = set(_words(msg.text))
        bucket = msg.timestamp.strftime("%Y-%m")
        monthly[bucket]["messages"] += 1
        monthly[bucket]["positive"] += len(words & POSITIVE_WORDS)
        monthly[bucket]["stress"] += len(words & STRESS_WORDS)
        if words & POSITIVE_WORDS or "you got this" in msg.text.lower():
            support[msg.sender] += 1
    timeline = [
        {"month": month, **scores}
        for month, scores in sorted(monthly.items())
    ]
    top_supporter = support.most_common(1)[0][0] if support else None
    return {
        "timeline": timeline,
        "top_supporter": top_supporter,
        "tone": "warm" if sum(x["positive"] for x in timeline) >= sum(x["stress"] for x in timeline) else "high-stress",
        "why": (
            "Your conversations are filled with reassurance, check-ins, and positive words."
            if sum(x["positive"] for x in timeline) >= sum(x["stress"] for x in timeline)
            else "Stress words and apology/pressure signals appear more often than reassurance signals."
        ),
    }


def _emotional_clock(messages: list[Message]) -> dict[str, Any]:
    """Hour-of-day mood buckets from vocabulary signals."""
    buckets = {
        "vulnerable_hours": Counter(),
        "comfort_hours": Counter(),
        "high_energy_hours": Counter(),
        "deep_talk_hours": Counter(),
    }
    vuln = {"sorry", "sad", "miss", "anxious", "tired", "alone", "scared", "cry"}
    comfort = {"tea", "chai", "coffee", "sleep", "hug", "home", "calm", "okay", "fine"}
    energy = {"haha", "lol", "party", "let's", "yay", "excited", "dance", "game"}
    deep = {"feel", "life", "future", "trust", "mean", "relationship", "forever", "why"}

    for msg in messages:
        if msg.kind != MessageKind.TEXT:
            continue
        words = set(_words(msg.text))
        hour = msg.timestamp.hour
        if words & vuln:
            buckets["vulnerable_hours"][hour] += 1
        if words & comfort:
            buckets["comfort_hours"][hour] += 1
        if words & energy or len(_emojis(msg.text)) >= 2:
            buckets["high_energy_hours"][hour] += 1
        if words & deep or len(msg.text) > 120:
            buckets["deep_talk_hours"][hour] += 1

    def _top_label(counter: Counter) -> dict[str, Any]:
        if not counter:
            return {"hour": None, "label": "subtle", "why": "No strong hourly pattern crossed the threshold."}
        hour, count = counter.most_common(1)[0]
        return {
            "hour": hour,
            "count": count,
            "label": f"{hour:02d}:00",
            "why": f"This hour appeared {count} times with this emotional texture in your export.",
        }

    return {
        "vulnerable": _top_label(buckets["vulnerable_hours"]),
        "comfort": _top_label(buckets["comfort_hours"]),
        "high_energy": _top_label(buckets["high_energy_hours"]),
        "deep_conversation": _top_label(buckets["deep_talk_hours"]),
    }


def _relationship_arc(messages: list[Message]) -> list[dict[str, str]]:
    if len(messages) < 20:
        return [{"phase": "current", "title": "Your story so far", "why": "Not enough messages yet for a multi-phase arc."}]

    start = messages[0].timestamp
    end = messages[-1].timestamp
    span = max((end - start).total_seconds(), 1)
    quintile_msgs: list[list[Message]] = [[] for _ in range(5)]
    for msg in messages:
        idx = min(4, int(((msg.timestamp - start).total_seconds() / span) * 5))
        quintile_msgs[idx].append(msg)

    phase_names = ["beginning", "comfort", "chaos", "support", "current"]
    phase_titles = [
        "The Beginning",
        "The Comfort Phase",
        "The Chaos Phase",
        "The Support Phase",
        "Where You Are Now",
    ]
    arc = []
    for i, chunk in enumerate(quintile_msgs):
        if not chunk:
            continue
        text_count = sum(1 for m in chunk if m.kind == MessageKind.TEXT)
        emoji_count = sum(len(_emojis(m.text)) for m in chunk if m.kind == MessageKind.TEXT)
        words = set(w for m in chunk if m.kind == MessageKind.TEXT for w in _words(m.text))
        stress = len(words & STRESS_WORDS)
        positive = len(words & POSITIVE_WORDS)
        if stress > positive + 2:
            tone = "intense"
        elif positive > stress + 2:
            tone = "warm"
        elif emoji_count > text_count * 0.4:
            tone = "playful"
        else:
            tone = "steady"
        arc.append({
            "phase": phase_names[i],
            "title": phase_titles[i],
            "tone": tone,
            "messages": len(chunk),
            "why": (
                f"In this stretch, your chat reads as {tone} — "
                f"{text_count} text messages and recurring rituals shaped this chapter."
            ),
        })
    return arc


def _cinematic_headline(persona: dict[str, str], jokes: list[dict], emotion: dict) -> str:
    joke_hint = jokes[0]["phrase"] if jokes else ""
    if joke_hint:
        return f"{persona.get('name', 'Your bond')} — carried by “{joke_hint}” and late-night honesty."
    return f"{persona.get('name', 'Your bond')} — {emotion.get('tone', 'a private')} rhythm only you two share."


def _build_teasers(
    persona: dict,
    jokes: list,
    clock: dict,
    arc: list,
    top_starter: tuple,
) -> list[str]:
    teasers = [
        f"Your dynamic persona: {persona.get('name', 'hidden')} — unlock to read why.",
    ]
    if jokes:
        teasers.append(f"Inside joke detected: “{jokes[0]['phrase']}” appears {jokes[0]['count']} times.")
    if clock.get("comfort", {}).get("hour") is not None:
        teasers.append(
            f"Comfort hour peaks around {clock['comfort']['label']} — full emotional clock is locked."
        )
    if arc:
        teasers.append(f"Relationship arc: you are in “{arc[-1]['title']}” — earlier phases are blurred.")
    if top_starter[0]:
        teasers.append(f"{top_starter[0]} restarts the thread after long silences most often.")
    return teasers[:5]


def _persona(messages: list[Message], emotion: dict[str, Any]) -> dict[str, str]:
    night_ratio = sum(1 for m in messages if m.timestamp.hour in NIGHT_HOURS) / max(len(messages), 1)
    emoji_total = sum(len(_emojis(m.text)) for m in messages if m.kind == MessageKind.TEXT)
    vocab = Counter(word for m in messages for word in (_words(m.text) if m.kind == MessageKind.TEXT else []))
    if night_ratio > 0.22:
        return {"name": "The Night Owls", "why": "A meaningful slice of your chat happens after midnight, when conversations tend to get slower and more honest."}
    if emotion["tone"] == "warm":
        return {"name": "The Comfort Pair", "why": "Reassurance, affection, and check-ins show up enough to make support one of your strongest patterns."}
    if emoji_total > len(messages) * 0.45:
        return {"name": "The Chaos Duo", "why": "Your messages carry high visual energy through emojis, reactions, and short expressive bursts."}
    if any(w in vocab for w in ("think", "life", "feel", "why", "maybe")):
        return {"name": "The Philosophers", "why": "Your vocabulary leans toward reflection, feelings, and open-ended conversation."}
    return {"name": "The Meme Ministers", "why": "Your bond is carried by recurring phrases, quick replies, and shared conversational rituals."}


def compute_wrapped(
    messages: list[Message],
    detected_format: str,
    senders: list[str],
    raw_message_count: int = 0,
) -> dict[str, Any]:
    if not messages:
        return {"error": "no messages"}

    text_messages = [m for m in messages if m.kind == MessageKind.TEXT]
    media_messages = [m for m in messages if m.kind == MessageKind.MEDIA_PLACEHOLDER]
    active_dates = {m.timestamp.date() for m in messages}
    hour_counts = Counter(m.timestamp.hour for m in messages)
    day_counts = Counter(m.timestamp.date().isoformat() for m in messages)
    sender_counts = Counter(m.sender for m in messages)

    emoji_counts = Counter()
    heatmap: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    night_counts = Counter()
    starter_counts = Counter()

    first = messages[0]
    starter_counts[first.sender] += 1
    for msg in messages:
        heatmap[msg.timestamp.date().isoformat()][str(msg.timestamp.hour)] += 1
        if msg.timestamp.hour in NIGHT_HOURS:
            night_counts[msg.sender] += 1
        if msg.kind == MessageKind.TEXT:
            emoji_counts.update(_emojis(msg.text))

    for prev, cur in zip(messages, messages[1:]):
        if (cur.timestamp - prev.timestamp).total_seconds() >= STARTER_GAP_SECONDS:
            starter_counts[cur.sender] += 1

    top_starter = starter_counts.most_common(1)[0] if starter_counts else ("", 0)
    emotion = _emotional_insights(messages)
    jokes = _inside_jokes(messages)
    persona = _persona(messages, emotion)
    clock = _emotional_clock(messages)
    arc = _relationship_arc(messages)
    headline = _cinematic_headline(persona, jokes, emotion)
    return {
        "detected_format": detected_format,
        "senders": senders,
        "cinematic_headline": headline,
        "total_messages": raw_message_count or len(messages),
        "conversation_turns": len(messages),
        "text_messages": len(text_messages),
        "media_count": len(media_messages),
        "active_days": len(active_dates),
        "first_message_at": messages[0].timestamp.isoformat(),
        "last_message_at": messages[-1].timestamp.isoformat(),
        "messages_per_sender": dict(sender_counts),
        "longest_conversation_session": _longest_session(messages),
        "emoji_frequency": [{"emoji": e, "count": c} for e, c in emoji_counts.most_common(25)],
        "shared_vocabulary": _shared_vocabulary(messages),
        "inside_jokes": jokes,
        "nicknames": _nicknames(messages, senders),
        "persona": persona,
        "emotional_clock": clock,
        "relationship_arc": arc,
        "night_owl": {
            "hours": sorted(NIGHT_HOURS),
            "total_messages": sum(night_counts.values()),
            "messages_per_sender": dict(night_counts),
            "top_sender": night_counts.most_common(1)[0][0] if night_counts else None,
        },
        "conversation_starters": {
            "gap_hours": STARTER_GAP_SECONDS // 3600,
            "counts": dict(starter_counts),
            "top_sender": top_starter[0],
            "top_count": top_starter[1],
        },
        "emotional_insights": emotion,
        "teasers": _build_teasers(persona, jokes, clock, arc, top_starter),
        "heatmap": {
            "by_day_hour": {day: dict(hours) for day, hours in heatmap.items()},
            "hour_totals": dict(hour_counts),
            "day_totals": dict(day_counts),
        },
    }


async def _render_wrapped_pdf(job_id: str, wrapped: dict[str, Any]) -> str:
    html = _templates.get_template("wrapped_pdf.html").render(stats=wrapped)
    pdf_path = OUTPUT_DIR / job_id / "chat_wrapped.pdf"
    await pdf_render.render_html_to_pdf(html, pdf_path)
    return str(pdf_path.relative_to(OUTPUT_DIR.parent))


async def run_chat_wrapped_pipeline(job_id: str, upload_path: Path) -> None:
    try:
        phases = [
            {"name": "Parse export", "status": "in_progress", "progress": 25},
            {"name": "Compute analytics", "status": "pending", "progress": 0},
            {"name": "Render PDF", "status": "pending", "progress": 0},
        ]
        jobs.update(job_id, state="parsing", progress=10, message="Reading chat export...", phases=phases)
        parsed = parse_chat(upload_path)
        if parsed.message_count > settings.MAX_MESSAGES:
            raise ValueError(f"Too many messages ({parsed.message_count}). Maximum is {settings.MAX_MESSAGES}.")

        phases[0] = {"name": "Parse export", "status": "done", "progress": 100}
        phases[1] = {"name": "Compute analytics", "status": "in_progress", "progress": 50}
        jobs.update(job_id, state="generating_wrapped", progress=45, message="Computing Chat Wrapped analytics...", phases=phases)
        wrapped = compute_wrapped(
            parsed.messages,
            parsed.detected_format,
            parsed.senders,
            parsed.raw_message_count,
        )
        wrapped["parser_warnings"] = parsed.parser_warnings

        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "chat_wrapped.json"
        result_path.write_text(json.dumps(wrapped, indent=2, ensure_ascii=False), encoding="utf-8")

        phases[1] = {"name": "Compute analytics", "status": "done", "progress": 100}
        phases[2] = {"name": "Render PDF", "status": "in_progress", "progress": 35}
        jobs.update(job_id, state="rendering", progress=82, message="Rendering branded PDF...", stats=wrapped, phases=phases)
        pdf_path = await _render_wrapped_pdf(job_id, wrapped)

        phases[2] = {"name": "Render PDF", "status": "done", "progress": 100}
        jobs.update(
            job_id,
            state="done",
            progress=100,
            message="Chat Wrapped analytics ready",
            stats=wrapped,
            preview_pdf=pdf_path,
            full_pdf=pdf_path,
            phases=phases,
        )
    except Exception as exc:
        jobs.update(
            job_id,
            state="failed",
            progress=100,
            message="Chat Wrapped failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"Chat Wrapped failed for job {job_id}:\n{traceback.format_exc()}")
