"""
Chat Wrapped — relationship intelligence from parsed messages (no LLM).
"""

import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import emoji as emoji_lib

from ..models import Message, MessageKind
from ..core import build_intelligence
from ..parsers import parse_chat
from ..pipeline import nlp_insights as nlp
from ..settings import OUTPUT_DIR, TEMPLATES_DIR, settings
from . import jobs, pdf_render
from jinja2 import Environment, FileSystemLoader, select_autoescape

SESSION_GAP_SECONDS = 30 * 60
STARTER_GAP_SECONDS = 6 * 60 * 60
NIGHT_HOURS = nlp.NIGHT_HOURS

_templates = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


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
    if not sessions:
        return {"messages": 0, "duration_seconds": 0, "top_sender": "", "start": "", "end": ""}
    longest = max(sessions, key=lambda g: (len(g), g[-1].timestamp - g[0].timestamp))
    duration = int((longest[-1].timestamp - longest[0].timestamp).total_seconds())
    return {
        "start": longest[0].timestamp.isoformat(),
        "end": longest[-1].timestamp.isoformat(),
        "duration_seconds": duration,
        "messages": len(longest),
        "top_sender": Counter(m.sender for m in longest).most_common(1)[0][0],
    }


def _support_counter(messages: list[Message]) -> Counter:
    c = Counter()
    for msg in messages:
        if msg.kind != MessageKind.TEXT or nlp.is_noise_message(msg.text):
            continue
        if nlp.token_set(msg.text) & nlp.POSITIVE or "you got this" in msg.text.lower():
            c[msg.sender] += 1
    return c


def _build_teasers(persona: dict, phrases: list, viral: dict, arc: list, top_starter: tuple) -> list[str]:
    teasers = [f"Your persona: {persona.get('name', 'hidden')} — unlock the full story behind it."]
    inside_reference = next(
        (phrase for phrase in phrases if phrase.get("phrase_type") == "relationship_specific"),
        None,
    )
    if inside_reference:
        teasers.append(
            f"“{inside_reference['phrase']}” returns in meaningful scenes — a shared reference worth remembering."
        )
    elif phrases:
        teasers.append(f"“{phrases[0]['phrase']}” recurs as one of your emotional rituals.")
    vp = viral.get("vulnerable_phrase")
    if vp:
        teasers.append(f"A phrase that only spikes in vulnerable moments is locked: “{vp['phrase']}”.")
    if arc:
        teasers.append(f"Latest chapter: {arc[-1]['title']} — earlier arcs are blurred.")
    if top_starter[0]:
        teasers.append(f"{top_starter[0]} breaks long silences most often.")
    return teasers[:6]


def compute_wrapped(
    messages: list[Message],
    detected_format: str,
    senders: list[str],
    raw_message_count: int = 0,
) -> dict[str, Any]:
    if not messages:
        return {"error": "no messages"}

    intelligence = build_intelligence(messages, senders)
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

    starter_counts[messages[0].sender] += 1
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
    support = _support_counter(messages)
    top_supporter = support.most_common(1)[0][0] if support else None

    phrases = intelligence.semantic_phrases
    language = intelligence.analytics["shared_language_evolution"]
    inside_jokes = language["inside_joke_candidates"]
    shared_vocab = language["recurring_phrases"]
    nicknames = language["nicknames"] or nlp.nicknames(messages, senders)
    persona = nlp.persona_from_signals(messages, phrases, hour_counts)
    arc = intelligence.analytics["relationship_timeline"]
    emotion = nlp.emotional_reading(messages, top_supporter)
    clock = nlp.emotional_clock_narrative(messages)
    viral = nlp.viral_moments(messages, phrases, persona, hour_counts)

    strongest_memory = intelligence.memories[0] if intelligence.memories else None
    if strongest_memory:
        evidence = strongest_memory.evidence_messages[0]
        emotion["why"] = strongest_memory.summary
        viral["most_human_moment"] = {
            "quote": evidence.text[:220],
            "sender": evidence.sender,
            "why": strongest_memory.summary,
        }
        cinematic = strongest_memory.summary
    elif persona.get("name"):
        cinematic = f"{persona['name']} — {emotion['why'][:120]}..."
    else:
        cinematic = emotion.get("why", "Your chat, read closely.")

    return {
        "detected_format": detected_format,
        "senders": senders,
        "cinematic_headline": cinematic,
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
        "shared_vocabulary": shared_vocab,
        "inside_jokes": inside_jokes,
        "nicknames": nicknames,
        "persona": {"name": persona["name"], "why": persona["why"]},
        "emotional_clock": clock,
        "relationship_arc": arc,
        "relationship_timeline": intelligence.analytics["relationship_timeline"],
        "emotional_trend": intelligence.analytics["emotional_trend"],
        "silence_drift": intelligence.analytics["silence_drift"],
        "communication_rhythm": intelligence.analytics["communication_rhythm"],
        "shared_language_evolution": intelligence.analytics["shared_language_evolution"],
        "meaningful_sessions": [
            session.to_dict() for session in intelligence.selected_sessions
        ],
        "strongest_moments": [memory.to_dict() for memory in intelligence.memories[:8]],
        "emotional_turning_points": [
            memory.to_dict() for memory in intelligence.memories
            if memory.memory_type in {"conflict", "reconnection", "reassurance", "comfort"}
        ][:8],
        "viral_moments": viral,
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
        "emotional_insights": {
            "tone": emotion["headline"],
            "why": emotion["why"],
            "top_supporter": top_supporter,
            "comfort_language": emotion.get("comfort_language", ""),
        },
        "teasers": _build_teasers(persona, phrases, viral, arc, top_starter),
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
            {"name": "Find rituals & arcs", "status": "pending", "progress": 0},
            {"name": "Render keepsake PDF", "status": "pending", "progress": 0},
        ]
        jobs.update(job_id, state="parsing", progress=10, message="Reading your conversations…", phases=phases)
        parsed = parse_chat(upload_path)
        if parsed.message_count > settings.MAX_MESSAGES:
            raise ValueError(f"Too many messages ({parsed.message_count}). Maximum is {settings.MAX_MESSAGES}.")

        phases[0] = {"name": "Parse export", "status": "done", "progress": 100}
        phases[1] = {"name": "Find rituals & arcs", "status": "in_progress", "progress": 50}
        jobs.update(job_id, state="generating_wrapped", progress=45, message="Finding inside jokes and emotional shifts…", phases=phases)
        wrapped = compute_wrapped(
            parsed.messages,
            parsed.detected_format,
            parsed.senders,
            parsed.raw_message_count,
        )
        wrapped["parser_warnings"] = parsed.parser_warnings

        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chat_wrapped.json").write_text(
            json.dumps(wrapped, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        phases[1] = {"name": "Find rituals & arcs", "status": "done", "progress": 100}
        phases[2] = {"name": "Render keepsake PDF", "status": "in_progress", "progress": 35}
        jobs.update(job_id, state="rendering", progress=82, message="Designing your keepsake PDF…", stats=wrapped, phases=phases)
        pdf_path = await _render_wrapped_pdf(job_id, wrapped)

        phases[2] = {"name": "Render keepsake PDF", "status": "done", "progress": 100}
        jobs.update(
            job_id,
            state="done",
            progress=100,
            message="Your Chat Wrapped is ready",
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
