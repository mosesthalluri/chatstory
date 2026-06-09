"""
Chat Wrapped — a grounded, story-driven recap built purely from the parsed
messages (no LLM, no hallucinations).

Everything here is derived directly from real messages and timestamps, so
nothing is invented. Forwarded/media/system lines and obvious noise are
excluded from anything we quote, so the recap never says a forwarded reel
caption "sounds like you two".

Output is a flat, JSON-safe dict (strings / numbers / lists / dicts only).
"""

from __future__ import annotations

import asyncio
import calendar
import json
import re
import traceback
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import emoji as emoji_lib

from ..models import Message, MessageKind
from ..pipeline import nlp_insights as nlp
from ..parsers import parse_chat
from ..settings import OUTPUT_DIR, TEMPLATES_DIR, settings
from . import jobs, pdf_render
from jinja2 import Environment, FileSystemLoader, select_autoescape

NIGHT_HOURS = nlp.NIGHT_HOURS

_templates = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)

# --------------------------------------------------------------------------
# Small lexicons (grounded keyword/phrase detection — no model)
# --------------------------------------------------------------------------
CARE_PHRASES = [
    "take care", "are you okay", "you okay", "i'm here", "im here", "i am here",
    "don't worry", "dont worry", "proud of you", "you got this", "get well",
    "feel better", "here for you", "stay safe", "be safe", "thinking of you",
]
LOVE_PHRASES = ["i love you", "love you", "love u", "ily", "luv u", "luv you"]
MISS_PHRASES = ["miss you", "miss u", "missing you", "missed you"]
PLAN_PHRASES = ["let's", "lets ", "let us", "shall we", "plan", "meet up", "meet at",
                "see you", "catch up", "wanna ", "want to go", "this weekend"]
APOLOGY_PHRASES = ["sorry", "my bad", "apologi", "forgive me", "didn't mean", "didnt mean"]
CELEBRATE_PHRASES = ["congrats", "congratulations", "happy birthday", "we did it",
                     "got the job", "cleared", "passed", "selected", "promotion",
                     "well done", "proud"]
GM_PHRASES = ["good morning", "gud morning", "gm", "morning!"]
GN_PHRASES = ["good night", "gud night", "gn", "goodnight", "night night"]

STOPWORDS = set("""
a an the and or but if then else when so for nor on at by to of in i me my mine
you your yours we us our he him his she her hers it its they them their this that
these those is am are was were be been being have has had do does did will would
could should can may might must not no yes ok okay yeah yea yep nope yo hi hey lol
lmao haha hehe just like get got go going gonna want need think know say said u ur
n r y idk btw tbh omg pls plz so much really very also there here what when how why
who which with from into out up down over only even still much many more most some
""".split())

URL_RE = re.compile(r"https?://\S+", re.I)
_WORD_RE = re.compile(r"[a-zA-Z']+")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _is_quotable(m: Message) -> bool:
    """A real, human, displayable text message — not media, not a forwarded
    link, not system noise."""
    if m.kind != MessageKind.TEXT:
        return False
    text = (m.text or "").strip()
    if not text or URL_RE.search(text):
        return False
    if nlp.is_noise_message(text):
        return False
    # media-ish placeholders that slipped through
    low = text.lower()
    if any(tag in low for tag in ("<media", "omitted", "this message was deleted",
                                  "missed voice call", "missed video call")):
        return False
    return True


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _emojis(text: str) -> list[str]:
    try:
        return [item["emoji"] for item in emoji_lib.emoji_list(text)]
    except Exception:
        return []


def _first_message_matching(messages: list[Message], phrases: list[str]) -> dict | None:
    for m in messages:
        if not _is_quotable(m):
            continue
        low = m.text.lower()
        if any(p in low for p in phrases):
            return {"sender": m.sender, "text": m.text[:200],
                    "date": m.timestamp.date().isoformat()}
    return None


def _count_matching(messages: list[Message], phrases: list[str]) -> int:
    n = 0
    for m in messages:
        if m.kind != MessageKind.TEXT:
            continue
        low = (m.text or "").lower()
        if any(p in low for p in phrases):
            n += 1
    return n


def _longest_streak(active_dates: set) -> int:
    if not active_dates:
        return 0
    days = sorted(active_dates)
    best = run = 1
    for prev, cur in zip(days, days[1:]):
        if (cur - prev).days == 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _longest_late_night(messages: list[Message]) -> dict:
    """Longest run of consecutive messages that happened during night hours
    within a ~1h gap window."""
    best: list[Message] = []
    cur: list[Message] = []
    for m in messages:
        if m.timestamp.hour in NIGHT_HOURS:
            if cur and (m.timestamp - cur[-1].timestamp) <= timedelta(hours=1):
                cur.append(m)
            else:
                cur = [m]
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    if not best:
        return {}
    return {
        "date": best[0].timestamp.date().isoformat(),
        "messages": len(best),
        "start": best[0].timestamp.strftime("%H:%M"),
        "end": best[-1].timestamp.strftime("%H:%M"),
    }


def _personality(stats: dict) -> dict:
    """Pick a varied, evidence-based chat personality title."""
    night = stats["night_ratio"]
    emoji = stats["emoji_ratio"]
    question = stats["question_ratio"]
    care = stats["care_ratio"]
    laugh = stats["laugh_ratio"]
    avg_len = stats["avg_len"]

    # Ordered rules — first strong signal wins, so different chats get
    # different titles instead of everyone being "The Meme Minister".
    if care >= 0.06:
        return {"title": "The Emotional Support Humans",
                "why": "Check-ins and comfort show up again and again."}
    if night >= 0.35:
        return {"title": "The Late-Night Philosophers",
                "why": "A big share of the talking happened after midnight."}
    if laugh >= 0.18 or emoji >= 0.45:
        return {"title": "The Chaos & Emoji Crew",
                "why": "Lots of laughter, emojis, and quick back-and-forth."}
    if question >= 0.22:
        return {"title": "The Curious Overthinkers",
                "why": "So many questions — this chat loves to dig into things."}
    if avg_len >= 90:
        return {"title": "The Essay Texters",
                "why": "Long, thoughtful messages are this chat's love language."}
    if emoji >= 0.25:
        return {"title": "The Soft Emoji Communicators",
                "why": "Feelings come through in emojis as much as words."}
    return {"title": "The Steady Everyday Talkers",
            "why": "Consistent, easygoing conversation — day in, day out."}


def _soundtrack(stats: dict) -> str:
    if stats["care_ratio"] >= 0.06:
        return "Comfort playlist energy"
    if stats["night_ratio"] >= 0.35:
        return "Late-night honesty"
    if stats["laugh_ratio"] >= 0.18:
        return "Soft chaos"
    if stats["question_ratio"] >= 0.22:
        return "Main-character overthinking"
    return "Easy everyday warmth"


def _clamp(v: float, lo: int = 5, hi: int = 100) -> int:
    return max(lo, min(hi, int(round(v))))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def compute_wrapped(
    messages: list[Message],
    detected_format: str,
    senders: list[str],
    raw_message_count: int = 0,
) -> dict[str, Any]:
    if not messages:
        return {"error": "no messages"}

    senders = senders or sorted({m.sender for m in messages})
    text_messages = [m for m in messages if m.kind == MessageKind.TEXT]
    media_messages = [m for m in messages if m.kind == MessageKind.MEDIA_PLACEHOLDER]
    quotable = [m for m in messages if _is_quotable(m)]
    n = len(messages)

    first_ts = messages[0].timestamp
    last_ts = messages[-1].timestamp
    year = last_ts.year
    active_dates = {m.timestamp.date() for m in messages}
    days_span = max((last_ts.date() - first_ts.date()).days, 1)

    sender_counts = Counter(m.sender for m in messages)

    # ---- distributions ----
    month_counts: Counter = Counter()
    day_counts: Counter = Counter()
    hour_counts: Counter = Counter()
    emoji_counts: Counter = Counter()
    night_msgs = 0
    emoji_msgs = 0
    question_msgs = 0
    laugh_msgs = 0
    care_msgs = 0
    total_len = 0

    for m in messages:
        month_counts[m.timestamp.strftime("%Y-%m")] += 1
        day_counts[m.timestamp.date().isoformat()] += 1
        hour_counts[m.timestamp.hour] += 1
        if m.timestamp.hour in NIGHT_HOURS:
            night_msgs += 1
        if m.kind == MessageKind.TEXT:
            text = m.text or ""
            total_len += len(text)
            es = _emojis(text)
            if es:
                emoji_msgs += 1
                emoji_counts.update(es)
            low = text.lower()
            if text.strip().endswith("?"):
                question_msgs += 1
            if "haha" in low or "lol" in low or "😂" in text or "🤣" in text:
                laugh_msgs += 1
            if any(p in low for p in CARE_PHRASES):
                care_msgs += 1

    text_n = max(len(text_messages), 1)
    ratios = {
        "night_ratio": night_msgs / n,
        "emoji_ratio": emoji_msgs / text_n,
        "question_ratio": question_msgs / text_n,
        "laugh_ratio": laugh_msgs / text_n,
        "care_ratio": care_msgs / text_n,
        "avg_len": total_len / text_n,
    }

    # ---- most active month / day ----
    top_month = month_counts.most_common(1)[0] if month_counts else ("", 0)
    top_day = day_counts.most_common(1)[0] if day_counts else ("", 0)

    def month_label(ym: str) -> str:
        try:
            y, mo = ym.split("-")
            return f"{calendar.month_name[int(mo)]} {y}"
        except Exception:
            return ym

    # ---- emotional timeline (gentle labels) ----
    timeline = []
    if month_counts:
        ordered = sorted(month_counts.items())
        loud = max(ordered, key=lambda kv: kv[1])[0]
        quiet = min(ordered, key=lambda kv: kv[1])[0]
        for ym, c in ordered:
            if ym == loud:
                label = "your loudest month"
            elif ym == quiet:
                label = "quieter, but still present"
            else:
                label = "steady"
            timeline.append({"month": month_label(ym), "count": c, "label": label})

    # ---- words / phrases / greetings / nicknames ----
    word_counter: Counter = Counter()
    bigram_counter: Counter = Counter()
    for m in quotable:
        toks = [t for t in _tokens(m.text) if t not in STOPWORDS and len(t) > 2]
        word_counter.update(toks)
        for a, b in zip(toks, toks[1:]):
            bigram_counter[f"{a} {b}"] += 1
    top_words = word_counter.most_common(12)
    top_phrases = [(p, c) for p, c in bigram_counter.most_common(20) if c >= 3][:8]
    greetings = [(g, _count_matching(messages, [g]))
                 for g in ("good morning", "good night", "lol", "haha", "miss you", "bro")]
    greetings = [(g, c) for g, c in greetings if c > 0]
    try:
        nicknames = nlp.nicknames(messages, senders)[:6]
    except Exception:
        nicknames = []

    # ---- emoji ----
    top_emojis = emoji_counts.most_common(10)
    signature = top_emojis[0][0] if top_emojis else ""
    emoji_share = int(round(100 * emoji_msgs / text_n)) if text_n else 0

    # ---- best moments ----
    first_msg = next(({"sender": m.sender, "text": m.text[:200],
                       "date": m.timestamp.date().isoformat()} for m in quotable), None)
    longest_msg = None
    if quotable:
        lm = max(quotable, key=lambda m: len(m.text))
        longest_msg = {"sender": lm.sender, "text": lm.text[:400],
                       "date": lm.timestamp.date().isoformat(), "length": len(lm.text)}
    sweet = None
    sweet_candidates = [m for m in quotable
                        if any(p in m.text.lower() for p in CARE_PHRASES + LOVE_PHRASES)
                        and len(m.text) <= 120]
    if sweet_candidates:
        s = sweet_candidates[len(sweet_candidates) // 2]
        sweet = {"sender": s.sender, "text": s.text[:160],
                 "date": s.timestamp.date().isoformat()}

    # ---- care moments (a few real examples) ----
    care_examples = []
    for m in quotable:
        if any(p in m.text.lower() for p in CARE_PHRASES):
            care_examples.append({"sender": m.sender, "text": m.text[:160],
                                  "date": m.timestamp.date().isoformat()})
        if len(care_examples) >= 5:
            break

    # ---- milestones (first occurrences) ----
    milestones = {
        "first_miss_you": _first_message_matching(messages, MISS_PHRASES),
        "first_love": _first_message_matching(messages, LOVE_PHRASES),
        "first_plan": _first_message_matching(messages, PLAN_PHRASES),
        "first_apology": _first_message_matching(messages, APOLOGY_PHRASES),
        "first_celebration": _first_message_matching(messages, CELEBRATE_PHRASES),
    }

    # ---- rituals ----
    rituals = {
        "good_mornings": _count_matching(messages, GM_PHRASES),
        "good_nights": _count_matching(messages, GN_PHRASES),
        "night_talks": night_msgs,
        "links_shared": sum(1 for m in text_messages if URL_RE.search(m.text or "")),
        "media_shared": len(media_messages),
    }

    # ---- connection meter ----
    msgs_per_active = n / max(len(active_dates), 1)
    meter = {
        "Consistency": _clamp(100 * len(active_dates) / days_span),
        "Humor": _clamp(100 * (ratios["laugh_ratio"] * 2 + ratios["emoji_ratio"])),
        "Comfort": _clamp(100 * ratios["care_ratio"] * 6),
        "Chaos": _clamp(100 * min(msgs_per_active / 80, 1) * 0.6 + 100 * ratios["emoji_ratio"] * 0.4),
        "Late-night": _clamp(100 * ratios["night_ratio"]),
    }

    # ---- then vs now ----
    half = max(len(text_messages) // 2, 1)
    early, recent = text_messages[:half], text_messages[half:]

    def _seg(seg):
        if not seg:
            return {"avg_len": 0, "emoji_rate": 0}
        ln = sum(len(m.text or "") for m in seg) / len(seg)
        er = sum(1 for m in seg if _emojis(m.text or "")) / len(seg)
        return {"avg_len": int(round(ln)), "emoji_rate": int(round(100 * er))}

    early_s, recent_s = _seg(early), _seg(recent)
    tvn_notes = []
    if recent_s["avg_len"] > early_s["avg_len"] + 5:
        tvn_notes.append("Messages got longer over time.")
    elif recent_s["avg_len"] + 5 < early_s["avg_len"]:
        tvn_notes.append("Messages got shorter and snappier.")
    if recent_s["emoji_rate"] > early_s["emoji_rate"] + 5:
        tvn_notes.append("More emojis crept in as you got comfortable.")

    # ---- quiet periods ----
    gaps = []
    for prev, cur in zip(messages, messages[1:]):
        d = (cur.timestamp - prev.timestamp).days
        if d >= 2:
            gaps.append(d)
    quiet = {
        "count": len(gaps),
        "longest_gap_days": max(gaps) if gaps else 0,
        "note": ("There were quiet days too — but the conversation always found "
                 "its way back." if gaps else "You barely went a day without talking."),
    }

    # ---- personality + soundtrack ----
    personality = _personality(ratios)
    soundtrack = _soundtrack(ratios)

    # ---- memory cards (shareable facts) ----
    cards = []
    if greetings:
        g, c = max(greetings, key=lambda x: x[1])
        cards.append(f"You said “{g}” {c:,} times.")
    if top_day[0]:
        cards.append(f"Your busiest day was {top_day[0]} with {top_day[1]:,} messages.")
    if top_phrases:
        cards.append(f"Your signature phrase was “{top_phrases[0][0]}”.")
    if signature:
        cards.append(f"{emoji_share}% of your messages carried an emoji — led by {signature}.")
    if quiet["longest_gap_days"]:
        cards.append(f"This chat survived a {quiet['longest_gap_days']}-day quiet spell and came back.")
    cards = cards[:5]

    # ---- generated letter (templated, grounded) ----
    name_str = " & ".join(senders[:2]) if senders else "you two"
    busiest = month_label(top_month[0]) if top_month[0] else "one stretch"
    letter = (
        f"This chat was made of {raw_message_count or n:,} small updates between "
        f"{name_str}. {busiest} was the loudest. There were "
        f"{rituals['good_mornings']:,} good mornings and "
        f"{care_msgs:,} moments of checking in on each other. "
        f"{quiet['note']} Mostly, it's proof that someone was there."
    )

    intro = ("A year of late replies, random jokes, comfort, chaos, and memories."
             if days_span > 200 else
             "A stretch of small updates, inside jokes, and staying in touch.")

    return {
        "detected_format": detected_format,
        "senders": senders,
        "year": year,
        "cover": {
            "title": f"Your {year} ChatWrapped",
            "names": name_str,
            "date_range": f"{first_ts.date().isoformat()} → {last_ts.date().isoformat()}",
            "intro": intro,
        },
        # core stats
        "total_messages": raw_message_count or n,
        "conversation_turns": n,
        "text_messages": len(text_messages),
        "media_count": len(media_messages),
        "active_days": len(active_dates),
        "days_span": days_span,
        "first_message_at": first_ts.isoformat(),
        "last_message_at": last_ts.isoformat(),
        "messages_per_sender": dict(sender_counts),
        "most_active_month": {"label": month_label(top_month[0]), "count": top_month[1]},
        "most_active_day": {"date": top_day[0], "count": top_day[1]},
        "longest_streak_days": _longest_streak(active_dates),
        "longest_late_night": _longest_late_night(messages),
        # story sections
        "emotional_timeline": timeline,
        "top_words": top_words,
        "top_phrases": top_phrases,
        "nicknames": nicknames,
        "greetings": greetings,
        "emoji": {"top": top_emojis, "signature": signature,
                  "line": f"You communicated {emoji_share}% through emojis."
                          if signature else "Words did most of the talking here."},
        "personality": personality,
        "best_moments": {
            "first_message": first_msg,
            "longest_message": longest_msg,
            "sweetest": sweet,
            "memorable_day": {"date": top_day[0], "count": top_day[1]} if top_day[0] else None,
        },
        "care_moments": care_examples,
        "milestones": milestones,
        "soundtrack": soundtrack,
        "rituals": rituals,
        "connection_meter": meter,
        "then_vs_now": {"early": early_s, "now": recent_s, "notes": tvn_notes},
        "quiet_periods": quiet,
        "memory_cards": cards,
        "letter": letter,
        "closing": ("Some chats are not just conversations. "
                    "They are proof that someone was there."),
        # teaser strings shown on the locked preview
        "teasers": [
            f"Your chat personality: {personality['title']}.",
            (f"Your busiest month was {month_label(top_month[0])}." if top_month[0] else ""),
            (f"Signature emoji: {signature}." if signature else ""),
            (f"{care_msgs:,} care-and-comfort moments are waiting inside." if care_msgs else ""),
        ],
        # small heatmap kept for the PDF / charts
        "heatmap": {
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
            {"name": "Build your story", "status": "pending", "progress": 0},
            {"name": "Render keepsake PDF", "status": "pending", "progress": 0},
        ]
        jobs.update(job_id, state="parsing", progress=10, message="Reading your conversations…", phases=phases)
        parsed = await asyncio.to_thread(parse_chat, upload_path)
        if parsed.message_count > settings.MAX_MESSAGES:
            raise ValueError(f"Too many messages ({parsed.message_count}). Maximum is {settings.MAX_MESSAGES}.")

        phases[0] = {"name": "Parse export", "status": "done", "progress": 100}
        phases[1] = {"name": "Build your story", "status": "in_progress", "progress": 50}
        jobs.update(job_id, state="generating_wrapped", progress=45, message="Turning messages into your story…", phases=phases)
        wrapped = await asyncio.to_thread(
            compute_wrapped,
            parsed.messages,
            parsed.detected_format,
            parsed.senders,
            parsed.raw_message_count,
        )
        wrapped["parser_warnings"] = parsed.parser_warnings
        # Guarantee JSON-safe before storing on the job / writing to disk.
        wrapped = json.loads(json.dumps(wrapped, default=str))

        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chat_wrapped.json").write_text(
            json.dumps(wrapped, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        phases[1] = {"name": "Build your story", "status": "done", "progress": 100}
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
