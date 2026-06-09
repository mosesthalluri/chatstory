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
APOLOGY_PHRASES = ["sorry", "my bad", "apologize", "apologies", "apology", "forgive me",
                   "didn't mean", "didnt mean"]
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

# WhatsApp / platform system annotations — these are NOT the users' words and
# must never appear in quotes, top phrases, or milestones. ("message edited"
# leaking in as a "signature phrase" is exactly this.)
_SYSTEM_RE = re.compile(
    r"<\s*this message was (?:edited|deleted)\s*>"
    r"|this message was (?:edited|deleted)"
    r"|you deleted this message"
    r"|<\s*media omitted\s*>|\bmedia omitted\b"
    r"|<\s*attached[^>]*>"
    r"|\blive location\b|location shared"
    r"|missed (?:voice|video) call"
    r"|\bnull\b",
    re.I,
)


def _clean_text(text: str) -> str:
    """Strip system annotations and collapse whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", _SYSTEM_RE.sub(" ", text)).strip()


def _compile_phrases(phrases: list[str]) -> "re.Pattern":
    """Word-boundary matcher for a phrase list. Word boundaries stop false
    positives like 'ily' matching inside 'easily' / 'Daily'."""
    parts = sorted({p.strip() for p in phrases if p.strip()}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in parts) + r")\b", re.I)


CARE_RE = _compile_phrases(CARE_PHRASES)
LOVE_RE = _compile_phrases(LOVE_PHRASES)
MISS_RE = _compile_phrases(MISS_PHRASES)
PLAN_RE = _compile_phrases(PLAN_PHRASES)
APOLOGY_RE = _compile_phrases(APOLOGY_PHRASES)
CELEBRATE_RE = _compile_phrases(CELEBRATE_PHRASES)
GM_RE = _compile_phrases(GM_PHRASES)
GN_RE = _compile_phrases(GN_PHRASES)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _is_quotable(m: Message) -> bool:
    """A real, human, displayable text message — not media, not a forwarded
    link, not system noise."""
    if m.kind != MessageKind.TEXT:
        return False
    raw = (m.text or "").strip()
    if not raw or URL_RE.search(raw):
        return False
    text = _clean_text(raw)
    if len(text) < 2 or nlp.is_noise_message(text):
        return False
    return True


def _qtext(m: Message, limit: int = 200) -> str:
    return _clean_text(m.text)[:limit]


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _emojis(text: str) -> list[str]:
    try:
        return [item["emoji"] for item in emoji_lib.emoji_list(text)]
    except Exception:
        return []


def _first_message_matching(messages: list[Message], regex: "re.Pattern") -> dict | None:
    for m in messages:
        if not _is_quotable(m):
            continue
        if regex.search(_clean_text(m.text)):
            return {"sender": m.sender, "text": _qtext(m),
                    "date": m.timestamp.date().isoformat()}
    return None


def _count_matching(messages: list[Message], regex: "re.Pattern") -> int:
    return sum(1 for m in messages
               if m.kind == MessageKind.TEXT and regex.search(_clean_text(m.text or "")))


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
# Story layer — topics, eras/arc, categorized moments, people, archetype
# --------------------------------------------------------------------------

# Life-phase topics. Each maps to trigger words; matched word-boundary so
# "exam" doesn't fire inside "example".
TOPIC_LEXICON: dict[str, list[str]] = {
    "college & exams": ["college", "exam", "exams", "assignment", "assignments",
                        "semester", "class", "classes", "professor", "marks",
                        "study", "studying", "lecture", "syllabus", "viva",
                        "internship", "attendance", "campus", "submission"],
    "faith & church": ["church", "bible", "prayer", "pray", "praying", "jesus",
                       "god", "lord", "sermon", "pastor", "worship", "verse",
                       "faith", "amen", "devotional", "christ", "blessing", "anna"],
    "work & jobs": ["office", "job", "jobs", "work", "salary", "interview",
                    "meeting", "boss", "client", "company", "resume", "hiring",
                    "shift", "career", "intern", "manager"],
    "food & cravings": ["biryani", "coffee", "tea", "chai", "pizza", "cake",
                        "dinner", "lunch", "restaurant", "food", "snack", "burger"],
    "travel & plans": ["trip", "travel", "flight", "train", "goa", "beach",
                       "mountain", "vacation", "hotel", "journey", "outing"],
    "family & home": ["mom", "mum", "mummy", "dad", "papa", "family", "sister",
                      "brother", "akka", "home", "parents", "uncle", "aunty"],
    "health & rest": ["sick", "fever", "hospital", "doctor", "medicine", "tired",
                      "sleep", "health", "rest", "headache"],
    "celebrations": ["birthday", "congrats", "congratulations", "party",
                     "festival", "christmas", "diwali", "celebrate", "wedding"],
    "shows & music": ["game", "movie", "song", "songs", "music", "series",
                      "netflix", "reels", "playlist", "concert"],
    "love & longing": ["miss", "missing", "love", "baby", "jaan", "cute",
                       "heart", "babe", "darling"],
}
_TOPIC_RE = {name: _compile_phrases(words) for name, words in TOPIC_LEXICON.items()}


def _topic_counts(messages: list[Message]) -> Counter:
    """Total keyword references per topic across messages (word-boundary)."""
    counts: Counter = Counter()
    for m in messages:
        if m.kind != MessageKind.TEXT:
            continue
        text = _clean_text(m.text)
        if not text:
            continue
        for name, rx in _TOPIC_RE.items():
            hits = len(rx.findall(text))
            if hits:
                counts[name] += hits
    return counts


def _top_topics(messages: list[Message], k: int = 2) -> list[str]:
    return [t for t, _ in _topic_counts(messages).most_common(k)]


def _eras(messages: list[Message]) -> list[dict]:
    """Split the timeline into up to 5 contiguous eras of roughly equal
    message VOLUME (so busy stretches get their own era), and label each by
    its dominant life-topics. This is the relationship 'arc'."""
    if len(messages) < 8:
        return []
    n = len(messages)
    k = min(5, max(2, n // 400 + 1))
    size = n / k
    eras = []
    for i in range(k):
        seg = messages[int(i * size): int((i + 1) * size) if i < k - 1 else n]
        if not seg:
            continue
        topics = _top_topics(seg, 2)
        night = sum(1 for m in seg if m.timestamp.hour in NIGHT_HOURS) / len(seg)
        vibe = "late-night" if night >= 0.35 else "daytime"
        eras.append({
            "label": f"{seg[0].timestamp.strftime('%b %Y')} – {seg[-1].timestamp.strftime('%b %Y')}",
            "messages": len(seg),
            "top_topics": topics or ["everyday life"],
            "vibe": vibe,
        })
    return eras


def _arc_narrative(eras: list[dict], senders: list[str], quiet: dict) -> str:
    if not eras:
        return ""
    name_str = " and ".join(senders[:2]) if senders else "you two"
    first_t = ", ".join(eras[0]["top_topics"][:2])
    last_t = ", ".join(eras[-1]["top_topics"][:2])
    parts = [f"In the early days ({eras[0]['label']}), {name_str} mostly talked about {first_t}."]
    if len(eras) > 2:
        mid = eras[len(eras) // 2]
        parts.append(f"Around {mid['label']} the conversation leaned into {', '.join(mid['top_topics'][:2])}.")
    if last_t and last_t != first_t:
        parts.append(f"By {eras[-1]['label']}, it had shifted toward {last_t}.")
    else:
        parts.append(f"Through {eras[-1]['label']}, {last_t or 'everyday life'} stayed at the center.")
    if quiet.get("longest_gap_days", 0) >= 7:
        parts.append(f"There was a {quiet['longest_gap_days']}-day silence along the way — and then you picked right back up.")
    return " ".join(parts)


_EMOTION_WORDS = _compile_phrases([
    "miss", "missing", "love", "sorry", "scared", "hurt", "proud", "happy",
    "sad", "cry", "crying", "worried", "grateful", "thank you", "thanks",
])


def _laugh_score(text: str) -> int:
    low = text.lower()
    return low.count("haha") + low.count("lol") + text.count("😂") + text.count("🤣") + text.count("😭")


def _moment(m: Message, label: str, why: str) -> dict:
    return {"sender": m.sender, "text": _qtext(m, 220), "date": m.timestamp.date().isoformat(),
            "label": label, "why": why}


def _highlight_moments(messages: list[Message], day_counts: Counter) -> dict:
    quotable = [m for m in messages if _is_quotable(m) and len(_clean_text(m.text)) >= 8]
    out: dict[str, dict | None] = {
        "most_supportive": None, "funniest": None, "most_wholesome": None,
        "most_emotional": None, "most_chaotic_day": None,
    }
    # supportive: care/encouragement, prefer a bit of length (heartfelt)
    care = [m for m in quotable if CARE_RE.search(_clean_text(m.text)) and len(_clean_text(m.text)) <= 220]
    if care:
        out["most_supportive"] = _moment(max(care, key=lambda m: len(_clean_text(m.text))),
                                         "Most supportive", "the kindest reassurance in the chat")
    # wholesome: care OR love OR gratitude, short and sweet
    whole = [m for m in quotable if (CARE_RE.search(_clean_text(m.text)) or LOVE_RE.search(_clean_text(m.text)))
             and len(_clean_text(m.text)) <= 120]
    if whole:
        out["most_wholesome"] = _moment(whole[len(whole) // 2], "Most wholesome", "soft and warm")
    # funniest: highest laugh score
    funny = max(quotable, key=lambda m: _laugh_score(m.text), default=None)
    if funny and _laugh_score(funny.text) >= 1 and len(_clean_text(funny.text)) <= 220:
        out["funniest"] = _moment(funny, "Funniest", "the one that cracked you up")
    # most emotional: emotion words + emoji, prefer short punchy
    emo = [m for m in quotable if _EMOTION_WORDS.search(_clean_text(m.text)) and len(_clean_text(m.text)) <= 160]
    if emo:
        out["most_emotional"] = _moment(max(emo, key=lambda m: _laugh_score(m.text) + len(_emojis(m.text))),
                                       "Most emotional", "raw and real")
    # most chaotic day: busiest day + a representative line from it
    if day_counts:
        d, c = day_counts.most_common(1)[0]
        rep = next((m for m in quotable if m.timestamp.date().isoformat() == d), None)
        out["most_chaotic_day"] = {
            "date": d, "count": c, "label": "Most chaotic day",
            "why": f"{c:,} messages in a single day",
            "text": _qtext(rep, 160) if rep else "", "sender": rep.sender if rep else "",
        }
    return out


def _people_insights(messages: list[Message], senders: list[str]) -> dict:
    text_msgs = [m for m in messages if m.kind == MessageKind.TEXT]
    per = {s: {"messages": 0, "chars": 0, "emoji_msgs": 0, "questions": 0,
               "night": 0, "apologies": 0, "good_mornings": 0} for s in senders}

    def bucket(s):
        return per.setdefault(s, {"messages": 0, "chars": 0, "emoji_msgs": 0,
                                  "questions": 0, "night": 0, "apologies": 0, "good_mornings": 0})

    for m in text_msgs:
        b = bucket(m.sender)
        t = _clean_text(m.text)
        b["messages"] += 1
        b["chars"] += len(t)
        if _emojis(m.text):
            b["emoji_msgs"] += 1
        if t.endswith("?"):
            b["questions"] += 1
        if m.timestamp.hour in NIGHT_HOURS:
            b["night"] += 1
        if APOLOGY_RE.search(t):
            b["apologies"] += 1
        if GM_RE.search(t):
            b["good_mornings"] += 1

    # conversation initiations (first msg after a 6h+ gap)
    starters: Counter = Counter()
    if messages:
        starters[messages[0].sender] += 1
    for prev, cur in zip(messages, messages[1:]):
        if (cur.timestamp - prev.timestamp).total_seconds() >= 6 * 3600:
            starters[cur.sender] += 1

    def _avg_len(s):
        b = per[s]
        return b["chars"] / b["messages"] if b["messages"] else 0

    def _emoji_rate(s):
        b = per[s]
        return b["emoji_msgs"] / b["messages"] if b["messages"] else 0

    insights = []
    if starters:
        insights.append(f"{starters.most_common(1)[0][0]} starts most conversations.")
    if len(senders) >= 2:
        longer = max(senders, key=_avg_len)
        insights.append(f"{longer} writes the longer messages (~{int(_avg_len(longer))} chars).")
        emo = max(senders, key=_emoji_rate)
        insights.append(f"{emo} uses the most emojis.")
        apo = max(senders, key=lambda s: per[s]["apologies"])
        if per[apo]["apologies"]:
            insights.append(f"{apo} is the first to say sorry most often.")
        gm = max(senders, key=lambda s: per[s]["good_mornings"])
        if per[gm]["good_mornings"]:
            insights.append(f"{gm} sends 'good morning' the most.")
    first_apology = _first_message_matching(messages, APOLOGY_RE)
    return {
        "insights": insights,
        "initiator": starters.most_common(1)[0][0] if starters else None,
        "per_sender": {s: {"messages": per[s]["messages"], "avg_len": int(_avg_len(s)),
                           "emoji_rate": int(round(100 * _emoji_rate(s))),
                           "questions": per[s]["questions"], "night": per[s]["night"]}
                       for s in senders},
        "first_apology_by": first_apology["sender"] if first_apology else None,
    }


def _archetype(messages: list[Message], topics: Counter, ratios: dict, moments: dict) -> dict:
    """A unique, evidence-grounded archetype built from the dominant topic +
    behaviour pattern (not a single generic axis)."""
    top_topic = topics.most_common(1)[0][0] if topics else "everyday life"
    night = ratios["night_ratio"] >= 0.3
    emoji = ratios["emoji_ratio"] >= 0.4
    care = ratios["care_ratio"] >= 0.04
    laugh = ratios["laugh_ratio"] >= 0.15
    question = ratios["question_ratio"] >= 0.2

    behaviour = ("Night Owls" if night else
                 "Emoji Chaos Crew" if emoji else
                 "Comfort Keepers" if care else
                 "Overthinkers" if question else
                 "Steady Regulars")
    topic_word = {
        "faith & church": "Church-and-Gossip",
        "college & exams": "Late-Night Study",
        "work & jobs": "Work-Grind",
        "love & longing": "Miss-You",
        "food & cravings": "Foodie",
        "travel & plans": "Wander",
        "family & home": "Family-Update",
        "shows & music": "Binge-and-Banter",
    }.get(top_topic, "Everyday")

    title = f"The {topic_word} {behaviour}"
    t2 = topics.most_common(2)
    topic_str = " and ".join(t for t, _ in t2) if t2 else "everyday life"
    why_bits = [f"Your conversations keep circling back to {topic_str}"]
    if night:
        why_bits.append("most of it after dark")
    if laugh:
        why_bits.append("with plenty of laughing")
    if care:
        why_bits.append("and a lot of checking in on each other")
    why = ", ".join(why_bits) + "."
    evidence = (moments.get("most_wholesome") or moments.get("most_supportive")
                or moments.get("most_emotional"))
    return {"title": title, "why": why,
            "evidence": {"text": evidence["text"], "sender": evidence["sender"]} if evidence else None}


def _yearly(messages: list[Message]) -> list[dict]:
    by_year: dict[int, list[Message]] = defaultdict(list)
    for m in messages:
        by_year[m.timestamp.year].append(m)
    out = []
    for y in sorted(by_year):
        seg = by_year[y]
        out.append({
            "year": y,
            "messages": len(seg),
            "active_days": len({m.timestamp.date() for m in seg}),
            "top_topic": (_top_topics(seg, 1) or ["everyday life"])[0],
        })
    return out


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
            if CARE_RE.search(_clean_text(text)):
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
        toks = [t for t in _tokens(_clean_text(m.text)) if t not in STOPWORDS and len(t) > 2]
        word_counter.update(toks)
        for a, b in zip(toks, toks[1:]):
            bigram_counter[f"{a} {b}"] += 1
    top_words = word_counter.most_common(12)
    top_phrases = [(p, c) for p, c in bigram_counter.most_common(20) if c >= 3][:8]
    greeting_defs = [
        ("good morning", ["good morning", "gm"]),
        ("good night", ["good night", "gn"]),
        ("lol", ["lol"]), ("haha", ["haha"]),
        ("miss you", MISS_PHRASES), ("bro", ["bro"]),
    ]
    greetings = [(lab, _count_matching(messages, _compile_phrases(variants)))
                 for lab, variants in greeting_defs]
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
    first_msg = next(({"sender": m.sender, "text": _qtext(m),
                       "date": m.timestamp.date().isoformat()} for m in quotable), None)
    longest_msg = None
    if quotable:
        lm = max(quotable, key=lambda m: len(_clean_text(m.text)))
        longest_msg = {"sender": lm.sender, "text": _qtext(lm, 400),
                       "date": lm.timestamp.date().isoformat(),
                       "length": len(_clean_text(lm.text))}
    sweet = None
    sweet_candidates = [m for m in quotable
                        if (CARE_RE.search(_clean_text(m.text)) or LOVE_RE.search(_clean_text(m.text)))
                        and len(_clean_text(m.text)) <= 120]
    if sweet_candidates:
        s = sweet_candidates[len(sweet_candidates) // 2]
        sweet = {"sender": s.sender, "text": _qtext(s, 160),
                 "date": s.timestamp.date().isoformat()}

    # ---- care moments (a few real examples) ----
    care_examples = []
    for m in quotable:
        if CARE_RE.search(_clean_text(m.text)):
            care_examples.append({"sender": m.sender, "text": _qtext(m, 160),
                                  "date": m.timestamp.date().isoformat()})
        if len(care_examples) >= 5:
            break

    # ---- milestones (first occurrences, word-boundary matched) ----
    milestones = {
        "first_miss_you": _first_message_matching(messages, MISS_RE),
        "first_love": _first_message_matching(messages, LOVE_RE),
        "first_plan": _first_message_matching(messages, PLAN_RE),
        "first_apology": _first_message_matching(messages, APOLOGY_RE),
        "first_celebration": _first_message_matching(messages, CELEBRATE_RE),
    }

    # ---- rituals ----
    rituals = {
        "good_mornings": _count_matching(messages, GM_RE),
        "good_nights": _count_matching(messages, GN_RE),
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

    # ---- story layer (arc, moments, people, archetype) ----
    topics = _topic_counts(messages)
    moments = _highlight_moments(messages, day_counts)
    people = _people_insights(messages, senders)
    eras = _eras(messages)
    arc_narrative = _arc_narrative(eras, senders, quiet)
    yearly = _yearly(messages)
    archetype = _archetype(messages, topics, ratios, moments)
    personality = archetype  # richer, topic+behaviour grounded (was generic)
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

    span_years = (last_ts.date() - first_ts.date()).days / 365.0
    if span_years >= 1.5:
        title = f"Your {first_ts.year}–{last_ts.year} ChatWrapped"
        intro = "Years of late replies, random jokes, comfort, chaos, and memories."
    elif days_span > 200:
        title = f"Your {year} ChatWrapped"
        intro = "A year of late replies, random jokes, comfort, chaos, and memories."
    else:
        title = f"Your {year} ChatWrapped"
        intro = "A stretch of small updates, inside jokes, and staying in touch."

    return {
        "detected_format": detected_format,
        "senders": senders,
        "year": year,
        "cover": {
            "title": title,
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
        "archetype": archetype,
        "relationship_arc": {"narrative": arc_narrative, "eras": eras},
        "topics": topics.most_common(8),
        "yearly": yearly,
        "highlight_moments": moments,
        "people": people,
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
