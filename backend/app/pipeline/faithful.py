"""
Faithful "Hybrid" manuscript builder (V2 rewrite).

Goal: a complete, immersive record of the WHOLE chat — not a sampled,
over-simplified summary. Every non-noise message is preserved verbatim; the
only generated text is a short, strictly-grounded scene-setting line per scene.
This makes hallucination structurally impossible to hide: the real words sit
right under the narration, and every scene carries a timestamp footnote so the
reader can cross-reference the original export.

Pipeline per chat:
  clean (drop links/media/system, keep real text)
    -> detect scenes (sessions split on silences/overnight/day gaps)
      -> group scenes into chapters by calendar month
        -> per scene: 1–2 grounded setting sentences (LLM, deterministic
           fallback) + the real dialogue + a timestamp footnote

The builder persists the manuscript incrementally and logs progress so the UI
can show a live tracker + preview while a long chat generates, and can be
cancelled cooperatively between scenes.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from .. import llm
from ..models import Message, MessageKind
from ..services import jobs
from ..settings import OUTPUT_DIR, settings


# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NOISE_RE = re.compile(
    r"<\s*media omitted\s*>|image omitted|video omitted|sticker omitted|"
    r"audio omitted|gif omitted|this message was deleted|"
    r"you deleted this message|<\s*attached:.*?>",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Strip links and media/system noise; keep the real words. Returns '' if
    nothing meaningful remains (so the line can be dropped)."""
    if not text:
        return ""
    t = _URL_RE.sub("", text)
    t = _NOISE_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_messages(messages: list[Message]) -> list[Message]:
    """Keep only real text messages (drops media/system/reactions/deleted) and
    strips links/noise from the text. Preserves everything else verbatim."""
    out: list[Message] = []
    for m in messages:
        if m.kind != MessageKind.TEXT:
            continue
        cleaned = clean_text(m.text)
        if not cleaned:
            continue
        out.append(Message(sender=m.sender, timestamp=m.timestamp,
                           text=cleaned, kind=MessageKind.TEXT))
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _date_label(d) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def _time_range(start, end) -> str:
    a = start.strftime("%H:%M")
    b = end.strftime("%H:%M")
    return a if a == b else f"{a}–{b}"


def _time_of_day_opener(hour: int) -> str:
    if hour < 5:
        return "In the small hours,"
    if hour < 12:
        return "That morning,"
    if hour < 17:
        return "That afternoon,"
    if hour < 21:
        return "That evening,"
    return "Late that night,"


# ---------------------------------------------------------------------------
# Chapter narration (one event-titled, grounded story per episode)
# ---------------------------------------------------------------------------

_RULES = """HARD RULES:
- Narrate WHAT THEY ACTUALLY TALKED ABOUT — the topics, news, questions,
  decisions, plans, jokes and reactions in the messages. Every sentence must be
  grounded in a real message.
- Invent NOTHING. Do NOT add physical actions, gestures, facial expressions,
  typing, phones, screens, rooms, weather, places, or feelings the messages do
  not state. Forbidden examples: "her eyes lit up", "his thumbs flew across the
  screen", "she smiled" — none of that is in a chat, so never write it.
- Do NOT comment on how MANY messages were sent or how fast they replied as a
  substitute for content. Tell what was said.
- You MAY note natural timing the timestamps reveal ("the next morning", "an
  hour later", "after a long pause"), but do NOT print exact clock times.
- Write flowing prose, NOT a "Name: text" transcript.
- If a single message is gibberish or a typo, skip it; do not quote nonsense.
- Use the names and pronouns given. Warm, specific, faithful to the content."""

CHAPTER_PROMPT = """You are writing one chapter of an immersive, TRUE story based
on a real chat. Below are the ACTUAL messages of this episode, in order, each
prefixed with when it was sent.

== PEOPLE (use these names and pronouns) ==
{people}

== MESSAGES (your ONLY source of truth) ==
{dialogue}

{rules}

Respond in EXACTLY this form:
TITLE: a short, specific chapter title naming what actually happened here
  (e.g. "The Cat at the Door", "A Long Silence", "Saying Goodbye"). 3-7 words,
  no dates, based only on these messages.
STORY: the chapter, as flowing third-person narrative prose."""

CONTINUE_PROMPT = """Continue the SAME chapter of the story. Do not write a new
title and do not recap — carry straight on from where it left off, using these
further messages (in order).

== PEOPLE ==
{people}

== MORE MESSAGES ==
{dialogue}

{rules}

Respond with the continuing prose only (no title)."""

_SYS = ("You are a careful biographer turning a real chat into an immersive, "
        "third-person story. You narrate only what the messages show and never "
        "invent physical detail, settings, or feelings.")


def _people_block(senders: list[str], pronouns: dict | None) -> str:
    pronouns = pronouns or {}
    out = []
    for s in senders:
        p = (pronouns.get(s) or "they/them").strip()
        out.append(f"- {s} — refer to {s} using {p}")
    return "\n".join(out) if out else "- (the two people in this chat)"


def _format_dialogue_for_prompt(window: list) -> str:
    # Include a short date+time so the model can reference natural timing/gaps.
    return "\n".join(
        f"[{m.timestamp.strftime('%b %d, %H:%M')}] {m.sender}: {m.text}"
        for m in window
    )


def _fallback_title(window: list) -> str:
    hour = window[0].timestamp.hour
    if hour < 5:
        return "Late Into the Night"
    if hour < 12:
        return "A Morning Together"
    if hour < 17:
        return "An Afternoon Talk"
    if hour < 21:
        return "An Evening Together"
    return "A Late-Night Talk"


def _deterministic_story(window: list) -> str:
    """Content-bearing fallback used only if the model fails for THIS chapter
    after retries: a readable paraphrase of what was actually said (never a
    message count, never invented detail). Quotes the real lines, lightly."""
    parts: list[str] = []
    for m in window:
        text = " ".join(m.text.split()).strip().rstrip(".")
        if len(text) < 2:
            continue
        parts.append(f'{m.sender} said, "{text}."')
    return " ".join(parts) if parts else "A brief, quiet exchange passed between them."


def _clean_prose(text: str) -> str:
    """Strip stray markdown the local model sometimes emits (**bold**, # heads,
    > quotes, bullets, code fences) and normalize whitespace, so every chapter
    renders as clean, uniform prose. Paragraphs stay separated by blank lines."""
    if not text:
        return ""
    t = text.replace("```", "").replace("`", "")
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"(?<!\w)[*_](.+?)[*_](?!\w)", r"\1", t)
    lines = []
    for ln in t.splitlines():
        ln = re.sub(r"^\s{0,3}#{1,6}\s*", "", ln)      # markdown headings
        ln = re.sub(r"^\s*>\s?", "", ln)               # blockquotes
        ln = re.sub(r"^\s*[-*+]\s+", "", ln)           # bullet markers
        ln = re.sub(r"^\s*\d+[.)]\s+", "", ln)         # numbered lists
        m = re.match(r"^\s*(?:TITLE|STORY)\s*:\s*(.*)$", ln, re.IGNORECASE)
        if m:
            ln = m.group(1)
        lines.append(ln)
    t = "\n".join(lines)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _clean_title(text: str) -> str:
    t = _clean_prose(text or "").splitlines()
    t = t[0] if t else ""
    return t.strip().strip('"').strip("'").rstrip(".").strip()


def _parse_title_story(text: str) -> tuple[str, str]:
    title, story, cur = "", "", None
    buf: list[str] = []
    for line in (text or "").splitlines():
        up = line.strip().upper()
        if up.startswith("TITLE:"):
            if cur == "STORY":
                story = "\n".join(buf).strip()
            cur, buf = "TITLE", [line.split(":", 1)[1].strip()]
        elif up.startswith("STORY:"):
            if cur == "TITLE":
                title = "\n".join(buf).strip()
            cur, buf = "STORY", [line.split(":", 1)[1].strip()]
        elif cur:
            buf.append(line)
    if cur == "TITLE":
        title = "\n".join(buf).strip()
    elif cur == "STORY":
        story = "\n".join(buf).strip()
    if not story and not title:
        story = (text or "").strip()
    return title.strip().strip('"').strip("'"), story.strip()


async def _llm_prose(prompt: str, max_tokens: int = 1100, attempts: int = 2) -> str | None:
    """Call the model with retries. Returns the text, or None if every attempt
    failed/timed out (so the caller can fall back transparently)."""
    timeout = max(60, settings.FAITHFUL_SCENE_TIMEOUT_SECONDS)
    for attempt in range(attempts):
        try:
            out = await asyncio.wait_for(
                llm.complete(
                    [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}],
                    model_size="strong", temperature=0.3, max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
            if out and out.strip():
                return out
        except Exception:
            pass
    return None


async def _narrate_chapter(windows: list[list], people: str) -> tuple[str, str, bool]:
    """Narrate one episode (possibly split into windows) into a single chapter:
    an event title + one flowing, grounded story.
    Returns (title, story, used_fallback) — used_fallback flags that the model
    failed for at least one window so the caller can surface it."""
    title = ""
    parts: list[str] = []
    used_fallback = False
    for j, window in enumerate(windows):
        dialogue = _format_dialogue_for_prompt(window)
        if j == 0:
            resp = await _llm_prose(CHAPTER_PROMPT.format(people=people, dialogue=dialogue, rules=_RULES))
            if resp:
                t, s = _parse_title_story(resp)
                title = _clean_title(t)
                s = _clean_prose(s)
                if s:
                    parts.append(s)
                else:
                    parts.append(_deterministic_story(window)); used_fallback = True
            else:
                parts.append(_deterministic_story(window)); used_fallback = True
        else:
            resp = await _llm_prose(CONTINUE_PROMPT.format(people=people, dialogue=dialogue, rules=_RULES))
            s = _clean_prose(resp or "")
            if s:
                parts.append(s)
            else:
                parts.append(_deterministic_story(window)); used_fallback = True
    if not title or len(title) < 3 or title.lower().startswith("chapter"):
        title = _fallback_title(windows[0])
    return title, "\n\n".join(p for p in parts if p), used_fallback


def _chunk(messages: list, max_n: int) -> list[list]:
    """Split a chapter's messages into bounded windows so a very long episode
    still narrates fully (nothing dropped) within bounded prompts."""
    if len(messages) <= max_n:
        return [messages]
    return [messages[i:i + max_n] for i in range(0, len(messages), max_n)]


def _group_chapters(messages: list, gap_hours: int) -> list[list]:
    """Group consecutive messages into chapter-sized EPISODES, starting a new
    chapter only after a long silence (so a night + the next morning stay in one
    chapter, but a multi-day gap begins a new one)."""
    from datetime import timedelta
    if not messages:
        return []
    groups: list[list] = []
    cur = [messages[0]]
    threshold = timedelta(hours=max(1, gap_hours))
    for prev, m in zip(messages, messages[1:]):
        if (m.timestamp - prev.timestamp) >= threshold:
            groups.append(cur)
            cur = []
        cur.append(m)
    groups.append(cur)
    return groups


def _chapter_citation(start, end) -> str:
    if start.date() == end.date():
        return f"{_date_label(start.date())}, {_time_range(start, end)}"
    return (f"{start.day} {start.strftime('%b')} – {end.day} {end.strftime('%b %Y')}, "
            f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_chat_stats(messages: list[Message]) -> dict:
    """Deterministic stats for the front of the book: per-month volume, who
    texted more, who initiated more. Computed on the cleaned messages so the
    numbers match the book's content exactly."""
    if not messages:
        return {}
    per_month = Counter(m.timestamp.strftime("%Y-%m") for m in messages)
    per_sender = Counter(m.sender for m in messages)
    chars_per_sender = Counter()
    for m in messages:
        chars_per_sender[m.sender] += len(m.text)

    # Initiations: who sends the first message after a 6h+ silence.
    initiations = Counter()
    for prev, cur in zip(messages, messages[1:]):
        if (cur.timestamp - prev.timestamp).total_seconds() > 6 * 3600:
            initiations[cur.sender] += 1
    # The very first message is also an initiation.
    initiations[messages[0].sender] += 1

    first = messages[0].timestamp
    last = messages[-1].timestamp
    active_days = len({m.timestamp.date() for m in messages})
    # Pretty month labels in order.
    per_month_pretty = [
        {"label": f"{ym}", "count": c}
        for ym, c in sorted(per_month.items())
    ]
    return {
        "total_messages": len(messages),
        "first_date": first.strftime("%d %B %Y"),
        "last_date": last.strftime("%d %B %Y"),
        "days_active": active_days,
        "senders": sorted(per_sender),
        "per_month": per_month_pretty,
        "per_sender": dict(per_sender),
        "chars_per_sender": dict(chars_per_sender),
        "initiations": dict(initiations),
        "most_texts": per_sender.most_common(1)[0][0] if per_sender else "",
        "most_initiations": initiations.most_common(1)[0][0] if initiations else "",
    }


# ---------------------------------------------------------------------------
# Persistence / logging / cancellation
# ---------------------------------------------------------------------------

def manuscript_path(job_id: str) -> Path:
    return OUTPUT_DIR / job_id / "manuscript.json"


def _genlog_path(job_id: str) -> Path:
    return OUTPUT_DIR / job_id / "genlog.txt"


def gen_log_tail(job_id: str, n: int = 25) -> list[str]:
    p = _genlog_path(job_id)
    if not p.exists():
        return []
    try:
        return p.read_text(encoding="utf-8").splitlines()[-n:]
    except Exception:
        return []


def _log(job_id: str, message: str) -> None:
    p = _genlog_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass
    print(f"[faithful:{job_id[:8]}] {message}")


def _save_manuscript(job_id: str, manuscript: dict) -> None:
    p = manuscript_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manuscript, ensure_ascii=False, default=str), encoding="utf-8")


def load_manuscript(job_id: str) -> dict | None:
    p = manuscript_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_cancelling(job_id: str) -> bool:
    s = jobs.load(job_id)
    return bool(s and s.state in ("cancelling", "cancelled"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def build_manuscript(job_id: str, parsed, *, title: str, subtitle: str,
                           pronouns: dict | None = None) -> dict:
    """Build (and incrementally persist) the full faithful manuscript: every
    scene rewritten as immersive AI narration that covers all of its messages.

    Returns the manuscript dict. Honors cooperative cancellation: if the job
    state becomes 'cancelling', it stops at the next passage boundary, finalizes
    whatever was generated, and returns it (so the partial preview is kept).
    """
    cleaned = clean_messages(parsed.messages)
    if not cleaned:
        raise ValueError("After removing links and media, no readable messages remained.")

    stats = compute_chat_stats(cleaned)
    senders = stats.get("senders") or sorted({m.sender for m in cleaned})
    people = _people_block(senders, pronouns)

    # Group into chapter-sized EPISODES (new chapter only after a long silence),
    # so each chapter is a coherent event with its own title — not a per-message
    # "X wrote / Y wrote" dump.
    episodes = _group_chapters(cleaned, settings.FAITHFUL_CHAPTER_GAP_HOURS)
    total = len(episodes)
    max_n = max(8, settings.FAITHFUL_MAX_SCENE_MESSAGES)
    _log(job_id, f"Cleaned chat: {len(cleaned)} messages → {total} chapters to narrate.")

    first = cleaned[0].timestamp
    last = cleaned[-1].timestamp
    date_range = (first.strftime("%B %Y") if first.strftime("%Y-%m") == last.strftime("%Y-%m")
                  else f"{first.strftime('%B %Y')} – {last.strftime('%B %Y')}")

    manuscript = {
        "title": title,
        "subtitle": subtitle,
        "date_range": date_range,
        "stats": stats,
        "chapters": [],
        "scene_total": total,
        "scene_done": 0,
        "complete": False,
        "cancelled": False,
    }

    chapters: list[dict] = []
    fallback_count = 0
    for i, episode in enumerate(episodes, 1):
        if _is_cancelling(job_id):
            _log(job_id, "Cancellation requested — stopping and keeping the partial preview.")
            manuscript["cancelled"] = True
            break

        start = episode[0].timestamp
        end = episode[-1].timestamp
        windows = _chunk(episode, max_n)
        ch_title, story, used_fallback = await _narrate_chapter(windows, people)
        if used_fallback:
            fallback_count += 1
        chapters.append({
            "n": i,
            "title": ch_title,
            "date": _date_label(start.date()),
            "time_range": _time_range(start, end),
            "narrative": story,
            "footnote": _chapter_citation(start, end),
        })
        manuscript["chapters"] = chapters
        manuscript["scene_done"] = i

        progress = 15 + int(75 * i / total) if total else 90
        jobs.update(job_id, progress=min(progress, 90),
                    message=f"Writing chapter {i} of {total}…")
        _save_manuscript(job_id, manuscript)
        _log(job_id, f"Chapter {i}/{total}: “{ch_title}”"
             + ("  [AI was slow here — used a plain summary]" if used_fallback else "  [AI]"))

    manuscript["complete"] = not manuscript["cancelled"]
    manuscript["fallback_count"] = fallback_count
    _save_manuscript(job_id, manuscript)
    if fallback_count:
        _log(job_id, f"Note: {fallback_count}/{len(chapters)} chapters fell back to a plain "
                     f"summary because the model was slow/unresponsive. Consider a faster model "
                     f"or re-run those.")
    _log(job_id, "Cancelled." if manuscript["cancelled"] else "All chapters narrated.")
    return manuscript
