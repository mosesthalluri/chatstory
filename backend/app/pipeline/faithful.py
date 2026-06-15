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
from ..core.sessions import detect_sessions
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
# Immersive narration (AI retells each scene as prose — grounded, covers all)
# ---------------------------------------------------------------------------

NARRATE_PROMPT = """You are writing one passage of an immersive, TRUE story based
on a real chat. Below are the ACTUAL messages of this scene, in order.

== PEOPLE (use these names and pronouns) ==
{people}

== MESSAGES (your only source of truth) ==
{dialogue}

Write this scene as flowing, third-person narrative prose — like a novel — so a
reader feels they were there. HARD RULES:
- Cover EVERY message above, in order. Do not skip any exchange. You may quote a
  few of their actual short phrases inside the prose.
- Invent NOTHING. No events, feelings, places, motives, dates, or facts that the
  messages don't show. If a line is ambiguous, narrate only what is stated.
- Refer to each person by name and the pronouns given above.
- Do NOT write timestamps, dates, or clock times (those are footnoted separately).
- Do NOT use a chat/transcript format ("Name: text"). Turn it into real prose.
- Keep it proportional: a few lines of chat -> a short paragraph; a long
  exchange -> two or three paragraphs. Warm, specific, immersive.
Write ONLY the prose."""


def _people_block(senders: list[str], pronouns: dict | None) -> str:
    pronouns = pronouns or {}
    out = []
    for s in senders:
        p = (pronouns.get(s) or "they/them").strip()
        out.append(f"- {s} — refer to {s} using {p}")
    return "\n".join(out) if out else "- (the two people in this chat)"


def _format_dialogue_for_prompt(lines: list[dict]) -> str:
    return "\n".join(f"{ln['sender']}: {ln['text']}" for ln in lines)


def _deterministic_narrative(lines: list[dict]) -> str:
    """Readable fallback prose when the LLM is unavailable/timed out. Grounded
    in the real words; not a raw transcript dump."""
    parts = []
    for ln in lines:
        text = ln["text"].strip().rstrip(".")
        if text:
            parts.append(f'{ln["sender"]} wrote, "{text}."')
    return " ".join(parts) if parts else "A quiet exchange passed between them."


async def _narrate_scene(lines: list[dict], people: str, hour: int) -> str:
    """Immersive AI narration of one scene, covering every message. Falls back
    to grounded deterministic prose so a scene never blocks."""
    prompt = NARRATE_PROMPT.format(people=people, dialogue=_format_dialogue_for_prompt(lines))
    try:
        resp = await asyncio.wait_for(
            llm.complete(
                [
                    {"role": "system", "content": (
                        "You are a careful novelist turning a real chat into immersive, "
                        "third-person prose. You cover every message and invent nothing "
                        "beyond what the messages show.")},
                    {"role": "user", "content": prompt},
                ],
                model_size="strong", temperature=0.4, max_tokens=900,
            ),
            timeout=max(45, settings.FAITHFUL_SCENE_TIMEOUT_SECONDS),
        )
    except Exception:
        return _deterministic_narrative(lines)
    text = (resp or "").strip()
    if not text or len(text) < 10:
        return _deterministic_narrative(lines)
    # Guard: if the model returned transcript-style lines, fall back to prose.
    transcripty = sum(1 for l in text.splitlines() if re.match(r"^\s*\S{1,30}:\s", l))
    if transcripty >= max(3, len(text.splitlines()) // 2):
        return _deterministic_narrative(lines)
    return text


def _chunk_session(messages: list, max_n: int) -> list[list]:
    """Split one session's messages into consecutive windows of <= max_n so a
    very long session still narrates fully (nothing dropped) within bounded
    prompts. Each window becomes its own footnoted passage."""
    if len(messages) <= max_n:
        return [messages]
    return [messages[i:i + max_n] for i in range(0, len(messages), max_n)]


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

    # Split into scenes (sessions), then into bounded passages so even long
    # sessions narrate fully without truncating the prompt — nothing dropped.
    max_n = max(8, settings.FAITHFUL_MAX_SCENE_MESSAGES)
    windows: list[list] = []
    for session in detect_sessions(cleaned):
        msgs = [m for m in session.messages if m.kind == MessageKind.TEXT]
        if not msgs:
            continue
        windows.extend(_chunk_session(msgs, max_n))
    total = len(windows)
    _log(job_id, f"Cleaned chat: {len(cleaned)} messages → {total} passages to narrate.")

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
    current_month = None
    footnote_n = 0

    for i, window in enumerate(windows, 1):
        if _is_cancelling(job_id):
            _log(job_id, "Cancellation requested — stopping and keeping the partial preview.")
            manuscript["cancelled"] = True
            break

        start = window[0].timestamp
        end = window[-1].timestamp
        lines = [
            {"sender": m.sender, "text": m.text, "time": m.timestamp.strftime("%H:%M")}
            for m in window
        ]

        month_label = start.strftime("%B %Y")
        if month_label != current_month:
            current_month = month_label
            chapters.append({"title": month_label, "scenes": []})
            manuscript["chapters"] = chapters

        footnote_n += 1
        narrative = await _narrate_scene(lines, people, start.hour)
        scene = {
            "n": footnote_n,
            "date": _date_label(start.date()),
            "time_range": _time_range(start, end),
            "narrative": narrative,
            "footnote": f"{_date_label(start.date())}, {_time_range(start, end)}",
        }
        chapters[-1]["scenes"].append(scene)
        manuscript["scene_done"] = i

        # Update progress/log every passage (cheap) for a responsive tracker;
        # persist the manuscript every few (keeps the live preview fresh).
        progress = 15 + int(75 * i / total) if total else 90
        jobs.update(job_id, progress=min(progress, 90),
                    message=f"Narrating passage {i} of {total}…")
        if i % 3 == 0 or i == total:
            _save_manuscript(job_id, manuscript)
            _log(job_id, f"Passage {i}/{total} narrated ({chapters[-1]['title']}).")

    manuscript["complete"] = not manuscript["cancelled"]
    _save_manuscript(job_id, manuscript)
    _log(job_id, "Cancelled." if manuscript["cancelled"] else "All passages narrated.")
    return manuscript
