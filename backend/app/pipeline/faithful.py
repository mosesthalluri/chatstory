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
# Scene-setting (the only generated text — strictly grounded)
# ---------------------------------------------------------------------------

SETTING_PROMPT = """You are setting the scene for one moment in a real chat, for
an immersive keepsake book. Below are the ACTUAL messages, in order.

{dialogue}

Write 1-2 short, warm sentences that set the scene for this exchange — the time
of day, the mood the words themselves show, what it is about. RULES:
- Use ONLY what these messages show. Do NOT invent events, names, places, or feelings.
- Do NOT quote or repeat the messages; the real lines are printed under your text.
- Third person. No dates or clock times (those are shown separately).
- If the messages are too sparse to say anything grounded, reply with exactly: -
Reply with ONLY the sentence(s), nothing else."""


def _format_dialogue_for_prompt(lines: list[dict]) -> str:
    return "\n".join(f"{ln['sender']}: {ln['text']}" for ln in lines)


async def _scene_setting(lines: list[dict], hour: int) -> str:
    """One grounded setting sentence. Deterministic time-of-day fallback so a
    scene is never blocked and nothing is invented when the model is unsure."""
    fallback = _time_of_day_opener(hour)
    if len(lines) < 2:
        return fallback
    prompt = SETTING_PROMPT.format(dialogue=_format_dialogue_for_prompt(lines[:40]))
    try:
        resp = await asyncio.wait_for(
            llm.complete(
                [
                    {"role": "system", "content": (
                        "You write short, grounded scene-setting for a real chat. "
                        "You never invent facts beyond the messages shown.")},
                    {"role": "user", "content": prompt},
                ],
                model_size="strong", temperature=0.2, max_tokens=120,
            ),
            timeout=max(30, settings.FAITHFUL_SCENE_TIMEOUT_SECONDS),
        )
    except Exception:
        return fallback
    text = (resp or "").strip().strip('"').strip()
    # Models sometimes prefix labels; keep it to the first 2 sentences.
    if not text or text == "-" or text.lower().startswith("(none"):
        return fallback
    text = re.sub(r"\s+", " ", text)
    return text[:400]


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

async def build_manuscript(job_id: str, parsed, *, title: str, subtitle: str) -> dict:
    """Build (and incrementally persist) the full faithful manuscript.

    Returns the manuscript dict. Honors cooperative cancellation: if the job
    state becomes 'cancelling', it stops at the next scene boundary, finalizes
    whatever was generated, and returns it (so the partial preview is kept).
    """
    cleaned = clean_messages(parsed.messages)
    if not cleaned:
        raise ValueError("After removing links and media, no readable messages remained.")

    stats = compute_chat_stats(cleaned)
    sessions = detect_sessions(cleaned)
    total = len(sessions)
    _log(job_id, f"Cleaned chat: {len(cleaned)} messages, {total} scenes to write.")

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

    for i, session in enumerate(sessions, 1):
        if _is_cancelling(job_id):
            _log(job_id, "Cancellation requested — stopping and keeping the partial preview.")
            manuscript["cancelled"] = True
            break

        lines = [
            {"sender": m.sender, "text": m.text, "time": m.timestamp.strftime("%H:%M")}
            for m in session.messages
            if m.kind == MessageKind.TEXT
        ]
        if not lines:
            continue

        month_label = session.start_time.strftime("%B %Y")
        if month_label != current_month:
            current_month = month_label
            chapters.append({"title": month_label, "scenes": []})
            manuscript["chapters"] = chapters

        footnote_n += 1
        setting = await _scene_setting(lines, session.start_time.hour)
        scene = {
            "n": footnote_n,
            "date": _date_label(session.start_time.date()),
            "time_range": _time_range(session.start_time, session.end_time),
            "setting": setting,
            "lines": lines,
            "footnote": f"{_date_label(session.start_time.date())}, {_time_range(session.start_time, session.end_time)}",
        }
        chapters[-1]["scenes"].append(scene)
        manuscript["scene_done"] = i

        # Persist + report every few scenes (and at the end) so the live
        # tracker/preview update without rewriting the file on every single one.
        if i % 5 == 0 or i == total:
            _save_manuscript(job_id, manuscript)
            progress = 15 + int(75 * i / total) if total else 90
            jobs.update(job_id, progress=min(progress, 90),
                        message=f"Writing scene {i} of {total}…")
            _log(job_id, f"Scene {i}/{total} written ({chapters[-1]['title']}).")

    manuscript["complete"] = not manuscript["cancelled"]
    _save_manuscript(job_id, manuscript)
    _log(job_id, "Cancelled." if manuscript["cancelled"] else "All scenes written.")
    return manuscript
