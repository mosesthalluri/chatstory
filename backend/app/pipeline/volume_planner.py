"""
Volume planning (V2). A long chat should never be forced into one rushed
PDF — it becomes a *collection* of volumes, each covering a meaningful era.

This is a deterministic, no-LLM stage. It takes the chapter chunks the
pipeline already produced (each is a (start_date, end_date, messages) tuple,
in chronological order) and groups CONSECUTIVE chapters into volumes.

Splitting is by *eras*, not the calendar: we cut at the longest silences
between chapters (the natural seams of a relationship) while keeping each
volume close to a target number of chapters. Short chats stay a single
volume so we never over-fragment a small story.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

from ..settings import settings


class Volume(NamedTuple):
    index: int                 # 1-based
    roman: str                 # "I", "II", ...
    name: str                  # "Volume I"
    era: str                   # soft, warm era name (creative, not asserted fact)
    date_range: str            # grounded, from real message dates
    start: date
    end: date
    chapter_indices: list[int]  # 1-based chapter indices in this volume

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "roman": self.roman,
            "name": self.name,
            "era": self.era,
            "date_range": self.date_range,
            "chapter_indices": list(self.chapter_indices),
            "chapter_count": len(self.chapter_indices),
        }


_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

# Soft, broadly-true era names. These are deliberately gentle and
# non-specific — the product Terms make clear outputs are a creative
# keepsake, and the grounded date range always sits beside them.
_ERA_NAMES = {
    1: ["Our Story"],
    2: ["The Early Days", "Closer Than Ever"],
    3: ["The Early Days", "Finding Our Rhythm", "Closer Than Ever"],
    4: ["The Early Days", "Finding Our Rhythm", "Through It All", "Closer Than Ever"],
    5: ["The Early Days", "Getting Closer", "Finding Our Rhythm", "Through It All", "Closer Than Ever"],
}


def _roman(i: int) -> str:
    return _ROMAN[i - 1] if 1 <= i <= len(_ROMAN) else str(i)


def _era_name(idx: int, total: int) -> str:
    names = _ERA_NAMES.get(total)
    if names and 1 <= idx <= len(names):
        return names[idx - 1]
    # For larger collections we lead with the date range instead of a label.
    return ""


def _fmt_range(start: date, end: date) -> str:
    if start.year == end.year:
        if start.month == end.month:
            return start.strftime("%B %Y")
        return f"{start.strftime('%b')} – {end.strftime('%b %Y')}"
    return f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}"


def _target_volume_count(n_chapters: int) -> int:
    """How many volumes for this many chapters (clamped to settings)."""
    if n_chapters < max(1, settings.VOLUME_MIN_TO_SPLIT):
        return 1
    target = round(n_chapters / max(1, settings.VOLUME_CHAPTERS_TARGET))
    return max(1, min(target, max(1, settings.VOLUME_MAX), n_chapters))


def _split_points(chunks: list[tuple], n_volumes: int) -> list[int]:
    """Pick the (n_volumes - 1) chapter boundaries with the largest silences.

    `chunks[i]` is (start_date, end_date, messages). The gap before chapter
    i+1 is chunks[i+1].start - chunks[i].end. We cut at the biggest gaps so
    volumes break on real lulls in the conversation, then return the sorted
    0-based indices that begin a new volume.
    """
    if n_volumes <= 1:
        return []
    gaps = []
    for i in range(len(chunks) - 1):
        end_i = chunks[i][1]
        start_next = chunks[i + 1][0]
        gap_days = (start_next - end_i).days
        gaps.append((gap_days, i + 1))  # i+1 = chapter index (0-based) that starts a new volume
    # Largest gaps first; tie-break on earlier boundary for determinism.
    gaps.sort(key=lambda g: (-g[0], g[1]))
    cut_at = sorted(idx for _, idx in gaps[: n_volumes - 1])
    return cut_at


def plan_volumes(chunks: list[tuple]) -> list[Volume]:
    """Group chronological chapter chunks into era-based volumes.

    `chunks` is the list the pipeline already built — each item is
    (start_date, end_date, messages). Returns a list of Volume records whose
    chapter_indices are 1-based and contiguous, covering every chapter.
    """
    n = len(chunks)
    if n == 0:
        return []

    n_volumes = _target_volume_count(n)
    cut_at = _split_points(chunks, n_volumes)

    # Build contiguous groups of 0-based chapter positions.
    groups: list[list[int]] = []
    current: list[int] = []
    for pos in range(n):
        if pos in cut_at and current:
            groups.append(current)
            current = []
        current.append(pos)
    if current:
        groups.append(current)

    total = len(groups)
    volumes: list[Volume] = []
    for vi, group in enumerate(groups, 1):
        start = chunks[group[0]][0]
        end = chunks[group[-1]][1]
        volumes.append(Volume(
            index=vi,
            roman=_roman(vi),
            name=f"Volume {_roman(vi)}",
            era=_era_name(vi, total),
            date_range=_fmt_range(start, end),
            start=start,
            end=end,
            chapter_indices=[p + 1 for p in group],  # 1-based
        ))
    return volumes


def estimate_volume_count(message_count: int) -> int:
    """Rough volume estimate from message volume alone — used to show an
    *estimated* price before generation (the final price is set from the
    real volume count once the book is planned). Heuristic: scale with the
    same thresholds used for chapter counts."""
    if message_count <= 0:
        return 1
    # ~ one volume per "large" tier worth of messages, gently.
    per_volume = max(1, settings.MSG_LARGE_MIN // 2)  # e.g. 5000 messages
    est = 1 + message_count // per_volume
    return max(1, min(est, max(1, settings.VOLUME_MAX)))
