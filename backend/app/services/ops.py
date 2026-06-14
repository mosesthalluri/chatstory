"""
Operational visibility for the admin panel (V2): storage/disk monitoring with
alerts, plus a small in-memory activity log.

Storage is computed on demand (admin views are infrequent). The activity log
is a bounded ring buffer in memory — it survives for the life of the process
and is enough to answer "what just happened / what failed" without standing up
a logging stack. A logging.Handler is attached so ordinary `logging` calls are
captured too.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import deque
from datetime import datetime

from ..settings import (
    STORAGE_ROOT, UPLOADS_DIR, OUTPUT_DIR, JOBS_DIR, settings,
)

# Recent activity, newest last. Bounded so memory can't grow unbounded.
_EVENTS: deque[dict] = deque(maxlen=500)

# Disk-usage alert thresholds (percent of the storage volume used).
WARN_PERCENT = 80
CRITICAL_PERCENT = 90
LOW_FREE_GB = 1.0


def record(message: str, level: str = "info") -> None:
    """Append an activity-log entry (shown on the admin panel)."""
    _EVENTS.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "message": str(message)[:500],
    })


def recent(n: int = 100) -> list[dict]:
    """Most recent activity, newest first."""
    return list(_EVENTS)[-n:][::-1]


class _RingHandler(logging.Handler):
    def emit(self, rec: logging.LogRecord) -> None:
        try:
            record(rec.getMessage(), rec.levelname.lower())
        except Exception:
            pass


def install_log_capture() -> None:
    """Attach the ring-buffer handler to the root logger (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h, _RingHandler) for h in root.handlers):
        return
    handler = _RingHandler()
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _dir_size(path) -> tuple[int, int]:
    """Return (total_bytes, file_count) under a directory, robust to races."""
    total = 0
    count = 0
    if not path or not os.path.isdir(path):
        return 0, 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
                count += 1
            except OSError:
                pass
    return total, count


def storage_report() -> dict:
    """Per-directory sizes, overall disk usage, and any alerts."""
    dirs = {
        "uploads": UPLOADS_DIR,
        "generated_output": OUTPUT_DIR,
        "job_metadata": JOBS_DIR,
        "payment_screenshots": STORAGE_ROOT / "payment_screenshots",
    }
    breakdown = []
    for label, path in dirs.items():
        size, count = _dir_size(path)
        breakdown.append({
            "label": label,
            "path": str(path),
            "bytes": size,
            "human": _human(size),
            "files": count,
        })

    try:
        usage = shutil.disk_usage(str(STORAGE_ROOT))
        total, used, free = usage.total, usage.used, usage.free
    except OSError:
        total = used = free = 0
    percent_used = round(used / total * 100, 1) if total else 0.0
    free_gb = free / (1024 ** 3) if free else 0.0

    alerts: list[str] = []
    if percent_used >= CRITICAL_PERCENT:
        alerts.append(f"Disk critically full: {percent_used}% used on the storage volume.")
    elif percent_used >= WARN_PERCENT:
        alerts.append(f"Disk getting full: {percent_used}% used on the storage volume.")
    if total and free_gb < LOW_FREE_GB:
        alerts.append(f"Low free space: only {_human(free)} left.")
    if settings.AUTO_DELETE_AFTER_HOURS <= 0:
        alerts.append("Upload retention is OFF (AUTO_DELETE_AFTER_HOURS=0) — uploads are never purged.")

    return {
        "breakdown": breakdown,
        "disk_total": _human(total),
        "disk_used": _human(used),
        "disk_free": _human(free),
        "percent_used": percent_used,
        "alerts": alerts,
        "retention_hours": settings.AUTO_DELETE_AFTER_HOURS,
    }
