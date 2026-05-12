"""
Cleanup script. Deletes uploaded chat data older than
AUTO_DELETE_AFTER_HOURS. Run via cron / Task Scheduler.

Usage:
    cd backend
    python -m scripts.cleanup        # one-time run
    python -m scripts.cleanup --dry  # show what would be deleted
"""

import shutil
import sys
import time
from pathlib import Path

# Allow running from project root or backend dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.settings import UPLOADS_DIR, settings


def cleanup(dry_run: bool = False) -> None:
    if settings.AUTO_DELETE_AFTER_HOURS <= 0:
        print("AUTO_DELETE_AFTER_HOURS=0, nothing to do.")
        return

    cutoff = time.time() - settings.AUTO_DELETE_AFTER_HOURS * 3600
    deleted = 0
    kept = 0

    for entry in UPLOADS_DIR.iterdir():
        if not entry.is_dir():
            continue
        # Use directory mtime as proxy for upload time
        if entry.stat().st_mtime < cutoff:
            print(f"{'[DRY] ' if dry_run else ''}Deleting {entry}")
            if not dry_run:
                shutil.rmtree(entry)
            deleted += 1
        else:
            kept += 1

    print(f"\nResult: {deleted} dirs deleted, {kept} kept.")


if __name__ == "__main__":
    cleanup(dry_run="--dry" in sys.argv)
