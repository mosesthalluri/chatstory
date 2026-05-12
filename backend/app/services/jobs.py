"""
Job manager. Persists job state to disk as JSON.

For v0.1 we use one JSON file per job — no database. This works fine up
to ~1000 books/day. Beyond that, switch to SQLite or Postgres.
"""

import json
import secrets
from datetime import datetime
from pathlib import Path

from ..models import JobStatus
from ..settings import JOBS_DIR


def new_job_id() -> str:
    """Unguessable job ID. Doubles as the download URL token."""
    return secrets.token_urlsafe(16)


def _job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def create(job_id: str) -> JobStatus:
    now = datetime.now()
    status = JobStatus(
        job_id=job_id,
        state="queued",
        progress=0,
        message="Job created, waiting to start",
        created_at=now,
        updated_at=now,
    )
    save(status)
    return status


def load(job_id: str) -> JobStatus | None:
    path = _job_file(job_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return JobStatus(**data)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def save(status: JobStatus) -> None:
    status.updated_at = datetime.now()
    _job_file(status.job_id).write_text(json.dumps(status.to_dict(), indent=2))


def update(
    job_id: str,
    *,
    state: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    error: str | None = None,
    preview_pdf: str | None = None,
    full_pdf: str | None = None,
    paid: bool | None = None,
    stats: dict | None = None,
) -> JobStatus | None:
    status = load(job_id)
    if status is None:
        return None
    if state is not None: status.state = state
    if progress is not None: status.progress = progress
    if message is not None: status.message = message
    if error is not None: status.error = error
    if preview_pdf is not None: status.preview_pdf = preview_pdf
    if full_pdf is not None: status.full_pdf = full_pdf
    if paid is not None: status.paid = paid
    if stats is not None: status.stats = stats
    save(status)
    return status
