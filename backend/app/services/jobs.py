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


def create(
    job_id: str,
    *,
    user_email: str | None = None,
    product: str | None = None,
) -> JobStatus:
    now = datetime.now()
    status = JobStatus(
        job_id=job_id,
        state="queued",
        progress=0,
        message="Job created, waiting to start",
        created_at=now,
        updated_at=now,
        user_email=user_email,
        product=product,
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
        if data.get("state") == "error":
            data["state"] = "failed"
        for key in ("user_email", "product"):
            data.setdefault(key, None)
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
    normalized_txt: str | None = None,
    normalized_json: str | None = None,
    phases: list[dict] | None = None,
    user_email: str | None = None,
    product: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    price: int | None = None,
) -> JobStatus | None:
    status = load(job_id)
    if status is None:
        return None
    if state == "error":
        state = "failed"
    if state is not None:
        status.state = state
    if progress is not None:
        status.progress = progress
    if message is not None:
        status.message = message
    if error is not None:
        status.error = error
    if preview_pdf is not None:
        status.preview_pdf = preview_pdf
    if full_pdf is not None:
        status.full_pdf = full_pdf
    if paid is not None:
        status.paid = paid
    if stats is not None:
        status.stats = stats
    if normalized_txt is not None:
        status.normalized_txt = normalized_txt
    if normalized_json is not None:
        status.normalized_json = normalized_json
    if phases is not None:
        status.phases = phases
    if user_email is not None:
        status.user_email = user_email
    if product is not None:
        status.product = product
    if date_from is not None:
        status.date_from = date_from
    if date_to is not None:
        status.date_to = date_to
    if price is not None:
        status.price = price
    save(status)
    return status


def list_all(*, user_email: str | None = None, limit: int = 100) -> list[JobStatus]:
    rows: list[JobStatus] = []
    for path in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        status = load(path.stem)
        if status is None:
            continue
        if user_email and status.user_email != user_email.strip().lower():
            continue
        rows.append(status)
        if len(rows) >= limit:
            break
    return rows


def result_url(status: JobStatus) -> str:
    product = status.product or "chatstory"
    if product == "chat-wrapped":
        return f"/wrapped/{status.job_id}"
    if product == "gift-engine":
        return f"/gift-engine/results/{status.job_id}"
    return f"/job/{status.job_id}"
