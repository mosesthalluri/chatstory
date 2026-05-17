"""Export unlock checks and download path helpers."""

from pathlib import Path

from ..settings import OUTPUT_DIR
from . import jobs, payments


def is_unlocked(job_id: str) -> bool:
    if payments.is_unlocked(job_id):
        return True
    status = jobs.load(job_id)
    return bool(status and status.paid)


def download_links(job_id: str, product: str | None = None) -> dict[str, str]:
    product = product or (jobs.load(job_id).product if jobs.load(job_id) else None) or "chatstory"
    if product == "chat-wrapped":
        return {
            "view": f"/wrapped/{job_id}",
            "pdf": f"/download/chat-wrapped/{job_id}/pdf",
            "unlock": f"/unlock/chat-wrapped/{job_id}",
        }
    if product == "gift-engine":
        return {
            "view": f"/gift-engine/results/{job_id}",
            "pdf": f"/download/gift-engine/{job_id}/pdf",
            "json": f"/download/gift-engine/{job_id}/json",
            "unlock": f"/unlock/gift-engine/{job_id}",
        }
    return {
        "view": f"/job/{job_id}",
        "preview_pdf": f"/download/chatstory/{job_id}/preview",
        "full_pdf": f"/download/chatstory/{job_id}/full",
        "unlock": f"/unlock/chat-wrapped/{job_id}",
    }


def resolve_pdf_path(job_id: str, field: str = "preview_pdf") -> Path | None:
    status = jobs.load(job_id)
    if status is None:
        return None
    stored = getattr(status, field, None) or status.preview_pdf
    if not stored:
        return None
    p = Path(stored)
    if p.is_absolute() and p.exists():
        return p
    for base in (OUTPUT_DIR.parent, OUTPUT_DIR):
        candidate = base / p
        if candidate.exists():
            return candidate
    canonical = OUTPUT_DIR / job_id / Path(stored).name
    if canonical.exists():
        return canonical
    named = {
        "chat_wrapped.pdf": OUTPUT_DIR / job_id / "chat_wrapped.pdf",
        "gift_engine.pdf": OUTPUT_DIR / job_id / "gift_engine.pdf",
        "preview.pdf": OUTPUT_DIR / job_id / "preview.pdf",
        "full.pdf": OUTPUT_DIR / job_id / "full.pdf",
    }
    return named.get(Path(stored).name)
