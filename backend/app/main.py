"""
FastAPI application. Defines:
  GET  /                — upload page (mobile-friendly HTML)
  POST /api/upload      — accepts file, starts job, returns job_id
  GET  /job/{job_id}    — status page (polls API)
  GET  /api/status/{id} — JSON status
  GET  /preview/{id}    — preview PDF download
  POST /api/retry/{id}  — retry just the PDF render step (uses checkpoint)
  POST /api/pay/{id}    — payment stub (replace with Stripe/Razorpay)
  GET  /full/{id}       — full PDF download (only if paid)
"""

import asyncio
import sys
from pathlib import Path

# --- Windows asyncio fix (belt-and-suspenders) ---
# Playwright needs subprocess support, which requires ProactorEventLoop on
# Windows. The actual rendering uses sync_playwright in a thread (see
# services/pdf_render.py) which sidesteps this entirely, but we set the
# policy here too so any future async subprocess code Just Works.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .pipeline.orchestrator import run_pipeline, retry_render
from .services import jobs
from .settings import (
    UPLOADS_DIR, OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR, STORAGE_ROOT,
    settings, BACKEND_ROOT,
)


app = FastAPI(title="ChatBook", version="0.1.0")

# Static files (CSS, JS for the frontend UI)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Jinja for the UI pages (separate from the book template)
ui_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


@app.get("/", response_class=HTMLResponse)
async def index():
    template = ui_env.get_template("upload.html")
    return template.render(
        max_size_mb=settings.MAX_UPLOAD_SIZE_MB,
        currency=settings.CURRENCY_SYMBOL,
        price=settings.FULL_BOOK_PRICE,
    )


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_page(job_id: str):
    status = jobs.load(job_id)
    if status is None:
        raise HTTPException(404, "Job not found")
    template = ui_env.get_template("status.html")
    return template.render(
        job_id=job_id,
        currency=settings.CURRENCY_SYMBOL,
        price=settings.FULL_BOOK_PRICE,
    )


@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Size check
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            413,
            f"File too large ({size_mb:.1f}MB). Max is {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    # Save the upload
    job_id = jobs.new_job_id()
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "chat").name
    upload_path = job_upload_dir / safe_name
    upload_path.write_bytes(contents)

    # Create job record
    jobs.create(job_id)

    # Kick off pipeline in background
    background_tasks.add_task(run_pipeline, job_id, upload_path)

    return {"job_id": job_id, "status_url": f"/job/{job_id}"}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    return s.to_dict()


def _resolve_output_path(stored_path: str) -> Path | None:
    """Resolve a path stored in the job record to an absolute path on disk.

    Orchestrator stores paths like "output/<job_id>/preview.pdf" relative
    to STORAGE_ROOT. Older builds may have stored absolute paths or paths
    relative to BACKEND_ROOT. We try multiple candidates so jobs from
    different versions still work.
    """
    if not stored_path:
        return None
    p = Path(stored_path)
    if p.is_absolute() and p.exists():
        return p
    # Try common roots in priority order
    for base in (STORAGE_ROOT, BACKEND_ROOT, BACKEND_ROOT.parent):
        candidate = base / p
        if candidate.exists():
            return candidate
    return None


@app.get("/preview/{job_id}")
async def preview_pdf(job_id: str):
    s = jobs.load(job_id)
    if s is None or not s.preview_pdf:
        raise HTTPException(404, "Preview not ready")
    full_path = _resolve_output_path(s.preview_pdf)
    if full_path is None:
        # Last-resort fallback: look in the canonical output location by job_id
        canonical = OUTPUT_DIR / job_id / "preview.pdf"
        if canonical.exists():
            full_path = canonical
        else:
            raise HTTPException(
                404,
                f"Preview file missing (looked for: {s.preview_pdf}). "
                f"Check storage/output/{job_id}/ — if the PDF exists there, "
                f"this is a path bug; report it."
            )
    return FileResponse(full_path, media_type="application/pdf", filename="preview.pdf")


@app.get("/full/{job_id}")
async def full_pdf(job_id: str):
    s = jobs.load(job_id)
    if s is None or not s.full_pdf:
        raise HTTPException(404, "Full PDF not ready")
    if not s.paid:
        raise HTTPException(402, "Payment required")
    full_path = _resolve_output_path(s.full_pdf)
    if full_path is None:
        canonical = OUTPUT_DIR / job_id / "full.pdf"
        if canonical.exists():
            full_path = canonical
        else:
            raise HTTPException(
                404,
                f"Full PDF file missing (looked for: {s.full_pdf}). "
                f"Check storage/output/{job_id}/."
            )
    return FileResponse(full_path, media_type="application/pdf", filename="full_book.pdf")


@app.post("/api/pay/{job_id}")
async def pay_stub(job_id: str):
    """STUB: marks job as paid. Replace with real payment webhook.

    For Stripe: create a checkout session, set the webhook URL to a new
    /api/payment-webhook endpoint that calls jobs.update(paid=True).
    Same idea for Razorpay.
    """
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    if s.state != "done":
        raise HTTPException(400, "Job not finished yet")
    jobs.update(job_id, paid=True)
    return {"ok": True, "download_url": f"/full/{job_id}"}


@app.post("/api/retry/{job_id}")
async def retry(job_id: str, background_tasks: BackgroundTasks):
    """Retry just the PDF render step using the saved checkpoint.

    Use this when a job died in the 'rendering' state — the LLM chapters
    and images are already done and saved to disk, so this re-runs only
    the cheap final step. No API spending.
    """
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")

    # Kick off retry in background, immediately return so the user can
    # watch progress on the status page.
    background_tasks.add_task(retry_render, job_id)
    return {"ok": True, "status_url": f"/job/{job_id}"}


@app.post("/api/dry-run")
async def dry_run(file: UploadFile = File(...)):
    """Parse-only endpoint. Runs the parser + stats, returns a summary —
    NO LLM calls, NO image gen, NO PDF render. Use this to validate that
    your file parses correctly before burning API budget on the full
    pipeline.

    Returns: message count, sender list, date range, top emojis, sample
    of first/last 5 real text messages, and any parser warnings.
    """
    from .parsers import parse_chat
    from .pipeline import stats as stats_mod
    from .models import MessageKind

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB).")

    # Save to a temp location so the parser can read from disk
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=Path(file.filename or "chat").suffix, delete=False
    ) as f:
        f.write(contents)
        tmp_path = Path(f.name)

    try:
        parsed = parse_chat(tmp_path)
        chat_stats = stats_mod.compute_stats(parsed)

        text_msgs = [m for m in parsed.messages if m.kind == MessageKind.TEXT]
        first_5 = [
            {
                "ts": m.timestamp.isoformat(),
                "sender": m.sender,
                "text": m.text[:100],
            } for m in text_msgs[:5]
        ]
        last_5 = [
            {
                "ts": m.timestamp.isoformat(),
                "sender": m.sender,
                "text": m.text[:100],
            } for m in text_msgs[-5:]
        ]

        return {
            "ok": True,
            "detected_format": parsed.detected_format,
            "senders": parsed.senders,
            "total_messages": chat_stats.get("total_messages", 0),
            "text_messages": len(text_msgs),
            "days_span": chat_stats.get("days_span", 0),
            "days_active": chat_stats.get("days_active", 0),
            "first_date": chat_stats.get("first_message_date", ""),
            "last_date": chat_stats.get("last_message_date", ""),
            "most_active_hour": chat_stats.get("most_active_hour", 0),
            "top_emojis": chat_stats.get("top_emojis", [])[:5],
            "messages_per_sender": chat_stats.get("messages_per_sender", {}),
            "sample_first_5": first_5,
            "sample_last_5": last_5,
            "parser_warnings": parsed.parser_warnings,
        }
    finally:
        tmp_path.unlink(missing_ok=True)
