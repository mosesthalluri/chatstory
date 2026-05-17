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

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Form, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .pipeline.orchestrator import run_pipeline, retry_render
from .services import auth, jobs, payments
from .services.chat_wrapped import run_chat_wrapped_pipeline
from .services.gift_engine import run_gift_engine_pipeline
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


PRODUCTS = {
    "chat-wrapped": {
        "title": "Chat Wrapped",
        "kicker": "Analytics",
        "description": "Upload a chat export to build a private wrapped dashboard and branded PDF.",
        "api_upload": "/api/chat-wrapped/upload",
        "api_status": "/api/chat-wrapped/status",
        "result_path": "/wrapped",
    },
    "gift-engine": {
        "title": "Meaningful Gift Engine",
        "kicker": "Recommendations",
        "description": "Find gift ideas from hobbies, stress, food, music, travel, routines, and support patterns.",
        "api_upload": "/api/gift-engine/upload",
        "api_status": "/api/gift-engine/status",
        "result_path": "/gift-engine/results",
    },
}


def _current_user(request: Request) -> dict | None:
    return auth.read_token(request.cookies.get("chatstory_session"))


def _is_admin(request: Request) -> bool:
    user = _current_user(request)
    return bool(user and user.get("role") == "admin")


def _require_admin(request: Request) -> dict:
    user = _current_user(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return user


def _has_unlock(request: Request, job_id: str) -> bool:
    return _is_admin(request) or payments.is_unlocked(job_id)


def _all_jobs() -> list[dict]:
    out = []
    for path in sorted((STORAGE_ROOT / "jobs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        status = jobs.load(path.stem)
        if status:
            out.append(status.to_dict())
    return out


def _wrapped_preview(data: dict, unlocked: bool) -> dict:
    if unlocked:
        return {"locked": False, "data": data}
    return {
        "locked": True,
        "preview": {
            "persona": data.get("persona", {}),
            "total_messages": data.get("total_messages", 0),
            "active_days": data.get("active_days", 0),
            "teasers": data.get("teasers", []),
            "payment_status": "locked",
        },
    }


def _gift_preview(data: dict, unlocked: bool) -> dict:
    if unlocked:
        return {"locked": False, "data": data}
    signals = {
        k: {"score": v.get("score", 0), "top_keywords": v.get("top_keywords", [])[:3]}
        for k, v in data.get("signals", {}).items()
        if v.get("score", 0) > 0
    }
    return {
        "locked": True,
        "preview": {
            "signals": signals,
            "teasers": data.get("teasers", []),
            "locked_cards": sum(len(v) for v in data.get("suggestions", {}).values()),
        },
    }


def _save_upload(contents: bytes, filename: str | None) -> tuple[str, Path]:
    job_id = jobs.new_job_id()
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_upload_dir / Path(filename or "chat").name
    upload_path.write_bytes(contents)
    jobs.create(job_id)
    return job_id, upload_path


async def _read_upload(file: UploadFile) -> bytes:
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max is {settings.MAX_UPLOAD_SIZE_MB}MB.")
    return contents


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    template = ui_env.get_template("upload.html")
    return template.render(
        max_size_mb=settings.MAX_UPLOAD_SIZE_MB,
        currency=settings.CURRENCY_SYMBOL,
        single_price=settings.SINGLE_EXPORT_PRICE,
        combined_price=settings.COMBINED_EXPORT_PRICE,
        user=_current_user(request),
    )


@app.get("/signup", response_class=HTMLResponse)
async def signup_page():
    return ui_env.get_template("auth.html").render(mode="signup", error="")


@app.post("/signup")
async def signup(response: Response, email: str = Form(...), password: str = Form(...)):
    try:
        user = auth.create_user(email, password)
    except ValueError as exc:
        return ui_env.get_template("auth.html").render(mode="signup", error=str(exc))
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("chatstory_session", auth.make_token(user), httponly=True, samesite="lax")
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return ui_env.get_template("auth.html").render(mode="login", error="")


@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    user = auth.authenticate(email, password)
    if not user:
        return ui_env.get_template("auth.html").render(mode="login", error="Invalid email or password")
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("chatstory_session", auth.make_token(user), httponly=True, samesite="lax")
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("chatstory_session")
    return response


@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password():
    return ui_env.get_template("forgot_password.html").render()


@app.get("/chat-wrapped", response_class=HTMLResponse)
async def chat_wrapped_page(request: Request):
    template = ui_env.get_template("product_upload.html")
    return template.render(product=PRODUCTS["chat-wrapped"], max_size_mb=settings.MAX_UPLOAD_SIZE_MB, user=_current_user(request))


@app.get("/gift-engine", response_class=HTMLResponse)
async def gift_engine_page(request: Request):
    template = ui_env.get_template("product_upload.html")
    return template.render(product=PRODUCTS["gift-engine"], max_size_mb=settings.MAX_UPLOAD_SIZE_MB, user=_current_user(request))


@app.get("/chatstory-coming-soon", response_class=HTMLResponse)
async def chatstory_coming_soon(request: Request):
    if _is_admin(request):
        return ui_env.get_template("chatstory_beta.html").render(user=_current_user(request))
    template = ui_env.get_template("coming_soon.html")
    return template.render()


@app.get("/processing/{product_slug}/{job_id}", response_class=HTMLResponse)
async def product_processing(product_slug: str, job_id: str):
    product = PRODUCTS.get(product_slug)
    if product is None or jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    template = ui_env.get_template("processing.html")
    return template.render(product=product, product_slug=product_slug, job_id=job_id)


@app.get("/wrapped/{job_id}", response_class=HTMLResponse)
async def wrapped_dashboard(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    template = ui_env.get_template("wrapped_result.html")
    return template.render(job_id=job_id, unlocked=_has_unlock(request, job_id))


@app.get("/gift-engine/results/{job_id}", response_class=HTMLResponse)
async def gift_dashboard(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    template = ui_env.get_template("gift_result.html")
    return template.render(job_id=job_id, unlocked=_has_unlock(request, job_id))


@app.get("/share/wrapped/{job_id}", response_class=HTMLResponse)
async def wrapped_share_card(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None or not s.stats:
        raise HTTPException(404, "Result not found")
    return ui_env.get_template("share_card.html").render(kind="wrapped", payload=_wrapped_preview(s.stats, _has_unlock(request, job_id)), job_id=job_id)


@app.get("/share/gift-engine/{job_id}", response_class=HTMLResponse)
async def gift_share_card(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None or not s.stats:
        raise HTTPException(404, "Result not found")
    return ui_env.get_template("share_card.html").render(kind="gift", payload=_gift_preview(s.stats, _has_unlock(request, job_id)), job_id=job_id)


@app.get("/unlock/{product_slug}/{job_id}", response_class=HTMLResponse)
async def unlock_page(product_slug: str, job_id: str):
    product = PRODUCTS.get(product_slug)
    if product is None or jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    return ui_env.get_template("unlock.html").render(
        product=product,
        product_slug=product_slug,
        job_id=job_id,
        payment=payments.for_job(job_id),
        single_price=settings.SINGLE_EXPORT_PRICE,
        combined_price=settings.COMBINED_EXPORT_PRICE,
        paytm_upi_id=settings.PAYTM_UPI_ID,
        paytm_qr_image=settings.PAYTM_QR_IMAGE,
    )


@app.post("/unlock/{product_slug}/{job_id}", response_class=HTMLResponse)
async def create_unlock(product_slug: str, job_id: str, email: str = Form(...), export_type: str = Form("single")):
    product = PRODUCTS.get(product_slug)
    if product is None or jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    payment = payments.create_intent(job_id, product_slug, email, export_type)
    return ui_env.get_template("unlock.html").render(
        product=product,
        product_slug=product_slug,
        job_id=job_id,
        payment=payment,
        single_price=settings.SINGLE_EXPORT_PRICE,
        combined_price=settings.COMBINED_EXPORT_PRICE,
        paytm_upi_id=settings.PAYTM_UPI_ID,
        paytm_qr_image=settings.PAYTM_QR_IMAGE,
    )


@app.post("/api/payments/{job_id}/submit")
async def submit_payment(job_id: str, transaction_id: str = Form(...), screenshot: UploadFile | None = File(None)):
    contents = await screenshot.read() if screenshot else None
    record = payments.submit_payment(job_id, transaction_id, contents, screenshot.filename if screenshot else "")
    return RedirectResponse(f"/unlock/{record['product']}/{job_id}", status_code=303)


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    _require_admin(request)
    return ui_env.get_template("admin.html").render(
        payments=payments.all_payments(),
        users=auth.all_users(),
        jobs=_all_jobs(),
    )


@app.post("/admin/payments/{payment_id}/verify")
async def admin_verify_payment(request: Request, payment_id: str, approved: str = Form("true")):
    user = _require_admin(request)
    record = payments.verify(payment_id, user["email"], approved == "true")
    if record["status"] == "verified":
        jobs.update(record["job_id"], paid=True)
    return RedirectResponse("/admin", status_code=303)


@app.post("/api/paytm/macro-verify")
async def paytm_macro_verify(amount: int = Form(...), transaction_id: str = Form(...), reference: str = Form(...), token: str = Form(...)):
    if token != settings.SECRET_KEY:
        raise HTTPException(403, "Bad webhook token")
    record = payments.macro_verify(amount, transaction_id, reference)
    if record:
        jobs.update(record["job_id"], paid=True)
    return {"ok": bool(record), "payment": record}


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


@app.post("/api/chat-wrapped/upload")
async def chat_wrapped_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await _read_upload(file)
    job_id, upload_path = _save_upload(contents, file.filename)
    background_tasks.add_task(run_chat_wrapped_pipeline, job_id, upload_path)

    return {
        "job_id": job_id,
        "status_url": f"/processing/chat-wrapped/{job_id}",
        "result_url": f"/wrapped/{job_id}",
    }


@app.get("/api/chat-wrapped/status/{job_id}")
async def chat_wrapped_status(job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    return s.to_dict()


@app.get("/api/chat-wrapped/result/{job_id}")
async def chat_wrapped_result(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    if s.state != "done" or not s.stats:
        raise HTTPException(409, "Chat Wrapped result is not ready")
    return _wrapped_preview(s.stats, _has_unlock(request, job_id))


@app.get("/api/chat-wrapped/pdf/{job_id}")
async def chat_wrapped_pdf(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None or not s.preview_pdf:
        raise HTTPException(404, "PDF not ready")
    if not _has_unlock(request, job_id):
        raise HTTPException(402, "Unlock required")
    full_path = _resolve_output_path(s.preview_pdf)
    if full_path is None:
        raise HTTPException(404, "PDF file missing")
    return FileResponse(full_path, media_type="application/pdf", filename="chat_wrapped.pdf")


@app.post("/api/gift-engine/upload")
async def gift_engine_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await _read_upload(file)
    job_id, upload_path = _save_upload(contents, file.filename)
    background_tasks.add_task(run_gift_engine_pipeline, job_id, upload_path)
    return {
        "job_id": job_id,
        "status_url": f"/processing/gift-engine/{job_id}",
        "result_url": f"/gift-engine/results/{job_id}",
    }


@app.get("/api/gift-engine/status/{job_id}")
async def gift_engine_status(job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    return s.to_dict()


@app.get("/api/gift-engine/result/{job_id}")
async def gift_engine_result(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    if s.state != "done" or not s.stats:
        raise HTTPException(409, "Gift Engine result is not ready")
    return _gift_preview(s.stats, _has_unlock(request, job_id))


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


def _find_uploaded_file(job_id: str) -> Path | None:
    """Find the original upload for a job so the LLM/PDF work can be rerun
    without making the user upload the same file again."""
    upload_dir = UPLOADS_DIR / job_id
    if not upload_dir.exists():
        return None
    files = [p for p in upload_dir.iterdir() if p.is_file()]
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


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


@app.get("/normalized/{job_id}/txt")
async def normalized_txt(job_id: str):
    """Download the normalized text — what the parser actually extracted
    from the user's raw input, in canonical [YYYY-MM-DD HH:MM:SS] sender: text
    format. Useful for verifying before chapter generation runs, and
    debugging when a chapter looks wrong."""
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    if not s.normalized_txt:
        # Fallback: try canonical location
        canonical = OUTPUT_DIR / job_id / "normalized.txt"
        if canonical.exists():
            return FileResponse(canonical, media_type="text/plain",
                                filename="normalized.txt")
        raise HTTPException(404, "Normalized output not available yet")
    path = _resolve_output_path(s.normalized_txt)
    if path is None:
        raise HTTPException(404, "Normalized file missing")
    return FileResponse(path, media_type="text/plain",
                        filename="normalized.txt")


@app.get("/normalized/{job_id}/summary")
async def normalized_summary(job_id: str):
    """Return the diagnostic summary as JSON. What was detected, how
    many messages, how many filtered as noise, which senders, etc."""
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    canonical = OUTPUT_DIR / job_id / "summary.json"
    if not canonical.exists():
        raise HTTPException(404, "Summary not yet available")
    import json as _json
    return _json.loads(canonical.read_text(encoding="utf-8"))


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


@app.post("/api/rewrite/{job_id}")
async def rewrite_story(job_id: str, background_tasks: BackgroundTasks):
    """Re-run story generation for an existing upload.

    This is intentionally heavier than /api/retry: it redoes the LLM story
    pass and PDF render, but it does not require the user to upload the chat
    again. Use it after prompt/code changes or when the normalized input looks
    good but the generated prose needs another pass.
    """
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")

    upload_path = _find_uploaded_file(job_id)
    if upload_path is None:
        raise HTTPException(
            404,
            "Original upload not found for this job. Please upload again."
        )

    jobs.update(
        job_id,
        state="queued",
        progress=0,
        message="Rewriting story from the saved upload...",
        error="",
        preview_pdf="",
        full_pdf="",
    )
    background_tasks.add_task(run_pipeline, job_id, upload_path)
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
