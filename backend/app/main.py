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
from contextlib import asynccontextmanager
from pathlib import Path

# --- Windows asyncio fix (belt-and-suspenders) ---
# Playwright needs subprocess support, which requires ProactorEventLoop on
# Windows. The actual rendering uses sync_playwright in a thread (see
# services/pdf_render.py) which sidesteps this entirely, but we set the
# policy here too so any future async subprocess code Just Works.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .pipeline.orchestrator import run_pipeline, retry_render
from .services import auth, jobs, payments
from .services.chat_wrapped import run_chat_wrapped_pipeline
from .services.gift_engine import run_gift_engine_pipeline
from .services.pdf_clipart_service import run_pdf_clipart_pipeline
from .services.queue import job_queue
from .services import exports as export_svc
from .settings import (
    UPLOADS_DIR, OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR, STORAGE_ROOT,
    settings, BACKEND_ROOT,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure a known admin account exists if ADMIN_EMAIL/ADMIN_PASSWORD are
    # configured. Without this, "first signup becomes admin" means a random
    # visitor could claim admin, and an existing non-admin user base would
    # lock admin out entirely.
    try:
        auth.seed_admin()
    except Exception as exc:  # never block startup on seeding
        print(f"[startup] admin seed skipped: {exc}")
    await job_queue.start()
    yield
    await job_queue.stop()


app = FastAPI(title="ChatBook", version="0.1.0", lifespan=lifespan)

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
    "pdf-clipart": {
        "title": "PDF Clipart",
        "kicker": "AI illustration",
        "description": "Upload a PDF and get it back with cute, minimal AI clipart added to pages that have a clear visual theme.",
        "api_upload": "/api/pdf-clipart/upload",
        "api_status": "/api/pdf-clipart/status",
        "result_path": "/pdf-clipart/result",
        "accept": ".pdf",
        "upload_hint": "Drop a PDF here — or tap to choose. Max {max}MB.",
        "free": True,
    },
    # ChatStory uses the legacy /job status page rather than the generic
    # /processing page, but it still needs an entry here so the shared
    # /unlock/{slug} and /download/{slug} hub routes resolve instead of 404.
    "chatstory": {
        "title": "ChatStory Storybook",
        "kicker": "Storybook",
        "description": "Turn a chat export into an illustrated PDF storybook.",
        "api_upload": "/api/upload",
        "api_status": "/api/status",
        "result_path": "/job",
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


def _is_free_product(job_id: str) -> bool:
    status = jobs.load(job_id)
    product = status.product if status else None
    return bool(product and PRODUCTS.get(product, {}).get("free"))


def _has_unlock(request: Request, job_id: str) -> bool:
    if _is_admin(request):
        return True
    if _is_free_product(job_id):
        return True
    return export_svc.is_unlocked(job_id)


def _unlock_context(request: Request, job_id: str, product_slug: str) -> dict:
    unlocked = _has_unlock(request, job_id)
    links = export_svc.download_links(job_id, product_slug)
    payment = payments.for_job(job_id)
    return {"unlocked": unlocked, "download_links": links, "payment": payment}


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
            "cinematic_headline": data.get("cinematic_headline", ""),
            "total_messages": data.get("total_messages", 0),
            "active_days": data.get("active_days", 0),
            "teasers": data.get("teasers", []),
            "payment_status": "locked",
            "locked_sections": [
                "emotional_clock",
                "relationship_arc",
                "inside_jokes",
                "nicknames",
                "full heatmap",
            ],
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


def _user_email(request: Request) -> str | None:
    user = _current_user(request)
    return user.get("email") if user else None


def _require_user(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(401, "Login required")
    return user


def _save_upload(
    contents: bytes,
    filename: str | None,
    *,
    user_email: str | None,
    product: str,
) -> tuple[str, Path]:
    job_id = jobs.new_job_id()
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_upload_dir / Path(filename or "chat").name
    upload_path.write_bytes(contents)
    jobs.create(job_id, user_email=user_email, product=product)
    return job_id, upload_path


async def _enqueue_pipeline(job_id: str, product: str, fn, upload_path: Path) -> None:
    await job_queue.enqueue(job_id, product, fn, job_id, upload_path)


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


@app.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    user = _require_user(request)
    user_jobs = []
    for status in jobs.list_all(user_email=user["email"]):
        row = status.to_dict()
        row["result_url"] = jobs.result_url(status)
        row["unlocked"] = _has_unlock(request, status.job_id)
        row["payment"] = payments.for_job(status.job_id)
        row["download_links"] = export_svc.download_links(status.job_id, status.product)
        user_jobs.append(row)
    return ui_env.get_template("dashboard.html").render(
        user=user,
        jobs=user_jobs,
        currency=settings.CURRENCY_SYMBOL,
    )


@app.get("/api/me/jobs")
async def api_my_jobs(request: Request):
    user = _require_user(request)
    rows = []
    for status in jobs.list_all(user_email=user["email"]):
        row = status.to_dict()
        row["result_url"] = jobs.result_url(status)
        row["unlocked"] = _has_unlock(request, status.job_id)
        row["payment_status"] = (payments.for_job(status.job_id) or {}).get("status", "none")
        rows.append(row)
    return {"jobs": rows, "queue": job_queue.snapshot()}


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


@app.get("/pdf-clipart", response_class=HTMLResponse)
async def pdf_clipart_page(request: Request):
    template = ui_env.get_template("product_upload.html")
    return template.render(product=PRODUCTS["pdf-clipart"], max_size_mb=settings.MAX_UPLOAD_SIZE_MB, user=_current_user(request))


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
    ctx = _unlock_context(request, job_id, "chat-wrapped")
    template = ui_env.get_template("wrapped_result.html")
    return template.render(job_id=job_id, **ctx)


@app.get("/gift-engine/results/{job_id}", response_class=HTMLResponse)
async def gift_dashboard(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    ctx = _unlock_context(request, job_id, "gift-engine")
    template = ui_env.get_template("gift_result.html")
    return template.render(job_id=job_id, **ctx)


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


def _render_unlock_page(request: Request, product_slug: str, job_id: str, payment=None):
    product = PRODUCTS.get(product_slug)
    if product is None or jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    ctx = _unlock_context(request, job_id, product_slug)
    return ui_env.get_template("unlock.html").render(
        product=product,
        product_slug=product_slug,
        job_id=job_id,
        payment=payment or ctx["payment"],
        unlocked=ctx["unlocked"],
        download_links=ctx["download_links"],
        single_price=settings.SINGLE_EXPORT_PRICE,
        combined_price=settings.COMBINED_EXPORT_PRICE,
        paytm_upi_id=settings.PAYTM_UPI_ID,
        paytm_qr_image=settings.PAYTM_QR_IMAGE,
    )


@app.get("/unlock/{product_slug}/{job_id}", response_class=HTMLResponse)
async def unlock_page(request: Request, product_slug: str, job_id: str):
    return _render_unlock_page(request, product_slug, job_id)


@app.post("/unlock/{product_slug}/{job_id}", response_class=HTMLResponse)
async def create_unlock(request: Request, product_slug: str, job_id: str, email: str = Form(...), export_type: str = Form("single")):
    product = PRODUCTS.get(product_slug)
    if product is None or jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    payment = payments.create_intent(job_id, product_slug, email, export_type)
    return _render_unlock_page(request, product_slug, job_id, payment=payment)


@app.get("/download/{product_slug}/{job_id}", response_class=HTMLResponse)
async def download_hub(request: Request, product_slug: str, job_id: str):
    if PRODUCTS.get(product_slug) is None or jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    ctx = _unlock_context(request, job_id, product_slug)
    if not ctx["unlocked"]:
        return RedirectResponse(ctx["download_links"]["unlock"], status_code=303)
    return ui_env.get_template("downloads.html").render(
        product=PRODUCTS[product_slug],
        product_slug=product_slug,
        job_id=job_id,
        download_links=ctx["download_links"],
        payment=ctx["payment"],
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
        queue=job_queue.snapshot(),
    )


@app.get("/api/admin/queue")
async def admin_queue_status(request: Request):
    _require_admin(request)
    return job_queue.snapshot()


@app.post("/admin/jobs/{job_id}/retry")
async def admin_retry_job(request: Request, job_id: str):
    _require_admin(request)
    status = jobs.load(job_id)
    if status is None:
        raise HTTPException(404, "Job not found")
    upload_path = _find_uploaded_file(job_id)
    if upload_path is None:
        raise HTTPException(404, "Original upload not found")
    product = status.product or "chatstory"
    runners = {
        "chat-wrapped": run_chat_wrapped_pipeline,
        "gift-engine": run_gift_engine_pipeline,
        "pdf-clipart": run_pdf_clipart_pipeline,
        "chatstory": run_pipeline,
    }
    fn = runners.get(product, run_pipeline)
    jobs.update(job_id, state="queued", progress=0, message="Queued for retry", error="")
    await _enqueue_pipeline(job_id, product, fn, upload_path)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/jobs/{job_id}/unlock")
async def admin_unlock_job(request: Request, job_id: str):
    _require_admin(request)
    if jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    jobs.update(job_id, paid=True)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/jobs/{job_id}/cancel")
async def admin_cancel_job(request: Request, job_id: str):
    _require_admin(request)
    await job_queue.cancel(job_id)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/payments/{payment_id}/verify")
async def admin_verify_payment(request: Request, payment_id: str, approved: str = Form("true")):
    user = _require_admin(request)
    record = payments.verify(payment_id, user["email"], approved == "true")
    if record["status"] == "verified":
        jobs.update(record["job_id"], paid=True)
        product = record.get("product") or "chat-wrapped"
        return RedirectResponse(f"/unlock/{product}/{record['job_id']}", status_code=303)
    return RedirectResponse("/admin", status_code=303)


@app.post("/api/paytm/macro-verify")
async def paytm_macro_verify(amount: int = Form(...), transaction_id: str = Form(...), reference: str = Form(...), token: str = Form(...)):
    import hmac
    if not hmac.compare_digest(token, settings.SECRET_KEY):
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
async def upload(request: Request, file: UploadFile = File(...)):
    contents = await _read_upload(file)
    job_id, upload_path = _save_upload(
        contents,
        file.filename,
        user_email=_user_email(request),
        product="chatstory",
    )
    await _enqueue_pipeline(job_id, "chatstory", run_pipeline, upload_path)
    return {"job_id": job_id, "status_url": f"/job/{job_id}"}


@app.post("/api/chat-wrapped/upload")
async def chat_wrapped_upload(request: Request, file: UploadFile = File(...)):
    contents = await _read_upload(file)
    job_id, upload_path = _save_upload(
        contents,
        file.filename,
        user_email=_user_email(request),
        product="chat-wrapped",
    )
    await _enqueue_pipeline(job_id, "chat-wrapped", run_chat_wrapped_pipeline, upload_path)
    return {
        "job_id": job_id,
        "status_url": f"/processing/chat-wrapped/{job_id}",
        "result_url": f"/wrapped/{job_id}",
    }


def _job_status_payload(request: Request, job_id: str) -> dict:
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    payload = s.to_dict()
    payload["unlocked"] = _has_unlock(request, job_id)
    payload["download_links"] = export_svc.download_links(job_id, s.product)
    payload["payment_status"] = (payments.for_job(job_id) or {}).get("status", "none")
    return payload


@app.get("/api/chat-wrapped/status/{job_id}")
async def chat_wrapped_status(request: Request, job_id: str):
    return _job_status_payload(request, job_id)


@app.get("/api/chat-wrapped/result/{job_id}")
async def chat_wrapped_result(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    if s.state != "done" or not s.stats:
        raise HTTPException(409, "Chat Wrapped result is not ready")
    return _wrapped_preview(s.stats, _has_unlock(request, job_id))


def _pdf_attachment(path: Path, filename: str) -> FileResponse:
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/chat-wrapped/pdf/{job_id}")
async def chat_wrapped_pdf(request: Request, job_id: str):
    if not _has_unlock(request, job_id):
        raise HTTPException(402, "Unlock required — complete payment and wait for admin verification")
    path = export_svc.resolve_pdf_path(job_id, "preview_pdf")
    if path is None:
        raise HTTPException(404, "PDF not ready yet")
    return _pdf_attachment(path, "chat_wrapped.pdf")


@app.get("/download/chat-wrapped/{job_id}/pdf")
async def download_wrapped_pdf(request: Request, job_id: str):
    if not _has_unlock(request, job_id):
        return RedirectResponse(f"/unlock/chat-wrapped/{job_id}", status_code=303)
    path = export_svc.resolve_pdf_path(job_id, "preview_pdf")
    if path is None:
        raise HTTPException(404, "PDF file missing")
    return _pdf_attachment(path, "chat_wrapped.pdf")


@app.post("/api/gift-engine/upload")
async def gift_engine_upload(request: Request, file: UploadFile = File(...)):
    contents = await _read_upload(file)
    job_id, upload_path = _save_upload(
        contents,
        file.filename,
        user_email=_user_email(request),
        product="gift-engine",
    )
    await _enqueue_pipeline(job_id, "gift-engine", run_gift_engine_pipeline, upload_path)
    return {
        "job_id": job_id,
        "status_url": f"/processing/gift-engine/{job_id}",
        "result_url": f"/gift-engine/results/{job_id}",
    }


@app.get("/api/gift-engine/status/{job_id}")
async def gift_engine_status(request: Request, job_id: str):
    return _job_status_payload(request, job_id)


@app.post("/api/pdf-clipart/upload")
async def pdf_clipart_upload(request: Request, file: UploadFile = File(...)):
    contents = await _read_upload(file)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    job_id, upload_path = _save_upload(
        contents,
        file.filename,
        user_email=_user_email(request),
        product="pdf-clipart",
    )
    await _enqueue_pipeline(job_id, "pdf-clipart", run_pdf_clipart_pipeline, upload_path)
    return {
        "job_id": job_id,
        "status_url": f"/processing/pdf-clipart/{job_id}",
        "result_url": f"/pdf-clipart/result/{job_id}",
    }


@app.get("/api/pdf-clipart/status/{job_id}")
async def pdf_clipart_status(request: Request, job_id: str):
    return _job_status_payload(request, job_id)


@app.get("/pdf-clipart/result/{job_id}", response_class=HTMLResponse)
async def pdf_clipart_result(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    return ui_env.get_template("pdf_clipart_result.html").render(
        job_id=job_id,
        status=s.to_dict(),
        download_url=f"/download/pdf-clipart/{job_id}/pdf",
    )


@app.get("/download/pdf-clipart/{job_id}/pdf")
async def download_pdf_clipart(job_id: str):
    path = export_svc.resolve_pdf_path(job_id, "preview_pdf")
    if path is None:
        raise HTTPException(404, "Annotated PDF not ready yet")
    return _pdf_attachment(path, "annotated.pdf")


@app.get("/download/gift-engine/{job_id}/pdf")
async def download_gift_pdf(request: Request, job_id: str):
    if not _has_unlock(request, job_id):
        return RedirectResponse(f"/unlock/gift-engine/{job_id}", status_code=303)
    path = export_svc.resolve_pdf_path(job_id, "preview_pdf")
    if path is None:
        raise HTTPException(404, "Gift PDF not ready")
    return _pdf_attachment(path, "gift_engine.pdf")


@app.get("/download/gift-engine/{job_id}/json")
async def download_gift_json(request: Request, job_id: str):
    if not _has_unlock(request, job_id):
        raise HTTPException(402, "Unlock required")
    path = OUTPUT_DIR / job_id / "gift_engine.json"
    if not path.exists():
        raise HTTPException(404, "Gift data missing")
    return FileResponse(path, media_type="application/json", filename="gift_engine.json")


@app.get("/download/chatstory/{job_id}/preview")
async def download_chatstory_preview(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    path = export_svc.resolve_pdf_path(job_id, "preview_pdf")
    if path is None:
        raise HTTPException(404, "Preview not ready")
    return _pdf_attachment(path, "chatstory_preview.pdf")


@app.get("/download/chatstory/{job_id}/full")
async def download_chatstory_full(request: Request, job_id: str):
    if not _has_unlock(request, job_id):
        return RedirectResponse(f"/job/{job_id}", status_code=303)
    path = export_svc.resolve_pdf_path(job_id, "full_pdf")
    if path is None:
        raise HTTPException(404, "Full PDF not ready")
    return _pdf_attachment(path, "chatstory_full.pdf")


@app.get("/api/gift-engine/result/{job_id}")
async def gift_engine_result(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    if s.state != "done" or not s.stats:
        raise HTTPException(409, "Gift Engine result is not ready")
    return _gift_preview(s.stats, _has_unlock(request, job_id))


@app.get("/api/status/{job_id}")
async def status(request: Request, job_id: str):
    return _job_status_payload(request, job_id)


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
async def full_pdf(request: Request, job_id: str):
    s = jobs.load(job_id)
    if s is None or not s.full_pdf:
        raise HTTPException(404, "Full PDF not ready")
    if not _has_unlock(request, job_id):
        raise HTTPException(402, "Payment required — unlock from your dashboard or unlock page")
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
async def retry(job_id: str):
    """Retry just the PDF render step using the saved checkpoint."""
    if jobs.load(job_id) is None:
        raise HTTPException(404, "Job not found")
    await job_queue.enqueue(job_id, "chatstory", retry_render, job_id)
    return {"ok": True, "status_url": f"/job/{job_id}"}


@app.post("/api/rewrite/{job_id}")
async def rewrite_story(job_id: str):
    """Re-run story generation for an existing upload."""
    s = jobs.load(job_id)
    if s is None:
        raise HTTPException(404, "Job not found")
    upload_path = _find_uploaded_file(job_id)
    if upload_path is None:
        raise HTTPException(404, "Original upload not found for this job. Please upload again.")
    jobs.update(
        job_id,
        state="queued",
        progress=0,
        message="Rewriting story from the saved upload...",
        error="",
        preview_pdf="",
        full_pdf="",
    )
    await _enqueue_pipeline(job_id, "chatstory", run_pipeline, upload_path)
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
