# ChatStory — Admin & Operations Guide

Everything an operator needs to **run, manage, and understand** this app.
For first-time setup from scratch see `docs/01_SETUP.md`; this file is the
practical day-to-day reference. Every individual setting is also documented
inline in `backend/.env.example`.

---

## 1. What this app is

A self-hosted FastAPI web app offering four products. Create an account →
upload a chat export (or a PDF) → a background job processes it →
preview free → pay to unlock → download. The philosophy is **memory
preservation, not summarization**: quality over speed, on modest hardware
(tuned for an i7 / 8 GB / GTX 1650 box with local Ollama).

| Product | Route | Needs an LLM? | Result |
|---|---|---|---|
| **ChatStory** (faithful book) | `/chatstory` | Yes (Ollama/Groq) | The **whole chat**, scene by scene, nothing skipped except noise — **A4 PDF + editable DOCX**, with timestamp footnotes; paywalled |
| **ChatWrapped** (recap) | `/chat-wrapped` | No | Story-style analytics web page + PDF |
| **GiftBook / Gift Engine** | `/gift-engine` | Optional (personalization) | Evidence-matched gift ideas + PDF + JSON |
| **Enhance a PDF** (clipart) | `/pdf-clipart` | Local Stable Diffusion (GPU) | Annotated PDF (free) |

ChatWrapped and Gift Engine are **deterministic** (no hallucination). Gift
Engine can optionally use the LLM to personalize ideas (`GIFT_ENGINE_USE_LLM`,
falls back if the model is down). PDF Clipart needs a GPU for the `local`
backend (or use `stub`/`api`). ChatStory needs Ollama (local) or a Groq key.

---

## 2. Architecture

```
                 ┌───────────────────────── FastAPI (app/main.py) ─────────────────────────┐
 Browser ──────► │  routes: pages, /api/*/upload, /api/*/status, /download/*, /admin        │
                 │   • auth (cookie sessions, MANDATORY for upload)                          │
                 │   • payments (manual UPI  OR  razorpay)   • payment_provider abstraction  │
                 │   • exports (download links)              • ops (storage + activity log)  │
                 └───────────────┬──────────────────────────────────────────────┬──────────┘
                                 │ enqueue                                        │ read/write
                                 ▼                                                ▼
                    JobQueue (services/queue.py)                          storage/  (JSON + files)
                    1 worker, FIFO, per-job timeout,                       ├── jobs/<id>.json
                    restart-safe (re-enqueues on boot)                     ├── uploads/<id>/...
                                 │ runs one of:                            ├── output/<id>/...
                                 ▼                                         ├── users.json
   ┌───────────────────────────────────────────────────────────┐         ├── payments.json
   │ run_chat_wrapped_pipeline   (services/chat_wrapped.py)      │         └── payment_screenshots/
   │ run_gift_engine_pipeline    (services/gift_engine.py)       │
   │ run_pdf_clipart_pipeline    (services/pdf_clipart_service)──┼──► pdf_clipart/ (standalone module)
   │ run_pipeline (ChatStory)    (pipeline/orchestrator.py)──────┼──► llm.py → Ollama/Groq
   │   └─ faithful (whole-chat scenes) → book_export → PDF + DOCX │
   └───────────────────────────────────────────────────────────┘
```

**Lifecycle:** upload saves the file under `storage/uploads/<job_id>/`, creates
`storage/jobs/<job_id>.json`, and enqueues a pipeline. The single queue worker
runs it (heavy work offloaded to a thread so status polling stays responsive),
writing progress to the job JSON. The browser polls `/api/<product>/status/<id>`
until `done`/`failed`. Jobs left running at shutdown are **re-enqueued on the
next boot**, so a restart never loses work.

### Key directories
```
backend/app/
  main.py                 FastAPI routes + PRODUCTS registry
  settings.py             all config (loaded from backend/.env)
  llm.py                  Ollama/Groq client
  pipeline/               faithful (whole-chat scenes), book_export (PDF+DOCX),
                          orchestrator, stats, normalizer, parsers/zip_utils
  services/               jobs, queue, auth, payments, payment_provider, exports,
                          image_gen, pdf_render, chat_wrapped, gift_engine,
                          gift_llm, pdf_clipart_service, telegram_bot, notify, ops
frontend/templates/       Jinja pages (book.html = the PDF)   frontend/static/  CSS + cliparts
pdf_clipart/              standalone PDF-clipart pipeline (also CLI)
backend/storage/          runtime data (gitignored)
```

---

## 3. Setup & run (server)

```bash
git checkout main && git pull
cd backend
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium                          # REQUIRED for any PDF output

# To enable the PDF Clipart product, also install its deps in THIS venv:
pip install -r ../pdf_clipart/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # for local SD

cp .env.example .env        # then edit (see §6). Set SECRET_KEY + ADMIN_PASSWORD.

# ChatStory only: run Ollama
ollama serve & ollama pull llama3.1:8b

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Storage dirs are created automatically. To expose
publicly see `docs/04_GO_LIVE.md` (Cloudflare Tunnel).

---

## 4. The user journey (V2)

1. **Accounts are mandatory.** A logged-out upload attempt redirects to
   **`/welcome`** ("A small favor before we begin" — why an account is
   required). Sign up → lands on **`/journey`** (the 7-step explainer).
2. **Upload** behind a **blocking terms modal** (must accept before any file is
   sent). ChatStory: upload → chat-at-a-glance + price → start. The **whole
   chat is kept** (no date picking, no sampling).
3. **Processing** shows live **queue position + ETA**. ChatStory adds a live
   tracker: progress, a **generation log**, a **"Preview so far"** page
   (`/job/{id}/live`, auto-refreshing) and a **Cancel** button — long faithful
   books can take a while, so you can watch scenes appear and stop if quality
   looks off (the partial book is kept).
4. **Preview** is free (the first chapter); the rest is paywalled.
5. **Pay & unlock** — manual UPI or Razorpay (§7).
6. **Download** from the unlock page or **`/dashboard`** ("My stuff"), which
   tracks every job's status, payment, and downloads. An **"Enhance my PDF"**
   CTA then points to the clipart product.
7. On failure the user sees a **Retry** button + a **Contact support** link.
8. A paid user who isn't logged in can re-open downloads with the **OTP access
   code** at **`/access`**.

---

## 5. Admin panel (`/admin`)

- **Become admin:** set `ADMIN_EMAIL`/`ADMIN_PASSWORD` in `.env` — that account
  is created/promoted on startup. Log in at `/login`. (If blank, the *first*
  signup becomes admin.)
- **Worker queue** — concurrency, pending/running jobs.
- **Storage & disk** — per-area sizes + file counts, volume used/free/percent,
  with **alerts** (>80%/>90% disk, <1 GB free, retention off). JSON at
  `/api/admin/storage`.
- **Payment verification** — Verify / Reject pending payments (manual flow).
- **Users**, **Jobs** (Retry / Unlock actions), and a **Recent activity** log
  (job finish/timeout/failure, payments, Razorpay confirmations, purges,
  startup — in-memory, last 500 events, resets on restart).

---

## 6. Configuration (`backend/.env`)

Defaults preserve current behavior, so you can run with almost nothing set.

| Key | Default | Notes |
|---|---|---|
| `USE_OLLAMA` | `true` | ChatStory LLM: Ollama (local) vs Groq |
| `GROQ_API_KEY` | — | needed if `USE_OLLAMA=false` |
| `OLLAMA_MODEL_STRONG`, `OLLAMA_HOST`, `OLLAMA_NUM_CTX` | llama3.1:8b … 8192 | **keep `NUM_CTX≥8192`** or prompts truncate |
| `GIFT_ENGINE_USE_LLM` | `true` | LLM-personalized gift ideas (falls back if down) |
| `FAITHFUL_SCENE_TIMEOUT_SECONDS` | 120 | per-scene LLM timeout (falls back to a grounded opener) |
| `QUEUE_JOB_TIMEOUT_SECONDS` | 86400 | generous so big faithful books aren't killed mid-run |
| `PDF_CLIPART_BACKEND` | `local` | Enhance-PDF product: `local` (SD/GPU) / `api` / `stub` (no GPU) |
| `PDF_CLIPART_MODEL` / `_MODEL_PATH` / `_STEPS` / `_MAX_PER_PAGE` | sd-turbo / — / 2 / 2 | reuse a local checkpoint via `_MODEL_PATH` |
| **Pricing** | | |
| `CURRENCY_SYMBOL` | ₹ | |
| `PRICE_SMALL/MEDIUM/LARGE`, `MSG_MEDIUM_MIN`, `MSG_LARGE_MIN` | 49/99/199, 2000, 10000 | tiered ChatStory price by total message volume |
| `SINGLE_EXPORT_PRICE`, `COMBINED_EXPORT_PRICE` | 50, 75 | Wrapped / Gift unlocks |
| **Payments** | | **see §7** |
| `PAYMENT_PROVIDER` | `manual` | or `razorpay` |
| `UPI_ID` / `PAYMENT_QR_PATH` | — | fall back to `PAYTM_UPI_ID` / `PAYTM_QR_IMAGE` |
| `RAZORPAY_KEY` / `RAZORPAY_SECRET` | — | only for `razorpay` |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_ADMIN_CHAT_ID` | — | Telegram approvals (optional) |
| `SMTP_HOST/PORT/USER/PASS/FROM` | — | email notifications (optional, no-op if unset) |
| **Queue & storage** | | |
| `QUEUE_MAX_CONCURRENT` | 1 | keep at 1 on a single-GPU box |
| `QUEUE_JOB_TIMEOUT_SECONDS`, `AVG_JOB_SECONDS` | 7200, 600 | job timeout / ETA estimate |
| `AUTO_DELETE_AFTER_HOURS` | 24 | uploads purged after this; **0 = never (alerts)** |
| `MAX_UPLOAD_SIZE_MB`, `MAX_MESSAGES` | 200, 300000 | upload limits |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD` | — | seeded admin account |
| `SUPPORT_EMAIL`, `*_EXPORT_VIDEO`, `REFUND_POLICY` | — | trust surface (homepage/terms) |
| `SECRET_KEY` | change-me | **set a random value** (signs sessions / webhook token) |

---

## 7. Payments

Two providers share the same records (`payments.json`) and unlock logic.
Switch with `PAYMENT_PROVIDER` — no code change.

**Manual (default):** user enters email → pays to your **UPI ID** / scans the
**QR** → submits transaction ID + screenshot → you approve via `/admin` **or**
the **Telegram bot** → job unlocks (and, if SMTP is set, the user is emailed
their access code).

- *Telegram approval (optional):* set `TELEGRAM_BOT_TOKEN` (from @BotFather) +
  `TELEGRAM_ADMIN_CHAT_ID` (from @userinfobot). Screenshots are pushed with
  Approve/Reject buttons; reply with `/msg <job_id> your message`. Long-polling,
  **no public webhook needed.**

**Razorpay:** set `PAYMENT_PROVIDER=razorpay` + `RAZORPAY_KEY`/`RAZORPAY_SECRET`
(test keys `rzp_test_…` work in test mode). The unlock page opens Razorpay
Checkout; the return signature is verified server-side (HMAC-SHA256) at
`/api/payments/razorpay/verify`, which unlocks the job. **No public webhook
needed.** Missing keys → transparently falls back to manual.

PDF Clipart is **free** — no payment/unlock.

---

## 8. The faithful ChatStory book

ChatStory keeps the **whole conversation** — quality and completeness over
speed. It does NOT sample or summarize.

- **What's kept / removed:** every real text message is preserved verbatim.
  Only noise is dropped — links, media/memes, system & deleted messages.
- **Hybrid rendering:** each scene (a conversation session) gets 1-2
  **grounded** scene-setting sentences (the only generated text; strict
  no-invention prompt + a deterministic time-of-day fallback, so it never
  hangs and never fabricates), followed by the **real dialogue** with per-line
  times and a **timestamp footnote** for cross-referencing the export.
- **Outputs** (`storage/output/<job>/`): `preview.pdf` (free — first chapter),
  `full.pdf` (A4, print margins, page numbers), and `full.docx` (**editable**,
  with real Word footnotes so the reader can fix any inaccuracy). Chapters are
  by calendar month. A statistics page leads the book: messages per month, who
  texted more, who initiated more.
- **Live generation:** the manuscript is persisted incrementally to
  `manuscript.json` with a `genlog.txt`; the tracker shows progress + log, a
  live preview (`/job/{id}/live`), and a **Cancel** button (stops at the next
  scene, keeps the partial book).
- **Speed:** one LLM call per scene → big chats can take a long time *by
  design*. `QUEUE_JOB_TIMEOUT_SECONDS` is generous; `FAITHFUL_SCENE_TIMEOUT_SECONDS`
  bounds each scene.
- **Pricing:** tiered by total message volume (`PRICE_SMALL/MEDIUM/LARGE`); one
  payment unlocks the full PDF + DOCX.

> Footnotes: the **DOCX** has true page-bottom Word footnotes; the **PDF** shows
> the timestamp at the bottom of each scene plus page numbers (Chromium can't do
> CSS page-bottom footnotes). Per-line times appear in both.

> Note: WhatsApp/Instagram/Telegram **ZIP** exports often bundle unrelated files
> (.vcf contacts, media). The parser finds the real chat member by name +
> content (iOS `_chat.txt` *and* Android `WhatsApp Chat with ….txt`) and ignores
> the rest; an unreadable ZIP gives a clear error instead of a bogus 1-day book.

---

## 9. Reliability, storage & backups

- **Restart-safe queue:** jobs left `queued`/`processing` at shutdown are
  re-enqueued from their saved upload on next boot (or failed with a clear
  message if the upload was already purged).
- **Retry:** failure screens give users a Retry button
  (`POST /api/jobs/{id}/retry`, owner/admin only); admins can retry from
  `/admin`. ChatStory re-renders PDF + DOCX from the saved `manuscript.json`
  (no LLM re-spend).
- **Retention:** an hourly task purges uploads older than
  `AUTO_DELETE_AFTER_HOURS` (default 24). Generated PDFs + metadata are kept.
  `0` disables purging — the admin panel **alerts** you.
- **Storage monitoring:** the `/admin` Storage card + `/api/admin/storage`
  (§5).
- **Backups:** copy `backend/storage/`. **Reset everything:** stop the server
  and delete `backend/storage/` (recreated on next start).
- **Scaling note:** the queue is right-sized for one server. For multiple
  worker processes / 100+ concurrent users, move to Redis + external workers —
  the enqueue/runner seam is isolated in `services/queue.py`.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Upload redirects to `/welcome` | Not logged in — accounts are mandatory. Sign in first. |
| Any PDF fails to render | `playwright install chromium` not run in the venv |
| "Chapter generation incomplete" / generic prose | Ollama context too small — keep `OLLAMA_NUM_CTX≥8192`; ensure the model is pulled and `OLLAMA_HOST` reachable |
| ChatStory job fails: "Could not reach Ollama" | start `ollama serve`; or `USE_OLLAMA=false` + `GROQ_API_KEY` |
| Job hangs / times out on big chats | narrow the date range, or raise `QUEUE_JOB_TIMEOUT_SECONDS`; keep `QUEUE_MAX_CONCURRENT=1` |
| PDF Clipart fails: "No module named fitz/torch" | install `pdf_clipart/requirements.txt` (+ torch) in the backend venv, or `PDF_CLIPART_BACKEND=stub` |
| PDF Clipart slow / re-downloads a 3.4 GB model | set `PDF_CLIPART_MODEL_PATH` to a local checkpoint, or `HF_HOME` to your cache |
| PDF Clipart OOM on 4 GB VRAM | keep size 512 / steps low; CPU offload is on; try `sd15-lcm` |
| Razorpay button doesn't appear | `PAYMENT_PROVIDER=razorpay` **and** both keys must be set; else it falls back to manual |
| Telegram approvals not arriving | both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ADMIN_CHAT_ID` set; message the bot once |
| Disk filling up | check the Storage card; ensure `AUTO_DELETE_AFTER_HOURS>0`; prune `backend/storage/output` (PDFs are kept by design) |
| `/admin` returns 403 | log in as the `ADMIN_EMAIL` account |

Logs go to the uvicorn console (progress, model loads, job tracebacks); recent
events also appear in the `/admin` activity log. Use `--reload` in dev only.

---

## 11. Testing without a GPU

ChatWrapped and Gift Engine (no-LLM) and the PDF Clipart *flow* work with no
GPU: set `PDF_CLIPART_BACKEND=stub` for placeholder art to verify upload →
progress → themed PDF → download end to end, then switch to `local` for real
Stable Diffusion. `scripts/test_core_intelligence.py` exercises the
deterministic analysis offline.
