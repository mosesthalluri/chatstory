# ChatBook — Admin & Operations Guide

Everything an operator needs to **run, manage, and understand** this app.
For first-time setup from scratch see `docs/01_SETUP.md`; this file is the
practical day-to-day reference.

---

## 1. What this app is

A self-hosted FastAPI web app offering four products. Upload a chat export
(or a PDF) → it's processed by a background job → you view/download the result.

| Product | Needs an LLM? | What it does | Result |
|---|---|---|---|
| **Chat Wrapped** | No | Grounded, story-style recap from a chat export (stats, personality, milestones, care moments, memory cards, letter) | Web page + PDF poster |
| **Meaningful Gift Engine** | No | Evidence-matched gift ideas from the chat | Web page + PDF + JSON |
| **PDF Clipart** | Local Stable Diffusion (GPU) | Adds mood-themed AI clipart to a PDF, one page at a time | Annotated PDF (free) |
| **ChatStory** | Yes (Ollama/Groq) | Illustrated storybook from a chat (paywalled, currently "coming soon" to public) | Preview + full PDF |

Chat Wrapped and Gift Engine are **fully deterministic** — no model, no API,
no hallucination. PDF Clipart needs a GPU for the `local` backend (or use the
`stub`/`api` backend). ChatStory needs Ollama (local) or a Groq key.

---

## 2. Architecture

```
                 ┌───────────────────────── FastAPI (app/main.py) ─────────────────────────┐
 Browser ──────► │  routes: pages, /api/*/upload, /api/*/status, /download/*, /admin        │
                 │   • auth (cookie sessions)      • PRODUCTS registry                       │
                 │   • payments (manual UPI)       • exports (download links)                │
                 └───────────────┬──────────────────────────────────────────────┬──────────┘
                                 │ enqueue                                        │ read/write
                                 ▼                                                ▼
                    JobQueue (services/queue.py)                          storage/  (JSON + files)
                    1 worker, FIFO, per-job timeout                        ├── jobs/<id>.json
                                 │ runs one of:                            ├── uploads/<id>/...
                                 ▼                                         ├── output/<id>/...
   ┌───────────────────────────────────────────────────────────┐         ├── users.json
   │ run_chat_wrapped_pipeline   (services/chat_wrapped.py)      │         ├── payments.json
   │ run_gift_engine_pipeline    (services/gift_engine.py)       │         └── payment_screenshots/
   │ run_pdf_clipart_pipeline    (services/pdf_clipart_service)──┼──► pdf_clipart/ (standalone module)
   │ run_pipeline (ChatStory)    (pipeline/orchestrator.py)──────┼──► llm.py → Ollama/Groq
   └───────────────────────────────────────────────────────────┘
                                 │ deterministic analysis
                                 ▼
                    core/ (sessions, scoring, semantics, analytics)  +  pipeline/ (stats, chunker, …)
```

**Request → result lifecycle:** upload saves the file under
`storage/uploads/<job_id>/`, creates `storage/jobs/<job_id>.json`, and enqueues
a pipeline. The single queue worker runs it (heavy CPU work is offloaded to a
thread so status polling stays responsive), writing progress back to the job
JSON. The browser polls `/api/<product>/status/<id>` until `done`/`failed`,
then shows the result and download links.

### Key directories
```
backend/app/
  main.py                 FastAPI routes + PRODUCTS registry
  settings.py             all config (loaded from backend/.env)
  llm.py                  Ollama/Groq client (ChatStory only)
  models.py               Message / JobStatus / dataclasses
  parsers/                WhatsApp / Instagram / Telegram / generic + detect
  core/                   deterministic relationship intelligence
  pipeline/               stats, chunker, chapter_gen, orchestrator (ChatStory)
  services/               jobs, queue, auth, payments, exports, image_gen,
                          pdf_render, chat_wrapped, gift_engine, pdf_clipart_service
frontend/templates/       Jinja pages   frontend/static/  CSS + cliparts
pdf_clipart/              standalone PDF-clipart pipeline (also CLI)
backend/storage/          runtime data (gitignored)
```

---

## 3. Setup & run (server)

```bash
cd backend
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium                          # REQUIRED for any PDF output

# To enable the PDF Clipart product, also install its deps in THIS venv:
pip install -r ../pdf_clipart/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # for local SD

cp .env.example .env        # then edit (see §5). Set SECRET_KEY + ADMIN_PASSWORD.

# ChatStory only: run Ollama
ollama serve & ollama pull llama3.1:8b

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. To expose publicly, see `docs/04_GO_LIVE.md`
(Cloudflare Tunnel).

---

## 4. Admin tasks

- **Become admin:** set `ADMIN_EMAIL`/`ADMIN_PASSWORD` in `.env` — that account
  is created/promoted automatically on startup. Log in at `/login`. (If those
  are blank, the *first* person to sign up becomes admin.)
- **Admin dashboard `/admin`:** lists payments, users, jobs, and the live queue.
- **Verify a payment:** user pays via UPI and submits a transaction ID on the
  unlock page → it appears in `/admin` → click verify → the job unlocks.
- **Retry / cancel / manually unlock a job:** buttons in `/admin` (retry re-runs
  the original upload; unlock marks it paid without payment).
- **PDF Clipart is free** — no payment/unlock needed.

---

## 5. Configuration (`backend/.env`)

| Key | Default | Notes |
|---|---|---|
| `USE_OLLAMA` | `true` | ChatStory LLM: Ollama (local) vs Groq |
| `GROQ_API_KEY` | — | needed if `USE_OLLAMA=false` |
| `OLLAMA_MODEL_FAST/STRONG`, `OLLAMA_HOST`, `OLLAMA_NUM_CTX` | llama3.1:8b … | local model settings |
| `USE_GEMINI_IMAGES`, `GEMINI_API_KEY` | false | ChatStory images (clipart by default) |
| `PDF_CLIPART_BACKEND` | `local` | `local` (SD/GPU) / `api` / `stub` (no GPU) |
| `PDF_CLIPART_MODEL` | `sd-turbo` | or `sd15-lcm` |
| `PDF_CLIPART_MODEL_PATH` | — | reuse a local `.safetensors`/diffusers folder (skip download) |
| `PDF_CLIPART_STEPS` / `_MAX_PER_PAGE` | 2 / 2 | inference steps / cliparts per page |
| `CHAPTERS_PER_BOOK`, `PREVIEW_CHAPTERS` | 0,1 | ChatStory |
| `SINGLE_EXPORT_PRICE`, `COMBINED_EXPORT_PRICE`, `FULL_BOOK_PRICE`, `CURRENCY_SYMBOL` | 50/75/399/₹ | pricing |
| `PAYTM_UPI_ID`, `PAYTM_QR_IMAGE` | — | shown on the unlock page |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD` | — | seeded admin account |
| `MAX_UPLOAD_SIZE_MB`, `MAX_MESSAGES` | 200, 300000 | upload limits |
| `QUEUE_MAX_CONCURRENT`, `QUEUE_JOB_TIMEOUT_SECONDS` | 1, 7200 | queue |
| `SECRET_KEY` | change-me | **set a random value** (signs session cookies) |

---

## 6. Data, storage & backups

All runtime state lives under `backend/storage/` (gitignored):
- `jobs/<id>.json` — job status + computed stats. `uploads/<id>/` — originals.
  `output/<id>/` — generated PDFs/JSON/images.
- `users.json`, `payments.json`, `payment_screenshots/` — accounts & payments.

**Back up:** copy `backend/storage/`. **Reset everything:** stop the server and
delete `backend/storage/` (it's recreated on next start). Old uploads can be
purged by `AUTO_DELETE_AFTER_HOURS` (see `scripts/cleanup.py`).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Any PDF fails to render | `playwright install chromium` not run in the venv |
| PDF Clipart job fails: "No module named fitz/torch" | install `pdf_clipart/requirements.txt` (+ torch) in the backend venv, or set `PDF_CLIPART_BACKEND=stub` |
| PDF Clipart slow / re-downloads a 3.4GB model | set `PDF_CLIPART_MODEL_PATH` to a local checkpoint, or `HF_HOME` to your model cache (see `pdf_clipart/README.md`) |
| PDF Clipart OOM on 4GB VRAM | keep size 512 / steps low; CPU offload is on by default; try `sd15-lcm` |
| ChatStory job fails: "Could not reach Ollama" | start `ollama serve`; or set `USE_OLLAMA=false` + `GROQ_API_KEY` |
| Gift Engine "datetime not JSON serializable" | fixed — pull latest |
| Chat Wrapped persona/quotes look wrong | fixed — now grounded in real messages; pull latest |
| Status page stuck / never finishes | check the uvicorn console for the job traceback; `/admin` shows the error |
| `/admin` returns 403 | log in as the `ADMIN_EMAIL` account |

Logs go to the uvicorn console (per-page progress, model loads, job
tracebacks). Run with `--reload` in dev only.

---

## 8. Testing without a GPU

The two no-LLM products and the PDF Clipart *flow* can be exercised with no
GPU. For PDF Clipart set `PDF_CLIPART_BACKEND=stub` to get placeholder art and
verify upload → progress → themed PDF → download end to end, then switch to
`local` for real Stable Diffusion output.
