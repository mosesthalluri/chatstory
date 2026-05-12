# Recreation prompt

If you ever want to rebuild ChatBook from scratch — using a different
LLM, different stack, or as an exercise — paste the prompt below into
GPT-4, Claude, Gemini, or any capable model. It encodes every design
decision so the rebuild matches the spirit of the original.

---

## THE PROMPT (copy everything below this line)

I want you to build a self-hostable web application called **ChatBook**.
It takes a chat export (WhatsApp, Instagram, Telegram, or generic text)
and produces an illustrated PDF storybook of the relationship plus a
Spotify-Wrapped-style stats page. There is a free preview and a paid
full version.

Build this as a complete, runnable starter project that a beginner can
operate. Prioritize clarity over cleverness, robustness over features.

### Hard requirements

- **Backend:** Python 3.11+, FastAPI, async throughout.
- **Frontend:** server-rendered HTML + vanilla JavaScript. No React, no
  build step, no framework. Mobile-responsive — usable on a phone with
  no app installed.
- **PDF:** generate HTML via Jinja templates, then print to PDF using
  Playwright headless Chromium. A5 page size.
- **LLM:** abstract a single `complete(messages, model_size)` interface
  that talks to either Groq (cloud, OpenAI-compatible API) or Ollama
  (local) based on a `USE_OLLAMA` env var. Both backends should support
  fast and strong model tiers.
- **Image gen:** Google Gemini 2.5 Flash Image API (free tier 500/day),
  with an SVG placeholder fallback when no key is configured.
- **Storage:** filesystem only. One JSON file per job in
  `backend/storage/jobs/`. No database in v0.1.
- **Background jobs:** FastAPI `BackgroundTasks` for v0.1. Single-process,
  in-memory queue is fine for the initial scale.

### Folder structure (must match)

```
chatbook/
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py              # FastAPI app, routes
│   │   ├── settings.py          # pydantic-settings, all env vars
│   │   ├── models.py            # Message, ParsedChat, JobStatus dataclasses
│   │   ├── llm.py               # unified LLM client (Groq + Ollama)
│   │   ├── parsers/
│   │   │   ├── __init__.py      # parse_chat() orchestrator + noise filter
│   │   │   ├── detect.py        # format detection
│   │   │   ├── whatsapp.py
│   │   │   ├── instagram.py
│   │   │   ├── telegram.py
│   │   │   └── generic.py       # last-resort fallback
│   │   ├── pipeline/
│   │   │   ├── stats.py         # deterministic stats engine
│   │   │   ├── chunker.py       # by_day / by_week / by_month / into_chapters
│   │   │   ├── highlights.py    # deterministic highlight selection
│   │   │   ├── summarize.py     # hierarchical: day → week → month → arc
│   │   │   ├── chapter_gen.py   # final chapter LLM prompt
│   │   │   └── orchestrator.py  # run_pipeline() master function
│   │   └── services/
│   │       ├── jobs.py          # JSON-per-job persistence
│   │       ├── image_gen.py     # Gemini + SVG fallback
│   │       └── pdf_render.py    # Jinja + Playwright
│   └── storage/
│       ├── uploads/   ├── jobs/   └── output/
├── frontend/
│   └── templates/
│       ├── upload.html          # mobile-first drag/drop upload
│       ├── status.html          # polls /api/status, shows progress
│       └── book.html            # the actual PDF template
└── docs/
    ├── 01_SETUP.md  ├── 02_HOW_IT_WORKS.md  ├── 03_CONFIG.md
    ├── 04_GO_LIVE.md  ├── 05_TROUBLESHOOTING.md  └── 06_RECREATION_PROMPT.md
```

### Critical design decisions (do not deviate)

**1. Normalized internal format.**
After parsing, every part of the codebase only sees a list of `Message`
objects with `(sender, timestamp, text, kind)`. Adding a new chat
platform = writing one new parser file. Nothing downstream changes.

**2. Hierarchical summarization, not single-pass.**
Real chats are 100k+ messages. Process them in layers:
- **Day digests** (60-100 words each) for active days. Parallel via
  `asyncio.Semaphore`.
- **Week digests** (100-200 words) roll up 7 days.
- **Month digests** (100-200 words) roll up 4 weeks.
- **Book arc** identifies 4-8 narrative threads from all month digests.
- **Chapter generation** sees: arc + relevant month digests + 25 verbatim
  highlight messages from a deterministic scorer. Never the raw stream.

**3. Stats and narrative are separate.**
Stats are pure Python (Counter, no LLM) — can't hallucinate.
Narrative is LLM. The Wrapped page comes from stats; chapters come from LLM.
This lets the free preview show a Wrapped page (cheap, deterministic) plus
one LLM chapter as a teaser.

**4. Hallucination prevention.**
Chapter prompts say "use only what's in the messages, do not invent
events." Highlights are picked deterministically by a scorer that
considers length, emotional keywords, late-night timestamps, post-silence
position, emoji density. Pull-quotes must be verbatim from the
highlights, never paraphrased.

**5. Noise filtering at parse time.**
Reactions, system messages, "this message was deleted," media-only
entries are tagged via `MessageKind` enum and filtered before anything
downstream sees them. Consecutive messages from same sender within 30
seconds are merged into single turns.

**6. Instagram mojibake fix.**
Instagram exports text as UTF-8 bytes interpreted as Latin-1. The
Instagram parser must re-encode every string with
`text.encode('latin-1').decode('utf-8')`. Without this, "héllo" becomes
"hÃ©llo" everywhere.

**7. HTML → PDF, not direct PDF.**
Jinja renders the book as HTML with proper CSS print rules (`@page`,
`page-break-after`). Playwright launches headless Chromium and prints
to PDF at A5 size. A designer can iterate on the template without
touching Python.

**8. Two PDFs per job.**
- `preview.pdf`: cover + Wrapped page + first chapter + paywall page.
  Watermarked with "PREVIEW". Always free.
- `full.pdf`: complete book, no watermark. Held in storage until
  `paid` flag flips true.

### Env config (`.env.example`)

```
USE_OLLAMA=false
GROQ_API_KEY=
GROQ_MODEL_FAST=llama-3.1-8b-instant
GROQ_MODEL_STRONG=llama-3.3-70b-versatile
OLLAMA_MODEL_FAST=llama3.1:8b
OLLAMA_MODEL_STRONG=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
GEMINI_API_KEY=
IMAGE_STYLE=flat vector cartoon illustration, soft pastel colors
CHAPTERS_PER_BOOK=8
PREVIEW_CHAPTERS=1
CURRENCY_SYMBOL=₹
FULL_BOOK_PRICE=399
AUTO_DELETE_AFTER_HOURS=24
MAX_UPLOAD_SIZE_MB=200
MAX_MESSAGES=300000
SECRET_KEY=change-me
DEBUG=true
```

### Routes

```
GET  /                  upload page (HTML)
GET  /job/{job_id}      status page (polls /api/status)
POST /api/upload        accepts file, returns {job_id, status_url}
GET  /api/status/{id}   JSON status
GET  /preview/{id}      preview PDF
POST /api/pay/{id}      stub — replace with Stripe/Razorpay webhook
GET  /full/{id}         full PDF, gated by paid=true
```

### Constraints

- Hardware target: i7 / 8GB RAM / GTX 1650 4GB VRAM (mid laptop)
- LLM cost target: under ₹50 per book on Groq paid; free on Ollama
- Time target: under 20 min per book on Groq, under 8 hours on Ollama
- Max input: 300,000 messages, 200MB upload
- No tracking, no analytics, no telemetry
- Auto-delete uploaded chat data after 24h; keep only derived PDF

### Documentation expected

- `01_SETUP.md` — total beginner: install Python, get API keys, run first time
- `02_HOW_IT_WORKS.md` — architecture explanation, why each decision
- `03_CONFIG.md` — every env var, when to change it
- `04_GO_LIVE.md` — Cloudflare Tunnel for laptop hosting; mention Oracle Free Tier as scale path
- `05_TROUBLESHOOTING.md` — every common error with the actual fix

### Style guide

- Type hints throughout
- Dataclasses for data models, not raw dicts
- Async functions for anything that touches I/O or external APIs
- Comments explain *why*, not *what*
- Every parser file follows the same shape: `parse(path: Path) -> ParsedChat`
- Errors that the user can fix produce friendly messages; bugs raise normal exceptions

### Now build it

Produce all files listed above with complete, runnable code. Don't
truncate. Don't say "the rest is similar." Test that imports resolve
and that a sample WhatsApp file parses end-to-end.

If you can run the code, do so and show the output. If anything
doesn't run, fix it before claiming you're done.

---

## END OF PROMPT

That prompt is ~1500 tokens. Most modern LLMs (GPT-4o, Claude 3.5+,
Gemini 1.5+) can produce ~80% of the codebase from this in one shot.
You'll need 2-3 follow-up prompts to fix bugs and fill in pieces it
truncated.

If you want to rebuild in a different language (Node, Go, Rust),
prepend: "Build this in [LANGUAGE] using equivalent libraries. Map
FastAPI → [their web framework], Playwright → [their headless browser
binding]." The architecture decisions transfer directly.
