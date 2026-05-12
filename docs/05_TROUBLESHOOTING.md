# Troubleshooting

The errors you'll actually hit, with fixes that actually work.

## ⚠️ Job crashes at 94% with `NotImplementedError` (Windows)

**Symptom:** Pipeline runs fine through parsing, summaries, chapter
writing, image generation. Hits 94% ("Rendering preview PDF…") and dies
with `NotImplementedError` and no other context.

**Cause:** Playwright launches headless Chromium as a subprocess. On
Windows, `uvicorn --reload` often uses `SelectorEventLoop`, which doesn't
support subprocess creation — Python's asyncio raises `NotImplementedError`
when Playwright tries to spawn the browser.

**Fix (v0.1.1+, already applied):**
- `services/pdf_render.py` uses `sync_playwright` in a worker thread via
  `asyncio.to_thread()`, sidestepping the event loop entirely.
- `main.py` also sets `WindowsProactorEventLoopPolicy()` at startup as
  belt-and-suspenders.

**If you're still on v0.1.0:** apply both changes from the v0.1.1 patch.
They are zero-impact on macOS/Linux, so safe to update everywhere.

**Recovering a job that already failed at 94%:** The expensive work (day
summaries, week/month rollups, chapter writing, image generation) is all
saved to disk in `backend/storage/output/<job_id>/checkpoint.json`. You
don't have to re-run from scratch — see "Retrying a failed job" below.

## ⚠️ `Image gen failed ... 429` messages in terminal

**Symptom:** Pipeline finishes "successfully" but the resulting PDF has
the same generic placeholder illustration on most or all chapters. Your
terminal shows lines like:

```
[image_gen] 429 rate limit, sleeping 5.0s (attempt 1/4)
[image_gen] fallback (rate_limited) for prompt '...'
```

**Cause:** Gemini's free tier has two limits:
- **500 requests/day** — easy to hit if you test multiple books in a day
- **~10 requests/minute** — easy to hit on a single book if you fire
  many image gens concurrently (default is 2 in flight)

**Fix (v0.1.1+, already applied):**
- Exponential backoff up to 4 retries before falling back
- Default concurrency lowered from 3 to 2
- Honors the `Retry-After` response header from Google
- Pipeline logs how many images fell back vs succeeded
- Status page shows a warning if any fell back to placeholder

**If you keep hitting it:**

1. **Spread out test runs.** Most free-tier 429s on a fresh day mean
   per-minute cap. Wait 60 seconds and try again.
2. **Slow down per-call.** In `.env`, lower `IMAGE_CONCURRENCY=2` to 1
   if you've added that knob, or edit `DEFAULT_PER_CALL_DELAY` in
   `services/image_gen.py` to `2.0` (2-second pause between calls).
3. **Check daily quota.** Visit
   https://aistudio.google.com/usage to see how many of your 500
   daily requests are left. Daily resets at midnight Pacific time.
4. **Generate fewer images per book.** In `.env` set `CHAPTERS_PER_BOOK=4`
   instead of 8 — halves the API spend per book.
5. **Upgrade to paid Gemini.** ~$0.04 per image, no per-minute cap.
   For ~17 images per book that's ~₹60 — still profitable at ₹399 sale.

## Retrying a failed job (no API spending)

If a job died in the rendering stage and you don't want to lose the
already-paid-for chapter generation and image generation work, you can
retry just the PDF step. The orchestrator saves a `checkpoint.json` after
chapter and image generation, before the PDF render starts.

**To retry:**

```bash
curl -X POST http://localhost:8000/api/retry/<job_id>
```

or just open the status page (`/job/<job_id>`) which polls and will show
fresh progress once the retry kicks off.

The retry uses zero LLM/Gemini API calls — only re-runs Playwright. If
the original failure was the Playwright bug above, make sure you've
applied the v0.1.1 patch first.

---

## Setup errors

### `python: command not found`

You don't have Python installed, or it's not on your PATH.

- **Windows:** reinstall from python.org and check "Add Python to PATH"
  on the first screen of the installer.
- **Mac:** try `python3` instead of `python`. Or install Python 3.12 via
  Homebrew: `brew install python@3.12`.
- **Linux:** `sudo apt install python3.12 python3.12-venv`.

### `pip install` fails with `error: Microsoft Visual C++ 14.0 or greater is required`

Some packages (like `pandas`, `numpy`) need a C compiler on Windows.
Install **Microsoft C++ Build Tools** from
https://visualstudio.microsoft.com/visual-cpp-build-tools/. Pick
"Desktop development with C++" workload. Restart your terminal afterwards.

### `pip install` fails with `Could not find a version that satisfies the requirement`

You're probably on Python 3.10 or older. Check with `python --version`.
ChatBook needs 3.11+.

### `playwright install chromium` fails

Network or permissions issue. Try:

```bash
# Force re-download
playwright install --force chromium

# On Linux, you may also need:
playwright install-deps chromium
```

If you're behind a corporate proxy, set `HTTPS_PROXY` first.

## Runtime errors

### `Could not detect chat format`

The parser couldn't figure out what kind of file you uploaded. Common causes:

- **It's a media-only WhatsApp export.** WhatsApp lets you export
  "without media" and "with media." Export *with* media — even though
  ChatBook only reads text, the .zip contains `_chat.txt` which we need.
- **The file is corrupted.** Try opening it in a text editor. If you
  see garbled characters, the export failed.
- **It's a screenshot, not a text export.** ChatBook doesn't OCR
  screenshots. Re-export as text from inside the chat app.

### `Too many messages (X). Maximum is 300000.`

Your chat is genuinely huge. Either:

- Bump `MAX_MESSAGES` in `.env` (and accept longer processing time and
  higher API cost).
- Pre-trim the file to a date range.
- Future feature (not in v0.1): UI date-range picker.

### `Could not reach Ollama at http://localhost:11434`

Ollama isn't running. Open a terminal and run:

```bash
ollama serve
```

Leave it running in that terminal. ChatBook needs it alive.

If `ollama serve` says "address already in use," Ollama is already
running — your error is something else (firewall, proxy). Check
`curl http://localhost:11434/api/tags` — if that works in a separate
terminal, the issue is in ChatBook's config.

### `Groq API error: 429`

You've hit the free-tier rate limit (14,400 requests per day, or 6,000
tokens per minute). Options:

- Wait 24 hours.
- Switch to Ollama (`USE_OLLAMA=true`) for the rest of today.
- Upgrade to Groq paid tier ($0.05 per million tokens).
- Reduce `CHAPTERS_PER_BOOK` to lower per-book request count.

### `Groq API error: 401`

Your `GROQ_API_KEY` is wrong or expired. Double-check it on
https://console.groq.com — copy it again carefully (no spaces, no quotes).

### `Image gen failed`

Probably a Gemini quota issue (500 images/day on free tier) or invalid
key. ChatBook silently falls back to placeholder SVGs, so books still
generate — they just look generic. Fix:

- Verify your `GEMINI_API_KEY` at https://aistudio.google.com/apikey
- If you've burned through the daily quota, wait 24 hours.
- Or simply leave `GEMINI_API_KEY` empty and accept SVG placeholders.

### Mojibake — text shows as `hÃ©llo` instead of `héllo`

Instagram-specific. The exporter writes UTF-8 bytes interpreted as
Latin-1, and we have to undo that. The Instagram parser handles this
automatically. If you see mojibake in the *output*, file an issue with
a sample message — it means our fix missed an edge case.

### PDF is blank or missing chapters

Probably a Playwright issue. Check:

- Did `playwright install chromium` finish successfully? Re-run it.
- Are there errors in the uvicorn terminal? They usually point to the
  Jinja template issue.
- Is the chapter body empty? Look at `backend/storage/jobs/{job_id}.json`
  — if `chapters` is empty, the LLM step failed.

### "Job not found" when checking status

The `backend/storage/jobs/` directory was deleted, or you're hitting
the wrong server. Re-upload.

## Performance issues

### Job sits at "Summarizing days" for ages

Normal — this is the bulk of the work. With Ollama on a GTX 1650, expect
**10-30 seconds per active day**. A 200k-message chat with 600 active
days takes 100-300 minutes.

Speed it up:

- Switch to Groq (`USE_OLLAMA=false`) — same step takes 5-10 minutes.
- Increase `max_concurrent` in `summarize.py` (default 4) if you have
  enough RAM. Watch memory: each concurrent Ollama request needs ~2GB.

### Laptop fan is screaming

That's Ollama maxing out your CPU/GPU. Options:

- Switch to Groq for any non-test job.
- Cap Ollama's CPU usage: `OLLAMA_NUM_THREAD=2 ollama serve`
- Run jobs overnight when you're not using the laptop.

### Out of memory

8GB RAM is tight when running Ollama (5GB for Llama 3.1 8B) plus
ChatBook plus a browser. Close other apps, or:

- Use a smaller Ollama model: `ollama pull llama3.2:3b` and set both
  Ollama model envs to `llama3.2:3b`.
- Switch to Groq, which uses no local RAM beyond the HTTP request.

## When things go really wrong

1. Check the uvicorn terminal — it logs full Python tracebacks.
2. Check `backend/storage/jobs/{job_id}.json` — the `error` field has
   the failure reason.
3. Set `DEBUG=true` in `.env` and restart for more verbose logs.
4. If it's a parser issue, try the file with the generic parser by
   renaming to `something.txt` (loses some structure but often works).
