# Running ChatBook fully local

Zero cloud, zero API keys, zero rate limits. Everything runs on your
laptop. Slower than cloud, but free forever and fully private.

This guide assumes you've already done the basic setup in `01_SETUP.md`
and have ChatBook running.

## What "fully local" means

| Component | Local replacement | What you give up |
|---|---|---|
| Groq cloud LLM | Ollama (Llama 3.1 8B) on your laptop | ~10× slower per call |
| Gemini image gen | Curated SVG cliparts (default) | AI-generated illustrations |
| Cloudflare Tunnel | localhost only | Phone access (unless on same WiFi) |

You keep: the entire pipeline (parse, stats, summarize, write chapters,
render PDF), the web UI, mobile-responsive design on the same network.

## Step 1 — Install Ollama (if you haven't already)

You said you already have Ollama, so this is just a check.

```bash
ollama --version
```

If that prints a version (e.g. `ollama version 0.4.2`), you're good.

If not, get it from https://ollama.com/download. Installer is ~200 MB,
takes 2 minutes.

## Step 2 — Pull the model

```bash
ollama pull llama3.1:8b
```

This downloads ~4.7 GB the first time. Subsequent runs use the cached copy.

**Verify it works:**

```bash
ollama run llama3.1:8b "say hi"
```

You should see a response within a few seconds. If you see "Error: model
not found", re-run the pull command.

**On your hardware (i7 / 8GB RAM / GTX 1650):** Llama 3.1 8B Q4 uses
about 5 GB of memory. With Ollama running, your free RAM drops to ~2-3 GB.
Close Chrome and other RAM hogs before running long jobs.

## Step 3 — Make sure Ollama is running

Ollama runs as a background service after install. Check:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response listing your models. If you see
"connection refused":

- **Windows:** Open the Ollama app from Start menu — it runs in the system
  tray. Right-click → "Start Ollama".
- **Mac:** Same — open Ollama from Applications.
- **Linux:** `sudo systemctl start ollama` or `ollama serve` in a separate
  terminal.

## Step 4 — Switch ChatBook to local mode

Edit `backend/.env` (create from `.env.example` if you haven't):

```ini
# Switch LLM to local Ollama
USE_OLLAMA=true

# These two can be empty — they're not used in local mode
GROQ_API_KEY=
GEMINI_API_KEY=

# Images use cliparts by default — leave this false
USE_GEMINI_IMAGES=false

# The model you pulled. If you pulled something else, change here.
OLLAMA_MODEL_FAST=llama3.1:8b
OLLAMA_MODEL_STRONG=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
```

**Why both FAST and STRONG point to the same model:** on consumer hardware
running two models means twice the RAM. With 8 GB total it'd thrash. One
model that's "good enough for everything" is the right call until you
upgrade hardware.

## Step 5 — Restart and run

```bash
# In the terminal where uvicorn is running, Ctrl+C to stop, then:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 and upload a chat. The status page will show
familiar stages — but each LLM step will be slower.

## What to expect performance-wise

For a 2-year LDR chat (~200k messages, ~700 active days), running on an
i7 / 8GB / GTX 1650:

| Stage | Local time (Ollama) | Cloud time (Groq) |
|---|---|---|
| Parse + clean | 30-60 sec | same |
| Stats engine | 1-2 min | same |
| Day digests (~700 calls) | **6-8 hours** | ~10 min |
| Week digests (~130 calls) | ~1 hour | ~2 min |
| Month digests (~30 calls) | ~15 min | ~30 sec |
| Book arc (1 call) | ~2 min | ~5 sec |
| Chapters (8 calls) | ~10-15 min | ~30 sec |
| Image picking (12 cliparts) | instant | instant |
| PDF render | ~5 sec | ~5 sec |
| **Total** | **~8-10 hours** | **~15 min** |

It's slow but it works. **Start a job before bed, come back to a
finished book.** A 1-month chat instead of 2 years finishes in about 30
minutes locally, which is fine for testing.

## Things that go wrong locally

### Laptop fan ramps up, gets hot

Normal. Llama 8B at 100% inference on 8 cores will get warm. Open
Task Manager / Activity Monitor — if CPU is at 90%+ and stays there,
that's expected work, not a bug.

To reduce thermal load: lower concurrency. In
`backend/app/pipeline/summarize.py` find `Semaphore(4)` and change to
`Semaphore(2)`. Halves the load, doubles the time. Worth it on a thin
laptop.

### "Out of memory" / system slows to a crawl

You don't have enough free RAM. Llama 8B needs 5+ GB. Close everything,
restart, try again.

If still happening, switch to a smaller model:

```bash
ollama pull llama3.2:3b
```

Then in `.env`:

```ini
OLLAMA_MODEL_FAST=llama3.2:3b
OLLAMA_MODEL_STRONG=llama3.2:3b
```

3B is dumber (you'll see it in the writing quality), but uses only ~2 GB.

### "Could not reach Ollama at http://localhost:11434"

Ollama isn't running. Re-do step 3.

### Job seems frozen at "Summarized 1/700 days"

Probably not frozen — the first call always seems slowest because the
model is being loaded into memory. Wait 60 seconds. If you see progress
move to 2/700, 3/700, etc, you're fine.

If it's been stuck for 5+ minutes, check the terminal where uvicorn is
running for errors. Most common: Ollama returned an error about the
model not being loaded — restart Ollama.

## Phone access without Cloudflare

If you want to upload from your phone but don't want to use Cloudflare
Tunnel, put your phone and laptop on the same WiFi and find your laptop's
local IP:

```bash
# Windows
ipconfig | findstr IPv4

# Mac/Linux
ifconfig | grep "inet "
```

You'll see something like `192.168.1.42`. On your phone, open
`http://192.168.1.42:8000` in any browser. Works on the same network.

You may need to allow incoming connections on port 8000 in your
firewall. The OS will usually prompt you the first time.

## When to switch back to cloud

You should consider re-enabling Groq once any of these is true:

- You have paying customers and 8-hour wait is unacceptable
- You're processing more than 1 book per day
- You've upgraded to a machine with a real GPU (3060+ class)
- You want the higher-quality 70B model output

You can switch by just flipping `USE_OLLAMA=false` and refilling
`GROQ_API_KEY`. No code changes needed.
