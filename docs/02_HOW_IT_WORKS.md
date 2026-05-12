# How ChatBook works

A conceptual tour of the pipeline, why it's built this way, and what to
change first when you want to improve it.

## The big picture

```
   ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
   │ Upload  │───>│ Parse + │───>│  Stats   │───>│ Pipeline │
   │ chat    │    │ Clean   │    │ engine   │    │  (LLM)   │
   └─────────┘    └─────────┘    └──────────┘    └──────────┘
                                                       │
   ┌─────────┐    ┌─────────┐    ┌──────────┐         │
   │   PDF   │<───│  Image  │<───│ Narrative│<────────┘
   │ render  │    │  gen    │    │ chapters │
   └─────────┘    └─────────┘    └──────────┘
```

## Stage by stage

### 1. Upload

The user drops a file. Could be:

- WhatsApp `.txt` export (most common)
- WhatsApp `.zip` export (text + media — we ignore media for now)
- Instagram JSON export (one file or zipped)
- Telegram JSON export
- A plain `.txt` file with no recognized format (best-effort)

The file goes into `backend/storage/uploads/{job_id}/`. We never delete
uploaded chat data automatically — see Privacy below.

### 2. Parse + Clean

This is where most of the engineering goes, and where most projects fail.

Each parser knows the structure of one format. The job:

1. Detect the format (sniff the first ~100 lines).
2. Route to the right parser.
3. Convert to a normalized internal format: a list of `Message` objects
   with `(sender, timestamp, text, kind)`.
4. Strip noise: reactions, "user added X", system messages, "this message
   was deleted", media-only entries (kept as `[shared media]` placeholders),
   reply quote duplicates.
5. Merge consecutive messages from the same sender within 30 seconds
   (treat as one logical "turn").

The output is a Python list. **Everything downstream only sees the
normalized list.** This is the most important architectural decision in
the codebase: it means adding a new format is a single new parser file,
not changes throughout the system.

### 3. Stats engine (no LLM, deterministic)

Pure Python on the normalized message list. Computes:

- Total messages, days span, days active
- Per-person message counts and median message length
- Hour-of-day distribution → "most active hour"
- Day-of-week distribution
- Response time distribution
- Top emojis (per person and combined)
- Recurring phrases (n-grams used 3+ times that aren't common English)
   — these are your "inside jokes"
- Sentiment trajectory (using a small classifier, not a generative LLM)
- Identified "narrative moments" — days with abnormally high activity,
  long unbroken conversations, or sentiment spikes

Numbers can't hallucinate. Everything that goes on the Wrapped page comes
from here.

### 4. Hierarchical summarization (the LLM, on a tight leash)

Real chats are 100k+ messages. You can't feed that to an LLM. Instead:

**Layer 1 — Day digests.** Active days (>20 messages) get a 100-word
summary from the LLM. Each day's prompt is small (~3k tokens), runs in
parallel, costs ~0.001 USD per day on Groq paid, free on Ollama.

**Layer 2 — Week digests.** Take 7 day-digests, summarize the week. ~300
tokens out per week.

**Layer 3 — Month digests.** Take 4 week-digests, summarize the month.
~600 tokens out.

**Layer 4 — Book arc.** Read all month digests once, identify the major
narrative threads of the whole relationship.

**Layer 5 — Chapter generation.** For each chapter, the LLM receives:
the book arc + the relevant month digests + the relevant week digests +
20-30 verbatim "highlight" messages picked deterministically by the stats
engine. It writes a 200-400 word chapter.

This compression is what makes the project tractable. A 200,000-message
chat collapses to ~30k tokens of structured context by the time the
chapter LLM call happens.

### 5. Verification (the trust layer)

After each chapter is written, a second LLM call checks:

> "Here is a chapter. Here are the source messages. List every factual
> claim in the chapter. For each claim, point to the supporting message.
> If any claim has no support, flag it."

Flagged chapters get regenerated. This costs ~0.001 USD per chapter and
catches the bulk of hallucinations.

### 6. Image generation

For each chapter, we generate one scene illustration via Gemini's free
tier (Nano Banana / Imagen). The prompt is constructed from the chapter's
themes — never from raw chat content (privacy and consistency reasons).

Style is a flat-vector cartoon style by default. Easy to swap by editing
`backend/app/services/image_gen.py`.

### 7. PDF render

We generate an HTML version of the book (cover, stats page, chapter
pages, paywall page if preview), then use Playwright to print HTML to PDF.
This gives us proper typography and CSS-grade layout — far better than
generating PDFs from Python directly.

Two PDFs come out:

- `preview.pdf` — 4 pages, free, watermarked. Cover + Wrapped + Chapter 1
  + paywall.
- `full.pdf` — complete book, no watermark. Held in storage until the
  payment webhook fires.

## Why this architecture

**Separation of stats from narrative.** Stats are deterministic and free;
narrative is LLM-generated and costs money. By splitting them, we can
ship the Wrapped page free as a viral acquisition tool and gate the book
behind payment.

**Hierarchical summarization, not single-pass.** Single-pass against a
huge context window produces vague averaged-out narratives. Multi-pass
preserves local detail at every level.

**HTML → PDF, not direct PDF generation.** Designers can iterate on
templates without touching Python. Web devs can contribute. CSS layout
beats every Python PDF library.

**Single-machine in v0.1.** Adding queues, workers, and databases adds
weeks of engineering and a long bug tail. Don't do it until you have
paying users telling you the wait is too long.

## What to change first when you want to improve quality

In rough order of impact per hour of effort:

1. **Edit the chapter generation prompt** in `backend/app/pipeline/chapter_gen.py`.
   This is the single biggest quality lever. Small wording changes shift
   tone dramatically.

2. **Improve the highlight selection** in `backend/app/pipeline/highlights.py`.
   The chapter is only as good as the messages it sees. Better selection
   = better book.

3. **Add more parsers** in `backend/app/parsers/`. Every format you
   support unlocks a new user segment.

4. **Improve the noise filter** in each parser. Real exports have weird
   edge cases. Watch for false positives in your dropped-messages log.

5. **Edit the HTML template** in `frontend/templates/book.html`. The look
   of the PDF is 100% controlled here.

## Privacy

By default ChatBook is self-hosted, so chat data only ever lives on your
machine. But if you deploy it for users, you're handling intimate data
about people who never consented to your service. Take this seriously:

- Set `AUTO_DELETE_AFTER_HOURS` in `.env` (default 24). After delivery,
  raw chat data is purged. Keep only derived stats and the final PDF.
- Never log chat content. Logs in this codebase deliberately log only
  message *counts* and timing.
- Never train models on user data.
- Have a privacy policy reviewed by an actual lawyer before launching
  publicly. This is non-optional.
