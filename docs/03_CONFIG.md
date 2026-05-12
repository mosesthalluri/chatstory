# Configuration — every knob explained

Everything in `backend/.env` controls how ChatBook behaves. This document
walks through each setting and what to change when.

## LLM settings

### `USE_OLLAMA` (true / false)

Decides where chat-to-text generation runs.

- `false` (default): use Groq's free cloud API. Fast, easy, sends
  message snippets to Groq's servers.
- `true`: use local Ollama. Free forever, fully private, ~10x slower on
  consumer hardware.

Switch this any time. No restart needed beyond the next request.

**When to use Ollama:** privacy-conscious users, no internet, building
something for friends/family. **When to use Groq:** anything with paying
users, faster turnaround, less laptop heat.

### `GROQ_API_KEY`

Required when `USE_OLLAMA=false`. Get from https://console.groq.com.
Starts with `gsk_`. No credit card needed — free tier is 14,400 requests/day.

### `GROQ_MODEL_FAST` and `GROQ_MODEL_STRONG`

Two models for two job types:
- **fast** — used for the bulk work (700+ daily summaries per book).
  Default `llama-3.1-8b-instant`. Cheap, plentiful quota.
- **strong** — used for chapter writing and book-arc identification.
  Default `llama-3.3-70b-versatile`. Smarter, lower quota (1000/day on
  free tier).

Other models you can try: `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`,
`gemma2-9b-it`. Check Groq's model list for current options.

### `OLLAMA_MODEL_FAST` and `OLLAMA_MODEL_STRONG`

When using Ollama, these are model tags. Defaults to `llama3.1:8b` for
both — on a GTX 1650 / 8GB RAM, you don't have headroom for anything
bigger. If you have a stronger GPU, try `llama3.1:70b` for strong.

### `OLLAMA_HOST`

Where Ollama is running. Default `http://localhost:11434`. Change only
if you're running Ollama on a different machine.

## Image generation

### `GEMINI_API_KEY`

Get from https://aistudio.google.com/apikey. Free tier: 500 images/day.
If empty, ChatBook uses a generic SVG placeholder for every chapter
illustration — books still work, just less pretty.

### `IMAGE_STYLE`

The style description prepended to every image prompt. Defaults to flat
vector cartoon. Edit to taste:

- Watercolor: `watercolor illustration, soft brushstrokes, paper texture`
- Anime-ish: `anime illustration, cel-shaded, warm lighting`
- Studio Ghibli vibe: `dreamy hand-painted illustration, soft pastels, ghibli style`
- Minimalist: `minimal line art, single color accent, lots of white space`

This is the single biggest visual lever in the product. Experiment.

## Book settings

### `CHAPTERS_PER_BOOK` (default: 8)

How many chapters in a full book. Each chapter = ~1 LLM call + 1 image.
More chapters = more cost + more time + more nuanced narrative.

- 6 chapters: faster, cheaper, feels brief
- 8 chapters: sweet spot
- 12+ chapters: rich but expensive; only worth it for very long chats
  (3+ year relationships)

### `PREVIEW_CHAPTERS` (default: 1)

How many chapters appear in the free preview before the paywall.
Higher = more generous preview = more conversion BUT also less perceived
value. 1-2 is standard for content paywalls.

### `CURRENCY_SYMBOL` and `FULL_BOOK_PRICE`

Cosmetic only — these only show on the paywall page. Real pricing
happens at your payment provider. Defaults to `₹399`. Other reasonable
values:

- `$9.99` for US
- `€8.99` for EU
- `£7.99` for UK
- `₹399` for India

## Storage

### `AUTO_DELETE_AFTER_HOURS` (default: 24)

Hours after job completion before raw uploaded chat data is purged.
The final PDF is kept regardless.

- `24`: standard. Lets users re-trigger if their first preview
  disappointed them.
- `0`: keep raw forever. **Don't do this in production** — it's a
  privacy liability.
- `1`: aggressive. Good for paranoid mode.

Note: v0.1 doesn't actually have a scheduled deletion job yet — you'll
need to run a cleanup script via cron. See `scripts/cleanup.py`.

### `MAX_UPLOAD_SIZE_MB` (default: 200)

Hard limit on upload size. Above this, the upload is rejected with HTTP
413. 200MB covers virtually any text chat export. Bump higher only if
you're accepting media-bundled WhatsApp ZIPs and want to extract text
from inside.

### `MAX_MESSAGES` (default: 300,000)

Hard limit on messages processed. Above this, ChatBook throws an error
asking the user to pick a date range. This protects you from runaway
LLM bills and laptop meltdown.

For reference: a 5-year LDR couple that texts heavily ≈ 200k-400k
messages. A 10-year family group chat ≈ 500k+. The limit is a safety
brake, not an aspiration.

## App settings

### `SECRET_KEY`

A random string used for signing internal stuff. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set it once and forget it. **Never commit your `.env` to git.**

### `DEBUG` (true / false)

When true: verbose logs, stack traces in error responses. When false:
production mode.

Always `false` when you're serving real users. The default `true` is
for development.

## Per-environment configs

Common pattern: have two .env files.

- `.env` — your local laptop config (Ollama, debug on, Groq for testing)
- `.env.prod` — your server config (Groq paid, debug off, real keys)

Switch with `cp .env.prod .env` before deploying.
