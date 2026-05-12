# ChatBook — Turn chats into illustrated storybooks

A self-hostable web app that takes WhatsApp / Instagram / Telegram chat exports
and turns them into illustrated PDF storybooks with a Spotify-Wrapped-style
stats page. Free preview, paid full book.

> **Status:** v0.1 — runnable starter. Built to be simple enough for a
> first-time web developer to operate, structured enough to grow.

## What's in here

- `backend/` — FastAPI server, the parsing + pipeline + PDF generation
- `frontend/` — Mobile-friendly HTML/JS UI (no framework, no build step)
- `docs/` — Setup guide, architecture notes, troubleshooting
- `scripts/` — Helper scripts for setup and maintenance

## Get started

If you've never run a web app before, open `docs/01_SETUP.md` and follow it
top to bottom. It assumes nothing.

If you already know your way around Python and want the short version:

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser. Or get the public URL with
Cloudflare Tunnel — see `docs/04_GO_LIVE.md`.

## Documentation

| File | What it covers |
|---|---|
| `docs/01_SETUP.md` | Install everything from scratch (Windows / Mac / Linux) |
| `docs/02_HOW_IT_WORKS.md` | The pipeline, why it's designed this way |
| `docs/03_CONFIG.md` | Every config knob, every API key |
| `docs/04_GO_LIVE.md` | Make your laptop reachable from the internet |
| `docs/05_TROUBLESHOOTING.md` | Common errors and fixes |
| `docs/06_RECREATION_PROMPT.md` | Use another LLM to rebuild this project |

## What works in v0.1

- Parses WhatsApp `.txt` exports (most localizations)
- Parses Instagram JSON exports (including ZIPs)
- Parses Telegram JSON exports
- Generic `.txt` fallback parser (best-effort)
- Aggressive noise filtering (reactions, system messages, media placeholders)
- Hierarchical summarization (day → week → month → chapter)
- Stats engine (Spotify-Wrapped style)
- Free preview PDF (4 pages, watermarked)
- Full PDF generation behind a paywall stub
- Mobile-responsive web UI
- Background job queue (in-process, suitable for low traffic)

## What's deliberately NOT in v0.1

- Real payment integration (there's a stub — wire your own Stripe/Razorpay)
- User accounts (job IDs are unguessable URLs — good enough for v0.1)
- Database (SQLite file per job — sufficient for <1000 books/day)
- Docker (added in v0.2 if you need it)
- Distributed workers (single-machine for now)

## License

Yours to use, modify, and sell. Provided as-is, no warranty.
