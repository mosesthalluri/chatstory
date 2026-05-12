# Setup — From zero to running

This guide assumes you've never set up a web app before. We'll go slowly.

## What you need on your machine

- **A computer** — Windows, Mac, or Linux. The i7 / 8GB / GTX 1650 spec is fine.
- **About 5GB of free disk space** — Python + dependencies + Ollama model + room for jobs.
- **Internet connection** — for installing things and calling the AI APIs.

## Step 1: Install Python 3.11 or newer

ChatBook needs Python 3.11+. Older versions will not work.

### On Windows

1. Go to https://www.python.org/downloads/
2. Download the latest Python 3.11 or 3.12 installer (NOT 3.13 — some libraries lag).
3. **Important:** check "Add Python to PATH" on the first install screen.
4. Click "Install Now."
5. Open Command Prompt and run `python --version`. You should see `Python 3.11.x` or `3.12.x`.

If `python --version` says "command not found," reinstall and make sure the
PATH checkbox is checked.

### On Mac

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.12
```

### On Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

## Step 2: Get the project files

If you got this as a ZIP, extract it. You should have a `chatbook/` folder
with `backend/`, `frontend/`, `docs/`, and `scripts/` inside.

Open a terminal and `cd` into that folder:

```bash
cd path/to/chatbook
```

## Step 3: Set up a virtual environment

A virtual environment is a sandbox so ChatBook's libraries don't conflict
with anything else on your computer. Always do this.

```bash
cd backend
python -m venv venv
```

Then activate it:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll know it worked when your terminal prompt starts with `(venv)`.

You'll need to do this `activate` step every time you open a new terminal
to work on ChatBook. The install (next step) only needs to happen once.

## Step 4: Install Python dependencies

```bash
pip install -r requirements.txt
```

This downloads about 200MB and takes 2-5 minutes. If it fails, check
`docs/05_TROUBLESHOOTING.md`.

## Step 5: Install Playwright browsers

ChatBook uses Playwright to render PDFs. After pip install, run:

```bash
playwright install chromium
```

This downloads another ~200MB. Only needs to happen once.

## Step 6: Get your API keys (free)

ChatBook uses two cloud AI services. Both have generous free tiers and
require no credit card.

### Groq (for the LLM that writes your books)

1. Go to https://console.groq.com
2. Sign up with your email (no credit card needed).
3. Click "API Keys" in the sidebar.
4. Click "Create API Key", give it any name, copy the key.
5. The key starts with `gsk_...` — save it somewhere.

### Google Gemini (for image generation)

1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account.
3. Click "Create API key."
4. Copy the key — starts with `AIza...`.

## Step 7: Configure ChatBook

In the `backend/` folder, copy the example config:

```bash
# Mac/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Open `.env` in any text editor (Notepad is fine). You'll see something like:

```
GROQ_API_KEY=
GEMINI_API_KEY=
USE_OLLAMA=false
```

Paste your keys after the `=` signs:

```
GROQ_API_KEY=gsk_your_actual_key_here
GEMINI_API_KEY=AIza_your_actual_key_here
USE_OLLAMA=false
```

Save the file. Don't add quotes around the keys.

## Step 8: (Optional) Set up Ollama for free local LLM

If you'd rather run the LLM locally on your machine and not use Groq's
free tier, install Ollama. This is slower but completely free and private.

1. Download from https://ollama.com/download
2. Install it (just click through the installer).
3. Open a terminal and run:
   ```bash
   ollama pull llama3.1:8b
   ```
   This downloads ~5GB. Takes 10-30 minutes.
4. Verify it works: `ollama run llama3.1:8b "say hi"`. If it responds, you're set.
5. In your `.env`, set `USE_OLLAMA=true`.

When `USE_OLLAMA=true`, ChatBook uses your local Ollama and ignores Groq.
You can flip this any time depending on whether you want speed or privacy.

## Step 9: Run ChatBook

From the `backend/` folder, with your virtual environment still active:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Open http://localhost:8000 in your browser. You should see ChatBook's
upload page.

## Step 10: Test it

1. Drag a chat export onto the upload area.
2. Watch the progress page.
3. After a few minutes (or longer for huge files), you'll get a free
   preview PDF.

If you want to test without your own chat, there's a sample in
`backend/storage/sample_chat.txt`.

## Going further

- Open it on your phone: see `docs/04_GO_LIVE.md` for Cloudflare Tunnel
- Configure pricing, watermarks, page count: see `docs/03_CONFIG.md`
- Stuck? Read `docs/05_TROUBLESHOOTING.md`
