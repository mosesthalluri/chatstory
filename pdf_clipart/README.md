# pdf_clipart — content-aware AI clipart for PDFs

Point it at a PDF and it adds small, cute, flat clipart to pages that have a
clear visual theme — leaving sparse, numeric, or themeless pages untouched.

Built to run on modest hardware (tested target: **Intel i7 9th gen, 8GB RAM,
NVIDIA GTX 1650 4GB**). It processes **one page at a time** and generates
**one image at a time**, freeing GPU memory after each, so it never blows
past 4GB of VRAM.

## Pipeline (per page)

1. **Read** the page text + the bounding boxes of text/images (PyMuPDF).
2. **Decide** if clipart is warranted — rule-based: skip pages that are too
   sparse, too numeric, or have no recognizable visual theme.
3. **Generate** up to N cliparts from auto-built prompts (page theme +
   style keywords) using Stable Diffusion (SD-Turbo or SD 1.5 + LCM-LoRA).
4. **Place** them in the largest empty squares, never overlapping text.
5. **Output** a new PDF; skipped pages are left unchanged.

## Install

```bash
cd pdf_clipart
python -m venv venv && venv\Scripts\activate        # Windows
# (or: source venv/bin/activate                        on Linux/Mac)

pip install -r requirements.txt

# For local GPU generation, install a CUDA build of torch that matches your
# driver (GTX 1650 works great with cu121):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> First local run downloads the model (~2-5 GB) from Hugging Face and caches
> it. No key needed for SD-Turbo / SD 1.5.

## Run

```bash
# Default: local SD-Turbo, 2 steps, 512px, up to 2 cliparts/page
python -m pdf_clipart input.pdf -o output.pdf

# SD 1.5 + LCM-LoRA (slightly nicer, ~4 steps)
python -m pdf_clipart input.pdf --model sd15-lcm --steps 4 --guidance 1.0

# No GPU? Test the whole pipeline with offline placeholder images:
python -m pdf_clipart input.pdf --backend stub

# Swap local generation for a hosted image API:
python -m pdf_clipart input.pdf --backend api \
    --api-url https://your-image-api/generate --api-key YOUR_KEY
```

You can also call it from Python:

```python
from pdf_clipart import Config, annotate_pdf

annotate_pdf("in.pdf", "out.pdf", Config(backend="stub", max_cliparts_per_page=1))
```

## Useful flags

| Flag | Meaning | Default |
|---|---|---|
| `--backend` | `local` / `api` / `stub` | `local` |
| `--model` | `sd-turbo` / `sd15-lcm` | `sd-turbo` |
| `--steps` | inference steps (keep low!) | 2 / 4 |
| `--guidance` | guidance scale | 0.0 / 1.0 |
| `--size` | square px (use 512 on 4GB) | 512 |
| `--max-per-page` | max cliparts per page | 2 |
| `--min-words` | sparse-page threshold | 25 |
| `--max-numeric-ratio` | digit fraction to skip a page | 0.4 |
| `--allow-themeless` | generate even without a theme | off |
| `--style` | override style suffix | minimal/cute/flat… |
| `--no-cpu-offload` | disable CPU offload (more VRAM) | offload on |
| `--keep-images` | save generated PNGs beside output | off |
| `-v` | verbose logging | off |

## VRAM notes (GTX 1650 / 4GB)

- `enable_model_cpu_offload()` is **on by default** — the single most
  important setting for 4GB. It streams model weights to the GPU per
  submodule instead of holding the whole model resident.
- Attention slicing + VAE slicing are enabled to cap activation memory.
- fp16 is used automatically when CUDA is available.
- Keep `--size 512` and `--steps` low (1-2 for SD-Turbo). 768px will likely
  OOM on 4GB.
- If you still hit OOM, close other GPU apps, or try `--model sd15-lcm`
  with `--steps 4`.

## Configurable / swappable

Everything in [`config.py`](config.py) is tunable: max cliparts per page,
the image style, the decision thresholds, image size, and the backend. To
plug in a different image service, set `--backend api` and adapt the request
/response shape in `ApiBackend.generate` ([`image_backends.py`](image_backends.py)).

## Error handling

Each page is wrapped in try/except: a failure to read, generate, or place
on one page is logged and that page is left unchanged — the run continues
and still produces a valid output PDF.
