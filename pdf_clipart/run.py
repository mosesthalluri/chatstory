"""
Command-line entry point.

Examples:
    # Real generation on the GTX 1650 (SD-Turbo, 2 steps, 512px):
    python -m pdf_clipart input.pdf -o output.pdf

    # SD 1.5 + LCM-LoRA instead:
    python -m pdf_clipart input.pdf --model sd15-lcm --steps 4 --guidance 1.0

    # No GPU — test the whole pipeline with placeholder images:
    python -m pdf_clipart input.pdf --backend stub

    # Use a hosted image API instead of local generation:
    python -m pdf_clipart input.pdf --backend api --api-url https://... --api-key KEY
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .pipeline import annotate_pdf


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_clipart",
        description="Add content-aware AI clipart to a PDF, one page at a time.",
    )
    p.add_argument("input", help="Path to the input PDF")
    p.add_argument("-o", "--output", default=None,
                   help="Output PDF path (default: <input>_clipart.pdf)")

    # Backend
    p.add_argument("--backend", choices=["local", "api", "stub"], default="local")
    p.add_argument("--model", choices=["sd-turbo", "sd15-lcm"], default="sd-turbo")
    p.add_argument("--model-path", dest="model_path", default=None,
                   help="Reuse a local model: a single-file .safetensors/.ckpt "
                        "(A1111/ComfyUI) or a diffusers folder. Skips downloading.")
    p.add_argument("--steps", dest="num_inference_steps", type=int, default=None,
                   help="Inference steps (default 2 for sd-turbo, 4 for sd15-lcm)")
    p.add_argument("--guidance", dest="guidance_scale", type=float, default=None,
                   help="Guidance scale (default 0.0 for sd-turbo, 1.0 for sd15-lcm)")
    p.add_argument("--size", dest="image_size", type=int, default=None,
                   help="Square image size in px (default 512)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false",
                   default=None, help="Disable model CPU offload (needs more VRAM)")

    # API backend
    p.add_argument("--api-url", dest="api_url", default=None)
    p.add_argument("--api-key", dest="api_key", default=None)

    # Decision / generation knobs
    p.add_argument("--max-per-page", dest="max_cliparts_per_page", type=int, default=None)
    p.add_argument("--min-words", dest="min_words", type=int, default=None)
    p.add_argument("--max-numeric-ratio", dest="max_numeric_ratio", type=float, default=None)
    p.add_argument("--style", dest="image_style", default=None,
                   help="Override the style suffix appended to prompts")
    p.add_argument("--allow-themeless", dest="require_visual_theme",
                   action="store_false", default=None,
                   help="Generate even without a recognized visual theme")
    p.add_argument("--keep-images", dest="keep_temp_images", action="store_true",
                   default=None, help="Keep generated PNGs next to the output")
    p.add_argument("--no-theme", dest="theme_pages", action="store_false",
                   default=None, help="Don't tint/theme annotated pages")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else \
        input_path.with_name(f"{input_path.stem}_clipart.pdf")

    config = Config.from_args(args)

    try:
        summary = annotate_pdf(input_path, output_path, config)
    except FileNotFoundError:
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2
    except Exception as exc:  # top-level safety net
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n{summary.annotated_pages}/{summary.total_pages} pages annotated, "
        f"{summary.cliparts_added} cliparts added, "
        f"{summary.skipped_pages} skipped, {summary.failed_pages} failed.\n"
        f"Output: {summary.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
