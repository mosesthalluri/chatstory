"""
Central configuration for the PDF clipart pipeline.

Every tunable knob lives here so the pipeline code never hardcodes a value.
Construct a Config in code, or build one from CLI args via `Config.from_args`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Style suffix appended to every generated-image prompt. Keeps the look
# consistent: minimal, cute, flat — never heavy or detailed.
DEFAULT_STYLE = (
    "minimal, cute, flat vector clipart, simple, clean, white background, "
    "soft pastel colors, centered, no text"
)

# Things we never want in the clipart. Pushes the model away from busy,
# photographic, or text-heavy output.
DEFAULT_NEGATIVE = (
    "photo, realistic, 3d, detailed, cluttered, busy background, text, "
    "watermark, signature, frame, border, noisy, gradient mesh, low quality"
)


@dataclass
class Config:
    # ---- Decision (whether a page gets clipart at all) ----
    min_words: int = 25
    """Pages with fewer real words than this are treated as too sparse."""

    max_numeric_ratio: float = 0.4
    """If digits make up more than this fraction of characters, skip
    (tables / invoices / number-heavy pages rarely have a visual theme)."""

    require_visual_theme: bool = True
    """If True, a page must contain at least one recognizable visual theme
    keyword to qualify. This is the 'no clear visual theme' guard."""

    min_theme_score: int = 1
    """Minimum number of visual-theme keyword hits required when
    require_visual_theme is True."""

    # ---- Generation ----
    max_cliparts_per_page: int = 2
    image_style: str = DEFAULT_STYLE
    negative_prompt: str = DEFAULT_NEGATIVE
    image_size: int = 512
    """Square render size in px. 512 is the sweet spot for 4GB VRAM."""

    # ---- Backend selection ----
    backend: str = "local"
    """One of: 'local' (Stable Diffusion via diffusers), 'api' (remote HTTP
    image service), 'stub' (offline placeholder — no GPU, for testing)."""

    # ---- Local diffusers backend ----
    model: str = "sd-turbo"
    """'sd-turbo' (1-2 steps, fastest) or 'sd15-lcm' (SD 1.5 + LCM-LoRA)."""
    model_path: str = ""
    """Reuse a model you already have instead of downloading. Either:
      * a single-file checkpoint (.safetensors/.ckpt) from A1111/ComfyUI, or
      * a local diffusers folder (contains model_index.json).
    When set, this overrides `model` for loading. Combine with
    model='sd15-lcm' to apply LCM-LoRA on top of an SD 1.5 checkpoint."""
    num_inference_steps: int = 2
    """Keep tiny on a GTX 1650: 1-2 for sd-turbo, ~4 for sd15-lcm."""
    guidance_scale: float = 0.0
    """0.0 for sd-turbo; ~1.0 for sd15-lcm."""
    seed: int | None = None
    cpu_offload: bool = True
    """enable_model_cpu_offload() — essential on 4GB VRAM."""

    # ---- API backend ----
    api_url: str = ""
    api_key: str = ""
    api_timeout: float = 120.0

    # ---- Placement ----
    page_margin_pt: float = 18.0
    """Keep cliparts at least this far from the page edge."""
    text_padding_pt: float = 8.0
    """Inflate text boxes by this much so clipart never crowds the text."""
    min_clipart_pt: float = 64.0
    """Don't place a clipart if the free area is smaller than this."""
    max_clipart_pt: float = 150.0
    """Cap clipart size so it stays a tasteful accent, not a hero image."""
    grid_cell_pt: float = 12.0
    """Resolution of the free-space scan. Smaller = finer but slower."""

    # ---- Theming ----
    theme_pages: bool = True
    """Apply a mood-based background tint + accent bands to annotated pages
    so the whole page looks themed (cute/dark/romantic/calm/…) instead of
    plain white with stamped clipart. Skipped pages stay untouched."""

    # ---- Output ----
    keep_temp_images: bool = False
    """If True, generated PNGs are kept on disk for inspection."""

    @classmethod
    def from_args(cls, args) -> "Config":
        """Build a Config from an argparse Namespace (see run.py)."""
        cfg = cls()
        for f in cls.__dataclass_fields__:
            val = getattr(args, f, None)
            if val is not None:
                setattr(cfg, f, val)
        # Sensible per-model defaults if the user didn't override them.
        if args.model == "sd15-lcm" and args.num_inference_steps is None:
            cfg.num_inference_steps = 4
        if args.model == "sd15-lcm" and args.guidance_scale is None:
            cfg.guidance_scale = 1.0
        return cfg
