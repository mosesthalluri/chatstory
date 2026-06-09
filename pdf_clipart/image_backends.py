"""
Image-generation backends.

Three interchangeable implementations behind one `ImageBackend` interface:

  * LocalDiffusersBackend — Stable Diffusion via `diffusers`, tuned for a
    4GB GTX 1650 (fp16, model CPU offload, attention/VAE slicing,
    SD-Turbo / SD1.5+LCM at 1-4 steps, 512px).
  * ApiBackend — POST the prompt to a remote image service. Swap local
    generation for a hosted API with `backend="api"`.
  * StubBackend — offline placeholder (PIL only, no torch). Lets the whole
    pipeline run and be tested on a machine with no GPU.

Each `generate(prompt, size)` returns PNG bytes. Backends generate exactly
one image per call so the pipeline never holds more than one image in
memory at a time.
"""

from __future__ import annotations

import gc
import hashlib
import io
import logging
from abc import ABC, abstractmethod

from .config import Config

log = logging.getLogger("pdf_clipart.backend")


class ImageBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, size: int) -> bytes:
        """Return PNG bytes for the prompt, or raise on failure."""

    def close(self) -> None:
        """Release any held resources (GPU memory, sessions)."""


# ---------------------------------------------------------------------------
# Local Stable Diffusion (diffusers)
# ---------------------------------------------------------------------------

class LocalDiffusersBackend(ImageBackend):
    """VRAM-frugal Stable Diffusion. torch/diffusers are imported lazily so
    importing this module costs nothing on a GPU-less box."""

    def __init__(self, config: Config):
        self.config = config
        self._pipe = None
        self._torch = None

    def _load(self) -> None:
        if self._pipe is not None:
            return

        import os
        import torch  # lazy — heavy import
        from diffusers import AutoPipelineForText2Image

        self._torch = torch
        has_cuda = torch.cuda.is_available()
        dtype = torch.float16 if has_cuda else torch.float32
        cfg = self.config

        is_lcm = cfg.model == "sd15-lcm"

        if cfg.model_path:
            # --- Reuse a model the user already has on disk ---
            path = os.path.expanduser(cfg.model_path)
            if path.lower().endswith((".safetensors", ".ckpt")):
                # A1111 / ComfyUI single-file checkpoint.
                from diffusers import StableDiffusionPipeline
                log.info("Loading single-file checkpoint: %s", path)
                pipe = StableDiffusionPipeline.from_single_file(
                    path, torch_dtype=dtype, safety_checker=None,
                )
            else:
                # A local diffusers folder (has model_index.json).
                log.info("Loading local diffusers model: %s", path)
                pipe = AutoPipelineForText2Image.from_pretrained(
                    path, torch_dtype=dtype, safety_checker=None,
                )
        elif is_lcm:
            model_id = "runwayml/stable-diffusion-v1-5"
            log.info("Loading SD 1.5 + LCM-LoRA (%s)…", model_id)
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=dtype, safety_checker=None,
            )
        else:  # default: sd-turbo
            model_id = "stabilityai/sd-turbo"
            log.info("Loading SD-Turbo (%s)…", model_id)
            # Prefer the fp16 weight variant — it's ~half the download/size.
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, torch_dtype=dtype, variant="fp16", safety_checker=None,
                )
            except Exception:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, torch_dtype=dtype, safety_checker=None,
                )

        # Apply LCM-LoRA on top of any SD 1.5 base (hub or local checkpoint)
        # when the user asked for sd15-lcm. This is what makes 4-step
        # generation work with a reused SD 1.5 checkpoint.
        if is_lcm:
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            pipe.fuse_lora()

        # --- 4GB VRAM survival kit ---
        # Offload weights to CPU and stream them to GPU per-submodule. This
        # is the single most important setting for a GTX 1650.
        if has_cuda and cfg.cpu_offload:
            pipe.enable_model_cpu_offload()
        elif has_cuda:
            pipe.to("cuda")
        # Slice attention + VAE so peak activation memory stays small.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        self._pipe = pipe
        log.info("Model ready (cuda=%s, dtype=%s).", has_cuda, dtype)

    def generate(self, prompt: str, size: int) -> bytes:
        self._load()
        cfg = self.config
        torch = self._torch

        generator = None
        if cfg.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(cfg.seed)

        # sd-turbo expects guidance_scale=0.0; LCM works best ~1.0.
        result = self._pipe(
            prompt=prompt,
            negative_prompt=cfg.negative_prompt or None,
            num_inference_steps=max(1, cfg.num_inference_steps),
            guidance_scale=cfg.guidance_scale,
            height=size,
            width=size,
            generator=generator,
        )
        image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        # Free per-image memory aggressively.
        del result, image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return buf.getvalue()

    def close(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            if self._torch and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Remote API backend
# ---------------------------------------------------------------------------

class ApiBackend(ImageBackend):
    """Generic HTTP image API. Expects a JSON endpoint that accepts a prompt
    and returns either raw image bytes or base64 in a JSON field. Adapt the
    request/response shape to your provider in `generate`."""

    def __init__(self, config: Config):
        if not config.api_url:
            raise ValueError("backend='api' requires --api-url")
        self.config = config

    def generate(self, prompt: str, size: int) -> bytes:
        import base64
        import httpx

        cfg = self.config
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        payload = {
            "prompt": prompt,
            "negative_prompt": cfg.negative_prompt,
            "width": size,
            "height": size,
            "steps": cfg.num_inference_steps,
        }
        with httpx.Client(timeout=cfg.api_timeout) as client:
            resp = client.post(cfg.api_url, json=payload, headers=headers)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if ctype.startswith("image/"):
                return resp.content
            data = resp.json()
            # Try a few common shapes.
            b64 = (
                data.get("image")
                or data.get("b64_json")
                or (data.get("data", [{}])[0].get("b64_json"))
            )
            if not b64:
                raise ValueError(f"Unrecognized API response shape: {list(data)[:6]}")
            return base64.b64decode(b64)


# ---------------------------------------------------------------------------
# Offline stub (no GPU) — for testing the pipeline end to end
# ---------------------------------------------------------------------------

class StubBackend(ImageBackend):
    """Deterministic placeholder image built with PIL. Draws a soft rounded
    card with a colour derived from the prompt, so you can verify decision,
    placement and PDF embedding without any model."""

    def generate(self, prompt: str, size: int) -> bytes:
        from PIL import Image, ImageDraw

        # Stable colour per subject so repeated runs look the same.
        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        r, g, b = (int(digest[i:i + 2], 16) for i in (0, 2, 4))
        # Pastel-ify.
        r, g, b = (128 + r // 2, 128 + g // 2, 128 + b // 2)

        img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        pad = size // 8
        draw.rounded_rectangle(
            [pad, pad, size - pad, size - pad],
            radius=size // 8, fill=(r, g, b, 255),
            outline=(90, 90, 90, 255), width=max(2, size // 128),
        )
        # A simple centered glyph (first letters of the subject).
        subject = prompt.split(",")[0].strip()[:2].upper() or "?"
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), subject, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((size - tw) / 2, (size - th) / 2 - bbox[1]),
                      subject, fill=(40, 40, 40, 255), font=font)
        except Exception:
            pass

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def make_backend(config: Config) -> ImageBackend:
    """Factory: pick a backend from config.backend."""
    if config.backend == "local":
        return LocalDiffusersBackend(config)
    if config.backend == "api":
        return ApiBackend(config)
    if config.backend == "stub":
        return StubBackend()
    raise ValueError(f"Unknown backend: {config.backend!r}")
