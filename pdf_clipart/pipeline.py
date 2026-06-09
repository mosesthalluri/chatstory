"""
Pipeline orchestrator.

Walks a PDF one page at a time:
    read text -> decide -> generate (1 image at a time) -> place -> next page

Memory discipline for the GTX 1650 / 8GB RAM box:
  * Only one page's content is held at a time.
  * Images are generated and embedded one by one; we never build a list of
    all images for the whole document.
  * The image backend frees GPU memory after every single generation.

A failure on one page is logged and skipped — it never aborts the run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import fitz

from .config import Config
from . import content_reader, decision
from .image_backends import ImageBackend, make_backend

log = logging.getLogger("pdf_clipart")


@dataclass
class PageResult:
    index: int
    decided: bool
    reason: str
    placed: int = 0
    error: str | None = None


@dataclass
class RunSummary:
    total_pages: int
    annotated_pages: int
    cliparts_added: int
    skipped_pages: int
    failed_pages: int
    seconds: float
    output_path: str
    pages: list[PageResult]


def annotate_pdf(
    input_path: str | Path,
    output_path: str | Path,
    config: Config | None = None,
    backend: ImageBackend | None = None,
    progress_callback: Optional[Callable[[int, int, "PageResult"], None]] = None,
) -> RunSummary:
    """Produce a new PDF with content-aware clipart added.

    Skipped/unwarranted pages are left byte-for-byte unchanged.

    `progress_callback(page_index, total_pages, result)` is invoked after
    each page (1-based index) so callers — e.g. a web job — can report
    progress. Exceptions from the callback are swallowed so they never
    abort the run.
    """
    config = config or Config()
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    own_backend = backend is None
    backend = backend or make_backend(config)

    start = time.time()
    results: list[PageResult] = []
    cliparts_added = 0
    annotated_pages = 0
    failed = 0

    def notify(idx: int, tot: int, res: "PageResult") -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(idx, tot, res)
        except Exception:
            pass  # progress reporting must never break the run

    doc = fitz.open(input_path)
    total = doc.page_count
    log.info("Opened %s (%d pages). Backend=%s, model=%s, max/page=%d",
             input_path.name, total, config.backend, config.model,
             config.max_cliparts_per_page)

    try:
        for i in range(total):
            page = doc[i]
            try:
                content = content_reader.read_page(page, i)
                want, reason, hits = decision.should_generate(content.text, config)

                if not want:
                    log.info("[page %d/%d] skip — %s", i + 1, total, reason)
                    results.append(PageResult(i, False, reason))
                    notify(i + 1, total, results[-1])
                    continue

                from .mood import detect_mood
                mood = detect_mood(content.text)
                prompts = decision.build_prompts(content.text, hits, config, mood.prompt_modifier)
                log.info("[page %d/%d] generate — %s, mood=%s (%d clipart)",
                         i + 1, total, reason, mood.name, len(prompts))

                # Generate + collect PNGs ONE AT A TIME. We keep at most the
                # prompts list and the small PNG bytes for this single page.
                pngs: list[bytes] = []
                for n, prompt in enumerate(prompts, 1):
                    try:
                        log.info("    img %d/%d: %s", n, len(prompts), prompt[:60])
                        png = backend.generate(prompt, config.image_size)
                        pngs.append(png)
                        if config.keep_temp_images:
                            tmp = output_path.with_name(
                                f"{output_path.stem}_p{i + 1}_{n}.png")
                            tmp.write_bytes(png)
                    except Exception as exc:
                        log.warning("    img %d/%d failed: %s — skipping image",
                                    n, len(prompts), exc)

                from . import placement
                # Theme the page first (background wash + accent bands), then
                # lay the clipart on top.
                placement.apply_theme(page, content, mood, config)
                placed = placement.place_cliparts(page, content, pngs, config)
                cliparts_added += placed
                if placed or config.theme_pages:
                    annotated_pages += 1
                results.append(PageResult(i, True, reason, placed=placed))
                notify(i + 1, total, results[-1])
                # Drop page-local image bytes promptly.
                del pngs

            except Exception as exc:
                failed += 1
                log.exception("[page %d/%d] FAILED: %s — leaving page unchanged",
                              i + 1, total, exc)
                results.append(PageResult(i, False, "error", error=str(exc)))
                notify(i + 1, total, results[-1])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # garbage_collect + deflate keeps the output size reasonable.
        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()
        if own_backend:
            backend.close()

    elapsed = time.time() - start
    skipped = sum(1 for r in results if not r.decided and not r.error)
    summary = RunSummary(
        total_pages=total,
        annotated_pages=annotated_pages,
        cliparts_added=cliparts_added,
        skipped_pages=skipped,
        failed_pages=failed,
        seconds=elapsed,
        output_path=str(output_path),
        pages=results,
    )
    log.info(
        "Done in %.1fs — %d/%d pages annotated, %d cliparts, %d skipped, %d failed -> %s",
        elapsed, annotated_pages, total, cliparts_added, skipped, failed, output_path,
    )
    return summary
