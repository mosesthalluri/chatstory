"""
PDF Clipart product runner.

Bridges the standalone top-level `pdf_clipart` package into the web app's
job queue. The heavy imports (PyMuPDF, and torch/diffusers for the local
backend) are done lazily inside the worker thread so the FastAPI server
starts fine even on a box where those aren't installed — a missing dep just
fails the individual job with a clear message instead of crashing startup.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

from ..settings import OUTPUT_DIR, PROJECT_ROOT, settings
from . import jobs


def _run_sync(job_id: str, upload_path: Path, output_path: Path) -> dict:
    """Blocking work: import the pipeline, run it with a progress callback
    that writes job progress. Runs inside asyncio.to_thread."""
    # Make the top-level `pdf_clipart` package importable from the backend.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from pdf_clipart import Config, annotate_pdf

    cfg = Config(
        backend=settings.PDF_CLIPART_BACKEND,
        model=settings.PDF_CLIPART_MODEL,
        num_inference_steps=settings.PDF_CLIPART_STEPS,
        max_cliparts_per_page=settings.PDF_CLIPART_MAX_PER_PAGE,
    )
    # sd-turbo wants guidance 0.0; sd15-lcm wants ~1.0.
    cfg.guidance_scale = 1.0 if settings.PDF_CLIPART_MODEL == "sd15-lcm" else 0.0

    phases = [
        {"name": "Read PDF", "status": "done", "progress": 100},
        {"name": "Generate clipart", "status": "in_progress", "progress": 0},
        {"name": "Write annotated PDF", "status": "pending", "progress": 0},
    ]

    def on_page(idx: int, total: int, result) -> None:
        frac = idx / max(total, 1)
        phases[1]["progress"] = int(100 * frac)
        msg = (
            f"Page {idx}/{total}: added {result.placed} clipart"
            if result.decided and not result.error
            else f"Page {idx}/{total}: skipped ({result.reason})"
        )
        # 10% (read) .. 90% (generation), final 10% for save.
        jobs.update(job_id, state="generating_clipart",
                    progress=10 + int(80 * frac), message=msg, phases=phases)

    summary = annotate_pdf(upload_path, output_path, cfg, progress_callback=on_page)
    return {
        "total_pages": summary.total_pages,
        "annotated_pages": summary.annotated_pages,
        "cliparts_added": summary.cliparts_added,
        "skipped_pages": summary.skipped_pages,
        "failed_pages": summary.failed_pages,
        "seconds": round(summary.seconds, 1),
    }


async def run_pdf_clipart_pipeline(job_id: str, upload_path: Path) -> None:
    try:
        upload_path = Path(upload_path)
        if upload_path.suffix.lower() != ".pdf":
            raise ValueError("Please upload a PDF file.")

        phases = [
            {"name": "Read PDF", "status": "in_progress", "progress": 50},
            {"name": "Generate clipart", "status": "pending", "progress": 0},
            {"name": "Write annotated PDF", "status": "pending", "progress": 0},
        ]
        jobs.update(job_id, state="parsing", progress=8,
                    message="Reading your PDF…", phases=phases)

        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "annotated.pdf"

        summary = await asyncio.to_thread(_run_sync, job_id, upload_path, output_path)

        rel = str(output_path.relative_to(OUTPUT_DIR.parent))
        done_phases = [
            {"name": "Read PDF", "status": "done", "progress": 100},
            {"name": "Generate clipart", "status": "done", "progress": 100},
            {"name": "Write annotated PDF", "status": "done", "progress": 100},
        ]
        jobs.update(
            job_id,
            state="done",
            progress=100,
            message=(
                f"Done — {summary['cliparts_added']} cliparts on "
                f"{summary['annotated_pages']}/{summary['total_pages']} pages"
            ),
            stats=summary,
            preview_pdf=rel,
            full_pdf=rel,
            phases=done_phases,
        )
    except Exception as exc:
        jobs.update(
            job_id,
            state="failed",
            progress=100,
            message="PDF Clipart failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"PDF Clipart failed for job {job_id}:\n{traceback.format_exc()}")
