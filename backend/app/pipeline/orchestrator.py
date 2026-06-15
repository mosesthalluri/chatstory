"""
ChatStory pipeline orchestrator (V2 — faithful mode).

Turns an uploaded chat export into a COMPLETE, immersive book: every non-noise
message preserved, scene-by-scene over the whole chat, with grounded
scene-setting and timestamp footnotes. Outputs an A4 PDF and an editable DOCX.

This deliberately does NOT sample or compress — quality and completeness over
speed. The heavy work is the per-scene narration in pipeline/faithful.py, which
persists incrementally so the UI can show a live tracker + preview and cancel a
long run.
"""

import asyncio
import traceback
from pathlib import Path

from ..services import jobs, pdf_render
from ..settings import OUTPUT_DIR, settings
from . import faithful, book_export


async def run_pipeline(job_id: str, upload_path: Path) -> None:
    """Parse → clean → faithful manuscript (all messages) → PDF + DOCX."""
    try:
        from ..parsers import parse_chat
        from . import normalizer

        phases = [
            {"name": "Reading", "status": "in_progress", "progress": 0},
            {"name": "Writing every scene", "status": "pending", "progress": 0},
            {"name": "Building PDF + DOCX", "status": "pending", "progress": 0},
        ]
        jobs.update(job_id, state="parsing", progress=4,
                    message="Reading your chat…", phases=phases)
        parsed = await asyncio.to_thread(parse_chat, upload_path)
        if not parsed.messages:
            raise ValueError("No messages could be read from this file.")

        # Persist the normalized export so the user can verify what we parsed.
        try:
            norm_dir = OUTPUT_DIR / job_id
            _, norm_txt_path, norm_json_path = normalizer.write_normalized_outputs(
                upload_path, norm_dir)
            jobs.update(
                job_id,
                normalized_txt=str(norm_txt_path.relative_to(OUTPUT_DIR.parent)),
                normalized_json=str(norm_json_path.relative_to(OUTPUT_DIR.parent)),
            )
        except Exception as e:
            print(f"[normalizer] failed to persist: {e}")

        if parsed.message_count > settings.MAX_MESSAGES:
            raise ValueError(
                f"Too many messages ({parsed.message_count}). Maximum is "
                f"{settings.MAX_MESSAGES}.")

        names = parsed.senders or sorted({m.sender for m in parsed.messages})
        title = "Our Story"
        subtitle = " & ".join(names[:2]) if names else "A Conversation"

        phases[0]["status"] = "done"; phases[0]["progress"] = 100
        phases[1]["status"] = "in_progress"
        jobs.update(job_id, state="generating_story", progress=12,
                    message="Cleaning noise and finding scenes…", phases=phases)

        manuscript = await faithful.build_manuscript(
            job_id, parsed, title=title, subtitle=subtitle)

        # Make the stats available to the dashboard/preview immediately.
        jobs.update(job_id, stats=manuscript.get("stats"))

        if manuscript.get("cancelled"):
            # Still render whatever was written so the partial preview is usable.
            await _render_outputs(job_id, manuscript, partial=True)
            jobs.update(job_id, state="cancelled", progress=100,
                        message="Cancelled — your partial book was kept.")
            return

        phases[1]["status"] = "done"; phases[1]["progress"] = 100
        phases[2]["status"] = "in_progress"
        jobs.update(job_id, state="rendering", progress=92,
                    message="Building your PDF and editable DOCX…", phases=phases)
        await _render_outputs(job_id, manuscript)

    except Exception as e:
        tb = traceback.format_exc()
        jobs.update(job_id, state="failed",
                    error=f"{type(e).__name__}: {e}",
                    message="Something went wrong — see error")
        print(f"Pipeline failed for job {job_id}:\n{tb}")


async def _render_outputs(job_id: str, manuscript: dict, *, partial: bool = False) -> None:
    """Render the preview PDF, full PDF, and editable DOCX from the manuscript."""
    out = OUTPUT_DIR / job_id

    preview_html = book_export.render_faithful_html(manuscript, preview=True)
    await pdf_render.render_a4_pdf(preview_html, out / "preview.pdf")

    jobs.update(job_id, progress=95, message="Rendering the full book (PDF)…")
    full_html = book_export.render_faithful_html(manuscript, preview=False)
    await pdf_render.render_a4_pdf(full_html, out / "full.pdf")

    jobs.update(job_id, progress=98, message="Writing the editable DOCX…")
    docx_ok = await asyncio.to_thread(book_export.build_docx, manuscript, out / "full.docx")

    update = dict(
        progress=100,
        preview_pdf=str((out / "preview.pdf").relative_to(OUTPUT_DIR.parent)),
        full_pdf=str((out / "full.pdf").relative_to(OUTPUT_DIR.parent)),
    )
    if not partial:
        update["state"] = "done"
        update["message"] = "Your book is ready (PDF + DOCX)" if docx_ok else "Your book is ready (PDF)"
    jobs.update(job_id, **update)


async def retry_render(job_id: str) -> bool:
    """Re-render PDF + DOCX from the saved manuscript (no LLM re-spend)."""
    manuscript = faithful.load_manuscript(job_id)
    if manuscript is None:
        return False
    try:
        jobs.update(job_id, state="rendering", progress=92,
                    message="Re-rendering your book…", error="")
        await _render_outputs(job_id, manuscript,
                              partial=bool(manuscript.get("cancelled")))
        if manuscript.get("cancelled"):
            jobs.update(job_id, state="cancelled", progress=100,
                        message="Cancelled — your partial book was kept.")
        return True
    except Exception as e:
        tb = traceback.format_exc()
        jobs.update(job_id, state="failed",
                    error=f"{type(e).__name__}: {e}",
                    message="Retry failed — see error")
        print(f"Retry failed for job {job_id}:\n{tb}")
        return True
