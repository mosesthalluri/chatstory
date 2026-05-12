"""
Pipeline orchestrator. The master function that takes an uploaded chat
file and produces preview + full PDFs.

Run end-to-end inside a single async task. For v0.1 we run jobs in-process
with FastAPI's BackgroundTasks. Move to a real queue (RQ, Celery) when you
have concurrent users.
"""

import json
import traceback
from dataclasses import asdict
from pathlib import Path

from .. import models
from ..parsers import parse_chat
from ..pipeline import stats, chunker, summarize, chapter_gen
from ..services import image_gen, pdf_render, jobs
from ..settings import OUTPUT_DIR, settings


def _checkpoint_path(job_id: str) -> Path:
    return OUTPUT_DIR / job_id / "checkpoint.json"


def _save_checkpoint(job_id: str, payload: dict) -> None:
    """Persist everything needed to re-render the PDF without re-spending
    API calls. Written right before the PDF step so a render crash doesn't
    waste hours of LLM work."""
    path = _checkpoint_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_checkpoint(job_id: str) -> dict | None:
    path = _checkpoint_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def run_pipeline(job_id: str, upload_path: Path) -> None:
    """End-to-end pipeline. Updates job status as it goes."""
    try:
        # 1. Parse
        jobs.update(job_id, state="parsing", progress=5, message="Reading chat file…")
        parsed = parse_chat(upload_path)

        if parsed.message_count > settings.MAX_MESSAGES:
            raise ValueError(
                f"Too many messages ({parsed.message_count}). "
                f"Maximum is {settings.MAX_MESSAGES}. Please upload a "
                f"shorter date range."
            )

        # 2. Stats
        jobs.update(job_id, state="analyzing", progress=15, message="Computing stats…")
        chat_stats = stats.compute_stats(parsed)
        jobs.update(job_id, stats=chat_stats)

        # 3. Day digests
        active_days = sum(
            1 for _, msgs in chunker.by_day(parsed.messages).items()
            if len(msgs) >= 5
        )
        jobs.update(
            job_id, state="analyzing", progress=20,
            message=f"Summarizing {active_days} active days…"
        )

        def day_progress(done, total):
            # Day summarization spans progress 20 → 60
            pct = 20 + int(40 * done / max(total, 1))
            jobs.update(job_id, progress=pct, message=f"Summarized {done}/{total} days")

        day_summaries = await summarize.summarize_all_days(
            parsed.messages, on_progress=day_progress
        )

        # 4. Week digests
        jobs.update(job_id, progress=62, message="Rolling up to weeks…")
        week_summaries = await summarize.summarize_weeks(day_summaries, parsed.messages)

        # 5. Month digests
        jobs.update(job_id, progress=70, message="Rolling up to months…")
        month_summaries = await summarize.summarize_months(week_summaries, parsed.messages)

        # 6. Book arc
        jobs.update(job_id, progress=75, message="Identifying narrative arcs…")
        arc = await summarize.identify_arc(month_summaries, chat_stats["days_span"])

        # 7. Chapter generation
        jobs.update(job_id, state="writing", progress=78, message="Writing chapters…")
        # CHAPTERS_PER_BOOK=0 means "auto" — pick based on chat length.
        # Any positive value forces that exact count.
        if settings.CHAPTERS_PER_BOOK and settings.CHAPTERS_PER_BOOK > 0:
            n_chapters = settings.CHAPTERS_PER_BOOK
        else:
            n_chapters = chunker.suggest_chapter_count(parsed.messages)
        print(f"[orchestrator] using {n_chapters} chapters "
              f"({'auto' if not settings.CHAPTERS_PER_BOOK else 'fixed'})")
        chapter_chunks = chunker.into_chapters(parsed.messages, n_chapters)
        chapters: list[chapter_gen.Chapter] = []
        for i, (start, end, msgs) in enumerate(chapter_chunks, 1):
            ch = await chapter_gen.generate_chapter(
                index=i, start_date=start, end_date=end,
                chapter_messages=msgs,
                month_summaries=month_summaries,
                arc_context=arc,
            )
            chapters.append(ch)
            jobs.update(
                job_id,
                progress=78 + int(10 * i / len(chapter_chunks)),
                message=f"Wrote chapter {i}/{len(chapter_chunks)}",
            )

        # 8. Image picking (cliparts by default, Gemini if configured)
        jobs.update(job_id, state="rendering", progress=88, message="Picking illustrations…")
        image_dir = OUTPUT_DIR / job_id / "images"
        chapter_images, image_stats = await image_gen.generate_images_for_chapters(
            [(ch.index, ch.illustration_prompt) for ch in chapters],
            image_dir,
        )
        # Cover gets a fresh clipart not used by any chapter, when possible
        used_so_far = {p.name for p in chapter_images.values()}
        cover_image_path = await image_gen.generate_image(
            "two chat bubbles, sparkles, warm and cozy",
            image_dir / "cover",
            stats=image_stats,
            avoid=used_so_far,
        )
        # Log summary; only surface to user if something actually went wrong
        print(f"[orchestrator] image summary: {image_stats.summary()}")
        if image_stats.fell_back > 0 and "rate_limited" in image_stats.fallback_reasons:
            jobs.update(
                job_id,
                message=(
                    f"Some images used clipart fallback ({image_stats.summary()}). "
                    f"Gemini rate-limited — try again later for a fully AI-illustrated book."
                ),
            )

        # 9. Save checkpoint — everything needed to re-render the PDF
        #    without re-spending API calls. Critical: PDF render can fail
        #    for reasons unrelated to the LLM/image work (Playwright/asyncio
        #    on Windows, disk full, etc).
        chapters_dict = [
            {
                "index": ch.index,
                "title": ch.title,
                "when": ch.when,
                "body": ch.body,
                "pull_quote": ch.pull_quote,
                "pull_quote_author": ch.pull_quote_author,
                "illustration_prompt": ch.illustration_prompt,
            }
            for ch in chapters
        ]
        _save_checkpoint(job_id, {
            "stats": chat_stats,
            "chapters": chapters_dict,
            "chapter_images": {str(k): str(v) for k, v in chapter_images.items()},
            "cover_image": str(cover_image_path) if cover_image_path else None,
        })

        # 10. Render PDFs
        jobs.update(job_id, state="rendering", progress=94, message="Rendering preview PDF…")
        await _render_pdfs_from_data(
            job_id=job_id,
            stats_data=chat_stats,
            chapters=chapters,
            chapter_images=chapter_images,
            cover_image_path=cover_image_path,
        )

    except Exception as e:
        tb = traceback.format_exc()
        jobs.update(
            job_id, state="error",
            error=f"{type(e).__name__}: {e}",
            message="Something went wrong — see error",
        )
        print(f"Pipeline failed for job {job_id}:\n{tb}")


async def _render_pdfs_from_data(
    *,
    job_id: str,
    stats_data: dict,
    chapters: list[chapter_gen.Chapter],
    chapter_images: dict[int, Path],
    cover_image_path: Path | None,
) -> None:
    """Render preview + full PDFs. Extracted so it can be called both from
    the normal pipeline AND from retry_render() when only the PDF step failed.
    """
    names = stats_data.get("senders", ["A", "B"])
    title = "Our Story"
    subtitle = " & ".join(names[:2])
    first = stats_data["first_message_date"][:10]
    last = stats_data["last_message_date"][:10]
    date_range = f"{first} — {last}"

    preview_html = pdf_render.render_book_html(
        title=title, subtitle=subtitle, date_range=date_range,
        stats=stats_data, chapters=chapters,
        chapter_images=chapter_images,
        is_preview=True,
        preview_chapter_count=settings.PREVIEW_CHAPTERS,
        cover_image=cover_image_path,
    )
    preview_path = OUTPUT_DIR / job_id / "preview.pdf"
    await pdf_render.render_html_to_pdf(preview_html, preview_path)

    jobs.update(job_id, progress=97, message="Rendering full PDF…")
    full_html = pdf_render.render_book_html(
        title=title, subtitle=subtitle, date_range=date_range,
        stats=stats_data, chapters=chapters,
        chapter_images=chapter_images,
        is_preview=False,
        preview_chapter_count=settings.PREVIEW_CHAPTERS,
        cover_image=cover_image_path,
    )
    full_path = OUTPUT_DIR / job_id / "full.pdf"
    await pdf_render.render_html_to_pdf(full_html, full_path)

    jobs.update(
        job_id, state="done", progress=100,
        message="Your book is ready",
        preview_pdf=str(preview_path.relative_to(OUTPUT_DIR.parent)),
        full_pdf=str(full_path.relative_to(OUTPUT_DIR.parent)),
    )


async def retry_render(job_id: str) -> bool:
    """Re-run just the PDF render step using the saved checkpoint.
    Returns True if a checkpoint was found and render was attempted.
    Use this when a job died in the rendering stage — no API calls re-spent.
    """
    cp = _load_checkpoint(job_id)
    if cp is None:
        return False

    try:
        # Reconstruct Chapter namedtuples from the dict
        chapters = [
            chapter_gen.Chapter(
                index=c["index"],
                title=c["title"],
                when=c["when"],
                body=c["body"],
                pull_quote=c.get("pull_quote", c.get("quote", "")),
                pull_quote_author=c.get("pull_quote_author", c.get("quote_by", "")),
                illustration_prompt=c["illustration_prompt"],
            )
            for c in cp["chapters"]
        ]
        chapter_images = {int(k): Path(v) for k, v in cp["chapter_images"].items()}
        cover_image_path = Path(cp["cover_image"]) if cp.get("cover_image") else None

        jobs.update(job_id, state="rendering", progress=94,
                    message="Re-rendering preview PDF…", error="")
        await _render_pdfs_from_data(
            job_id=job_id,
            stats_data=cp["stats"],
            chapters=chapters,
            chapter_images=chapter_images,
            cover_image_path=cover_image_path,
        )
        return True
    except Exception as e:
        tb = traceback.format_exc()
        jobs.update(
            job_id, state="error",
            error=f"{type(e).__name__}: {e}",
            message="Retry failed — see error",
        )
        print(f"Retry failed for job {job_id}:\n{tb}")
        return True
