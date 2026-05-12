"""
PDF rendering. We render the book as HTML (via Jinja templates), then
print to PDF using headless Chromium.

This separation matters: a designer can iterate on the HTML/CSS without
touching Python. CSS print layout beats every Python PDF library.

IMPORTANT — Windows asyncio fix:
We use Playwright's SYNC API inside a worker thread (via asyncio.to_thread)
rather than the async API directly. Reason: Playwright's async API needs
asyncio subprocess support, which requires ProactorEventLoop on Windows.
But uvicorn --reload often uses SelectorEventLoop, which raises
NotImplementedError when starting subprocesses. Running sync_playwright in
a thread sidesteps the event loop entirely and works identically on
Windows, macOS, and Linux.
"""

import asyncio
import base64
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..settings import TEMPLATES_DIR, settings
from ..pipeline.chapter_gen import Chapter


_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def _image_to_data_uri(path: Path | None) -> str:
    """Inline images as data URIs so Playwright doesn't need to fetch
    them from disk during PDF render."""
    if not path or not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}.get(suffix, "image/png")
    if suffix == ".svg":
        return path.read_text(encoding="utf-8")
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{data}" style="width:100%;height:auto;"/>'


def render_book_html(
    *,
    title: str,
    subtitle: str,
    date_range: str,
    stats: dict,
    chapters: list[Chapter],
    chapter_images: dict[int, Path],
    is_preview: bool,
    preview_chapter_count: int,
    cover_image: Path | None = None,
) -> str:
    """Render the full HTML for the book."""
    template = _jinja_env.get_template("book.html")

    chapter_image_html: dict[int, str] = {}
    for idx, path in chapter_images.items():
        chapter_image_html[idx] = _image_to_data_uri(path)

    cover_image_html = _image_to_data_uri(cover_image) if cover_image else ""

    return template.render(
        title=title,
        subtitle=subtitle,
        date_range=date_range,
        stats=stats,
        chapters=chapters,
        chapter_images=chapter_image_html,
        is_preview=is_preview,
        preview_chapter_count=preview_chapter_count,
        cover_image=cover_image_html,
        currency=settings.CURRENCY_SYMBOL,
        full_price=settings.FULL_BOOK_PRICE,
    )


def _render_sync(html: str, output_path: Path) -> Path:
    """Synchronous Playwright render. Runs in a worker thread to avoid
    the Windows asyncio subprocess bug. Do NOT call from async code
    directly — it blocks. Use render_html_to_pdf() instead.
    """
    from playwright.sync_api import sync_playwright

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        try:
            context = browser.new_context()
            page = context.new_page()
            # Use 'load' rather than 'networkidle' since all assets are
            # inlined as data URIs — no network activity to wait for.
            page.set_content(html, wait_until="load")
            page.pdf(
                path=str(output_path),
                width="148mm",
                height="210mm",
                print_background=True,
                prefer_css_page_size=True,
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            )
        finally:
            browser.close()

    return output_path


async def render_html_to_pdf(html: str, output_path: Path) -> Path:
    """Render HTML string to PDF. A5 page size.

    Offloads to a worker thread so the FastAPI event loop stays responsive
    AND so we avoid the Windows asyncio subprocess NotImplementedError.
    """
    return await asyncio.to_thread(_render_sync, html, output_path)
