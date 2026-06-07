"""
Page content reader (PyMuPDF).

Pulls the text and the text/image bounding boxes for a single page at a
time. We deliberately never read the whole document into memory — the
pipeline opens the doc once and asks this module for one page's data when
it needs it.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class PageContent:
    index: int                  # 0-based page number
    text: str                   # concatenated visible text
    width: float                # page width in points
    height: float               # page height in points
    occupied_boxes: list[tuple[float, float, float, float]]
    """Bounding boxes (x0, y0, x1, y1) of text and existing images —
    regions clipart must avoid."""


def read_page(page: "fitz.Page", index: int) -> PageContent:
    """Extract text + occupied regions for one already-open page."""
    text = page.get_text("text") or ""

    boxes: list[tuple[float, float, float, float]] = []
    # Text blocks: get_text("blocks") -> (x0, y0, x1, y1, text, no, type)
    for block in page.get_text("blocks"):
        if len(block) >= 5 and str(block[4]).strip():
            boxes.append((float(block[0]), float(block[1]),
                          float(block[2]), float(block[3])))

    # Existing raster images on the page — don't draw on top of them.
    try:
        for info in page.get_image_info():
            bbox = info.get("bbox")
            if bbox:
                boxes.append((float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])))
    except Exception:
        # get_image_info isn't critical; ignore if the build lacks it.
        pass

    rect = page.rect
    return PageContent(
        index=index,
        text=text,
        width=float(rect.width),
        height=float(rect.height),
        occupied_boxes=boxes,
    )
