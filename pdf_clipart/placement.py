"""
Clipart placement.

Finds empty rectangles on a page that don't overlap text or existing
images, and inserts generated PNGs there. Strategy:

  1. Build a coarse occupancy grid (text + images + page margins = blocked).
  2. Find the largest empty square via dynamic programming.
  3. Place a clipart (clamped to a tasteful size) centered in that square.
  4. Mark it occupied and repeat for the next clipart.

Because we re-mark after every placement, multiple cliparts on one page can
never overlap each other or the text.
"""

from __future__ import annotations

import logging
import math

import fitz

from .config import Config
from .content_reader import PageContent

log = logging.getLogger("pdf_clipart.placement")


def _build_grid(page: PageContent, config: Config) -> tuple[list[list[bool]], int, int, float]:
    """Return (occupied, rows, cols, cell) where occupied[r][c] is True if
    that cell is blocked. Cells outside the page margin are blocked too."""
    cell = config.grid_cell_pt
    cols = max(1, math.ceil(page.width / cell))
    rows = max(1, math.ceil(page.height / cell))
    occupied = [[False] * cols for _ in range(rows)]

    margin = config.page_margin_pt

    def block_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        c0 = max(0, int(x0 // cell))
        c1 = min(cols - 1, int(x1 // cell))
        r0 = max(0, int(y0 // cell))
        r1 = min(rows - 1, int(y1 // cell))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                occupied[r][c] = True

    # Margins: block the outer frame.
    block_rect(0, 0, page.width, margin)                       # top
    block_rect(0, page.height - margin, page.width, page.height)  # bottom
    block_rect(0, 0, margin, page.height)                      # left
    block_rect(page.width - margin, 0, page.width, page.height)   # right

    # Text + image boxes, inflated by text padding.
    pad = config.text_padding_pt
    for (x0, y0, x1, y1) in page.occupied_boxes:
        block_rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    return occupied, rows, cols, cell


def _largest_empty_square(occupied: list[list[bool]], rows: int, cols: int):
    """Classic maximal-square DP. Returns (r0, c0, side_cells) of the
    biggest empty square (bottom-right tracked, converted to top-left)."""
    dp = [[0] * cols for _ in range(rows)]
    best = 0
    best_rc = None
    for r in range(rows):
        for c in range(cols):
            if occupied[r][c]:
                dp[r][c] = 0
                continue
            if r == 0 or c == 0:
                dp[r][c] = 1
            else:
                dp[r][c] = 1 + min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1])
            if dp[r][c] > best:
                best = dp[r][c]
                best_rc = (r, c)
    if not best_rc:
        return None
    br, bc = best_rc
    r0 = br - best + 1
    c0 = bc - best + 1
    return r0, c0, best


def _mark(occupied, r0, c0, side):
    for r in range(r0, r0 + side):
        for c in range(c0, c0 + side):
            occupied[r][c] = True


def place_cliparts(page: "fitz.Page", content: PageContent,
                   images: list[bytes], config: Config) -> int:
    """Insert each PNG in `images` into a free square on the page.

    Returns the number actually placed (may be fewer than supplied if the
    page runs out of room).
    """
    if not images:
        return 0

    occupied, rows, cols, cell = _build_grid(content, config)
    placed = 0

    for png in images:
        found = _largest_empty_square(occupied, rows, cols)
        if not found:
            log.info("  page %d: no free space left for more clipart", content.index + 1)
            break
        r0, c0, side = found
        side_pt = side * cell
        if side_pt < config.min_clipart_pt:
            log.info("  page %d: largest gap %.0fpt < min %.0fpt — stopping",
                     content.index + 1, side_pt, config.min_clipart_pt)
            break

        # Clamp to a tasteful max, centered within the empty square.
        draw_pt = min(side_pt, config.max_clipart_pt)
        cx = (c0 * cell) + side_pt / 2
        cy = (r0 * cell) + side_pt / 2
        rect = fitz.Rect(cx - draw_pt / 2, cy - draw_pt / 2,
                         cx + draw_pt / 2, cy + draw_pt / 2)

        try:
            page.insert_image(rect, stream=png, keep_proportion=True)
        except Exception as exc:
            log.warning("  page %d: insert_image failed: %s", content.index + 1, exc)
            continue

        # Mark the used cells (the full empty square) so the next clipart
        # picks a different region.
        _mark(occupied, r0, c0, side)
        placed += 1

    return placed
