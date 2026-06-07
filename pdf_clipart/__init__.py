"""
pdf_clipart — add content-aware AI-generated clipart to a PDF.

Public API:
    from pdf_clipart import Config, annotate_pdf
    annotate_pdf("in.pdf", "out.pdf", Config(backend="stub"))
"""

from .config import Config
from .pipeline import annotate_pdf, RunSummary, PageResult

__all__ = ["Config", "annotate_pdf", "RunSummary", "PageResult"]
