"""PDF text extraction and SoA page heuristics."""

from __future__ import annotations

from collections.abc import Sequence

import pdfplumber


def extract_pages(pdf_path: str) -> dict[int, str]:
    """Return 1-based page index → extracted plain text."""
    pages: dict[int, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            pages[i] = (text or "").strip()
    return pages


def find_soa_pages(
    all_pages: dict[int, str],
    keywords: Sequence[str],
    top_n: int,
) -> list[int]:
    """
    Rank pages by how often SoA-related keywords appear; return up to `top_n` pages.
    If nothing scores, fall back to the first `top_n` page numbers present.
    """
    scored: list[tuple[int, int]] = []
    for pg, text in all_pages.items():
        if not text:
            continue
        lower = text.lower()
        score = sum(lower.count(k.lower()) for k in keywords)
        if score > 0:
            scored.append((score, pg))

    scored.sort(reverse=True)
    out: list[int] = []
    seen: set[int] = set()
    for _, pg in scored:
        if pg not in seen:
            seen.add(pg)
            out.append(pg)
        if len(out) >= top_n:
            return out

    if out:
        return out

    ordered = sorted(all_pages.keys())
    return ordered[:top_n]
