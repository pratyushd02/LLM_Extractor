"""
Pipeline orchestrator.

Connects the four steps in order:
  1. Extract raw footnotes from PDF text
  2. Resolve each footnote to procedures + visits
  3. Parse conditions and key fields
  4. Write structured JSON

This file should stay thin — logic lives in the step modules.
"""

import time

from config import PipelineConfig
from extractor.footnote_extractor import extract_footnotes
from output.writer import write_json
from parser.condition_parser import parse_conditions
from resolver.reference_resolver import resolve_footnotes
from utils.pdf_reader import extract_pages, find_soa_pages


def run(pdf_path: str, out_path: str, config: PipelineConfig) -> dict:
    """
    Run the full pipeline.

    Args:
        pdf_path: Path to the protocol PDF.
        out_path: Where to write the output JSON.
        config:   Full pipeline configuration.

    Returns:
        The output dict (same as what is written to disk).
    """
    t_start = time.time()
    _log("START", f"pdf={pdf_path}  model={config.ollama.model}")

    # ── Step 0: read PDF ──────────────────────────────────────────────────────
    _log("0/4", "Reading PDF …")
    all_pages = extract_pages(pdf_path)
    _log("0/4", f"{len(all_pages)} pages extracted")

    soa_pages = find_soa_pages(
        all_pages,
        keywords=config.extraction.soa_keywords,
        top_n=config.extraction.soa_top_n_pages,
    )
    _log("0/4", f"SoA candidate pages: {soa_pages}")

    soa_texts = {pg: all_pages[pg] for pg in soa_pages if pg in all_pages}
    soa_combined = "\n\n".join(
        f"--- Page {pg} ---\n{txt}" for pg, txt in sorted(soa_texts.items())
    )

    # ── Step 1: extract footnotes ─────────────────────────────────────────────
    _log("1/4", "Extracting footnotes …")
    raw_footnotes = extract_footnotes(soa_texts, config)
    _log("1/4", f"{len(raw_footnotes)} footnote(s) found")

    if not raw_footnotes:
        _log("WARN", "No footnotes found — check --pages or PDF quality")

    # ── Step 2: resolve references ────────────────────────────────────────────
    _log("2/4", "Resolving references …")
    resolved = resolve_footnotes(raw_footnotes, soa_combined, config)
    _log("2/4", f"{len(resolved)} footnote(s) resolved")

    # ── Step 3: parse conditions ──────────────────────────────────────────────
    _log("3/4", "Parsing conditions …")
    structured = parse_conditions(resolved, config)
    _log("3/4", f"{len(structured)} footnote(s) structured")

    # ── Step 4: write output ──────────────────────────────────────────────────
    _log("4/4", f"Writing JSON → {out_path}")
    output = write_json(structured, out_path, config)

    elapsed = time.time() - t_start
    _log("DONE", f"{len(structured)} footnotes in {elapsed:.1f}s → {out_path}")

    return output


def _log(step: str, msg: str) -> None:
    print(f"  [{step:>4}] {msg}")
