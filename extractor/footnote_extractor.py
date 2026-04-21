"""
Step 1 — Footnote extraction.

Takes raw page text, returns a list of RawFootnote objects.
Two strategies run in sequence:
  1. LLM-based  — more accurate, handles unusual symbols and wrapped text
  2. Regex-based — fast fallback; fills gaps the LLM misses

After extraction, a confidence-scoring pass filters out footnotes that are
unlikely to belong to the Schedule of Assessments (SOA). Only footnotes with
SOA confidence > 0.5 are returned for further processing.
"""

import re
import time
import json

from config import PipelineConfig
from models import RawFootnote
from utils.ollama_client import chat, parse_json


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are a clinical trial data engineer. "
    "You extract information from protocol documents and return only valid JSON — "
    "no markdown fences, no commentary."
)

PROMPT = """\
Extract ALL footnote definitions from the protocol text below.

Footnotes appear below the Schedule of Assessments table. They are labelled with
letters (a, b, c…), numbers (1, 2, 3…), or symbols (†, ‡, *, §) — sometimes in
parentheses. Each footnote has a symbol and a body of explanatory text.

Return JSON only:
{{
  "footnotes": [
    {{
      "symbol": "<label e.g. 'a' or '(b)' or '†'>",
      "text":   "<complete verbatim footnote body>",
      "page":   <integer page number>
    }}
  ]
}}

Return {{"footnotes": []}} if none are found. Do NOT invent footnotes.

--- TEXT ---
{text}
--- END ---
"""

SOA_CONFIDENCE_SYSTEM = (
    "You are a clinical trial data engineer. "
    "You classify footnotes and return only valid JSON — "
    "no markdown fences, no commentary."
)

SOA_CONFIDENCE_PROMPT = """\
You are given a list of footnotes extracted from a clinical trial protocol document.

Your task: for each footnote, estimate the probability (0.0 to 1.0) that it belongs
to a Schedule of Assessments (SOA) table — i.e. it describes timing, frequency,
conditions, or procedures for assessments listed in the SOA.

Indicators of a SOA footnote:
- References visit windows, cycles, days, or timepoints (e.g. "Day 1", "Cycle 2", "within 72 hours")
- Describes when/how an assessment should be performed
- Mentions procedures like blood draws, ECGs, biopsies, questionnaires
- Uses clinical trial language: "predose", "postdose", "screening", "follow-up"
- Clarifies exceptions or conditions for specific SOA rows/columns

Indicators it is NOT a SOA footnote:
- Defines abbreviations or acronyms only (e.g. "AE = Adverse Event")
- Describes general regulatory or ethical statements
- References sections of the protocol unrelated to assessments
- Is a general document header/footer note

Return JSON only:
{{
  "footnotes": [
    {{
      "symbol": "<symbol>",
      "soa_confidence": <float 0.0–1.0>
    }}
  ]
}}

--- FOOTNOTES ---
{footnotes}
--- END ---
"""


# ── Regex fallback ─────────────────────────────────────────────────────────────

_PATTERN = re.compile(
    r"""
    (?:^|\n)\s*
    (?P<sym>
        \([a-z0-9]{1,3}\)       # (a), (b), (12)
      | \([†‡§¶*#]{1,2}\)       # (†)
      | [a-z]\.\s               # a.
      | \d+\.\s                 # 1.
      | [†‡§¶*#]+\s             # † or *
    )
    \s*
    (?P<text>[A-Z][^\n]{25,})   # body starts uppercase, min 25 chars
    """,
    re.VERBOSE | re.MULTILINE,
)


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol variants (e.g. '"a"', '(a)', 'A.') to a stable key."""
    sym = symbol.strip()
    while len(sym) >= 2 and sym[0] == sym[-1] and sym[0] in {"'", '"'}:
        sym = sym[1:-1].strip()
    sym = sym.rstrip(".").strip()

    if re.fullmatch(r"\(([a-z0-9]{1,3}|[†‡§¶*#]{1,2})\)", sym, flags=re.IGNORECASE):
        sym = sym[1:-1]
    if re.fullmatch(r"[a-z0-9]{1,3}", sym, flags=re.IGNORECASE):
        sym = sym.lower()

    return sym


def _regex_extract(page_texts: dict[int, str]) -> list[RawFootnote]:
    found, seen = [], set()
    for pg, text in sorted(page_texts.items()):
        for m in _PATTERN.finditer(text):
            sym = _normalize_symbol(m.group("sym"))
            body = m.group("text").strip()
            if sym not in seen and len(body) >= 25:
                seen.add(sym)
                found.append(RawFootnote(symbol=sym, text=body, page=pg))
    return found


# ── LLM extraction ─────────────────────────────────────────────────────────────

def _llm_extract(
    page_texts: dict[int, str],
    config: PipelineConfig,
) -> list[RawFootnote]:
    chunk = config.extraction.pages_per_chunk
    pages = sorted(page_texts)
    results: list[RawFootnote] = []
    seen_to_idx: dict[str, int] = {}

    for i in range(0, len(pages), chunk):
        batch = pages[i : i + chunk]
        combined = "\n\n".join(
            f"--- Page {pg} ---\n{page_texts[pg]}" for pg in batch
        )
        prompt = PROMPT.format(text=combined)
        raw = chat(prompt, config.ollama, system=SYSTEM, verbose=config.verbose)
        try:
            parsed = parse_json(raw)
        except json.JSONDecodeError:
            parsed = {"footnotes": []}

        for fn in parsed.get("footnotes", []):
            sym = _normalize_symbol(str(fn.get("symbol", "")))
            text = str(fn.get("text", "")).strip()
            if not sym or not text:
                continue

            page = int(fn.get("page", batch[0]))
            if sym not in seen_to_idx:
                seen_to_idx[sym] = len(results)
                results.append(RawFootnote(symbol=sym, text=text, page=page))
                continue

            # If duplicate symbol appears, keep the richer variant.
            idx = seen_to_idx[sym]
            if len(text) > len(results[idx].text):
                results[idx] = RawFootnote(symbol=sym, text=text, page=page)
        time.sleep(0.2)

    return results


# ── SOA confidence filtering ───────────────────────────────────────────────────

def _filter_soa_footnotes(
    footnotes: list[RawFootnote],
    config: PipelineConfig,
    threshold: float = 0.5,
    batch_size: int = 20,
) -> list[RawFootnote]:
    """
    Score each footnote for SOA relevance using the LLM.
    Returns only those with soa_confidence > threshold.

    Footnotes are sent in batches to avoid overflowing the context window.
    Any footnote whose symbol is missing from the LLM response (parse error,
    dropout) is kept by default so we don't silently discard data.
    """
    if not footnotes:
        return []

    scores: dict[str, float] = {}

    for i in range(0, len(footnotes), batch_size):
        batch = footnotes[i : i + batch_size]
        serialised = json.dumps(
            [{"symbol": fn.symbol, "text": fn.text} for fn in batch],
            indent=2,
        )
        prompt = SOA_CONFIDENCE_PROMPT.format(footnotes=serialised)
        raw = chat(
            prompt, config.ollama, system=SOA_CONFIDENCE_SYSTEM, verbose=config.verbose
        )
        try:
            parsed = parse_json(raw)
        except json.JSONDecodeError:
            # Keep all footnotes in this batch on parse failure.
            if config.verbose:
                print(
                    f"[soa_filter] JSON parse error on batch {i}–{i+batch_size}; "
                    "keeping all footnotes in batch."
                )
            for fn in batch:
                scores[fn.symbol] = 1.0
            continue

        for item in parsed.get("footnotes", []):
            sym = _normalize_symbol(str(item.get("symbol", "")))
            try:
                confidence = float(item.get("soa_confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            scores[sym] = confidence

        time.sleep(0.2)

    kept, dropped = [], []
    for fn in footnotes:
        # Default to keeping if the LLM didn't return a score for this symbol.
        confidence = scores.get(fn.symbol, 1.0)
        if confidence > threshold:
            kept.append(fn)
        else:
            dropped.append((fn.symbol, confidence))

    if config.verbose and dropped:
        print(
            f"[soa_filter] Dropped {len(dropped)} non-SOA footnote(s) "
            f"(threshold={threshold}): "
            + ", ".join(f"{s}={c:.2f}" for s, c in dropped)
        )

    return kept


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_footnotes(
    page_texts: dict[int, str],
    config: PipelineConfig,
    soa_confidence_threshold: float = 0.5,
) -> list[RawFootnote]:
    """
    Extract footnotes from page text using LLM + regex fallback,
    then filter to only SOA-relevant footnotes (confidence > soa_confidence_threshold).

    Returns deduplicated, SOA-filtered RawFootnote list.
    """
    llm_results = _llm_extract(page_texts, config)

    # Fill gaps with regex
    llm_syms = {_normalize_symbol(fn.symbol) for fn in llm_results}
    regex_results = _regex_extract(page_texts)
    extras = [fn for fn in regex_results if _normalize_symbol(fn.symbol) not in llm_syms]

    all_footnotes = llm_results + extras

    # Filter: keep only footnotes likely belonging to the SOA
    return _filter_soa_footnotes(
        all_footnotes,
        config,
        threshold=soa_confidence_threshold,
    )