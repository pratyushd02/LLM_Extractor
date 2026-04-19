"""
Step 1 — Footnote extraction.

Takes raw page text, returns a list of RawFootnote objects.
Two strategies run in sequence:
  1. LLM-based  — more accurate, handles unusual symbols and wrapped text
  2. Regex-based — fast fallback; fills gaps the LLM misses
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
            # Keep pipeline running; regex fallback can still recover many footnotes.
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


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_footnotes(
    page_texts: dict[int, str],
    config: PipelineConfig,
) -> list[RawFootnote]:
    """
    Extract footnotes from page text using LLM + regex fallback.
    Returns deduplicated RawFootnote list.
    """
    llm_results = _llm_extract(page_texts, config)

    # Fill gaps with regex
    llm_syms = {_normalize_symbol(fn.symbol) for fn in llm_results}
    regex_results = _regex_extract(page_texts)
    extras = [fn for fn in regex_results if _normalize_symbol(fn.symbol) not in llm_syms]

    return llm_results + extras
