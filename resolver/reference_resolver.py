"""
Step 2 — Reference resolution.

For each RawFootnote, asks the LLM to identify:
  - which procedure it describes
  - which study visits it applies to
  - broad tags (timing / scope / condition / population / ordering)

Input:  list[RawFootnote]  +  SoA table text
Output: list[ResolvedFootnote]
"""

import time

from config import PipelineConfig
from models import RawFootnote, ResolvedFootnote
from utils.ollama_client import chat, parse_json


SYSTEM = (
    "You are a clinical trial data engineer. "
    "You map protocol footnotes to their procedures and visit scope. "
    "Return only valid JSON — no markdown, no commentary."
)

PROMPT = """\
You are given a clinical trial protocol footnote and the Schedule of Assessments (SoA) table.

Footnote symbol : {symbol}
Footnote text   : {text}

SoA table (truncated):
{soa}

Determine:
1. The primary procedure or assessment this footnote describes.
2. Which study visits it applies to or qualifies.
3. Broad tags from: timing, scope, condition, population, ordering

Return JSON only:
{{
  "procedure": "<procedure name>",
  "visits":    ["<visit label, e.g. V1 (Screening), V2 (Day 0), All visits>"],
  "tags":      ["<tag>"]
}}
"""

# Max SoA characters sent per call — keeps context manageable
_SOA_LIMIT = 2500


def resolve_footnotes(
    footnotes: list[RawFootnote],
    soa_text: str,
    config: PipelineConfig,
) -> list[ResolvedFootnote]:
    """
    Resolve each footnote's procedure + visit scope against the SoA table.
    Returns one ResolvedFootnote per input RawFootnote.
    """
    resolved = []
    soa_snippet = soa_text[:_SOA_LIMIT]

    for fn in footnotes:
        prompt = PROMPT.format(
            symbol=fn.symbol,
            text=fn.text,
            soa=soa_snippet,
        )
        raw = chat(prompt, config.ollama, system=SYSTEM, verbose=config.verbose)
        data = parse_json(raw)

        resolved.append(ResolvedFootnote(
            symbol=fn.symbol,
            procedure=data.get("procedure", "Unknown"),
            visits=data.get("visits", []),
            tags=data.get("tags", []),
            raw=fn,
        ))
        time.sleep(0.2)

    return resolved
