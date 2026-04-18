"""
Shared data models for the SoA footnote digitization pipeline.
These are simple dataclasses — no business logic here.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawFootnote:
    """A footnote exactly as found in the PDF — no interpretation yet."""
    symbol: str        # e.g. "a", "(b)", "†"
    text: str          # verbatim prose
    page: int          # 1-indexed page number


@dataclass
class ResolvedFootnote:
    """Footnote after its scope has been mapped to SoA procedures and visits."""
    symbol: str
    procedure: str             # e.g. "12-lead ECG"
    visits: list[str]          # e.g. ["V1 (Screening)", "V2 (Day 0)"]
    tags: list[str]            # e.g. ["timing", "condition"]
    raw: RawFootnote           # keep the original for traceability


@dataclass
class StructuredFootnote:
    """Final output: fully parsed with conditions and key fields."""
    symbol: str
    procedure: str
    visits: list[str]
    tags: list[str]
    conditions: list[dict]     # [{type, description, applies_if, value}]
    key_fields: dict           # domain-specific KV pairs (schedule, population, etc.)
    raw_text: str
    source_page: int
    confidence: float = 1.0
    notes: str = ""
