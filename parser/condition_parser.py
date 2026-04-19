"""
Step 3 — Condition & field parsing.

For each ResolvedFootnote, extracts:
  - Typed conditions (timing rules, population filters, ordering constraints, etc.)
  - Key domain fields (schedule_days, lab_type, population, etc.)
  - Confidence score

Input:  list[ResolvedFootnote]
Output: list[StructuredFootnote]
"""

import time

from config import PipelineConfig
from models import ResolvedFootnote, StructuredFootnote
from utils.ollama_client import chat, parse_json


SYSTEM = (
    "You are a clinical trial data engineer. "
    "You parse structured logic from protocol footnote text. "
    "Return only valid JSON — no markdown, no commentary."
)

PROMPT = """\
Parse the clinical trial footnote below and extract structured information.

Procedure : {procedure}
Footnote  : {text}

Return JSON only:
{{
  "conditions": [
    {{
      "type":        "<timing | population | conditional | scope | ordering | storage | lab>",
      "description": "<plain English>",
      "applies_if":  "<optional trigger condition or null>",
      "value":       "<optional concrete value/threshold or null>"
    }}
  ],
  "key_fields": {{
    "<field_name>": "<value>"
  }},
  "confidence":  <0.0–1.0>,
  "notes":       "<any ambiguities or assumptions>"
}}

Guidelines for key_fields by procedure type:
- Drug administration → schedule_days, frequency, infusion_duration_hours, observation_hours
- Laboratory          → lab_type, collection_weeks, ship_within_hours
- PK substudy        → population, n_patients, sampling_type, timepoints_hours
- Pregnancy test     → population, acceptable_local_test_condition
- Biomarkers         → analytes, collection_weeks, storage_temp_celsius
- Endoscopy          → reader_type, modalities, biopsy_analyses

Only extract fields that are explicitly stated in the footnote.
"""


def parse_conditions(
    resolved: list[ResolvedFootnote],
    config: PipelineConfig,
) -> list[StructuredFootnote]:
    """
    Parse each ResolvedFootnote into a StructuredFootnote with typed conditions
    and key domain fields.
    """
    structured = []

    for fn in resolved:
        prompt = PROMPT.format(procedure=fn.procedure, text=fn.raw.text)
        raw = chat(prompt, config.ollama, system=SYSTEM, verbose=config.verbose)
        data = parse_json(raw)

        structured.append(StructuredFootnote(
            symbol=fn.symbol,
            procedure=fn.procedure,
            visits=fn.visits,
            tags=fn.tags,
            conditions=data.get("conditions", []),
            key_fields=data.get("key_fields", {}),
            raw_text=fn.raw.text,
            source_page=fn.raw.page,
            confidence=float(data.get("confidence", 1.0)),
            notes=data.get("notes", ""),
        ))
        time.sleep(0.2)

    return structured
