"""
Step 4 — Output writer.

Serializes StructuredFootnote objects to a JSON file.
Keeping this separate means we can easily add CSV, FHIR, or USDM writers later.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from config import PipelineConfig
from models import StructuredFootnote


def write_json(
    footnotes: list[StructuredFootnote],
    out_path: str,
    config: PipelineConfig,
) -> dict:
    """
    Write structured footnotes to a JSON file.
    Returns the output dict (useful for tests and debugging).
    """
    output = {
        "protocol_id": config.protocol_id,
        "sponsor": config.sponsor,
        "indication": config.indication,
        "investigational_product": config.investigational_product,
        "model": config.ollama.model,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_footnotes": len(footnotes),
        "footnotes": [asdict(fn) for fn in footnotes],
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output
