# SoA Footnote Digitization Pipeline

Extracts, resolves, and structures footnotes from clinical trial protocol PDFs
using a local Ollama LLM.

## Project layout

```
.
├── main.py                        # CLI entry point
├── pipeline.py                    # Orchestrator
├── config.py, models.py
├── extractor/  resolver/  parser/  output/  utils/
├── data/
│   └── sample_protocols/
└── requirements.txt
```

## Quickstart

```bash
# 1. Install Ollama and pull a model
ollama pull llama3.1
ollama serve

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run (from the repo root — this folder is the project / git root)
python main.py --pdf data/sample_protocols/CTJ301UC201_Protocol.pdf

# With options
python main.py \
  --pdf  data/sample_protocols/CTJ301UC201_Protocol.pdf \
  --model mistral \
  --out  results/soa_footnotes.json \
  --pages 28,29,30 \
  --verbose
```

## Pipeline steps

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 | `utils/pdf_reader.py` | PDF file | `{page: text}` |
| 1 | `extractor/footnote_extractor.py` | page texts | `list[RawFootnote]` |
| 2 | `resolver/reference_resolver.py` | `RawFootnote` + SoA text | `list[ResolvedFootnote]` |
| 3 | `parser/condition_parser.py` | `ResolvedFootnote` | `list[StructuredFootnote]` |
| 4 | `output/writer.py` | `StructuredFootnote` | JSON file |

## Output schema

```json
{
  "protocol_id": "CTJ301UC201",
  "sponsor": "Leading Biopharm Limited",
  "total_footnotes": 14,
  "footnotes": [
    {
      "symbol": "a",
      "procedure": "Patient diary",
      "visits": ["V2 (Day 0)", "V3 (Week 2)"],
      "tags": ["timing", "scope"],
      "conditions": [
        {
          "type": "timing",
          "description": "Diary collected and reviewed at each subsequent visit",
          "applies_if": null,
          "value": null
        }
      ],
      "key_fields": {
        "dispense_visit": "V2",
        "collect_at": "each subsequent visit"
      },
      "raw_text": "Patient diary to be dispensed at V2 ...",
      "source_page": 31,
      "confidence": 0.95,
      "notes": ""
    }
  ]
}
```

## Running tests

```bash
pytest tests/ -v
```

## What's next (incremental improvements)

- [ ] Table-aware extraction using `pdfplumber` table parser
- [ ] Symbol-to-cell mapping (which SoA cell carries which footnote marker)
- [ ] Cross-reference resolution (footnote A references Section 5.2)
- [ ] CSV / USDM output writer
- [ ] Evaluation harness with gold-standard annotations
