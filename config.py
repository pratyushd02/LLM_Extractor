"""
Pipeline configuration — edit this file to change models, paths, and behaviour.
"""

from dataclasses import dataclass, field


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "qwen3-coder:480b-cloud"
    temperature: float = 0.05
    timeout_seconds: int = 120


@dataclass
class ExtractionConfig:
    # How many PDF pages to send to the LLM in one call
    pages_per_chunk: int = 4
    # Keywords used to auto-detect which pages likely contain the SoA
    soa_keywords: list[str] = field(default_factory=lambda: [
        "schedule of assessments", "schedule of activities",
        "procedure", "assessment", "screening", "randomis",
        "end of treatment", "follow-up",
    ])
    # How many top-scoring pages to consider as SoA candidates
    soa_top_n_pages: int = 6


@dataclass
class PipelineConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    verbose: bool = False

    # Protocol metadata written into the output JSON
    protocol_id: str = "CTJ301UC201"
    sponsor: str = "Leading Biopharm Limited"
    indication: str = "Active Ulcerative Colitis"
    investigational_product: str = "TJ301 (Olamkicept)"
