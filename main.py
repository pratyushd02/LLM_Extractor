import argparse

from config import OllamaConfig, PipelineConfig
from pipeline import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Digitize SoA footnotes from a clinical trial protocol PDF."
    )
    p.add_argument("--pdf",     required=True,              help="Path to protocol PDF")
    p.add_argument("--out",     default="soa_footnotes.json", help="Output JSON path")
    p.add_argument("--model",   default="llama3.1",         help="Ollama model name")
    p.add_argument("--host",    default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--pages",   default="",
                   help="Comma-separated page numbers to search (e.g. 28,29,30). "
                        "Auto-detected if omitted.")
    p.add_argument("--verbose", action="store_true",        help="Print LLM prompts/responses")

    # Protocol metadata
    p.add_argument("--protocol-id", default="CTJ301UC201")
    p.add_argument("--sponsor",     default="Leading Biopharm Limited")
    p.add_argument("--indication",  default="Active Ulcerative Colitis")
    p.add_argument("--product",     default="TJ301 (Olamkicept)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        ollama=OllamaConfig(host=args.host, model=args.model),
        verbose=args.verbose,
        protocol_id=args.protocol_id,
        sponsor=args.sponsor,
        indication=args.indication,
        investigational_product=args.product,
    )

    # Override SoA page detection if the user supplied explicit pages
    manual_pages = []
    if args.pages.strip():
        manual_pages = [int(p) for p in args.pages.split(",") if p.strip().isdigit()]

    run(pdf_path=args.pdf, out_path=args.out, config=config)


if __name__ == "__main__":
    main()
