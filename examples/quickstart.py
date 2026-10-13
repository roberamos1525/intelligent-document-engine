"""Library quickstart: ingest the sample corpus, then search, ask and summarise.

Run from the project root:

    .venv/Scripts/python examples/quickstart.py      (Windows)
    .venv/bin/python examples/quickstart.py          (Linux/macOS)

Everything runs offline; data goes to a temporary directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from idengine.config import load_settings
from idengine.engine import Engine

CORPUS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden" / "corpus"


def main() -> None:
    with tempfile.TemporaryDirectory() as data_dir:
        settings = load_settings(environ={}, overrides={"storage": {"data_dir": data_dir}})
        with Engine(settings) as engine:
            for path in sorted(CORPUS.iterdir()):
                outcome, _ = engine.ingest_path(path, tags=["sample"])
                doc = outcome.document
                print(f"ingested {path.name:<18} -> {doc.id} ({doc.chunk_count} chunks)")

            print("\n== hybrid search: 'how long is personal data kept' ==")
            for rank, hit in enumerate(
                engine.search("how long is personal data kept", top_k=3).hits, 1
            ):
                print(f"{rank}. [{hit.score:.4f}] {hit.citation_label}\n   {hit.snippet}")

            print("\n== filtered search (markdown only) ==")
            filtered = engine.search("policy", filters={"doc_type": "markdown"}, top_k=3)
            for hit in filtered.hits:
                print(f"- {hit.citation_label}")

            for question in (
                "What is the monthly rent for the office lease?",
                "Who approved the FY2025 budget?",
                "What is the boiling point of mercury?",
            ):
                answer = engine.ask(question)
                print(f"\nQ: {question}\nA: {answer.answer}")
                if answer.short_answer:
                    print(f"   short answer: {answer.short_answer}")
                for citation in answer.citations:
                    print(f"   [{citation.number}] {citation.source}")

            first = engine.documents.all_ready()[0]
            summary = engine.summarize_document(first.id, max_sentences=2)
            print(f"\n== summary of {first.title} ==\n{summary.text}")


if __name__ == "__main__":
    main()
