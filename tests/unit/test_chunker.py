from __future__ import annotations

from itertools import pairwise

from idengine.config import ChunkingSettings
from idengine.ingest.chunker import Chunker, split_by_tokens
from idengine.ingest.model import Block, BlockKind, DocType, ExtractedDocument
from idengine.text.tokenize import count_tokens


def sentence(i: int) -> str:
    return f"Sentence number {i} talks about topic {i} in some detail."


def make_doc(blocks: list[Block]) -> ExtractedDocument:
    return ExtractedDocument(doc_type=DocType.TXT, blocks=blocks)


def test_chunks_respect_token_budget_and_overlap() -> None:
    body = " ".join(sentence(i) for i in range(40))
    doc = make_doc([Block(BlockKind.PARAGRAPH, body)])
    settings = ChunkingSettings(max_tokens=60, overlap_tokens=15, min_tokens=5)
    chunks = Chunker(settings).chunk(doc)
    assert len(chunks) > 5
    assert all(c.token_count <= 60 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    # Consecutive chunks share the trailing sentence (sentence-level overlap).
    for prev, nxt in pairwise(chunks):
        last_sentence = prev.text.split(". ")[-1]
        assert nxt.text.startswith(last_sentence.rstrip("."))
    # All original sentences are preserved somewhere.
    joined = " ".join(c.text for c in chunks)
    assert all(sentence(i) in joined for i in range(40))


def test_no_overlap_when_disabled() -> None:
    body = " ".join(sentence(i) for i in range(20))
    settings = ChunkingSettings(max_tokens=40, overlap_tokens=0, min_tokens=1)
    chunks = Chunker(settings).chunk(make_doc([Block(BlockKind.PARAGRAPH, body)]))
    total = sum(c.token_count for c in chunks)
    assert total == count_tokens(body)


def test_heading_aware_chunks_never_cross_sections() -> None:
    doc = make_doc(
        [
            Block(BlockKind.HEADING, "Contract", level=1),
            Block(BlockKind.HEADING, "Payment", level=2),
            Block(BlockKind.PARAGRAPH, "Pay within 30 days."),
            Block(BlockKind.HEADING, "Termination", level=2),
            Block(BlockKind.PARAGRAPH, "Either party may terminate with notice."),
        ]
    )
    chunks = Chunker(ChunkingSettings(max_tokens=200, overlap_tokens=10)).chunk(doc)
    assert [c.heading_path for c in chunks] == [
        ("Contract", "Payment"),
        ("Contract", "Termination"),
    ]
    assert chunks[0].section == "Payment"
    assert chunks[0].index_text() == "Contract > Payment\nPay within 30 days."


def test_flattened_mode_keeps_heading_text() -> None:
    doc = make_doc(
        [
            Block(BlockKind.HEADING, "Intro", level=1),
            Block(BlockKind.PARAGRAPH, "Hello there."),
            Block(BlockKind.HEADING, "Body", level=1),
            Block(BlockKind.PARAGRAPH, "More text."),
        ]
    )
    settings = ChunkingSettings(max_tokens=200, overlap_tokens=0, respect_headings=False)
    chunks = Chunker(settings).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ()
    assert chunks[0].text == "Intro\n\nHello there.\n\nBody\n\nMore text."


def test_page_ranges_follow_units() -> None:
    doc = make_doc(
        [
            Block(BlockKind.PARAGRAPH, sentence(1), page=1),
            Block(BlockKind.PARAGRAPH, sentence(2), page=2),
            Block(BlockKind.PARAGRAPH, sentence(3), page=3),
        ]
    )
    chunks = Chunker(ChunkingSettings(max_tokens=500, overlap_tokens=0)).chunk(doc)
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 3)


def test_overlong_sentence_is_hard_split() -> None:
    long_sentence = " ".join(f"word{i}" for i in range(250))
    doc = make_doc([Block(BlockKind.PARAGRAPH, long_sentence)])
    chunks = Chunker(ChunkingSettings(max_tokens=100, overlap_tokens=0)).chunk(doc)
    assert [c.token_count for c in chunks] == [100, 100, 50]


def test_tables_split_by_rows_and_code_kept_whole() -> None:
    table = "\n".join(f"row {i} | value {i}" for i in range(30))
    doc = make_doc(
        [
            Block(BlockKind.TABLE, table),
            Block(BlockKind.CODE, "def f():\n    return 1"),
        ]
    )
    chunks = Chunker(ChunkingSettings(max_tokens=40, overlap_tokens=0, min_tokens=1)).chunk(doc)
    assert all(c.token_count <= 40 for c in chunks)
    assert chunks[-1].text.endswith("def f():\n    return 1")


def test_tiny_trailing_chunk_is_merged() -> None:
    body = " ".join(sentence(i) for i in range(5)) + " Ok."
    settings = ChunkingSettings(max_tokens=70, overlap_tokens=0, min_tokens=5)
    chunks = Chunker(settings).chunk(make_doc([Block(BlockKind.PARAGRAPH, body)]))
    assert chunks[-1].text.endswith("Ok.")
    assert chunks[-1].token_count > 5


def test_split_by_tokens_windows() -> None:
    assert split_by_tokens("a b c d e", 2) == ["a b", "c d", "e"]
    assert split_by_tokens("  ", 5) == []
    assert split_by_tokens("short", 5) == ["short"]
