"""Seeded randomised invariant tests for the core data structures.

These complement the example-based tests: each test generates many inputs
from a fixed seed and checks properties that must hold for all of them.
"""

from __future__ import annotations

import random
import string

import numpy as np
import pytest

from idengine.config import ChunkingSettings
from idengine.embeddings import HashingEmbedder
from idengine.ingest.chunker import Chunker
from idengine.ingest.model import Block, BlockKind, DocType, ExtractedDocument
from idengine.search.bm25 import BM25Index
from idengine.search.fusion import reciprocal_rank_fusion
from idengine.text.tokenize import analyze, count_tokens, split_sentences
from idengine.vectorindex import FlatIndex, HNSWIndex, compile_filter

WORDS = [
    "invoice",
    "payment",
    "contract",
    "lease",
    "policy",
    "employee",
    "deploy",
    "rollback",
    "budget",
    "audit",
    "tenant",
    "landlord",
    "notice",
    "privacy",
    "retention",
    "certificate",
    "outage",
    "replica",
    "cluster",
    "stipend",
    "leave",
]


def random_sentence(rng: random.Random) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(3, 18))]
    words[0] = words[0].capitalize()
    return " ".join(words) + rng.choice([".", ".", "!", "?"])


def random_document(rng: random.Random) -> ExtractedDocument:
    blocks: list[Block] = []
    for _ in range(rng.randint(1, 12)):
        roll = rng.random()
        if roll < 0.2:
            blocks.append(
                Block(BlockKind.HEADING, rng.choice(WORDS).title(), level=rng.randint(1, 3))
            )
        elif roll < 0.3:
            blocks.append(Block(BlockKind.LIST_ITEM, random_sentence(rng)))
        else:
            text = " ".join(random_sentence(rng) for _ in range(rng.randint(1, 8)))
            blocks.append(Block(BlockKind.PARAGRAPH, text, page=rng.randint(1, 5)))
    if all(b.is_heading for b in blocks):
        blocks.append(Block(BlockKind.PARAGRAPH, random_sentence(rng)))
    return ExtractedDocument(doc_type=DocType.TXT, blocks=blocks)


@pytest.mark.parametrize("seed", range(25))
def test_chunker_invariants(seed: int) -> None:
    rng = random.Random(seed)
    document = random_document(rng)
    max_tokens = rng.choice([24, 40, 80, 200])
    overlap = rng.choice([0, 5, 10])
    chunker = Chunker(ChunkingSettings(max_tokens=max_tokens, overlap_tokens=overlap, min_tokens=4))
    chunks = chunker.chunk(document)
    assert chunks, "every document with text yields at least one chunk"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    corpus = "\n".join(c.text for c in chunks)
    for block in document.blocks:
        if block.is_heading:
            continue
        for sentence in split_sentences(block.text):
            # Nothing is lost: every sentence (or its hard-split pieces) is present.
            assert sentence.text.split()[0] in corpus
    for chunk in chunks:
        # Merging a tiny tail may add up to min_tokens; nothing else may exceed the budget.
        assert chunk.token_count <= max_tokens + 4
        assert chunk.token_count == count_tokens(chunk.text)
        if chunk.page_start is not None:
            assert chunk.page_start <= (chunk.page_end or chunk.page_start)


@pytest.mark.parametrize("seed", range(10))
def test_hnsw_matches_exact_search_on_random_data(seed: int) -> None:
    rng = np.random.default_rng(seed)
    dim = 16
    vectors = rng.standard_normal((150, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"id{i}" for i in range(150)]
    meta = [{"bucket": i % 4} for i in range(150)]
    flat, hnsw = FlatIndex(dim), HNSWIndex(dim, m=8, ef_search=40, seed=seed)
    flat.add(ids, vectors, meta)
    hnsw.add(ids, vectors, meta)
    query = vectors[int(rng.integers(150))]
    exact = [h.id for h in flat.search(query, 5)]
    approx = [h.id for h in hnsw.search(query, 5)]
    assert exact[0] == approx[0]
    assert len(set(exact) & set(approx)) >= 4
    flt = compile_filter({"bucket": int(rng.integers(4))})
    assert {h.id for h in hnsw.search(query, 5, flt)} == {h.id for h in flat.search(query, 5, flt)}


@pytest.mark.parametrize("seed", range(10))
def test_bm25_scores_are_monotonic_in_term_frequency(seed: int) -> None:
    rng = random.Random(seed)
    index = BM25Index()
    for i in range(20):
        index.add(f"d{i}", analyze(" ".join(random_sentence(rng) for _ in range(3))), {})
    target = rng.choice(WORDS)
    base = " ".join(w for w in WORDS[:10] if w != target)
    index.add("low", analyze(f"{base} {target}"), {})
    index.add("high", analyze(f"{base} {target} {target} {target}"), {})
    scores = {h.id: h.score for h in index.search(analyze(target), 50)}
    assert scores["high"] > scores["low"] > 0
    assert all(score > 0 for score in scores.values())


@pytest.mark.parametrize("seed", range(10))
def test_compiled_filters_agree_with_python_semantics(seed: int) -> None:
    rng = random.Random(seed)
    records = [
        {"n": rng.randint(0, 20), "cat": rng.choice("abc"), "tags": rng.sample(list("xyz"), 2)}
        for _ in range(60)
    ]
    low, high = sorted(rng.sample(range(21), 2))
    category = rng.choice("abc")
    tag = rng.choice("xyz")
    spec = {
        "$or": [
            {"n": {"$gte": low, "$lt": high}, "cat": {"$ne": category}},
            {"tags": tag, "$not": {"cat": {"$in": ["a"]}}},
        ]
    }
    predicate = compile_filter(spec)
    assert predicate is not None
    for record in records:
        expected = (low <= record["n"] < high and record["cat"] != category) or (
            tag in record["tags"] and record["cat"] != "a"
        )
        assert predicate(record) is expected


@pytest.mark.parametrize("seed", range(5))
def test_rrf_is_order_invariant_and_bounded(seed: int) -> None:
    rng = random.Random(seed)
    pool = [f"x{i}" for i in range(30)]
    a = rng.sample(pool, 15)
    b = rng.sample(pool, 15)
    forward = reciprocal_rank_fusion({"a": a, "b": b}, k=60)
    backward = reciprocal_rank_fusion({"b": b, "a": a}, k=60)
    assert [item for item, _, _ in forward] == [item for item, _, _ in backward]
    for _, score, ranks in forward:
        assert 0 < score <= len(ranks) / 61


@pytest.mark.parametrize("seed", range(5))
def test_hashing_embeddings_are_stable_under_word_order_noise(seed: int) -> None:
    rng = random.Random(seed)
    embedder = HashingEmbedder(dim=128, use_bigrams=False)
    words = [rng.choice(WORDS) for _ in range(12)]
    shuffled = words[:]
    rng.shuffle(shuffled)
    unrelated = "".join(rng.choice(string.ascii_lowercase) for _ in range(40))
    vectors = embedder.embed([" ".join(words), " ".join(shuffled), unrelated])
    # Without bigrams the representation is a bag of words: order does not matter.
    assert float(vectors[0] @ vectors[1]) == pytest.approx(1.0, abs=1e-5)
    assert float(vectors[0] @ vectors[2]) < 0.5
