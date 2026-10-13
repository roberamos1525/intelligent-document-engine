from __future__ import annotations

import pytest

from idengine.text.normalize import clean_text, dehyphenate, fingerprint_text, unwrap_lines
from idengine.text.stemmer import stem
from idengine.text.tokenize import (
    analyze,
    bigrams,
    count_tokens,
    sentences,
    split_sentences,
    words,
)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("motoring", "motor"),
        ("hopping", "hop"),
        ("filing", "file"),
        ("happy", "happi"),
        ("relational", "relat"),
        ("conditional", "condit"),
        ("digitizer", "digit"),
        ("generalizations", "gener"),
        ("formative", "form"),
        ("electrical", "electr"),
        ("adjustment", "adjust"),
        ("adoption", "adopt"),
        ("controll", "control"),
        ("invoices", "invoic"),
        ("invoicing", "invoic"),
        ("is", "is"),
    ],
)
def test_porter_stemmer(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_words_and_tokens() -> None:
    assert words("The U.S. office's e-mail: Hello!") == [
        "the",
        "u.s",
        "office's",
        "e-mail",
        "hello",
    ]
    assert count_tokens("Hello, world!") == 4


def test_analyze_drops_stopwords_keeps_numbers_and_stems() -> None:
    terms = analyze("The invoices were paid in 2024 for order INV-77")
    assert "the" not in terms
    assert "invoic" in terms
    assert "2024" in terms
    assert "inv-77" in terms
    assert analyze("the and of", keep_stopwords=True) == ["the", "and", "of"]


def test_bigrams() -> None:
    assert bigrams(["a", "b", "c"]) == ["a_b", "b_c"]
    assert bigrams(["a"]) == []


def test_sentence_split_handles_abbreviations_and_decimals() -> None:
    text = "Dr. Smith paid $3.50 at 5 p.m. on Jan. 5. He left! Did he? yes he did.\n\nNew para"
    assert sentences(text) == [
        "Dr. Smith paid $3.50 at 5 p.m. on Jan. 5.",
        "He left!",
        "Did he? yes he did.",
        "New para",
    ]


def test_sentence_offsets_point_into_source() -> None:
    text = "  First sentence here.   Second one follows."
    for span in split_sentences(text):
        assert text[span.start : span.end] == span.text


def test_initials_do_not_split() -> None:
    assert sentences("Signed by J. Doe and K. Roe. Done.") == [
        "Signed by J. Doe and K. Roe.",
        "Done.",
    ]


def test_clean_text_normalises_unicode_and_whitespace() -> None:
    raw = "eﬃcient text “quoted”\x07\r\ninfor-\nmation\n\n\n\nnext"
    assert clean_text(raw) == 'efficient text "quoted"\ninformation\n\nnext'


def test_dehyphenate_only_joins_lowercase_breaks() -> None:
    assert dehyphenate("co-\noperate") == "cooperate"
    assert dehyphenate("Covid-\n19") == "Covid-\n19"


def test_unwrap_lines_keeps_list_items() -> None:
    text = "This is a hard\nwrapped line.\n\n- item one\n- item two"
    assert unwrap_lines(text) == "This is a hard wrapped line.\n\n- item one\n- item two"


def test_fingerprint_is_format_independent() -> None:
    assert fingerprint_text("Hello,   WORLD!\n") == fingerprint_text("hello world")
