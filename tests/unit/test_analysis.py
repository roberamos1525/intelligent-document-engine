from __future__ import annotations

import pytest

from idengine.analysis.dates import find_dates, parse_date
from idengine.analysis.entities import EntityExtractor, summarize_entities
from idengine.analysis.keywords import extract_keywords, rake, repeated_phrases, tfidf_keywords
from idengine.analysis.language import detect_language, ngram_profile, out_of_place
from idengine.analysis.metadata import MetadataExtractor
from idengine.ingest.model import Block, BlockKind, DocType, ExtractedDocument


class TestLanguage:
    @pytest.mark.parametrize(
        ("text", "code"),
        [
            ("This agreement becomes effective on the first day of the following month.", "en"),
            ("El contrato entre las partes entra en vigor el primer día del mes siguiente.", "es"),
            (
                "Le contrat entre les parties entre en vigueur le premier jour du mois suivant.",
                "fr",
            ),
            (
                "Der Vertrag zwischen den Parteien tritt am ersten Tag des Folgemonats in Kraft.",
                "de",
            ),
            (
                "Il contratto tra le parti entra in vigore il primo giorno del mese successivo.",
                "it",
            ),
            ("O contrato entre as partes entra em vigor no primeiro dia do mês seguinte.", "pt"),
            (
                "Het contract tussen de partijen treedt in werking op de eerste dag van de maand.",
                "nl",
            ),
        ],
    )
    def test_detects_supported_languages(self, text: str, code: str) -> None:
        guess = detect_language(text)
        assert guess.code == code
        assert guess.scores[code] == max(guess.scores.values())

    def test_short_text_is_undetermined_unless_hinted(self) -> None:
        assert detect_language("ok").code == "und"
        assert detect_language("ok", hint="de").code == "de"
        assert detect_language("ok", hint="xx").code == "und"

    def test_profile_primitives(self) -> None:
        profile = ngram_profile("abc abc", size=5)
        assert len(profile) == 5
        assert out_of_place(["a"], {"a": 0}) == 0
        assert out_of_place(["zz"], {"a": 0, "b": 1}) == 2


class TestDates:
    def test_formats_are_normalised(self) -> None:
        text = (
            "Signed 2024-03-05, due March 15, 2024, renewed 1st of April 2025, "
            "paid 31/12/2023, reviewed in June 2022."
        )
        assert [d.iso for d in find_dates(text)] == [
            "2024-03-05",
            "2024-03-15",
            "2025-04-01",
            "2023-12-31",
            "2022-06",
        ]

    def test_ambiguous_numeric_dates(self) -> None:
        assert find_dates("on 03/04/2024")[0].iso == "2024-03-04"
        assert find_dates("on 03/04/2024", day_first=True)[0].iso == "2024-04-03"

    def test_invalid_dates_and_modal_may(self) -> None:
        assert find_dates("2024-02-30 and 13/13/2024") == []
        assert find_dates("staff may 2024 plans") == []

    def test_parse_date(self) -> None:
        assert parse_date("2024-01-02T10:00:00Z") == "2024-01-02"
        assert parse_date("Jan 2, 2024") == "2024-01-02"
        assert parse_date("garbage") is None
        assert parse_date("") is None


class TestEntities:
    def test_all_labels(self) -> None:
        text = (
            "Invoice from Acme Widgets Inc. dated March 5, 2024. Contact Dr. Jane Smith at "
            "jane.smith@acme.example or +1 415-555-0199. Total: $12,450.00, later USD 3.5 million "
            "and 200 euros, a 15% discount. Offices in San Francisco and the State of Oregon. "
            "Department of Energy approved it. Visit https://acme.example/pay. "
            "Approved by Maria Lopez. Chief Financial Officer, Tom Baker agreed."
        )
        found = {(e.label, e.normalized) for e in EntityExtractor().extract(text)}
        expected = {
            ("ORG", "Acme Widgets Inc."),
            ("DATE", "2024-03-05"),
            ("PERSON", "Jane Smith"),
            ("EMAIL", "jane.smith@acme.example"),
            ("PHONE", "+14155550199"),
            ("MONEY", "USD 12450.00"),
            ("MONEY", "USD 3500000.00"),
            ("MONEY", "EUR 200.00"),
            ("PERCENT", "15%"),
            ("LOCATION", "San Francisco"),
            ("LOCATION", "State of Oregon"),
            ("ORG", "Department of Energy"),
            ("URL", "https://acme.example/pay"),
            ("PERSON", "Maria Lopez"),
            ("PERSON", "Tom Baker"),
        }
        assert expected <= found

    def test_defined_terms_are_not_organisations(self) -> None:
        labels = {
            e.text for e in EntityExtractor().extract("The Company shall pay. The Group agrees.")
        }
        assert "Company" not in labels and "The Company" not in labels

    def test_custom_gazetteers(self) -> None:
        extractor = EntityExtractor(organizations={"Initech"}, locations={"Gotham"})
        found = {
            (e.label, e.text) for e in extractor.extract("Initech opened an office in Gotham.")
        }
        assert found == {("ORG", "Initech"), ("LOCATION", "Gotham")}

    def test_summary_groups_unique_values(self) -> None:
        entities = EntityExtractor().extract("Mail a@x.io, then a@x.io, then b@x.io.")
        assert summarize_entities(entities) == {"EMAIL": ["a@x.io", "b@x.io"]}


class TestKeywords:
    TEXT = (
        "Machine learning models require training data. Training data quality determines model "
        "accuracy. Data labeling and data cleaning improve machine learning accuracy."
    )

    def test_repeated_phrases_and_rake(self) -> None:
        phrases = [k.phrase for k in repeated_phrases(self.TEXT)]
        assert phrases[:2] == ["machine learning", "training data"]
        assert "data labeling" in [k.phrase for k in rake(self.TEXT)]

    def test_tfidf_prefers_rare_terms_with_corpus_statistics(self) -> None:
        common = {"data": 100}
        terms = tfidf_keywords(
            self.TEXT, document_frequency=lambda t: common.get(t, 1), total_documents=100
        )
        assert terms[0].phrase != "data"

    def test_blend_has_no_duplicate_stems(self) -> None:
        keywords = extract_keywords(self.TEXT, top_k=6)
        assert keywords[0] == "machine learning"
        assert len(keywords) == len(set(keywords)) <= 6

    def test_empty_inputs(self) -> None:
        assert rake("the and of") == []
        assert tfidf_keywords("") == []


class TestMetadata:
    def test_combines_native_and_text_signals(self) -> None:
        doc = ExtractedDocument(
            doc_type=DocType.TXT,
            blocks=[
                Block(BlockKind.HEADING, "Quarterly Report", level=1),
                Block(BlockKind.PARAGRAPH, "By Ann Lee and Bob Kim"),
                Block(BlockKind.PARAGRAPH, "Date: 5 March 2024"),
                Block(BlockKind.PARAGRAPH, "Revenue reached $2 million in Germany."),
            ],
            title="Quarterly Report",
        )
        meta = MetadataExtractor().extract(doc, filename="q.txt")
        assert meta.title == "Quarterly Report"
        assert meta.authors == ["Ann Lee", "Bob Kim"]
        assert meta.document_date == "2024-03-05"
        assert meta.language == "en"
        assert meta.entities["MONEY"] == ["USD 2000000.00"]
        assert meta.heading_count == 1
        assert meta.word_count > 10

    def test_title_falls_back_to_filename(self) -> None:
        doc = ExtractedDocument(
            doc_type=DocType.TXT,
            blocks=[Block(BlockKind.TABLE, "a | b\n" * 80)],
        )
        assert MetadataExtractor().extract(doc, filename="my_report-v2.txt").title == "my report v2"

    def test_keywords_skipped_for_non_english(self) -> None:
        doc = ExtractedDocument(
            doc_type=DocType.TXT,
            blocks=[
                Block(BlockKind.PARAGRAPH, "Der Vertrag tritt am ersten Tag des Monats in Kraft.")
            ],
        )
        meta = MetadataExtractor().extract(doc)
        assert meta.language == "de"
        assert meta.keywords == []
