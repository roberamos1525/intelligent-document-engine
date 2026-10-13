"""Golden evaluation dataset: schema, loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from idengine.errors import ConfigurationError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusDocument(_Strict):
    file: str
    category: str | None = None


class RetrievalCase(_Strict):
    query: str = Field(min_length=1)
    relevant: list[str] = Field(min_length=1)
    graded: dict[str, float] = Field(default_factory=dict)


class QACase(_Strict):
    question: str = Field(min_length=3)
    answers: list[str] = Field(default_factory=list)
    document: str | None = None
    answerable: bool = True


class LabeledText(_Strict):
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ClassificationSplit(_Strict):
    train: list[LabeledText] = Field(min_length=4)
    test: list[LabeledText] = Field(min_length=1)


class GoldenDataset(_Strict):
    name: str
    documents: dict[str, CorpusDocument]
    retrieval: list[RetrievalCase] = Field(default_factory=list)
    qa: list[QACase] = Field(default_factory=list)
    classification: ClassificationSplit | None = None

    @model_validator(mode="after")
    def _references_exist(self) -> GoldenDataset:
        known = set(self.documents)
        for case in self.retrieval:
            missing = (set(case.relevant) | set(case.graded)) - known
            if missing:
                raise ValueError(
                    f"retrieval case '{case.query}' references unknown {sorted(missing)}"
                )
        for qa in self.qa:
            if qa.document is not None and qa.document not in known:
                raise ValueError(f"qa case '{qa.question}' references unknown '{qa.document}'")
            if qa.answerable and not qa.answers:
                raise ValueError(f"answerable qa case '{qa.question}' needs answers")
        return self


def load_dataset(directory: Path) -> tuple[GoldenDataset, Path]:
    """Load ``dataset.json`` from ``directory`` and check that corpus files exist."""
    path = directory / "dataset.json"
    if not path.is_file():
        raise ConfigurationError(f"evaluation dataset not found: {path}")
    try:
        dataset = GoldenDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ConfigurationError(f"invalid evaluation dataset {path}: {exc}") from exc
    for key, doc in dataset.documents.items():
        if not (directory / doc.file).is_file():
            raise ConfigurationError(f"corpus file for '{key}' not found: {doc.file}")
    return dataset, directory
