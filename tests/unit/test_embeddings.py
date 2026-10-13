from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from idengine.config import EmbeddingSettings
from idengine.embeddings import HashingEmbedder, LsaEmbedder, create_embedder, l2_normalize
from idengine.embeddings.lsa import CsrMatrix, randomized_svd
from idengine.errors import ConfigurationError, ModelNotTrainedError, ValidationError

CORPUS = [
    "The invoice total is due within thirty days of the billing date.",
    "Late invoice payments incur a monthly fee on the outstanding balance.",
    "Employees may work remotely three days per week after probation.",
    "Remote employees must use the corporate VPN for all work.",
    "The autoscaler adds replicas when CPU utilisation is high.",
    "Roll back a failed deployment with kubectl rollout undo.",
]


class TestHashing:
    def test_deterministic_unit_vectors(self) -> None:
        embedder = HashingEmbedder(dim=64, seed=1)
        a = embedder.embed(CORPUS)
        b = HashingEmbedder(dim=64, seed=1).embed(CORPUS)
        assert a.shape == (6, 64) and a.dtype == np.float32
        assert np.allclose(a, b)
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
        assert not np.allclose(a, HashingEmbedder(dim=64, seed=2).embed(CORPUS))

    def test_similar_texts_score_higher(self) -> None:
        embedder = HashingEmbedder(dim=256)
        vectors = embedder.embed(CORPUS)
        query = embedder.embed_query("When is the invoice payment due?")
        scores = vectors @ query
        assert int(np.argmax(scores)) in (0, 1)
        assert scores[0] > scores[4]

    def test_empty_text_is_zero_vector_and_features(self) -> None:
        embedder = HashingEmbedder(dim=16, use_bigrams=False)
        assert not embedder.embed(["the of and"]).any()
        assert "invoic" in embedder.features("invoices")
        assert all("_" not in f for f in embedder.features("big red invoices"))
        with pytest.raises(ValueError):
            HashingEmbedder(dim=4)

    def test_signature(self) -> None:
        assert HashingEmbedder(dim=32, seed=5).signature() == {
            "provider": "hashing",
            "dim": 32,
            "seed": 5,
            "bigrams": True,
        }


class TestLsa:
    def test_fit_embed_and_persist(self, tmp_path: Path) -> None:
        embedder = LsaEmbedder(dim=4)
        assert embedder.requires_fit and not embedder.is_fitted
        with pytest.raises(ModelNotTrainedError):
            embedder.embed(["x"])
        embedder.fit(CORPUS)
        vectors = embedder.embed(CORPUS)
        assert vectors.shape == (6, 4)
        query = embedder.embed_query("remote work VPN")
        assert int(np.argmax(vectors @ query)) in (2, 3)
        embedder.save(tmp_path)
        restored = LsaEmbedder(dim=4)
        assert restored.load(tmp_path)
        assert np.allclose(restored.embed(CORPUS), vectors, atol=1e-6)
        assert not LsaEmbedder(dim=8).load(tmp_path)  # dimensionality changed: refit
        assert not LsaEmbedder(dim=4).load(tmp_path / "missing")

    def test_fit_validation(self) -> None:
        with pytest.raises(ValidationError):
            LsaEmbedder(dim=2).fit(["only one"])
        with pytest.raises(ValidationError):
            LsaEmbedder(dim=2).fit(["the of", "and the"])

    def test_randomized_svd_matches_exact_svd(self) -> None:
        rng = np.random.default_rng(0)
        dense = rng.standard_normal((40, 6)) @ rng.standard_normal((6, 30))  # rank 6
        dense[dense < 0.5] = 0.0
        rows, cols = np.nonzero(dense)
        indptr = np.searchsorted(rows, np.arange(dense.shape[0] + 1))
        matrix = CsrMatrix(
            indptr.astype(np.int64), cols.astype(np.int64), dense[rows, cols], dense.shape
        )
        assert np.allclose(matrix.dot(np.eye(30)), dense)
        assert np.allclose(matrix.tdot(np.eye(40)), dense.T)
        _, s, _ = randomized_svd(matrix, 4, seed=1)
        exact = np.linalg.svd(dense, compute_uv=False)[:4]
        assert np.allclose(s, exact, rtol=1e-3)


class TestFactoryAndOptionalAdapter:
    def test_factory(self) -> None:
        assert isinstance(create_embedder(EmbeddingSettings()), HashingEmbedder)
        assert isinstance(create_embedder(EmbeddingSettings(provider="lsa", dim=16)), LsaEmbedder)

    def test_missing_sentence_transformers_is_a_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ConfigurationError, match="pip install sentence-transformers"):
            create_embedder(EmbeddingSettings(provider="sentence-transformers"))

    def test_sentence_transformers_adapter_with_fake_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeModel:
            def __init__(self, name: str) -> None:
                self.name = name

            def get_sentence_embedding_dimension(self) -> int:
                return 3

            def encode(self, texts: list[str], **_: object) -> np.ndarray:
                return np.array([[len(t), 1.0, 0.0] for t in texts])

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embedder = create_embedder(
            EmbeddingSettings(provider="sentence-transformers", model_name="m")
        )
        vectors = embedder.embed(["ab", "abcd"])
        assert embedder.dim == 3
        assert np.allclose(
            vectors, l2_normalize(np.array([[2, 1, 0], [4, 1, 0]], dtype=np.float32))
        )
        assert embedder.embed([]).shape == (0, 3)
        assert embedder.signature()["model"] == "m"
