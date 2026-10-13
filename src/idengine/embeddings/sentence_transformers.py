"""Optional adapter for the ``sentence-transformers`` package.

The package is *not* a dependency of this project. The adapter imports it
lazily and raises a configuration error with installation instructions when
it is missing. Models are loaded from the local Hugging Face cache; the engine
never downloads anything unless the user installs and configures this
provider explicitly.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from idengine.embeddings.base import EmbeddingProvider, Matrix, l2_normalize
from idengine.errors import ConfigurationError


class SentenceTransformerEmbedder(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise ConfigurationError(
                "embeddings.provider='sentence-transformers' requires the optional package: "
                "pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = module.SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if not isinstance(dim, int) or dim <= 0:
            raise ConfigurationError(f"model {model_name} did not report an embedding dimension")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=False
        )
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def signature(self) -> dict[str, Any]:
        return {"provider": self.name, "dim": self._dim, "model": self.model_name}
