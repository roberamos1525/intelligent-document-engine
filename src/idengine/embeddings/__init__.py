"""Embedding providers behind the :class:`EmbeddingProvider` interface."""

from __future__ import annotations

from idengine.config import EmbeddingSettings
from idengine.embeddings.base import EmbeddingProvider, l2_normalize
from idengine.embeddings.hashing import HashingEmbedder
from idengine.embeddings.lsa import LsaEmbedder


def create_embedder(settings: EmbeddingSettings) -> EmbeddingProvider:
    """Build the configured provider."""
    if settings.provider == "hashing":
        return HashingEmbedder(settings.dim, seed=settings.seed, use_bigrams=settings.bigrams)
    if settings.provider == "lsa":
        return LsaEmbedder(settings.dim, seed=settings.seed)
    from idengine.embeddings.sentence_transformers import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings.model_name)


__all__ = [
    "EmbeddingProvider",
    "HashingEmbedder",
    "LsaEmbedder",
    "create_embedder",
    "l2_normalize",
]
