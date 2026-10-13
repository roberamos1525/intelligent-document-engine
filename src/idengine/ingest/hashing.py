"""Content hashing for exact and near-duplicate detection."""

from __future__ import annotations

import hashlib

from idengine.text.normalize import fingerprint_text

SIMHASH_BITS = 64
_MASK = (1 << SIMHASH_BITS) - 1


def content_hash(data: bytes) -> str:
    """SHA-256 of the raw bytes: identical uploads collapse to one document."""
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """SHA-256 of the canonicalised text, so the same text in TXT and DOCX is detected."""
    return hashlib.sha256(fingerprint_text(text).encode("utf-8")).hexdigest()


def _shingles(tokens: list[str], size: int) -> list[str]:
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def simhash(text: str, *, shingle_size: int = 3) -> int:
    """64-bit SimHash (Charikar) over word shingles.

    Documents that differ only slightly produce fingerprints with a small
    Hamming distance, which is what :func:`hamming_distance` measures.
    """
    tokens = fingerprint_text(text).split()
    weights = [0] * SIMHASH_BITS
    for shingle in _shingles(tokens, shingle_size):
        digest = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for bit in range(SIMHASH_BITS):
            weights[bit] += 1 if digest >> bit & 1 else -1
    value = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            value |= 1 << bit
    return value & _MASK


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def simhash_to_signed(value: int) -> int:
    """Map an unsigned 64-bit value into SQLite's signed INTEGER range."""
    return value - (1 << 64) if value >= 1 << 63 else value


def simhash_from_signed(value: int) -> int:
    return value + (1 << 64) if value < 0 else value
