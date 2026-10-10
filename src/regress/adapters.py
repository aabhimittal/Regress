"""Helpers for agents whose memories and answers are text, not vectors.

Most real agent memories are strings (reflections, summaries, skill descriptions). REGRESS
needs vectors for diversity, and stable identities for drift. These helpers provide
dependency-free defaults; swap in a real sentence-embedding model when you have one —
anything with ``embed(list[str]) -> (n, d) array`` works.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Callable, Protocol, Sequence

import numpy as np

from .core import MemorySnapshot

__all__ = ["Embedder", "HashingEmbedder", "text_key", "memory_from_texts"]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing of word unigrams and character n-grams, L2-normalised.

    Deterministic, no model download, and good enough to see memories converge (near-
    duplicate summaries hash to near-identical vectors). It does not know that "car" and
    "automobile" are related; use a semantic model for that.
    """

    def __init__(self, dim: int = 256, char_ngrams: tuple[int, ...] = (3, 4)) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim
        self.char_ngrams = char_ngrams

    def _features(self, text: str) -> list[str]:
        t = _normalize(text)
        feats = [f"w:{w}" for w in t.split()]
        padded = f" {t} "
        for n in self.char_ngrams:
            feats += [f"c{n}:{padded[i:i + n]}" for i in range(max(0, len(padded) - n + 1))]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for f in self._features(str(text)):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                out[row, h % self.dim] += 1.0 if (h >> 63) else -1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


def _normalize(text: str, strip_punct: bool = True) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    if strip_punct:
        t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def text_key(strip_punct: bool = True, max_len: int = 64) -> Callable[[object], str]:
    """An ``output_key`` that ignores case, Unicode form, whitespace and (optionally)
    punctuation, so "Paris." and " paris" count as the same answer."""
    from ._canon import output_key

    def key(output: object) -> str:
        if isinstance(output, str):
            norm = _normalize(output, strip_punct)
            if len(norm) <= max_len:
                return "t:" + norm
            return "t#" + hashlib.sha256(norm.encode()).hexdigest()[:24]
        return output_key(output)

    return key


def memory_from_texts(ids: Sequence[str], texts: Sequence[str], embedder: Embedder | None = None) -> MemorySnapshot:
    """Build a :class:`MemorySnapshot` from text memories."""
    if len(ids) != len(texts):
        raise ValueError(f"{len(ids)} ids but {len(texts)} texts")
    emb = (embedder or HashingEmbedder()).embed(list(texts)) if texts else np.zeros((0, 0))
    return MemorySnapshot(ids=list(ids), embeddings=np.asarray(emb, dtype=float))
