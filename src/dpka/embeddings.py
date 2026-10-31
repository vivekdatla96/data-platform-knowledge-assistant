"""Embedding backends.

``FastEmbedEmbedder`` runs an open-source model (default ``BAAI/bge-small-en-v1.5``, MIT
licensed, 384 dimensions) locally through ONNX Runtime. No API key, no GPU, no PyTorch.

``HashingEmbedder`` is a deterministic bag-of-words hashing embedder. It has no semantic
understanding; it exists so unit tests and CI run fast, offline and reproducibly.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from dpka.models import sha256_hex
from dpka.text import tokenize

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported lazily: heavy import, optional in tests

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.name = f"fastembed:{model_name}"
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.array(list(self._model.embed(texts, batch_size=32))))

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(list(self._model.embed([self._query_prefix + text])))
        return _normalize(vec)[0]


class HashingEmbedder:
    """Deterministic feature hashing. Test/CI only: not a semantic model."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(sha256_hex(tok)[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name)
    if backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedding backend {backend!r}")
