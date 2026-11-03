"""Optional cross-encoder reranking.

A cross-encoder reads the question and a passage together and returns a relevance logit, which is
more precise than comparing two independently computed embeddings but much slower. It is used
only on the top candidates from hybrid retrieval.

Default model: ``Xenova/ms-marco-MiniLM-L-6-v2`` (Apache-2.0, about 80 MB, ONNX via fastembed).
A logit of 0 corresponds to a predicted relevance probability of 0.5.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


class FastEmbedReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # heavy import, lazy

        self._model = TextCrossEncoder(model_name=model_name)
        self.name = f"fastembed-rerank:{model_name}"

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(list(self._model.rerank(query, texts)), dtype=np.float32)


def build_reranker(model_name: str) -> Reranker | None:
    """``"none"`` (or empty) disables reranking."""
    if not model_name or model_name.lower() == "none":
        return None
    return FastEmbedReranker(model_name)
