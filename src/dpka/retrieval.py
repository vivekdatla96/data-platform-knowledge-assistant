"""Dense and hybrid (dense + BM25) retrieval over the local index."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from dpka.embeddings import Embedder
from dpka.models import Chunk
from dpka.store import EmbeddingModelMismatch, VectorStore
from dpka.text import content_terms, tokenize

RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float  # cosine similarity in [-1, 1]
    bm25_score: float
    fused_score: float
    rank: int


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each element when sorted descending (ties broken by position)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.lens.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        for i, tf in enumerate(self.tfs):
            denom_norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg_len or 1))
            s = 0.0
            for t in set(query):
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + denom_norm)
            out[i] = s
        return out


class Retriever:
    """Loads the index into memory once; call :meth:`refresh` after re-ingesting."""

    def __init__(self, store: VectorStore, embedder: Embedder, mode: str = "hybrid"):
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        self.store, self.embedder, self.mode = store, embedder, mode
        self.refresh()

    def refresh(self) -> None:
        # Check name and dimension: a same-named embedder with a different dimension would
        # otherwise fail later inside the similarity matrix product.
        model = self.store.get_meta("embedding_model")
        dim = self.store.get_meta("embedding_dim")
        if model is not None and (model != self.embedder.name or dim != str(self.embedder.dim)):
            raise EmbeddingModelMismatch(
                f"index built with {model!r} ({dim} dimensions), query embedder is "
                f"{self.embedder.name!r} ({self.embedder.dim} dimensions); "
                "re-run ingestion with --rebuild"
            )
        self.chunks, self.matrix = self.store.load_all()
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks or not query.strip():
            return []
        q = self.embedder.embed_query(query)
        dense = self.matrix @ q
        bm25 = self.bm25.scores(tokenize(query))

        if self.mode == "dense":
            fused = dense.astype(np.float64)
        else:
            # Reciprocal rank fusion. A chunk with no keyword overlap gets no BM25 credit.
            fused = 1.0 / (RRF_K + _ranks(dense))
            fused += np.where(bm25 > 0, 1.0 / (RRF_K + _ranks(bm25)), 0.0)

        top = np.argsort(-fused, kind="stable")[:k]
        return [
            RetrievedChunk(self.chunks[i], float(dense[i]), float(bm25[i]), float(fused[i]), r)
            for r, i in enumerate(top, start=1)
        ]

    def coverage(self, query: str, hits: list[RetrievedChunk]) -> float:
        """IDF-weighted share of the query's content terms that appear in the retrieved text.

        Terms that never occur in the corpus get the highest weight, so a question about
        something the corpus does not mention (a URL, a password, a Kafka topic) scores low
        even when embeddings find superficially similar chunks.
        """
        terms = content_terms(query)
        if not terms or not hits:
            return 0.0
        unknown = max(self.bm25.idf.values(), default=1.0) + 1.0
        seen = set().union(*(set(tokenize(h.chunk.text)) for h in hits))
        weights = {t: self.bm25.idf.get(t, unknown) for t in terms}
        covered = sum(w for t, w in weights.items() if t in seen)
        return covered / sum(weights.values())
