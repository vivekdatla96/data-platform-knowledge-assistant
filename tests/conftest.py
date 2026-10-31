from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dpka.assistant import Assistant
from dpka.embeddings import HashingEmbedder
from dpka.ingest import ingest
from dpka.providers import ExtractiveProvider
from dpka.retrieval import Retriever
from dpka.store import VectorStore

REPO = Path(__file__).absolute().parents[1]
CORPUS = REPO / "data" / "corpus"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A private, writable copy of the synthetic corpus (the original is never modified)."""
    dst = tmp_path / "corpus"
    shutil.copytree(CORPUS, dst)
    return dst


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path: Path):
    s = VectorStore(tmp_path / "index.sqlite")
    yield s
    s.close()


@pytest.fixture
def indexed(corpus: Path, store: VectorStore, embedder: HashingEmbedder) -> VectorStore:
    report = ingest(corpus, store, embedder)
    assert report.ok, report.failed
    return store


@pytest.fixture
def retriever(indexed: VectorStore, embedder: HashingEmbedder) -> Retriever:
    return Retriever(indexed, embedder, "hybrid")


@pytest.fixture
def assistant(retriever: Retriever) -> Assistant:
    # The hashing embedder is not semantic, so the similarity gate is disabled in tests that use
    # it; the coverage gate and the provider-level checks still apply.
    return Assistant(retriever, ExtractiveProvider(), min_relevance=0.0, min_coverage=0.3)
