from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import CORPUS

from dpka.embeddings import HashingEmbedder
from dpka.ingest import ingest
from dpka.retrieval import Retriever
from dpka.store import EmbeddingModelMismatch, VectorStore

RUNBOOK = "runbooks/incremental_load_fact_orders.md"


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_first_ingest_indexes_every_document(corpus, store, embedder):
    report = ingest(corpus, store, embedder)
    assert report.ok
    assert len(report.added) == 12
    assert report.chunks_embedded == store.stats()["chunks"] > 0
    types = store.stats()["chunks_by_type"]
    assert set(types) == {
        "snowflake_metadata",
        "snowflake_lineage",
        "informatica_mapping",
        "runbook",
    }


def test_reingest_is_a_no_op(corpus, store, embedder):
    ingest(corpus, store, embedder)
    before = store.load_all()[0]
    report = ingest(corpus, store, embedder)
    assert len(report.unchanged) == 12
    assert report.added == report.updated == report.deleted == []
    assert report.chunks_embedded == 0
    assert [c.chunk_id for c in store.load_all()[0]] == [c.chunk_id for c in before]


def test_updated_document_replaces_chunks_and_reuses_unchanged_embeddings(corpus, store, embedder):
    ingest(corpus, store, embedder)
    path = corpus / RUNBOOK
    old = "`LOOKBACK_MINUTES` is 120."
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(
        text.replace(old, "`LOOKBACK_MINUTES` is 240 (raised for the zebra test)."),
        encoding="utf-8",
    )

    report = ingest(corpus, store, embedder)
    assert report.updated == [RUNBOOK]
    assert report.chunks_embedded == 1  # only the edited section
    assert report.chunks_reused == store.documents()[RUNBOOK].chunk_count - 1

    chunks = [c for c in store.load_all()[0] if c.doc_id == RUNBOOK]
    joined = "\n".join(c.text for c in chunks)
    assert "zebra" in joined and "is 120." not in joined
    hits = Retriever(store, embedder).search("zebra test lookback", k=1)
    assert hits[0].chunk.doc_id == RUNBOOK


def test_deleted_document_is_removed_from_index_and_results(corpus, store, embedder):
    ingest(corpus, store, embedder)
    doomed = "informatica/m_PRODUCTS_FULL_REFRESH.xml"
    (corpus / doomed).unlink()

    report = ingest(corpus, store, embedder)
    assert report.deleted == [doomed]
    assert doomed not in store.documents()
    assert all(c.doc_id != doomed for c in store.load_all()[0])
    hits = Retriever(store, embedder).search("How is DIM_PRODUCT loaded? truncate insert", k=10)
    assert all(h.chunk.doc_id != doomed for h in hits)


def test_invalid_update_keeps_last_known_good_version(corpus, store, embedder):
    ingest(corpus, store, embedder)
    target = "snowflake/lineage.json"
    good_hash = store.documents()[target].content_hash
    (corpus / target).write_text('{"format": "dpka.snowflake_lineage.v1"', encoding="utf-8")

    report = ingest(corpus, store, embedder)
    assert target in report.failed
    assert store.documents()[target].content_hash == good_hash
    assert any(c.doc_id == target for c in store.load_all()[0])


def test_new_invalid_document_is_reported_not_indexed(corpus, store, embedder):
    (corpus / "runbooks" / "broken.md").write_text("no heading", encoding="utf-8")
    report = ingest(corpus, store, embedder)
    assert "runbooks/broken.md" in report.failed
    assert "runbooks/broken.md" not in store.documents()
    assert len(report.added) == 12


def test_hidden_and_unsupported_files_are_ignored(corpus, store, embedder):
    (corpus / ".hidden.md").write_text("# Hidden\n\n## A\n\nsecret", encoding="utf-8")
    (corpus / "notes.txt").write_text("ignored", encoding="utf-8")
    report = ingest(corpus, store, embedder)
    assert ".hidden.md" not in report.added and "notes.txt" not in report.added


def test_ingest_never_modifies_the_source_corpus(tmp_path, embedder):
    before = _tree_hash(CORPUS)
    store = VectorStore(tmp_path / "i.sqlite")
    ingest(CORPUS, store, embedder)
    ingest(CORPUS, store, embedder)
    store.close()
    assert _tree_hash(CORPUS) == before


def test_mixing_embedding_models_is_refused(corpus, store, embedder):
    ingest(corpus, store, embedder)
    with pytest.raises(EmbeddingModelMismatch):
        ingest(corpus, store, HashingEmbedder(dim=256))
    report = ingest(corpus, store, HashingEmbedder(dim=256), rebuild=True)
    assert len(report.added) == 12


def test_missing_corpus_dir_raises(tmp_path, store, embedder):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "nope", store, embedder)
