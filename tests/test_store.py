from dataclasses import replace

import numpy as np
import pytest

from dpka.store import EmbeddingModelMismatch


def test_same_model_name_with_changed_dimension_is_rejected(indexed, embedder):
    before = indexed.stats()
    with pytest.raises(EmbeddingModelMismatch, match="dimensions"):
        indexed.bind_embedding_model(embedder.name, embedder.dim + 1)
    assert indexed.stats() == before
    assert indexed.get_meta("embedding_dim") == str(embedder.dim)


@pytest.mark.parametrize("dim", [0, -1, True, 1.5])
def test_invalid_dimension_does_not_bind_model(store, dim):
    with pytest.raises(ValueError, match="positive integer"):
        store.bind_embedding_model("bad", dim)
    assert store.get_meta("embedding_model") is None


@pytest.mark.parametrize("invalid", ["flat", "width", "rows", "nan", "inf", "overflow"])
def test_invalid_vectors_preserve_existing_document(indexed, invalid):
    chunks, matrix = indexed.load_all()
    record = next(iter(indexed.documents().values()))
    selected = [i for i, chunk in enumerate(chunks) if chunk.doc_id == record.doc_id]
    document_chunks = [chunks[i] for i in selected]
    vectors = matrix[selected].astype(np.float64)
    if invalid == "flat":
        vectors = vectors.ravel()
    elif invalid == "width":
        vectors = vectors[:, :-1]
    elif invalid == "rows":
        vectors = vectors[:-1]
    else:
        vectors[0, 0] = {"nan": np.nan, "inf": np.inf, "overflow": 1e100}[invalid]
    with pytest.raises(ValueError, match="shape|finite"):
        indexed.replace_document(replace(record, content_hash="bad"), document_chunks, vectors)
    assert indexed.documents()[record.doc_id] == record
    after_chunks, after_matrix = indexed.load_all()
    assert after_chunks == chunks
    np.testing.assert_array_equal(after_matrix, matrix)


@pytest.mark.parametrize("invalid", ["count", "ownership"])
def test_inconsistent_provenance_is_rejected(indexed, invalid):
    chunks, matrix = indexed.load_all()
    record = next(iter(indexed.documents().values()))
    selected = [i for i, chunk in enumerate(chunks) if chunk.doc_id == record.doc_id]
    document_chunks = [chunks[i] for i in selected]
    bad_record = (
        replace(record, chunk_count=record.chunk_count + 1) if invalid == "count" else record
    )
    if invalid == "ownership":
        document_chunks[0] = replace(document_chunks[0], doc_id="another-document")
    with pytest.raises(ValueError, match="chunk count|belong"):
        indexed.replace_document(bad_record, document_chunks, matrix[selected])
    assert indexed.documents()[record.doc_id] == record
    np.testing.assert_array_equal(indexed.load_all()[1], matrix)


def test_valid_float64_vectors_round_trip_as_float32(indexed):
    chunks, matrix = indexed.load_all()
    record = next(iter(indexed.documents().values()))
    selected = [i for i, chunk in enumerate(chunks) if chunk.doc_id == record.doc_id]
    indexed.replace_document(
        record, [chunks[i] for i in selected], matrix[selected].astype(np.float64)
    )
    result = indexed.load_all()[1]
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, matrix)
