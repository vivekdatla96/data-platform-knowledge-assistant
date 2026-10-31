from __future__ import annotations

from dpka.retrieval import BM25, Retriever
from dpka.text import content_terms, stem, tokenize


def test_tokenize_splits_identifiers_and_stems():
    toks = tokenize("DW.FACT_ORDERS orders loaded")
    assert "fact_orders" in toks and "fact" in toks and "order" in toks
    assert stem("calculated") == stem("calculate")
    assert "the" not in content_terms("what is the grain")


def test_bm25_prefers_documents_with_rare_query_terms():
    bm = BM25([["alpha", "beta"], ["gamma", "delta", "alpha"], ["epsilon"]])
    scores = bm.scores(["gamma"])
    assert scores.argmax() == 1 and scores[0] == 0 and scores[2] == 0


def test_hybrid_retrieves_mapping_for_identifier_question(retriever):
    hits = retriever.search("How is NET_AMOUNT calculated in FACT_ORDERS?", k=5)
    docs = [h.chunk.doc_id for h in hits]
    assert "informatica/m_ORDERS_STG_TO_FACT_ORDERS.xml" in docs
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


def test_retrieval_by_category(retriever):
    cases = {
        "How do I backfill FACT_ORDERS?": "runbooks/incremental_load_fact_orders.md",
        "escalate session fails twice in a row": "runbooks/failed_etl_job_triage.md",
        "Which downstream objects depend on DW.FACT_ORDERS lineage?": "snowflake/lineage.json",
        "What is the grain of FACT_RETURNS?": "snowflake/dw_tables.json",
    }
    for q, expected in cases.items():
        docs = [h.chunk.doc_id for h in retriever.search(q, k=5)]
        assert expected in docs, (q, docs)


def test_dense_mode_and_empty_query(indexed, embedder):
    r = Retriever(indexed, embedder, "dense")
    hits = r.search("SCD Type 2 tracked attributes DIM_CUSTOMER", k=3)
    assert len(hits) == 3
    assert hits[0].fused_score == hits[0].dense_score
    assert r.search("   ") == []


def test_coverage_is_low_for_terms_absent_from_corpus(retriever):
    q_in = "How do I backfill FACT_ORDERS?"
    q_out = "Which Kafka topic publishes inventory events?"
    assert retriever.coverage(q_in, retriever.search(q_in)) > 0.9
    assert retriever.coverage(q_out, retriever.search(q_out)) < 0.2
