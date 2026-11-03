"""Reranking and the reranker-rescue abstention rule, using a deterministic fake reranker."""

from __future__ import annotations

import numpy as np
import pytest

from dpka.assistant import Assistant
from dpka.config import Settings
from dpka.providers import ExtractiveProvider
from dpka.reranker import build_reranker
from dpka.retrieval import Retriever


class KeywordReranker:
    """Scores a passage by whether it contains a marker string; records every call."""

    name = "fake-rerank"

    def __init__(self, marker: str, hit: float = 5.0, miss: float = -5.0):
        self.marker, self.hit, self.miss = marker, hit, miss
        self.calls: list[int] = []

    def score(self, query, texts):
        self.calls.append(len(texts))
        return np.array([self.hit if self.marker in t else self.miss for t in texts])


def test_reranker_reorders_within_the_candidate_pool(indexed, embedder):
    q = "How do I backfill FACT_ORDERS for a past date range?"
    base = Retriever(indexed, embedder).search(q, k=5)
    pool = Retriever(indexed, embedder).search(q, k=10)
    # Promote whichever pool chunk the plain ranking put last.
    marker = pool[-1].chunk.text[:40]
    rr = KeywordReranker(marker)
    hits = Retriever(indexed, embedder, reranker=rr, rerank_candidates=10).search(q, k=5)

    assert rr.calls == [10]
    assert hits[0].chunk.chunk_id == pool[-1].chunk.chunk_id
    assert hits[0].rerank_score == 5.0 and hits[0].rank == 1
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.rerank_score is not None for h in hits)
    assert all(h.rerank_score is None for h in base)


def test_pool_is_never_smaller_than_k(indexed, embedder):
    rr = KeywordReranker("zzz")
    hits = Retriever(indexed, embedder, reranker=rr, rerank_candidates=2).search("FACT_ORDERS", 5)
    assert rr.calls == [5] and len(hits) == 5


def _assistant(retriever, **gate):
    return Assistant(retriever, ExtractiveProvider(), **gate)


def test_reranker_rescues_a_question_the_first_gate_rejects(indexed, embedder):
    q = "Is it safe to rerun a failed FACT_ORDERS session?"
    strict = {"min_relevance": 0.999, "min_coverage": 0.3}  # similarity gate always fails
    plain = _assistant(Retriever(indexed, embedder), **strict).ask(q)
    assert plain.status == "abstained"

    rescued = _assistant(
        Retriever(indexed, embedder, reranker=KeywordReranker("rerun")), **strict
    ).ask(q)
    assert rescued.status == "answered"
    assert rescued.gate["rescued_by_reranker"] == 1.0
    assert rescued.gate["best_rerank_score"] == 5.0
    assert all(e["rerank_score"] is not None for e in rescued.evidence)


def test_reranker_below_threshold_does_not_rescue(indexed, embedder):
    rr = KeywordReranker("no-such-marker")  # every score is -5
    res = _assistant(
        Retriever(indexed, embedder, reranker=rr), min_relevance=0.999, min_coverage=0.3
    ).ask("Is it safe to rerun a failed FACT_ORDERS session?")
    assert res.status == "abstained"
    assert "best reranker score -5.00" in res.abstain_reason
    assert "rescued_by_reranker" not in res.gate


def test_reranker_never_vetoes_a_question_the_first_gate_accepts(indexed, embedder):
    rr = KeywordReranker("no-such-marker")
    res = _assistant(
        Retriever(indexed, embedder, reranker=rr), min_relevance=0.0, min_coverage=0.3
    ).ask("Is it safe to rerun a failed FACT_ORDERS session?")
    assert res.status == "answered"


def test_out_of_corpus_question_still_abstains_with_a_low_reranker(indexed, embedder):
    rr = KeywordReranker("no-such-marker")
    res = _assistant(
        Retriever(indexed, embedder, reranker=rr), min_relevance=0.0, min_coverage=0.3
    ).ask("Which Kafka topic publishes inventory events?")
    assert res.status == "abstained"


@pytest.mark.parametrize("value", ["none", "NONE", ""])
def test_reranker_can_be_disabled(value):
    assert build_reranker(value) is None


def test_rerank_settings_validation():
    s = Settings.from_env(
        {"DPKA_RERANKER_MODEL": "none", "DPKA_RERANK_CANDIDATES": "3", "DPKA_TOP_K": "8"}
    )
    assert s.reranker_model == "none" and s.rerank_candidates == 3 and s.top_k == 8
    with pytest.raises(ValueError, match="DPKA_RERANK_CANDIDATES"):
        Settings(rerank_candidates=0)


def test_reranking_cannot_veto_through_the_evidence_it_selects(indexed, embedder):
    """Devin review on PR #5: the first gate must use the pre-rerank top-k.

    With k=1 the plain top hit clears the gate. The reranker promotes an off-topic chunk with a
    negative logit (so no rescue). Judged on that chunk the gate would fail; judged on the
    baseline it passes, which is the documented behaviour.
    """
    q = "Is it safe to rerun a failed FACT_ORDERS session?"
    gate = {"min_relevance": 0.0, "min_coverage": 0.3, "top_k": 1}
    rr = KeywordReranker("WH_ANALYTICS_S", hit=-1.0, miss=-5.0)
    retriever = Retriever(indexed, embedder, reranker=rr, rerank_candidates=54)

    baseline, final = retriever.search_with_baseline(q, k=1)
    assert "WH_ANALYTICS_S" in final[0].chunk.text
    assert final[0].chunk.chunk_id != baseline[0].chunk.chunk_id
    assert retriever.coverage(q, final) < 0.3 <= retriever.coverage(q, baseline)

    res = _assistant(retriever, **gate).ask(q)
    # The gate must not reject it. (The offline provider may still decline on its own, since the
    # promoted evidence is off-topic; that is a different, legitimate abstention.)
    assert not (res.abstain_reason or "").startswith("Insufficient evidence")
    assert res.gate["term_coverage"] >= 0.3
    assert "rescued_by_reranker" not in res.gate
    assert res.evidence[0]["chunk_id"] == final[0].chunk.chunk_id


def test_search_returns_identical_lists_without_a_reranker(retriever):
    baseline, final = retriever.search_with_baseline("backfill FACT_ORDERS", k=5)
    assert baseline == final == retriever.search("backfill FACT_ORDERS", k=5)
