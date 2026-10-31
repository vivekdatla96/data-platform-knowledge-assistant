"""Integration tests with the real open-source embedding model.

Skipped unless DPKA_RUN_MODEL_TESTS=1 (the first run downloads ~70 MB from Hugging Face).
"""

from __future__ import annotations

import os

import pytest

from dpka.assistant import Assistant
from dpka.config import Settings
from dpka.ingest import ingest
from dpka.providers import ExtractiveProvider
from dpka.retrieval import Retriever

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("DPKA_RUN_MODEL_TESTS") != "1",
        reason="set DPKA_RUN_MODEL_TESTS=1 to run real-model tests",
    ),
]


@pytest.fixture(scope="module")
def real_assistant(tmp_path_factory):
    from conftest import CORPUS

    from dpka.embeddings import FastEmbedEmbedder
    from dpka.store import VectorStore

    emb = FastEmbedEmbedder()
    store = VectorStore(tmp_path_factory.mktemp("m") / "i.sqlite")
    assert ingest(CORPUS, store, emb).ok
    s = Settings()
    return Assistant(
        Retriever(store, emb, "hybrid"),
        ExtractiveProvider(),
        min_relevance=s.min_relevance,
        min_coverage=s.min_coverage,
    )


def test_real_model_answers_and_cites(real_assistant):
    res = real_assistant.ask("How do I backfill FACT_ORDERS for a past date range?")
    assert res.status == "answered"
    assert res.citations[0]["doc_id"] == "runbooks/incremental_load_fact_orders.md"


def test_real_model_abstains_on_out_of_corpus_question(real_assistant):
    assert (
        real_assistant.ask("What is the Snowflake account URL for production?").status
        == "abstained"
    )
