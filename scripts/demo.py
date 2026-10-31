"""End-to-end local demonstration (offline, no paid API).

1. Copies the synthetic corpus to a temporary directory (the original stays untouched).
2. Ingests it, then re-ingests to show the run is a no-op.
3. Asks representative questions: incremental loads, failed jobs, mappings, lineage, an
   out-of-corpus question (abstention) and one that retrieves the planted prompt injection.
4. Edits one runbook and deletes one mapping, re-ingests, and shows the index followed.

Usage:  python scripts/demo.py [--embedder fastembed|hashing]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from dpka.assistant import Assistant
from dpka.config import Settings
from dpka.embeddings import build_embedder
from dpka.ingest import ingest
from dpka.providers import ExtractiveProvider
from dpka.retrieval import Retriever
from dpka.store import VectorStore

REPO = Path(__file__).absolute().parents[1]

QUESTIONS = [
    "How does the FACT_ORDERS incremental load decide which source rows to extract?",
    "What should I do when an ETL session fails with a statement timeout?",
    "How is NET_AMOUNT calculated in FACT_ORDERS?",
    "Which downstream objects depend on DW.FACT_ORDERS?",
    "What is the Snowflake account URL for production?",
    "What do the partner notes say about the returns feed?",
]


def show(res) -> None:
    print(f"\nQ: {res.question}")
    print(f"   status={res.status} gate={res.gate}")
    if res.answer:
        for line in res.answer.splitlines():
            print(f"   {line}")
        for c in res.citations:
            lines = f" lines {c['start_line']}-{c['end_line']}" if c["start_line"] else ""
            print(f"   [{c['label']}] {c['doc_id']} :: {c['locator']}{lines}")
    if res.abstain_reason:
        print(f"   abstained: {res.abstain_reason}")
    for w in res.warnings:
        print(f"   ! {w}")
    flagged = [e["label"] for e in res.evidence if e["suspicious"]]
    if flagged:
        print(f"   evidence flagged as untrusted (instruction-like text): {flagged}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="fastembed", choices=["fastembed", "hashing"])
    args = ap.parse_args()
    settings = Settings()
    embedder = build_embedder(args.embedder, settings.embedding_model)
    min_rel = settings.min_relevance if args.embedder == "fastembed" else 0.0

    with tempfile.TemporaryDirectory(prefix="dpka-demo-") as tmp:
        corpus = Path(tmp) / "corpus"
        shutil.copytree(REPO / "data" / "corpus", corpus)
        store = VectorStore(Path(tmp) / "index.sqlite")

        print(f"== Embedder: {embedder.name}")
        print("== Provider: Offline retrieval demonstration (extractive, no LLM)")
        print("\n== 1. Initial ingestion")
        print("   " + ingest(corpus, store, embedder).summary())
        print(f"   {store.stats()}")
        print("== 2. Re-ingestion (should change nothing)")
        print("   " + ingest(corpus, store, embedder).summary())

        retriever = Retriever(store, embedder, "hybrid")
        assistant = Assistant(
            retriever,
            ExtractiveProvider(),
            min_relevance=min_rel,
            min_coverage=settings.min_coverage,
        )
        print("\n== 3. Questions")
        for q in QUESTIONS:
            show(assistant.ask(q))

        print("\n== 4. Update one runbook and delete one mapping")
        rb = corpus / "runbooks" / "snowflake_warehouse_operations.md"
        rb.write_text(
            rb.read_text(encoding="utf-8").replace("= 3600", "= 7200"),
            encoding="utf-8",
        )
        (corpus / "informatica" / "m_PRODUCTS_FULL_REFRESH.xml").unlink()
        print("   " + ingest(corpus, store, embedder).summary())
        retriever.refresh()
        show(assistant.ask("What is the statement timeout of warehouse WH_ETL_XS?"))
        hits = retriever.search("How is PRODUCT_SK generated in the product full refresh?", k=5)
        print("\n   Retrieval after deleting m_PRODUCTS_FULL_REFRESH.xml:")
        for h in hits:
            print(f"   {h.rank}. {h.chunk.doc_id} :: {h.chunk.locator}")
        assert all("m_PRODUCTS_FULL_REFRESH" not in h.chunk.doc_id for h in hits)
        store.close()
    print("\nDemo finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
