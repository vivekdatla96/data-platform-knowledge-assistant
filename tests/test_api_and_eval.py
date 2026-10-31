from __future__ import annotations

import asyncio

import httpx
from conftest import CORPUS, REPO

from dpka.api import create_app
from dpka.cli import main
from dpka.evaluation import evaluate, load_questions


def test_api_health_ask_and_ui(assistant):
    app = create_app(assistant=assistant)

    async def run_test() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = (await client.get("/api/health")).json()
            assert health["index"]["documents"] == 12
            assert "no LLM" in health["provider_label"]

            q = "Is it safe to rerun a failed FACT_ORDERS session?"
            r = await client.post("/api/ask", json={"question": q})
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "answered" and body["citations"] and body["evidence"]

            assert (await client.post("/api/ask", json={"question": ""})).status_code == 422
            assert (
                await client.post("/api/ask", json={"question": "x", "top_k": 99})
            ).status_code == 422

            html = (await client.get("/")).text
            assert "synthetic" in html and "innerHTML" not in html

    asyncio.run(run_test())


def test_eval_labels_point_at_real_documents():
    docs = {p.relative_to(CORPUS).as_posix() for p in CORPUS.rglob("*") if p.is_file()}
    for name in ("questions.jsonl", "holdout.jsonl"):
        qs = load_questions(REPO / "data" / "eval" / name)
        assert any(not q.answerable for q in qs)
        for q in qs:
            assert set(q.relevant_docs) <= docs, (q.id, q.relevant_docs)


def test_evaluate_produces_reproducible_metrics(assistant):
    qs = load_questions(REPO / "data" / "eval" / "questions.jsonl")
    a = evaluate(assistant, qs, k=5)
    b = evaluate(assistant, qs, k=5)
    assert a["summary"] == b["summary"]
    s = a["summary"]
    assert s["answerable_questions"] == 26 and s["unanswerable_questions"] == 8
    for key in ("hit_at_1", "recall_at_5", "mrr", "unanswerable_abstained_rate"):
        assert 0.0 <= s[key] <= 1.0


def test_cli_ingest_and_ask(tmp_path, corpus, capsys, monkeypatch):
    monkeypatch.setenv("DPKA_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("DPKA_MIN_RELEVANCE", "0")
    index = str(tmp_path / "cli.sqlite")
    assert main(["ingest", "--corpus", str(corpus), "--index", index]) == 0
    assert "added=12" in capsys.readouterr().out
    assert main(["ingest", "--corpus", str(corpus), "--index", index]) == 0
    assert "unchanged=12" in capsys.readouterr().out
    assert main(["ask", "--index", index, "Is", "it", "safe", "to", "rerun", "FACT_ORDERS?"]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out and "Retrieved evidence:" in out
