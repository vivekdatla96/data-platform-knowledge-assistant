"""Reproducible retrieval and abstention evaluation on a labeled question set.

Relevance is labeled at document level (a question lists the source files that contain its
answer). Metrics are computed over the ranked list of distinct documents in the top-k chunks.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpka.assistant import Assistant


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    category: str
    answerable: bool
    relevant_docs: list[str]


def load_questions(path: Path) -> list[EvalQuestion]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        d = json.loads(line)
        q = EvalQuestion(
            d["id"],
            d["question"],
            d["category"],
            bool(d["answerable"]),
            list(d.get("relevant_docs", [])),
        )
        if q.answerable and not q.relevant_docs:
            raise ValueError(f"line {n}: answerable question {q.id} has no relevant_docs")
        out.append(q)
    return out


def _ranked_docs(doc_ids: list[str]) -> list[str]:
    seen: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.append(d)
    return seen


def evaluate(assistant: Assistant, questions: list[EvalQuestion], k: int = 5) -> dict[str, Any]:
    rows = []
    for q in questions:
        res = assistant.ask(q.question, top_k=k)
        docs = _ranked_docs([e["doc_id"] for e in res.evidence])
        rel = set(q.relevant_docs)
        first = next((i for i, d in enumerate(docs, start=1) if d in rel), None)
        cited = {c["doc_id"] for c in res.citations}
        rows.append(
            {
                "id": q.id,
                "category": q.category,
                "answerable": q.answerable,
                "question": q.question,
                "retrieved_docs": docs,
                "hit_at_1": bool(docs and docs[0] in rel),
                "recall_at_k": len(rel & set(docs)) / len(rel) if rel else None,
                "reciprocal_rank": 1.0 / first if first else 0.0,
                "status": res.status,
                "gate": res.gate,
                "cited_docs": sorted(cited),
                "citations_in_relevant": (len(cited & rel) / len(cited)) if cited and rel else None,
            }
        )

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if r["status"] == "answered"]
    cite_prec = [
        r["citations_in_relevant"] for r in answered if r["citations_in_relevant"] is not None
    ]

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "answerable_questions": len(ans),
        "unanswerable_questions": len(una),
        "k": k,
        "hit_at_1": mean([float(r["hit_at_1"]) for r in ans]),
        f"recall_at_{k}": mean([r["recall_at_k"] for r in ans]),
        "mrr": mean([r["reciprocal_rank"] for r in ans]),
        "answerable_answered_rate": mean([float(r["status"] == "answered") for r in ans]),
        "unanswerable_abstained_rate": mean([float(r["status"] == "abstained") for r in una]),
        "cited_docs_in_relevant_set": mean(cite_prec),
    }
    by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in ans}):
        rs = [r for r in ans if r["category"] == cat]
        by_cat[cat] = {
            "n": len(rs),
            f"recall_at_{k}": mean([r["recall_at_k"] for r in rs]),
            "mrr": mean([r["reciprocal_rank"] for r in rs]),
        }
    return {
        "config": {
            "embedder": assistant.retriever.embedder.name,
            "retrieval_mode": assistant.retriever.mode,
            "provider": assistant.provider.name,
            "min_relevance": assistant.min_relevance,
            "min_coverage": assistant.min_coverage,
            "python": platform.python_version(),
        },
        "summary": summary,
        "by_category": by_cat,
        "questions": rows,
    }


def to_markdown(report: dict[str, Any]) -> str:
    c, s = report["config"], report["summary"]
    k = s["k"]
    lines = [
        f"Embedder `{c['embedder']}`, retrieval `{c['retrieval_mode']}`, provider "
        f"`{c['provider']}`, gate: similarity >= {c['min_relevance']}, coverage >= "
        f"{c['min_coverage']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answerable questions | {s['answerable_questions']} |",
        f"| Unanswerable questions | {s['unanswerable_questions']} |",
        f"| Hit@1 (doc level) | {s['hit_at_1']} |",
        f"| Recall@{k} (doc level) | {s[f'recall_at_{k}']} |",
        f"| MRR (doc level) | {s['mrr']} |",
        f"| Answerable questions answered | {s['answerable_answered_rate']} |",
        f"| Unanswerable questions abstained | {s['unanswerable_abstained_rate']} |",
        f"| Cited docs inside labeled relevant set | {s['cited_docs_in_relevant_set']} |",
        "",
        f"| Category | n | Recall@{k} | MRR |",
        "|---|---|---|---|",
    ]
    for cat, v in report["by_category"].items():
        lines.append(f"| {cat} | {v['n']} | {v[f'recall_at_{k}']} | {v['mrr']} |")
    misses = [
        r
        for r in report["questions"]
        if (r["answerable"] and r["status"] != "answered")
        or (not r["answerable"] and r["status"] != "abstained")
    ]
    if misses:
        lines += ["", "Questions where the answer/abstain decision was wrong:", ""]
        for r in misses:
            lines.append(f"- `{r['id']}` ({r['status']}, gate {r['gate']}): {r['question']}")
    return "\n".join(lines)
