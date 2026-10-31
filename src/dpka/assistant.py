"""Question answering: retrieve -> abstention gate -> generate -> verify citations."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from dpka.citations import check_citations
from dpka.config import Settings
from dpka.embeddings import build_embedder
from dpka.prompting import INSUFFICIENT, Evidence
from dpka.providers import AnswerProvider, ProviderError, build_provider
from dpka.retrieval import RetrievedChunk, Retriever
from dpka.store import VectorStore

MAX_QUESTION_CHARS = 1000


@dataclass
class AnswerResult:
    question: str
    status: str  # answered | abstained | generation_failed
    answer: str | None
    abstain_reason: str | None
    provider: str
    provider_label: str
    retrieval_mode: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gate: dict[str, float] = field(default_factory=dict)  # abstention signals
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evidence_dict(label: str, hit: RetrievedChunk) -> dict[str, Any]:
    c = hit.chunk
    return {
        "label": label,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "doc_type": c.doc_type,
        "locator": c.locator,
        "title": c.title,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "dense_score": round(hit.dense_score, 4),
        "bm25_score": round(hit.bm25_score, 3),
        "suspicious": c.suspicious,
        "text": c.text,
    }


class Assistant:
    def __init__(
        self,
        retriever: Retriever,
        provider: AnswerProvider,
        *,
        top_k: int = 5,
        min_relevance: float = 0.5,
        min_coverage: float = 0.6,
    ):
        self.retriever = retriever
        self.provider = provider
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.min_coverage = min_coverage

    @classmethod
    def from_settings(cls, settings: Settings) -> Assistant:
        store = VectorStore(settings.index_path)
        embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
        retriever = Retriever(store, embedder, settings.retrieval_mode)
        return cls(
            retriever,
            build_provider(settings),
            top_k=settings.top_k,
            min_relevance=settings.min_relevance,
            min_coverage=settings.min_coverage,
        )

    def ask(self, question: str, top_k: int | None = None) -> AnswerResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question longer than {MAX_QUESTION_CHARS} characters")

        result = AnswerResult(
            question=question,
            status="abstained",
            answer=None,
            abstain_reason=None,
            provider=self.provider.name,
            provider_label=self.provider.label,
            retrieval_mode=self.retriever.mode,
        )
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=top_k or self.top_k)

        labeled = [(f"S{i}", h) for i, h in enumerate(hits, start=1)]
        result.evidence = [_evidence_dict(lab, h) for lab, h in labeled]
        if any(h.chunk.suspicious for h in hits):
            result.warnings.append(
                "Some retrieved evidence contains instruction-like text and is treated as "
                "untrusted data."
            )

        best = max((h.dense_score for h in hits), default=0.0)
        coverage = self.retriever.coverage(question, hits)
        result.timings_ms["retrieval"] = round((time.perf_counter() - t0) * 1000, 1)
        result.gate = {"best_similarity": round(best, 4), "term_coverage": round(coverage, 4)}
        if best < self.min_relevance or coverage < self.min_coverage:
            result.abstain_reason = (
                f"Insufficient evidence: best similarity {best:.2f} (min "
                f"{self.min_relevance:.2f}), question-term coverage {coverage:.2f} "
                f"(min {self.min_coverage:.2f})."
            )
            return result

        evidence = [Evidence(lab, h.chunk) for lab, h in labeled]
        t1 = time.perf_counter()
        try:
            raw = self.provider.generate_answer(question, evidence)
        except ProviderError as exc:
            result.status = "generation_failed"
            result.warnings.append(
                f"{exc} - showing retrieved evidence only."
                + (" The error looks transient; retry later." if exc.retryable else "")
            )
            return result
        finally:
            result.timings_ms["generation"] = round((time.perf_counter() - t1) * 1000, 1)

        if raw.strip().startswith(INSUFFICIENT):
            result.abstain_reason = "The provider reported that the evidence is insufficient."
            return result

        check = check_citations(raw, {e.label: e.chunk.text for e in evidence})
        result.warnings.extend(check.warnings)
        if not check.valid_labels:
            result.abstain_reason = (
                "The generated answer cited no retrieved evidence, so it was withheld."
            )
            return result

        by_label = {d["label"]: d for d in result.evidence}
        result.status = "answered"
        result.answer = check.text
        result.citations = [
            {
                k: by_label[lab][k]
                for k in ("label", "chunk_id", "doc_id", "locator", "start_line", "end_line")
            }
            for lab in check.valid_labels
        ]
        return result
