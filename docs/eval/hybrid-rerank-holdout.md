Embedder `fastembed:BAAI/bge-small-en-v1.5`, retrieval `hybrid`, provider `offline-extractive`, reranker `fastembed-rerank:Xenova/ms-marco-MiniLM-L-6-v2`, gate: similarity >= 0.68 and coverage >= 0.3, or reranker score >= 0.0

| Metric | Value |
|---|---|
| Answerable questions | 8 |
| Unanswerable questions | 4 |
| Hit@1 (doc level) | 0.875 |
| Recall@5 (doc level) | 0.9375 |
| MRR (doc level) | 0.9375 |
| Answerable questions answered | 0.875 |
| Unanswerable questions abstained | 1.0 |
| Cited docs inside labeled relevant set | 0.6905 |

| Category | n | Recall@5 | MRR |
|---|---|---|---|
| failed_job | 2 | 1.0 | 1.0 |
| incremental_load | 1 | 1.0 | 1.0 |
| lineage | 1 | 0.5 | 1.0 |
| mapping | 2 | 1.0 | 0.75 |
| metadata | 2 | 1.0 | 1.0 |

Questions where the answer/abstain decision was wrong:

- `h03` (abstained, gate {'best_similarity': 0.6276, 'term_coverage': 0.6085, 'best_rerank_score': -0.781}): A session failed because the lookup cache could not be built. How do I fix it?
