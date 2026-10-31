Embedder `fastembed:BAAI/bge-small-en-v1.5`, retrieval `dense`, provider `offline-extractive`, gate: similarity >= 0.68, coverage >= 0.3

| Metric | Value |
|---|---|
| Answerable questions | 8 |
| Unanswerable questions | 4 |
| Hit@1 (doc level) | 0.75 |
| Recall@5 (doc level) | 1.0 |
| MRR (doc level) | 0.875 |
| Answerable questions answered | 0.875 |
| Unanswerable questions abstained | 1.0 |
| Cited docs inside labeled relevant set | 0.7619 |

| Category | n | Recall@5 | MRR |
|---|---|---|---|
| failed_job | 2 | 1.0 | 1.0 |
| incremental_load | 1 | 1.0 | 1.0 |
| lineage | 1 | 1.0 | 1.0 |
| mapping | 2 | 1.0 | 0.75 |
| metadata | 2 | 1.0 | 0.75 |

Questions where the answer/abstain decision was wrong:

- `h03` (abstained, gate {'best_similarity': 0.6276, 'term_coverage': 0.6085}): A session failed because the lookup cache could not be built. How do I fix it?
