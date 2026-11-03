Embedder `fastembed:BAAI/bge-small-en-v1.5`, retrieval `hybrid`, provider `offline-extractive`, reranker `fastembed-rerank:Xenova/ms-marco-MiniLM-L-6-v2`, gate: similarity >= 0.68 and coverage >= 0.3, or reranker score >= 0.0

| Metric | Value |
|---|---|
| Answerable questions | 26 |
| Unanswerable questions | 8 |
| Hit@1 (doc level) | 0.9615 |
| Recall@5 (doc level) | 0.9808 |
| MRR (doc level) | 0.9808 |
| Answerable questions answered | 1.0 |
| Unanswerable questions abstained | 1.0 |
| Cited docs inside labeled relevant set | 0.6603 |

| Category | n | Recall@5 | MRR |
|---|---|---|---|
| failed_job | 6 | 0.9167 | 1.0 |
| incremental_load | 6 | 1.0 | 1.0 |
| lineage | 4 | 1.0 | 1.0 |
| mapping | 6 | 1.0 | 0.9167 |
| metadata | 4 | 1.0 | 1.0 |
