# Evaluation

All numbers on this page were produced by the commands below on 2026-10-31 (Windows 11,
Python 3.11.18, fastembed 0.7.4, onnxruntime 1.31.1, numpy 2.4.6). The raw per-question
results are in [`docs/eval/`](eval/). Nothing here is extrapolated beyond these runs.

```bash
dpka ingest --rebuild
dpka eval --mode hybrid --questions data/eval/questions.jsonl --out docs/eval/hybrid-questions.json
dpka eval --mode hybrid --questions data/eval/holdout.jsonl   --out docs/eval/hybrid-holdout.json
dpka eval --mode dense  --questions data/eval/questions.jsonl --out docs/eval/dense-questions.json
dpka eval --mode dense  --questions data/eval/holdout.jsonl   --out docs/eval/dense-holdout.json
```

## Datasets

| File | Answerable | Unanswerable | Purpose |
|---|---|---|---|
| `data/eval/questions.jsonl` | 26 | 8 | Development set. The abstention thresholds were chosen by looking at it. |
| `data/eval/holdout.jsonl` | 8 | 4 | Written after the thresholds were fixed and not used for tuning. |

Each answerable question lists the source documents that contain its answer
(`relevant_docs`). Categories: incremental loads, failed ETL jobs, source-to-target mappings,
warehouse lineage, and metadata. Unanswerable questions ask about things the corpus does not
contain (account URLs, passwords, Kafka, Tableau, dbt, Airflow, Spark, SLAs).

Both sets were written by the project author against a 12-document synthetic corpus. They are
small and share a style with the corpus, so treat the results as a regression baseline for this
repository, not as evidence about performance on real enterprise documentation.

## Metrics

* **Hit@1, Recall@5, MRR** are computed at document level: the top-5 chunks are collapsed into an
  ordered list of distinct source documents and compared with `relevant_docs`.
* **Answerable answered / unanswerable abstained** run the full pipeline with the offline
  extractive provider, so they measure the abstention gate plus the provider's own
  "insufficient evidence" check.
* **Cited docs inside relevant set** is the share of documents cited in answered responses that
  are in the labeled relevant set. Citations outside the set are not necessarily wrong (for
  example, a lineage edge that also mentions the column), so this is a rough signal, not a
  precision measure.

## Results (embedder `BAAI/bge-small-en-v1.5`, offline extractive provider, k = 5)

| Retrieval | Set | Hit@1 | Recall@5 | MRR | Answerable answered | Unanswerable abstained | Cited docs in relevant set |
|---|---|---|---|---|---|---|---|
| hybrid (dense + BM25, RRF) | dev | 0.9231 | 0.9615 | 0.9615 | 1.0 | 1.0 | 0.6923 |
| hybrid (dense + BM25, RRF) | holdout | 1.0 | 1.0 | 1.0 | 0.75 | 1.0 | 0.6944 |
| dense only | dev | 0.8846 | 0.9423 | 0.9327 | 1.0 | 1.0 | 0.8077 |
| dense only | holdout | 0.75 | 1.0 | 0.875 | 0.875 | 1.0 | 0.7619 |

Gate used for all rows: best cosine similarity >= 0.68 and question-term coverage >= 0.30.

## Reading the results honestly

* **Hybrid retrieval helps on identifier-heavy questions.** Questions that name `NET_AMOUNT`,
  `OVERRIDE_START_TS` or `FACT_RETURNS` benefit from BM25's exact token matching. Dense-only
  ranked the right document first less often (Hit@1 0.88 vs 0.92 on dev, 0.75 vs 1.0 on
  holdout). With 8 holdout questions, one question moves Hit@1 by 0.125.
* **The abstention thresholds are overfit to the dev set.** On the dev set every answerable
  question cleared the similarity threshold (minimum 0.697) and every unanswerable question fell
  below it (maximum 0.671). The margin is 0.03. On the holdout set, 2 of 8 answerable questions
  in hybrid mode (`h03`, lookup cache failure; `h05`, returns arriving before their order) were
  wrongly abstained, although retrieval had found the right documents. All 12 unanswerable
  questions across both sets were correctly abstained. The design therefore errs toward
  abstaining.
* **Coverage alone does not separate the classes.** IDF-weighted term coverage catches questions
  with vocabulary missing from the corpus (Kafka, Airflow, Tableau score <= 0.16), but some
  answerable questions phrased in everyday words also score low (for example "Which downstream
  objects depend on DW.FACT_ORDERS?" scores 0.34). That is why the coverage threshold is low and
  similarity does most of the work.
* **No answer-quality metric is reported for LLM generation.** The Bedrock path is implemented
  and tested against stubbed responses, but no live Bedrock calls were made for this evaluation,
  so there are no faithfulness or answer-accuracy numbers for generated answers.

## Recalibrating

Thresholds depend on the embedding model and the corpus. After changing either, run `dpka eval`
on a labeled set, inspect the `gate` values per question in the JSON output, and set
`DPKA_MIN_RELEVANCE` / `DPKA_MIN_COVERAGE`. Keep a holdout set you do not tune on.
