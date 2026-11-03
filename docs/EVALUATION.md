# Evaluation

All numbers on this page were produced by the commands below (Windows 11, Python 3.11.18,
fastembed 0.7.4, onnxruntime 1.31.1, numpy 2.4.6): the no-rerank rows on 2026-10-31, the
reranking rows on 2026-11-03. The raw per-question results are in [`docs/eval/`](eval/).
Nothing here is extrapolated beyond these runs.

```bash
dpka ingest --rebuild
# default since 2026-11-03: hybrid retrieval + cross-encoder reranking
dpka eval --questions data/eval/questions.jsonl --out docs/eval/hybrid-rerank-questions.json
dpka eval --questions data/eval/holdout.jsonl   --out docs/eval/hybrid-rerank-holdout.json
# earlier configurations (re-running them today reproduces the same numbers)
dpka eval --no-rerank --mode hybrid --questions data/eval/questions.jsonl --out docs/eval/hybrid-questions.json
dpka eval --no-rerank --mode hybrid --questions data/eval/holdout.jsonl   --out docs/eval/hybrid-holdout.json
dpka eval --no-rerank --mode dense  --questions data/eval/questions.jsonl --out docs/eval/dense-questions.json
dpka eval --no-rerank --mode dense  --questions data/eval/holdout.jsonl   --out docs/eval/dense-holdout.json
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
| **hybrid + rerank (default)** | dev | 0.9615 | 0.9808 | 0.9808 | 1.0 | 1.0 | 0.6603 |
| **hybrid + rerank (default)** | holdout | 0.875 | 0.9375 | 0.9375 | 0.875 | 1.0 | 0.6905 |
| hybrid (dense + BM25, RRF) | dev | 0.9231 | 0.9615 | 0.9615 | 1.0 | 1.0 | 0.6923 |
| hybrid (dense + BM25, RRF) | holdout | 1.0 | 1.0 | 1.0 | 0.75 | 1.0 | 0.6944 |
| dense only | dev | 0.8846 | 0.9423 | 0.9327 | 1.0 | 1.0 | 0.8077 |
| dense only | holdout | 0.75 | 1.0 | 0.875 | 0.875 | 1.0 | 0.7619 |

Gate for all rows: best cosine similarity >= 0.68 and question-term coverage >= 0.30. The rerank
rows add one rule: a question that fails that gate is still answered if the best reranker logit
is >= 0.

## Cross-encoder reranking (added 2026-11-03)

The top 10 hybrid candidates are re-scored by `Xenova/ms-marco-MiniLM-L-6-v2` and the best 5
are kept. How the settings were chosen:

* **Candidate pool:** 20 and 10 gave identical dev results; 10 was chosen because it is faster.
  Decided on the dev set and latency only.
* **Rescue threshold 0:** the model's own 50% relevance point, fixed before the holdout set was
  run. It was not tuned: on the dev set the best reranker score of every unanswerable question was
  <= -1.38, while two answerable ones scored -1.02 and -0.79, a margin too thin to tune on.
* **Rescue only, never veto:** a negative reranker score does not override a question the first
  gate accepted, because those two answerable dev questions would then have been rejected.

What it changed:

* **Dev:** Hit@1 0.92 -> 0.96, Recall@5 and MRR 0.96 -> 0.98; abstention unchanged (all
  answered / all abstained).
* **Holdout:** one more answerable question answered (`h05` was rescued: 6 -> 7 of 8), still
  4 of 4 unanswerable abstained, but retrieval got worse on two questions:
  * `h04` (SCD2 change detection): the reranker ranked `snowflake/dw_tables.json` first. Its
    `ROW_HASH` column note does describe change detection, but the label lists the mapping
    first, so it counts as a miss. Labels were not edited after seeing results.
  * `h06` (columns from `RAW.PRODUCTS_STG` into `DIM_PRODUCT`): the labeled lineage file stayed
    first, but the mapping dropped out of the top 5 (Recall@5 for that question 1.0 -> 0.5).
  * `h03` (lookup cache failure) is still wrongly abstained (best reranker score -0.78).
* **Cost:** median retrieval time went from about 10 ms to about 1.05 s per question on the
  development laptop's CPU (12 dev questions after warm-up). Disable it with
  `DPKA_RERANKER_MODEL=none` or `--no-rerank`.

With 8 holdout questions every change of one question moves Hit@1 by 0.125, so the holdout result
is best read as "a trade: better abstention, slightly worse ranking", not as a regression or a
win. The holdout set has now been used to report two design decisions; neither was tuned on it,
but a fresh holdout set would be needed for an unbiased estimate of any further changes.

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
  abstaining. (With reranking, `h05` is now answered; `h03` is still abstained.)
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
