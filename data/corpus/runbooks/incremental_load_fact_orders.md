# Runbook: Incremental load of DW.FACT_ORDERS

> Synthetic runbook written for the Data Platform Knowledge Assistant demo. Systems, names and
> schedules are fictional.

## How the incremental window works

The session `s_m_ORDERS_STG_TO_FACT_ORDERS` runs hourly at :15 in workflow `wf_DAILY_SALES`.
Its pre-session SQL reads the row for this job from `AUDIT.ETL_CONTROL`:

- `LAST_EXTRACT_TS` is the high-water mark: the largest `UPDATED_AT` loaded by the last
  successful run.
- `LOOKBACK_MINUTES` is 120. The source filter is
  `UPDATED_AT > LAST_EXTRACT_TS - 120 minutes`.
- `OVERRIDE_START_TS`, when not null, replaces `LAST_EXTRACT_TS` for one run (used for backfills).

The source qualifier keeps only the newest row per `ORDER_ID`, and the target is an upsert keyed
on `ORDER_ID`, so overlapping windows do not create duplicates.

## Why there is a lookback window

The order-management system commits some transactions late: a row can become visible in
`RAW.ORDERS_STG` with an `UPDATED_AT` that is earlier than rows that were already loaded. Without
the 120 minute overlap those late-committed rows would fall below the watermark and never be
extracted. The overlap re-reads a little data every hour; the upsert makes that harmless.

## When the watermark moves

The post-session SQL updates `LAST_EXTRACT_TS` to `MAX(SOURCE_UPDATED_AT)` of the rows just loaded
and writes a row to `AUDIT.JOB_RUN_LOG`. It runs **only when the session succeeds**. A failed or
aborted run leaves the watermark where it was, so the next run re-extracts the same window.

## Rerunning a failed run

Rerunning is safe and is the default recovery:

1. Confirm in `AUDIT.JOB_RUN_LOG` that the last run for the job has `STATUS = 'FAILED'` and read
   `ERROR_MESSAGE`.
2. Fix the cause (see the failed ETL job triage runbook).
3. Restart the session from the workflow monitor. Do not edit `AUDIT.ETL_CONTROL` by hand for a
   normal rerun.

Because the target is an upsert and the update strategy rejects rows that are not newer than the
existing row, replaying the same window cannot overwrite newer data.

## Backfilling a past date range

Use a backfill when historical orders were corrected in the source, or after a source outage.

1. Set `OVERRIDE_START_TS` for job `s_m_ORDERS_STG_TO_FACT_ORDERS` in `AUDIT.ETL_CONTROL` to the
   start of the range.
2. If the range is more than about 30 days, temporarily resize warehouse `WH_ETL_XS` (see the
   Snowflake warehouse runbook).
3. Run the session once. The post-session SQL clears `OVERRIDE_START_TS` on success.
4. Reconcile row counts: compare `COUNT(DISTINCT ORDER_ID)` in `RAW.ORDERS_STG` for the range
   with `DW.FACT_ORDERS`.

Never truncate `DW.FACT_ORDERS` to backfill. Downstream `MART.DAILY_SALES` and `DW.FACT_RETURNS`
depend on stable `ORDER_SK` values, and a truncate would generate new surrogate keys.
