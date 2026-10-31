# Runbook: Triage a failed ETL job

> Synthetic runbook written for the Data Platform Knowledge Assistant demo. Error messages below
> are illustrative examples, not copies of vendor error catalogs.

## First five minutes

1. Open the workflow monitor for `wf_DAILY_SALES` and find the failed session.
2. Query `AUDIT.JOB_RUN_LOG` for the latest row of that job. `ERROR_MESSAGE` holds the first error
   the session reported, and `ROWS_REJECTED` tells you whether rows went to the reject file.
3. Check whether downstream sessions were skipped. The customer dimension session fails the
   workflow before the orders session starts, by design.
4. Post a short status in the data platform on-call channel: job, first error, start time.

## Common failures and fixes

| Symptom in ERROR_MESSAGE | Likely cause | Fix |
|---|---|---|
| Statement reached its statement timeout | Warehouse too small for the window, often after a backfill or a large late batch | Rerun; if it repeats, follow the Snowflake warehouse runbook to resize temporarily |
| Warehouse is suspended and cannot be resumed | Resource monitor quota reached on `WH_ETL_XS` | Ask the platform admin to raise the quota or wait for reset, then rerun |
| Duplicate row detected during DML action | More than one source row per key reached the MERGE, usually because the dedupe in the source qualifier was edited | Restore the `QUALIFY ROW_NUMBER()` dedupe, then rerun |
| NULL result in a non-nullable column CUSTOMER_SK | Lookup default for unknown customers was removed | Restore the -1 default in `LKP_DIM_CUSTOMER`, then rerun |
| Lookup cache build failed | Lookup source too large for the cache directory on the integration node | Clear old cache files or switch the lookup to uncached for one run |
| Number of rows rejected exceeds threshold | Bad source rows, for example negative `ORDER_TOTAL` | Inspect the reject file, raise a data issue with the source owner |

## Rerun rules

- Incremental sessions (`FACT_ORDERS`, `FACT_RETURNS`) do not advance their watermark on failure,
  so a plain restart re-extracts the same window. Do not edit `AUDIT.ETL_CONTROL` for a normal
  rerun.
- `DIM_PRODUCT` is a single-transaction truncate and insert, so a failed run leaves the old data
  in place and can be rerun at any time.
- Run `DIM_CUSTOMER` before `FACT_ORDERS` so orders see current customer versions.

## Escalation

- If the same session fails twice in a row after a rerun, page the secondary data platform on-call
  engineer.
- If `MART.DAILY_SALES` will be stale past 07:00 UTC, notify the analytics team lead so reports
  carry a freshness warning.
- Source data problems (rejected rows, missing files) go to the source system owner as a ticket;
  do not patch source data in `RAW` tables.
