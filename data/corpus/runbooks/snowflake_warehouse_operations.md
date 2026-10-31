# Runbook: Snowflake warehouse operations for ETL

> Synthetic runbook written for the Data Platform Knowledge Assistant demo. Settings are
> illustrative for a fictional account.

## Warehouses

| Warehouse | Size | Auto-suspend | Used by |
|---|---|---|---|
| `WH_ETL_XS` | X-Small | 60 seconds | All `wf_DAILY_SALES` sessions |
| `WH_ANALYTICS_S` | Small | 300 seconds | BI queries on `MART` |

`WH_ETL_XS` has `STATEMENT_TIMEOUT_IN_SECONDS = 3600` and a resource monitor that suspends it when
the monthly credit quota is reached.

## Statement timeouts

A timeout on an ETL session usually means the window is larger than normal. Check
`AUDIT.JOB_RUN_LOG.ROWS_READ` for the last successful run, compare with the source count for the
current window, and rerun once. If it times out again, resize temporarily.

## Temporary resize for backfills

```sql
ALTER WAREHOUSE WH_ETL_XS SET WAREHOUSE_SIZE = 'MEDIUM';
-- run the backfill session
ALTER WAREHOUSE WH_ETL_XS SET WAREHOUSE_SIZE = 'XSMALL';
```

Always size back down in the same change window, and record the resize in the change log. Leaving
the warehouse at Medium multiplies the credit burn of every hourly run.

## Cost hygiene

- Keep auto-suspend at 60 seconds for ETL warehouses; hourly sessions finish in minutes.
- Do not point BI tools at `WH_ETL_XS`.
- Review `WAREHOUSE_METERING_HISTORY` weekly for unexpected spikes.
