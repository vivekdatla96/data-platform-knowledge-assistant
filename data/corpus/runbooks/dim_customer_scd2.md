# Runbook: DW.DIM_CUSTOMER SCD Type 2 maintenance

> Synthetic runbook written for the Data Platform Knowledge Assistant demo.

## What is tracked

`DW.DIM_CUSTOMER` keeps history for three attributes: `CUSTOMER_SEGMENT`, `LOYALTY_TIER` and
`REGION`. A change in any of them expires the current version and inserts a new one.
`SIGNUP_DATE` is Type 1: it is overwritten in place and never creates a version.

Change detection compares `ROW_HASH`, a SHA-256 of the three tracked attributes, with the hash
computed from the incoming snapshot row.

## Version rules

- New version: `EFFECTIVE_FROM = SNAPSHOT_DATE`, `EFFECTIVE_TO = 9999-12-31`, `IS_CURRENT = 'Y'`.
- Expired version: `EFFECTIVE_TO = SNAPSHOT_DATE - 1`, `IS_CURRENT = 'N'`.
- Surrogate key `-1` is a seeded "Unknown customer" row and must never be expired or deleted.

## Health checks

Run these after any manual fix. Each should return zero rows.

```sql
-- more than one current row per customer
SELECT CUSTOMER_ID, COUNT(*) AS current_rows
FROM DW.DIM_CUSTOMER
WHERE IS_CURRENT = 'Y'
GROUP BY CUSTOMER_ID
HAVING COUNT(*) > 1;

-- overlapping validity ranges
SELECT a.CUSTOMER_ID
FROM DW.DIM_CUSTOMER a
JOIN DW.DIM_CUSTOMER b
  ON a.CUSTOMER_ID = b.CUSTOMER_ID
 AND a.CUSTOMER_SK < b.CUSTOMER_SK
 AND a.EFFECTIVE_FROM <= b.EFFECTIVE_TO
 AND b.EFFECTIVE_FROM <= a.EFFECTIVE_TO;
```

## Repairing duplicate current rows

Duplicate current rows usually mean the same snapshot was processed twice after a partial failure.

1. Pause `wf_DAILY_SALES` so the orders session does not look up ambiguous rows.
2. For each affected `CUSTOMER_ID`, keep the version with the highest `CUSTOMER_SK` as current and
   set the others to `IS_CURRENT = 'N'` with `EFFECTIVE_TO` one day before the kept version's
   `EFFECTIVE_FROM`.
3. Re-run both health checks.
4. Resume the workflow. Orders loaded while duplicates existed may point at an older version;
   re-run the orders session with an `OVERRIDE_START_TS` covering that period.
