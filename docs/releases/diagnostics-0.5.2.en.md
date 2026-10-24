# mindie-diagnostics 0.5.2: Cooperative Intake Draining, Offline Stage-Admission Budgets & Record-Boundary Cursor Preservation

[中文](https://github.com/mindie-agent/diagnostics/blob/v0.5.2/docs/releases/diagnostics-0.5.2.md)

Building on the existing per-cycle file and byte scan limits, `mindie-diagnostics 0.5.2` adds cooperative `cancel` and `deadline` cutoffs to JSONL log ingestion and stage admission in the background worker and offline `reporting maintain` command—stopping scans at complete record boundaries, saving the SQLite byte cursor for already-enqueued records, and skipping subsequent stages once stopped or out of admission budget.

---

## 1. Key Fixes & Behavior Changes

### 1. Cooperative Record-Boundary Cutoff & Cursor Persistence (`mindie_diagnostics/ingestion.py`, PR `#29`)

- A single `ingest()` pass was already bounded by `max_files=256`, `max_bytes=8 MiB`, and `4096` directory entries, but could not exit early within those bounds when the worker was stopping or a maintenance budget expired.
- `0.5.2` adds cooperative `cancel` and monotonic `deadline` checks to `ingest(..., cancel=None, deadline=None)`:
  - Checks `cancel` and `deadline` between candidate JSONL files and after each complete JSONL record;
  - When triggered, stops at a **complete record boundary**, sets `counts["limited"] = 1`, and writes the byte `offset` of the last enqueued complete record to the SQLite `cursors` table;
  - The next cycle resumes from that byte `offset` without marking unread records as consumed (note: this is a cooperative drain for normal shutdown and deadline expiry, not an `exactly-once` guarantee across `SIGKILL` or process crashes).

### 2. Stopping Further Worker Stages on Shutdown (`mindie_diagnostics/cli.py`, PR `#29`)

- `run_cycle(..., cancel=None)` passes the `SIGINT` / `SIGTERM` `cancel` event into `ingest(..., cancel=cancel)` and gates subsequent stages:
  - Once `cancel` is set, the worker keeps data already collected in the cycle and starts no further root `ingest`, local retention (`prune`), or issue reporting (`publish_one`) stages;
  - The interrupted cycle reports `status="degraded"` and `limited=1`.

### 3. Offline Stage-Admission Budget for `reporting maintain --budget-seconds` (`mindie_diagnostics/reporting_commands.py`, PR `#29`)

- `reporting maintain` is strictly an offline maintenance stage (`"network": False`) covering queue expiry, consent-withdrawal cleanup, per-root `ingest`, per-root local `prune`, and optional running-service handoff; **it does not build support bundles, reconcile remote GitHub issues, or perform `POST` requests**.
- `budget_seconds` (default `75` seconds) is a **stage-admission budget**, not an OS-level 20-second service stop guarantee or a hard total wall-clock cap:
  - Each root `ingest` or `prune` stage starts only while at least `3.0` seconds remain, and `ingest` stops scanning new records at `deadline - 3.0s` so the final `Outbox` SQLite enqueue and `cursors` write retain their `3.0`-second SQLite lock wait budget;
  - When fewer than `3.0` seconds remain, `maintain` skips the remaining stage, records `{"status": "unavailable", "category": "budget_exhausted", "limited": True}`, and degrades aggregate `status` to `"degraded"`.

### 4. Bounded Outbound Reconciliation Documented as Design Follow-Up (PR `#29`)

- Bounding remote GitHub issue/comment reconciliation reads is documented separately as an unimplemented design follow-up in [`docs/outbound-reconciliation-followup.en.md`](https://github.com/mindie-agent/diagnostics/blob/v0.5.2/docs/outbound-reconciliation-followup.en.md), kept distinct from the local offline intake drain in `0.5.2`.
- Outbound safety invariants remain unchanged: `unknown` `POST` receipts are never replayed, and GitHub search misses are never treated as proof of remote absence.

---

## 2. Verification & Known Boundaries

- **Local Contract Suite**: Passed **`132 passed`** on both Python 3.11 and Python 3.13.
- **Honest Boundary**: During an earlier Windows test run, a legacy consent-isolation test ingested 1 record when 2 were expected and passed on rerun. Its root cause remains undetermined, and `0.5.2` does not claim a root-cause fix for that intermittent Windows test flake.
