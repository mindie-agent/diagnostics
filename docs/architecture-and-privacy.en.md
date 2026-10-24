# Architecture, Privacy Allowlist & Log Rotation Guide

[中文](./architecture-and-privacy.md) · **English** · [Back to Documentation Index](./README.en.md)

This guide explains the four core mechanisms inside `mindie-diagnostics`: **how component failures are attributed, how the strict privacy allowlist guarantees zero data leaks, why user consent is kept separate from config storage errors, and how ring-buffer log rotation keeps disk usage bounded.**

---

## 1. Failure Attribution: Product Bugs vs. Everyday User Errors

Inside MindIE Agent components, only the boundary that owns a fault calls `mindie_diagnostics.integration.record_failure`:

```python
from mindie_diagnostics.integration import record_failure

result = record_failure(
    "remote-dev",
    "mcp.request",
    stage="decode",
    category="protocol_mismatch",
    exception=error,
)
```

- **Fixed Code Labels Only**: `category` and `operation` must be static string constants defined in product code — user inputs, commands, and file paths are never passed into these fields.
- **Logging Failures Never Break Your Task**: When local persistence succeeds, `record_failure` returns a random `incident_id`. If the disk is full or unwritable, it simply flags `logging_failed` in the diagnostic envelope without altering the tool's return value, calling a model, or touching the network.
- **Separating User Errors from Product Bugs**:
  - Invalid arguments, missing configuration, insufficient permissions, SSH authentication failures, or non-zero exit codes from your own training scripts are normal operational events, not MindIE Agent bugs. They are recorded locally with `reportable=False` and **never open GitHub Issues**.
  - Only confirmed internal exceptions and broken own-tool protocols are marked as reportable candidates.

---

## 2. Strict Allowlist Privacy for Public Issue Reporting

When you explicitly opt in (`reporting configure --enabled true` and `reporting ensure`), the background worker files GitHub Issues using a deterministic allowlist — **with zero LLM calls**:

| Category | Fields |
| --- | --- |
| **✅ Allowlisted (Included)** | Component name, package version & Git commit revision, operation & stage name, static error category, monotonic duration, process exit/cleanup flags, and product code stack frames (file & line numbers within MindIE packages) |
| **🚫 Strictly Excluded (Never Sent)** | Raw exception message text (`str(e)`), shell commands, `stdout` / `stderr`, prompts, conversation transcripts, environment variable values, credentials/keys, session or job IDs, user file paths, and remote host IPs or hostnames |

**Anti-Spam & Instant Revocation Guarantees**:
- **Idempotent Reconciliation (No Duplicate Issue Spam)**: Each candidate goes through at most 3 processing cycles. If a network drop leaves a `POST` request uncertain, the next cycle checks GitHub first rather than blindly reposting. If an open or closed Issue with the same fault fingerprint already exists, it links to that Issue without commenting or reopening it.
- **Immediate Revocation When Disabled**: The active queue holds at most 1,000 candidates (for 7 days) and keeps at most 1,000 compact receipts (for 30 days). Running `configure --enabled false` **immediately invalidates all pending items in the queue**; re-enabling later never backfills events recorded while reporting was off.

---

## 3. Keeping User Consent Separate from Config File Errors

Many tools treat a corrupted or unreadable config file the same as "first run" or "feature disabled," silently resetting user state or wiping queues.

`mindie-diagnostics` strictly separates consent state from storage errors (`consent_status()`):
1. **`allowed`**: The policy file is valid and reporting is explicitly enabled.
2. **`withdrawn`**: The policy file does not exist yet or reporting is explicitly set to `false`. Queued items are withdrawn **only** when `withdrawn` is verified.
3. **`unavailable`**: If the policy file exists on disk but is malformed, has unsafe permissions, or cannot be read, `fallback.read_policy()` raises `PolicyUnavailable`. In this state, the worker **blocks all network transmission while preserving pending evidence and log cursors** until the file is fixed.

Similarly, the SQLite outbox database records an initialization marker; if a previously initialized database goes missing or becomes empty, it fails visibly as a storage error rather than silently creating a blank history.

---

## 4. Bounded Ring-Buffer Logs & Offline Maintenance

- **4 × 1 MiB Segment Rotation**:
  Each process/component writes to four fixed `1 MiB` JSONL segments, reusing a closed segment only after all registered readers have finished reading it. If disk space runs low, dropped record counts and `logging_failed` are surfaced visibly and writing resumes automatically once space is freed.
- **What actually trips backpressure (`retention-state.json`), and what it costs**:
  The previous offline maintenance pass sets backpressure whenever the log root is still over budget, and that budget is not only bytes: remaining above **128 MiB or 512 files**, or failing to complete the scan, sets it. The 7-day age threshold only makes consumed old segments eligible for reclamation; age alone does not set backpressure. Because segments are named per process (`<pid>-<hex>.jsonl`), the **file count** grows with the number of short-lived processes rather than with bytes, so backpressure can trip far below 128 MiB. While it is set, a **newly started process cannot open a segment at all**: its local failure records are honestly reported as `logging_failed` and dropped, while processes that already own a segment keep rotating within their own credit. This is a deliberate bounded-ness tradeoff - the file budget also bounds inodes and per-scan cost, and dropping it would allow hundreds of thousands of tiny files inside 128 MiB. Backpressure clears on the next successful offline maintenance once the reporter has consumed the backlog, so **a reporting policy file that cannot be read keeps it set** until that file is repaired or removed.
- **Zero-Dependency Startup Fallback**:
  Early startup faults before full logger setup are capped at a single `1 MiB` fallback file.
- **Offline Maintenance (`reporting maintain`)**:
  - Enforces a default retention budget of **7 days, 128 MiB, and 512 files** per log root.
  - Runs completely offline by default. Passing `--update-running` smoothly upgrades an already enabled, healthy background worker to a newer semantic version without ever starting or reviving stopped workers.
  - Client adapters assign a stage-admission budget via `--budget-seconds` (default 75 seconds, capped at `max(75, _UPDATE_BUDGET_S + 15)`, currently 195 seconds). Intake reserves 3 seconds for settlement; no new root or retention phase starts below that reserve. This is not a hard wall-clock ceiling: a lawful SQLite wait already in progress may finish after the budget.
