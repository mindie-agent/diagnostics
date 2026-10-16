# Architecture, Privacy & Log Retention Reference

[中文](./architecture-and-privacy.md) · **English** · [Back to Documentation Index](./README.en.md)

This document specifies the failure ownership boundary, strict public Issue allowlist, consent versus policy-storage separation, and bounded local log rotation in `mindie-diagnostics`.

---

## 1. Failure Ownership Boundary

Invoke `mindie_diagnostics.integration.record_failure` only at the component boundary that owns the failure:

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

- **Static Metadata Only**: Pass static product identifiers and categories—never pass user input as `category` or `operation`.
- **Non-Throwing Recording**: The result includes a random `incident_id` when local persistence succeeds; otherwise it exposes `logging_failed` without altering the caller's business result, invoking a model, or performing network I/O.
- **Caller Errors vs. Product Defects**: Ordinary caller errors, missing configuration, inactive admission, cancellation, permissions, network/authentication failures, non-zero business command exits/timeouts, and native model failures are not product defects and stay local (`reportable=False`). Only confirmed internal faults and malformed own-tool protocols are reportable.

---

## 2. Public Reporting & Strict Allowlist Privacy

When a user explicitly enables automatic reporting (`reporting configure --enabled true` followed by `reporting ensure`), the background worker constructs GitHub Issues without any LLM calls from a strict allowlist:

- **Allowlisted Fields**: Component name, loaded version/revision, operation and stage, static error category, monotonic duration, exit/cleanup facts, and product code stack frames.
- **Excluded Data**: Raw exception messages, shell commands, stdout/stderr, prompts, transcripts, environment values, credentials, session/job IDs, file paths, and host addresses.
- **Bounded Cycles & Idempotent Reconciliation**: Each candidate has at most three persisted processing cycles. Uncertain `POST` outcomes only reconcile on subsequent cycles and are never blindly reposted. Existing open or closed Issues matching the public fault fingerprint are linked without commenting or reopening.
- **Queue Caps & Immediate Revocation**: The active queue holds at most 1,000 candidates (for 7 days), and terminal evidence is compacted to small receipts (at most 1,000 for 30 days). Disabling reporting immediately invalidates pending remote actions under the prior authorization; re-enabling never backfills events produced while off or resurrects withdrawn candidates.

---

## 3. Consent Status vs. Policy Storage

Failure observation keeps reporting authorization distinct from policy file availability:

- `fallback.read_policy()` returns `None` only when the selected policy file does not exist; an existing malformed, unsafe, or unreadable policy raises `PolicyUnavailable`.
- `consent_status()` distinguishes `allowed`, `withdrawn`, and `unavailable`:
  - `unavailable` blocks network transmission while preserving pending evidence and log cursors;
  - only a verified disabled or changed authorization (`withdrawn`) withdraws queued work.
- Outbox initialization is serialized and guarded by an ownership marker. An existing empty/incomplete database or a missing previously initialized outbox fails visibly as a storage error rather than silently recreating publication history.

---

## 4. Bounded Local Log Rotation & Maintenance

- **Four-Segment Rotation**: Installed writers maintain four fixed 1 MiB JSONL segments per process/component and reuse closed segments only after all registered readers have consumed that inode. Space pressure surfaces visibly as dropped records / `logging_failed` and resumes automatically once safe space is reclaimed.
- **Dependency-Free Startup Fallback**: Capped at a single 1 MiB fault file that fails visibly when full.
- **Offline Maintenance (`reporting maintain`)**:
  - Default retention target: **7 days, 128 MiB, and 512 files** per configured root.
  - Runs completely offline unless `--update-running` is passed, which only upgrades an already enabled, healthy worker to a higher semantic version (never starting or reviving stopped/unhealthy workers).
  - Adapters pass their remaining time budget via `--budget-seconds` (capped at 75 seconds).
