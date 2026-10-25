# MindIE Diagnostics & Reporting (`mindie-diagnostics`) — Agent Architecture Guide

This repository implements the shared local failure recorder, privacy redaction scanner (`r2`), offline support bundle projector, SQLite incident outbox, and opt-in native background GitHub issue reporter for the MindIE Agent ecosystem.

## 1. Module Responsibilities & File Navigation

| Module | Responsibility & Key Entry Points |
| :--- | :--- |
| [`mindie_diagnostics/fallback.py`](mindie_diagnostics/fallback.py) | Standalone stdlib-only local failure recorder (`record_failure`) and reporting policy reader (`read_policy`, `consent_status`, `scope_key`). Can be vendored as a single file; never spawns services or raises into callers. |
| [`mindie_diagnostics/integration.py`](mindie_diagnostics/integration.py) | Installed-package metadata enrichment (`record_failure`), atomic policy configuration (`configure_reporting`), and read-only status aggregation (`reporting_status`). |
| [`mindie_diagnostics/logging.py`](mindie_diagnostics/logging.py) | Per-process structured JSONL logger (`Recorder`, `Operation`, `append_failure_event`) with `O_EXCL` first-use slots, reader-aware segment recycling (`MAX_LOG_BYTES = 1 MiB`, `BACKUP_COUNT = 3`) and fork-safe lock reset. |
| [`mindie_diagnostics/redact.py`](mindie_diagnostics/redact.py) | Pure `r2` privacy/credential scanner (`scan_text`, `scan_tree`, `redact_text`, `Allowlist`) with 21 deterministic rules and zero external dependencies. |
| [`mindie_diagnostics/bundle.py`](mindie_diagnostics/bundle.py) | Allowlist-only public event projection (`export_public_event`, `export_issue_event`) and bounded offline support bundle builder (`collect_bundle`). |
| [`mindie_diagnostics/outbox.py`](mindie_diagnostics/outbox.py) | SQLite incident outbox (`Outbox`, `reporter.sqlite3`) managing active (`pending`, `retry`, `uncertain`) and compacted terminal (`published`, `withdrawn`, `expired`, `exhausted`, `blocked`, `permanent-failed`) incident lifecycles. |
| [`mindie_diagnostics/ingestion.py`](mindie_diagnostics/ingestion.py) | Incremental log consumer (`ingest`) with durable per-inode byte cursors (`cursors` table in `reporter.sqlite3`). |
| [`mindie_diagnostics/reporter.py`](mindie_diagnostics/reporter.py) | Conservative GitHub issue publisher (`GitHub`, `publish_one`, `fingerprint`, `issue_payload`, `render_issue`) using `gh api --include` with marker reconciliation (`<!-- mindie-incident:<sha256> -->`). |
| [`mindie_diagnostics/reader_registry.py`](mindie_diagnostics/reader_registry.py) | Lifecycle-owned reader registration (`register_reader`, `unregister_reader`, `reader_guard`), read-only cursor snapshot inspection (`cursor_snapshot`), and `PressureReader`. |
| [`mindie_diagnostics/maintenance.py`](mindie_diagnostics/maintenance.py) | Offline log segment retention (`prune`) and `retention-state.json` backpressure marker management. |
| [`mindie_diagnostics/reporting_runtime.py`](mindie_diagnostics/reporting_runtime.py) | Immutable content-addressed runtime generation builder (`prepare_runtime`, `runtime_transaction`, `retire_other_generations`) under `<policy>.runtime/generations/<sha256>/venv`. |
| [`mindie_diagnostics/reporting_commands.py`](mindie_diagnostics/reporting_commands.py) | Explicit lifecycle operations: offline maintenance (`maintain`), service provisioning (`ensure`), and bounded automatic runtime upgrade (`upgrade_running`). |
| [`mindie_diagnostics/service.py`](mindie_diagnostics/service.py) | Linux `systemd --user` unit installer/controller (`install_service`, `ensure_reporter_service`, `service_status`, `start_service`, `remove_service`) and cross-platform dispatcher. |
| [`mindie_diagnostics/platform_service.py`](mindie_diagnostics/platform_service.py) | macOS `LaunchAgent` (`launchctl`) and Windows Scheduled Task (`powershell.exe`) native service controller and `run_worker` entry point. |
| [`mindie_diagnostics/health.py`](mindie_diagnostics/health.py) | Worker heartbeat writer (`Health`, 15s daemon thread) and status reader (`read_health`, `worker-health.json`). |
| [`mindie_diagnostics/process_output.py`](mindie_diagnostics/process_output.py) | Scoped FD 1/2 (`stdout`/`stderr`) pipe capture (`capture_output`) for long-lived service processes. |
| [`mindie_diagnostics/context.py`](mindie_diagnostics/context.py) | `ContextVar` correlation ID propagation (`bind_context`, `current_context`, `wrap_context`). |
| [`mindie_diagnostics/cli.py`](mindie_diagnostics/cli.py) | CLI entry point (`mindie-diagnostics`) for `bundle`, `status`, `service`, `reporting`, and `worker` commands. |

## 2. Core State & Timeout Arithmetic Invariants

1. **External-Effect Honesty (`pending` / `retry` / `uncertain` / `published`)**:
   - `publish_one` always queries existing repository issues (`find_issue`) for `<!-- mindie-incident:<sha256> -->` before creating an issue.
   - `Outbox.begin_post` transitions leased items to `state='uncertain'` **before** spawning a mutating `POST repos/<repo>/issues` request.
   - If a `POST` fails after process start or times out, the item remains `uncertain` and will reconcile against GitHub markers on the next cycle before ever attempting another `POST`.
   - Transient network, authentication, and authority-read failures retain pending evidence with capped backoff; read attempts never permanently exhaust an incident. Unknown POSTs only reconcile and never replay.
   - `_record_publication_outcome` never turns an acknowledged external GitHub publication (`published` / `reconciled`) into a remote failure if the local SQLite receipt update fails (`status="recording_failed"`, `publication_status="published"`).
2. **Consent Gate & Non-Backfill Guarantee**:
   - Local fault logging (`record_failure`, `Recorder`) works without network or reporting consent.
   - Events recorded while reporting is disabled carry `reporting=None` and are never backfilled if reporting is enabled later.
   - `ConsentUnavailable` (missing or corrupt policy/linked contribution authority) pauses ingestion/publication without advancing byte cursors or discarding queued evidence; `ConsentWithdrawn` (disabled choice or revision mismatch) immediately transitions queued items to `withdrawn` and compacts their payloads into minimal receipts.
   - A plugin policy links its profile's `mindie-consent/1` authority. Only an explicit `contribute` choice carrying `choice_revision` authorizes reporting. Events, ingestion and every remote action recheck this same authority. Each profile owns its service, queue and policy; shared log roots are filtered by policy path.
3. **Timeout & Lock Budgets**:
   - **Hot Path (`record_failure` / `Recorder._emit`)**: Non-blocking locks (`blocking=False`). A never-existing slot is claimed with `O_EXCL` across the fixed `BACKUP_COUNT + 1` window and makes zero reader-registry or SQLite calls. Recycling an existing owned segment keeps the existing `100ms` cooperative work cutoff (`50ms` `reader_guard` wait, then `cursor_snapshot` with the remaining request); that is a best-effort admission boundary, not a hard latency ceiling, because `cursor_snapshot` retains the project-wide `3.0s` SQLite wait floor. Backpressure forbids new-slot growth.
   - **Read-Only Status (`reporting_status` / `local_snapshot`)**: `5.0s` SQLite read-only progress-handler deadline (`busy_timeout=3000`) and `150ms` local log tail scan deadline (`<= 256` entries, `<= 8` 16KiB tails).
   - **Outbox SQLite (`Outbox.connect`)**: `timeout=3.0s`, `PRAGMA busy_timeout=3000`, `BEGIN IMMEDIATE` write transactions.
   - **GitHub Transport (`GitHub` / `_run_gh`)**: `30s` default cycle timeout (`lease_seconds = timeout + 20`), reserving up to `15s` inside that same deadline for `POST`; process-group reaping uses `SIGTERM` -> `2.0s` grace -> `SIGKILL`. Explicit worker ensure shares one `75s` deadline across preparation and native service commands.
   - **Runtime Handoff (`maintain` / `upgrade_running`)**: Shared monotonic budget capped at `75s` in `maintain` (requires `>= 61s` remaining to enter `upgrade_running`), with `_LOCK_WAIT_S = 12.0s`, `_READBACK_S = 8.0s`, and `_ROLLBACK_RESERVE_S = 20.0s` reserved for rollback to the previous generation on failed readback.

## 3. Verification Commands

Run the full test suite from the repository root:

```bash
pytest -q
```
