# Production fault reporting for the first internal release

MindIE retains the VAWS automatic production-fault reporting capability. The
existing component is the owner: a component records a bounded local fault,
the installed independent reporter ingests it, and a model-free allowlist
produces a GitHub Issue. There is no transcript scan or extra model invocation.
The caller's result remains authoritative when diagnostics fail.

The independent saved reporting choice is offered at initial setup. A missing
choice means off; enabled, disabled and later are retained without per-task
prompts. An enabled policy and a healthy installed reporter are different
facts. Reporter installation happens once outside native Hooks. No user's live
choice was changed during this validation.

The existing publication contract is unchanged: fingerprints deduplicate faults;
an uncertain Issue POST is reconciled by its original marker without another
POST; revoked authorization cannot resurrect pending evidence. Public reports
exclude raw exception text, task content, commands, paths, credentials, local
incident IDs and session IDs. Reporter and logging failures remain visible
separately from the business result.

## Windows readiness correction

The previous `_worker_view` probed process liveness only on POSIX, making an
otherwise healthy Windows reporter permanently unverified. Version 0.4.1 adds
read-only process-handle and creation-time checks. A health record captures the
worker's creation FILETIME; a reader matches that value with the live process,
rejecting PID reuse. Access denial, unsupported inspection and old records
without identity remain unverified. No signal or process mutation is used.

## Evidence and limits

Local component suite: 291 passed and four skipped. Focused tests cover identity
match/mismatch, an exited or inaccessible process, missing legacy identity,
authorization transitions, redaction, deduplication, and uncertain POST
reconciliation. Existing macOS component behavior remains covered.

Native Lenovo Windows 11, Python 3.13.15, validated the actual Win32 API path:
current-worker creation identity, simulated stale health identity, and an owned
child's exit. The Claude Code adapter was exercised through real local fault
files, ingestion and SQLite outbox, with a synthetic in-memory Issue publisher.
That pipeline checks private-content canaries, duplicate faults, an uncertain
POST followed by readback, off-by-default behavior and no historical backfill.
The first pipeline run passed its functional assertions but failed test-directory
cleanup because the test retained log handles; the fixture now closes only its
own recorder before removing its directory.

These are process/API and adapter/component checks. Native scheduled-task
installation, Claude/Codex host event delivery, an actual GitHub Issue, and a
complete production fault under the frozen release package still require their
separate acceptance evidence. No public Issue was created by these tests.
