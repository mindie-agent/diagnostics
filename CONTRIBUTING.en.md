# Contributing to `mindie-diagnostics`

[中文](CONTRIBUTING.md) | **English** · [Back to Homepage](README.en.md)

Thank you for contributing to **MindIE Diagnostics** (`mindie-agent/diagnostics`, repository class: `state-service`)! This repository provides zero-dependency offline failure attribution, strict allowlist projection, Triage Card v2 rendering, bounded ring-buffer logging, and Scale-to-Zero consent-gated GitHub Issue reporting.

---

## 1. Core Engineering Rules (`state-service` Checklist Contract)

1. **`SVC-CONSENT` — Allowlist Privacy & Zero Retroactive Backfill**:
   Public reporting is opt-in by default and never exports raw exception text (`str(e)`), shell commands, environment values, file paths, or host endpoints. Disabling reporting withdraws unsent items immediately without retroactive backfill upon re-enablement; unreadable policy files raise `PolicyUnavailable` rather than resetting state.
2. **`SVC-STATE-COMPAT` — Zero Runtime Dependencies & Rollback-Safe State**:
   Keep the runtime package 100% Python stdlib-only. Preserve `fallback.record_failure`, `service_environment.py`, `reporter.sqlite3`, and `reporting_commands.ensure()` (`status="running"`, `mode="on_demand"`) compatibility.
3. **`SVC-LIFETIME` — Scale-to-Zero Drain Worker, Safe Service Retirement & Symlink Support**:
   Never run a 24/7 idle loop, a 15-second heartbeat `fsync` thread, or a `KeepAlive` respawn loop: the worker drains a bounded batch, schedules a durable retry wake **only** if active work remains, and exits `0` immediately when idle. Retire legacy OS services across `<config>.state/service` and user service directories (awaiting `launchctl bootout` on macOS, using windowless `pythonw.exe` + `CREATE_NO_WINDOW` on Windows, and handling non-systemd WSL honestly) while preserving user-disabled choices before retiring runtime venvs. Support Stow/Nix/worktree symlinks on config and log roots. Keep core modules `< 800` lines with `[Role]` / `[Invariants]` / `[Timeouts & Locks]` / `[Side-Effect Policy]` headers.
4. **`SVC-EXPORT` — Pending-Only `7d` Expiry & Uncertain POST Reconciliation**:
   Expire **only** never-dispatched `pending` incidents (`attempts == 0` and `attempt_timestamp IS NULL`) older than 7 days (`state="expired"`). **Never** age-expire `uncertain` POST receipts, `published` deduplication fingerprints, consent revisions, or `withdrawn` receipts. Always place `except TimeoutError: raise` before `except OSError:`.

---

## 2. Local Isolated Testing & Required Gates

```bash
# 1. Run isolated unit and integration tests (conftest.py isolates all user/host paths)
python3 -B -m pytest -q -m "not live_npu"

# 2. Verify offline zero-dependency wheel install outside the checkout
uv build --wheel --out-dir /tmp/diag-dist
uv venv /tmp/diag-smoke && uv pip install --python /tmp/diag-smoke/bin/python --no-index --find-links /tmp/diag-dist mindie-diagnostics
(cd /tmp && /tmp/diag-smoke/bin/mindie-diagnostics --help >/dev/null)

# 3. Check Unix LF line endings and state-service PR checklist
python3 .github/scripts/check_lf_eol.py
python3 .github/scripts/check_pr_checklist.py --profile state-service --body-file <pr_body.md>
```

---

## 3. Submitting a Pull Request

Complete [`.github/pull_request_template.md`](.github/pull_request_template.md) with concrete verification evidence for `SVC-CONSENT`, `SVC-STATE-COMPAT`, `SVC-LIFETIME`, and `SVC-EXPORT` (or `not-applicable: <specific reason>` for docs-only items).
