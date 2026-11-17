# MindIE Diagnostics (`mindie-diagnostics`) — State-Service Architecture & Operating Guide

`mindie-diagnostics` (`state-service` class, zero runtime third-party dependencies) owns offline failure attribution (`fallback.record_failure`, `integration.record_failure`), strict allowlist projection (`bundle.py`, `redact.py`), Triage Card v2 rendering (`triage_card.py` including `mindie-harness` co-owned stack attribution), bounded ring-buffer log retention (`4 x 1 MiB`), and the Scale-to-Zero consent-gated GitHub Issue outbox (`outbox.py`, `reporter.py`, `wake_schedule.py`).

---

## 1. Submodule Map

- **Zero-Dependency Local Recording & Privacy Projection**:
  - [`fallback.py`](mindie_diagnostics/fallback.py): Stdlib-only non-blocking failure recorder (`record_failure`) and consent policy reader (`read_policy`, `consent_status`, `PolicyUnavailable`), supporting symlinked user config/log roots via `realpath` while enforcing `O_NOFOLLOW` on internal `.jsonl` segments.
  - [`integration.py`](mindie_diagnostics/integration.py) & [`evidence.py`](mindie_diagnostics/evidence.py): Installed-package metadata enrichment, co-owned `mindie_harness.*` frame retention on host-plugin incidents, `configure_reporting`, and read-only `reporting_status` (`mode="on_demand"`).
  - [`bundle.py`](mindie_diagnostics/bundle.py), [`redact.py`](mindie_diagnostics/redact.py) & [`triage_card.py`](mindie_diagnostics/triage_card.py): Allowlist projection, deterministic `r2` leak scan, `"mindie-harness": ("mindie-agent/harness", "package")` registration, and Triage Card v2 rendering.
- **Scale-to-Zero Drain Worker, Outbox & OS Service Lifecycle**:
  - [`cli.py`](mindie_diagnostics/cli.py), [`health.py`](mindie_diagnostics/health.py) & [`wake_schedule.py`](mindie_diagnostics/wake_schedule.py): Coalesced on-demand drain worker (`worker-drain.lock`, bounded batch `<= 8`), zero background heartbeat thread (`mode="on_demand"`, `status="idle"`), durable OS retry wake scheduling (`<state>/next-wake.json` + companion timer only when active work remains), and dual-path legacy OS service retirement.
  - [`outbox.py`](mindie_diagnostics/outbox.py) & [`reporter.py`](mindie_diagnostics/reporter.py): SQLite WAL queue (`reporter.sqlite3`), `7d` expiry (`PENDING_TTL_SECONDS`) strictly on never-dispatched `pending` rows while preserving `uncertain` POST receipts, `published` dedup fingerprints, and `withdrawn` receipts, plus `begin_post` atomic uncertainty persistence.
  - [`service.py`](mindie_diagnostics/service.py), [`platform_service.py`](mindie_diagnostics/platform_service.py), [`reporting_commands.py`](mindie_diagnostics/reporting_commands.py), [`reporting_runtime.py`](mindie_diagnostics/reporting_runtime.py) & [`runtime_retirement.py`](mindie_diagnostics/runtime_retirement.py): Cross-platform user service management (`launchd` awaiting `bootout`, `systemd --user` + non-systemd WSL local fallback, Windows Scheduled Task `pythonw.exe` `Subsystem == 2` + `CREATE_NO_WINDOW`), preserving user-disabled choices and retiring legacy services before venv GC.

---

## 2. State-Service Invariants & Five Iron Rules (`SVC-*`)

1. **`SVC-CONSENT`**: `record_failure` never triggers network calls or starts workers. Public GitHub reporting requires explicit opt-in (`mindie.diagnostics.reporting.v1` + linked `mindie-consent/1` `choice_revision`); withdrawing consent immediately marks queued items `withdrawn` and re-enabling never backfills prior events. Unreadable policy files raise `PolicyUnavailable` without wiping queued evidence.
2. **`SVC-STATE-COMPAT`**: Keep `mindie_diagnostics` 100% stdlib-only at runtime. Preserve `reporter.sqlite3` schema, `reporting_commands.ensure()` return compatibility (`status="running"`, `mode="on_demand"`), and `service_environment.py`. Configure SQLite WAL + `busy_timeout=3000` once in `Outbox.connect`.
3. **`SVC-LIFETIME`**: Scale-to-Zero worker exits `0` immediately when idle—zero 24/7 idle loops, zero 15s heartbeat `fsync` threads, and zero `KeepAlive` respawn loops. Missing idle heartbeats are healthy. Support legitimate Stow/Nix/worktree symlinks on config and log roots while keeping `O_NOFOLLOW` on internal `.jsonl` segments, DrvFS `.env` files, and `site-packages` source collection. Keep all modules `< 800` lines with `[Role]` / `[Invariants]` / `[Timeouts & Locks]` / `[Side-Effect Policy]` headers.
4. **`SVC-EXPORT`**: Precede every `except OSError:` with `except TimeoutError: raise`. Persist `uncertain` via `begin_post` before any mutating GitHub POST; reconcile `uncertain` strictly via read-only search/listing without blind replay. Expire **only** never-dispatched `pending` rows older than `7d`—never expire `uncertain`, `published`, `withdrawn`, or consent revisions.

---

## 3. Isolated Verification & PR Gates

Isolate `HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `XDG_*`, `MINDIE_*`, `REMOTE_DEV_AUTH_DIR`, `UV_CACHE_DIR`, and `TMPDIR` in all tests:

```bash
python3 -B -m pytest -q -m "not live_npu"
python3 .github/scripts/check_lf_eol.py
python3 .github/scripts/check_pr_checklist.py --profile state-service --body-file <pr_body.md>
```
