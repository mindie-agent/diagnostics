# Changelog

## 0.5.8 (2026-11-06)

- Validate the actual PE GUI subsystem before installing or starting a Windows scheduled worker; reject console executables named `pythonw.exe` and allow a normal retry after environment repair.

## 0.5.7 (2026-11-04)

- Publish `errno` and a safe path label (`mcp-binds.sqlite3`) for OSError events (`#35`).

## 0.5.6 (2026-11-04)

- Enforce `subprocess.CREATE_NO_WINDOW` across all Windows background reporting, Task Scheduler (`schtasks`), and `gh` CLI subprocess calls (`reporter.py`, `platform_service.py`, `reporting_commands.py`, `reporting_runtime.py`) to eliminate console window popups.

## 0.5.5 (2026-11-02)

- Extract Ascend NPU fault triage (`triage_npu_fault`) and the 3-primitive precision debugging toolkit (`npu-watch`, `lockstep-diff`, `call-repro`) into the standalone `mindie-debugger` package.
- Bound GitHub outbound reconciliation and align offline `maintenance.prune` budgets.
- Fix Windows `gh` CLI stub handling, `sitecustomize` subprocess URL `&` escaping, unprivileged symlink/path test guards, and Ubuntu 26.04 `systemd-analyze verify` warning handling.
- Update documentation tables, unified MindIE Agent brand assets, and Google Antigravity (`mindie-agent-agy`) host support docs.
