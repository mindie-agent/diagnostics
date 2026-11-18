# mindie-diagnostics 0.5.17: Unified Read-Only `mindie_doctor` Installation Check

[中文](mindie-diagnostics-0.5.17.md)

`mindie-diagnostics v0.5.17` adds one unified installation check, `mindie-diagnostics doctor` / `run_doctor()`, that summarizes runtime, privacy consent, remote NPU, GitHub channel, and local diagnostics in one structured JSON report:

- **Unified `mindie-doctor/1` Report with Three Outcomes**: Emits `healthy` / `attention_needed` / `degraded`. Missing optional `gh`, unconfigured endpoint aliases, or disabled reporting remain `healthy` under `local-only` / `read-only`; only genuine faults such as corrupt `mindie-consent.json` / `endpoints.json`, unauthenticated `gh` under `contribute`, or a failed remote probe become `degraded`.
- **Strictly Read-Only, Zero Host Pollution**: On an empty installation doctor never creates directories, `reporter.sqlite3`, `-wal` / `-shm` sidecars, `worker-health.json`, or `mindie-consent.json`, and never starts a background worker. Endpoint fields such as `password` never enter the report.
- **Bounded Timeouts and Remote Reachability**: The local-only budget is `5.0s`; `--probe-remote` uses a `15.0s` outer budget (`5.0s` local + `8.0s` remote + `2.0s` cleanup grace). `gh` is capped at `2.5s` and SSH at `8.0s`, and every subprocess inherits the remaining monotonic deadline. Windows uses `CREATE_NO_WINDOW`.
- **Actionable Remediations**: Concise hints cover choosing a consent tier, missing `gh` login under `contribute`, incomplete community publishing configuration, damaged endpoint JSON, and connecting an NPU host, while reminding agents that native `ssh` or local scripts remain available when a helper tool has issues.
- **Version**: `0.5.17`.
