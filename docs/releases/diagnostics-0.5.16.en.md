# mindie-diagnostics 0.5.16: Scale-to-Zero Outbox Drain, 7-Day Pending Expiry, and `mindie-harness` Triage Routing

[中文](diagnostics-0.5.16.md)

`mindie-diagnostics v0.5.16` delivers clean-architecture state-service hardening and cross-platform determinism improvements:

- **Scale-to-Zero On-Demand Outbox Drain & 7-Day Pending TTL**: Replaces idle background polling with bounded on-demand outbox draining that exits immediately once empty when opt-in reporting is enabled, and expires unsent `pending` outbox items after `7d` to prevent unbounded stale accumulation.
- **`mindie-harness` Attribution & Explicit `TimeoutError` Propagation**: Registers `mindie-agent/harness` (`mindie_harness`) in deterministic stack-frame component routing and precedes `except OSError:` handlers with `except TimeoutError: raise` so timeouts never masquerade as benign I/O misses.
- **`state-service` Governance Alignment**: Aligns contributor documentation and CI gates with `--profile state-service` (`SVC-CONSENT`, `SVC-STATE-COMPAT`, `SVC-LIFETIME`, `SVC-EXPORT`).
- **Version**: `0.5.16`.
