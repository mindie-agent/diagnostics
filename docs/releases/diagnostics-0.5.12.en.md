# mindie-diagnostics 0.5.12: Transport Classification, Windows PowerShell 5.1 / CP936 & WSL2 Service Hardening

[中文](diagnostics-0.5.12.md)

Hardens cross-platform transport error policy, windowless PowerShell UTF-8 stream handling, and WSL2 executable/mount/service boundaries:

- **Transport Failure Policy Alignment (`fallback.py`)**: Classifies `ssl.SSLCertVerificationError` as `("credentials", False, "update_credentials_required")` and mid-stream `ssl.SSLError`/`SSLEOFError`, `socket.gaierror`/`herror`, and `http.client.IncompleteRead`/`RemoteDisconnected`/`BadStatusLine` as `("connectivity", False, "update_transport_unavailable")` so transient TLS/network drops never auto-file false `internal` GitHub Issues, while keeping `InvalidURL`/`ImproperConnectionState`/`NotConnected` as `internal`.
- **Windowless PowerShell 5.1 Direct UTF-8 Pipe Binding (`platform_service.py`)**: Binds `[Console]::SetOut` and `[Console]::SetError` directly over `[Console]::OpenStandardOutput()` / `[Console]::OpenStandardError()` with UTF-8 (`no BOM`), preventing headless `CREATE_NO_WINDOW` fallback to `CP936` on non-ASCII task paths.
- **WSL2 `.exe`/PE/DrvFS Shadowing Defense & `metadata`-Aware Permissions (`filesystem.py`, `github_transport.py`, `service.py`, `reporting_runtime.py`)**: Executes the exact native binary path resolved by `which_native`; enforces `0700`/`0600` permissions on native POSIX filesystems and `metadata`-enabled DrvFS mounts while accommodating fixed-mode single-owner DrvFS mounts without `metadata`.
- **Non-`systemd` Linux / WSL2 Local Worker Identity & Transaction Rollback (`platform_service.py`, `service.py`)**: Falls back to a managed local background worker when `systemctl --user` is unavailable, requiring non-empty verified process identity (`uid` + start time) before recording or signaling any PID, cleaning up unregistered children on registration failure, and restoring unit/manifest transactions before propagating `TimeoutError`.

In-package `__version__` is aligned to `0.5.12`. Actual test and benchmark results are recorded in the packaged `benchmark-results.json`, `benchmark-results.md`, and CI runs.
