# mindie-diagnostics 0.5.5

This release focuses on component boundary decoupling, bounded GitHub outbound reconciliation aligned with maintenance prune budgets, and cross-platform (Windows / Ubuntu 26.04) service and test hardening:

- **Boundary Decoupling & Extraction of `mindie-debugger`**:
  - Extracted Ascend NPU CANN `plog` fault triage (`triage_npu_fault`) and the three-primitive NPU precision/memory-stomp toolkit (`npu-watch`, `lockstep-diff`, `call-repro`) into the standalone `mindie-debugger` (`v0.1.0`) package. `mindie-diagnostics` remains a zero-dependency package dedicated to bounded local event recording, R2 redaction, offline support bundles (`collect_bundle` / `export_public_event`), and opt-in telemetry reporting.
- **Bounded GitHub Outbound Reconciliation & Prune Budget Alignment**:
  - Enforced strict time/item bounds on GitHub outbound reconciliation and aligned `maintenance.prune` budgets so slow network probes or pending outbox entries cannot starve maintenance windows.
- **Cross-Platform Service Verification & Windows Test Hardening**:
  - Handled Windows `gh` CLI stubs and `sitecustomize` subprocess invocation (preventing `cmd.exe` from splitting query URLs on `&`), accommodated unprivileged Windows symlink (`WinError 1314`) and `MAX_PATH` boundaries in tests, and adapted service unit verification for Ubuntu 26.04 `systemd-analyze verify` warnings.
- **Documentation & Host Ecosystem Sync**:
  - Updated documentation tables, unified family brand assets, and added Google Antigravity (`mindie-agent-agy`) host plugin documentation.

The package `__version__` is synchronized to `0.5.5`.
