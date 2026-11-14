# mindie-diagnostics 0.5.13

## 关键修复与兼容性增强

- **Windows 无弹窗跨仓门禁（`R4`）兼容导出**：在 `mindie_diagnostics.github_transport` 中显式定义 `_CREATE_NO_WINDOW` 并在 `mindie_diagnostics.reporter` 中同步重导出，修复 `v0.5.11` 拆分 `github_transport.py` 后 `check_auto_update_readiness.py` 对 `mindie_diagnostics/reporter.py` 的无窗口子进程门禁检查。
