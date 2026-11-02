# mindie-diagnostics 0.5.5

本次更新聚焦组件职责收敛、GitHub 外发对账预算边界对齐以及跨平台（Windows / Ubuntu 26.04）服务与测试稳健性增强：

- **职责解耦与 `mindie-debugger` 独立拆仓**：
  - 将昇腾 NPU CANN `plog` 故障分诊（`triage_npu_fault`）与三原语 NPU 精度/内存踩踏调试工具包（`npu-watch`、`lockstep-diff`、`call-repro`）正式提取至独立组件 `mindie-debugger`（`v0.1.0`）。`mindie-diagnostics` 保持零第三方依赖，专注本地有界事件记录、R2 脱敏、离线诊断包（`collect_bundle` / `export_public_event`）与显式授权遥测上报。
- **GitHub 外发对账有界化与清理预算对齐**：
  - 为 GitHub Issue 外发对账（Outbound Reconciliation）施加严格的有界预算，并与 `maintenance.prune` 的离线清理预算对齐，防止慢速网络或存量挂起项挤占维护窗口。
- **跨平台服务校验与 Windows 测试兼容修复**：
  - 处理 Windows 环境下的 `gh` CLI Stub 调用与 `sitecustomize` 进程参数转义（避免 `cmd.exe` 截断含 `&` 的查询 URL），兼容无符号链接特权（`WinError 1314`）及长路径边界，并适配 Ubuntu 26.04 `systemd-analyze verify` 告警行为。
- **文档与宿主生态同步**：
  - 更新组件文档表与统一品牌标识，新增 Google Antigravity (`mindie-agent-agy`) 宿主插件支持说明。

包内 `__version__` 同步为 `0.5.5`。
