# mindie-diagnostics 0.5.16：按需 Scale-to-Zero 上报排空、7d 过期治理与 `mindie-harness` 归属分诊

[English](diagnostics-0.5.16.en.md)

`mindie-diagnostics v0.5.16` 完成状态服务洁净架构收敛与跨平台确定性治理：

- **Scale-to-Zero 按需排空与零后台常驻驻留**：取消空闲态长驻后台轮询守护，改为仅在用户显式开启上报（Opt-in）且存在待发项时按需有界排空并在完成后退出；对未发送的 `pending` 条目强制执行 `7d` TTL 过期清理，杜绝断网或撤回授权后的陈旧积压。
- **`mindie-harness` 底座异常归属识别与 `TimeoutError` 严格透传**：在自动故障分类与堆栈归属表中正式注册 `mindie-agent/harness`（`mindie_harness`），确保共享底座自身缺陷精准路由至对应仓库；全量 `except OSError:` 边界显式前置 `except TimeoutError: raise`，避免超时被静默吞没为普通文件 I/O 异常。
- **`state-service` 贡献门禁与文档对齐**：补齐扁平同级 Worktree 贡献指南与 `--profile state-service` PR 设计原则门禁（`SVC-CONSENT` / `SVC-STATE-COMPAT` / `SVC-LIFETIME` / `SVC-EXPORT`）。
- **版本**：`0.5.16`。
