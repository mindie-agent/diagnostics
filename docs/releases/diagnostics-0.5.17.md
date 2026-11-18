# mindie-diagnostics 0.5.17：统一只读 `mindie_doctor` 安装体检

[English](mindie-diagnostics-0.5.17.en.md)

`mindie-diagnostics v0.5.17` 新增统一的安装体检入口 `mindie-diagnostics doctor` / `run_doctor()`，用一份结构化 JSON 汇总运行时、隐私授权、远程 NPU、GitHub 通道与本地诊断状态：

- **统一 `mindie-doctor/1` 报告与三态结论**：输出 `healthy` / `attention_needed` / `degraded`；缺失可选的 `gh`、未配置端点别名或未启用上报在 `local-only` / `read-only` 模式下保持 `healthy`，只有损坏的 `mindie-consent.json` / `endpoints.json`、`contribute` 模式下 `gh` 未认证或远程探测失败等真实故障才标记 `degraded`。
- **严格零副作用只读体检**：在空安装上绝不创建目录、`reporter.sqlite3`、`-wal` / `-shm` 伴随文件、`worker-health.json`、`mindie-consent.json`，也绝不启动任何后台 Worker；端点文件中的 `password` 等字段永不进入报告。
- **有界超时与远程可达性**：默认本地预算 `5.0s`，`--probe-remote` 时外层预算 `15.0s`（`5.0s` 本地 + `8.0s` 远程 + `2.0s` 清理余量）；`gh` 最多 `2.5s`，SSH 最多 `8.0s`，每个子进程都继承剩余单调时钟截止时间；Windows 使用 `CREATE_NO_WINDOW`。
- **可执行的修复提示**：针对未选择授权档位、`contribute` 模式缺少 `gh` 登录、社区发布配置不完整、端点 JSON 损坏或未配置 NPU 主机等场景给出简洁 `remediations`，并说明任何辅助工具异常时仍可通过原生 `ssh` / 本地脚本继续任务。
- **版本**：`0.5.17`。
