# mindie-diagnostics 0.5.2：协作式摄取排空、离线阶段准入预算与记录边界游标保持

[English](https://github.com/mindie-agent/diagnostics/blob/v0.5.2/docs/releases/diagnostics-0.5.2.en.md)

`mindie-diagnostics 0.5.2` 在单轮日志摄取既有的文件数与字节上限基础上，为后台诊断 Worker 与离线维护命令（`reporting maintain`）补齐了协作式停止信号（`cancel`）与截止时刻（`deadline`）控制：触发停止或维护预算耗尽时，在完整 JSONL 记录边界停止扫描、保存已入队记录的字节游标，并停止准入后续阶段。

---

## 一、核心修复与行为变更

### 1. 在既有限额上增加记录边界协作停止与游标保持（`mindie_diagnostics/ingestion.py`，PR `#29`）

- 单轮 `ingest()` 原本已有 `max_files=256`、`max_bytes=8 MiB` 与 `4096` 个目录项的扫描上限，但在该限额内遇到后台服务停止或维护预算到期时无法提前退出。
- `0.5.2` 为 `ingest(..., cancel=None, deadline=None)` 增加了协作式停止检查：
  - 在候选 JSONL 文件之间以及每条完整 JSONL 记录处理后检查 `cancel` 事件与单调时钟 `deadline`；
  - 一旦触发，立即在**完整记录边界**退出扫描并标记 `counts["limited"] = 1`，同时将本文件最后一条已完成入队记录的字节偏移（`offset`）写入 SQLite `cursors` 表；
  - 下一周期从该 `offset` 继续增量读取，不把未扫描记录标为已消费（注：该机制是面向正常停止与预算截止的协作式记录边界排空，不宣称在进程被强杀 `SIGKILL` 或崩溃时提供跨进程 `exactly-once` 保证）。

### 2. Worker 收到停止信号后停止准入后续阶段（`mindie_diagnostics/cli.py`，PR `#29`）

- 后台 Worker 的 `run_cycle(..., cancel=None)` 将 `SIGINT` / `SIGTERM` 停止事件传入 `ingest(..., cancel=cancel)`，并在各阶段开始前检查停止状态：
  - 一旦 `cancel` 置位，保留本轮已收集的数据，不再启动后续诊断根目录的 `ingest`、本地留存清理（`prune`）或远端上报（`publish_one`）阶段；
  - 被中断的周期如实返回 `status="degraded"` 与 `limited=1`。

### 3. `reporting maintain --budget-seconds` 离线阶段准入预算（`mindie_diagnostics/reporting_commands.py`，PR `#29`）

- `reporting maintain` 是纯离线维护阶段（返回 `"network": False`），仅负责队列过期清理、撤回授权清理、按根目录增量 `ingest`、本地 `prune` 以及可选的运行中服务切换，**不构建上报压缩包、不执行远端 GitHub 对账，也不发起 `POST` 请求**。
- `budget_seconds`（默认 `75` 秒）是**阶段准入预算**，而非操作系统级 20 秒停机保证或总耗时硬上界：
  - 启动每个根目录的 `ingest` 或 `prune` 前，要求至少剩余 `3.0` 秒预算，且 `ingest` 在 `deadline - 3.0s` 停止扫描新记录，为最后一条记录的 `Outbox` SQLite 入队与 `cursors` 写入预留合法的 `3.0` 秒锁等待窗口；
  - 若剩余时间不足 `3.0` 秒，则跳过后续阶段，记录 `{"status": "unavailable", "category": "budget_exhausted", "limited": True}`，并将汇总状态标为 `status="degraded"`。

### 4. 出站对账读取有界化后续设计说明（PR `#29`）

- 远端 GitHub Issue 搜索与评论对账读取的有界化方案作为未实施的后续设计记录在仓内 [`docs/outbound-reconciliation-followup.md`](https://github.com/mindie-agent/diagnostics/blob/v0.5.2/docs/outbound-reconciliation-followup.md) 中，与本次已落地的本地离线维护收敛严格区分。
- 外部副作用保护规则保持不变：结果未知（`unknown`）的 `POST` 回执绝不重放，GitHub 搜索未命中绝不当作远端未创建。

---

## 二、测试验证与已知边界

- **本地契约测试**：在 Python 3.11 与 Python 3.13 下运行本地测试套件，**`132 passed`**。
- **已知边界说明**：此前在 Windows 环境下的历史测试运行中，曾出现过一次旧同意隔离测试（legacy consent isolation test）预期摄取 2 条记录但实际摄取 1 条、随后重跑通过的现象。该偶发波动的根因尚未查明，`0.5.2` 不宣称已根因修复该问题。
