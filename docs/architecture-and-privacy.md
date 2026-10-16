# 架构、隐私白名单与日志保留规范

**中文** · [English](./architecture-and-privacy.en.md) · [返回文档中心](./README.md)

本文档说明 `mindie-diagnostics` 的故障归属边界、公开上报的隐私白名单、授权与策略存储分离机制，以及有界本地日志轮转与维护规范。

---

## 1. 故障归属边界（Failure Ownership）

仅在真正拥有该故障的组件边界调用 `mindie_diagnostics.integration.record_failure`：

```python
from mindie_diagnostics.integration import record_failure

result = record_failure(
    "remote-dev",
    "mcp.request",
    stage="decode",
    category="protocol_mismatch",
    exception=error,
)
```

- **静态元数据约束**：仅传入静态产品标识与类别，绝不将用户输入作为 `category` 或 `operation`。
- **非阻塞降级**：当本地写入成功时返回随机 `incident_id`；若磁盘满或写失败，则暴露 `logging_failed`，绝不改变原工具的业务返回值，也不触发安装、网络发布或模型调用。
- **区分调用方错误与产品缺陷**：调用方入参错误、缺失配置、未激活准入、取消、权限不足、网络/SSH 认证失败、业务命令非零退出或超时、以及外部模型自身报错，均不属于 MindIE Agent 产品缺陷，应以 `reportable=False` 仅保留在本地；仅确认的内部异常与自有工具协议损坏可标记为可上报。

---

## 2. 公开上报与严格白名单隐私

当用户显式开启自动上报（`reporting configure --enabled true` 并执行 `reporting ensure`）后，后台 Worker 在不调用任何模型的前提下，严格按白名单构造公开 GitHub Issue：

- **白名单允许字段**：组件名称、已加载版本与 Git 修订号、操作与阶段名、静态错误类别、单调时钟耗时、退出/清理事实、产品代码栈帧。
- **绝对排除项**：原始异常消息文本、Shell 命令、标准输出/错误、Prompt、对话转录（Transcript）、环境变量值、凭据、会话/作业 ID、文件路径与主机地址。
- **有界重试与幂等对账**：每个候选事件最多经历 3 次持久化处理周期。当 `POST` 结果不确定时，后续周期仅执行幂等对账，绝不盲目重复建单。已存在的 Open 或 Closed 同指纹 Issue 仅作关联，不自动评论或重新打开。
- **队列容量与即时撤权**：活动队列上限为 1,000 条候选（保留 7 天），终态回执上限为 1,000 条（保留 30 天）。关闭上报会立即吊销此前授权下的待发送队列；重新开启不会回填关闭期间的事件，也不会复活已撤权的旧队列。

---

## 3. 授权状态与策略存储分离

故障观测将“上报授权状态”与“策略文件存储可用性”严格区分：

- `fallback.read_policy()` 仅在策略文件不存在时返回 `None`；若策略文件损坏、权限不安全或不可读，则显式抛出 `PolicyUnavailable`。
- `consent_status()` 明确区分 `allowed`、`withdrawn` 与 `unavailable` 三种状态：
  - 当策略处于 `unavailable` 时，阻止对外发送但保留待处理证据与日志游标；
  - 仅当明确验证到用户关闭或变更授权（`withdrawn`）时，才撤回队列中的待发项。
- Outbox 数据库初始化由所有权标记串行保护；若发现已初始化的 Outbox 丢失或为空库，则作为存储故障显式报错，绝不静默重建历史记录。

---

## 4. 本地日志轮转与离线维护

- **四分段环形轮转**：每个已安装写入器按进程/组件维护 4 个固定的 1 MiB JSONL 分段，仅当所有已注册读取器均消费完某分段 inode 后才复用该分段。空间压力下显式记录丢弃计数与 `logging_failed`，待安全空间恢复后自动继续。
- **无依赖启动兜底**：启动兜底写入器限制为单个 1 MiB 故障文件，写满后显式报错。
- **离线维护（`reporting maintain`）**：
  - 默认保留目标为每个配置根目录 **7 天、128 MiB、512 个文件**。
  - 默认完全离线运行；仅在显式传入 `--update-running` 且当前存在已启用、健康的运行中 Worker 时，才会将其平滑升级至更高语义版本（不会自动拉起已停止或异常的 Worker）。
  - 适配器可通过 `--budget-seconds` 传入剩余时间预算（上限 75 秒）。
