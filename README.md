<h1 align="center">MindIE Agent Diagnostics (<code>mindie-diagnostics</code>)</h1>

<p align="center">
  <strong>面向 MindIE Agent 组件的有界本地结构化日志、离线诊断包与可选 GitHub Issue 自动上报底座</strong>
</p>

<p align="center">
  <strong>中文</strong> · <a href="./README.en.md">English</a> · <a href="./docs/README.md">文档中心</a> · <a href="https://github.com/mindie-agent/mindie-agent">MindIE Agent 主仓库</a>
</p>

---

## 项目简介

`mindie-diagnostics` 是 [MindIE Agent](https://github.com/mindie-agent/mindie-agent) 的共享可观测性与故障诊断组件（Python 包名 `mindie-diagnostics`，导入名 `mindie_diagnostics`）。它为 `remote-dev`、`coordinator`、`npu-top`、`knowledge` 及各原生客户端适配器提供统一、无模型调用（Model-free）的本地结构化日志轮转、故障归属记录、离线脱敏诊断包导出，以及经用户显式授权的 GitHub Issue 自动上报服务。

- **100% 无模型执行路径**：所有诊断记录、维护、导出与上报路径均不调用任何大语言模型，不抓取对话转录（Transcript），也绝不自动重放业务命令。
- **有界本地日志轮转**：每个进程/组件使用 4 个固定 1 MiB JSONL 分段，仅在所有已注册读取器消费完毕后复用，默认按 7 天 / 128 MiB / 512 个文件上限执行离线维护。
- **严格的隐私白名单**：公开 Issue 与离线诊断包仅投影静态组件名、版本号、阶段、错误类别、单调时钟耗时、退出码与产品代码栈帧；命令原文、输出、Prompt、路径、端点地址与凭据一律排除。
- **显式授权的自动上报**：GitHub Issue 自动上报默认关闭（Opt-in），配置与服务安装完全独立于知识库贡献，关闭后立即吊销待发送队列。

---

## 快速开始

### 1. 环境要求与安装

需要 Python `>= 3.11`。如需启用自动 GitHub Issue 上报，需本地已安装并登录 `gh` CLI。

```bash
uv pip install -e .
```

### 2. 自动上报状态查询与配置（CLI）

各原生客户端适配器均提供内置诊断入口；独立组件也可直接调用 CLI：

```bash
# 查看只读状态（包含本地近期故障引用、授权状态、队列与 Worker 健康度）
python -m mindie_diagnostics.cli reporting status

# 开启自动上报（默认目标仓库为 mindie-agent/mindie-agent，可通过 --repository 指定）
python -m mindie_diagnostics.cli reporting configure --enabled true

# 准备独立运行时并启动后台上报 Worker
python -m mindie_diagnostics.cli reporting ensure

# 关闭自动上报（立即失效未发送的远端操作）
python -m mindie_diagnostics.cli reporting configure --enabled false
```

### 3. 日志维护与离线诊断包导出

```bash
# 执行离线日志保留维护
python -m mindie_diagnostics.cli reporting maintain

# 升级运行中且健康的 Worker 至更高语义版本（不自动拉起已停止的 Worker）
python -m mindie_diagnostics.cli reporting maintain --update-running

# 针对指定 operation_id 导出离线脱敏诊断包（不上传网络）
python -m mindie_diagnostics.cli bundle --root /absolute/log-root --operation-id INCIDENT_ID
```

---

## Python 集成示例

在拥有该故障边界的组件入口记录结构化故障：

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

- 仅传入静态产品元数据，切勿将用户输入作为 `category` 或 `operation`。
- 调用方参数错误、缺配置、权限不足、网络/认证失败或业务命令非零退出不属于产品自身缺陷，应标记为 `reportable=False` 仅保留在本地。
- 常规操作可通过 `configure(...).operation(...)` 记录有界阶段、单调时钟耗时与清理证据。

---

## 环境变量与配置路径

| 配置项 / 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MINDIE_DIAGNOSTICS_CONFIG` | `~/.config/mindie-agent/diagnostics.json`（或 `$XDG_CONFIG_HOME/mindie-agent/diagnostics.json`） | 上报策略配置文件路径（队列、回执、健康度与独立运行时存于同级目录） |
| `MINDIE_DIAGNOSTICS_ROOT` | 平台用户状态目录 | 本地结构化诊断日志根目录 |
| `MINDIE_LOG_LEVEL` | `INFO` | 日志级别（`DEBUG` / `INFO` / `WARNING` / `ERROR`） |

---

## 文档导航

| 文档 | 中文 | English |
| --- | --- | --- |
| 文档中心索引 | [docs/README.md](./docs/README.md) | [docs/README.en.md](./docs/README.en.md) |
| 架构、隐私白名单与日志保留规范 | [docs/architecture-and-privacy.md](./docs/architecture-and-privacy.md) | [docs/architecture-and-privacy.en.md](./docs/architecture-and-privacy.en.md) |

---

## 开发与测试

```bash
python -m pytest
```

---

## 许可证

本项目基于 [MIT License](./LICENSE) 开源。
