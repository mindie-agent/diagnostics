# mindie-diagnostics 0.5.11：双键去重、Triage Card v2 与主仓集中归档

[English](diagnostics-0.5.11.en.md)

升级自动故障诊断与上报链路（Diagnostics 2.0），默认归档目标统一至 `mindie-agent/mindie-agent`，消除跨版本重复建单并补齐精确到提交哈希的根因定位信息：

- **双键去重与回归追加（`fingerprint` + `fault_signature`）**：在保留按构建版本 `fingerprint` 本地队列去重的同时，引入跨版本稳定的 `fault_signature`（忽略包版本号、修订号与单帧行号微移）。命中主仓已有 `OPEN` Issue 时自动追加 **Additional Occurrence** 评论，命中 `CLOSED` Issue 时追加 **Regression Observed After Issue Closure** 评论，避免升级版本后重复创建同根因新单；翻页扫描对中途删除条目具备跨周期游标连续性保护。
- **Triage Card v2 与精确源码 Permalink**：上报正文与回归评论内嵌结构化根因定位卡片（`reason_code`、`substage`、`contract_name`、`soc_generation`、`harness`、`diagnostics_version` 及最近执行阶段 `phases`），并在具备完整 40 位提交哈希与自有模块栈帧时自动生成对应组件/插件仓精确提交行号链接（正确区分普通模块与包 `__init__.py`）。
- **64 层因果异常链遍历与原子构建元数据**：共享记录器与 `Operation` 日志统一遍历最多 64 层异常因果链，超限未达根因叶子节点时显式标记 `stack_incomplete=True`，禁止将外层包装帧误标为故障点；显式部分构建元数据或 `infer_build_meta=False` 严格禁止混入父进程包修订号。
- **零自由文本隐私红线与调用溯源令牌**：`MINDIE_DIAGNOSTICS_INVOCATION_TOKEN` 仅用于本地子进程同次调用溯源核验，在诊断包导出与 GitHub 上报时全量剥离且不计入属性省略计数；异常消息、`__notes__`、本地路径、环境变量与凭据一律不进入事件或公开内容。

包内 `__version__` 同步为 `0.5.11`。本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
