# 异常证据采集与校验接口（Exception Evidence API）

**中文** · [English](./exception-evidence.en.md) · [返回文档中心](./README.md)

`mindie_diagnostics.integration.exception_evidence(component, exception, *, source_root=None)`（底层实现位于 `mindie_diagnostics.evidence.exception_evidence`）在不执行任何磁盘写入或网络上报的前提下，从活跃异常中提取有界、可安全 JSON 序列化的诊断证据字典。跨进程调用时，可在子进程仍持有原始异常与调用栈（traceback）时调用该函数，将序列化结果回传给父进程并传入 `record_failure(..., evidence=evidence)`；同进程直接记录时，也可直接调用 `record_failure(..., exception=exc, source_root=owned_source_directory)` 原地采集。

`source_root` 是调用方显式声明的受信任自有组件源码绝对目录（传入相对路径会被直接忽略）。位于该目录下的独立脚本文件名仅按词法规则映射为规范组件命名空间（例如 `release_activation.py` 映射为 `mindie_agent_codex.release_activation`，映射自 `__init__.py` 时额外标记 `init=True`）。该映射不做文件系统属主校验，调用方**严禁**传入用户业务代码目录；`source_root` 路径本身绝不会写入返回证据中，跨盘符或目录外的栈帧亦不做映射。原本就归属于组件规范模块名（以及 `mindie-agent-*` 宿主插件共管的 `mindie_harness.*` 模块）的栈帧会自动保留。

活跃异常采集函数 `exception_evidence()` 从当前异常链中提取：`stack_frames`（`module`、`function`、`line` 及可选 `init`）、基于合法栈帧重新计算的 SHA-256 `stack_fingerprint`（发生截断时附带 `stack_truncated` / `stack_incomplete` 标记）、`exception_chain`（最多 8 个异常类名标识符）、严格从非 `TimeoutError` 的 `OSError` 实例提取的有符号 32 位整型 `errno`，以及异常对象的 `.reason_code` / `.code` 标签。采集过程最多遍历 64 层异常链、每层最多扫描 4,096 个 traceback 节点（滑动保留末尾 256 个节点），最终最多保留 24 个归属栈帧；遇到循环异常链会自动安全终止。异常消息（`str(exc)`）、文件路径、源码正文、局部变量、`stdout` 与 `stderr` 一律不予采集。

`mindie_diagnostics.evidence.validate_evidence(component, evidence)` 在无需持有活跃异常对象的前提下，将调用方或子进程传入的 JSON 字典投影到白名单并联动 `redact.scan_text` 执行敏感词复核：除校验归属 `stack_frames`（丢弃外部传入的 `stack_fingerprint` 并基于合法栈帧重新计算）与 `exception_chain` 外，还校验调用方显式提供的有符号 32 位整型 `errno`、`exit_code` 以及受格式约束的标签字段（`reason_code`、`substage`、`contract_name` 与取值限定在 `{"a2", "a3", "a5"}` 内的 `soc_generation`），未知或格式不合法的字段一律剔除。由于纯 JSON 传输本身无法证明整型 `errno` 源自真实的 `OSError` 实例，子进程生产端应优先通过 `exception_evidence()` 生成异常证据。校验仅保留归属组件（及宿主插件共管的 `mindie_harness`）命名空间下的栈帧，拒绝任意用户代码命名空间。

`record_failure` 保持尽力而为（best-effort）记录语义（内部锁获取采用非阻塞 `blocking=False`，绝不向业务调用方抛异常或拉起后台 Worker，而 JSONL 日志追加本身同步写入磁盘），并完整沿用既有的用户授权检查与 `incident_id` 幂等去重逻辑。附带 `evidence` 绝不会绕过或授予公开上报权限；当父进程同时传入本地包装异常 `exception` 与经校验的子进程 `evidence` 时，子进程传回的已校验证据字段优先覆盖父进程包装异常上的同名字段。
