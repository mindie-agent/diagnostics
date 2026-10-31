# `mindie-diagnostics` 文档中心

**中文** · [English](./README.en.md) · [返回项目主页](../README.md)

这里详细说明了 `mindie-diagnostics` 是如何保护你的隐私安全、区分“用户报错”与“产品 Bug”、以及管理本地循环日志和授权队列的。

## 文档导航

| 文档主题 | 中文 | English | 你能在这里查到什么 |
| --- | --- | --- | --- |
| 架构、隐私白名单与日志轮转详解 | [architecture-and-privacy.md](./architecture-and-privacy.md) | [architecture-and-privacy.en.md](./architecture-and-privacy.en.md) | 哪些信息允许进诊断包、哪些绝对禁止？为什么策略文件坏了不能当成首次使用？4 × 1 MiB 循环日志怎么防丢数据？ |
| 出站对账窗口有界化与日志回收预算对齐 | [outbound-reconciliation-followup.md](./outbound-reconciliation-followup.md) | [outbound-reconciliation-followup.en.md](./outbound-reconciliation-followup.en.md) | 为什么不确定状态的 POST 绝不能重放？如何通过 `attempt_timestamp` + 阳性搜索 + 有界列举完成 8 MiB 预算内的安全对账，并与 `prune` 游标读取预算对齐？ |
