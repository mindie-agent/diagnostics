# `mindie-diagnostics` 文档中心

**中文** · [English](./README.en.md) · [返回项目主页](../README.md)

这里详细说明了 `mindie-diagnostics` 是如何保护你的隐私安全、区分“用户报错”与“产品 Bug”、以及管理本地循环日志和授权队列的。

## 文档导航

| 文档主题 | 中文 | English | 你能在这里查到什么 |
| --- | --- | --- | --- |
| 架构、隐私白名单与日志轮转详解 | [architecture-and-privacy.md](./architecture-and-privacy.md) | [architecture-and-privacy.en.md](./architecture-and-privacy.en.md) | 哪些信息允许进诊断包、哪些绝对禁止？为什么策略文件坏了不能当成首次使用？4 × 1 MiB 循环日志怎么防丢数据？ |

## 设计跟进（尚未实施）

| 文档主题 | 中文 | English | 你能在这里查到什么 |
| --- | --- | --- | --- |
| 出站对账读取的有界化跟进 | [outbound-reconciliation-followup.md](./outbound-reconciliation-followup.md) | [outbound-reconciliation-followup.en.md](./outbound-reconciliation-followup.en.md) | 为什么 8 MiB 周期预算与 20 页对账窗口不自洽？为什么只用搜索接口不能证明「不存在」？有界窗口需要哪一份单一归属的意图元数据？ |
