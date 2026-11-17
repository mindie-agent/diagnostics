# `mindie-diagnostics` 文档中心

**中文** · [English](./README.en.md) · [返回项目主页](../README.md)

这里详细说明了 `mindie-diagnostics` 如何保护隐私安全、区分“用户日常报错”与“组件自身缺陷”、通过 Scale-to-Zero 按需排空消灭常驻后台开销，以及安全管理本地循环日志和授权队列。

## 文档导航

| 文档主题 | 中文 | English | 你能在这里查到什么 |
| --- | --- | --- | --- |
| 架构、Scale-to-Zero 排空、隐私白名单与日志轮转 | [architecture-and-privacy.md](./architecture-and-privacy.md) | [architecture-and-privacy.en.md](./architecture-and-privacy.en.md) | 白名单投影、`mindie-harness` 共管栈归因、Scale-to-Zero 按需排空（零常驻空闲进程与零 15 秒心跳 `fsync`）、`7d` 仅限未发送 `pending` 过期（保留 `uncertain` 对账与已发布指纹）以及 4 × 1 MiB 环形日志轮转 |
| 历史方案归档索引 | [archive/README.md](./archive/README.md) | [archive/README.md](./archive/README.md) | 早期出站对账后续设计讨论归档（仅供历史溯源，不作为当前门禁） |
