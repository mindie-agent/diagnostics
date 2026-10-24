# 出站对账读取的有界化跟进（设计稿，本轮未实施）

**中文** · [English](./outbound-reconciliation-followup.en.md) · [返回文档中心](./README.md)

本文只记录设计意图与约束，**本轮没有改动任何生产传输或 API**（`reporter.py` 的请求形状、`Outbox` 的 schema 与状态机均保持原样）。
行号基准：当前分支工作树（基于 `838480a`，`reporter.py` 头部精简 2 行后更新）。

---

## 1. 现状中可复现的矛盾

`GitHub.find_issue`（[`reporter.py:468-497`](../mindie_diagnostics/reporter.py)）用一条「全量列举 + 下载每条 Issue 正文」的读取同时承担两件事：跨会话去重，以及自己那次不确定 POST 的对账。它最多翻 20 页、每页 100 条（`reporter.py:481-482`），而整个发布周期（一次对账 + 一次 POST）共享 8 MiB 字节预算（`reporter.py:457-458`，`self._bytes` 只在 `begin_cycle` 清零）。

同时，本组件自己发布的 Issue 正文上限是 60000 字节（`reporter.py:416-417`）。用本地假 `gh` 脚本（真实子进程 + 真实 `_parse_response`，无网络）喂入 250 条、正文 52,698 字节的 Issue 列举后：第 1 页读入 6,074,551 字节，第 2 页因剩余预算不足直接抛 `github_output_limit`。

这只是**本地 fixture 层面的输入证据**，用来说明预算算术，不构成任何检索质量、NPU 或线上验收结论。

后果（源码可推）：`github_output_limit` 不在 `publish_one` 的 `recoverable` 白名单内且 `permanent=False`（`reporter.py:574-577`），于是被记为可重试并按退避重放，而仓库体量只会变大——对账窗口实际不可能恢复，`pending` 走不到 POST、`uncertain` 也完不成对账，却被呈现为瞬时故障。

## 2. 任何方案都必须保留的既有不变量

1. **不确定的 POST 永不重放**：`begin_post` 的租约栅栏（`outbox.py:428-434`）只接受 `pending`/`retry`，`uncertain` 行只能走只读对账分支（`reporter.py:543-547`）。
2. **“读不到”绝不等于“不存在”**：窗口不完整、字节耗尽、索引延迟都必须保持为可见的失败，不得折叠成「远端没有这条 Issue」。
3. **本地回执失败不反杀已确认的远端结果**（`_record_publication_outcome`，`reporter.py:509-517`）。
4. **公开身份仍是 Issue 正文里的 `<!-- mindie-incident:<sha256> -->`**，不因读取形状改变。

## 3. 为什么“改用 search 接口”不能单独成立

`GET search/issues` 的响应确实与仓库体量无关，但 GitHub 的搜索索引是**最终一致**的：刚创建的 Issue 可能短时间查不到。因此：

- search **只能用于确认存在**（命中即可据此 `reconciled`，这是安全方向）；
- search **未命中不得作为不存在的证明**，尤其不得据此让一条 `pending` 记录去 POST——那正是重复建单的来源。

也就是说 search 可以削减去重读取的字节量，但必须与一条“按时间有界、无索引延迟”的列举读取配合，且在两者都无法给出确定结论时如实上报窗口不完整。

## 4. `first_seen` 不是 POST 时刻

`Outbox.claim` 返回的 `first_seen`（`outbox.py:307-329`）是**该故障第一次被本地看见**的时间，不是发起 POST 的时间：

- 它确实 ≤ 真实 POST 时刻，所以 `[first_seen, now]` 是一个**正确的**下界窗口（不会漏掉自己建的单）；
- 但它的**宽度不受约束**：一个几个月前首次出现、最近才排到队首的故障，窗口会退化成全量扫描；
- 复活分支会把 `first_seen` 重置为当前时间（`outbox.py:196-201`），虽然该分支显式排除了携带 `submission_uncertain*` 的行（所以带未知写证据的行不会被重置），但这说明该字段的语义归属是“本地首见”，不是“外部意图”。

结论：`first_seen` 只能当作**正确性下界**使用，不能宣称它能给出一个小窗口。

## 5. 若要有界窗口，需要的是单一归属的意图元数据

当前 `incidents` 表里没有任何稳定记录“本进程在何时发起过一次 POST”的列：`lease_until` 在任何状态写入时被清零（`outbox.py:409`），`next_attempt` 每轮对账都会被重写，二者都不能重建原始尝试时刻。

需要的最小形态是：**在既有 `incidents` 行上增加一个可空的尝试时刻列**，由 `begin_post` 在**同一个 `BEGIN IMMEDIATE` 事务内**与 `state='uncertain'` 一起写入（单一归属，与不确定状态同生共死），迁移采用与现有 schema 校验一致的追加式方案（`PRAGMA table_info` 判定 + `ALTER TABLE ... ADD COLUMN`，老库读为 `NULL` 时退回 `first_seen` 下界）。

**不要**为此另建一本回执/审计账本或第二个状态机；对账窗口的输入必须来自拥有该状态的同一行。

## 6. 有界读取与诚实的窗口上报

- 每个用途各自声明页数、字节与单条正文上限，并计入同一个 `begin_cycle` 预算；
- 字节/页数耗尽时使用 `github_reconciliation_window_exhausted`（`reporter.py:493` 已有此代码，目前因字节预算先行耗尽而几乎不可达），与传输异常 `github_output_limit` 区分开，让 `reporting status` 能分辨“窗口不足”与“网络异常”；
- 窗口不完整时保持 `uncertain` / `retry`，绝不据此 POST。

## 7. 明确不属于本设计的主张

- **不主张跨 profile / 跨主机的 exactly-once**：marker 稳定只能保证「当某个读取者已经能看到该 marker 时不会重复建单」，并不能排除两个 profile 同时处于 POST 中途的窗口。
- **不动 `publications` / 限流 schema**：`publish_one` 目前确实不传 `hourly_limit`（`reporter.py:550`），该表因此为空；但在没有真实缺陷与迁移验证之前，不在本轮删除表或分支。
- **不合并健康判定模块**：`health.read_health` 与 `integration._worker_view` 的重复判定同理，留待带迁移验证的独立改动。

## 8. 相邻待办：`prune` 与 `cursor_snapshot` 的预算对齐

`maintenance.prune` 的 1.0s 期限与 `reader_registry.cursor_snapshot` 的 3.0s 数据库等待下限不自洽（见两处模块头部已更正的说明）。在没有可证实的真实锁等待路径之前，本轮只更正了文档承诺；真要对齐，应当是**按剩余预算做准入**（剩余时间不足以容纳一次合规的 3.0s 等待时直接跳过并记为 cursors 未知，从而保守地不删除任何分片），而不是把外层期限简单抬高、或把 SQLite 等待压到亚秒级重新引入热路径误杀。
