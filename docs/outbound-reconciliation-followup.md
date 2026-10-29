# 出站对账读取的有界化跟进

**中文** · [English](./outbound-reconciliation-followup.en.md) · [返回文档中心](./README.md)

**实施状态（2026-10-29）**：已实现尝试时间戳迁移、阳性搜索与有界列举、窗口耗尽分类，以及 prune 剩余预算准入。第 1–7 节保留实施前的设计依据；其中行号基于 `838480a`，不代表当前实现位置。当前行为及本地验证范围见第 8–9 节。

---

## 1. 实施前可复现的矛盾

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

## 8. 已实施：`prune` 与 `cursor_snapshot` 的预算对齐

`maintenance.prune(timeout=1.0)` 保留默认期限，并预留 50ms 给释放锁与写入 retention marker。它将实际剩余时间传给 `cursor_snapshot(..., remaining_budget=...)`；每个 reader 在连接 SQLite 前必须还有 `3.0s + 50ms`，否则返回 `None`。schema 检查与 cursor 查询处于同一个只读事务，避免两次独立的数据库锁等待。

默认预算下，有注册 reader 的 prune 会返回 `reader_unknown=True`、`limited=True`，不删除分片；没有 reader 时不需要 SQLite 等待。调用方可以明确传入更充足的 `timeout`。`reporting_commands.maintain` 将自身剩余预算减去 settlement reserve 后传入，因此有预算的显式维护仍可完成回收。没有降低 `busy_timeout=3000`，也没有提高 prune 的默认期限。

既有日志回收调用只传 `timeout`，仍保留原先的协作式执行期限语义；本次不宣称关闭该热路径的所有延迟边界。

## 9. 已实施：出站窗口与验证范围

- `outbox.py`（本仓库没有 `storage.py`）在 `incidents` 追加可空 `attempt_timestamp REAL`。`begin_post` 在同一个 `BEGIN IMMEDIATE` 事务中写入尝试时间和 `uncertain`。历史 `uncertain` 行第一次被 claim 时在租约事务内回填 `first_seen`，后续对账不移动下界。
- 公开标记保持 `<!-- mindie-incident:<sha256> -->`。搜索请求限定 `repo:`、`is:issue`、`in:body` 与指纹，一页最多 5 条，只以完整标记且非 PR 的结果确认命中；搜索未命中或 `incomplete_results` 不代表不存在。
- 列举按 `created` 倒序，使用 `state=all`，每页 30 条、最多 5 页。下界是尝试时间（尚未尝试则为 `first_seen`）减去 300 秒时钟偏差，向下取整到秒；`since` 再前移一秒以包含恰好等于下界的记录。`since` 过滤的是更新时间，因此页面尾部创建时间早于下界时停止；短页也代表窗口完整。
- 搜索、列举和 POST 共用一次 `begin_cycle` 的默认 8 MiB 响应预算，单条读取正文上限 256 KiB。可通过 `max_response_bytes` 设置更小预算。对账读取字节、正文或页数耗尽均返回 `github_reconciliation_window_exhausted`，至少退避 3600 秒，保持 `retry`／`uncertain`，不授权 POST。非对账传输保留 `github_output_limit`。
- 验证使用纯内存标记匹配、真实 SQLite 迁移／回滚／租约，以及本地 HTTP 服务和真实 gh 替身子进程，保留生产字节计数器。覆盖搜索命中、延迟索引、边界时间、共享字节预算、页数耗尽、未知 POST 不重发和 prune 准入。测试入口为 `tests/test_reporter.py`、`tests/test_outbox.py`、`tests/test_reconciliation_http.py`、`tests/test_reader_registry.py` 与 `tests/test_maintenance.py`。

这些证据属于本地 Tier 0／Tier 1，不构成线上 GitHub 检索质量、NPU 或跨主机 exactly-once 验收。
