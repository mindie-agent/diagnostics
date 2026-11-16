# mindie-diagnostics 0.5.15

## 更新故障策略分类补齐（防止 DNS 解析与并发锁竞争误报内部故障）

- **DNS 解析故障归入连通性（`resolver`）**：`update_failure_policy()` 将子进程或传输层已标记的 `category="resolver"` 统一映射为 `("connectivity", False, "update_transport_unavailable")`，消除本机断网/DNS 解析失败被兜底误判为 `internal / update_internal_failure` 并自动提交 GitHub Issue 的问题。
- **并发锁竞争与前置条件受阻归入非上报前置条件（`busy` / `deferred` / `prerequisite`）**：`update_failure_policy()` 支持识别异常链上的 `category in {"busy", "deferred", "prerequisite"}` 并映射为 `("prerequisite", False, "update_prerequisite_unavailable")`，确保跨子进程透传的锁忙/延迟更新状态仅保留本地可观测记录而不触发虚假故障上报。
- **版本**：`0.5.15`。
