## 变更摘要 (Summary)

- 

## 设计原则 Review (Pre-PR Checklist — `state-service`)

<!-- 请核对以下四项 state-service 契约，并在下方提供逐项依据（不适用项请写 `not-applicable: 具体原因`，勿填裸 `N/A` 或 `TODO`）。全仓文件须保持 Unix LF 换行。 -->

- [ ] **SVC-CONSENT** — 白名单隐私投影与同意书撤回零补发完备；`PolicyUnavailable` 严格区分损坏与撤回；无未授权网络或后台启动副作用。
- [ ] **SVC-STATE-COMPAT** — 运行时保持 100% 标准库零外部依赖；`fallback.record_failure`、`reporter.sqlite3`、`service_environment.py` 与 `reporting_commands.ensure()`（`mode="on_demand"`）保持回滚兼容。
- [ ] **SVC-LIFETIME** — Scale-to-Zero 按需排空（零常驻空闲循环、零 15 秒心跳线程、零 `KeepAlive` 自拉起）；旧服务先完成卸载（含 `launchctl bootout` 等待与保留手动禁用状态）再回收虚拟环境；兼容合法软链；核心模块 `< 800` 行且 Windows 子进程携带 `CREATE_NO_WINDOW`。
- [ ] **SVC-EXPORT** — `7d` 过期严格限于从未投递的 `pending` 项；永久保护 `uncertain` POST 对账、`published` 查重指纹与 `withdrawn` 回执；`except OSError:` 前置 `except TimeoutError: raise`。

## 自查证据 (Evidence)

- **SVC-CONSENT**: 
- **SVC-STATE-COMPAT**: 
- **SVC-LIFETIME**: 
- **SVC-EXPORT**: 

## 验证结果 (Verification)

- 
