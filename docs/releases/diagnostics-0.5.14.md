# mindie-diagnostics 0.5.14

## 存储生命周期与 Windows/macOS/Linux 保留修复

- **Windows 写入者存活判定（`D-3`）**：`_writer_exited` 改用 `kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess`，正确区分已退出 PID（`ERROR_INVALID_PARAMETER=87`）、受保护/其他用户的存活进程（`ERROR_ACCESS_DENIED=5`）与 `STILL_ACTIVE=259`，不再调用 `tasklist`/`powershell`。Windows 上 `.jsonl` 段清理不再永久为 0。
- **保留扫描收敛**：单次扫描改为最多 `16384` 个 `DirEntry` 元数据条目（组件目录上限 `4096`），避免旧的 `4096` 截断在 `scandir` 返回新条目时饿死最旧段；最后一段被删除后同时回收空的 `<root>/events/<component>` 目录。
- **Worker 自日志清理（`D-2`）**：`maintain()` 在策略根之后额外对 `<state>/diagnostics` 执行仅本地 `prune(state_diag, queue=None, ...)`，清理 60s 周期写入的 `worker.cycle` 段，且绝不把自日志递归摄取进 GitHub Outbox。
- **关闭上报后的轮转保留**：日志段轮转时（同一写入者最多每 300s 一次）机会式执行 `prune(root, timeout=0.5)`，只删除已被读者消费的段；任何清理失败都不影响业务日志写入。
- **原子运行时代际退役（`D-1`）**：`current.json` 记录至多一个 `previous_source_hash`，形成 `current + previous + 活动 worker` 的回滚闭包。`retire_other_generations` 先原子重命名为 `.trash-*` 再受限删除；Windows 上打开中的 `pythonw.exe`/`.dll` 使重命名失败时保留原代际 100% 完整并记为 deferred；超过 30 分钟的 `.staging-*` 与本代际外的历史代际会收敛清理。`maintain()` 还会在运行时事务空闲时机会式清理，使已安装用户在下一次 60s 维护周期自动收敛。
- **Windows profile 标记**：新建 `%LOCALAPPDATA%\mindie\diagnostics\<profile>` 时写入 `profile.json`；`prune_orphan_windows_profiles` 仅清理记录了已不存在策略根且可原子重命名的标记目录。
- **服务暂存目录（`D-4`）**：systemd 单元发布失败时使用 `shutil.rmtree(..., ignore_errors=True)` 清理临时目录，不再因残留文件触发 `rmdir` 失败。
- **版本**：`0.5.14`。
