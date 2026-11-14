# mindie-diagnostics 0.5.12：传输层错误分类收敛、Windows PowerShell 5.1 / CP936 与 WSL2 服务加固

[English](diagnostics-0.5.12.en.md)

收敛自动更新与后台诊断上报的跨平台边界，杜绝瞬时传输抖动误报、Windows 无控制台代码页损坏及 WSL2 跨系统遮蔽与权限误放行：

- **传输层错误分类收敛（`fallback.py`）**：将 `ssl.SSLCertVerificationError` 归类为 `("credentials", False, "update_credentials_required")`；将流中直接抛出的 `ssl.SSLError`/`SSLEOFError`、`socket.gaierror`/`herror` 及 `http.client.IncompleteRead`/`RemoteDisconnected`/`BadStatusLine` 归类为 `("connectivity", False, "update_transport_unavailable")`，杜绝因中途 TLS/连接抖动自动创建虚假 `internal` Issue，同时保持 `InvalidURL`/`ImproperConnectionState`/`NotConnected` 等调用状态缺陷为 `internal`。
- **Windows PowerShell 5.1 无控制台标准流 UTF-8 直连（`platform_service.py`）**：通过 `[Console]::OpenStandardOutput()` / `[Console]::OpenStandardError()` 显式绑定无 BOM UTF-8 `StreamWriter`，消除 `CREATE_NO_WINDOW` 无控制台句柄场景下 `[Console]::OutputEncoding` 抛出 `IOException` 退回 `CP936` 导致的非 ASCII 计划任务路径校验损坏。
- **WSL2 `.exe`/PE/DrvFS 遮蔽防护与 `metadata` 权限分级（`filesystem.py`、`github_transport.py`、`service.py`、`reporting_runtime.py`）**：`which_native` 在 POSIX 下拒绝 Windows PE 可执行文件与 DrvFS 路径并直接将解析出的原生路径传入 `Popen`/`systemctl`；仅对未开启 `metadata` 且单用户匹配的 WSL2 DrvFS 挂载豁免固定模式位，对开启 `metadata` 的 DrvFS 与原生 POSIX 文件系统严格执行 `0700`/`0600` fail-closed 校验。
- **非 `systemd` Linux / WSL2 本地受管后台 Worker 与事务回滚完整性（`platform_service.py`、`service.py`）**：无 `systemctl --user` 会话总线时自动回退至本地后台 Worker，强制校验非空进程身份（UID + 启动时刻）与注册失败子进程回收，杜绝 PID 复用误杀与孤儿进程残留；单元与清单发布在 `TimeoutError` 下同样保证先完成原子回滚再重抛超时异常。

包内 `__version__` 同步为 `0.5.12`。本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
