# mindie-diagnostics 0.5.12：传输层错误分类收敛、Windows PowerShell 5.1 / CP936 与 WSL2 服务加固

[English](diagnostics-0.5.12.en.md)

收敛自动更新与后台诊断上报的跨平台边界，杜绝瞬时传输抖动误报、Windows 无控制台代码页损坏及 WSL2 跨系统遮蔽与权限误放行：

- **传输层错误分类收敛（`fallback.py`）**：将 `ssl.SSLCertVerificationError` 归类为 `("credentials", False, "update_credentials_required")`；将流中直接抛出的 `ssl.SSLError`/`SSLEOFError`、`socket.gaierror`/`herror` 及 `http.client.IncompleteRead`/`RemoteDisconnected`/`BadStatusLine` 归类为 `("connectivity", False, "update_transport_unavailable")`，杜绝因中途 TLS/连接抖动自动创建虚假 `internal` Issue，同时保持 `InvalidURL`/`ImproperConnectionState`/`NotConnected` 等调用状态缺陷为 `internal`。
- **Windows PowerShell 5.1 无控制台标准流 UTF-8 直连（`platform_service.py`）**：通过 `[Console]::OpenStandardOutput()` / `[Console]::OpenStandardError()` 显式绑定无 BOM UTF-8 `StreamWriter`，消除 `CREATE_NO_WINDOW` 无控制台句柄场景下 `[Console]::OutputEncoding` 抛出 `IOException` 退回 `CP936` 导致的非 ASCII 计划任务路径校验损坏。
- **WSL2 `.exe`/PE/DrvFS 遮蔽防护与私有权限严格闭合（`filesystem.py`、`github_transport.py`、`service.py`、`reporting_runtime.py`）**：`which_native` 在 POSIX 下通过严格符号链接解析（`realpath(..., strict=True)`）拒绝 Windows PE 可执行文件与 DrvFS 路径并直接将解析出的原生路径传入 `Popen`/`systemctl`，且全程透传 `os.stat`/`os.lstat` `TimeoutError`；后台 Worker 代码运行目录与凭据环境文件对所有 POSIX 与 DrvFS 挂载严格执行 `0700`/`0600` fail-closed 校验，拒绝未启用 `metadata` 的 `0777`/`0666` DrvFS 共享目录。
- **非 `systemd` Linux / WSL2 本地受管后台 Worker 与排空预算保证（`platform_service.py`、`service.py`）**：无 `systemctl --user` 会话总线时自动回退至本地后台 Worker，强制校验非空进程身份（`boot_id` + UID + 启动时刻）、停止前预留完整排空预算（`>= 5.5s`，确保 `gh` 子进程组优雅回收绝不被截断）与注册失败子进程回收；单元与清单发布在 `TimeoutError` 下同样保证先完成原子回滚再重抛超时异常。

包内 `__version__` 同步为 `0.5.12`。本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
