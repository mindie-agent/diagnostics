# mindie-diagnostics 0.5.6

本次更新聚焦 Windows 后台诊断上报与定时维护子进程的零控制台弹窗（Zero Console Popup）加固：

- **Windows 零控制台弹窗加固（`reporter.py` / `platform_service.py` / `reporting_commands.py` / `reporting_runtime.py`）**：
  - 为后台 `gh` Issue 上报、Windows 计划任务（`schtasks`）注册/查询以及诊断运行时子进程调用统一显式附加 `subprocess.CREATE_NO_WINDOW`，彻底消除从 GUI 宿主或无控制台父进程触发诊断维护时闪现黑色控制台窗口的问题。

包内 `__version__` 同步为 `0.5.6`。
