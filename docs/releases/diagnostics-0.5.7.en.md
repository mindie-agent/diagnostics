# mindie-diagnostics 0.5.7

This release hardens Windows background diagnostic reporting and scheduled maintenance subprocess execution against console window popups:

- **Windows Zero Console Popup Hardening (`reporter.py` / `platform_service.py` / `reporting_commands.py` / `reporting_runtime.py`)**:
  - Explicitly passes `subprocess.CREATE_NO_WINDOW` to background `gh` Issue reporting, Windows Task Scheduler (`schtasks`) operations, and diagnostic runtime subprocess invocations so GUI hosts never flash transient console windows.

The package `__version__` is synchronized to `0.5.7`.
