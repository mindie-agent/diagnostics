# mindie-diagnostics 0.5.8：Windows 计划任务真实 GUI 入口校验

[English](diagnostics-0.5.8.en.md)

Windows 计划任务原先只检查 `pythonw.exe` 文件存在。当该文件实际是控制台型启动器时，任务仍会弹出控制台窗口。现在安装服务及启动已有注册任务前，都读取 PE32/PE32+ 头并要求 GUI 子系统（`Subsystem == 2`）。缺失、不可读、损坏或控制台型入口明确报 `windowless_python_required`，不会注册或启动新任务。

校验仅作定长头部读取，不缓存、不改写解释器。修复运行环境后可正常重试；服务配置、授权、原虚拟环境及数据保留行为保持现有协议。

验证：Windows 本地服务安装、重启及网络上下文范围 71 passed、6 skipped（平台限定用例）。新增回归覆盖新安装拒绝、已注册入口被替换后拒绝启动、修复后恢复。正式发布流水线另行运行完整组件测试及精确标签源码 benchmark，结果随 Release 资产交付。

关联：mindie-agent/knowledge#115 的同类入口审查。
