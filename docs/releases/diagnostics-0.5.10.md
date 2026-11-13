# mindie-diagnostics 0.5.10：Windows 长路径与企业网络修复

[English](diagnostics-0.5.10.en.md)

修复 mindie-agent-codex#113：接受合法的 Windows 本地扩展路径，正确编码 SQLite URI，并统一普通路径与扩展路径的读者、日志和游标身份。保留清理默认预算由 1 秒调整为 5 秒，满足 SQLite 3 秒等待与收尾预算，避免空闲读者被持续误报为未知。

Windows 生成的诊断虚拟环境存放在 `%LOCALAPPDATA%/mindie/diagnostics/<profile-hash>/<source-hash>/venv`，减少原生 Python 启动器的 MAX_PATH 限制；配置侧的 `current.json` 与锁仍是唯一切换依据。旧布局保留回滚读取能力。存量队列、游标、授权修订和不确定发布回执不迁移或重建。存活且身份核验通过的降级 worker 可以获取修复，新一代切换仍要求健康读回。

后台网络环境保留调用方通过 Git 配置槽传入的 HTTP 代理、CA 与 TLS 设置，排除凭据助手和认证头；撤销代理后会清除后台遗留设置。DNS、TCP、TLS 阶段失败仍保留真实错误，不解释为账号失效，也不绕过企业网络策略。

测试覆盖实际长路径 SQLite、虚拟环境构建、代际切换与队列保留，并在 CI 中增加 Python 3.12。正式发布流水线继续核验多平台合同测试、精确源码 benchmark、发行资产和旧客户端升级。用户现场的 TCP 超时仍需在其网络中核验；本版本不声称改变现场防火墙或代理策略。
