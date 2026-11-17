# 参与贡献（Contributing to `mindie-diagnostics`）

**中文** | [English](CONTRIBUTING.en.md) · [返回项目主页](README.md)

欢迎参与 **MindIE Diagnostics**（`mindie-agent/diagnostics`，仓库分类：`state-service`）的开发与改进！本仓库提供零第三方运行时依赖的本地结构化故障记录、白名单隐私投影、Triage Card v2 归因以及按需排空（Scale-to-Zero）的 GitHub Issue 上报能力。

---

## 1. 核心工程准则（`state-service` 四大铁律）

1. **`SVC-CONSENT` — 严格白名单隐私与撤回零补发**：
   公开上报默认关闭；严禁上传原始异常消息（`str(e)`）、命令参数、环境变量、路径或主机地址。关闭上报时立即撤回队列中未发送项（`withdrawn`），重新开启绝不回溯补发；策略文件损坏须抛出 `PolicyUnavailable`，绝不能当成首次运行清空队列。
2. **`SVC-STATE-COMPAT` — 零第三方运行时依赖与状态兼容**：
   运行时包保持 100% Python 标准库实现；保持 `fallback.record_failure`、`service_environment.py`、`reporter.sqlite3` 与 `reporting_commands.ensure()`（`status="running"`、`mode="on_demand"`）前后向兼容。
3. **`SVC-LIFETIME` — Scale-to-Zero 按需排空、服务先退后清与软链友好**：
   禁止常驻空闲守护循环、15 秒心跳 `fsync` 线程与 `KeepAlive` 自拉起循环：Worker 排空后立即以 `mode="on_demand"`、`status="idle"` 退出，仅在仍有待重试项时登记下一次持久化唤醒。服务迁移须覆盖 `<config>.state/service` 与系统默认目录（macOS 等待 `launchctl bootout` 完成，Windows 使用无窗 `pythonw.exe` + `CREATE_NO_WINDOW`，Linux 诚实处理无 systemd 的 WSL 环境），保留用户手动禁用状态，并在确认旧服务卸载后再回收旧虚拟环境。兼容 Stow/Nix/Worktree 软链配置与日志根目录。所有核心模块保持 `< 800` 行及标准四段头部注释。
4. **`SVC-EXPORT` — `7d` 仅限未发送 `pending` 过期与 `uncertain` 对账诚实性**：
   仅对超过 7 天且从未发起投递（`attempts == 0`、`attempt_timestamp IS NULL`）的 `pending` 项标记 `expired`；**绝不**因超期丢弃 `uncertain` POST 回执、`published` 查重指纹、`choice_revision` 或 `withdrawn` 回执。所有 `except OSError:` 前必须显式保留 `except TimeoutError: raise`。

---

## 2. 本地隔离测试与必跑门禁

```bash
# 1. 运行完整隔离测试套件（tests/conftest.py 自动隔离全部宿主目录）
python3 -B -m pytest -q -m "not live_npu"

# 2. 校验零依赖 Wheel 离线安装与外部目录冒烟
uv build --wheel --out-dir /tmp/diag-dist
uv venv /tmp/diag-smoke && uv pip install --python /tmp/diag-smoke/bin/python --no-index --find-links /tmp/diag-dist mindie-diagnostics
(cd /tmp && /tmp/diag-smoke/bin/mindie-diagnostics --help >/dev/null)

# 3. 校验 Unix LF 换行符与 state-service PR 自查清单
python3 .github/scripts/check_lf_eol.py
python3 .github/scripts/check_pr_checklist.py --profile state-service --body-file <pr_body.md>
```

---

## 3. 提交 PR 说明

请按 [`.github/pull_request_template.md`](.github/pull_request_template.md) 勾选 `SVC-CONSENT`、`SVC-STATE-COMPAT`、`SVC-LIFETIME`、`SVC-EXPORT` 并填写具体核验依据（纯文档等不适用项请写 `not-applicable: <具体原因>`）。
