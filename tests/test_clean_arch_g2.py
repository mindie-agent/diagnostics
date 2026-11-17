"""[Role] Orthogonal behavioral regression tests for Group G2 (`diagnostics-clean-arch`).

[Invariants]
- Exercises real Scale-to-Zero worker subprocesses, real filesystem symlinks,
  real SQLite Outbox state machines, and real Triage Card v2 Markdown projection.
- Isolates all user/host environment roots; zero host pollution.

[Timeouts & Locks]
- Every test completes well under Tier 0/1 bounds (`< 5s`–`< 15s`).

[Side-Effect Policy]
- Writes strictly within pytest `tmp_path`.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from mindie_diagnostics import configure, fallback, integration, triage_card
from mindie_diagnostics.bundle import collect_bundle
from mindie_diagnostics.evidence import validate_evidence
from mindie_diagnostics.health import Health, read_health
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.maintenance import prune
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.wake_schedule import read_wake_receipt, sync_retry_wake


def test_mindie_harness_attribution_and_coowned_host_plugin_frames() -> None:
    """`triage_card` attributes `mindie-harness` and retains `mindie_harness.*` frames on host plugins."""
    sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    repo, tree_url = triage_card.component_commit_url("mindie-harness", sha)
    assert repo == "mindie-agent/harness"
    assert tree_url == f"https://github.com/mindie-agent/harness/tree/{sha}"

    rel, blob_url = triage_card.component_source_permalink(
        "mindie-harness", sha, "mindie_harness.mcp_gateway", 88
    )
    assert rel == "mindie_harness/mcp_gateway.py"
    assert blob_url == f"https://github.com/mindie-agent/harness/blob/{sha}/mindie_harness/mcp_gateway.py#L88"

    # Host plugin incident containing both plugin entry frame and co-owned mindie_harness frame.
    raw_evidence = {
        "stack_frames": [
            {"module": "mindie_agent_codex.mcp_entry", "function": "main", "line": 21},
            {"module": "urllib.request", "function": "urlopen", "line": 215},
            {"module": "mindie_harness.mcp_gateway", "function": "dispatch_call", "line": 142},
        ]
    }
    validated = validate_evidence("mindie-agent-codex", raw_evidence)
    kept_mods = [f["module"] for f in validated["stack_frames"]]
    assert kept_mods == ["mindie_agent_codex.mcp_entry", "mindie_harness.mcp_gateway"]

    payload = {
        "schema": 1,
        "events": [
            {
                "schema": 1,
                "component": "mindie-agent-codex",
                "operation": "mcp.dispatch",
                "severity": "ERROR",
                "event": "operation.end",
                "status": "error",
                "package_version": "0.2.0",
                "package_revision": sha,
                "attributes": {
                    "stage": "rpc",
                    "category": "internal",
                    "error_type": "RuntimeError",
                    "stack_frames": validated["stack_frames"],
                },
            }
        ],
    }
    fp = triage_card.build_fingerprint(payload)
    sig = triage_card.fault_signature(payload)
    title, body = triage_card.render_triage_card(payload, incident_fp=fp, fault_sig=sig, occurrences=1)
    assert "mindie-agent-codex" in title
    assert "`plugins/mindie-agent/scripts/mcp_entry.py#L21`" in body
    assert "`mindie_harness/mcp_gateway.py#L142`" in body
    assert "**(fault site)**" in body


def test_scale_to_zero_worker_exits_immediately_with_zero_heartbeat_threads(tmp_path: Path) -> None:
    """Scale-to-Zero worker spawns zero heartbeat threads, exits immediately when idle, and stays healthy hours later."""
    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "reporting.state"
    cfg_path = tmp_path / "reporting.json"
    assert integration.configure_reporting(
        True, repository="mindie-agent/diagnostics", config=cfg_path, roots=[str(log_dir)]
    )["status"] == "configured"

    # 1. Verify Health spawns zero background heartbeat threads and missing idle heartbeats are never stale.
    now = [1000.0]
    rec = configure("scale-zero-test", root=tmp_path / "diag-internal")
    before_threads = {t.name for t in threading.enumerate()}
    with Health(state_dir, rec, clock=lambda: now[0]) as h:
        during_threads = {t.name for t in threading.enumerate()}
        assert "mindie-worker-heartbeat" not in during_threads
        assert during_threads == before_threads
        h.finish_drain(status="idle", next_wake_at=None)
    # 2 hours later, idle Scale-to-Zero worker is healthy and NOT stale!
    now[0] += 7200.0
    health_doc = read_health(state_dir, clock=lambda: now[0])
    assert health_doc["mode"] == "on_demand"
    assert health_doc["status"] == "idle"
    assert health_doc["healthy"] is True
    assert health_doc["stale"] is False

    # 2. Run actual `mindie-diagnostics worker` subprocess WITHOUT `--once`: must exit 0 immediately!
    fake_gh = tmp_path / ("gh.cmd" if os.name == "nt" else "gh")
    fake_gh.write_text("@echo off\r\nexit /b 0\r\n" if os.name == "nt" else "#!/bin/sh\nexit 0\n", encoding="utf-8")
    if os.name != "nt":
        fake_gh.chmod(0o700)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mindie_diagnostics.cli",
            "worker",
            "--reporting-config",
            str(cfg_path),
            "--state",
            str(state_dir),
            "--repository",
            "mindie-agent/diagnostics",
            "--gh",
            str(fake_gh),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["status"] == "ok"
    assert out["mode"] == "on_demand"
    assert out["next_wake_at"] is None
    assert read_wake_receipt(state_dir) is None

    # Subprocess PID has already exited; _worker_view must report healthy=True, running=False, mode='on_demand'!
    view = integration._worker_view(cfg_path)
    assert view["mode"] == "on_demand"
    assert view["status"] == "idle"
    assert view["healthy"] is True
    assert view["running"] is False

    # 3. Without an installed OS service unit, sync_retry_wake persists receipt honestly with scheduled=False.
    uninstalled = sync_retry_wake(state_dir, cfg_path, next_wake_at=2060.0, now=2000.0)
    assert uninstalled["scheduled"] is False
    assert uninstalled["receipt_persisted"] is True
    receipt = read_wake_receipt(state_dir)
    assert receipt is not None
    assert receipt["next_wake_at"] == 2060.0
    assert receipt["wake_not_before"] == 2060.0
    assert receipt["delay_seconds"] == 60
    cleared = sync_retry_wake(state_dir, cfg_path, next_wake_at=None, now=2065.0)
    assert cleared["scheduled"] is False
    assert read_wake_receipt(state_dir) is None


def test_linux_systemd_linked_profile_rearm_and_wsl_local_retry_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linked-profile systemd timers link via `enable <full_path>` + `restart` (`RemainAfterElapse=no`), and WSL local wakes consume `next-wake.json`."""
    from mindie_diagnostics import platform_service as ps, service as svc, wake_schedule as ws

    state_dir = tmp_path / "linked.state"
    unit_dir = state_dir / "service"
    unit_dir.mkdir(parents=True)
    cfg_path = tmp_path / "linked.json"
    integration.configure_reporting(True, repository="mindie-agent/diagnostics", config=cfg_path, roots=[str(tmp_path / "logs")])

    # Case A: Linux with systemd --user on a linked profile outside ~/.config/systemd/user
    monkeypatch.setattr(ws.sys, "platform", "linux")
    monkeypatch.setattr(svc.sys, "platform", "linux")
    unit_path = svc._unit_path(unit_dir)
    unit_path.write_text(
        svc.MARKER + svc.METADATA + json.dumps({"schema": 1, "since": "1970-01-01T00:00:00Z"}) + "\n[Service]\nExecStart=/bin/true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(svc, "_systemd_user_available", lambda _r: True)
    seen_cmds: list[list[str]] = []

    def fake_systemd_runner(argv, **kwargs):
        seen_cmds.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    res = ws.sync_retry_wake(state_dir, cfg_path, next_wake_at=1300.0, now=1000.0, unit_dir=unit_dir, runner=fake_systemd_runner)
    timer_path = unit_path.with_name(unit_path.stem + "-retry.timer")
    assert res["scheduled"] is True
    assert timer_path.is_file()
    timer_body = timer_path.read_text(encoding="utf-8")
    assert "RemainAfterElapse=no" in timer_body
    assert ["systemctl", "--user", "enable", str(timer_path)] in seen_cmds
    assert ["systemctl", "--user", "restart", timer_path.name] in seen_cmds

    ws.clear_retry_wake(state_dir, cfg_path, unit_dir=unit_dir, runner=fake_systemd_runner)
    assert ["systemctl", "--user", "disable", "--now", timer_path.name] in seen_cmds
    assert not timer_path.exists()

    # Case B: Non-systemd Linux / WSL2 local manifest consumes `next-wake.json` via `run_delayed_wake`.
    unit_path.unlink()
    monkeypatch.setattr(svc, "_systemd_user_available", lambda _r: False)
    manifest, label, local_unit = ps._paths(unit_dir)
    local_payload = b'{"label":"local"}\n'
    local_cfg = {
        "marker": ps.MARKER,
        "platform": "linux",
        "since": "1970-01-01T00:00:00Z",
        "label": label,
        "launcher": sys.executable,
        "launch_args": ["-I", "-m", "mindie_diagnostics.platform_service", "run", "--config", str(manifest)],
        "argv": ["worker", "--reporting-config", str(cfg_path), "--state", str(state_dir), "--repository", "mindie-agent/diagnostics"],
        "environment_file": None,
        "state": str(state_dir),
        "unit_sha256": __import__("hashlib").sha256(local_payload).hexdigest(),
    }
    local_unit.write_bytes(local_payload)
    manifest.write_text(json.dumps(local_cfg), encoding="utf-8")

    invoked: list[str] = []
    monkeypatch.setattr(ps, "run_worker", lambda c: invoked.append(str(c)) or 0)
    now_epoch = __import__("time").time()
    (state_dir / "next-wake.json").write_text(
        json.dumps({"schema": 1, "next_wake_at": now_epoch - 0.1, "wake_not_before": now_epoch - 0.1, "delay_seconds": 5}),
        encoding="utf-8",
    )
    assert ws.run_delayed_wake(manifest, state_dir) == 0
    assert invoked == [str(manifest)]

    # Case C: Self-completing macOS `<label>.retry` spawns detached bootout so StartInterval never leaks in launchd.
    monkeypatch.setattr(ws.sys, "platform", "darwin")
    monkeypatch.setattr(ps.sys, "platform", "darwin")
    monkeypatch.setattr(ps.os, "getuid", lambda: 501, raising=False)
    dar_manifest, dar_label, _dar_unit = ps._paths(unit_dir)
    retry_label = f"{dar_label}.retry"
    retry_plist = dar_manifest.parent / f"{retry_label}.plist"
    retry_plist.write_bytes(b"plist")
    detached_bootouts: list[tuple[str, str]] = []
    monkeypatch.setattr(ws, "_spawn_detached_bootout", lambda exe, dom: detached_bootouts.append((exe, dom)))
    monkeypatch.setenv("XPC_SERVICE_NAME", retry_label)
    ws.clear_retry_wake(state_dir, cfg_path, unit_dir=unit_dir)
    assert not retry_plist.exists()
    assert len(detached_bootouts) == 1
    assert detached_bootouts[0][1] == f"gui/501/{retry_label}"


def test_ensure_accepts_post_drain_idle_native_status_and_preserves_disabled_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`reporting_commands.ensure()` returns `running` + `mode='on_demand'` when worker already exited to `idle`, and respects disabled services."""
    from mindie_diagnostics import reporting_commands as commands, reporting_runtime, service

    cfg = tmp_path / "diagnostics.json"
    logs = tmp_path / "logs"
    integration.configure_reporting(True, config=cfg, repository="mindie-agent/diagnostics", roots=[str(logs)])
    src_hash = "c" * 64
    monkeypatch.setattr(commands, "which_native", lambda *a, **kw: sys.executable)
    monkeypatch.setattr(reporting_runtime, "_prepare_locked", lambda *a, **kw: {"python": sys.executable, "source_hash": src_hash})
    monkeypatch.setattr(
        commands,
        "_worker_view",
        lambda _: {"status": "idle", "mode": "on_demand", "healthy": True, "running": False, "runtime_source": src_hash},
    )
    monkeypatch.setattr(service, "ensure_reporter_service", lambda *a, **kw: {"status": "installed", "start_requested": True})
    monkeypatch.setattr(service, "service_status", lambda **_: {"status": "idle"})
    monkeypatch.setattr(commands, "_retirement_decision", lambda *a, **kw: {"allowed": True, "keep": ()})

    ensured = commands.ensure(cfg, unit_dir=tmp_path / "svc")
    assert ensured["status"] == "running"
    assert ensured["mode"] == "on_demand"

    # When user explicitly disabled the OS service, ensure preserves that choice instead of force-starting.
    monkeypatch.setattr(service, "service_status", lambda **_: {"status": "disabled"})
    disabled_res = commands.ensure(cfg, unit_dir=tmp_path / "svc")
    assert disabled_res["status"] == "disabled_by_user"
    assert disabled_res["mode"] == "on_demand"


def test_macos_remove_service_awaits_bootout_before_unlinking_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`platform_service.remove_service` awaits `launchctl print` exit 113 after `launchctl bootout`."""
    from mindie_diagnostics import platform_service as ps

    monkeypatch.setattr(ps.sys, "platform", "darwin")
    monkeypatch.setattr(ps.os, "getuid", lambda: 501, raising=False)
    unit_dir = tmp_path / "LaunchAgents"
    manifest, label, unit = ps._paths(unit_dir)
    unit_dir.mkdir(parents=True)
    plist_bytes = plistlib.dumps({"Label": label, "ProgramArguments": ["/venv/bin/python", "run"]})
    cfg = {
        "marker": ps.MARKER,
        "platform": "darwin",
        "since": "1970-01-01T00:00:00Z",
        "label": label,
        "launcher": "/venv/bin/python",
        "launch_args": ["-I", "-m", "mindie_diagnostics.platform_service", "run", "--config", str(manifest)],
        "argv": ["worker", "--reporting-config", str(tmp_path / "reporting.json"), "--state", str(tmp_path / "state"), "--repository", "mindie-agent/diagnostics"],
        "environment_file": None,
        "state": str(tmp_path / "state"),
        "unit_sha256": __import__("hashlib").sha256(plist_bytes).hexdigest(),
    }
    unit.write_bytes(plist_bytes)
    manifest.write_text(json.dumps(cfg), encoding="utf-8")

    calls: list[tuple[str, ...]] = []
    booted_out = [False]
    print_after_bootout = [0]

    def fake_runner(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[:2] == ["launchctl", "bootout"]:
            booted_out[0] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[:2] == ["launchctl", "print"]:
            if not booted_out[0]:
                return SimpleNamespace(returncode=0, stdout=f"state = running\n{manifest}\n/venv/bin/python\n", stderr="")
            print_after_bootout[0] += 1
            if print_after_bootout[0] == 1:
                # Still unloading on first poll; unit file MUST still exist here!
                assert unit.exists()
                return SimpleNamespace(returncode=0, stdout="state = exiting\n", stderr="")
            return SimpleNamespace(returncode=113, stdout="", stderr="Could not find service")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    res = ps.remove_service(unit_dir=unit_dir, runner=fake_runner)
    assert res["status"] == "removed"
    assert print_after_bootout[0] >= 2
    assert not unit.exists() and not manifest.exists()


def test_symlinked_config_consent_and_log_root_work_cleanly(tmp_path: Path) -> None:
    """Stow/Nix/worktree symlinks on `diagnostics.json`, `mindie-consent.json`, and `--root` work end to end."""
    dotfiles = tmp_path / "dotfiles-repo"
    real_logs = tmp_path / "real-logs"
    dotfiles.mkdir()
    real_logs.mkdir()

    real_consent = dotfiles / "mindie-consent.json"
    real_consent.write_text(
        json.dumps(
            {
                "schema": "mindie-consent/1",
                "choice": "contribute",
                "choice_revision": "1" * 32,
            }
        ),
        encoding="utf-8",
    )
    real_cfg = dotfiles / "diagnostics.json"
    real_cfg.write_text("{}", encoding="utf-8")

    sym_dir = tmp_path / "symlinked-home"
    sym_dir.mkdir()
    sym_consent = sym_dir / "mindie-consent.json"
    sym_cfg = sym_dir / "diagnostics.json"
    sym_logs = sym_dir / "diag-logs"
    sym_consent.symlink_to(real_consent)
    sym_cfg.symlink_to(real_cfg)
    sym_logs.symlink_to(real_logs, target_is_directory=True)

    # Temporarily initialize valid policy via configure_reporting through symlinked paths.
    real_cfg.unlink()
    sym_cfg.unlink()
    assert integration.configure_reporting(
        True,
        repository="mindie-agent/diagnostics",
        config=real_cfg,
        roots=[str(sym_logs)],
        consent_config=sym_consent,
    )["status"] == "configured"
    sym_cfg.symlink_to(real_cfg)

    # Reading policy and updating policy through symlinked diagnostics.json works cleanly.
    loaded = fallback.read_policy(sym_cfg)
    assert loaded is not None and loaded["decision"] == "enabled"
    assert integration.configure_reporting(
        True,
        repository="mindie-agent/diagnostics",
        config=sym_cfg,
        roots=[str(sym_logs)],
        consent_config=sym_consent,
    )["status"] == "configured"

    # Recording, ingesting, bundling, and pruning through symlinked log root works cleanly.
    os.environ["MINDIE_DIAGNOSTICS_CONFIG"] = str(sym_cfg)
    rec = fallback.record_failure(
        "mindie-harness", "seed.boot", stage="bootstrap", category="internal", root=sym_logs
    )
    assert rec["recorded"] is True

    bundle = collect_bundle(sym_logs)
    assert bundle["summary"]["event_count"] == 1

    q = Outbox(tmp_path / "sym-state" / "reporter.sqlite3")
    ing = ingest(sym_logs, q, config=sym_cfg)
    assert ing["enqueued"] == 1
    pruned = prune(sym_logs, queue=q)
    assert pruned["limited"] is False


def test_outbox_7d_pending_only_expiry_preserves_uncertain_published_and_withdrawn(tmp_path: Path) -> None:
    """`maintain()` expires ONLY never-dispatched `pending` (>7d) and NEVER expires `uncertain`, `published`, or `withdrawn`."""
    cfg = tmp_path / "reporting.json"
    integration.configure_reporting(True, config=cfg, repository="mindie-agent/diagnostics", roots=[str(tmp_path)])
    consent = fallback.current_consent(cfg)
    assert consent is not None

    now = [10_000.0]
    queue = Outbox(tmp_path / "outbox.sqlite3", clock=lambda: now[0])

    # 1. Published incident
    queue.enqueue("fp-pub", "1" * 32, {"component": "mindie-diagnostics", "operation": "op.pub"}, consent=consent)
    pub_item = queue.claim()
    queue.update(pub_item, state="published", issue_number=42, issue_url="https://github.com/mindie-agent/diagnostics/issues/42")

    # 2. Uncertain POST incident
    queue.enqueue("fp-unc", "2" * 32, {"component": "mindie-diagnostics", "operation": "op.unc"}, consent=consent)
    unc_item = queue.claim()
    assert queue.begin_post(unc_item)
    queue.update(unc_item, state="uncertain", last_error="submission_uncertain_timeout", next_attempt=now[0] + 60)

    # 3. Withdrawn incident
    other_cfg = tmp_path / "withdrawn.json"
    integration.configure_reporting(True, config=other_cfg, repository="mindie-agent/diagnostics", roots=[str(tmp_path)])
    other_consent = fallback.current_consent(other_cfg)
    queue.enqueue("fp-wdr", "3" * 32, {"component": "mindie-diagnostics", "operation": "op.wdr"}, consent=other_consent)
    integration.configure_reporting(False, config=other_cfg, repository="mindie-agent/diagnostics", roots=[str(tmp_path)])
    assert queue.withdraw_unconsented() == 1

    # 4. Stale never-dispatched pending incident (created at t=10_000)
    queue.enqueue("fp-stale-pending", "4" * 32, {"component": "mindie-diagnostics", "operation": "op.stale"}, consent=consent)

    # Advance +8 days (> 7d TTL), add a fresh pending incident, and run maintain().
    now[0] += 8 * 86400
    queue.enqueue("fp-fresh-pending", "5" * 32, {"component": "mindie-diagnostics", "operation": "op.fresh"}, consent=consent)
    queue.maintain()

    rows_8d = {r["fingerprint"]: r for r in queue.rows()}
    assert rows_8d["fp-stale-pending"]["state"] == "expired"
    assert rows_8d["fp-stale-pending"]["last_error"] == "pending_ttl_expired"
    assert rows_8d["fp-fresh-pending"]["state"] == "pending"
    assert rows_8d["fp-unc"]["state"] == "uncertain"
    assert rows_8d["fp-pub"]["state"] == "published"
    assert rows_8d["fp-wdr"]["state"] == "withdrawn"

    # Advance +40 more days (48 days total > 30d TERMINAL_HISTORY_SECONDS):
    # `uncertain`, `published` dedup fingerprint, and `withdrawn` receipt STILL survive!
    now[0] += 40 * 86400
    queue.maintain()
    rows_48d = {r["fingerprint"]: r for r in queue.rows()}
    assert rows_48d["fp-unc"]["state"] == "uncertain"
    assert rows_48d["fp-pub"]["state"] == "published"
    assert rows_48d["fp-pub"]["issue_number"] == 42
    assert rows_48d["fp-wdr"]["state"] == "withdrawn"


def test_diagnostics_state_service_pr_checklist_validator() -> None:
    """`.github/scripts/check_pr_checklist.py --profile state-service` validates PR descriptions cleanly."""
    script_path = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_pr_checklist.py"
    spec = importlib.util.spec_from_file_location("check_pr_checklist_diag", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    valid_body = """
## 变更摘要 (Summary)
- Replace 24/7 idle worker and 15s heartbeat thread with Scale-to-Zero drain worker and 7d pending expiry.

## 设计原则 Review (Pre-PR Checklist — `state-service`)
- [x] **SVC-CONSENT** — Allowlist and withdrawal verified
- [x] **SVC-STATE-COMPAT** — Stdlib zero-dep and ensure compatibility verified
- [x] **SVC-LIFETIME** — Scale-to-Zero and bootout retirement verified
- [x] **SVC-EXPORT** — 7d pending-only expiry and uncertain reconciliation verified

## 自查证据 (Evidence)
- **SVC-CONSENT**: `test_contribution_reporting.py` and `test_clean_arch_g2.py` passed.
- **SVC-STATE-COMPAT**: Offline wheel smoke check confirmed zero external dependencies.
- **SVC-LIFETIME**: Zero background threads in `Health`; worker exits 0 immediately when idle.
- **SVC-EXPORT**: `uncertain`, `published`, and `withdrawn` rows verified retained at +48d.

## 验证结果 (Verification)
- Ran `pytest -q -m "not live_npu"` (all passed).
"""
    assert mod.validate_pr_body(valid_body, profile="state-service") == []
