"""Durable Scale-to-Zero retry wake scheduling and legacy service migration.

[Role]
Manages on-demand OS retry wake companions (`<label>.retry.plist`, `<label> Retry`,
`<unit>-retry.timer`, WSL/non-systemd single-flight `delayed-run`, and
`<state>/next-wake.json`) strictly when active Outbox work remains, clears
them (including self-completing `launchd` retry jobs) immediately when the queue
drains empty, and retires legacy OS service registrations before runtime venv GC.

[Invariants]
- Zero processes and zero repeating timers remain loaded once Outbox drains empty.
- Linked-profile systemd timers outside `~/.config/systemd/user` are linked via
  `systemctl --user enable <timer_path>` and rearmed via `RemainAfterElapse=no` +
  `systemctl --user restart <timer_name>`.
- Non-systemd Linux / WSL2 `delayed-run` honors the `>= 5s` wake floor and waits
  for any retiring parent `worker-drain.lock` holder before invoking `run_worker`
  so batch handoffs are never lost to false `coalesced` exits.
- When a macOS `<label>.retry` job drains the queue itself, a detached shell-free
  `bootout-label` helper unloads `<label>.retry` from `launchd` right after exit.

[Timeouts & Locks]
- Subprocess probes use bounded 5s timeouts (or caller `runner` deadline) and
  `CREATE_NO_WINDOW` on Windows; `delayed-run` uses non-blocking `flock`.

[Side-Effect Policy]
- Writes/unlinks `<state>/next-wake.json` and companion retry timer definitions;
  best-effort wake scheduling errors never roll back completed Outbox drains.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import plistlib
import signal
import subprocess
import sys
import time

from . import fallback as f
from .bundle import _write_output


def _wake_receipt_path(state: Path) -> Path:
    return Path(state) / "next-wake.json"


def _timer_pid_path(state: Path) -> Path:
    return Path(state) / "retry-timer.json"


def read_wake_receipt(state: str | Path) -> dict | None:
    raw = f._read_regular_bounded(_wake_receipt_path(Path(state)), 4096)
    if not raw:
        return None
    try:
        doc = json.loads(raw)
        return doc if isinstance(doc, dict) and doc.get("schema") == 1 else None
    except (ValueError, TypeError):
        return None


def _run_quiet(runner, argv: list[str], *, timeout: float = 5.0):
    run_fn = runner or subprocess.run
    kwargs: dict = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "check": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return run_fn(argv, **kwargs)
    except TimeoutError:
        raise
    except (OSError, subprocess.SubprocessError):
        return None


def _spawn_detached_bootout(launcher: str, domain_target: str) -> None:
    """Unload a self-completing macOS `<label>.retry` job after the worker exits."""
    try:
        subprocess.Popen(
            [
                launcher,
                "-I",
                "-m",
                "mindie_diagnostics.wake_schedule",
                "bootout-label",
                "--domain-target",
                domain_target,
            ],
            cwd="/",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except TimeoutError:
        raise
    except OSError:
        pass


def _stop_local_retry_timer(state_dir: Path) -> None:
    if os.name != "posix":
        return
    pid_file = _timer_pid_path(state_dir)
    raw = f._read_regular_bounded(pid_file, 4096)
    if raw:
        try:
            from . import platform_service as ps
            doc = json.loads(raw)
            pid, ident = doc.get("pid"), doc.get("identity")
            if ps._linux_worker_alive(pid, ident):
                os.kill(pid, signal.SIGTERM)
        except TimeoutError:
            raise
        except Exception:
            pass
    try:
        pid_file.unlink(missing_ok=True)
    except TimeoutError:
        raise
    except OSError:
        pass


def _start_local_retry_timer(state_dir: Path, manifest: Path, cfg: dict) -> bool:
    if os.name != "posix":
        return False
    from . import platform_service as ps
    raw = f._read_regular_bounded(_timer_pid_path(state_dir), 4096)
    if raw:
        try:
            doc = json.loads(raw)
            if ps._linux_worker_alive(doc.get("pid"), doc.get("identity")):
                return True
        except Exception:
            pass
    worker_env = {
        k: v for k, v in os.environ.items()
        if k in ps._SAFE_WORKER_ENV or k.startswith(("XDG_", "MINDIE_"))
    }
    try:
        proc = subprocess.Popen(
            [
                str(cfg["launcher"]),
                "-I",
                "-m",
                "mindie_diagnostics.wake_schedule",
                "delayed-run",
                "--config",
                str(manifest),
                "--state",
                str(state_dir),
            ],
            cwd="/",
            env=worker_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            umask=0o077,
        )
    except TimeoutError:
        raise
    except OSError:
        return False
    ident = ps._linux_worker_identity(proc.pid)
    if ident:
        try:
            _write_output(
                _timer_pid_path(state_dir),
                (json.dumps({"pid": proc.pid, "identity": ident}, ensure_ascii=True) + "\n").encode("utf-8"),
            )
        except TimeoutError:
            raise
        except OSError:
            pass
    return True


def clear_retry_wake(
    state: str | Path,
    reporting_config=None,
    *,
    unit_dir=None,
    runner=None,
    force_probe: bool = False,
) -> None:
    """Clear `<state>/next-wake.json` and any companion OS or local retry wake timer."""
    state_dir = Path(state)
    try:
        _wake_receipt_path(state_dir).unlink(missing_ok=True)
    except TimeoutError:
        raise
    except OSError:
        pass
    _stop_local_retry_timer(state_dir)
    try:
        eff_dir = unit_dir if unit_dir is not None else f.service_directory(reporting_config)
    except Exception:
        eff_dir = unit_dir
    try:
        from . import platform_service as ps
        manifest, label, _unit = ps._paths(eff_dir)
        retry_label = f"{label}.retry"
        if sys.platform == "darwin":
            retry_plist = manifest.parent / f"{retry_label}.plist"
            domain_target = ps._domain(retry_label)
            plist_present = retry_plist.exists() or retry_plist.is_symlink()
            if os.environ.get("XPC_SERVICE_NAME") == retry_label:
                retry_plist.unlink(missing_ok=True)
                cfg = ps._read(manifest) if manifest.is_file() and not manifest.is_symlink() else None
                launcher = str(cfg["launcher"]) if cfg and cfg.get("launcher") else sys.executable
                if runner is None:
                    _spawn_detached_bootout(launcher, domain_target)
            elif plist_present or force_probe:
                probe = _run_quiet(runner, ["launchctl", "print", domain_target])
                if plist_present or (probe is not None and probe.returncode == 0):
                    _run_quiet(runner, ["launchctl", "bootout", domain_target])
                retry_plist.unlink(missing_ok=True)
        elif sys.platform == "win32":
            retry_xml = manifest.parent / "mindie-diagnostics-retry-task.xml"
            if retry_xml.exists() or retry_xml.is_symlink() or force_probe:
                task_name = f"{label} Retry"
                ps._ps(
                    runner or subprocess.run,
                    f"$n={ps._ps_quote(task_name)};"
                    "if(@(Get-ScheduledTask -ErrorAction SilentlyContinue|Where-Object{$_.TaskPath -eq '\\' -and $_.TaskName -eq $n}).Count -gt 0){"
                    "Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue}",
                    action="task.retry_clear",
                )
                retry_xml.unlink(missing_ok=True)
        elif sys.platform.startswith("linux"):
            from . import service as svc
            unit_path = svc._unit_path(eff_dir)
            timer_path = unit_path.with_name(unit_path.stem + "-retry.timer")
            if timer_path.exists() or timer_path.is_symlink() or force_probe:
                if timer_path.exists() or timer_path.is_symlink():
                    _run_quiet(runner, ["systemctl", "--user", "disable", "--now", timer_path.name])
                    timer_path.unlink(missing_ok=True)
                    _run_quiet(runner, ["systemctl", "--user", "daemon-reload"])
    except TimeoutError:
        raise
    except Exception:
        pass


def sync_retry_wake(
    state: str | Path,
    reporting_config,
    *,
    next_wake_at: float | None,
    now: float | None = None,
    unit_dir=None,
    runner=None,
) -> dict:
    """Persist durable retry wake ONLY when active Outbox work remains; else clear."""
    state_dir = Path(state)
    if next_wake_at is None or not math.isfinite(next_wake_at):
        clear_retry_wake(state_dir, reporting_config, unit_dir=unit_dir, runner=runner)
        return {"scheduled": False, "next_wake_at": None}

    ref_now = time.time() if now is None else float(now)
    delay = max(5, min(86400, int(math.ceil(next_wake_at - ref_now))))
    effective_wake = max(float(next_wake_at), ref_now + delay)
    receipt = {
        "schema": 1,
        "next_wake_at": float(next_wake_at),
        "wake_not_before": effective_wake,
        "delay_seconds": delay,
    }
    try:
        _write_output(_wake_receipt_path(state_dir), (json.dumps(receipt, ensure_ascii=True) + "\n").encode("utf-8"))
    except TimeoutError:
        raise
    except OSError:
        return {"scheduled": False, "next_wake_at": float(next_wake_at), "warning": "wake_receipt_write_failed"}

    os_scheduled = False
    try:
        eff_dir = unit_dir if unit_dir is not None else f.service_directory(reporting_config)
        from . import platform_service as ps
        manifest, label, _unit = ps._paths(eff_dir)
        cfg = ps._read(manifest) if manifest.is_file() and not manifest.is_symlink() else None
        if cfg is not None and sys.platform == "darwin":
            retry_label = f"{label}.retry"
            retry_plist = manifest.parent / f"{retry_label}.plist"
            interval = max(60, delay)
            payload = plistlib.dumps(
                {
                    "Label": retry_label,
                    "ProgramArguments": [str(cfg["launcher"]), *cfg["launch_args"]],
                    "RunAtLoad": False,
                    "StartInterval": interval,
                    "WorkingDirectory": str(manifest.parent),
                    "Umask": 0o077,
                    "StandardOutPath": "/dev/null",
                    "StandardErrorPath": "/dev/null",
                }
            )
            prior = retry_plist.read_bytes() if retry_plist.is_file() and not retry_plist.is_symlink() else None
            ps._write(retry_plist, payload)
            probe = _run_quiet(runner, ["launchctl", "print", ps._domain(retry_label)])
            loaded = probe is not None and probe.returncode == 0
            if loaded and prior != payload and os.environ.get("XPC_SERVICE_NAME") != retry_label:
                _run_quiet(runner, ["launchctl", "bootout", ps._domain(retry_label)])
                loaded = False
            if not loaded:
                boot = _run_quiet(runner, ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(retry_plist)])
                os_scheduled = boot is not None and boot.returncode == 0
            else:
                os_scheduled = True
        elif cfg is not None and sys.platform == "win32":
            retry_xml = manifest.parent / "mindie-diagnostics-retry-task.xml"
            start_iso = (datetime.now(timezone.utc) + timedelta(seconds=delay)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = (datetime.now(timezone.utc) + timedelta(seconds=delay + 3600)).strftime("%Y-%m-%dT%H:%M:%SZ")
            sid_reply = ps._ps(runner or subprocess.run, "[Security.Principal.WindowsIdentity]::GetCurrent().User.Value", action="task.identity")
            sid = sid_reply.stdout.strip()
            if sid.startswith("S-1-"):
                xml_body = (
                    f'<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
                    f"<RegistrationInfo><Description>{ps.MARKER}:{manifest}:retry</Description></RegistrationInfo>"
                    f"<Triggers><TimeTrigger><StartBoundary>{start_iso}</StartBoundary>"
                    f"<EndBoundary>{end_iso}</EndBoundary><Enabled>true</Enabled></TimeTrigger></Triggers>"
                    f'<Principals><Principal id="Author"><UserId>{sid}</UserId>'
                    f"<LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>"
                    f"<Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>"
                    f"<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>"
                    f"<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>"
                    f"<StartWhenAvailable>true</StartWhenAvailable>"
                    f"<DeleteExpiredTaskAfter>PT10M</DeleteExpiredTaskAfter>"
                    f"<Enabled>true</Enabled><Hidden>true</Hidden><ExecutionTimeLimit>PT10M</ExecutionTimeLimit></Settings>"
                    f'<Actions Context="Author"><Exec><Command>{cfg["launcher"]}</Command>'
                    f'<Arguments>{subprocess.list2cmdline(cfg["launch_args"])}</Arguments>'
                    f'<WorkingDirectory>{Path(cfg["launcher"]).parent}</WorkingDirectory></Exec></Actions></Task>'
                ).encode("utf-8")
                ps._write(retry_xml, xml_body)
                task_name = f"{label} Retry"
                reg = ps._ps(
                    runner or subprocess.run,
                    f"Register-ScheduledTask -TaskName {ps._ps_quote(task_name)} -TaskPath '\\' "
                    f"-Xml ([IO.File]::ReadAllText({ps._ps_quote(retry_xml)})) -Force|Out-Null",
                    action="task.retry_register",
                )
                os_scheduled = reg.returncode == 0
        elif sys.platform.startswith("linux"):
            from . import service as svc
            unit_path = svc._unit_path(eff_dir)
            if svc._read_owned(unit_path) is not None and svc._systemd_user_available(runner):
                timer_path = unit_path.with_name(unit_path.stem + "-retry.timer")
                timer_text = (
                    "[Unit]\nDescription=MindIE diagnostics bounded retry wake\n\n"
                    f"[Timer]\nOnActiveSec={delay}s\nAccuracySec=5s\nRemainAfterElapse=no\nUnit={unit_path.name}\n\n"
                    "[Install]\nWantedBy=timers.target\n"
                )
                _write_output(timer_path, timer_text.encode("utf-8"))
                _run_quiet(runner, ["systemctl", "--user", "daemon-reload"])
                en = _run_quiet(runner, ["systemctl", "--user", "enable", str(timer_path)])
                st = _run_quiet(runner, ["systemctl", "--user", "restart", timer_path.name])
                os_scheduled = bool(en is not None and en.returncode == 0 and st is not None and st.returncode == 0)
            elif cfg is not None:
                os_scheduled = _start_local_retry_timer(state_dir, manifest, cfg)
    except TimeoutError:
        raise
    except Exception:
        os_scheduled = False
    return {
        "scheduled": os_scheduled,
        "receipt_persisted": True,
        "next_wake_at": float(next_wake_at),
        "delay_seconds": delay,
    }


@contextmanager
def _try_timer_lock(state: Path):
    import fcntl
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(state / "retry-timer.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        try:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _await_drain_lock_release(state_dir: Path, max_wait: float = 5.0) -> None:
    """Ensure a retiring parent drain worker releases `worker-drain.lock` before handoff exec."""
    if os.name != "posix":
        return
    import fcntl
    lock_path = state_dir / "worker-drain.lock"
    if not lock_path.exists():
        return
    deadline = time.monotonic() + max_wait
    fd = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        while time.monotonic() < deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return
            except BlockingIOError:
                time.sleep(0.05)
    finally:
        os.close(fd)


def run_delayed_wake(config_path: str | Path, state_path: str | Path) -> int:
    """Single-flight non-systemd Linux / WSL2 one-shot retry sleeper; exits immediately if cleared."""
    state_dir = Path(os.path.realpath(state_path))
    stop = False

    def _handle_sig(*_a):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_sig)
    with _try_timer_lock(state_dir) as acquired:
        if not acquired:
            return 0
        try:
            while not stop:
                receipt = read_wake_receipt(state_dir)
                if receipt is None:
                    return 0
                due = receipt.get("wake_not_before")
                if not isinstance(due, (int, float)) or not math.isfinite(due):
                    due = receipt.get("next_wake_at")
                if not isinstance(due, (int, float)) or not math.isfinite(due):
                    return 0
                remaining = float(due) - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(1.0, remaining))
            if stop or read_wake_receipt(state_dir) is None:
                return 0
        finally:
            try:
                _timer_pid_path(state_dir).unlink(missing_ok=True)
            except TimeoutError:
                raise
            except OSError:
                pass
    _await_drain_lock_release(state_dir, max_wait=5.0)
    from .platform_service import run_worker
    return run_worker(config_path)


def run_bootout_label(domain_target: str) -> int:
    """Wait briefly for the parent `<label>.retry` process to exit, then unload its `StartInterval` job."""
    time.sleep(0.25)
    _run_quiet(None, ["launchctl", "bootout", domain_target], timeout=5.0)
    return 0


def retire_legacy_services(config, *, active_unit_dir, runner=None):
    """Unload and remove owned legacy services bound to `config` outside `active_unit_dir`."""
    from . import service as svc

    policy = f.read_policy(config)
    candidates: list[Path | None] = []
    if policy and policy.get("consent_config"):
        candidates.append(None)
    state_svc = f.state_path(config) / "service"
    if active_unit_dir is not None and Path(active_unit_dir).resolve() != state_svc.resolve():
        candidates.append(state_svc)
    primary_result = None
    for cand in candidates:
        if cand is not None and active_unit_dir is not None and Path(cand).resolve() == Path(active_unit_dir).resolve():
            continue
        if cand is None and active_unit_dir is None:
            continue
        if cand is None:
            res = svc.remove_service(expected_reporting_config=config, runner=runner)
            primary_result = res
        elif cand.is_dir():
            res = svc.remove_service(unit_dir=cand, expected_reporting_config=config, runner=runner)
            if primary_result is None:
                primary_result = res
    return primary_result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    dr = sub.add_parser("delayed-run")
    dr.add_argument("--config", required=True)
    dr.add_argument("--state", required=True)
    bl = sub.add_parser("bootout-label")
    bl.add_argument("--domain-target", required=True)
    opts = p.parse_args()
    if opts.action == "delayed-run":
        raise SystemExit(run_delayed_wake(opts.config, opts.state))
    raise SystemExit(run_bootout_label(opts.domain_target))
