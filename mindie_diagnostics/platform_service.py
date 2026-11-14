"""macOS LaunchAgent, Windows Scheduled Task, and non-systemd Linux worker management.

[Role]
Installs, queries, starts, and removes the user-scoped background worker on
macOS (`~/Library/LaunchAgents`), Windows (Scheduled Tasks), and non-systemd
Linux / WSL2 (`~/.config/mindie-agent/diagnostics-service`) using a JSON manifest
(`mindie-diagnostics-service.json`) and SHA-256 unit verification.

[Invariants]
- Never installs a machine-wide service or modifies user login policy.
- Re-verifies `reporting.json` consent and revision before any native unit write
  or service start/stop mutation.
- On macOS, `_await_bootout` confirms label removal (`launchctl print` exit 113)
  before running `launchctl bootstrap`.
- On Linux local supervision, worker PIDs are verified against `/proc/<pid>/stat`
  `starttime` so recycled PIDs are never signaled or mistaken for active workers.

[Timeouts & Locks]
- `_locked` acquires an exclusive `flock`/`msvcrt` lock on `.lock` with a 10s
  default ceiling capped by `runner.deadline`.
- `_await_bootout` and `_linux_stop_worker` bound stop waits to 3.0s capped by
  `runner.deadline`.

[Side-Effect Policy]
- If writing the updated unit or manifest fails after stopping the previous
  worker, the prior unit and manifest bytes are restored without starting it.
- `remove_service` unloads and deletes only owned unit/manifest files while
  preserving diagnostic logs and outbox state.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

from . import service as common
from .fallback import _private_mkdir
from . import service_environment
from .windows_interpreter import require_windowless

MARKER = "mindie-diagnostics.native-service.v1"
MANIFEST = "mindie-diagnostics-service.json"
LIMIT = 32768


def _physical(path):
    path = common._absolute(path)
    return path.resolve() if os.name == "nt" and path.exists() else path


def _paths(unit_dir=None):
    override = unit_dir or os.environ.get("MINDIE_DIAGNOSTICS_SERVICE_DIR")
    if override:
        root = common._absolute(override)
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local") / "mindie/diagnostics-service"
    elif sys.platform == "darwin":
        root = Path.home() / "Library/LaunchAgents"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "mindie-agent/diagnostics-service"
    root = _physical(root)
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    if sys.platform == "win32":
        return root / MANIFEST, "MindIE Diagnostics " + digest, root / "mindie-diagnostics-task.xml"
    if sys.platform == "darwin":
        label = "org.mindie.diagnostics." + digest
        return root / MANIFEST, label, root / (label + ".plist")
    label = "mindie-diagnostics-" + digest
    return root / MANIFEST, label, root / (label + ".local.json")


@contextmanager
def _locked(path, deadline=None):
    _private_mkdir(path.parent, parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    if lock.is_symlink():
        raise common.ServiceError("unowned_lock")
    with lock.open("a+b") as stream:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"0")
            stream.flush()
        stop = time.monotonic() + 10
        if deadline is not None:
            stop = min(stop, deadline)
        while True:
            if time.monotonic() >= stop:
                raise common.ServiceError("installation_busy")
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except TimeoutError:
                raise
            except (BlockingIOError, OSError):
                if time.monotonic() >= stop:
                    raise common.ServiceError("installation_busy") from None
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _read(path):
    if path.is_symlink():
        raise common.ServiceError("unowned_unit")
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise common.ServiceError("unowned_unit")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["marker"] != MARKER or result["platform"] not in {"win32", "darwin", "linux"}:
            raise ValueError("owner")
        common._since(result["since"])
        if not isinstance(result["argv"], list) or not result["argv"] or result["argv"][0] != "worker":
            raise ValueError("argv")
        return result
    except (ValueError, KeyError, TypeError) as exc:
        raise common.ServiceError("unowned_unit") from exc


def _write(path, data):
    if len(data) > LIMIT:
        raise common.ServiceError("unit_too_large")
    fd, temporary = tempfile.mkstemp(prefix=".mindie-service-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run(runner, argv, *, action, check=True):
    if os.name == "nt":
        underlying = runner
        runner = lambda command, **kwargs: underlying(
            command,
            **{
                **kwargs,
                "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
                **({"errors": "replace"} if kwargs.get("text") else {}),
            },
        )
    return common._run(runner, argv, action=action, check=check)


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _ps(runner, script, *, action):
    prefix = (
        "$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';"
        "$u=[Text.UTF8Encoding]::new($false);$OutputEncoding=$u;"
        "try{[Console]::InputEncoding=$u}catch [System.IO.IOException]{};"
        "try{[Console]::OutputEncoding=$u}catch [System.IO.IOException]{};"
        "$o=[IO.StreamWriter]::new([Console]::OpenStandardOutput(),$u);$o.AutoFlush=$true;[Console]::SetOut($o);"
        "$e=[IO.StreamWriter]::new([Console]::OpenStandardError(),$u);$e.AutoFlush=$true;[Console]::SetError($e);"
        "$env:PSModulePath=(Join-Path $PSHOME 'Modules')+[IO.Path]::PathSeparator+$env:PSModulePath;"
    )
    encoded = base64.b64encode((prefix + script).encode("utf-16-le")).decode("ascii")
    return _run(runner, ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                         "-EncodedCommand", encoded], action=action)


def _task(runner, label):
    script = ("$t=@(Get-ScheduledTask -ErrorAction Stop | Where-Object { $_.TaskPath -eq '\\' -and $_.TaskName -eq "
              + _ps_quote(label) + " });if($t.Count -eq 0){'null'}else{"
              "$i=Get-ScheduledTaskInfo -InputObject $t[0];"
              "@{xml=(Export-ScheduledTask -InputObject $t[0]);state=[string]$t[0].State;last_result=$i.LastTaskResult}"
              "|ConvertTo-Json -Compress}")
    reply = _ps(runner, script, action="task.query")
    try:
        return json.loads(reply.stdout.lstrip("\ufeff"))
    except ValueError as exc:
        raise common.ServiceError("task_status_invalid") from exc


def _verify_task(task, manifest, config):
    if task is None:
        return
    if config is None:
        raise common.ServiceError("unowned_loaded_unit")
    try:
        root = ET.fromstring(task["xml"])
        def value(suffix):
            return next((item.text or "" for item in root.iter() if item.tag.endswith("}" + suffix)), "")
        if (value("Description") != MARKER + ":" + str(manifest)
                or value("Command") != config["launcher"]
                or value("Arguments") != subprocess.list2cmdline(config["launch_args"])):
            raise ValueError("owner")
    except (ET.ParseError, KeyError, TypeError, ValueError) as exc:
        raise common.ServiceError("unowned_loaded_unit") from exc


def _windows_xml(config, manifest, sid):
    namespace = "http://schemas.microsoft.com/windows/2004/02/mit/task"
    ET.register_namespace("", namespace)
    task = ET.Element("{" + namespace + "}Task", version="1.2")
    def element(parent, name, text=None, **attributes):
        node = ET.SubElement(parent, "{" + namespace + "}" + name, attributes)
        node.text = text
        return node
    registration = element(task, "RegistrationInfo")
    element(registration, "Description", MARKER + ":" + str(manifest))
    trigger = element(element(task, "Triggers"), "LogonTrigger")
    element(trigger, "Enabled", "true")
    element(trigger, "UserId", sid)
    principal = element(element(task, "Principals"), "Principal", id="Author")
    element(principal, "UserId", sid)
    element(principal, "LogonType", "InteractiveToken")
    element(principal, "RunLevel", "LeastPrivilege")
    settings = element(task, "Settings")
    for name, value in (("MultipleInstancesPolicy", "IgnoreNew"), ("DisallowStartIfOnBatteries", "false"),
                        ("StopIfGoingOnBatteries", "false"), ("StartWhenAvailable", "true"),
                        ("Enabled", "true"), ("Hidden", "true"), ("ExecutionTimeLimit", "PT0S")):
        element(settings, name, value)
    action = element(element(task, "Actions", Context="Author"), "Exec")
    element(action, "Command", config["launcher"])
    element(action, "Arguments", subprocess.list2cmdline(config["launch_args"]))
    # Windows rejects a long working directory before launching the short venv.
    element(action, "WorkingDirectory", str(Path(config["launcher"]).parent))
    # Register-ScheduledTask receives a .NET UTF-16 string. An UTF-8 XML
    # declaration conflicts with that transport even though the file is UTF-8.
    return ET.tostring(task, encoding="utf-8", xml_declaration=False)


def _private_environment(value, runner):
    if value is None:
        return None
    path = common._absolute(value)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT:
        raise common.ServiceError("unsafe_environment_file")
    path = _physical(path)
    if sys.platform != "win32":
        return common._environment_file(path)
    return path


def _launchctl(runner, *args, check=True):
    return _run(runner, ["launchctl", *args], action="launchctl." + args[0], check=check)


def _domain(label):
    return "gui/" + str(os.getuid()) + "/" + label


def _launchd_status(runner, label, manifest, config):
    reply = _launchctl(runner, "print", _domain(label), check=False)
    if reply.returncode:
        if "Could not find service" not in reply.stderr:
            raise common.ServiceError("service_manager_unavailable", action="launchctl.print", returncode=reply.returncode)
        return {"status": "inactive", "returncode": reply.returncode}
    if config is None or str(manifest) not in reply.stdout or config["launcher"] not in reply.stdout:
        raise common.ServiceError("unowned_loaded_unit")
    match = re.search(r"\bstate = ([a-z]+)", reply.stdout)
    return {"status": "active" if match and match[1] == "running" else "idle", "returncode": 0}


def _pid_path(manifest: Path) -> Path:
    return manifest.with_name("worker-process.json")


def _linux_worker_identity(pid: int) -> str | None:
    if type(pid) is not int or pid <= 1:
        return None
    if Path("/proc/self/stat").exists():
        try:
            proc_dir = Path(f"/proc/{pid}")
            uid = proc_dir.stat().st_uid
            if uid != os.getuid():
                return None
            raw = (proc_dir / "stat").read_bytes()
        except TimeoutError:
            raise
        except OSError:
            return None
        rparen = raw.rfind(b")")
        if rparen < 0:
            return None
        fields = raw[rparen + 2:].split()
        if len(fields) <= 19 or fields[0] == b"Z":
            return None
        try:
            return f"proc:{uid}:{int(fields[19])}"
        except ValueError:
            return None
    try:
        ps_exe = common.which_native("ps", which_fn=common.shutil.which) or "/bin/ps"
        reply = subprocess.run(
            [str(ps_exe), "-o", "uid=,state=,lstart=", "-p", str(pid)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "LC_ALL": "C"}, timeout=3, check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except TimeoutError:
        raise
    except OSError:
        return None
    parts = reply.stdout.strip().split(None, 2)
    if reply.returncode != 0 or len(parts) < 3 or parts[1].startswith("Z"):
        return None
    try:
        uid = int(parts[0])
    except ValueError:
        return None
    return f"ps:{uid}:{parts[2]}" if uid == os.getuid() else None


def _linux_worker_alive(pid, expected_identity) -> bool:
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        return False
    if type(expected_identity) is not str or not expected_identity:
        return False
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    except ChildProcessError:
        pass
    except TimeoutError:
        raise
    except OSError:
        pass
    observed = _linux_worker_identity(pid)
    if observed is None or observed != expected_identity:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except TimeoutError:
        raise
    except OSError:
        return False


def _linux_local_status(manifest: Path, config) -> dict:
    pid_file = _pid_path(manifest)
    if pid_file.is_symlink():
        raise common.ServiceError("unowned_unit")
    if not pid_file.exists():
        return {"status": "inactive", "returncode": 3}
    if config is None or not pid_file.is_file() or pid_file.stat().st_size > LIMIT:
        raise common.ServiceError("unowned_loaded_unit")
    try:
        doc = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = doc["pid"]
        identity = doc.get("identity")
        config_sha = doc.get("config_sha256")
        if doc.get("manifest") != str(manifest) or type(pid) is not int or pid <= 1:
            raise ValueError("owner")
        if type(identity) is not str or not identity:
            raise ValueError("identity")
        if config_sha is not None and type(config_sha) is not str:
            raise ValueError("config_sha256")
    except (ValueError, KeyError, TypeError) as exc:
        raise common.ServiceError("unowned_loaded_unit") from exc
    alive = _linux_worker_alive(pid, identity)
    return {"status": "active" if alive else "idle", "pid": pid, "identity": identity,
            "config_sha256": config_sha, "returncode": 0 if alive else 1}


def _linux_stop_worker(manifest: Path, config, *, deadline=None) -> None:
    status = _linux_local_status(manifest, config)
    if status["returncode"] == 0:
        pid = status["pid"]
        identity = status["identity"]
        if _linux_worker_alive(pid, identity):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except TimeoutError:
                raise
            except OSError:
                pass
            stop = time.monotonic() + 3.0
            if deadline is not None:
                stop = min(stop, deadline)
            while _linux_worker_alive(pid, identity) and time.monotonic() < stop:
                time.sleep(0.025)
            if _linux_worker_alive(pid, identity):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except TimeoutError:
                    raise
                except OSError:
                    pass
                kill_stop = time.monotonic() + 0.5
                while _linux_worker_alive(pid, identity) and time.monotonic() < kill_stop:
                    time.sleep(0.02)
                if _linux_worker_alive(pid, identity):
                    raise common.ServiceError("service_removal_unconfirmed", action="local.stop", returncode=1)
    _pid_path(manifest).unlink(missing_ok=True)


_SAFE_WORKER_ENV = frozenset({
    "HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "LC_CTYPE",
    "TZ", "TMPDIR", "GH_HOST", "GH_CONFIG_DIR",
})


def _cleanup_unregistered_child(proc) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=0.5)
    except TimeoutError:
        raise
    except OSError:
        pass


def _linux_start_worker(manifest: Path, launcher: Path, launch_args: list[str], *, config_sha256: str) -> None:
    worker_env = {
        k: v for k, v in os.environ.items()
        if k in _SAFE_WORKER_ENV or k.startswith(("XDG_", "MINDIE_"))
    }
    try:
        proc = subprocess.Popen(
            [str(launcher), *launch_args],
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
    except OSError as exc:
        raise common.ServiceError("command_failed", action="local.start", returncode=1) from exc
    try:
        identity = _linux_worker_identity(proc.pid)
        if not identity:
            raise common.ServiceError("command_failed", action="local.start", returncode=1)
        payload = json.dumps(
            {"manifest": str(manifest), "config_sha256": config_sha256, "pid": proc.pid, "identity": identity},
            ensure_ascii=True, separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        _write(_pid_path(manifest), payload)
    except BaseException:
        _cleanup_unregistered_child(proc)
        _pid_path(manifest).unlink(missing_ok=True)
        raise


def _publish_unit_and_manifest(unit: Path, payload: bytes, manifest: Path, manifest_bytes: bytes) -> None:
    prior_unit = unit.read_bytes() if unit.is_file() and not unit.is_symlink() else None
    prior_manifest = manifest.read_bytes() if manifest.is_file() and not manifest.is_symlink() else None
    try:
        _write(unit, payload)
        _write(manifest, manifest_bytes)
    except (TimeoutError, OSError, common.ServiceError):
        try:
            if prior_unit is None:
                unit.unlink(missing_ok=True)
            else:
                _write(unit, prior_unit)
            if prior_manifest is None:
                manifest.unlink(missing_ok=True)
            else:
                _write(manifest, prior_manifest)
        except TimeoutError:
            raise
        except (OSError, common.ServiceError) as exc:
            raise common.ServiceError("unit_restore_failed", action="unit.restore") from exc
        raise


def _verify_unit(unit, config):
    if unit.is_symlink() or (unit.exists() and (config is None or unit.stat().st_size > LIMIT
            or hashlib.sha256(unit.read_bytes()).hexdigest() != config.get("unit_sha256"))):
        raise common.ServiceError("unowned_unit")


def _await_bootout(runner, label):
    """Confirm one bootout removed the label before one bootstrap; read-only."""
    stop = time.monotonic() + 3
    outer = getattr(runner, 'deadline', None)
    if outer is not None:
        stop = min(stop, outer)

    def probe(command, **kwargs):
        remaining = stop - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, 0)
        timeout = kwargs.get('timeout')
        kwargs['timeout'] = remaining if timeout is None else min(timeout, remaining)
        return runner(command, **kwargs)

    while True:
        reply = _launchctl(probe, "print", _domain(label), check=False)
        if reply.returncode == 113:
            return
        if reply.returncode != 0:
            raise common.ServiceError("service_manager_unavailable",
                                      action="launchctl.print", returncode=reply.returncode)
        if time.monotonic() >= stop:
            raise common.ServiceError("service_removal_unconfirmed",
                                      action="launchctl.print", returncode=reply.returncode)
        time.sleep(.05)


def install_service(roots, state, repository, *, python=None, gh=None, interval=60, since=None,
                    environment_file=None, unit_dir=None, runner=None, start=True,
                    ensure=False, reporting_config=None, expected_consent_revision=None):
    if reporting_config is None:
        raise common.ServiceError('reporting_config_required')
    runner = runner or subprocess.run
    manifest, label, unit = _paths(unit_dir)
    reporting_config = common._absolute(reporting_config)
    roots = list(dict.fromkeys(str(common._absolute(root)) for root in roots))
    if not 1 <= len(roots) <= 32:
        raise common.ServiceError("diagnostic_roots_required")
    state = common._absolute(state)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise common.ServiceError("invalid_repository")
    if type(interval) not in (int, float) or not 5 <= interval <= 86400:
        raise common.ServiceError("invalid_interval")
    with _locked(manifest, deadline=getattr(runner, 'deadline', None)):
        common._check_reporting_consent(reporting_config, expected_consent_revision)
        # Resolve again after creation, accounting for Windows package redirection.
        manifest, label, unit = _paths(manifest.parent)
        existing = _read(manifest)
        if ensure and existing:
            common._check_reporter_binding(existing['argv'], existing['environment_file'], _physical(state), repository, reporting_config)
            if environment_file is None:
                environment_file = existing["environment_file"]
        fixed_since = existing["since"] if existing else common._since(since)
        interpreter, version = common._interpreter(
            (lambda command, **kwargs: runner(
                command,
                **{
                    **kwargs,
                    "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    **({"errors": "replace"} if kwargs.get("text") else {}),
                },
            )) if os.name == "nt" else runner,
            python,
        )
        launcher = _physical(interpreter)
        if sys.platform == "win32":
            launcher = launcher.with_name("pythonw.exe")
            require_windowless(launcher)
            loaded = _task(runner, label)
            _verify_task(loaded, manifest, existing)
        elif sys.platform == "darwin":
            loaded = _launchd_status(runner, label, manifest, existing)
        else:
            loaded = _linux_local_status(manifest, existing)
        _verify_unit(unit, existing)
        physical_roots = []
        for root in roots:
            directory = Path(root)
            if directory.is_symlink():
                raise common.ServiceError('unsafe_diagnostic_root')
            _private_mkdir(directory, parents=True, exist_ok=True)
            physical_roots.append(str(_physical(directory)))
        roots = list(dict.fromkeys(physical_roots))
        if any(item.is_symlink() for item in (state, *state.parents)):
            raise common.ServiceError('unsafe_state_directory')
        _private_mkdir(state, parents=True, exist_ok=True)
        state = _physical(state)
        environment_file, network_changed = service_environment.prepare(manifest.parent, environment_file)
        env_file = _private_environment(environment_file, runner)
        gh_path = (str(common._executable(gh)) if ensure and existing and env_file is None
                   else common._reporter_executable(gh, env_file))
        argv = ["worker", "--reporting-config", str(reporting_config)]
        for root in roots:
            argv += ["--root", root]
        argv += ["--state", str(state), "--repository", repository, "--gh", gh_path,
                 "--since", fixed_since, "--interval", str(interval)]
        config = {"marker": MARKER, "platform": sys.platform, "since": fixed_since, "label": label,
                  "launcher": str(launcher), "launch_args": ["-I", "-m", "mindie_diagnostics.platform_service", "run", "--config", str(manifest)],
                  "argv": argv, "environment_file": str(env_file) if env_file else None, "state": str(state)}
        if sys.platform == "win32":
            sid = _ps(runner, "[Security.Principal.WindowsIdentity]::GetCurrent().User.Value", action="task.identity").stdout.strip()
            if not re.fullmatch(r"S-1-[0-9-]+", sid):
                raise common.ServiceError("user_identity_unavailable")
            payload = _windows_xml(config, manifest, sid)
        elif sys.platform == "darwin":
            payload = plistlib.dumps({"Label": label, "ProgramArguments": [str(launcher), *config["launch_args"]],
                                      "RunAtLoad": True,
                                      "WorkingDirectory": str(manifest.parent), "Umask": 0o077,
                                      "StandardOutPath": "/dev/null", "StandardErrorPath": "/dev/null"})
        else:
            payload = (json.dumps({"label": label, "program": [str(launcher), *config["launch_args"]],
                                   "working_directory": "/"}, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
        config["unit_sha256"] = hashlib.sha256(payload).hexdigest()
        changed = existing != config or network_changed
        _private_mkdir(state, parents=True, exist_ok=True)
        manifest_bytes = (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
        if sys.platform == "win32":
            common._check_reporting_consent(reporting_config, expected_consent_revision)
            if changed:
                _write(unit, payload)
                _write(manifest, manifest_bytes)
            common._check_reporting_consent(reporting_config, expected_consent_revision)
            if changed or loaded is None:
                _ps(runner, "Register-ScheduledTask -TaskName " + _ps_quote(label) + " -TaskPath '\\' -Xml "
                    "([IO.File]::ReadAllText(" + _ps_quote(unit) + ")) -Force|Out-Null", action="task.register")
            if start:
                common._check_reporting_consent(reporting_config, expected_consent_revision)
                if existing and changed:
                    _ps(runner, "Stop-ScheduledTask -TaskName " + _ps_quote(label), action="task.stop")
                common._check_reporting_consent(reporting_config, expected_consent_revision)
                _ps(runner, "Start-ScheduledTask -TaskName " + _ps_quote(label), action="task.start")
        elif sys.platform == "darwin":
            if loaded["returncode"] == 0 and changed:
                common._check_reporting_consent(reporting_config, expected_consent_revision)
                _launchctl(runner, "bootout", _domain(label))
                _await_bootout(runner, label)
            if changed:
                common._check_reporting_consent(reporting_config, expected_consent_revision)
                _publish_unit_and_manifest(unit, payload, manifest, manifest_bytes)
            if start:
                common._check_reporting_consent(reporting_config, expected_consent_revision)
                if changed or loaded["returncode"]:
                    _launchctl(runner, "bootstrap", _domain(label).rsplit("/", 1)[0], str(unit))
                common._check_reporting_consent(reporting_config, expected_consent_revision)
                _launchctl(runner, "kickstart", _domain(label))
        else:
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            if changed:
                common._check_reporting_consent(reporting_config, expected_consent_revision)
                _publish_unit_and_manifest(unit, payload, manifest, manifest_bytes)
            if start:
                stale_worker = loaded["returncode"] == 0 and (changed or loaded.get("config_sha256") != manifest_sha)
                if stale_worker:
                    common._check_reporting_consent(reporting_config, expected_consent_revision)
                    _linux_stop_worker(manifest, existing or config, deadline=getattr(runner, "deadline", None))
                if stale_worker or loaded["returncode"] != 0:
                    common._check_reporting_consent(reporting_config, expected_consent_revision)
                    _linux_start_worker(manifest, launcher, config["launch_args"], config_sha256=manifest_sha)
        return {"status": "installed", "unit": str(unit), "manifest": str(manifest), "task_name": label,
                "since": fixed_since, "changed": changed, "start_requested": start, "python": str(interpreter),
                "package_version": version, "state": str(state), "platform": sys.platform}


def service_status(*, unit_dir=None, runner=None):
    runner = runner or subprocess.run
    manifest, label, unit = _paths(unit_dir)
    config = _read(manifest)
    if config is None:
        return {"status": "absent", "unit": str(unit), "task_name": label}
    _verify_unit(unit, config)
    if sys.platform == "win32":
        loaded = _task(runner, label)
        _verify_task(loaded, manifest, config)
        facts = {"status": "absent" if loaded is None else {"Running": "active", "Ready": "idle", "Disabled": "disabled"}.get(loaded["state"], "unknown"),
                 "last_result": loaded.get("last_result") if loaded else None}
    elif sys.platform == "darwin":
        facts = _launchd_status(runner, label, manifest, config)
    else:
        local = _linux_local_status(manifest, config)
        facts = {"status": local["status"], "returncode": local["returncode"]}
        if "pid" in local:
            facts["pid"] = local["pid"]
    return {**facts, "unit": str(unit), "manifest": str(manifest), "task_name": label, "since": config["since"]}


def start_service(*, unit_dir=None, runner=None):
    runner = runner or subprocess.run
    manifest, label, unit = _paths(unit_dir)
    with _locked(manifest):
        config = _read(manifest)
        if config is None:
            raise common.ServiceError("service_not_installed")
        _verify_unit(unit, config)
        if not unit.is_file():
            raise common.ServiceError("service_not_installed")
        if sys.platform == "win32":
            loaded = _task(runner, label)
            _verify_task(loaded, manifest, config)
            if loaded is None:
                raise common.ServiceError("service_not_installed")
            require_windowless(config["launcher"])
            _ps(runner, "Start-ScheduledTask -TaskName " + _ps_quote(label), action="task.start")
        elif sys.platform == "darwin":
            loaded = _launchd_status(runner, label, manifest, config)
            if loaded["returncode"]:
                _launchctl(runner, "bootstrap", _domain(label).rsplit("/", 1)[0], str(unit))
            _launchctl(runner, "kickstart", _domain(label))
        else:
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            loaded = _linux_local_status(manifest, config)
            if loaded["returncode"] == 0 and loaded.get("config_sha256") != manifest_sha:
                _linux_stop_worker(manifest, config, deadline=getattr(runner, "deadline", None))
                loaded = {"returncode": 3}
            if loaded["returncode"] != 0:
                _linux_start_worker(manifest, Path(config["launcher"]), config["launch_args"], config_sha256=manifest_sha)
        return {"status": "start_requested", "unit": str(unit), "since": config["since"]}


def remove_service(*, unit_dir=None, runner=None, expected_reporting_config=None):
    runner = runner or subprocess.run
    manifest, label, unit = _paths(unit_dir)
    with _locked(manifest):
        config = _read(manifest)
        if config is None:
            return {"status": "absent", "unit": str(unit)}
        if expected_reporting_config is not None:
            from .service_config import worker_options
            binding = worker_options(config["argv"], config["environment_file"])
            if binding["reporting_config"] != str(common._absolute(expected_reporting_config)):
                return {"status": "different_profile", "unit": str(unit)}
        _verify_unit(unit, config)
        if sys.platform == "win32":
            loaded = _task(runner, label)
            _verify_task(loaded, manifest, config)
            if loaded is not None:
                _ps(runner, "Stop-ScheduledTask -TaskName " + _ps_quote(label) + ";Unregister-ScheduledTask -TaskName "
                    + _ps_quote(label) + " -Confirm:$false", action="task.remove")
        elif sys.platform == "darwin":
            if _launchd_status(runner, label, manifest, config)["returncode"] == 0:
                _launchctl(runner, "bootout", _domain(label))
        else:
            _linux_stop_worker(manifest, config, deadline=getattr(runner, "deadline", None))
        if _read(manifest) != config:
            raise common.ServiceError("unit_changed_during_removal")
        unit.unlink(missing_ok=True)
        manifest.unlink()
        return {"status": "removed", "unit": str(unit), "state_preserved": True}


def run_worker(config_path):
    path = common._absolute(config_path)
    config = _read(path)
    if config is None:
        raise common.ServiceError("service_not_installed")
    for key in ("PYTHONPATH", "PYTHONHOME", "GH_TOKEN", "GITHUB_TOKEN"):
        os.environ.pop(key, None)
    env_file = _private_environment(config.get("environment_file"), subprocess.run)
    if env_file is not None:
        try:
            service_environment.apply(service_environment.read(env_file))
        except ValueError:
            raise common.ServiceError("invalid_environment_file") from None
    from .cli import main
    # JSONL and health are already bounded and owned by the worker. Avoid a
    # second unbounded console log or pythonw's absent standard streams.
    with open(os.devnull, "w", encoding="utf-8") as output, redirect_stdout(output), redirect_stderr(output):
        return main(config["argv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    try:
        raise SystemExit(run_worker(options.config))
    except Exception as error:
        try:
            config_path = common._absolute(options.config)
            _write(config_path.with_name("service-failure.json"), json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "category": getattr(error, "category", "worker_start_failed"),
                 "at": common._since()}).encode("utf-8"))
        finally:
            raise SystemExit(1)
