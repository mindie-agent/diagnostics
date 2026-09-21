"""Owned Windows scheduled tasks and macOS launch agents for the local worker.

Managers supervise an installed Python process. Their configuration contains
paths and arguments only; optional credentials are read by that process from a
private file. Nothing installs a machine-wide service or changes login policy.
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
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

from . import service as common

MARKER = "mindie-diagnostics.native-service.v1"
MANIFEST = "mindie-diagnostics-service.json"
LIMIT = 32768
ENV_KEYS = {"GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR", "XAI_API_KEY", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"}


def _physical(path):
    path = common._absolute(path)
    if os.name != "nt" or not path.exists():
        return path
    # MSIX may virtualize LOCALAPPDATA. The scheduler must receive the physical
    # path visible outside the calling app's package, not its virtual alias.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0, 7, None, 3, 0x02000000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise common.ServiceError("physical_path_unavailable")
    try:
        output = ctypes.create_unicode_buffer(32768)
        count = kernel.GetFinalPathNameByHandleW(handle, output, len(output), 0)
        if not 0 < count < len(output):
            raise common.ServiceError("physical_path_unavailable")
        value = output.value
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value)
    finally:
        kernel.CloseHandle(handle)


def _paths(unit_dir=None):
    override = unit_dir or os.environ.get("MINDIE_DIAGNOSTICS_SERVICE_DIR")
    if override:
        root = common._absolute(override)
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local") / "mindie/diagnostics-service"
    else:
        root = Path.home() / "Library/LaunchAgents"
    root = _physical(root)
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    label = ("MindIE Diagnostics " if sys.platform == "win32" else "org.mindie.diagnostics.") + digest
    return root / MANIFEST, label, root / ("mindie-diagnostics-task.xml" if sys.platform == "win32" else label + ".plist")


@contextmanager
def _locked(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = path.with_suffix(".lock")
    if lock.is_symlink():
        raise common.ServiceError("unowned_lock")
    with lock.open("a+b") as stream:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + 10
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
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
        if result["marker"] != MARKER or result["platform"] not in {"win32", "darwin"}:
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
        runner = lambda command, **kwargs: underlying(command, **kwargs, creationflags=subprocess.CREATE_NO_WINDOW)
    return common._run(runner, argv, action=action, check=check)


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _ps(runner, script, *, action):
    prefix = ("$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
              "$OutputEncoding=[Console]::OutputEncoding;$ProgressPreference='SilentlyContinue';"
              "$env:PSModulePath=(Join-Path $PSHOME 'Modules')+[IO.Path]::PathSeparator+$env:PSModulePath;")
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
    element(action, "WorkingDirectory", str(manifest.parent))
    # Register-ScheduledTask receives a .NET UTF-16 string. An UTF-8 XML
    # declaration conflicts with that transport even though the file is UTF-8.
    return ET.tostring(task, encoding="utf-8", xml_declaration=False)


def _private_environment(value, runner):
    if value is None:
        return None
    path = common._absolute(value)
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file() or path.stat().st_size > LIMIT:
        raise common.ServiceError("unsafe_environment_file")
    path = _physical(path)
    if sys.platform != "win32":
        return common._environment_file(path)
    # Read ACL metadata only. Secret values are never passed through PowerShell.
    script = ("$p=" + _ps_quote(path) + ";$a=[IO.File]::GetAccessControl($p);"
              "$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
              "$ok=($a.GetOwner([Security.Principal.SecurityIdentifier]).Value -eq $sid);"
              "foreach($r in $a.Access){$id=$r.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value;"
              "if($r.AccessControlType -eq 'Allow' -and $id -notin @($sid,'S-1-5-18','S-1-5-32-544')){$ok=$false}};"
              "if(-not $ok){exit 9}")
    try:
        _ps(runner, script, action="credentials.permissions")
    except common.ServiceError:
        raise common.ServiceError("unsafe_environment_file") from None
    return path


def save_environment_token(state, runner=None):
    """Persist an explicitly authorized environment PAT with private permissions."""
    runner = runner or subprocess.run
    name = next((key for key in ("GH_TOKEN", "GITHUB_TOKEN") if os.environ.get(key)), None)
    if name is None:
        raise common.ServiceError("environment_token_required")
    token = os.environ[name]
    if len(token) > 8192 or any(character.isspace() or ord(character) < 32 for character in token):
        raise common.ServiceError("invalid_environment_token")
    root = common._absolute(state)
    if any(item.is_symlink() for item in (root, *root.parents)):
        raise common.ServiceError("unsafe_environment_file")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = _physical(root)
    path = root / "credentials.env"
    if path.exists() or path.is_symlink():
        _private_environment(path, runner)
    descriptor, temporary = tempfile.mkstemp(prefix=".mindie-credential-", dir=root)
    temporary_path = _physical(temporary)
    try:
        if sys.platform == "win32":
            # Restrict the empty file before writing the credential. No secret
            # enters a PowerShell argument, script, scheduler XML or receipt.
            script = ("$p=" + _ps_quote(temporary_path) + ";$id=[Security.Principal.WindowsIdentity]::GetCurrent().User;"
                      "$owner=[IO.File]::GetAccessControl($p).GetOwner([Security.Principal.SecurityIdentifier]).Value;"
                      "if($owner -ne $id.Value){throw 'credential_file_owner_mismatch'};"
                      "$a=New-Object Security.AccessControl.FileSecurity;"
                      "$a.SetAccessRuleProtection($true,$false);"
                      "$a.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($id,'FullControl','Allow')));"
                      "$system=New-Object Security.Principal.SecurityIdentifier('S-1-5-18');"
                      "$a.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($system,'FullControl','Allow')));"
                      "[IO.File]::SetAccessControl($p,$a)")
            _ps(runner, script, action="credentials.protect")
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            stream.write(name + "=" + token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        return _private_environment(path, runner)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


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


def _verify_unit(unit, config):
    if unit.is_symlink() or (unit.exists() and (config is None or unit.stat().st_size > LIMIT
            or hashlib.sha256(unit.read_bytes()).hexdigest() != config.get("unit_sha256"))):
        raise common.ServiceError("unowned_unit")


def install_service(roots, state, repository, *, python=None, gh=None, grok=None,
                    grok_home=None, grok_work=None, interval=60, since=None,
                    environment_file=None, unit_dir=None, runner=None, start=True, save_token=False, central_bot=False,
                    ensure=False, reporting_config=None):
    runner = runner or subprocess.run
    manifest, label, unit = _paths(unit_dir)
    reporting_config = common._absolute(reporting_config) if reporting_config is not None else None
    if reporting_config is not None and (grok or grok_home or grok_work or central_bot or save_token):
        raise common.ServiceError('reporter_configuration_conflict')
    roots = list(dict.fromkeys(str(common._absolute(root)) for root in roots))
    if ensure and central_bot:
        raise common.ServiceError('central_bot_is_not_a_local_reporter')
    if ensure and (grok or grok_home or grok_work):
        raise common.ServiceError('legacy_model_worker_requires_explicit_removal')
    if central_bot and roots:
        raise common.ServiceError("central_bot_does_not_watch_local_logs")
    if not (0 if central_bot else 1) <= len(roots) <= 32:
        raise common.ServiceError("diagnostic_roots_required")
    if central_bot and not grok:
        raise common.ServiceError("central_bot_requires_grok")
    state = common._absolute(state)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise common.ServiceError("invalid_repository")
    if type(interval) not in (int, float) or not 5 <= interval <= 86400:
        raise common.ServiceError("invalid_interval")
    if bool(grok) != bool(grok_home and grok_work) or ((grok_home or grok_work) and not grok):
        raise common.ServiceError("grok_profile_required")
    if save_token and environment_file is not None:
        raise common.ServiceError("choose_existing_file_or_save_token")
    with _locked(manifest):
        # Resolve again after creation, accounting for Windows package redirection.
        manifest, label, unit = _paths(manifest.parent)
        existing = _read(manifest)
        if ensure and existing:
            from .service_config import merged_reporter_options
            common._check_reporter_binding(existing['argv'], existing['environment_file'], state, repository, reporting_config)
            if reporting_config is None:
                retained = merged_reporter_options(existing['argv'], existing['environment_file'], roots, repository)
                roots, state, gh = retained['roots'], Path(retained['state']), retained['gh']
                environment_file = retained['environment_file']
                interval = retained['interval']
            save_token = False
        fixed_since = existing["since"] if existing else common._since(since)
        interpreter, version = common._interpreter(runner, python)
        launcher = _physical(interpreter)
        if sys.platform == "win32":
            launcher = launcher.with_name("pythonw.exe")
            if not launcher.is_file():
                raise common.ServiceError("windowless_python_required")
            loaded = _task(runner, label)
            _verify_task(loaded, manifest, existing)
        else:
            loaded = _launchd_status(runner, label, manifest, existing)
        _verify_unit(unit, existing)
        # The manager runs outside the calling application's MSIX namespace.
        # Materialize explicit local directories before resolving their physical
        # paths, including first use when the diagnostic directory is absent.
        physical_roots = []
        for root in roots:
            directory = Path(root)
            if any(item.is_symlink() for item in (directory, *directory.parents)):
                raise common.ServiceError('unsafe_diagnostic_root')
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            physical_roots.append(str(_physical(directory)))
        roots = list(dict.fromkeys(physical_roots))
        if any(item.is_symlink() for item in (state, *state.parents)):
            raise common.ServiceError('unsafe_state_directory')
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        state = _physical(state)
        if save_token:
            environment_file = save_environment_token(state, runner)
        env_file = _private_environment(environment_file, runner)
        gh_path = (str(common._executable(gh)) if ensure and existing and env_file is None
                   else common._reporter_executable(gh, env_file))
        argv = ["worker"]
        if central_bot:
            argv.append("--central-bot")
        if reporting_config is not None:
            argv += ["--reporting-config", str(reporting_config)]
        for root in roots:
            argv += ["--root", root]
        argv += ["--state", str(state), "--repository", repository, "--gh", gh_path,
                 "--since", fixed_since, "--interval", str(interval)]
        if grok:
            argv += ["--grok", str(common._executable(grok)), "--grok-home", str(common._absolute(grok_home)),
                     "--grok-work", str(common._absolute(grok_work))]
        config = {"marker": MARKER, "platform": sys.platform, "since": fixed_since, "label": label,
                  "launcher": str(launcher), "launch_args": ["-I", "-m", "mindie_diagnostics.platform_service", "run", "--config", str(manifest)],
                  "argv": argv, "environment_file": str(env_file) if env_file else None, "state": str(state)}
        if sys.platform == "win32":
            sid = _ps(runner, "[Security.Principal.WindowsIdentity]::GetCurrent().User.Value", action="task.identity").stdout.strip()
            if not re.fullmatch(r"S-1-[0-9-]+", sid):
                raise common.ServiceError("user_identity_unavailable")
            payload = _windows_xml(config, manifest, sid)
        else:
            payload = plistlib.dumps({"Label": label, "ProgramArguments": [str(launcher), *config["launch_args"]],
                                      "RunAtLoad": True,
                                      "WorkingDirectory": str(manifest.parent), "Umask": 0o077,
                                      "StandardOutPath": "/dev/null", "StandardErrorPath": "/dev/null"})
        config["unit_sha256"] = hashlib.sha256(payload).hexdigest()
        changed = existing != config
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        if changed:
            _write(unit, payload)
            _write(manifest, (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode("utf-8"))
        if sys.platform == "win32":
            if changed or loaded is None:
                _ps(runner, "Register-ScheduledTask -TaskName " + _ps_quote(label) + " -TaskPath '\\' -Xml "
                    "([IO.File]::ReadAllText(" + _ps_quote(unit) + ")) -Force|Out-Null", action="task.register")
            if start:
                if existing and (changed or save_token):
                    _ps(runner, "Stop-ScheduledTask -TaskName " + _ps_quote(label), action="task.stop")
                _ps(runner, "Start-ScheduledTask -TaskName " + _ps_quote(label), action="task.start")
        else:
            if loaded["returncode"] == 0 and (changed or save_token):
                _launchctl(runner, "bootout", _domain(label))
            if start:
                if changed or save_token or loaded["returncode"]:
                    _launchctl(runner, "bootstrap", _domain(label).rsplit("/", 1)[0], str(unit))
                _launchctl(runner, "kickstart", _domain(label))
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
    else:
        facts = _launchd_status(runner, label, manifest, config)
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
            _ps(runner, "Start-ScheduledTask -TaskName " + _ps_quote(label), action="task.start")
        else:
            loaded = _launchd_status(runner, label, manifest, config)
            if loaded["returncode"]:
                _launchctl(runner, "bootstrap", _domain(label).rsplit("/", 1)[0], str(unit))
            _launchctl(runner, "kickstart", _domain(label))
        return {"status": "start_requested", "unit": str(unit), "since": config["since"]}


def remove_service(*, unit_dir=None, runner=None):
    runner = runner or subprocess.run
    manifest, label, unit = _paths(unit_dir)
    with _locked(manifest):
        config = _read(manifest)
        if config is None:
            return {"status": "absent", "unit": str(unit)}
        _verify_unit(unit, config)
        if sys.platform == "win32":
            loaded = _task(runner, label)
            _verify_task(loaded, manifest, config)
            if loaded is not None:
                _ps(runner, "Stop-ScheduledTask -TaskName " + _ps_quote(label) + ";Unregister-ScheduledTask -TaskName "
                    + _ps_quote(label) + " -Confirm:$false", action="task.remove")
        else:
            if _launchd_status(runner, label, manifest, config)["returncode"] == 0:
                _launchctl(runner, "bootout", _domain(label))
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
        with env_file.open(encoding="utf-8") as stream:
            content = stream.read(LIMIT + 1)
        if len(content) > LIMIT:
            raise common.ServiceError("invalid_environment_file")
        for line in content.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, equal, value = line.partition("=")
            if not equal or key not in ENV_KEYS or any(ord(character) < 32 for character in value):
                raise common.ServiceError("invalid_environment_file")
            os.environ[key] = value
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
        # Safe bounded startup evidence even when credentials/package loading
        # fail before the worker's normal health loop is available.
        try:
            config_path = common._absolute(options.config)
            _write(config_path.with_name("service-failure.json"), json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "category": getattr(error, "category", "worker_start_failed"),
                 "at": common._since()}).encode("utf-8"))
        finally:
            raise SystemExit(1)
