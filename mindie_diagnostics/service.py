"""Install an explicitly enabled native user service; never edit personal units.

This is installation work, not a task entry or background execution authority.
The unit rate-limits its journal output; system-wide journal capacity remains
under the administrator's existing policy. No journald configuration is changed.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

UNIT = "mindie-diagnostics.service"
MARKER = "# Owned by mindie-diagnostics user-service schema=1\n"
METADATA = "# Installation: "
MAX_UNIT_BYTES = 32768


class ServiceError(RuntimeError):
    def __init__(self, category, *, action=None, returncode=None):
        self.category, self.action, self.returncode = category, action, returncode
        super().__init__(category + (f" ({action}, exit {returncode})" if action else ""))


def _linux():
    if sys.platform != "linux":
        raise ServiceError("unsupported_platform")


def _absolute(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or any(c in str(path) for c in "\0\r\n"):
        raise ServiceError("absolute_path_required")
    # Do not resolve a venv's bin/python symlink into its base interpreter.
    return path.absolute()


def _unit_path(unit_dir):
    if unit_dir is None:
        unit_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd/user"
    return _absolute(unit_dir) / UNIT


def _quoted(value):
    value = str(value)
    if any(c in value for c in "\0\r\n"):
        raise ServiceError("invalid_unit_argument")
    # systemd expands specifiers and environment variables independently of its
    # argument quoting. These escapes prevent both; no shell ever interprets it.
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def _read_owned(path):
    if path.is_symlink():
        raise ServiceError("unowned_unit")
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > MAX_UNIT_BYTES:
        raise ServiceError("unowned_unit")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not text.startswith(MARKER) or len(lines) < 2 or not lines[1].startswith(METADATA):
        raise ServiceError("unowned_unit")
    try:
        metadata = json.loads(lines[1][len(METADATA):])
        if metadata.get("schema") != 1:
            raise ValueError("schema")
        _since(metadata["since"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ServiceError("invalid_owned_unit") from exc
    return text, metadata


def _since(value=None):
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError as exc:
        raise ServiceError("invalid_since") from exc


def _run(runner, command, *, check=True, action):
    try:
        result = runner(command, capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)
    except FileNotFoundError as exc:
        raise ServiceError("command_unavailable", action=action) from exc
    except subprocess.TimeoutExpired as exc:
        raise ServiceError("command_timeout", action=action) from exc
    if check and result.returncode:
        raise ServiceError("command_failed", action=action, returncode=result.returncode)
    return result


def _systemctl(runner, *args, check=True):
    return _run(runner, ["systemctl", "--user", *args], check=check, action="systemctl." + args[0])


def _loaded_properties(runner):
    reply = _systemctl(runner, "show", UNIT, "--no-pager",
                       "--property=LoadState,ActiveState,SubState,UnitFileState,FragmentPath,DropInPaths", check=False)
    facts = dict(line.split("=", 1) for line in reply.stdout.splitlines() if "=" in line)
    return reply, facts


def _check_loaded_owner(runner, path, *, allow_missing=False, creating=False, repair=False):
    reply, facts = _loaded_properties(runner)
    if facts.get("DropInPaths"):
        raise ServiceError("unowned_unit_override")
    fragment = facts.get("FragmentPath")
    if fragment and Path(fragment).resolve() != path.resolve():
        raise ServiceError("unowned_loaded_unit")
    if creating and facts.get("LoadState") == "loaded":
        raise ServiceError("unowned_loaded_unit")
    if (repair and not creating and reply.returncode == 0 and fragment
            and facts.get('LoadState') == 'bad-setting' and _read_owned(path) is not None):
        return  # Our exact marked fragment can be repaired; no foreign drop-ins.
    if allow_missing and facts.get("LoadState") == "not-found" and reply.returncode in (0, 1, 4):
        return
    if reply.returncode or facts.get("LoadState") not in {"loaded", "not-found"}:
        raise ServiceError("unit_ownership_unknown", action="systemctl.show", returncode=reply.returncode)
    if ((facts.get("LoadState") == "loaded" and not fragment)
            or (facts.get("LoadState") == "not-found" and not allow_missing)):
        raise ServiceError("unit_ownership_unknown")


def _executable(value):
    found = value if Path(value).is_absolute() else shutil.which(str(value))
    if not found:
        raise ServiceError("executable_unavailable")
    path = _absolute(found)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ServiceError("executable_unavailable")
    return path


def _interpreter(runner, python):
    path = _executable(python or sys.executable)
    code = ("import importlib.metadata as m,json,sys;d=m.distribution('mindie-diagnostics');"
            "u=json.loads(d.read_text('direct_url.json') or '{}');"
            "print(json.dumps({'prefix':sys.prefix,'base':sys.base_prefix,'version':d.version,"
            "'editable':u.get('dir_info',{}).get('editable',False),"
            "'package':str(d.locate_file('mindie_diagnostics'))}))")
    reply = _run(runner, [str(path), "-I", "-c", code], action="interpreter.verify")
    try:
        facts = json.loads(reply.stdout)
        if (facts["prefix"] == facts["base"] or facts["editable"] is not False
                or not Path(facts["package"]).resolve().is_relative_to(Path(facts["prefix"]).resolve())):
            raise ValueError("not an installed venv package")
    except (ValueError, KeyError, TypeError) as exc:
        raise ServiceError("installed_venv_required") from exc
    return path, facts["version"]


def _environment_file(value):
    path = _absolute(value)
    if str(path) != str(path).rstrip() or any(char in str(path) for char in '*?['):
        raise ServiceError('unsupported_environment_path')
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ServiceError("unsafe_environment_file")
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()):
        raise ServiceError("unsafe_environment_file")
    return path


def _reporter_executable(value, environment_file):
    if environment_file is None and any(os.environ.get(key) for key in ("GH_TOKEN", "GITHUB_TOKEN")):
        raise ServiceError("persistent_credentials_required")
    try:
        return str(_executable(value or "gh"))
    except ServiceError:
        if environment_file is not None:
            # The worker's HTTPS adapter can use a private token when no CLI is
            # installed. The service manager stores the file path, never its value.
            return str(value or "gh")
        raise ServiceError("github_authentication_required") from None


@contextmanager
def _locked(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / ".mindie-diagnostics-service.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ServiceError("installation_busy")
                time.sleep(.05)
        yield
    finally:
        os.close(descriptor)


def _publish(path, text, runner):
    if len(text.encode("utf-8")) > MAX_UNIT_BYTES:
        raise ServiceError("unit_too_large")
    # An isolated sibling directory keeps systemd-analyze from loading a broken
    # old unit alongside the candidate. It remains on the same filesystem.
    staging = Path(tempfile.mkdtemp(prefix='.mindie-unit-', dir=path.parent))
    descriptor, temporary = tempfile.mkstemp(prefix="mindie-diagnostics-", suffix=".service", dir=staging)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Parse the exact staged file before replacing a working installation.
        # This does not contact a service manager or execute ExecStart. Some
        # minimal systemd installations omit this optional inspection binary.
        analyzer = shutil.which("systemd-analyze")
        if analyzer:
            checked = _run(runner, [analyzer, "verify", "--man=no", temporary], action="unit.verify")
            if checked.stderr.strip():
                # Invalid non-fatal directives may be ignored with exit zero.
                raise ServiceError('unit_verification_warning', action='unit.verify')
        os.replace(temporary, path)
        return "verified" if analyzer else "unavailable"
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
        staging.rmdir()


def install_service(roots, state, repository, *, python=None, gh=None, grok=None,
                    grok_home=None, grok_work=None, interval=60, since=None,
                    environment_file=None, unit_dir=None, runner=None, start=True, save_token=False, central_bot=False,
                    ensure=False):
    """Install/update the owned unit, preserving the first installation's since.

    ``start=False`` writes/enables it without starting or restarting a process.
    State and source logs are never removed, including during uninstall.
    """
    if sys.platform in {"win32", "darwin"}:
        from .platform_service import install_service as install_native
        return install_native(roots, state, repository, python=python, gh=gh, grok=grok,
                              grok_home=grok_home, grok_work=grok_work, interval=interval, since=since,
                              environment_file=environment_file, unit_dir=unit_dir, runner=runner, start=start,
                              save_token=save_token, central_bot=central_bot, ensure=ensure)
    _linux()
    runner = runner or subprocess.run
    path = _unit_path(unit_dir)
    if ensure and central_bot:
        raise ServiceError('central_bot_is_not_a_local_reporter')
    roots = [_absolute(root) for root in roots]
    if central_bot and roots:
        raise ServiceError("central_bot_does_not_watch_local_logs")
    if not (0 if central_bot else 1) <= len(roots) <= 32:
        raise ServiceError("diagnostic_roots_required")
    if central_bot and not grok:
        raise ServiceError("central_bot_requires_grok")
    state = _absolute(state)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ServiceError("invalid_repository")
    if type(interval) not in (int, float) or not 5 <= interval <= 86400:
        raise ServiceError("invalid_interval")
    if bool(grok) != bool(grok_home and grok_work) or ((grok_home or grok_work) and not grok):
        raise ServiceError("grok_profile_required")
    if save_token and environment_file is not None:
        raise ServiceError("choose_existing_file_or_save_token")
    with _locked(path):
        existing = _read_owned(path)
        if ensure and existing:
            from .service_config import linux_worker_configuration, merged_reporter_options
            old_argv, old_environment = linux_worker_configuration(existing[0])
            retained = merged_reporter_options(old_argv, old_environment, roots, repository)
            roots, state, gh = retained['roots'], Path(retained['state']), retained['gh']
            environment_file = retained['environment_file']
            save_token = False  # Onboarding never rotates another clone's authentication.
            grok, grok_home, grok_work = (retained[key] for key in ('grok', 'grok_home', 'grok_work'))
            interval = retained['interval']
        fixed_since = existing[1]["since"] if existing else _since(since)
        interpreter, version = _interpreter(runner, python)
        _check_loaded_owner(runner, path, allow_missing=True, creating=existing is None,
                            repair=existing is not None)
        if save_token:
            from .platform_service import save_environment_token
            environment_file = save_environment_token(state, runner)
        environment_file = _environment_file(environment_file) if environment_file is not None else None
        gh_path = (str(_executable(gh)) if ensure and existing and environment_file is None
                   else _reporter_executable(gh, environment_file))
        argv = [str(interpreter), "-I", "-m", "mindie_diagnostics.cli", "worker"]
        if central_bot:
            argv.append("--central-bot")
        for root in dict.fromkeys(roots):
            argv += ["--root", str(root)]
        argv += ["--state", str(state), "--repository", repository, "--gh", gh_path,
                 "--since", fixed_since, "--interval", str(interval)]
        if grok:
            argv += ["--grok", str(_executable(grok)), "--grok-home", str(_absolute(grok_home)),
                     "--grok-work", str(_absolute(grok_work))]
        metadata = {"schema": 1, "since": fixed_since}
        text = (MARKER + METADATA + json.dumps(metadata, separators=(",", ":")) + "\n"
                "[Unit]\nDescription=MindIE local diagnostics reporter\nAfter=network-online.target\n"
                "\n[Service]\nType=exec\nExecStart=" + " ".join(map(_quoted, argv)) + "\n"
                "WorkingDirectory=/\n"
                + ("EnvironmentFile=" + str(environment_file).replace('%', '%%') + "\n" if environment_file else "") +
                "UnsetEnvironment=PYTHONPATH PYTHONHOME\nRestart=on-failure\nRestartSec=10\n"
                "TimeoutStopSec=20\nKillMode=control-group\nUMask=0077\n"
                "StandardOutput=journal\nStandardError=journal\nSyslogIdentifier=mindie-diagnostics\n"
                "LogRateLimitIntervalSec=30s\nLogRateLimitBurst=100\n"
                "\n[Install]\nWantedBy=default.target\n")
        changed = existing is None or existing[0] != text
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        verification = _publish(path, text, runner) if changed else "unchanged"
        _systemctl(runner, "daemon-reload")
        _systemctl(runner, "enable", str(path))
        _check_loaded_owner(runner, path)
        if start:
            _systemctl(runner, "restart" if existing and (changed or save_token) else "start", UNIT)
        return {"status": "installed", "unit": str(path), "since": fixed_since,
                "changed": changed, "start_requested": start, "python": str(interpreter),
                "package_version": version, "state": str(state), "unit_verification": verification}


def ensure_reporter_service(roots, state, repository, **options):
    """Add log roots to the owned local reporter, preserving its durable setup.

    Existing roots, state, credential file and optional model profile survive a
    new clone's onboarding. The explicit runtime may change. Repository changes
    and central-bot replacement are refused. Merge and update share one lock.
    """
    return install_service(roots, state, repository, **options, ensure=True)


def service_status(*, unit_dir=None, runner=None):
    if sys.platform in {"win32", "darwin"}:
        from .platform_service import service_status as status_native
        return status_native(unit_dir=unit_dir, runner=runner)
    _linux()
    path = _unit_path(unit_dir)
    existing = _read_owned(path)
    if existing is None:
        return {"status": "absent", "unit": str(path)}
    reply, facts = _loaded_properties(runner or subprocess.run)
    allowed = {key: facts.get(key) for key in ("LoadState", "ActiveState", "SubState", "UnitFileState", "FragmentPath")}
    status = facts.get("ActiveState", "unknown") if reply.returncode == 0 else "unknown"
    return {"status": status, "unit": str(path), "since": existing[1]["since"],
            "systemctl_returncode": reply.returncode, "systemd": allowed}


def start_service(*, unit_dir=None, runner=None):
    """Start an existing owned unit without recreating configuration or since."""
    if sys.platform in {"win32", "darwin"}:
        from .platform_service import start_service as start_native
        return start_native(unit_dir=unit_dir, runner=runner)
    _linux()
    path = _unit_path(unit_dir)
    runner = runner or subprocess.run
    with _locked(path):
        existing = _read_owned(path)
        if existing is None:
            raise ServiceError("service_not_installed")
        _systemctl(runner, "daemon-reload")
        _check_loaded_owner(runner, path)
        _systemctl(runner, "start", UNIT)
        return {"status": "start_requested", "unit": str(path), "since": existing[1]["since"]}


def remove_service(*, unit_dir=None, runner=None):
    if sys.platform in {"win32", "darwin"}:
        from .platform_service import remove_service as remove_native
        return remove_native(unit_dir=unit_dir, runner=runner)
    _linux()
    path = _unit_path(unit_dir)
    runner = runner or subprocess.run
    with _locked(path):
        existing = _read_owned(path)
        if existing is None:
            return {"status": "absent", "unit": str(path)}
        _check_loaded_owner(runner, path, allow_missing=True, repair=True)
        _systemctl(runner, "disable", "--now", UNIT)
        if _read_owned(path) != existing:
            raise ServiceError("unit_changed_during_removal")
        path.unlink()
        _systemctl(runner, "daemon-reload")
        return {"status": "removed", "unit": str(path), "state_preserved": True}
