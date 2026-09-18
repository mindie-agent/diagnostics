"""Native service manager contracts; real Windows activation has separate evidence."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
from unittest import mock

import pytest

from mindie_diagnostics import platform_service as native, service


class Runner:
    def __init__(self, prefix):
        self.prefix, self.calls, self.xml = prefix, [], None
        self.running = False
        self.launch = None
        self.fail = None

    def __call__(self, argv, **options):
        self.calls.append((argv, options))
        code, output, error = 0, "", ""
        if argv[0] == "powershell.exe":
            script = base64.b64decode(argv[-1]).decode("utf-16-le")
            if self.fail and self.fail in script:
                code = 1
            elif "Get-ScheduledTask -ErrorAction" in script:
                output = json.dumps(None if self.xml is None else {
                    "xml": self.xml, "state": "Running" if self.running else "Ready", "last_result": 0})
            elif "Register-ScheduledTask -TaskName" in script:
                path = script.split("ReadAllText('", 1)[1].split("'))", 1)[0].replace("''", "'")
                self.xml = Path(path).read_text(encoding="utf-8")
            elif "Unregister-ScheduledTask" in script:
                self.xml, self.running = None, False
            elif "Start-ScheduledTask" in script:
                self.running = True
            elif "Stop-ScheduledTask" in script:
                self.running = False
            elif "::GetCurrent().User.Value" in script:
                output = "S-1-5-21-1000"
        elif argv[0] == "launchctl":
            if argv[1] == "print":
                code = int(self.launch is None)
                if code:
                    error = "Could not find service in domain"
                if self.launch:
                    output = "state = running\nprogram = " + self.launch[0] + "\n" + "\n".join(self.launch[1:])
            elif argv[1] == "bootstrap":
                self.launch = plistlib.loads(Path(argv[-1]).read_bytes())["ProgramArguments"]
            elif argv[1] == "bootout":
                self.launch = None
        else:
            assert argv[1:3] == ["-I", "-c"]
            output = json.dumps({"prefix": str(self.prefix), "base": str(self.prefix.parent / "base"),
                                 "version": "0.2.0", "editable": False, "package": str(self.prefix / "lib/mindie_diagnostics")})
        return subprocess.CompletedProcess(argv, code, output, error or ("private fixture must never appear in errors" if code else ""))


@pytest.fixture(params=["win32", "darwin"])
def installation(request, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", request.param)
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    if request.param == "darwin" and os.name == "nt":
        # Real POSIX ownership/mode enforcement runs on Linux/macOS; Windows
        # executes this launchd command fixture without emulating POSIX stat.
        monkeypatch.setattr(service, "_environment_file", lambda value: Path(value))
    prefix = tmp_path / "Python env with spaces"
    prefix.mkdir()
    python = prefix / "python.exe"
    for path in (python, prefix / "pythonw.exe", prefix / "gh.exe"):
        path.write_bytes(b"fixture executable")
        path.chmod(0o700)
    runner = Runner(prefix)
    values = {"roots": [tmp_path / "logs $fixture"], "state": tmp_path / "state with spaces",
              "repository": "owner/project", "python": python, "gh": prefix / "gh.exe",
              "unit_dir": tmp_path / "user service", "runner": runner, "since": "2026-09-14T01:02:03Z"}
    return values, runner


def scripts(runner):
    return [base64.b64decode(argv[-1]).decode("utf-16-le") for argv, _ in runner.calls if argv[0] == "powershell.exe"]


def test_install_reuse_status_start_and_remove_preserve_data(installation):
    values, runner = installation
    first = service.install_service(**values)
    assert first["status"] == "installed" and first["start_requested"] is True
    assert first["platform"] == sys.platform
    manifest = Path(first["manifest"])
    before = manifest.read_bytes()
    data = values["state"] / "keep.sqlite3"
    data.write_bytes(b"do not delete")
    second = service.install_service(**{**values, "since": "2040-01-01T00:00:00Z"})
    assert second["changed"] is False and second["since"] == first["since"]
    assert manifest.read_bytes() == before
    assert service.service_status(unit_dir=values["unit_dir"], runner=runner)["status"] == "active"
    assert service.start_service(unit_dir=values["unit_dir"], runner=runner)["status"] == "start_requested"
    assert service.remove_service(unit_dir=values["unit_dir"], runner=runner)["state_preserved"] is True
    assert data.read_bytes() == b"do not delete"
    assert service.service_status(unit_dir=values["unit_dir"], runner=runner)["status"] == "absent"


@pytest.mark.parametrize('token_only', [False, True])
def test_ensure_preserves_two_clones_roots_state_and_existing_authentication(installation, monkeypatch, token_only):
    from mindie_diagnostics.service_config import worker_options
    values, runner = installation
    if token_only:
        monkeypatch.setenv('GH_TOKEN', 'initial-private-fixture')
        monkeypatch.setattr(service.shutil, 'which', lambda value: None)
        values = {**values, 'gh': None}
    first = service.ensure_reporter_service(**values, save_token=token_only)
    config = json.loads(Path(first['manifest']).read_text())
    credential = Path(config['environment_file']) if token_only else None
    before = credential.read_bytes() if credential else None
    state_file = values['state'] / 'pending.sqlite3'
    state_file.write_bytes(b'existing queue must survive')
    second_root = values['state'].parent / 'another clone logs'
    unused_state = values['state'].parent / 'unneeded clone state'
    monkeypatch.setenv('GH_TOKEN', 'different-clone-private-fixture')
    second_values = {**values, 'roots': [second_root], 'state': unused_state, 'save_token': True}
    second = service.ensure_reporter_service(**second_values)
    current = json.loads(Path(second['manifest']).read_text())
    options = worker_options(current['argv'], current['environment_file'])
    assert options['roots'] == [str(values['roots'][0]), str(second_root)]
    assert options['state'] == str(values['state'])
    assert options['environment_file'] == config['environment_file'] and current['since'] == config['since']
    assert options['gh'] == ('gh' if token_only else str(values['gh']))
    assert not unused_state.exists() and state_file.read_bytes() == b'existing queue must survive'
    if credential:
        assert credential.read_bytes() == before
    monkeypatch.delenv('GH_TOKEN')
    runner.calls.clear()
    third = service.ensure_reporter_service(**{**second_values, 'save_token': False})
    assert not third['changed']
    assert not any('Stop-ScheduledTask' in script or 'Register-ScheduledTask' in script for script in scripts(runner))
    assert not any(argv[:2] == ['launchctl', 'bootout'] for argv, _ in runner.calls)


def test_ensure_preserves_optional_grok_and_refuses_central_or_repository_replacement(installation):
    from mindie_diagnostics.service_config import worker_options
    values, runner = installation
    profile = dict(grok=values['gh'], grok_home=values['state'] / 'grok', grok_work=values['state'] / 'work')
    first = service.install_service(**values, **profile)
    second = service.ensure_reporter_service(**values)
    assert not second['changed']
    manifest = Path(first['manifest'])
    before = manifest.read_bytes()
    with pytest.raises(service.ServiceError, match='reporter_repository_mismatch'):
        service.ensure_reporter_service(**{**values, 'repository': 'different/project'})
    assert manifest.read_bytes() == before
    service.install_service(**{**values, 'roots': [], 'central_bot': True}, **profile)
    before = manifest.read_bytes()
    runner.calls.clear()
    with pytest.raises(service.ServiceError, match='central_bot_is_not_a_local_reporter'):
        service.ensure_reporter_service(**values)
    assert manifest.read_bytes() == before and runner.calls == []


def test_manager_configuration_is_hidden_bounded_and_shell_free(installation):
    values, runner = installation
    result = service.install_service(**values)
    config = json.loads(Path(result["manifest"]).read_text())
    assert config["launch_args"][:4] == ["-I", "-m", "mindie_diagnostics.platform_service", "run"]
    assert config["environment_file"] is None
    if sys.platform == "win32":
        text = Path(result["unit"]).read_text()
        assert not text.startswith("<?xml")  # TaskScheduler receives a .NET string, not UTF-8 bytes.
        assert "pythonw.exe" in text and "<Hidden>true</Hidden>" in text
        assert "<LogonType>InteractiveToken</LogonType>" in text
        assert "<RunLevel>LeastPrivilege</RunLevel>" in text
        assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in text
        assert "<RestartOnFailure>" in text and "<MultipleInstancesPolicy>IgnoreNew" in text
        assert all("-WindowStyle" in argv for argv, _ in runner.calls if argv[0] == "powershell.exe")
    else:
        plist = plistlib.loads(Path(result["unit"]).read_bytes())
        assert plist["KeepAlive"] == {"SuccessfulExit": False} and plist["ThrottleInterval"] == 10
        assert plist["StandardOutPath"] == plist["StandardErrorPath"] == "/dev/null"
        assert "EnvironmentVariables" not in plist


def test_manager_receives_physical_log_and_state_paths_on_first_creation(installation, monkeypatch):
    values, runner = installation
    logical_root, logical_state = values['roots'][0], values['state']
    mapping = {logical_root: logical_root.with_name('physical logs'),
               logical_state: logical_state.with_name('physical state')}
    original = native._physical
    def physical(path):
        path = Path(path)
        if path in mapping:
            assert path.is_dir()  # Resolve only after MSIX has materialized the directory.
            mapping[path].mkdir(exist_ok=True)
            return mapping[path]
        return original(path)
    monkeypatch.setattr(native, '_physical', physical)
    result = service.ensure_reporter_service(**values)
    config = json.loads(Path(result['manifest']).read_text())
    assert config['argv'][config['argv'].index('--root') + 1] == str(mapping[logical_root])
    assert config['argv'][config['argv'].index('--state') + 1] == str(mapping[logical_state])
    assert config['state'] == str(mapping[logical_state])
    assert service.ensure_reporter_service(**values)['changed'] is False


def test_foreign_manifest_or_unit_never_overwritten(installation):
    values, runner = installation
    manifest, _, unit = native._paths(values["unit_dir"])
    manifest.parent.mkdir()
    manifest.write_text('{"personal":"do not modify"}')
    with pytest.raises(service.ServiceError, match="unowned_unit"):
        service.install_service(**values)
    assert manifest.read_text() == '{"personal":"do not modify"}'
    manifest.unlink()
    unit.write_bytes(b"foreign service")
    with pytest.raises(service.ServiceError, match="unowned_unit"):
        service.install_service(**values)
    assert unit.read_bytes() == b"foreign service"


def test_foreign_loaded_manager_entry_is_not_stopped_or_replaced(installation):
    values, runner = installation
    if sys.platform == "win32":
        runner.xml = "<Task><Description>personal</Description></Task>"
    else:
        runner.launch = ["/personal/python", "private-task"]
    with pytest.raises(service.ServiceError, match="unowned_loaded_unit"):
        service.install_service(**values)
    assert not any("Register-ScheduledTask" in script or "Stop-ScheduledTask" in script for script in scripts(runner))
    assert not any(argv[:2] in [["launchctl", "bootout"], ["launchctl", "bootstrap"]] for argv, _ in runner.calls)


def test_no_start_registers_without_launching_worker(installation):
    values, runner = installation
    result = service.install_service(**values, start=False)
    assert result["start_requested"] is False
    assert not any("Start-ScheduledTask" in script for script in scripts(runner))
    assert not any(argv[:2] in [["launchctl", "bootstrap"], ["launchctl", "kickstart"]] for argv, _ in runner.calls)


def test_changed_configuration_preserves_since_and_updates_manager(installation):
    values, runner = installation
    first = service.install_service(**values)
    second = service.install_service(**{**values, "interval": 120})
    assert second["changed"] is True and second["since"] == first["since"]
    config = json.loads(Path(second["manifest"]).read_text())
    assert config["argv"][-2:] == ["--interval", "120"]


def test_temporary_token_requires_explicit_persistent_action(installation, monkeypatch):
    values, runner = installation
    monkeypatch.setenv("GH_TOKEN", "private-temporary-fixture")
    with pytest.raises(service.ServiceError, match="persistent_credentials_required"):
        service.install_service(**values)
    assert not any("Register-ScheduledTask" in script for script in scripts(runner))


def test_explicit_save_token_never_enters_manager_configuration_or_result(installation, monkeypatch):
    values, runner = installation
    secret = "private-token-saving-fixture"
    monkeypatch.setenv("GH_TOKEN", secret)
    result = service.install_service(**values, save_token=True)
    config = json.loads(Path(result["manifest"]).read_text())
    credentials = Path(config["environment_file"])
    assert credentials.read_text() == "GH_TOKEN=" + secret + "\n"
    assert secret not in json.dumps(result) + Path(result["manifest"]).read_text() + Path(result["unit"]).read_text()
    assert all(secret not in script for script in scripts(runner))
    if sys.platform == "win32":
        assert any("SetAccessRuleProtection($true,$false)" in script for script in scripts(runner))
        assert all(".SetOwner(" not in script for script in scripts(runner))  # DACL change must not require WRITE_OWNER.
    elif os.name != "nt":
        assert credentials.stat().st_mode & 0o777 == 0o600


def test_existing_private_file_is_not_read_by_installer(installation, monkeypatch):
    values, runner = installation
    credentials = values["state"].parent / "private.env"
    credentials.write_text("GH_TOKEN=private-no-read-fixture\n")
    credentials.chmod(0o600)
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if path == credentials:
            raise AssertionError("installer must not read credential values")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    result = service.install_service(**values, environment_file=credentials)
    assert result["status"] == "installed"


def test_worker_loads_private_file_in_process_and_drops_inherited_token(installation, monkeypatch):
    values, runner = installation
    credentials = values["state"].parent / "worker.env"
    credentials.write_text("GH_TOKEN=owned-service-fixture\n")
    credentials.chmod(0o600)
    result = service.install_service(**values, environment_file=credentials)
    monkeypatch.setenv("GH_TOKEN", "old-inherited-fixture")
    monkeypatch.setenv("GITHUB_TOKEN", "wrong-inherited-fixture")
    monkeypatch.setattr(native, "_private_environment", lambda value, run: Path(value))
    from mindie_diagnostics import cli
    def worker(argv):
        assert argv[0] == "worker"
        assert os.environ["GH_TOKEN"] == "owned-service-fixture"
        assert "GITHUB_TOKEN" not in os.environ
        return 0
    monkeypatch.setattr(cli, "main", worker)
    assert native.run_worker(result["manifest"]) == 0


def test_replaced_unit_is_preserved_during_removal(installation):
    values, runner = installation
    result = service.install_service(**values)
    unit = Path(result["unit"])
    unit.write_bytes(b"personal replacement")
    with pytest.raises(service.ServiceError, match="unowned_unit"):
        service.remove_service(unit_dir=values["unit_dir"], runner=runner)
    with pytest.raises(service.ServiceError, match="unowned_unit"):
        service.start_service(unit_dir=values["unit_dir"], runner=runner)
    assert unit.read_bytes() == b"personal replacement"


def test_isolated_profiles_have_distinct_manager_names(installation):
    values, _ = installation
    first = native._paths(values["unit_dir"])
    second = native._paths(values["unit_dir"].with_name("second profile"))
    assert first[1] != second[1]


def test_central_bot_service_requires_grok_and_accepts_no_log_roots(installation):
    values, runner = installation
    with pytest.raises(service.ServiceError, match="central_bot_does_not_watch_local_logs"):
        service.install_service(**{**values, "central_bot": True})
    with pytest.raises(service.ServiceError, match="central_bot_requires_grok"):
        service.install_service(**{**values, "roots": [], "central_bot": True})
    result = service.install_service(**{**values, "roots": [], "central_bot": True,
                                        "grok": values["gh"], "grok_home": values["state"] / "bot-home",
                                        "grok_work": values["state"] / "bot-work"})
    config = json.loads(Path(result["manifest"]).read_text())
    assert config["argv"][:2] == ["worker", "--central-bot"]
    assert "--root" not in config["argv"]
