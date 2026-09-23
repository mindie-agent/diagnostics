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
                code = 113 if self.launch is None else 0
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
              "unit_dir": tmp_path / "user service", "runner": runner, "since": "2026-09-14T01:02:03Z",
              "reporting_config": tmp_path / "reporting.json"}
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


def test_reporter_policy_roundtrip_and_binding_protection(installation):
    from mindie_diagnostics.service_config import worker_options
    values, runner = installation
    config = values['reporting_config']
    first = service.ensure_reporter_service(**values)
    same = service.ensure_reporter_service(**values)
    assert not same['changed']
    manifest = Path(first['manifest'])
    current = json.loads(manifest.read_text())
    options = worker_options(current['argv'], current['environment_file'])
    assert options['reporting_config'] == str(config)
    assert current['environment_file'] is None
    before = manifest.read_bytes()
    with pytest.raises(service.ServiceError, match='reporter_authorization_mismatch'):
        service.ensure_reporter_service(**{**values, 'reporting_config': config.with_name('other.json')})
    assert manifest.read_bytes() == before


def test_failed_mac_bootout_preserves_old_descriptions(installation):
    values, runner = installation
    if sys.platform != 'darwin':
        pytest.skip('macOS descriptor transaction only')
    first = service.ensure_reporter_service(**values)
    unit, manifest = Path(first['unit']), Path(first['manifest'])
    before = unit.read_bytes(), manifest.read_bytes()
    def fail_bootout(argv, **options):
        if argv[:2] == ['launchctl', 'bootout']:
            runner.calls.append((argv, options))
            return subprocess.CompletedProcess(argv, 112, '', 'controlled failure')
        return runner(argv, **options)
    with pytest.raises(service.ServiceError, match='command_failed'):
        service.ensure_reporter_service(**{**values, 'interval': 61, 'runner': fail_bootout})
    assert (unit.read_bytes(), manifest.read_bytes()) == before
    assert runner.launch is not None


def test_partial_mac_descriptor_write_restores_verified_pair_without_starting(installation, monkeypatch):
    values, runner = installation
    if sys.platform != 'darwin':
        pytest.skip('macOS descriptor transaction only')
    first = service.ensure_reporter_service(**values)
    unit, manifest = Path(first['unit']), Path(first['manifest'])
    before = unit.read_bytes(), manifest.read_bytes()
    original, failed = native._write, []
    def fail_manifest_once(path, payload):
        if path == manifest and not failed:
            failed.append(True)
            raise OSError('controlled manifest write failure')
        return original(path, payload)
    monkeypatch.setattr(native, '_write', fail_manifest_once)
    runner.calls.clear()
    with pytest.raises((OSError, service.ServiceError)):
        service.ensure_reporter_service(**{**values, 'interval': 61})
    assert (unit.read_bytes(), manifest.read_bytes()) == before
    assert runner.launch is None
    actions = [argv[1] for argv, _ in runner.calls if argv[0] == 'launchctl']
    assert actions.count('bootout') == 1 and 'bootstrap' not in actions and 'kickstart' not in actions


def test_consent_withdrawn_during_interpreter_probe_prevents_native_side_effect(installation):
    from mindie_diagnostics.integration import configure_reporting
    from mindie_diagnostics.fallback import read_policy
    values, runner = installation
    configure_reporting(True, config=values['reporting_config'], repository=values['repository'],
                        roots=[str(path) for path in values['roots']])
    revision = read_policy(values['reporting_config'])['revision']
    def withdraw_during_probe(argv, **options):
        result = runner(argv, **options)
        if argv[0] not in ('launchctl', 'powershell.exe'):
            configure_reporting(False, config=values['reporting_config'], repository=values['repository'])
        return result
    with pytest.raises(service.ServiceError, match='reporting_authorization_changed'):
        service.ensure_reporter_service(**{**values, 'runner': withdraw_during_probe,
                                          'expected_consent_revision': revision})
    assert runner.launch is None and runner.xml is None
    assert not (values['unit_dir'] / native.MANIFEST).exists()


def test_consent_withdrawn_between_native_commands_prevents_next_start(installation):
    from mindie_diagnostics.integration import configure_reporting
    from mindie_diagnostics.fallback import read_policy
    values, runner = installation
    service.ensure_reporter_service(**values)
    configure_reporting(True, config=values['reporting_config'], repository=values['repository'],
                        roots=[str(path) for path in values['roots']])
    revision = read_policy(values['reporting_config'])['revision']
    withdrawn = []
    after = []
    def withdraw_after_command(argv, **options):
        text = base64.b64decode(argv[-1]).decode('utf-16-le') if argv[0] == 'powershell.exe' else ' '.join(argv[:2])
        if withdrawn:
            after.append(text)
        result = runner(argv, **options)
        if text == 'launchctl bootstrap' or 'Stop-ScheduledTask -TaskName' in text:
            configure_reporting(False, config=values['reporting_config'], repository=values['repository'])
            withdrawn.append(True)
        return result
    with pytest.raises(service.ServiceError, match='reporting_authorization_changed'):
        service.ensure_reporter_service(**{**values, 'interval': 61, 'runner': withdraw_after_command,
                                          'expected_consent_revision': revision})
    assert withdrawn
    assert not any(text == 'launchctl kickstart' or 'Start-ScheduledTask -TaskName' in text for text in after)


@pytest.mark.parametrize('legacy_flag', ['--grok', '--central-bot'])
def test_reporter_refuses_legacy_model_worker(installation, legacy_flag):
    values, runner = installation
    first = service.install_service(**values)
    manifest = Path(first['manifest'])
    doc = json.loads(manifest.read_text())
    doc['argv'].append(legacy_flag)
    if legacy_flag == '--grok':
        doc['argv'].append(str(values['gh']))
    manifest.write_text(json.dumps(doc))
    before = manifest.read_bytes()
    runner.calls.clear()
    with pytest.raises(service.ServiceError, match='legacy_worker'):
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
        assert "<RestartOnFailure>" not in text and "<MultipleInstancesPolicy>IgnoreNew" in text
        assert all("-WindowStyle" in argv for argv, _ in runner.calls if argv[0] == "powershell.exe")
    else:
        plist = plistlib.loads(Path(result["unit"]).read_bytes())
        assert "KeepAlive" not in plist
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


def test_service_uses_gh_login_without_copying_temporary_token(installation, monkeypatch):
    values, runner = installation
    monkeypatch.setenv("GH_TOKEN", "private-temporary-fixture")
    result = service.install_service(**values)
    manifest = json.loads(Path(result['manifest']).read_text())
    assert manifest['environment_file'] is None
    assert 'private-temporary-fixture' not in Path(result['manifest']).read_text()
    assert not (values['state'] / 'credentials.env').exists()


def test_removed_token_copy_option_cannot_write_credentials(installation, monkeypatch):
    values, runner = installation
    monkeypatch.setenv('GH_TOKEN', 'private-token-saving-fixture')
    with pytest.raises(TypeError, match='save_token'):
        service.install_service(**values, save_token=True)
    assert runner.calls == []
    assert not values['state'].exists()


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


def test_missing_policy_rejected_without_manager_mutation(installation):
    values, runner = installation
    with pytest.raises(service.ServiceError, match='reporting_config_required'):
        service.install_service(**{**values, 'reporting_config': None})
    assert runner.calls == []
    assert not values['unit_dir'].exists()
