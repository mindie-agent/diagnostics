from contextlib import nullcontext
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

from mindie_diagnostics import service


class Runner:
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = []
        self.editable = False
        self.fail = None
        self.timeout = None
        self.state = "active"
        self.fragment = ""
        self.dropins = ""
        self.verify_failure = False
        self.verify_warning = False

    def __call__(self, argv, **options):
        self.calls.append((argv, options))
        assert options == {"capture_output": True, "text": True, "encoding": "utf-8", "timeout": 30, "check": False}
        if Path(argv[0]).name == "systemd-analyze":
            assert argv[1:3] == ["verify", "--man=no"]
            assert Path(argv[3]).suffix == '.service' and Path(argv[3]).is_file()
            return subprocess.CompletedProcess(argv, int(self.verify_failure), '',
                                               'EnvironmentFile ignored' if self.verify_warning else '')
        if argv[0] != "systemctl":
            assert argv[1:3] == ["-I", "-c"]
            facts = {"prefix": str(self.prefix), "base": "/base-python", "version": "0.1.0",
                     "editable": self.editable, "package": str(self.prefix / "lib/mindie_diagnostics")}
            return subprocess.CompletedProcess(argv, 0, json.dumps(facts), "")
        if argv[2] == self.timeout:
            raise subprocess.TimeoutExpired(argv, 30)
        if argv[2] == self.fail:
            return subprocess.CompletedProcess(argv, 1, "", "private raw failure never copied")
        if argv[2] == "show":
            load = 'loaded' if Path(self.fragment).is_file() else 'not-found'
            return subprocess.CompletedProcess(argv, 0,
                f"LoadState={load}\nActiveState={self.state}\nSubState=running\nUnitFileState=enabled\nFragmentPath={self.fragment}\nDropInPaths={self.dropins}\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def install(tmp_path, monkeypatch):
    # Unit construction is tested on every platform without touching its real
    # service manager. Actual Linux locking gets its own test below.
    monkeypatch.setattr(service.sys, "platform", "linux")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(service, "_locked", lambda path, **kwargs: (path.parent.mkdir(parents=True, exist_ok=True), nullcontext())[1])
    prefix = tmp_path / 'immutable env % $HOME;not-a-shell'
    python = prefix / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture")
    python.chmod(0o700)
    gh = tmp_path / "Windows tools/gh.exe"
    gh.parent.mkdir()
    gh.write_bytes(b"fixture")
    gh.chmod(0o700)
    runner = Runner(prefix)
    values = {"roots": [tmp_path / 'logs %u "quoted" $TOKEN; echo secret'],
              "state": tmp_path / 'state %u $HOME', "repository": "example/repository",
              "python": python, "gh": gh, "unit_dir": tmp_path / "user units", "runner": runner,
              "since": "2026-09-13T01:02:03Z", "reporting_config": tmp_path / "reporting.json"}
    runner.fragment = str(values['unit_dir'] / service.UNIT)
    return values, runner


def test_install_uses_literal_arguments_immutable_python_and_bounded_journal(install):
    values, runner = install
    result = service.install_service(**values)
    text = Path(result["unit"]).read_text()
    assert text.startswith(service.MARKER)
    assert '%%u' in text and '$$TOKEN' in text and '\\"quoted\\"' in text
    assert '"-I" "-m" "mindie_diagnostics.cli" "worker"' in text
    assert 'UnsetEnvironment=PYTHONPATH PYTHONHOME' in text
    assert 'LogRateLimitIntervalSec=30s' in text and 'LogRateLimitBurst=100' in text
    assert 'Restart=no' in text and 'KillMode=control-group' in text
    assert '\nWorkingDirectory=/\n' in text
    assert all(call[0][0] in {str(values['python']), 'systemctl'} or Path(call[0][0]).name == 'systemd-analyze' for call in runner.calls)
    assert [call[0][2] for call in runner.calls if call[0][0] == 'systemctl'] == ['show', 'daemon-reload', 'enable', 'show', 'start']
    assert result['since'] == '2026-09-13T01:02:03Z' or result['since'] == '2026-09-13T01:02:03+00:00'


def test_reinstall_preserves_original_since_and_only_restarts_changed_config(install):
    values, runner = install
    first = service.install_service(**values)
    path = Path(first['unit'])
    previous = path.read_bytes()
    runner.calls.clear()
    second = service.install_service(**{**values, 'since': '2030-01-01T00:00:00Z'})
    assert second['since'] == first['since'] and not second['changed']
    assert path.read_bytes() == previous
    assert runner.calls[-1][0][-2:] == ['start', service.UNIT]
    third = service.install_service(**{**values, 'interval': 120})
    assert third['changed'] and third['since'] == first['since']
    assert runner.calls[-1][0][-2:] == ['restart', service.UNIT]
    previous = path.read_bytes()
    started = service.start_service(unit_dir=values['unit_dir'], runner=runner)
    assert started['since'] == first['since'] and path.read_bytes() == previous


def test_ensure_uses_only_current_authorized_roots_and_rejects_rebinding(install):
    from mindie_diagnostics.service_config import linux_worker_configuration, worker_options
    values, runner = install
    config = values['reporting_config']
    first = service.ensure_reporter_service(**values)
    new_root = values['state'].parent / 'other logs'
    second = service.ensure_reporter_service(**{**values, 'roots': [new_root]})
    options = worker_options(*linux_worker_configuration(Path(second['unit']).read_text()))
    assert options['roots'] == [str(new_root)]
    assert options['reporting_config'] == str(config)
    before = Path(second['unit']).read_bytes()
    with pytest.raises(service.ServiceError, match='reporter_authorization_mismatch'):
        service.ensure_reporter_service(**{**values, 'reporting_config': config.with_name('other.json')})
    assert Path(second['unit']).read_bytes() == before


def test_reporter_requires_gh_even_when_environment_file_exists(install, monkeypatch):
    values, _ = install
    monkeypatch.setattr(shutil, 'which', lambda value: None)
    with pytest.raises(service.ServiceError, match='gh_unavailable'):
        service._reporter_executable('absent-gh', values['state'] / 'credentials.env')


def test_ensure_rejects_legacy_and_repository_rebinding_before_mutation(install):
    values, runner = install
    installed = service.install_service(**values)
    path = Path(installed['unit'])
    original = path.read_text()
    # Existing old-generation file, not a newly supported legacy install path.
    path.write_text(original.replace('"worker"', '"worker" "--central-bot"'))
    before = path.read_bytes()
    runner.calls.clear()
    with pytest.raises(service.ServiceError, match='legacy_worker_requires_explicit_removal'):
        service.ensure_reporter_service(**values)
    assert path.read_bytes() == before and runner.calls == []
    path.write_text(original)
    runner.calls.clear()
    with pytest.raises(service.ServiceError, match='reporter_authorization_mismatch'):
        service.ensure_reporter_service(**{**values, 'repository': 'another/repository'})
    assert path.read_text() == original and runner.calls == []


def test_reinstall_repairs_own_invalid_unit_but_never_foreign_overrides(install):
    values, runner = install
    first = service.install_service(**values)
    path = Path(first['unit'])
    path.write_text(path.read_text().replace('WorkingDirectory=/', 'WorkingDirectory="/invalid"'))
    class InvalidBeforeReload:
        def __init__(self):
            self.reloaded = False
        def __call__(self, argv, **options):
            if argv[:3] == ['systemctl', '--user', 'daemon-reload']:
                self.reloaded = True
            reply = runner(argv, **options)
            if argv[:3] == ['systemctl', '--user', 'show'] and not self.reloaded:
                reply.stdout = reply.stdout.replace('LoadState=loaded', 'LoadState=bad-setting')
            return reply
    result = service.install_service(**{**values, 'runner': InvalidBeforeReload()})
    assert result['changed'] and result['since'] == first['since']
    assert '\nWorkingDirectory=/\n' in path.read_text()
    runner.dropins = '/personal/override.conf'
    with pytest.raises(service.ServiceError, match='unowned_unit_override'):
        service.install_service(**{**values, 'runner': InvalidBeforeReload()})


def test_personal_unit_is_never_overwritten_or_stopped(install):
    values, runner = install
    path = values['unit_dir'] / service.UNIT
    path.parent.mkdir(parents=True)
    original = b'[Service]\nExecStart=/personal/worker\n'
    path.write_bytes(original)
    with pytest.raises(service.ServiceError, match='unowned_unit'):
        service.install_service(**values)
    with pytest.raises(service.ServiceError, match='unowned_unit'):
        service.remove_service(unit_dir=path.parent, runner=runner)
    assert path.read_bytes() == original and runner.calls == []


def test_editable_or_base_python_is_not_installed_in_unit(install):
    values, runner = install
    runner.editable = True
    with pytest.raises(service.ServiceError, match='installed_venv_required'):
        service.install_service(**values)
    assert not (values['unit_dir'] / service.UNIT).exists()
    assert not any(call[0][0] == 'systemctl' for call in runner.calls)


def test_remove_only_owned_unit_preserves_state_and_other_configuration(install):
    values, runner = install
    installed = service.install_service(**values)
    state_file = values['state'] / 'reporter.sqlite3'
    state_file.write_bytes(b'preserved private state')
    other = values['unit_dir'] / 'personal.service'
    other.write_bytes(b'personal configuration')
    runner.calls.clear()
    result = service.remove_service(unit_dir=values['unit_dir'], runner=runner)
    assert result['status'] == 'removed' and result['state_preserved']
    assert not Path(installed['unit']).exists()
    assert state_file.read_bytes() == b'preserved private state'
    assert other.read_bytes() == b'personal configuration'
    assert runner.calls[1][0] == ['systemctl', '--user', 'disable', '--now', service.UNIT]
    assert service.remove_service(unit_dir=values['unit_dir'], runner=runner)['status'] == 'absent'


def test_unknown_stop_result_preserves_unit_without_retry(install):
    values, runner = install
    installed = service.install_service(**values)
    runner.calls.clear()
    runner.timeout = 'disable'
    with pytest.raises(service.ServiceError, match='command_timeout'):
        service.remove_service(unit_dir=values['unit_dir'], runner=runner)
    assert Path(installed['unit']).is_file()
    assert len(runner.calls) == 2


def test_status_reports_observed_state_and_no_cleanup(install):
    values, runner = install
    service.install_service(**values)
    runner.calls.clear()
    runner.state = 'failed'
    result = service.service_status(unit_dir=values['unit_dir'], runner=runner)
    assert result['status'] == 'failed' and len(runner.calls) == 1
    assert runner.calls[0][0][2] == 'show'


def test_missing_policy_is_rejected_before_any_manager_call(install):
    values, runner = install
    with pytest.raises(service.ServiceError, match='reporting_config_required'):
        service.install_service(**{**values, 'reporting_config': None})
    assert runner.calls == []
    assert not values['unit_dir'].exists()


@pytest.mark.parametrize('change', [{'roots': []}, {'roots': ['relative']}, {'state': 'relative'},
                                    {'repository': 'bad\nExecStart=/anything'}, {'since': 'tomorrow'},
                                    {'interval': float('nan')}, {'interval': 0}])
def test_invalid_configuration_does_not_call_systemctl(install, change):
    values, runner = install
    with pytest.raises(service.ServiceError):
        service.install_service(**{**values, **change})
    assert not any(call[0][0] == 'systemctl' for call in runner.calls)


def test_unsupported_platform_is_explicit(monkeypatch):
    monkeypatch.setattr(service.sys, 'platform', 'freebsd')
    for action in (lambda: service.install_service([], '/state', 'example/repository', reporting_config='/reporting.json'),
                   service.service_status, service.remove_service):
        with pytest.raises(service.ServiceError, match='unsupported_platform'):
            action()


@pytest.mark.parametrize('changed', ['fragment', 'dropins'])
def test_loaded_personal_unit_or_dropin_is_not_started_or_stopped(install, changed):
    values, runner = install
    setattr(runner, changed, '/personal/' + ('other.service' if changed == 'fragment' else 'override.conf'))
    with pytest.raises(service.ServiceError, match='unowned_'):
        service.install_service(**values)
    assert not (values['unit_dir'] / service.UNIT).exists()
    assert not any(call[0][2] in {'start', 'restart'} for call in runner.calls if call[0][0] == 'systemctl')


@pytest.mark.skipif(os.name == 'nt', reason='POSIX file permission contract')
def test_environment_file_requires_private_regular_owned_file_without_reading_contents(install, tmp_path, monkeypatch):
    values, runner = install
    secret = tmp_path / 'github.env'
    secret.write_text('GH_TOKEN=fixture-never-copy\nWSLENV=GH_TOKEN/w\n')
    secret.chmod(0o644)
    with pytest.raises(service.ServiceError, match='unsafe_environment_file'):
        service.install_service(**values, environment_file=secret)
    secret.chmod(0o600)
    original_read = Path.read_text
    def guarded_read(path, *args, **kwargs):
        if path == secret:
            raise AssertionError('installer must not read credentials')
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded_read)
    installed = service.install_service(**values, environment_file=secret)
    unit = Path(installed['unit']).read_text()
    assert 'EnvironmentFile=' in unit and 'fixture-never-copy' not in unit
    assert 'EnvironmentFile=' + str(secret).replace('%', '%%') + '\n' in unit
    link = tmp_path / 'linked.env'
    link.symlink_to(secret)
    with pytest.raises(service.ServiceError, match='unsafe_environment_file'):
        service.install_service(**values, environment_file=link)


@pytest.mark.skipif(os.name == 'nt', reason='Linux persistent flock')
def test_install_lock_has_stable_inode_and_no_shared_state_deletion(tmp_path):
    path = tmp_path / service.UNIT
    with service._locked(path):
        inode = (tmp_path / '.mindie-diagnostics-service.lock').stat().st_ino
    with service._locked(path):
        assert (tmp_path / '.mindie-diagnostics-service.lock').stat().st_ino == inode


def test_rejected_staging_unit_preserves_existing_unit_and_never_reloads(install, monkeypatch):
    values, runner = install
    original_which = service.shutil.which
    monkeypatch.setattr(service.shutil, 'which', lambda name: '/test/systemd-analyze' if name == 'systemd-analyze' else original_which(name))
    result = service.install_service(**values)
    assert result['unit_verification'] == 'verified'
    previous = Path(result['unit']).read_bytes()
    runner.calls.clear()
    runner.verify_failure = True
    with pytest.raises(service.ServiceError, match='command_failed') as failure:
        service.install_service(**{**values, 'interval': 120})
    assert failure.value.action == 'unit.verify'
    assert Path(result['unit']).read_bytes() == previous
    assert [call[0][2] for call in runner.calls if call[0][0] == 'systemctl'] == ['show']
    assert not list(values['unit_dir'].glob('.mindie-diagnostics-*.service'))


def test_missing_optional_unit_analyzer_is_reported_without_installing_it(install, monkeypatch):
    values, runner = install
    monkeypatch.setattr(service.shutil, 'which', lambda name: None)
    result = service.install_service(**values)
    assert result['unit_verification'] == 'unavailable'
    assert Path(result['unit']).is_file()


def test_zero_exit_parser_warning_does_not_publish_a_partially_ignored_unit(install, monkeypatch):
    values, runner = install
    monkeypatch.setattr(service.shutil, 'which', lambda name: 'systemd-analyze')
    runner.verify_warning = True
    with pytest.raises(service.ServiceError, match='unit_verification_warning'):
        service.install_service(**values)
    assert not (values['unit_dir'] / service.UNIT).exists()
    assert not list(values['unit_dir'].glob('.mindie-unit-*'))
    assert not any(call[0][:3] == ['systemctl', '--user', 'daemon-reload'] for call in runner.calls)


@pytest.mark.skipif(sys.platform != 'linux', reason='actual Linux systemd unit parser')
def test_generated_unit_is_accepted_by_actual_systemd_analyze_and_old_workdir_rejected(install, tmp_path):
    analyzer = shutil.which('systemd-analyze')
    if analyzer is None:
        pytest.skip('systemd-analyze is not installed on this Linux platform')
    values, stub = install
    # The verifier requires an executable, but does not execute it. Keep the
    # actual service manager stubbed; only the static unit parser is real.
    values['python'] = Path(sys.executable)
    values['gh'] = Path(sys.executable)
    environment = tmp_path / 'github % $HOME.env'
    environment.write_text('FIXTURE=value\n')
    environment.chmod(0o600)
    values['environment_file'] = environment
    replies = []
    def runner(argv, **options):
        if Path(argv[0]).name == 'systemd-analyze':
            reply = subprocess.run(argv, **options)
            replies.append(reply)
            assert reply.returncode == 0, reply.stderr
            return reply
        return stub(argv, **options)
    values['runner'] = runner
    result = service.install_service(**values)
    assert result['unit_verification'] == 'verified' and len(replies) == 1
    text = Path(result['unit']).read_text()
    invalid = tmp_path / 'old-workdir.service'
    invalid.write_text(text.replace('WorkingDirectory=/', 'WorkingDirectory="/tmp/old state"'))
    old = subprocess.run([analyzer, 'verify', '--man=no', str(invalid)],
                         capture_output=True, text=True, timeout=30, check=False)
    assert old.returncode != 0
    assert 'WorkingDirectory' in old.stderr and 'not absolute' in old.stderr
    invalid_environment = tmp_path / 'old-env.service'
    directive = 'EnvironmentFile=' + str(environment).replace('%', '%%')
    invalid_environment.write_text(text.replace(directive, 'EnvironmentFile="' + directive.split('=',1)[1] + '"'))
    ignored = subprocess.run([analyzer, 'verify', '--man=no', str(invalid_environment)],
                             capture_output=True, text=True, timeout=30, check=False)
    assert ignored.returncode == 0 and 'EnvironmentFile' in ignored.stderr and 'not absolute' in ignored.stderr
