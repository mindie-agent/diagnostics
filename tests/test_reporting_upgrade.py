"""Fault-boundary regressions; native launchd acceptance is recorded separately."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mindie_diagnostics import reporting_commands as commands, reporting_runtime as runtime, service
from mindie_diagnostics.integration import configure_reporting


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = tmp_path / 'diagnostics.json'
    configure_reporting(True, config=config, repository='owner/repo', roots=[str(tmp_path / 'logs')])
    root = config.with_suffix('.runtime'); root.mkdir(mode=0o700)
    old = {'schema': 1, 'version': '0.3.0', 'revision': None, 'source_hash': 'a' * 64}
    (root / 'current.json').write_text(json.dumps(old))
    state = {'healthy': True, 'runtime_source': old['source_hash'], 'status': 'idle', 'pid': 314159}
    monkeypatch.setattr(commands, '_worker_view', lambda config: dict(state))
    monkeypatch.setattr(commands.shutil, 'which', lambda name: sys.executable)
    monkeypatch.setattr(service, 'service_status', lambda **kwargs: {'status': 'active'})
    monkeypatch.setattr(runtime, '_collect_source', lambda: ('0.4.0', None, [], 'b' * 64))
    monkeypatch.setattr(runtime, '_generation_python', lambda *args: sys.executable)
    native_calls = []
    monkeypatch.setattr(service, 'ensure_reporter_service', lambda *args, **kwargs: native_calls.append(kwargs))
    return config, root, state, native_calls


def test_timed_out_preparation_is_visible_and_same_target_is_not_retried(prepared, monkeypatch):
    config, root, _state, calls = prepared
    preparations = []
    def timeout(*args, **kwargs):
        preparations.append(kwargs['deadline'])
        raise subprocess.TimeoutExpired(['controlled-prepare'], 0.1)
    monkeypatch.setattr(runtime, '_prepare_locked', timeout)
    first = commands.upgrade_running(config)
    second = commands.upgrade_running(config)
    assert first['status'] == 'degraded' and first['rollback'] == 'not_needed'
    assert second['status'] == 'suppressed'
    assert len(preparations) == 1 and calls == []
    record = json.loads((root / ('update-' + 'b' * 64 + '.json')).read_text())
    assert record['status'] == 'failed'


@pytest.mark.parametrize('change', ['stopped', 'new_grant', 'restarted'])
def test_preparation_cannot_revive_stopped_worker_or_cross_consent_revision(prepared, monkeypatch, change):
    config, root, state, calls = prepared
    def prepare(*args, **kwargs):
        if change == 'stopped':
            state['healthy'] = False
        elif change == 'restarted':
            state['pid'] += 1
        else:
            configure_reporting(False, config=config, repository='owner/repo')
            configure_reporting(True, config=config, repository='owner/repo', roots=[str(config.parent / 'logs')])
        return {'python': sys.executable, 'source_hash': 'b' * 64}
    monkeypatch.setattr(runtime, '_prepare_locked', prepare)
    result = commands.upgrade_running(config)
    assert result['status'] == 'skipped' and calls == []
    assert json.loads((root / 'current.json').read_text())['source_hash'] == 'a' * 64


def test_invalid_worker_directory_is_a_failure_not_a_normal_update_skip(prepared, monkeypatch):
    from mindie_diagnostics.integration import _worker_view

    config, root, _state, calls = prepared
    state = config.with_suffix('.state')
    state.write_bytes(b'retained invalid state')
    monkeypatch.setattr(commands, '_worker_view', _worker_view)
    assert commands.upgrade_running(config) == {
        'status': 'degraded', 'reason': 'reporter_worker_unavailable'}
    assert calls == [] and state.read_bytes() == b'retained invalid state'
    assert json.loads((root / 'current.json').read_text())['source_hash'] == 'a' * 64
    assert not list(root.glob('update-*.json'))


@pytest.mark.parametrize('bad_hash', ['../outside', 'g' * 64, 'a' * 65])
def test_current_source_hash_cannot_select_arbitrary_generation(tmp_path, bad_hash):
    (tmp_path / 'current.json').write_text(json.dumps({'schema': 1, 'version': '0.3.0', 'source_hash': bad_hash}))
    with pytest.raises(RuntimeError, match='runtime_untrusted'):
        runtime._read_current(tmp_path)


def test_absolute_subprocess_budget_is_enforced_on_a_real_child():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        commands._bounded_runner(started + 0.1)([sys.executable, '-I', '-c', 'import time;time.sleep(5)'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    assert time.monotonic() - started < 2


def test_installed_git_revision_is_distinct_from_consent_revision(prepared, monkeypatch):
    config, root, state, calls = prepared
    monkeypatch.setattr(runtime, '_collect_source', lambda: ('0.4.0', 'c' * 40, [], 'b' * 64))
    monkeypatch.setattr(runtime, '_prepare_locked', lambda *args, **kwargs: {'python': sys.executable})
    def installed(*args, **kwargs):
        calls.append(kwargs)
        state.update(runtime_source='b' * 64, pid=state['pid'] + 1)
        return {'changed': True}
    monkeypatch.setattr(service, 'ensure_reporter_service', installed)
    result = commands.upgrade_running(config)
    assert result['status'] == 'updated' and len(calls) == 1
    assert json.loads((root / 'current.json').read_text())['revision'] == 'c' * 40


@pytest.mark.parametrize(('candidate', 'expected'), [('0.3.0', 'source_conflict'), ('0.2.0', 'skipped')])
def test_shared_reporter_rejects_conflicting_or_older_source_without_restart(prepared, monkeypatch, candidate, expected):
    config, _root, state, calls = prepared
    monkeypatch.setattr(runtime, '_collect_source', lambda: (candidate, None, [], 'b' * 64))
    def unexpected(*args, **kwargs):
        pytest.fail('conflicting/older source must not prepare a generation')
    monkeypatch.setattr(runtime, '_prepare_locked', unexpected)
    assert commands.upgrade_running(config)['status'] == expected
    assert calls == [] and state['pid'] == 314159


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX lock implementation; Windows native acceptance separate')
@pytest.mark.parametrize('name', ['service', 'platform_service'])
def test_expired_service_lock_budget_is_rejected_even_without_contention(tmp_path, name):
    from mindie_diagnostics import platform_service
    module = service if name == 'service' else platform_service
    with pytest.raises(service.ServiceError, match='installation_busy'):
        with module._locked(tmp_path / 'unit', deadline=time.monotonic() - 1):
            pytest.fail('an already expired operation must not enter the service mutation region')


@pytest.mark.parametrize(('budget', 'offline_seconds'), [(0, 0), (5, 0), (75, 20)])
def test_maintenance_deducts_offline_time_before_deciding_to_replace(prepared, monkeypatch, budget, offline_seconds):
    from mindie_diagnostics import maintenance
    config, root, state, calls = prepared
    clock = [100.0]
    monkeypatch.setattr(commands.time, 'monotonic', lambda: clock[0])
    def prune(*args, **kwargs):
        clock[0] += offline_seconds
        return {'status': 'ok'}
    monkeypatch.setattr(maintenance, 'prune', prune)
    monkeypatch.setattr(commands, 'upgrade_running', lambda *args, **kwargs: pytest.fail('insufficient shared budget must not enter upgrade'))
    result = commands.maintain(config, update_running=True, budget_seconds=budget)
    assert result['reporter_update'] == {'status': 'skipped', 'reason': 'insufficient_budget'}
    assert not list(root.glob('update-*.json')) and calls == [] and state['pid'] == 314159


@pytest.mark.parametrize('budget', [float('nan'), float('inf'), -1, True])
def test_maintenance_rejects_invalid_budget_before_work(prepared, budget):
    config, root, _state, calls = prepared
    with pytest.raises(ValueError, match='invalid_budget'):
        commands.maintain(config, update_running=True, budget_seconds=budget)
    assert not list(root.glob('update-*.json')) and calls == []


def test_committed_runtime_is_not_rolled_back_when_terminal_record_write_fails(prepared, monkeypatch):
    config, root, state, calls = prepared
    monkeypatch.setattr(runtime, '_prepare_locked', lambda *args, **kwargs: {'python': sys.executable})
    def installed(*args, **kwargs):
        calls.append(kwargs)
        state.update(runtime_source='b' * 64, pid=state['pid'] + 1)
        return {'changed': True}
    monkeypatch.setattr(service, 'ensure_reporter_service', installed)
    original = runtime._write_update_record
    def fail_terminal(root, source, record):
        if record['status'] == 'updated':
            raise OSError('controlled terminal-record write failure')
        return original(root, source, record)
    monkeypatch.setattr(runtime, '_write_update_record', fail_terminal)
    result = commands.upgrade_running(config)
    assert result['status'] == 'degraded' and result['rollback'] == 'not_needed'
    assert result['reason'] == 'update_record_write_failed'
    assert len(calls) == 1 and state['runtime_source'] == 'b' * 64
    assert json.loads((root / 'current.json').read_text())['source_hash'] == 'b' * 64
    assert json.loads((root / ('update-' + 'b' * 64 + '.json')).read_text())['status'] == 'attempting'


def test_native_failure_action_and_code_are_preserved_without_raw_message(prepared, monkeypatch):
    config, root, _state, calls = prepared
    monkeypatch.setattr(runtime, '_prepare_locked', lambda *args, **kwargs: {'python': sys.executable})
    def install(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise service.ServiceError('command_failed', action='launchctl.bootstrap', returncode=5)
        return {'changed': True}
    monkeypatch.setattr(service, 'ensure_reporter_service', install)
    result = commands.upgrade_running(config)
    record = json.loads((root / ('update-' + 'b' * 64 + '.json')).read_text())
    for value in (result, record):
        assert value['failure_action'] == 'launchctl.bootstrap' and value['failure_returncode'] == 5
        assert value['rollback'] == 'restored'
