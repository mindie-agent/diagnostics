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


def test_timed_out_preparation_is_visible_and_allows_idempotent_retry(prepared, monkeypatch):
    assert commands._UPDATE_BUDGET_S == 180.0
    assert commands._UPDATE_BUDGET_S - commands._ROLLBACK_RESERVE_S >= 160.0
    config, root, state, calls = prepared
    preparations = []
    started = time.monotonic()
    def flaky_prepare(*args, **kwargs):
        preparations.append(kwargs['deadline'])
        if len(preparations) == 1:
            raise subprocess.TimeoutExpired(['controlled-prepare'], 0.1)
        return {'python': sys.executable, 'source_hash': 'b' * 64}
    def installed(*args, **kwargs):
        calls.append(kwargs)
        state.update(runtime_source='b' * 64, pid=state['pid'] + 1)
        return {'changed': True}
    monkeypatch.setattr(runtime, '_prepare_locked', flaky_prepare)
    monkeypatch.setattr(service, 'ensure_reporter_service', installed)
    first = commands.upgrade_running(config)
    assert first['status'] == 'degraded' and first['rollback'] == 'not_needed'
    assert len(preparations) == 1 and calls == []
    assert preparations[0] - started >= 150.0
    assert not list(root.glob('update-*.json'))
    second = commands.upgrade_running(config)
    assert second['status'] == 'updated' and len(preparations) == 2 and len(calls) == 1


@pytest.mark.parametrize('change', ['stopped', 'new_grant'])
def test_preparation_cannot_revive_stopped_worker_or_cross_consent_revision(prepared, monkeypatch, change):
    config, root, state, calls = prepared
    def prepare(*args, **kwargs):
        if change == 'stopped':
            state['healthy'] = False
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


def test_native_failure_action_and_code_are_preserved_without_raw_message(prepared, monkeypatch):
    config, _root, _state, calls = prepared
    monkeypatch.setattr(runtime, '_prepare_locked', lambda *args, **kwargs: {'python': sys.executable})
    def install(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise service.ServiceError('command_failed', action='launchctl.bootstrap', returncode=5)
        return {'changed': True}
    monkeypatch.setattr(service, 'ensure_reporter_service', install)
    result = commands.upgrade_running(config)
    assert result['failure_action'] == 'launchctl.bootstrap' and result['failure_returncode'] == 5
    assert result['rollback'] == 'restored'


def test_collect_source_reads_real_distribution_tree_and_rejects_symlinks(tmp_path, monkeypatch):
    import importlib.metadata as metadata
    import mindie_diagnostics

    site = tmp_path / "site-packages"
    pkg_dir = site / "mindie_diagnostics"
    dist_info = site / "mindie_diagnostics-0.4.2.dist-info"
    pycache = pkg_dir / "__pycache__"
    pkg_dir.mkdir(parents=True, mode=0o700)
    dist_info.mkdir(parents=True, mode=0o700)
    pycache.mkdir(parents=True, mode=0o700)

    init_file = pkg_dir / "__init__.py"
    init_file.write_text('__version__ = "0.4.2"\n', encoding="utf-8")
    (pkg_dir / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    (pycache / "helper.cpython-313.pyc").write_bytes(b"\x00\x01\x02\x03")

    commit = "d" * 40
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: mindie-diagnostics\nVersion: 0.4.2\n", encoding="utf-8")
    (dist_info / "REQUESTED").write_bytes(b"")
    (dist_info / "direct_url.json").write_text(
        json.dumps({"url": "https://github.com/mindie-agent/diagnostics.git", "vcs_info": {"vcs": "git", "commit_id": commit}}),
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text(
        "mindie_diagnostics/__init__.py,,\n"
        "mindie_diagnostics/helper.py,,\n"
        "mindie_diagnostics/__pycache__/helper.cpython-313.pyc,,\n"
        "mindie_diagnostics-0.4.2.dist-info/METADATA,,\n"
        "mindie_diagnostics-0.4.2.dist-info/REQUESTED,,\n"
        "mindie_diagnostics-0.4.2.dist-info/direct_url.json,,\n"
        "mindie_diagnostics-0.4.2.dist-info/RECORD,,\n",
        encoding="utf-8",
    )

    real_dist = metadata.PathDistribution(dist_info)
    monkeypatch.setattr(metadata, "distribution", lambda name: real_dist)
    monkeypatch.setattr(mindie_diagnostics, "__file__", str(init_file))
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))

    version, revision, files, source_hash = runtime._collect_source()
    assert version == "0.4.2"
    assert revision == commit
    assert runtime._is_source_hash(source_hash)
    keys = {k for k, _ in files}
    assert "mindie_diagnostics/__init__.py" in keys
    assert "mindie_diagnostics/helper.py" in keys
    assert "mindie_diagnostics-0.4.2.dist-info/REQUESTED" in keys
    assert not any("__pycache__" in k or k.endswith(".pyc") for k in keys)

    # Replacing a package file with a symlink must be rejected by real _collect_source
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 99\n", encoding="utf-8")
    (pkg_dir / "helper.py").unlink()
    (pkg_dir / "helper.py").symlink_to(outside)
    with pytest.raises(RuntimeError, match="source_rejected"):
        runtime._collect_source()


def test_real_generation_atomic_pointer_switch_and_retire_other_generations(tmp_path):
    config = tmp_path / "diagnostics.json"
    configure_reporting(True, config=config, repository="owner/repo", roots=[str(tmp_path / "logs")])

    def make_source(version, rev_char):
        files = [
            ("mindie_diagnostics/__init__.py", f'__version__ = "{version}"\n'.encode()),
            (
                f"mindie_diagnostics-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: mindie-diagnostics\nVersion: {version}\n".encode(),
            ),
            (
                f"mindie_diagnostics-{version}.dist-info/RECORD",
                f"mindie_diagnostics/__init__.py,,\nmindie_diagnostics-{version}.dist-info/METADATA,,\n".encode(),
            ),
        ]
        import hashlib
        digest = hashlib.sha256()
        digest.update(version.encode() + b"\0" + (rev_char * 40).encode() + b"\0")
        digest.update(b"mindie_diagnostics/__init__.py\0" + files[0][1] + b"\0")
        return version, rev_char * 40, files, digest.hexdigest()

    src_v1 = make_source("0.3.0", "1")
    src_v2 = make_source("0.4.0", "2")

    with runtime.runtime_transaction(config) as root:
        first = runtime._prepare_locked(config, publish=True, source=src_v1)
        assert first["published"] is True and first["changed"] is True
        current = runtime._read_current(root)
        assert current["source_hash"] == src_v1[3] and current["version"] == "0.3.0"
        assert (root / "generations" / src_v1[3] / "source.json").is_file()

        # Prepare v2 with publish=False: v2 generation is materialized & verified,
        # but current.json still points to v1 until explicitly published.
        second = runtime._prepare_locked(config, publish=False, source=src_v2)
        assert second["published"] is False and second["changed"] is True
        assert runtime._read_current(root)["source_hash"] == src_v1[3]
        assert (root / "generations" / src_v2[3] / "source.json").is_file()

        # Atomically switch current.json pointer and retire superseded generations.
        runtime._publish(root, {"schema": 1, "source_hash": src_v2[3], "version": src_v2[0], "revision": src_v2[1]})
        assert runtime._read_current(root)["source_hash"] == src_v2[3]
        retired = runtime.retire_other_generations(root, src_v2[3])
        assert retired == [src_v1[3]]
        assert not (root / "generations" / src_v1[3]).exists()
        assert (root / "generations" / src_v2[3]).is_dir()

