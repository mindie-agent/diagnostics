import os
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mindie_diagnostics.maintenance import prune
from mindie_diagnostics import maintenance


def test_retention_preserves_fresh_and_unrelated_files(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, '_writer_exited', lambda pid: True)
    folder = tmp_path / 'events' / 'test'
    folder.mkdir(parents=True)
    old = folder / ('1-' + 'a' * 32 + '.jsonl')
    old.write_text('old')
    os.utime(old, (1, 1))
    fresh = folder / ('2-' + 'b' * 32 + '.jsonl')
    fresh.write_text('fresh')
    unrelated = folder / 'state.db'
    unrelated.write_text('preserve')
    result = prune(tmp_path, max_bytes=1)
    assert not old.exists()
    assert fresh.exists() and unrelated.exists()
    assert result['removed_files'] == 1 and result['limited']


def test_worker_retention_preserves_unread_old_evidence(tmp_path, monkeypatch):
    # This test exercises cursor protection after the writer has exited.
    monkeypatch.setattr(maintenance, '_writer_exited', lambda pid: True)
    from mindie_diagnostics import configure
    from mindie_diagnostics.ingestion import ingest
    from mindie_diagnostics.outbox import Outbox
    from pathlib import Path

    recorder = configure('retention-test', root=tmp_path)
    with recorder.operation('retained') as operation:
        operation.fail('transport')
    path = Path(recorder.record_ref)
    recorder.close()
    os.utime(path, (1, 1))
    queue = Outbox(tmp_path / 'outbox.db')
    result = prune(tmp_path, max_bytes=1, queue=queue)
    assert path.exists() and result['unread_files'] == 1 and result['limited']
    consumed = ingest(tmp_path, queue)
    assert consumed['scanned_bytes'] > 0 and consumed['enqueued'] == 0
    # Retention consumes local-only records without authorizing an Issue.
    skipped = prune(tmp_path, max_bytes=1, queue=queue, timeout=1)
    assert skipped['reader_unknown'] and skipped['removed_files'] == 0
    assert prune(tmp_path, max_bytes=1, queue=queue)['removed_files'] == 1
    assert not path.exists()


def test_prune_skips_locked_reader_when_remaining_budget_cannot_admit_sqlite(tmp_path):
    from mindie_diagnostics.reader_registry import register_reader

    folder = tmp_path / 'events' / 'test'
    folder.mkdir(parents=True)
    path = folder / ('1-' + 'a' * 32 + '.jsonl')
    path.write_text('evidence')
    os.utime(path, (1, 1))
    db_path = tmp_path / 'reader.db'
    db = sqlite3.connect(db_path)
    try:
        db.execute('CREATE TABLE cursors (path TEXT PRIMARY KEY,inode TEXT,offset INTEGER)')
        db.execute('INSERT INTO cursors VALUES (?,?,?)',
                   (str(path), str(path.stat().st_ino), path.stat().st_size))
        db.commit()
        register_reader(tmp_path, db_path)
        # A rollback-journal exclusive lock would make the real SQLite reader
        # wait 3s if prune admitted it into this explicit 1s outer budget.
        db.execute('BEGIN EXCLUSIVE')
        started = time.monotonic()
        result = prune(tmp_path, max_bytes=1, timeout=1)
        assert time.monotonic() - started < 1
        assert result['reader_unknown'] and result['limited']
        assert result['removed_files'] == 0 and result['unread_files'] == 1
        assert path.read_text() == 'evidence'
    finally:
        db.close()


def test_retention_keeps_actual_live_writer_and_rotated_family(tmp_path):
    from mindie_diagnostics import configure
    from pathlib import Path

    recorder = configure('live-retention', root=tmp_path)
    try:
        with recorder.operation('before'):
            pass
        path = Path(recorder.record_ref)
        rotated = path.with_name(path.name + '.1')
        rotated.write_text('earlier segment\n')
        before = path.stat().st_size
        result = prune(tmp_path, max_bytes=1, clock=lambda: max(path.stat().st_mtime, rotated.stat().st_mtime) + 301)
        assert result['removed_files'] == 0
        assert result['active_or_unknown_files'] == 2 and result['limited']
        with recorder.operation('after'):
            pass
        assert path.stat().st_size == before and rotated.exists()
        assert recorder.logging_failed and recorder.dropped_records == 2
        # A live file remains open, but maintenance pressure forbids growth.
        assert recorder.operation("still-available").summary()["status"] == "pending"
    finally:
        recorder.close()


@pytest.mark.parametrize('error,exited', [(None, False), (ProcessLookupError(), True),
                                       (PermissionError(), False), (OSError(), False),
                                       (OverflowError(), False)])
def test_only_definitive_posix_process_absence_permits_deletion(monkeypatch, error, exited):
    def probe(pid, signum):
        assert pid == 123 and signum == 0
        if error:
            raise error
    monkeypatch.setattr(maintenance, 'os', SimpleNamespace(name='posix', kill=probe))
    assert maintenance._writer_exited(123) is exited
    assert maintenance._writer_exited(0) is False


def test_windows_never_uses_os_kill_as_a_probe(monkeypatch):
    def unsafe(*args):
        pytest.fail('Windows os.kill must not be used for retention')
    monkeypatch.setattr(maintenance, 'os', SimpleNamespace(name='nt', kill=unsafe))
    monkeypatch.setattr(maintenance, '_kernel32', lambda: None)
    assert maintenance._writer_exited(123) is False


def test_real_exited_process_pid_is_definitively_dead_on_current_os():
    windowless = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    proc = subprocess.Popen([sys.executable, '-I', '-c', 'pass'], **windowless)
    proc.wait(timeout=10)
    assert maintenance._writer_exited(proc.pid) is True


def _fake_kernel32(*, handle=0, error=0, exit_code=None, exit_ok=True):
    calls = []

    def open_process(access, inherit, pid):
        calls.append(('open', access, inherit, pid))
        return handle

    def get_exit_code(opened, pointer):
        calls.append(('exit', opened))
        if exit_ok:
            pointer._obj.value = exit_code
        return 1 if exit_ok else 0

    def close(opened):
        calls.append(('close', opened))
        return 1

    return SimpleNamespace(
        OpenProcess=open_process,
        GetExitCodeProcess=get_exit_code,
        CloseHandle=close,
    ), calls


@pytest.mark.parametrize(('handle', 'error', 'exit_code', 'closed', 'exited'), [
    (0, 87, None, False, True),    # ERROR_INVALID_PARAMETER: PID is gone.
    (0, 5, None, False, False),    # ERROR_ACCESS_DENIED: live protected PID.
    (0, 6, None, False, False),    # Unknown observation is not proof of exit.
    (0x1234, 0, 259, True, False),  # STILL_ACTIVE.
    (0x1234, 0, 0, True, True),     # Exited with code 0.
    (0x1234, 0, None, True, False),  # GetExitCodeProcess failed.
])
def test_windows_writer_exit_contract_uses_kernel32_only(
        monkeypatch, handle, error, exit_code, closed, exited):
    kernel32, calls = _fake_kernel32(
        handle=handle, error=error, exit_code=exit_code,
        exit_ok=exit_code is not None,
    )
    monkeypatch.setattr(maintenance, '_kernel32', lambda: kernel32)
    monkeypatch.setattr(maintenance.ctypes, 'get_last_error', lambda: error, raising=False)
    monkeypatch.setattr(maintenance, 'os', SimpleNamespace(
        name='nt', kill=lambda *args: pytest.fail('os.kill is not a Windows probe')))

    assert maintenance._writer_exited(4242) is exited
    assert calls[0] == ('open', 0x1000, False, 4242)
    assert (('close', handle) in calls) is closed


def test_prune_collects_more_than_the_old_4096_entry_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, '_writer_exited', lambda pid: True)
    folder = tmp_path / 'events' / 'test'
    folder.mkdir(parents=True)
    for index in range(4100):
        (folder / (str(100000 + index) + '-' + 'a' * 32 + '.jsonl')).write_text('x')
    result = prune(tmp_path, max_bytes=1, clock=lambda: 10**10, timeout=10)
    assert result['removed_files'] == 4100
    assert not folder.exists()
    assert (tmp_path / 'retention-state.json').exists()


def test_prune_removes_emptied_component_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, '_writer_exited', lambda pid: True)
    folder = tmp_path / 'events' / 'only'
    folder.mkdir(parents=True)
    path = folder / ('1-' + 'a' * 32 + '.jsonl')
    path.write_text('x')
    os.utime(path, (1, 1))
    assert prune(tmp_path, max_bytes=1)['removed_files'] == 1
    assert not folder.exists()
