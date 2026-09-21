import os
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


def test_worker_retention_preserves_unread_old_evidence(tmp_path, community_consent, monkeypatch):
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
    assert ingest(tmp_path, queue)['enqueued'] == 1
    assert prune(tmp_path, max_bytes=1, queue=queue)['removed_files'] == 1
    assert not path.exists()


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
        assert path.stat().st_size > before and rotated.exists()
        assert not recorder.logging_failed
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
    assert maintenance._writer_exited(123) is False
