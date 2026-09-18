import os

from mindie_diagnostics.maintenance import prune


def test_retention_preserves_fresh_and_unrelated_files(tmp_path):
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


def test_worker_retention_preserves_unread_old_evidence(tmp_path, community_consent):
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
