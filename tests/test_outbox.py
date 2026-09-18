import concurrent.futures

import pytest

from mindie_diagnostics.outbox import Outbox, QueueFull


def test_dedup_capacity_and_claim(tmp_path):
    queue = Outbox(tmp_path / 'queue.db', capacity=1)
    assert queue.enqueue('a', 'op1', {'evidence': 1})
    assert not queue.enqueue('a', 'op1', {'evidence': 1})
    assert queue.enqueue('a', 'op2', {'evidence': 1})
    with pytest.raises(QueueFull):
        queue.enqueue('b', 'op3', {})
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        claims = list(pool.map(lambda _: queue.claim(), range(4)))
    assert len([item for item in claims if item]) == 1
    item = next(item for item in claims if item)
    assert item['occurrences'] == 2
    queue.update(item, state='published')
    assert queue.claim() is None


def test_crash_after_begin_post_retains_uncertainty(tmp_path):
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    queue.enqueue('a', 'o', {})
    item = queue.claim(lease_seconds=10)
    assert queue.begin_post(item)
    now[0] += 11
    reclaimed = queue.claim()
    assert reclaimed['state'] == 'uncertain'
    with pytest.raises(RuntimeError, match='lease lost'):
        queue.update(item, state='published')


def test_rate_limit_and_unwritable_database(tmp_path):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('a', 'o', {})
    item = queue.claim()
    assert not queue.begin_post(item, hourly_limit=0)
    assert queue.begin_generation(item, hourly_limit=1)
    assert not queue.begin_generation(item, hourly_limit=1)
    file = tmp_path / 'file'
    file.write_text('not a directory')
    with pytest.raises(OSError):
        Outbox(file / 'queue.db')
