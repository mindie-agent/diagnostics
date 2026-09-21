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


def test_three_crashed_leases_exhaust_and_new_occurrence_does_not_reset(tmp_path):
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    queue.enqueue('a', 'a' * 32, {})
    for expected in (1, 2, 3):
        assert queue.claim(lease_seconds=1)['attempts'] == expected
        now[0] += 2
    assert queue.claim() is None
    queue.enqueue('a', 'b' * 32, {})
    assert queue.claim() is None
    row = queue.rows()[0]
    assert row['state'] == 'exhausted' and row['attempts'] == 3
    assert row['incident_ids'] == ['a' * 32, 'b' * 32]


def test_expired_uncertain_receipt_does_not_occupy_active_capacity(tmp_path):
    from mindie_diagnostics.outbox import UNSENT_TTL_SECONDS
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', capacity=1, clock=lambda: now[0])
    queue.enqueue('a', 'a' * 32, {'events': []})
    item = queue.claim(lease_seconds=1)
    queue.begin_post(item)
    now[0] += UNSENT_TTL_SECONDS + 1
    queue.maintain()
    assert queue.rows()[0]['state'] == 'expired'
    assert 'uncertain' in queue.rows()[0]['last_error']
    queue.enqueue('b', 'b' * 32, {})
    assert queue.claim()['fingerprint'] == 'b'


def test_valid_lease_protects_evidence_from_expiry(tmp_path):
    from mindie_diagnostics.outbox import UNSENT_TTL_SECONDS
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    queue.enqueue('a', 'a' * 32, {'evidence': True})
    queue.claim(lease_seconds=2 * UNSENT_TTL_SECONDS)
    now[0] += UNSENT_TTL_SECONDS + 1
    queue.maintain()
    assert queue.rows()[0]['state'] == 'pending'


def test_withdrawal_preserves_unknown_submission_fact(tmp_path):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('a', 'a' * 32, {})
    item = queue.claim()
    queue.begin_post(item)
    queue.update(item, state='uncertain', last_error='github_timeout')
    assert queue.withdraw_unconsented() == 1
    row = queue.rows()[0]
    assert row['state'] == 'withdrawn' and row['last_error'] == 'submission_uncertain_reporting_consent_withdrawn'
