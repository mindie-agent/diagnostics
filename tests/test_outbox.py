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


def test_published_receipt_survives_consent_disable_and_pending_withdraws(tmp_path):
    import sqlite3
    from mindie_diagnostics.integration import configure_reporting
    from mindie_diagnostics.reporting import current_consent
    path = tmp_path / 'reporting.json'
    roots = [str(tmp_path)]
    assert configure_reporting(True, config=path, repository='example/project', roots=roots)['status'] == 'configured'
    consent = current_consent(path)
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    with queue.connect() as db:
        columns = {row[1] for row in db.execute('PRAGMA table_info(incidents)')}
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'diagnosis' not in columns and 'diagnosis_state' not in columns and 'generations' not in tables
    assert not hasattr(Outbox, 'begin_generation') and not hasattr(Outbox, 'published')
    queue.enqueue('pub', 'a' * 32, {'component': 'reporter'}, consent=consent)
    now[0] += 1
    queue.enqueue('unc', 'b' * 32, {'component': 'reporter'}, consent=consent)
    now[0] += 1
    queue.enqueue('pen', 'c' * 32, {'component': 'reporter'}, consent=consent)
    published = queue.claim()
    queue.update(published, state='published', issue_number=7, issue_url='https://github.com/example/project/issues/7')
    uncertain = queue.claim()
    assert queue.begin_post(uncertain)
    assert queue.claim()['fingerprint'] == 'pen'
    configure_reporting(False, config=path, repository='example/project', roots=roots)
    assert queue.withdraw_unconsented() == 2
    rows = {row['fingerprint']: row for row in queue.rows()}
    assert rows['pub']['state'] == 'published'
    assert rows['pub']['issue_number'] == 7
    assert rows['pub']['issue_url'] == 'https://github.com/example/project/issues/7'
    assert rows['pub']['attempts'] == 1
    assert 'diagnosis_state' not in rows['pub']
    assert rows['pen']['state'] == 'withdrawn'
    assert rows['unc']['state'] == 'withdrawn'
    assert rows['unc']['last_error'] == 'submission_uncertain_reporting_consent_withdrawn'
    legacy = tmp_path / 'legacy.db'
    raw = sqlite3.connect(legacy)
    raw.executescript("""
        CREATE TABLE incidents (
            fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL,
            first_seen REAL NOT NULL, last_seen REAL NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
            lease_token TEXT, issue_number INTEGER, issue_url TEXT, last_error TEXT,
            diagnosis TEXT, diagnosis_state TEXT NOT NULL DEFAULT 'pending', consent TEXT);
        CREATE TABLE seen (operation_id TEXT PRIMARY KEY, observed REAL NOT NULL);
        CREATE TABLE publications (at REAL NOT NULL);
        CREATE TABLE generations (at REAL NOT NULL);
    """)
    raw.execute(
        "INSERT INTO incidents (fingerprint,payload,first_seen,last_seen,state,attempts,issue_number,issue_url,diagnosis,diagnosis_state) "
        "VALUES ('legacy','{}',1,1,'published',2,9,'https://github.com/example/project/issues/9','old','pending')"
    )
    raw.commit()
    raw.close()
    opened = Outbox(legacy)
    kept = opened.rows()[0]
    assert kept['state'] == 'published' and kept['attempts'] == 2 and kept['issue_number'] == 9
    assert kept['issue_url'] == 'https://github.com/example/project/issues/9'
    with opened.connect() as db:
        stored = dict(db.execute("SELECT diagnosis,diagnosis_state,attempts,issue_number,issue_url FROM incidents").fetchone())
    assert stored == {'diagnosis': 'old', 'diagnosis_state': 'pending', 'attempts': 2,
                      'issue_number': 9, 'issue_url': 'https://github.com/example/project/issues/9'}
