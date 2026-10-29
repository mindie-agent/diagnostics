import concurrent.futures
import sqlite3

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
    now[0] = 1100.0
    item = queue.claim(lease_seconds=10)
    assert item['attempt_timestamp'] is None
    now[0] = 1101.0
    assert queue.begin_post(item)
    now[0] += 11
    reclaimed = queue.claim()
    assert reclaimed['state'] == 'uncertain'
    assert reclaimed['first_seen'] == 1000.0
    assert reclaimed['attempt_timestamp'] == 1101.0
    with pytest.raises(RuntimeError, match='lease expired'):
        queue.begin_post(item)
    with queue.connect() as db:
        assert db.execute("SELECT attempt_timestamp FROM incidents").fetchone()[0] == 1101.0
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


def test_repeated_crashes_and_old_unsent_work_retain_payload(tmp_path):
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', capacity=1, clock=lambda: now[0])
    queue.enqueue('a', 'a' * 32, {'evidence': 'retained'})
    for expected in range(1, 6):
        item = queue.claim(lease_seconds=1)
        assert item['attempts'] == expected
        assert item['payload']['evidence'] == 'retained'
        now[0] += 2
    item = queue.claim()
    queue.begin_post(item)
    now[0] += 31 * 86400
    queue.maintain()
    retained = queue.claim()
    assert retained['state'] == 'uncertain'
    assert retained['payload']['evidence'] == 'retained'
    with pytest.raises(QueueFull):
        queue.enqueue('b', 'b' * 32, {})


def test_new_occurrence_restores_legacy_known_unsent_failure(tmp_path):
    queue = Outbox(tmp_path / "queue.db")
    queue.enqueue("fault", "a" * 32, {"evidence": "old"})
    item = queue.claim()
    queue.begin_post(item)
    queue.update(item, state="permanent-failed", last_error="github_http_422")
    queue.enqueue("fault", "b" * 32, {"evidence": "new"})
    item = queue.claim()
    assert item["state"] == "pending" and item["payload"]["evidence"] == "new"
    assert item["attempt_timestamp"] is None


def test_begin_post_timestamp_and_uncertainty_commit_or_rollback_together(tmp_path):
    queue = Outbox(tmp_path / "queue.db", clock=lambda: 1000.0)
    queue.enqueue("fault", "op", {})
    item = queue.claim()
    with queue.connect() as db:
        db.executescript("""
            CREATE TRIGGER require_attempt_timestamp BEFORE UPDATE OF state ON incidents
            WHEN NEW.state = 'uncertain' AND NEW.attempt_timestamp IS NULL
            BEGIN SELECT RAISE(ABORT, 'uncertainty requires attempt timestamp'); END;
            CREATE TRIGGER fail_rate_receipt BEFORE INSERT ON publications
            BEGIN SELECT RAISE(ABORT, 'rate receipt failed'); END;
        """)
    with pytest.raises(sqlite3.IntegrityError, match="rate receipt failed"):
        queue.begin_post(item, hourly_limit=1)
    with queue.connect() as db:
        row = db.execute("SELECT state,attempt_timestamp FROM incidents").fetchone()
        assert tuple(row) == ("pending", None)
        db.execute("DROP TRIGGER fail_rate_receipt")
    assert queue.begin_post(item, hourly_limit=1)
    with queue.connect() as db:
        row = db.execute("SELECT state,attempt_timestamp FROM incidents").fetchone()
        assert tuple(row) == ("uncertain", 1000.0)
        assert db.execute("SELECT at FROM publications").fetchone()[0] == 1000.0


def test_legacy_schema_migrates_and_uncertain_claim_freezes_first_seen(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE incidents (
                fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL,
                first_seen REAL NOT NULL, last_seen REAL NOT NULL,
                occurrences INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
                lease_token TEXT, issue_number INTEGER, issue_url TEXT, last_error TEXT,
                consent TEXT);
            CREATE TABLE seen (operation_id TEXT PRIMARY KEY, observed REAL NOT NULL);
            CREATE TABLE publications (at REAL NOT NULL);
            INSERT INTO incidents (fingerprint,payload,first_seen,last_seen,state)
                VALUES ('unknown','{"evidence":"retained"}',100,100,'uncertain');
            INSERT INTO incidents (fingerprint,payload,first_seen,last_seen)
                VALUES ('unsent','{}',200,200);
        """)
    now = [1000.0]
    queue = Outbox(path, clock=lambda: now[0])
    with queue.connect() as db:
        assert [row[0] for row in db.execute("SELECT attempt_timestamp FROM incidents")] == [None, None]
    item = queue.claim()
    assert item["attempt_timestamp"] == 100.0
    assert item["payload"] == {"evidence": "retained"}
    queue.update(item, state="uncertain", next_attempt=1100.0)
    pending = queue.claim()
    assert pending["fingerprint"] == "unsent" and pending["attempt_timestamp"] is None
    now[0] = 1200.0
    reopened = Outbox(path, clock=lambda: now[0])
    reclaimed = reopened.claim()
    assert reclaimed["fingerprint"] == "unknown" and reclaimed["attempt_timestamp"] == 100.0
    with reopened.connect() as db:
        assert db.execute(
            "SELECT attempt_timestamp FROM incidents WHERE fingerprint='unknown'"
        ).fetchone()[0] == reclaimed["attempt_timestamp"]


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
