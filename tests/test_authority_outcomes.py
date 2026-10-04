"""Existing policy/queue damage never turns into first use or consent revoke."""
import json
import sqlite3
from pathlib import Path

import pytest

from mindie_diagnostics import fallback as f
from mindie_diagnostics.integration import configure_reporting, reporting_status, _runtime_view
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.reporting import ConsentUnavailable
from mindie_diagnostics.reporter import publish_one
from test_reporting_consent import Publisher, reporting_failure, reporting_policy
from test_reporter import payload as public_payload


@pytest.mark.parametrize('damage', ['invalid_json', 'empty', 'wrong_type', 'permission'])
def test_invalid_existing_policy_cannot_be_overwritten_or_withdraw_pending(tmp_path, monkeypatch, damage):
    config = tmp_path / 'config.json'
    consent = reporting_policy(config)
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('fault', 'a' * 32, {'evidence': 'retained'}, consent=consent)
    original_reader = f._read_regular_bounded
    if damage == 'permission':
        monkeypatch.setattr(f, '_read_regular_bounded', lambda path, limit: None if path == config else original_reader(path, limit))
    else:
        config.write_text({'invalid_json': '{', 'empty': '', 'wrong_type': '[]'}[damage])
    before = config.read_bytes()
    with pytest.raises(f.PolicyUnavailable):
        f.read_policy(config)
    assert reporting_status(config=config)['status'] == 'configuration_unavailable'
    assert configure_reporting(True, config=config)['category'] == 'reporting_policy_unavailable'
    assert config.read_bytes() == before
    with pytest.raises(ConsentUnavailable):
        queue.withdraw_unconsented()
    assert queue.rows()[0]['state'] == 'pending'
    with queue.connect() as db:
        assert json.loads(db.execute('SELECT payload FROM incidents').fetchone()[0])['evidence'] == 'retained'


def test_policy_fault_during_ingestion_retains_cursor_for_recovery(tmp_path):
    config = tmp_path / 'config.json'
    reporting_policy(config)
    reporting_failure(tmp_path, config)
    valid = config.read_bytes()
    queue = Outbox(tmp_path / 'queue.db')
    config.write_text('{broken')
    with pytest.raises(ConsentUnavailable):
        ingest(tmp_path, queue)
    with queue.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM cursors').fetchone()[0] == 0
    config.write_bytes(valid)
    assert ingest(tmp_path, queue)['enqueued'] == 1


@pytest.mark.parametrize('mode', ['policy_read_failure', 'malformed_reply'])
def test_post_phase_stays_uncertain_and_never_reposts(tmp_path, mode):
    config = tmp_path / 'config.json'
    consent = reporting_policy(config)
    now = [1000.]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    queue.enqueue('fault', 'a' * 32, public_payload(), consent=consent)
    valid = config.read_bytes()
    class UnknownPost(Publisher):
        def create_issue(self, title, body):
            self.calls.append('post')
            if mode == 'policy_read_failure':
                config.write_text('{')
                raise ConsentUnavailable()
            raise ValueError('malformed external response')
    publisher = UnknownPost()
    assert publish_one(queue, publisher)['status'] == 'uncertain'
    config.write_bytes(valid)
    for _ in range(3):
        now[0] += 4000
        publish_one(queue, publisher)
    row = queue.rows()[0]
    assert publisher.calls.count('post') == 1
    assert row['state'] == 'exhausted' and row['attempts'] == 3
    assert 'uncertain' in row['last_error']


@pytest.mark.parametrize('damage', ['empty', 'missing_table', 'deleted'])
def test_outbox_damage_is_not_initialized_as_an_empty_authority(tmp_path, damage):
    path = tmp_path / 'queue.db'
    queue = Outbox(path)
    queue.enqueue('fault', 'a' * 32, {})
    item = queue.claim()
    queue.begin_post(item)
    if damage == 'empty':
        path.write_bytes(b'')
    elif damage == 'missing_table':
        with queue.connect() as db:
            db.execute('DROP TABLE incidents')
    else:
        path.unlink()
    before = path.read_bytes() if path.exists() else None
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        Outbox(path)
    assert (path.read_bytes() if path.exists() else None) == before
    if damage == 'deleted':
        with pytest.raises(sqlite3.OperationalError):
            queue.rows()
        assert not path.exists()


def test_unreadable_existing_runtime_pointer_is_not_first_use(tmp_path):
    config = tmp_path / 'config.json'
    pointer = config.with_suffix('.runtime') / 'current.json'
    pointer.parent.mkdir()
    pointer.write_bytes(b'')
    assert _runtime_view(config)['status'] == 'unavailable'
    pointer.unlink()
    assert _runtime_view(config)['status'] == 'not_prepared'


@pytest.mark.parametrize('raw', ['{broken', '[]', 'null', '{}'])
@pytest.mark.parametrize('state', ['pending', 'uncertain'])
def test_damaged_saved_consent_keeps_original_row_and_never_publishes(tmp_path, raw, state):
    config = tmp_path / 'config.json'
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('fault', 'a' * 32, {'evidence': 'retained'}, consent=reporting_policy(config))
    with queue.connect() as db:
        db.execute('UPDATE incidents SET state=?,consent=?', (state, raw))
        before = dict(db.execute('SELECT * FROM incidents').fetchone())
    publisher = Publisher()
    with pytest.raises(ConsentUnavailable):
        queue.withdraw_unconsented()
    with pytest.raises(ConsentUnavailable):
        publish_one(queue, publisher)
    with queue.connect() as db:
        assert dict(db.execute('SELECT * FROM incidents').fetchone()) == before
    assert publisher.calls == []


def test_absent_saved_consent_is_still_an_unreportable_legacy_event(tmp_path):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('legacy', 'a' * 32, {})
    assert queue.withdraw_unconsented() == 1
    assert queue.rows()[0]['state'] == 'withdrawn'


@pytest.mark.parametrize("mode", ["ack", "reconcile", "transport_failure"])
def test_external_outcome_survives_receipt_failure(tmp_path, monkeypatch, mode):
    from mindie_diagnostics.reporter import TransportError
    config = tmp_path / 'config.json'
    consent = reporting_policy(config)
    now = [1000.]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    queue.enqueue('fault', 'a' * 32, public_payload(), consent=consent)
    reply = {"number": 7, "html_url": "https://github.com/example/project/issues/7"}
    class Writer(Publisher):
        def find_issue(self, item):
            return reply if mode == "reconcile" else None
        def create_issue(self, title, body):
            self.calls.append('post')
            if mode == "transport_failure":
                raise TransportError("original_transport_failure", uncertain=True)
            return reply
    publisher = Writer()
    def failed_record(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic disk failure")
    original = queue.update
    monkeypatch.setattr(queue, "update", failed_record)
    result = publish_one(queue, publisher)
    assert result["status"] == "recording_failed"
    assert result["local_recording"]["error_type"] == "OperationalError"
    if mode == "transport_failure":
        assert result["error"] == "original_transport_failure" and result["submission_state"] == "uncertain"
    else:
        assert result["issue_url"] == reply["html_url"] and result["operation_completed"]
        assert result["publication_status"] == ("published" if mode == "ack" else "reconciled")
    monkeypatch.setattr(queue, "update", original)
    now[0] += 4000
    publish_one(queue, publisher)
    assert publisher.calls.count('post') == (0 if mode == "reconcile" else 1)
