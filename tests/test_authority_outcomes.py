"""Existing policy/queue damage never turns into first use or consent revoke."""
import errno
import json
import os
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


@pytest.mark.parametrize('parts', [(), ('missing', 'nested')])
def test_policy_below_file_is_unavailable_not_unconfigured(tmp_path, parts, capsys):
    from mindie_diagnostics import cli, reporting_commands as commands

    parent = tmp_path / 'not-a-directory'
    parent.write_bytes(b'retained file')
    config = parent.joinpath(*parts, 'diagnostics.json')
    before = list(tmp_path.iterdir())
    with pytest.raises(f.PolicyUnavailable):
        f.read_policy(config)
    status = reporting_status(config=config)
    assert status['status'] == 'configuration_unavailable'
    assert status['category'] == 'reporting_policy_unavailable'
    assert commands.upgrade_running(config) == {
        'status': 'degraded', 'reason': 'reporting_policy_unavailable'}
    assert commands.maintain(config)['category'] == 'reporting_policy_unavailable'
    assert cli.main(['reporting', 'status', '--config', str(config)]) == 1
    assert json.loads(capsys.readouterr().out)['category'] == 'reporting_policy_unavailable'
    assert list(tmp_path.iterdir()) == before and parent.read_bytes() == b'retained file'


@pytest.mark.parametrize('parts', [(), ('missing', 'nested')])
def test_genuinely_missing_policy_remains_read_only_first_use(tmp_path, parts):
    from mindie_diagnostics import reporting_commands as commands

    config = tmp_path.joinpath(*parts, 'diagnostics.json')
    before = list(tmp_path.iterdir())
    assert f.read_policy(config) is None
    assert reporting_status(config=config)['status'] == 'not_configured'
    assert commands.upgrade_running(config) == {
        'status': 'skipped', 'reason': 'reporting_not_configured'}
    assert list(tmp_path.iterdir()) == before


def test_missing_policy_keeps_existing_no_symlink_ancestor_policy(tmp_path):
    actual = tmp_path / 'actual'
    actual.mkdir()
    link = tmp_path / 'linked'
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation unavailable')
    with pytest.raises(f.PolicyUnavailable):
        f.read_policy(link / 'missing' / 'diagnostics.json')
    assert list(actual.iterdir()) == [] and link.is_symlink()


@pytest.mark.parametrize('code', [errno.EACCES, errno.EIO])
def test_missing_policy_preserves_ancestor_io_error(tmp_path, monkeypatch, code):
    parent = tmp_path / 'unreadable'
    config = parent / 'diagnostics.json'
    original_lstat = os.lstat
    fault = OSError(code, 'synthetic ancestor fault')
    def lstat(path, *args, **kwargs):
        if Path(path) == parent:
            raise fault
        return original_lstat(path, *args, **kwargs)
    monkeypatch.setattr(f.os, 'lstat', lstat)
    with pytest.raises(f.PolicyUnavailable) as caught:
        f.read_policy(config)
    assert caught.value.__cause__ is fault
    assert not parent.exists()


def test_missing_policy_without_any_verified_ancestor_is_unavailable(tmp_path, monkeypatch):
    config = tmp_path / 'missing' / 'diagnostics.json'
    candidates = {config, *config.parents}
    original_lstat = os.lstat
    def lstat(path, *args, **kwargs):
        if Path(path) in candidates:
            raise FileNotFoundError(errno.ENOENT, 'synthetic inaccessible root', str(path))
        return original_lstat(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(f.os, 'lstat', lstat)
        with pytest.raises(f.PolicyUnavailable):
            f.read_policy(config)
    assert list(tmp_path.iterdir()) == []


def test_file_in_place_of_state_or_runtime_is_not_an_empty_status(tmp_path, capsys):
    from mindie_diagnostics import cli
    from mindie_diagnostics.health import read_health

    config = tmp_path / 'diagnostics.json'
    assert configure_reporting(True, config=config, roots=[str(tmp_path / 'logs')])['status'] == 'configured'
    state, runtime = config.with_suffix('.state'), config.with_suffix('.runtime')
    state.write_bytes(b'retained state')
    runtime.write_bytes(b'retained runtime')
    status = reporting_status(config=config)
    assert status['worker']['status'] == 'unavailable'
    assert status['queue']['status'] == 'unavailable'
    assert status['runtime']['status'] == 'unavailable'
    assert read_health(state) == {'status': 'unreadable', 'healthy': False}
    with pytest.raises(NotADirectoryError):
        cli.main(['status', '--state', str(state)])
    assert capsys.readouterr().out == ''
    assert cli.main(['reporting', 'maintain', '--config', str(config)]) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'degraded'
    assert state.read_bytes() == b'retained state' and runtime.read_bytes() == b'retained runtime'


def test_missing_health_queue_and_runtime_do_not_create_state(tmp_path, capsys):
    from mindie_diagnostics import cli
    from mindie_diagnostics.health import read_health

    config = tmp_path / 'diagnostics.json'
    assert configure_reporting(True, config=config, roots=[str(tmp_path / 'logs')])['status'] == 'configured'
    before = list(tmp_path.iterdir())
    status = reporting_status(config=config)
    assert status['worker']['status'] == 'not_started'
    assert status['queue'] == {'counts': {}, 'recent': []}
    assert status['runtime']['status'] == 'not_prepared'
    state = config.with_suffix('.state')
    assert read_health(state) == {'status': 'not_started', 'healthy': False}
    assert cli.main(['status', '--state', str(state)]) == 0
    assert json.loads(capsys.readouterr().out)['reporter'] == []
    assert list(tmp_path.iterdir()) == before


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
        with pytest.raises((sqlite3.OperationalError, FileNotFoundError)):
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


@pytest.mark.parametrize("damage", ["marker_missing", "marker_changed", "database_replaced", "missing_primary_key"])
def test_open_outbox_never_uses_replaced_or_incomplete_authority(tmp_path, damage):
    path = tmp_path / "queue.db"
    queue = Outbox(path)
    queue.enqueue('fault', 'a' * 32, {'retained': True})
    marker = path.with_name(path.name + '.initialized')
    if damage == 'marker_missing':
        marker.unlink()
    elif damage == 'marker_changed':
        marker.write_bytes(b'bad-marker')
    elif damage == 'database_replaced':
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    else:
        with sqlite3.connect(path) as db:
            rows = db.execute('SELECT * FROM seen').fetchall()
            db.execute('DROP TABLE seen')
            db.execute('CREATE TABLE seen(operation_id TEXT, observed REAL NOT NULL)')
            db.executemany('INSERT INTO seen VALUES(?,?)', rows)
    with pytest.raises((ValueError, OSError)):
        queue.claim()
    if damage == 'missing_primary_key':
        with pytest.raises(ValueError, match='schema'):
            Outbox(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT attempts,state FROM incidents').fetchone() == (0, 'pending')
