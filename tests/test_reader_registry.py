from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import pytest

from mindie_diagnostics.reader_registry import (
    cursor_snapshot, fully_consumed, pressure_blocked, reader_guard,
    register_reader, unregister_reader,
)


def queue(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE cursors (path TEXT PRIMARY KEY,inode TEXT,offset INTEGER)')
    path.chmod(0o600)
    return path


def test_all_registered_readers_must_consume(tmp_path):
    a, b = queue(tmp_path / 'a.db'), queue(tmp_path / 'b.db')
    for path in (a, b):
        register_reader(tmp_path, path)
    with closing(sqlite3.connect(a)) as db, db:
        db.execute('INSERT INTO cursors VALUES (?,?,?)', ('/log', '123', 42))
    with reader_guard(tmp_path) as state:
        assert state['status'] == 'ok'
        cursors = cursor_snapshot(state['readers'])
        assert not fully_consumed(cursors, '/log', 123, 42)
    with closing(sqlite3.connect(b)) as db, db:
        db.execute('INSERT INTO cursors VALUES (?,?,?)', ('/log', '123', 42))
    with reader_guard(tmp_path) as state:
        assert fully_consumed(cursor_snapshot(state['readers']), '/log', 123, 42)
        assert not fully_consumed(cursor_snapshot(state['readers']), '/log', 124, 42)
    unregister_reader(tmp_path, a)
    unregister_reader(tmp_path, b)
    with reader_guard(tmp_path) as state:
        assert state == {'status': 'ok', 'readers': []}


def test_deleted_or_disabled_reader_self_heals_without_bricking_retention(tmp_path):
    path = queue(tmp_path / 'a.db')
    register_reader(tmp_path, path)
    path.unlink()
    with reader_guard(tmp_path) as state:
        assert state == {'status': 'ok', 'readers': []}
        assert cursor_snapshot(state['readers']) == []
    # A non-regular reader binding still fails closed during cursor inspection.
    folder = tmp_path / 'dir.db'
    folder.mkdir()
    assert cursor_snapshot([str(folder)]) is None


def test_corrupt_registry_is_unknown_not_empty(tmp_path):
    with reader_guard(tmp_path):
        pass
    path = tmp_path / '.readers' / 'registry.json'
    path.write_text('{}')
    path.chmod(0o600)
    with reader_guard(tmp_path) as state:
        assert state['status'] == 'unknown'


def test_pressure_marker_only_blocks_on_explicit_blocked_true(tmp_path):
    assert not pressure_blocked(tmp_path)
    path = tmp_path / 'retention-state.json'
    path.write_text('{')
    path.chmod(0o600)
    assert not pressure_blocked(tmp_path)
    path.write_text(json.dumps({'schema': 1, 'blocked': False, 'limited': True}))
    assert not pressure_blocked(tmp_path)
    path.write_text(json.dumps({'schema': 1, 'blocked': True, 'limited': True}))
    path.chmod(0o644)
    assert pressure_blocked(tmp_path)


def test_missing_cursor_table_is_unread(tmp_path):
    path = tmp_path / 'reader.db'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE harmless (id INTEGER)')
    path.chmod(0o600)
    register_reader(tmp_path, path)
    assert cursor_snapshot([str(path)]) == [{}]
    assert not fully_consumed([{}], '/log', 1, 1)


def test_reader_guard_timeout_is_explicit_and_not_swallowed(tmp_path, monkeypatch):
    import mindie_diagnostics.reader_registry as rr

    monkeypatch.setattr(rr, '_try_lock', lambda _fd: False)
    with reader_guard(tmp_path, timeout=0.02) as state:
        assert state == {'status': 'unknown', 'reason': 'timeout', 'readers': []}
    path = queue(tmp_path / 'busy.db')
    with pytest.raises(TimeoutError, match='diagnostic reader registry is busy'):
        register_reader(tmp_path, path, timeout=0.02)
    with pytest.raises(TimeoutError, match='diagnostic reader registry is busy'):
        unregister_reader(tmp_path, path, timeout=0.02)


def test_snapshot_budget_admission_preserves_unread_and_consumed_cursors(tmp_path):
    path = queue(tmp_path / 'reader.db')
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('INSERT INTO cursors VALUES (?,?,?)', ('/log', '123', 42))
    skipped = cursor_snapshot([str(path)], remaining_budget=1)
    assert skipped is None and not fully_consumed(skipped, '/log', 123, 42)
    assert cursor_snapshot([str(path)], remaining_budget=3.2) == [
        {'/log': {'inode': '123', 'offset': 42}},
    ]
    # No database wait is needed when there are no registered consumers.
    assert cursor_snapshot([], remaining_budget=0) == []


def test_snapshot_does_not_enter_locked_database_without_wait_and_cleanup_budget(tmp_path):
    path = queue(tmp_path / 'busy.db')
    with closing(sqlite3.connect(path)) as db:
        db.execute('BEGIN EXCLUSIVE')
        started = time.monotonic()
        assert cursor_snapshot([str(path)], remaining_budget=3.0) is None
        assert time.monotonic() - started < .5


def test_snapshot_readers_share_actual_remaining_budget(tmp_path, monkeypatch):
    import mindie_diagnostics.reader_registry as rr

    a, b = queue(tmp_path / 'a.db'), queue(tmp_path / 'b.db')
    clock = [0.0]
    opened = []
    transactions = []
    original_connect = sqlite3.connect

    def connect(database, **kwargs):
        db = original_connect(database, **kwargs)
        opened.append(database)

        def after_query(sql):
            if sql.startswith('SELECT path,inode,offset'):
                # The first reader really queries SQLite in one transaction;
                # advancing the injected clock makes its cost deterministic.
                transactions.append(db.in_transaction)
                clock[0] += .2
        db.set_trace_callback(after_query)
        return db

    with closing(original_connect(b)) as locked:
        locked.execute('BEGIN EXCLUSIVE')
        monkeypatch.setattr(rr, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
        monkeypatch.setattr(rr.sqlite3, 'connect', connect)
        assert cursor_snapshot([str(a), str(b)], remaining_budget=3.2) is None
    assert opened == [a.as_uri() + '?mode=ro']
    assert transactions == [True]

