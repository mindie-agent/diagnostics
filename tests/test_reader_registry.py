import json
import os
from pathlib import Path
import sqlite3

import pytest

from mindie_diagnostics.reader_registry import (
    cursor_snapshot, fully_consumed, pressure_blocked, reader_guard,
    register_reader, unregister_reader,
)


def queue(path):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE cursors (path TEXT PRIMARY KEY,inode TEXT,offset INTEGER)')
    path.chmod(0o600)
    return path


def test_all_registered_readers_must_consume(tmp_path):
    a, b = queue(tmp_path / 'a.db'), queue(tmp_path / 'b.db')
    for path in (a, b):
        register_reader(tmp_path, path)
    with sqlite3.connect(a) as db:
        db.execute('INSERT INTO cursors VALUES (?,?,?)', ('/log', '123', 42))
    with reader_guard(tmp_path) as state:
        assert state['status'] == 'ok'
        cursors = cursor_snapshot(state['readers'])
        assert not fully_consumed(cursors, '/log', 123, 42)
    with sqlite3.connect(b) as db:
        db.execute('INSERT INTO cursors VALUES (?,?,?)', ('/log', '123', 42))
    with reader_guard(tmp_path) as state:
        assert fully_consumed(cursor_snapshot(state['readers']), '/log', 123, 42)
        assert not fully_consumed(cursor_snapshot(state['readers']), '/log', 124, 42)
    unregister_reader(tmp_path, a)
    unregister_reader(tmp_path, b)
    with reader_guard(tmp_path) as state:
        assert state == {'status': 'ok', 'readers': []}


def test_missing_registered_reader_protects_all(tmp_path):
    path = queue(tmp_path / 'a.db')
    register_reader(tmp_path, path)
    path.unlink()
    with reader_guard(tmp_path) as state:
        assert cursor_snapshot(state['readers']) is None
    unregister_reader(tmp_path, path)


def test_corrupt_registry_is_unknown_not_empty(tmp_path):
    with reader_guard(tmp_path):
        pass
    path = tmp_path / '.readers' / 'registry.json'
    path.write_text('{}')
    path.chmod(0o600)
    with reader_guard(tmp_path) as state:
        assert state['status'] == 'unknown'


def test_pressure_marker_fail_closed(tmp_path):
    assert not pressure_blocked(tmp_path)
    path = tmp_path / 'retention-state.json'
    path.write_text('{')
    path.chmod(0o600)
    assert pressure_blocked(tmp_path)
    path.write_text(json.dumps({'schema': 1, 'blocked': False, 'limited': True}))
    assert not pressure_blocked(tmp_path)
    path.chmod(0o644)
    if os.name != 'nt':
        assert pressure_blocked(tmp_path)


def test_symlink_registry_rejected(tmp_path):
    target = tmp_path / 'target'
    target.mkdir()
    (tmp_path / '.readers').symlink_to(target)
    with reader_guard(tmp_path) as state:
        assert state['status'] == 'unknown'


def test_missing_cursor_table_is_unread(tmp_path):
    path = tmp_path / 'reader.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE harmless (id INTEGER)')
    path.chmod(0o600)
    register_reader(tmp_path, path)
    assert cursor_snapshot([str(path)]) == [{}]
    assert not fully_consumed([{}], '/log', 1, 1)
