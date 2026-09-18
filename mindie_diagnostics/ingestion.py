"""Incremental bounded log consumption, with durable byte cursors and dedup."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import re

from .bundle import _safe_open, collect_bundle, export_public_event
from .outbox import Outbox
from .community import consent_allowed, scope_key

_FILE = re.compile(r"\d+-[0-9a-f]{32}\.jsonl(?:\.[1-3])?\Z")


def _candidates(root, counts):
    events = root / 'events'
    if not events.is_dir() or events.is_symlink():
        return []
    files, visited = [], 0
    with os.scandir(events) as folders:
        for folder in folders:
            visited += 1
            if visited > 4096:
                counts['limited'] = 1
                return files
            if folder.is_symlink() or not folder.is_dir(follow_symlinks=False):
                continue
            with os.scandir(folder.path) as entries:
                for entry in entries:
                    visited += 1
                    if visited > 4096:
                        counts['limited'] = 1
                        return files
                    if entry.is_symlink() or not _FILE.fullmatch(entry.name) or not entry.is_file(follow_symlinks=False):
                        continue
                    path = Path(entry.path)
                    try:
                        info = path.stat()
                    except FileNotFoundError:
                        continue
                    files.append((info.st_mtime, path, info.st_ino, info.st_size))
    return files


def ingest(root, queue: Outbox, *, max_files=256, max_bytes=8 * 1024 * 1024, since=None):
    from .reporter import fingerprint, issue_payload
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError('diagnostic root must not traverse symlinks')
    counts = {'enqueued': 0, 'invalid': 0, 'scanned_bytes': 0, 'limited': 0, 'files': 0, 'caller_errors': 0, 'before_start': 0, 'consent_skipped': 0}
    with queue.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS cursors (path TEXT PRIMARY KEY, inode TEXT, offset INTEGER, touched REAL, discard INTEGER DEFAULT 0)')
        db.execute('DELETE FROM cursors WHERE touched < ?', (queue.clock() - 30 * 86400,))
        db.execute('DELETE FROM cursors WHERE path IN (SELECT path FROM cursors ORDER BY touched DESC LIMIT -1 OFFSET 10000)')
        cursors = {row['path']: dict(row) for row in db.execute('SELECT * FROM cursors')}
    candidates = []
    for modified, path, inode, size in _candidates(root, counts):
        cursor = cursors.get(str(path), {})
        offset = cursor.get('offset', 0) if cursor.get('inode') == str(inode) else 0
        if size < offset:
            offset = 0
        if size > offset:
            candidates.append((cursor.get('touched', 0), modified, path, str(inode), offset, cursor.get('discard', 0) if offset else 0))
    for _, _, path, candidate_identity, offset, discard in sorted(candidates):
        if counts['files'] >= max_files or counts['scanned_bytes'] >= max_bytes:
            counts['limited'] = 1
            break
        counts['files'] += 1
        recent = deque(maxlen=100)
        with _safe_open(path) as stream:
            identity = str(os.fstat(stream.fileno()).st_ino)
            # Rotation can replace the path after candidate discovery. Offsets
            # and oversized-line state belong to the inode actually observed.
            if identity != candidate_identity:
                offset, discard = 0, 0
            stream.seek(offset)
            while counts['scanned_bytes'] < max_bytes:
                start = stream.tell()
                line = stream.readline(min(65537, max_bytes - counts['scanned_bytes']))
                if not line:
                    break
                counts['scanned_bytes'] += len(line)
                if discard:
                    discard = int(not line.endswith(b'\n'))
                    offset = stream.tell()
                    continue
                if len(line) > 65536:
                    counts['invalid'] += 1
                    discard = int(not line.endswith(b'\n'))
                    offset = stream.tell()
                    continue
                if not line.endswith(b'\n'):
                    # A writer may still be appending this event. Keep the cursor
                    # before it; never turn a later fragment into a new record.
                    counts['limited'] = 1
                    stream.seek(start)
                    break
                offset = stream.tell()
                try:
                    raw = json.loads(line)
                    event = export_public_event(raw)
                    if event is None:
                        counts['invalid'] += 1
                        continue
                    if since is not None and datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).timestamp() < since:
                        counts['before_start'] += 1
                        continue
                    consent = raw.get('community')
                    # Older unscoped logs stay local. Enabling a workspace later
                    # must never backfill logs produced without that consent.
                    recent.append((event, consent))
                    if event.get('event') != 'operation.end' or event.get('status') != 'error':
                        continue
                    attributes = event.get('attributes', {})
                    if (attributes.get('category') in {'caller', 'cancelled'}
                            or attributes.get('classification') in {'caller', 'cancelled'}):
                        counts['caller_errors'] += 1
                        continue
                    if not consent_allowed(consent):
                        counts['consent_skipped'] += 1
                        continue
                    records = [row for row, scope in recent if scope == consent and row['operation_id'] == event['operation_id']]
                    bundle = collect_bundle(root, operation_id=event['operation_id'], records=records,
                                            include_logs=False, max_bytes=36000)
                    payload = issue_payload(bundle)
                    counts['enqueued'] += int(queue.enqueue(scope_key(fingerprint(payload), consent),
                                                            scope_key(event['operation_id'], consent),
                                                            payload, consent=consent))
                except (ValueError, KeyError, TypeError):
                    counts['invalid'] += 1
            with queue.connect() as db:
                db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?,?)', (str(path), identity, offset, queue.clock(), discard))
    return counts
