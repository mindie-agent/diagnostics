"""Offline retention. Called by the worker, never by ordinary tool operations."""
from __future__ import annotations

import os
from pathlib import Path
import re
import time

_FILE = re.compile(r"(\d+)-[0-9a-f]{32}\.jsonl(?:\.[1-3])?\Z")


def _writer_exited(pid: int) -> bool:
    """Only positive POSIX PIDs with a definitive ESRCH are safe to prune.

    An idle writer can still hold an append handle. PID reuse, permission
    failures and platforms without this probe all retain the file. In
    particular, Windows os.kill(pid, 0) is not a POSIX liveness check.
    """
    if os.name != "posix" or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError, OverflowError):
        return False
    return False


def prune(root: str | Path, *, max_bytes: int = 128 * 1024 * 1024,
          max_age_days: int = 7, max_files: int = 512, clock=time.time, queue=None) -> dict:
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("retention root must not traverse symlinks")
    events = root / 'events'
    result = {"removed_files": 0, "removed_bytes": 0, "remaining_bytes": 0, "limited": False,
              "unread_files": 0, "active_or_unknown_files": 0}
    cursors = None
    if queue is not None:
        with queue.connect() as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cursors'").fetchone()
            cursors = {row['path']: dict(row) for row in db.execute('SELECT * FROM cursors')} if exists else {}
    if not events.is_dir() or events.is_symlink():
        return result
    files, visited = [], 0
    with os.scandir(events) as folders:
        for folder in folders:
            visited += 1
            if visited > 4096:
                result['limited'] = True
                break
            if folder.is_symlink() or not folder.is_dir(follow_symlinks=False):
                continue
            with os.scandir(folder.path) as entries:
                for entry in entries:
                    visited += 1
                    if visited > 4096:
                        result['limited'] = True
                        break
                    if entry.is_symlink() or not _FILE.fullmatch(entry.name) or not entry.is_file(follow_symlinks=False):
                        continue
                    path = Path(entry.path)
                    # Windows DirEntry.stat can report st_ino=0; Path.stat opens
                    # the file metadata used by the pre-unlink comparison.
                    info = path.stat(follow_symlinks=False)
                    files.append((info.st_mtime, info.st_size, path, info.st_ino))
            if result['limited']:
                break
    total, remaining, now = sum(row[1] for row in files), len(files), clock()
    exited = {}
    for modified, size, path, inode in sorted(files):
        if cursors is not None:
            cursor = cursors.get(str(path), {})
            if cursor.get('inode') != str(inode) or cursor.get('offset', 0) < size:
                result['unread_files'] += 1
                continue  # Bounded ingestion/backpressure must not erase pending evidence.
        if now - modified < 300:
            continue  # Allow current writers and just-generated evidence to settle.
        if now - modified <= max_age_days * 86400 and total <= max_bytes and remaining <= max_files:
            continue
        pid = int(_FILE.fullmatch(path.name).group(1))
        if pid not in exited:
            exited[pid] = _writer_exited(pid)
        if not exited[pid]:
            result['active_or_unknown_files'] += 1
            continue
        # Check the absolute target again immediately before a nonrecursive removal.
        if not path.resolve().is_relative_to(events.resolve()) or path.is_symlink():
            continue
        try:
            current = path.stat()
            if (current.st_ino, current.st_mtime, current.st_size) != (inode, modified, size):
                continue
            path.unlink()
        except (FileNotFoundError, PermissionError):
            continue
        total -= size
        remaining -= 1
        result['removed_files'] += 1
        result['removed_bytes'] += size
    result['remaining_bytes'] = total
    result['limited'] |= total > max_bytes or remaining > max_files
    return result
