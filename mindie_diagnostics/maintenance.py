"""Offline retention. Called by the worker, never by ordinary tool operations."""
from __future__ import annotations

import os
import json
import stat
import uuid
from pathlib import Path
import re
import time

from .reader_registry import reader_guard, cursor_snapshot, fully_consumed

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


def _marker(root, result, blocked, now):
    data = {"schema": 1, "blocked": bool(blocked), "limited": bool(result['limited']),
            "checked_at": now, "remaining_bytes": result['remaining_bytes'],
            "remaining_files": result['remaining_files'],
            "unread_files": result['unread_files'],
            "active_or_unknown_files": result['active_or_unknown_files']}
    raw = json.dumps(data, separators=(",", ":")).encode()
    if len(raw) > 2048:
        raise ValueError("retention status exceeds bound")
    temporary = root / ('.retention-' + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
        os.replace(temporary, root / 'retention-state.json')
    finally:
        temporary.unlink(missing_ok=True)


def prune(root: str | Path, *, max_bytes: int = 128 * 1024 * 1024,
          max_age_days: int = 7, max_files: int = 512, clock=time.time, queue=None) -> dict:
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("retention root must not traverse symlinks")
    events = root / 'events'
    result = {"removed_files": 0, "removed_bytes": 0, "remaining_bytes": 0,
              "remaining_files": 0, "limited": False, "reader_unknown": False,
              "unread_files": 0, "active_or_unknown_files": 0}
    if not root.exists():
        return result
    deadline = time.monotonic() + 1
    files, visited, incomplete = [], 0, False
    total, remaining, now = 0, 0, clock()
    try:
        if events.is_symlink():
            raise ValueError("retention event directory must not be a symlink")
        if events.is_dir():
            with os.scandir(events) as folders:
                for folder in folders:
                    visited += 1
                    if visited > 4096 or time.monotonic() >= deadline:
                        incomplete = True
                        break
                    if folder.is_symlink() or not folder.is_dir(follow_symlinks=False):
                        continue
                    with os.scandir(folder.path) as entries:
                        for entry in entries:
                            visited += 1
                            if visited > 4096 or time.monotonic() >= deadline:
                                incomplete = True
                                break
                            if (entry.is_symlink() or not _FILE.fullmatch(entry.name)
                                    or not entry.is_file(follow_symlinks=False)):
                                continue
                            path = Path(entry.path)
                            info = path.stat(follow_symlinks=False)
                            files.append((info.st_mtime, info.st_size, path,
                                          info.st_ino, info.st_dev, info.st_mtime_ns))
                    if incomplete:
                        break
        total, remaining = sum(row[1] for row in files), len(files)
        with reader_guard(root, timeout=min(.1, max(0, deadline - time.monotonic())),
                          queue_path=queue.path if queue is not None else None) as state:
            cursors = cursor_snapshot(state['readers'], timeout=min(.2, max(0, deadline - time.monotonic()))) if state['status'] == 'ok' else None
            result['reader_unknown'] = cursors is None
            exited = {}
            for modified, size, path, inode, device, modified_ns in sorted(files):
                if time.monotonic() >= deadline:
                    incomplete = True
                    break
                if not fully_consumed(cursors, path, inode, size):
                    result['unread_files'] += 1
                    continue
                if now - modified < 300:
                    continue
                if (now - modified <= max_age_days * 86400
                        and total <= max_bytes and remaining <= max_files):
                    continue
                pid = int(_FILE.fullmatch(path.name).group(1))
                if pid not in exited:
                    exited[pid] = _writer_exited(pid)
                if not exited[pid]:
                    result['active_or_unknown_files'] += 1
                    continue
                if (path.is_symlink() or path.parent.is_symlink()
                        or not path.resolve().is_relative_to(events.resolve())):
                    incomplete = True
                    continue
                try:
                    current = path.stat(follow_symlinks=False)
                    if (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
                            or (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size)
                            != (device, inode, modified_ns, size)):
                        incomplete = True
                        continue
                    path.unlink()
                except FileNotFoundError:
                    incomplete = True
                    continue
                total -= size
                remaining -= 1
                result['removed_files'] += 1
                result['removed_bytes'] += size
    except (OSError, ValueError):
        incomplete = True
        result['reader_unknown'] = True
        total, remaining = sum(row[1] for row in files), len(files)
    result['remaining_bytes'] = total
    result['remaining_files'] = remaining
    blocked = incomplete or total > max_bytes or remaining > max_files
    result['limited'] = bool(blocked or result['reader_unknown'])
    try:
        _marker(root, result, blocked, now)
    except (OSError, ValueError):
        result['limited'] = True
        result['marker_failed'] = True
    return result
