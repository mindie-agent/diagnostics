"""Offline log retention and backpressure marker management.

[Role]
Prunes consumed, aged, or over-quota JSONL event segments under `<root>/events`
whose writer process has exited, and writes `<root>/retention-state.json` to
signal backpressure to active log writers when retention limits remain exceeded.

[Invariants]
- Called only by maintenance/worker flows, never inline during ordinary tool
  operations.
- Never deletes a segment unless: (1) all registered readers have consumed it
  (`fully_consumed`), (2) it is at least 300s old, and (3) its POSIX writer PID
  definitively exited (`ESRCH`).

[Timeouts & Locks]
- A 1.0s monotonic deadline (`deadline = time.monotonic() + 1`) bounds the
  directory scan and the delete loop, and caps the `reader_guard` lock wait at
  100ms and the requested `cursor_snapshot` budget at 200ms.
- That 1.0s is not a hard ceiling for the whole call: `cursor_snapshot` keeps
  the project-wide `3.0s` floor on each reader database wait (WAL plus
  `busy_timeout=3000`, see `reader_registry.py`), which a contended reader
  store can spend inside the 200ms request. Prune stays conservative either
  way - an unknown cursor snapshot deletes nothing - so the overrun costs
  latency, never evidence. Aligning the two budgets is tracked in
  `docs/outbound-reconciliation-followup.md`.

[Side-Effect Policy]
- Unlinks only verified consumed log segments and atomically updates
  `retention-state.json`.
"""
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
    """Only positive POSIX PIDs with a definitive ESRCH are safe to prune."""
    if os.name != "posix" or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except TimeoutError:
        raise
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
    if root.is_symlink():
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
                if path.is_symlink() or path.parent.is_symlink():
                    incomplete = True
                    continue
                try:
                    current = path.stat(follow_symlinks=False)
                    if (not stat.S_ISREG(current.st_mode)
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
    except TimeoutError:
        raise
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
    except TimeoutError:
        raise
    except (OSError, ValueError):
        result['limited'] = True
        result['marker_failed'] = True
    return result
