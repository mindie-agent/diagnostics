"""Offline log retention and backpressure marker management.

[Role]
Prunes consumed, aged, or over-quota JSONL event segments under `<root>/events`
whose writer process has exited, and writes `<root>/retention-state.json` to
signal backpressure to active log writers when retention limits remain exceeded.

[Invariants]
- Called by maintenance/worker flows and by a rate-limited recovery path from a
  blocked log writer, never inline during ordinary tool operations.
- Never deletes a segment unless: (1) all registered readers have consumed it
  (`fully_consumed`), (2) it is at least 300s old, and (3) its writer PID
  definitively exited (POSIX `ESRCH`, Windows `OpenProcess` +
  `GetExitCodeProcess`; no console subprocess probe).
- File identity is re-read with `os.stat(follow_symlinks=False)` whenever a
  `DirEntry` reports zero `st_dev`/`st_ino` (Windows `DirEntry.stat` leaves
  them zero), so cursor matching and the pre-unlink check agree.
- A stale `blocked=True` marker may be recovered at most once per 300s. After
  acquiring the exclusive claim the claimant re-reads the marker, verifies
  `blocked`/`checked_at` are still stale, then atomically advances
  `checked_at` before scanning; a caller holding a stale pre-claim read
  therefore cannot run a second recovery inside the same cooldown.

[Timeouts & Locks]
- The caller's monotonic budget (default 5.0s) reserves 50ms for releasing the
  registry lock and writing the retention marker. Scanning and deletion share
  the remainder; `reader_guard` waits at most 100ms.
- `cursor_snapshot` receives the actual remaining budget. Its 3.0s SQLite
  wait plus cleanup must fit before it opens each reader; otherwise cursors
  are unknown and this call deletes nothing. Callers may explicitly provide
  a smaller budget; that call explicitly reports incomplete reader inspection.
- The writer-triggered recovery prune is capped at 0.5s and is admitted only
  through `claim_recovery`'s exclusive claim file; no unbounded work is added
  to the emit path.

[Side-Effect Policy]
- Unlinks only verified consumed log segments, writes a temporary exclusive
  recovery claim, and atomically updates `retention-state.json`.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import stat
import uuid
from pathlib import Path
import re
import time
from types import SimpleNamespace

from .reader_registry import reader_guard, cursor_snapshot, fully_consumed

_FILE = re.compile(r"(\d+)-[0-9a-f]{32}\.jsonl(?:\.[1-3])?\Z")
_CLEANUP_S = .05
_FOLDER_LIMIT = 4096
# Metadata-only DirEntry scans are cheap; a 4096-entry cutoff starved old
# segments on Windows and in state/diagnostics backlogs when scandir returned
# newest entries first. Bound one prune pass, never a single component.
_ENTRY_LIMIT = 16384
_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _kernel32():
    """Configured kernel32 functions; never a tasklist/powershell call."""
    try:
        dll = ctypes.WinDLL("kernel32", use_last_error=True)
    except TimeoutError:
        raise
    except (AttributeError, OSError):
        try:
            dll = ctypes.windll.kernel32
        except TimeoutError:
            raise
        except (AttributeError, OSError):
            return None
    # CDLL.__getattr__ returns a fresh function object on every access, so the
    # prototypes below must be attached to retained objects. HANDLE is
    # pointer-sized; without restype ctypes truncates it to c_int.
    open_process = dll.OpenProcess
    get_exit_code = dll.GetExitCodeProcess
    close_handle = dll.CloseHandle
    try:
        open_process.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        open_process.restype = ctypes.c_void_p
        get_exit_code.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        get_exit_code.restype = ctypes.c_int
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
    except (AttributeError, TypeError):
        pass
    return SimpleNamespace(
        OpenProcess=open_process,
        GetExitCodeProcess=get_exit_code,
        CloseHandle=close_handle,
    )


def _writer_exited(pid: int) -> bool:
    """True only when the OS proves this writer PID no longer exists."""
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except TimeoutError:
            raise
        except (OSError, ValueError, OverflowError):
            return False
        return False
    if os.name != "nt":
        return False
    kernel32 = _kernel32()
    if kernel32 is None:
        # Non-Windows host masquerading as nt (tests) or a broken ctypes table.
        return False
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # 87 means the PID does not exist; 5 means a protected or other-user
        # live process owns it. Any other observation is not proof of exit.
        return ctypes.get_last_error() == _ERROR_INVALID_PARAMETER
    try:
        exit_code = ctypes.c_uint32(0)
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value != _STILL_ACTIVE
        return False
    finally:
        kernel32.CloseHandle(handle)


def _write_marker(root, data):
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


def _marker(root, result, blocked, now):
    _write_marker(root, {
        "schema": 1, "blocked": bool(blocked), "limited": bool(result['limited']),
        "checked_at": now, "remaining_bytes": result['remaining_bytes'],
        "remaining_files": result['remaining_files'],
        "unread_files": result['unread_files'],
        "active_or_unknown_files": result['active_or_unknown_files'],
    })


def _marker_state(path):
    """Read a bounded regular retention marker without following a symlink."""
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 2048:
            return None
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except TimeoutError:
        raise
    except OSError:
        return None
    try:
        raw = os.read(descriptor, 2049)
    except TimeoutError:
        raise
    except OSError:
        return None
    finally:
        os.close(descriptor)
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, UnicodeError):
        return None
    if not isinstance(data, dict) or data.get('schema') != 1:
        return None
    return data


def claim_recovery(root, *, cooldown: float = 300.0, now=None) -> bool:
    """Atomically claim a stale blocked-marker recovery pass.

    The marker is re-read under the exclusive claim file before `checked_at`
    is advanced, so a caller that read the stale marker before another caller
    claimed cannot run a second recovery prune inside the same cooldown. A
    crashed claimant is reclaimable once its claim file is itself older than
    the cooldown.
    """
    root = Path(root)
    if root.is_symlink():
        return False
    state = _marker_state(root / 'retention-state.json')
    if state is None or state.get('blocked') is not True:
        return False
    checked = state.get('checked_at')
    if isinstance(checked, bool) or not isinstance(checked, (int, float)):
        return False
    if not math.isfinite(checked):
        return False
    now = time.time() if now is None else float(now)
    cooldown = max(0.0, float(cooldown))
    if now - checked < cooldown:
        return False
    claim = root / '.retention-recovery'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)

    def acquire():
        try:
            return os.open(claim, flags, 0o600)
        except FileExistsError:
            try:
                stale = now - claim.lstat().st_mtime >= cooldown
            except TimeoutError:
                raise
            except OSError:
                return None
            if not stale:
                return None
            try:
                claim.unlink()
                return os.open(claim, flags, 0o600)
            except TimeoutError:
                raise
            except (FileExistsError, OSError):
                return None

    try:
        descriptor = acquire()
    except TimeoutError:
        raise
    except OSError:
        return False
    if descriptor is None:
        return False
    try:
        os.close(descriptor)
        # Two callers can read the stale marker before either claims. Re-read
        # it under the exclusive claim so the loser sees the winner's advanced
        # checked_at (or a cleared blocked flag) and stays inside the cooldown.
        current = _marker_state(root / 'retention-state.json')
        checked = current.get('checked_at') if current is not None else None
        if (current is None or current.get('blocked') is not True
                or isinstance(checked, bool)
                or not isinstance(checked, (int, float))
                or not math.isfinite(checked) or now - checked < cooldown):
            claim.unlink()
            return False
        state = dict(current)
        state['checked_at'] = now
        _write_marker(root, state)
        return True
    except TimeoutError:
        raise
    except (OSError, ValueError):
        try:
            claim.unlink()
        except TimeoutError:
            raise
        except OSError:
            pass
        return False


def release_recovery(root) -> None:
    try:
        (Path(root) / '.retention-recovery').unlink()
    except TimeoutError:
        raise
    except OSError:
        pass


def prune(root: str | Path, *, max_bytes: int = 128 * 1024 * 1024,
          max_age_days: int = 7, max_files: int = 512, clock=time.time, queue=None,
          timeout: float = 5.0) -> dict:
    root = Path(os.path.realpath(root))
    events = root / 'events'
    result = {"removed_files": 0, "removed_bytes": 0, "remaining_bytes": 0,
              "remaining_files": 0, "limited": False, "reader_unknown": False,
              "unread_files": 0, "active_or_unknown_files": 0}
    if not root.exists():
        return result
    deadline = time.monotonic() + max(0, float(timeout) - _CLEANUP_S)
    files, folders_seen, entries_seen, incomplete = [], 0, 0, False
    total, remaining, now = 0, 0, clock()
    try:
        if events.is_symlink():
            raise ValueError("retention event directory must not be a symlink")
        if events.is_dir():
            with os.scandir(events) as folders:
                for folder in folders:
                    folders_seen += 1
                    if folders_seen > _FOLDER_LIMIT or time.monotonic() >= deadline:
                        incomplete = True
                        break
                    if folder.is_symlink() or not folder.is_dir(follow_symlinks=False):
                        continue
                    with os.scandir(folder.path) as entries:
                        for entry in entries:
                            entries_seen += 1
                            if entries_seen > _ENTRY_LIMIT or time.monotonic() >= deadline:
                                incomplete = True
                                break
                            if (entry.is_symlink() or not _FILE.fullmatch(entry.name)
                                    or not entry.is_file(follow_symlinks=False)):
                                continue
                            # DirEntry.stat avoids a path lookup per candidate,
                            # but Windows leaves st_dev/st_ino zero there while
                            # Path.stat/os.stat fills them from the file handle.
                            try:
                                info = entry.stat(follow_symlinks=False)
                                if info.st_ino == 0 or info.st_dev == 0:
                                    info = os.stat(entry.path, follow_symlinks=False)
                            except TimeoutError:
                                raise
                            except OSError:
                                continue
                            files.append((info.st_mtime, info.st_size, Path(entry.path),
                                          info.st_ino, info.st_dev, info.st_mtime_ns))
                    if incomplete:
                        break
        total, remaining = sum(row[1] for row in files), len(files)
        with reader_guard(root, timeout=min(.1, max(0, deadline - time.monotonic())),
                          queue_path=queue.path if queue is not None else None) as state:
            cursors = cursor_snapshot(
                state['readers'], timeout=.2,
                remaining_budget=max(0, deadline - time.monotonic()),
            ) if state['status'] == 'ok' else None
            result['reader_unknown'] = cursors is None
            if state.get('unreadable_readers'):
                result['reader_unknown'] = True
                result['reader_error'] = state.get('reader_error')
                if 'errno' in state:
                    result['reader_errno'] = state['errno']
            if state.get('policy_unavailable'):
                result['policy_unavailable'] = True
                if state.get('reader_error') and 'reader_error' not in result:
                    result['reader_error'] = state['reader_error']
            if cursors is None:
                result['unread_files'] = len(files)
            exited = {}
            for modified, size, path, inode, device, modified_ns in sorted(files) if cursors is not None else ():
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
                # Keep `<root>/events` from accumulating thousands of empty
                # component directories after their last segment is pruned.
                if path.parent.parent == events:
                    try:
                        path.parent.rmdir()
                    except TimeoutError:
                        raise
                    except OSError:
                        pass
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
