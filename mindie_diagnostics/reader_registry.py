"""Explicit local log readers and bounded, conservative cursor inspection.

Registration is lifecycle-owned, never inferred from age or filesystem discovery.
The registry lock spans a caller's reclaim operation so registration cannot race
with deletion of evidence that a newly registered reader still needs.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
import uuid

_MAX_READERS = 32
_MAX_REGISTRY_BYTES = 32768
_MAX_CURSOR_ROWS = 10000


def _real_path(value) -> Path:
    path = Path(os.path.abspath(value))
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("diagnostic reader paths must not traverse symlinks")
    return path


def _regular(info) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and (os.name == "nt" or
                 (info.st_uid == os.getuid() and not stat.S_IMODE(info.st_mode) & 0o077)))


def _read(path: Path, limit: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(fd)
        if not _regular(info) or info.st_size > limit:
            raise ValueError("unsafe diagnostic metadata")
        data = os.read(fd, limit + 1)
        if len(data) > limit:
            raise ValueError("diagnostic metadata exceeds bound")
        return data
    finally:
        os.close(fd)


def _folder(root) -> Path:
    root = _real_path(root)
    # Only create our missing root/registry directories, never chmod parents.
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    folder = root / ".readers"
    folder.mkdir(mode=0o700, exist_ok=True)
    info = folder.lstat()
    if not stat.S_ISDIR(info.st_mode) or (os.name != "nt" and
            (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077)):
        raise ValueError("diagnostic reader directory is not private")
    return folder


def _readers(folder: Path) -> list[str]:
    try:
        data = json.loads(_read(folder / "registry.json", _MAX_REGISTRY_BYTES))
    except FileNotFoundError:
        return []
    if (not isinstance(data, dict) or data.get("schema") != 1
            or not isinstance(data.get("readers"), list)
            or len(data["readers"]) > _MAX_READERS):
        raise ValueError("invalid diagnostic reader registry")
    readers = data["readers"]
    for value in readers:
        if (not isinstance(value, str) or not value or len(value) > 1024
                or not Path(value).is_absolute() or "\0" in value):
            raise ValueError("invalid diagnostic reader binding")
    if len(set(readers)) != len(readers):
        raise ValueError("duplicate diagnostic reader binding")
    return readers


def _save(folder: Path, readers: list[str]) -> None:
    data = json.dumps({"schema": 1, "readers": readers}, separators=(",", ":")).encode()
    if len(readers) > _MAX_READERS or len(data) > _MAX_REGISTRY_BYTES:
        raise ValueError("diagnostic reader registry is full")
    temporary = folder / (".registry-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, folder / "registry.json")
    finally:
        temporary.unlink(missing_ok=True)


def _queue_path(queue_path) -> str:
    path = _real_path(queue_path)
    if len(str(path)) > 1024 or not _regular(path.lstat()):
        raise ValueError("diagnostic reader must be an existing private database")
    return str(path)


def _try_lock(fd: int) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def reader_guard(root, timeout: float = .1, queue_path=None):
    """Hold the registry lock; uncertainty is explicit and never an empty set."""
    deadline = time.monotonic() + max(0, min(float(timeout), .2))
    fd = None
    locked = False
    state = {"status": "unknown", "readers": []}
    try:
        folder = _folder(root)
        fd = os.open(folder / "registry.lock", os.O_RDWR | os.O_CREAT
                     | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o600)
        if not _regular(os.fstat(fd)):
            raise ValueError("unsafe diagnostic reader lock")
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"0")
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic reader registry is busy")
            time.sleep(min(.005, max(0, deadline - time.monotonic())))
        locked = True
        readers = _readers(folder)
        if queue_path is not None:
            binding = _queue_path(queue_path)
            if binding not in readers:
                readers.append(binding)
                _save(folder, readers)
        state = {"status": "ok", "readers": readers}
    except (OSError, ValueError, TypeError, UnicodeError):
        pass
    try:
        yield state
    finally:
        if fd is not None:
            try:
                if locked:
                    _unlock(fd)
            finally:
                os.close(fd)


def register_reader(root, queue_path) -> None:
    with reader_guard(root, queue_path=queue_path) as state:
        if state["status"] != "ok":
            raise RuntimeError("diagnostic reader registration unavailable")


def unregister_reader(root, queue_path) -> None:
    # A missing queue may be deliberately unregistered after its reader stopped.
    binding = str(_real_path(queue_path))
    with reader_guard(root) as state:
        if state["status"] != "ok":
            raise RuntimeError("diagnostic reader registration unavailable")
        if binding in state["readers"]:
            _save(_real_path(root) / ".readers",
                  [value for value in state["readers"] if value != binding])


def cursor_snapshot(readers, timeout: float = .1):
    """Read existing cursor tables; never create/migrate a reader database."""
    if not isinstance(readers, list) or len(readers) > _MAX_READERS:
        return None
    deadline = time.monotonic() + max(0, min(float(timeout), .2))
    snapshots = []
    try:
        for binding in readers:
            if time.monotonic() >= deadline:
                return None
            path = _real_path(binding)
            info = path.lstat()
            if not _regular(info):
                return None
            remaining = max(0, deadline - time.monotonic())
            db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=remaining)
            try:
                db.execute("PRAGMA query_only=ON")
                db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 100)
                table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cursors'").fetchone()
                rows = db.execute("SELECT path,inode,offset FROM cursors LIMIT ?", (_MAX_CURSOR_ROWS + 1,)).fetchall() if table else []
                if len(rows) > _MAX_CURSOR_ROWS:
                    return None
                snapshot = {}
                for name, inode, offset in rows:
                    if (not isinstance(name, str) or len(name) > 4096
                            or not isinstance(inode, str) or not inode.isdecimal()
                            or type(offset) is not int or offset < 0):
                        return None
                    snapshot[name] = {"inode": inode, "offset": offset}
                current = path.lstat()
                if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                    return None
                snapshots.append(snapshot)
            finally:
                db.close()
        return snapshots
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return None


def fully_consumed(snapshot, path, inode, size) -> bool:
    if snapshot is None:
        return False
    for reader in snapshot:
        cursor = reader.get(str(path), {})
        if cursor.get("inode") != str(inode) or cursor.get("offset", -1) < size:
            return False
    return True


def pressure_blocked(root) -> bool:
    """Offline-maintenance backpressure, not an instantaneous global quota."""
    try:
        path = _real_path(root) / "retention-state.json"
        data = json.loads(_read(path, 2048))
        if (not isinstance(data, dict) or data.get("schema") != 1
                or type(data.get("blocked")) is not bool
                or type(data.get("limited")) is not bool):
            return True
        return data["blocked"]
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError, UnicodeError):
        return True
