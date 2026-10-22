"""Lifecycle-owned log reader registry and bounded cursor inspection.

[Role]
Coordinates log writers (`logging.py`) and maintenance pruning (`maintenance.py`)
with active log consumers (`ingestion.py`) via `<root>/.readers/registry.json`
and read-only SQLite cursor snapshots.

[Invariants]
- Reader registration is explicit and lifecycle-owned, never inferred from file
  age or directory scanning.
- Lock acquisition failure or timeout yields `status="unknown"`, never an empty
  reader list, so log segments are never prematurely recycled or deleted.

[Timeouts & Locks]
- `reader_guard` acquires `flock`/`msvcrt` on `.readers/registry.lock` with a
  bounded timeout (capped at 2.0s; 50ms when called from `_Handler._select`).
- `cursor_snapshot` queries each registered reader SQLite database in `?mode=ro`
  with `PRAGMA query_only=ON` and a 100-opcode progress handler bound to the
  shared deadline.

[Side-Effect Policy]
- `cursor_snapshot` and `PressureReader` are strictly read-only and never create
  or migrate reader databases.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from .fallback import _private_mkdir
import sqlite3
import stat
import time
import uuid

_MAX_READERS = 32
_MAX_REGISTRY_BYTES = 32768
_MAX_CURSOR_ROWS = 10000


def _real_path(value) -> Path:
    return Path(os.path.abspath(value))


def _regular(info) -> bool:
    return stat.S_ISREG(info.st_mode)


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
    _private_mkdir(root, parents=True, exist_ok=True)
    folder = root / ".readers"
    if folder.is_symlink():
        raise ValueError("diagnostic reader directory is not private")
    _private_mkdir(folder, exist_ok=True)
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
    except TimeoutError:
        raise
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


def _active_reader(binding: str) -> bool:
    """Prune deleted reader DBs or disabled profile queues so retention never self-bricks."""
    path = _real_path(binding)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except TimeoutError:
        raise
    except OSError:
        return True
    if not _regular(info):
        return True
    if path.name == "reporter.sqlite3" and path.parent.suffix == ".state":
        from .fallback import PolicyUnavailable, read_policy
        try:
            policy = read_policy(path.parent.with_suffix(".json"))
        except PolicyUnavailable:
            return True
        if policy is None or policy.get("decision") != "enabled":
            return False
    return True


@contextmanager
def reader_guard(root, timeout: float = 2.0, queue_path=None):
    """Hold the registry lock; uncertainty is explicit and never an empty set."""
    deadline = time.monotonic() + max(0, min(float(timeout), 2.0))
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
        existing = _readers(folder)
        binding = _queue_path(queue_path) if queue_path is not None else None
        readers = [r for r in existing if r == binding or _active_reader(r)]
        if binding is not None and binding not in readers:
            readers.append(binding)
        if readers != existing:
            _save(folder, readers)
        state = {"status": "ok", "readers": readers}
    except TimeoutError:
        state = {"status": "unknown", "reason": "timeout", "readers": []}
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


def register_reader(root, queue_path, *, timeout: float = 2.0) -> None:
    with reader_guard(root, timeout=timeout, queue_path=queue_path) as state:
        if state["status"] != "ok":
            if state.get("reason") == "timeout":
                raise TimeoutError("diagnostic reader registry is busy")
            raise RuntimeError("diagnostic reader registration unavailable")


def unregister_reader(root, queue_path, *, timeout: float = 2.0) -> None:
    # A missing queue may be deliberately unregistered after its reader stopped.
    binding = str(_real_path(queue_path))
    with reader_guard(root, timeout=timeout) as state:
        if state["status"] != "ok":
            if state.get("reason") == "timeout":
                raise TimeoutError("diagnostic reader registry is busy")
            raise RuntimeError("diagnostic reader registration unavailable")
        if binding in state["readers"]:
            _save(_real_path(root) / ".readers",
                  [value for value in state["readers"] if value != binding])


def cursor_snapshot(readers, timeout: float = 2.0):
    """Read existing cursor tables; never create/migrate a reader database."""
    if not isinstance(readers, list) or len(readers) > _MAX_READERS:
        return None
    deadline = time.monotonic() + max(0, min(float(timeout), 2.0))
    snapshots = []
    try:
        for binding in readers:
            if time.monotonic() >= deadline:
                return None
            if not _active_reader(binding):
                continue
            path = _real_path(binding)
            if not _regular(path.lstat()):
                return None
            remaining = max(0, deadline - time.monotonic())
            db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=max(3.0, remaining))
            try:
                db.execute("PRAGMA query_only=ON")
                db.execute("PRAGMA busy_timeout=3000")
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
                snapshots.append(snapshot)
            finally:
                db.close()
        return snapshots
    except TimeoutError:
        raise
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


class PressureReader:
    """Stat-cached reader for `retention-state.json` backpressure."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'retention-state.json'
        self.signature = object()
        self.blocked = False

    def __call__(self, *, force=False):
        try:
            info = self.path.lstat()
            signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        except FileNotFoundError:
            signature = None
        except TimeoutError:
            raise
        except OSError:
            return False
        if force or signature != self.signature:
            self.blocked = pressure_blocked(self.root)
            self.signature = signature
        return self.blocked


def pressure_blocked(root) -> bool:
    """Offline-maintenance backpressure, not an instantaneous global quota."""
    try:
        path = _real_path(root) / "retention-state.json"
        data = json.loads(_read(path, 2048))
        return isinstance(data, dict) and data.get("schema") == 1 and data.get("blocked") is True
    except TimeoutError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError):
        return False
