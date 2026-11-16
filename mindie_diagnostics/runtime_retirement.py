"""Bounded two-phase retirement helpers for immutable runtime generations.

[Role]
  Provides atomic rename-to-trash retirement primitives, Windows extended-path
  normalization, bounded read-only-tolerant tree removal, and marked Windows
  profile orphan cleanup for `reporting_runtime`.

[Invariants]
  * A caller renames an eligible entry to `.trash-*` before this module deletes
    anything; a failed rename therefore leaves the original entry intact.
  * Removal never follows symlinks or NTFS junctions: a junction is removed as
    a link with `os.rmdir`, and its target children are never traversed.
  * Orphan profile cleanup deletes only sibling directories naming an existing
    `profile.json` whose recorded policy root no longer exists.
  * A profile whose recorded root is on an unavailable volume anchor is
    deferred (`deferred_volume_unavailable`), never deleted.
  * Windows liveness checks inspect `<profile>/<source_hash>/venv/Scripts`,
    the real per-generation layout, before retiring a profile.
  * Interrupted `.trash-*` removals are resumed at the start of every profile
    cleanup pass.

[Timeouts & Locks]
  * Every traversal checks the caller's absolute monotonic deadline while
    iterating `os.scandir`; no unbounded `list()`/`sorted()` scan precedes the
    first deadline check, and `_ENTRY_LIMIT` bounds each directory.
  * No lock is acquired here; callers own `runtime_transaction`.

[Side-Effect Policy]
  * Unlinks only owned generation/trash directories, symlink/junction links,
    and marked orphan profile directories; read-only files are made writable
    before a bounded retry.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import time
import uuid

_EXTENDED_PREFIX = "\\\\?\\"
_PROFILE_NAME = re.compile(r"[0-9a-f]{20}\Z")
_SOURCE_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE_ANCHOR = re.compile(r"^[A-Za-z]:[\\/]")
_ENTRY_LIMIT = 16384


class _Retirement(list):
    """List-compatible generation names plus deferred/staging detail."""

    def __init__(self, retired=(), deferred=(), staging=(), *,
                 volume_unavailable=(), swept=()):
        super().__init__(retired)
        self.deferred = tuple(deferred)
        self.staging = tuple(staging)
        self.volume_unavailable = tuple(volume_unavailable)
        self.deferred_volume_unavailable = self.volume_unavailable
        self.swept = tuple(swept)


def _native_path(path):
    """Add the Windows extended prefix for deep venv artifacts near MAX_PATH."""
    path = Path(path)
    text = str(path)
    if (os.name == "nt" and len(text) >= 240
            and not text.startswith(_EXTENDED_PREFIX)):
        return Path(_EXTENDED_PREFIX + text)
    return path


def _is_junction(path):
    """True for an NTFS directory junction; false where the API is absent."""
    return bool(getattr(os.path, "isjunction", lambda _: False)(path))


def _unlink(path):
    try:
        os.unlink(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        os.unlink(path)


def _rmdir(path):
    try:
        os.rmdir(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        os.rmdir(path)


def _remove_link(path, *, junction):
    """Remove only the link itself; a junction target is never traversed."""
    if junction and not Path(path).is_symlink():
        _rmdir(path)
    else:
        _unlink(path)


def _bounded_entries(path, deadline, limit=_ENTRY_LIMIT):
    """Collect at most `limit` DirEntries, checking the deadline while scanning.

    The returned boolean is false when the scan was cut short by the deadline or
    the entry cap; callers may still process the bounded prefix and must stop
    before starting new work once the deadline is spent.
    """
    entries = []
    with os.scandir(path) as scan:
        for entry in scan:
            if time.monotonic() >= deadline or len(entries) >= limit:
                return entries, False
            entries.append(entry)
    return entries, True


def _remove_tree(path, deadline):
    """Bounded symlink/junction-safe removal; False when the budget expires."""
    path = _native_path(path)
    if time.monotonic() >= deadline:
        return False
    if _is_junction(path):
        if Path(path).is_symlink():
            _unlink(path)
        else:
            _rmdir(path)
        return True
    if Path(path).is_symlink():
        _unlink(path)
        return True
    try:
        with os.scandir(path) as scan:
            for entry in scan:
                if time.monotonic() >= deadline:
                    return False
                junction = _is_junction(entry.path)
                if junction or entry.is_symlink():
                    _remove_link(entry.path, junction=junction)
                elif entry.is_dir(follow_symlinks=False):
                    if not _remove_tree(Path(entry.path), deadline):
                        return False
                else:
                    _unlink(entry.path)
    except FileNotFoundError:
        return True
    except NotADirectoryError:
        _unlink(path)
        return True
    _rmdir(path)
    return True


def _image_available(path):
    """Windows: True unless a live process has mapped this image.

    `CreateFileW` with an empty share mode fails with
    `ERROR_SHARING_VIOLATION=32` while python.exe/pythonw.exe is running. Any
    other outcome (including ACL denial or a missing file) is not proof of a
    live process and leaves the normal atomic-rename guard in charge.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        close = kernel32.CloseHandle
        create.argtypes = (
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        )
        create.restype = wintypes.HANDLE
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        handle = create(str(_native_path(path)), 0x80000000, 0, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            return ctypes.get_last_error() != 32
    except TimeoutError:
        raise
    except (AttributeError, OSError, TypeError, ValueError):
        return True
    close(handle)
    return True


def _image_candidates(profile, deadline):
    """Yield `<profile>/<source_hash>/venv/Scripts` images plus legacy depth."""
    try:
        children, _complete = _bounded_entries(profile, deadline)
    except TimeoutError:
        raise
    except OSError:
        return
    for child in children:
        if time.monotonic() >= deadline:
            return
        if child.is_symlink() or not child.is_dir(follow_symlinks=False):
            continue
        if not _SOURCE_HASH.fullmatch(child.name):
            continue
        for name in ("python.exe", "pythonw.exe"):
            yield Path(child.path) / "venv" / "Scripts" / name
    # Pre-0.5.14 profiles kept a single venv directly under the profile.
    for name in ("python.exe", "pythonw.exe"):
        yield Path(profile) / "venv" / "Scripts" / name


def _profile_idle(entry, deadline=None):
    if deadline is None:
        deadline = time.monotonic() + 3.0
    for candidate in _image_candidates(entry, deadline):
        if time.monotonic() >= deadline:
            return False
        try:
            if candidate.exists() and not _image_available(candidate):
                return False
        except TimeoutError:
            raise
        except OSError:
            return False
    return True


def _volume_available(target):
    """False when the target's drive/anchor itself is not currently mounted."""
    anchor = target.anchor
    if not anchor:
        match = _DRIVE_ANCHOR.match(str(target))
        if match is None:
            return True
        anchor = match.group(0)
    try:
        return Path(anchor).exists()
    except TimeoutError:
        raise
    except OSError:
        return False


def _sweep_trash(parent, deadline):
    """Finish interrupted `.trash-*` removals from an earlier profile pass."""
    from mindie_diagnostics.fallback import _owned

    swept, deferred = [], []
    try:
        entries, _complete = _bounded_entries(parent, deadline)
    except TimeoutError:
        raise
    except OSError:
        return swept, deferred
    for entry in entries:
        if time.monotonic() >= deadline:
            if entry.name.startswith(".trash-"):
                deferred.append(entry.name)
            break
        if not entry.name.startswith(".trash-"):
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            continue
        try:
            if not _owned(entry.stat(follow_symlinks=False)):
                continue
            if _remove_tree(Path(entry.path), deadline):
                swept.append(entry.name)
            else:
                deferred.append(entry.name)
        except TimeoutError:
            raise
        except OSError:
            deferred.append(entry.name)
    return swept, deferred


def prune_orphan_windows_profiles(active_root, *, budget_seconds=3.0):
    """Retire only marked sibling profiles whose recorded policy root is gone.

    An unmarked directory is never deleted. A still-running pythonw.exe is
    detected by an exclusive image open and by a failed atomic rename; either
    way the profile is left untouched. Existing `.trash-*` directories are
    swept first so an interrupted pass resumes instead of leaking forever.
    """
    if os.name != "nt":
        return _Retirement()
    from mindie_diagnostics.fallback import (
        _as_local_absolute,
        _owned,
        _read_regular_bounded,
    )
    from .reporting_runtime import _OUT_CAP, _generations, _load_json

    active = Path(_generations(Path(active_root)))
    parent = active.parent
    if parent.is_symlink() or not parent.is_dir():
        return _Retirement()
    deadline = time.monotonic() + max(0.0, float(budget_seconds))
    retired, deferred, volume_unavailable = [], [], []
    swept, sweep_deferred = _sweep_trash(parent, deadline)
    deferred.extend(sweep_deferred)
    try:
        entries, _complete = _bounded_entries(parent, deadline)
    except TimeoutError:
        raise
    except OSError:
        return _Retirement(retired, deferred, volume_unavailable=volume_unavailable,
                           swept=swept)
    for entry in entries:
        if time.monotonic() >= deadline:
            break
        if entry.name == active.name or not _PROFILE_NAME.fullmatch(entry.name):
            continue
        if entry.is_symlink() or _is_junction(entry.path):
            continue
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            if not _owned(entry.stat(follow_symlinks=False)):
                continue
            marker = _load_json(Path(entry.path) / "profile.json",
                                _read_regular_bounded, _OUT_CAP)
            target = marker.get("root")
            if marker.get("schema") != 1 or not isinstance(target, str):
                continue
            target_path = _as_local_absolute(target)
            if target_path is None or target_path.exists():
                continue
            if not _volume_available(target_path):
                deferred.append(entry.name)
                volume_unavailable.append(entry.name)
                continue
            if not _profile_idle(Path(entry.path), deadline):
                deferred.append(entry.name)
                continue
            trash = parent / f".trash-{entry.name[:16]}-{uuid.uuid4().hex[:8]}"
            os.replace(_native_path(entry.path), _native_path(trash))
        except TimeoutError:
            raise
        except (OSError, ValueError, RuntimeError):
            continue
        retired.append(entry.name)
        try:
            if not _remove_tree(trash, deadline):
                deferred.append(trash.name)
        except TimeoutError:
            raise
        except OSError:
            deferred.append(trash.name)
    return _Retirement(retired, deferred, volume_unavailable=volume_unavailable,
                       swept=swept)
