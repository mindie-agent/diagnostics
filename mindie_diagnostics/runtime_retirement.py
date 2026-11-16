"""Bounded two-phase retirement helpers for immutable runtime generations.

[Role]
  Provides atomic rename-to-trash retirement primitives, Windows extended-path
  normalization, bounded read-only-tolerant tree removal, and marked Windows
  profile orphan cleanup for `reporting_runtime`.

[Invariants]
  * A caller renames an eligible entry to `.trash-*` before this module deletes
    anything; a failed rename therefore leaves the original entry intact.
  * Removal never follows symlinks and never exceeds its monotonic deadline.
  * Orphan profile cleanup deletes only sibling directories naming an existing
    `profile.json` whose recorded policy root no longer exists.

[Timeouts & Locks]
  * Every traversal checks the caller's absolute monotonic deadline.
  * No lock is acquired here; callers own `runtime_transaction`.

[Side-Effect Policy]
  * Unlinks only owned generation/trash directories and marked orphan profile
    directories; read-only files are made writable before a bounded retry.
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


class _Retirement(list):
    """List-compatible generation names plus deferred/staging detail."""

    def __init__(self, retired=(), deferred=(), staging=()):
        super().__init__(retired)
        self.deferred = tuple(deferred)
        self.staging = tuple(staging)


def _native_path(path):
    """Add the Windows extended prefix for deep venv artifacts near MAX_PATH."""
    path = Path(path)
    text = str(path)
    if (os.name == "nt" and len(text) >= 240
            and not text.startswith(_EXTENDED_PREFIX)):
        return Path(_EXTENDED_PREFIX + text)
    return path


def _unlink(path):
    try:
        os.unlink(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        os.unlink(path)


def _remove_tree(path, deadline):
    """Bounded symlink-safe removal; False when the monotonic budget expires."""
    path = _native_path(path)
    if time.monotonic() >= deadline:
        return False
    try:
        with os.scandir(path) as scan:
            entries = list(scan)
    except FileNotFoundError:
        return True
    except NotADirectoryError:
        _unlink(path)
        return True
    for entry in entries:
        if time.monotonic() >= deadline:
            return False
        if entry.is_dir(follow_symlinks=False):
            if not _remove_tree(Path(entry.path), deadline):
                return False
        else:
            _unlink(entry.path)
    try:
        os.rmdir(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        os.rmdir(path)
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


def _profile_idle(entry):
    for name in ("python.exe", "pythonw.exe"):
        candidate = Path(entry) / "venv" / "Scripts" / name
        if candidate.exists() and not _image_available(candidate):
            return False
    return True


def prune_orphan_windows_profiles(active_root, *, budget_seconds=3.0):
    """Retire only marked sibling profiles whose recorded policy root is gone.

    An unmarked directory is never deleted. A still-running pythonw.exe is
    detected by an exclusive image open and by a failed atomic rename; either
    way the profile is left untouched.
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
    try:
        entries = sorted(parent.iterdir(), key=lambda path: path.name)
    except TimeoutError:
        raise
    except OSError:
        return _Retirement()
    deadline = time.monotonic() + max(0.0, float(budget_seconds))
    retired, deferred = [], []
    for entry in entries:
        if time.monotonic() >= deadline:
            break
        if entry.name == active.name or not _PROFILE_NAME.fullmatch(entry.name):
            continue
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            if not _owned(entry.lstat()):
                continue
            marker = _load_json(entry / "profile.json", _read_regular_bounded, _OUT_CAP)
            target = marker.get("root")
            if marker.get("schema") != 1 or not isinstance(target, str):
                continue
            if _as_local_absolute(target) is None or Path(target).exists():
                continue
            if not _profile_idle(entry):
                deferred.append(entry.name)
                continue
            trash = parent / f".trash-{entry.name[:16]}-{uuid.uuid4().hex[:8]}"
            os.replace(_native_path(entry), _native_path(trash))
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
    return _Retirement(retired, deferred)
