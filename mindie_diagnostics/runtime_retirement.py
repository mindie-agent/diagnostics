"""Bounded two-phase retirement helpers for immutable runtime generations.

[Role]
  Provides atomic rename-to-trash retirement primitives, Windows extended-path
  normalization, bounded read-only-tolerant tree removal, and marked Windows
  profile orphan cleanup for `reporting_runtime`.

[Invariants]
  * A caller renames an eligible entry to `.trash-*` before this module deletes
    anything; a failed rename therefore leaves the original entry intact.
  * Link classification uses one `os.lstat`: `stat.S_ISLNK` is a symlink, and
    Windows `st_reparse_tag`/`st_file_attributes` marks a junction/reparse
    point. This works on Python 3.11, which has no `os.path.isjunction` (and
    3.12's `ntpath.isjunction` swallows `OSError`, including `TimeoutError`).
  * Removal never follows symlinks or NTFS junctions: a junction is removed as
    a link with `os.rmdir`, and its target children are never traversed. A
    vanished child never aborts removal of its siblings or the root.
  * Orphan profile cleanup deletes only sibling directories naming an existing
    `profile.json` whose recorded policy root no longer exists.
  * A profile whose recorded root is on an unavailable volume anchor is
    deferred (`deferred_volume_unavailable`), never deleted.
  * A profile is retired only after a complete process-image scan. Windows
    liveness checks inspect `<profile>/<source_hash>/venv/Scripts`, the real
    per-generation layout; legacy profiles are scanned too.
  * Interrupted `.trash-*` removals are resumed at the start of every profile
    cleanup pass.

[Timeouts & Locks]
  * Every traversal checks the caller's absolute monotonic deadline while
    iterating `os.scandir`; no unbounded `list()`/`sorted()` scan precedes the
    first deadline check, and `_ENTRY_LIMIT` bounds each directory.
  * Process inspection is read-only and bounded: POSIX uses a
    `ps -axo command=` subprocess with the remaining budget as its timeout;
    Windows enumerates `CreateToolhelp32Snapshot` and reads each candidate
    image with `QueryFullProcessImageNameW`, checking the deadline per entry.
    A scan that cannot complete reports `complete=False` and callers defer.
  * No lock is acquired here; callers own `runtime_transaction`.

[Side-Effect Policy]
  * Unlinks only owned generation/trash directories, symlink/junction links,
    and marked orphan profile directories; read-only files are made writable
    before a bounded retry. Process inspection only opens read-only handles.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import subprocess
import time
import uuid

_EXTENDED_PREFIX = "\\\\?\\"
_PROFILE_NAME = re.compile(r"[0-9a-f]{20}\Z")
_SOURCE_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE_ANCHOR = re.compile(r"^[A-Za-z]:[\\/]")
_ENTRY_LIMIT = 16384
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TH32CS_SNAPPROCESS = 0x2
_ERROR_NO_MORE_FILES = 18
_IMAGE_CHARS = 32768
_PYTHON_IMAGES = frozenset(("python.exe", "pythonw.exe"))


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


def _lstat(path):
    """`os.lstat` with `TimeoutError` preserved and other stat failures hidden."""
    try:
        return os.lstat(path)
    except TimeoutError:
        raise
    except OSError:
        return None


def _classify(info):
    """Classify an lstat result as `"symlink"`, `"junction"` or `None`."""
    if info is None:
        return None
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    # Python 3.11 has no os.path.isjunction; 3.12's ntpath.isjunction also
    # swallows OSError (including TimeoutError). The lstat metadata below is
    # available on 3.11, 3.12 and 3.13 without either pitfall.
    reparse = (bool(getattr(info, "st_reparse_tag", 0))
               or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT))
    return "junction" if reparse else None


def _link_kind(path):
    """Return the link kind of `path` from one lstat, or None for a non-link."""
    return _classify(_lstat(path))


def _is_junction(path):
    """True for an NTFS junction/reparse point (symlinks excluded)."""
    return _link_kind(path) == "junction"


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
    if junction:
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


def _present(path):
    """True when `path` still exists, preserving `TimeoutError`."""
    return _lstat(path) is not None


def _remove_tree(path, deadline):
    """Bounded link-safe removal; True only when the entry is really gone.

    Each `unlink`/`rmdir` handles its own `FileNotFoundError`, so one child
    disappearing concurrently cannot abort the remaining traversal or leave
    the caller believing a still-present root was swept.
    """
    root = Path(path)
    fs_root = _native_path(root)
    if time.monotonic() >= deadline:
        return False
    info = _lstat(fs_root)
    if info is None:
        return not _present(root) and not _present(fs_root)
    kind = _classify(info)
    if kind is not None:
        try:
            _remove_link(fs_root, junction=(kind == "junction"))
        except TimeoutError:
            raise
        except FileNotFoundError:
            pass
        except OSError:
            return False
        return not _present(root) and not _present(fs_root)
    if not stat.S_ISDIR(info.st_mode):
        try:
            _unlink(fs_root)
        except TimeoutError:
            raise
        except FileNotFoundError:
            pass
        except OSError:
            return False
        return not _present(root) and not _present(fs_root)
    try:
        with os.scandir(fs_root) as scan:
            for entry in scan:
                if time.monotonic() >= deadline:
                    return False
                child = Path(entry.path)
                child_info = _lstat(child)
                if child_info is None:
                    continue
                child_kind = _classify(child_info)
                try:
                    if child_kind is not None:
                        _remove_link(child, junction=(child_kind == "junction"))
                    elif stat.S_ISDIR(child_info.st_mode):
                        if not _remove_tree(child, deadline):
                            return False
                    else:
                        _unlink(child)
                except TimeoutError:
                    raise
                except FileNotFoundError:
                    # Only this child vanished; keep deleting its siblings.
                    continue
                except OSError:
                    return False
    except TimeoutError:
        raise
    except FileNotFoundError:
        # The directory itself disappeared mid-scan; the final honest check
        # below decides whether removal completed.
        pass
    except NotADirectoryError:
        pass
    try:
        _rmdir(fs_root)
    except TimeoutError:
        raise
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return not _present(root) and not _present(fs_root)


def _generation_prefixes(roots):
    """Absolute generation path spellings that can appear in process data."""
    prefixes = set()
    for root in roots:
        text = os.fspath(root).rstrip("\\/")
        if not text:
            continue
        prefixes.add(text)
        if text.startswith(_EXTENDED_PREFIX):
            prefixes.add(text[len(_EXTENDED_PREFIX):])
    return tuple(sorted(prefixes))


def _generation_patterns(roots):
    return tuple(
        re.compile(re.escape(prefix) + r"[\\/]+([0-9a-f]{64})(?![0-9a-f])",
                   re.IGNORECASE)
        for prefix in _generation_prefixes(roots)
    )


def _generation_hashes(text, patterns):
    found = set()
    for pattern in patterns:
        found.update(match.group(1).lower() for match in pattern.finditer(text))
    return found


def _posix_process_text(deadline):
    """Bounded `ps` output, or None when the scan is unavailable/incomplete."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    try:
        completed = subprocess.run(
            ["ps", "-axo", "command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except TimeoutError:
        raise
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        return completed.stdout.decode("utf-8", "surrogateescape")
    except (AttributeError, UnicodeError):
        return None


def _windows_process_image(open_process, query_image, close_handle, pid):
    import ctypes
    from ctypes import wintypes
    handle = open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(_IMAGE_CHARS)
        buffer = ctypes.create_unicode_buffer(_IMAGE_CHARS)
        if not query_image(handle, 0, buffer, ctypes.byref(size)):
            return None
        return buffer.value
    finally:
        close_handle(handle)


def _windows_process_images(deadline):
    """Full image paths of live Python processes, or None when incomplete."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except TimeoutError:
        raise
    except (ImportError, OSError):
        return None

    class _ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_snapshot = kernel32.CreateToolhelp32Snapshot
        create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        create_snapshot.restype = wintypes.HANDLE
        first = kernel32.Process32FirstW
        first.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W))
        first.restype = wintypes.BOOL
        next_entry = kernel32.Process32NextW
        next_entry.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W))
        next_entry.restype = wintypes.BOOL
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        query_image = kernel32.QueryFullProcessImageNameW
        query_image.argtypes = (wintypes.HANDLE, wintypes.DWORD,
                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
        query_image.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
    except TimeoutError:
        raise
    except (AttributeError, OSError, TypeError):
        return None
    snapshot = create_snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return None
    entry = _ProcessEntry32W()
    entry.dwSize = ctypes.sizeof(_ProcessEntry32W)
    images = []
    try:
        present = first(snapshot, ctypes.byref(entry))
        while present:
            if time.monotonic() >= deadline:
                return None
            if entry.th32ProcessID and entry.szExeFile.lower() in _PYTHON_IMAGES:
                image = _windows_process_image(
                    open_process, query_image, close_handle,
                    entry.th32ProcessID,
                )
                if image is None:
                    # A live Python process we cannot attribute: incomplete.
                    return None
                images.append(image)
            present = next_entry(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != _ERROR_NO_MORE_FILES:
            return None
    finally:
        close_handle(snapshot)
    return images


def generation_roots(config):
    """Both generation layouts a live process may reference."""
    from .reporting_runtime import _generations, _root_path
    try:
        root = _root_path(config)
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError):
        return ()
    primary = Path(_generations(root))
    legacy = Path(root) / "generations"
    if os.path.normcase(os.fspath(primary)) == os.path.normcase(os.fspath(legacy)):
        return (primary,)
    return (primary, legacy)


def live_generation_hashes(generation_roots, *, deadline=None):
    """Return `(source_hashes, complete)` for live processes using a generation.

    Process inspection is read-only. `complete` is False whenever the platform
    scan cannot finish within `deadline` (missing helper, subprocess timeout,
    an unreadable candidate image, or an exhausted budget); callers must keep
    every generation in that case.
    """
    patterns = _generation_patterns(tuple(generation_roots))
    if not patterns:
        return set(), True
    if deadline is None:
        deadline = time.monotonic() + 3.0
    if time.monotonic() >= deadline:
        return set(), False
    if os.name == "posix":
        commands = _posix_process_text(deadline)
        if commands is None:
            return set(), False
        return _generation_hashes(commands, patterns), True
    if os.name == "nt":
        images = _windows_process_images(deadline)
        if images is None:
            return set(), False
        return _generation_hashes("\n".join(images), patterns), True
    return set(), False


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
    """Return `(candidate images, complete)` for the real per-generation layout."""
    try:
        children, complete = _bounded_entries(profile, deadline)
    except TimeoutError:
        raise
    except OSError:
        return [], False
    candidates = []
    for child in children:
        if time.monotonic() >= deadline:
            return candidates, False
        info = _lstat(child.path)
        if info is None:
            # A child we cannot inspect leaves the profile's liveness unknown.
            return candidates, False
        if _classify(info) is not None or not stat.S_ISDIR(info.st_mode):
            continue
        if not _SOURCE_HASH.fullmatch(child.name):
            continue
        for name in ("python.exe", "pythonw.exe"):
            candidates.append(Path(child.path) / "venv" / "Scripts" / name)
    # Pre-0.5.14 profiles kept a single venv directly under the profile.
    for name in ("python.exe", "pythonw.exe"):
        candidates.append(Path(profile) / "venv" / "Scripts" / name)
    return candidates, complete


def _profile_idle(entry, deadline=None):
    """True only after a complete scan proved no generation image is mapped."""
    if deadline is None:
        deadline = time.monotonic() + 3.0
    candidates, complete = _image_candidates(entry, deadline)
    if not complete:
        return False
    for candidate in candidates:
        if time.monotonic() >= deadline:
            return False
        try:
            if _present(candidate) and not _image_available(candidate):
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
        info = _lstat(entry.path)
        if info is None or _classify(info) is not None:
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        try:
            if not _owned(info):
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
            if time.monotonic() >= deadline:
                # The scan itself may have spent the last of the budget.
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
