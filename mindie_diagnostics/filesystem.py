"""Cross-platform path identity, SQLite URIs, WSL DrvFS, and native CLI lookup.

[Role]
- Compare local path spellings (`path_identity`) and encode SQLite file URIs
  (`sqlite_file_uri`) across Windows extended-length (`\\\\?\\`) paths.
- Detect Windows PE binaries (`is_pe_executable`), WSL2 DrvFS mounts
  (`is_wsl_drvfs`), and native platform executables (`which_native`) so WSL2
  never accidentally executes `/mnt/c/.../*.exe` host shims from `PATH`.

[Invariants]
- Extended Windows paths retain long-path support and virtualenv identity.
- On POSIX (`os.name != "nt"` and `sys.platform != "win32"`), `which_native`
  never returns a Windows PE executable or a bare-name match on a WSL DrvFS mount.
- `except TimeoutError: raise` precedes every `except OSError:` handler.

[Timeouts & Locks]
- Pure path arithmetic plus bounded local header/mountinfo reads (`<= 1 KiB` PE
  probe using `O_NONBLOCK`); acquires no locks and spawns no subprocesses.

[Side-Effect Policy]
- Strictly read-only inspection; never mutates files, permissions, or config.
"""
from __future__ import annotations

import ntpath
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from urllib.parse import quote

_STDLIB_WHICH = shutil.which
_MOUNTINFO_PATH = "/proc/self/mountinfo"
_OCTAL_ESCAPE_RE = re.compile(r"\\([0-7]{3})")
_WINDOWS_SUFFIXES = (".exe", ".com", ".bat", ".cmd", ".ps1")


def path_identity(value):
    """Compare drive paths without resolving a virtualenv into its base Python."""
    text = os.fspath(value)
    if os.name == "nt":
        logical = text[4:] if text.startswith("\\\\?\\") else text
        drive, tail = ntpath.splitdrive(logical)
        if drive and tail.startswith(("\\", "/")):
            return ntpath.normcase(ntpath.normpath(logical))
    return text


def sqlite_file_uri(value):
    path = Path(value).absolute()
    # Path.as_uri() treats the extended namespace's '?' as a URI authority.
    if os.name == "nt" and str(path).startswith("\\\\?\\"):
        return "file:" + quote(str(path), safe=":")
    return path.as_uri()


def is_pe_executable(value) -> bool:
    """Return True iff `value` resolves to a regular file with a valid MZ+PE header."""
    try:
        target = os.fspath(value)
        info = os.stat(target)
        if not stat.S_ISREG(info.st_mode) or info.st_size < 68:
            return False
        fd = os.open(target, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        try:
            head = os.read(fd, 1024)
            if len(head) < 64 or head[:2] != b"MZ":
                return False
            pe_offset = int.from_bytes(head[60:64], "little", signed=False)
            if not 64 <= pe_offset <= 16 * 1024 * 1024:
                return False
            if pe_offset + 4 <= len(head):
                sig = head[pe_offset : pe_offset + 4]
            else:
                os.lseek(fd, pe_offset, os.SEEK_SET)
                sig = os.read(fd, 4)
            return sig == b"PE\x00\x00"
        finally:
            os.close(fd)
    except TimeoutError:
        raise
    except OSError:
        return False


def _unescape_mount_path(raw: str) -> str:
    return _OCTAL_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 8)), raw)


def _is_drvfs_entry(fstype: str, source: str, super_opts: str) -> bool:
    if fstype == "drvfs":
        return True
    if fstype == "9p":
        if "drvfs" in super_opts or "aname=drvfs" in super_opts:
            return True
        if len(source) >= 2 and source[0].isalpha() and source[1] == ":":
            return True
    return False


def _parse_drvfs_mounts(raw_mountinfo: str) -> list[tuple[str, bool, bool]]:
    """Return `(mount_point, is_drvfs, fixed_mode_for_euid)` entries from mountinfo."""
    euid = os.geteuid() if hasattr(os, "geteuid") else -1
    mounts: list[tuple[str, bool, bool]] = []
    for line in raw_mountinfo.splitlines():
        lhs, sep, rhs = line.partition(" - ")
        if not sep:
            continue
        lhs_parts = lhs.split()
        rhs_parts = rhs.split()
        if len(lhs_parts) < 6 or len(rhs_parts) < 3:
            continue
        mount_point = _unescape_mount_path(lhs_parts[4])
        on_drvfs = _is_drvfs_entry(rhs_parts[0], rhs_parts[1], rhs_parts[2])
        fixed_mode = False
        if on_drvfs:
            tokens = [tok.strip() for tok in re.split(r"[,;]", f"{lhs_parts[5]},{rhs_parts[2]}") if tok.strip()]
            has_metadata = "metadata" in tokens
            uid_tokens = [tok.split("=", 1)[1] for tok in tokens if tok.startswith("uid=")]
            uid_matches = bool(uid_tokens) and all(tok.isdigit() and int(tok) == euid for tok in uid_tokens)
            fixed_mode = (not has_metadata) and uid_matches
        mounts.append((mount_point, on_drvfs, fixed_mode))
    return mounts


def _match_drvfs_mount(value, *, require_fixed_mode: bool) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        raw_mountinfo = Path(_MOUNTINFO_PATH).read_text(encoding="utf-8", errors="replace")
    except TimeoutError:
        raise
    except OSError:
        return False
    mounts = _parse_drvfs_mounts(raw_mountinfo)
    if not mounts:
        return False
    raw_text = os.fspath(value)
    try:
        real = os.path.realpath(raw_text)
    except TimeoutError:
        raise
    except OSError:
        real = os.path.abspath(raw_text)
    candidates = [real] if require_fixed_mode else list(dict.fromkeys([os.path.abspath(raw_text), real]))
    for target in candidates:
        best_len = -1
        best_flag = False
        for mount_point, on_drvfs, fixed_mode in mounts:
            prefix = "/" if mount_point == "/" else mount_point.rstrip("/")
            if target == prefix or (prefix == "/" and target.startswith("/")) or target.startswith(prefix + "/"):
                if len(prefix) > best_len:
                    best_len = len(prefix)
                    best_flag = fixed_mode if require_fixed_mode else on_drvfs
        if best_flag:
            return True
    return False


def is_wsl_drvfs(value) -> bool:
    """Return True when `value` (or its symlink target) lives on any WSL DrvFS mount."""
    return _match_drvfs_mount(value, require_fixed_mode=False)


def is_wsl_drvfs_fixed_mode(value) -> bool:
    """Return True only when the resolved target lives on single-owner fixed-mode WSL DrvFS."""
    return _match_drvfs_mount(value, require_fixed_mode=True)


def _is_exec_regular_file(path_str: str) -> bool:
    try:
        st = os.stat(path_str)
        return stat.S_ISREG(st.st_mode) and os.access(path_str, os.X_OK)
    except TimeoutError:
        raise
    except OSError:
        return False


def which_native(cmd, *, path=None, which_fn=None) -> str | None:
    """Resolve `cmd` without picking Windows PE `.exe` or WSL DrvFS shims on POSIX."""
    cmd_str = os.fspath(cmd) if isinstance(cmd, (str, Path)) else ""
    if not cmd_str:
        return None
    lookup = which_fn or shutil.which
    if os.name == "nt" or sys.platform == "win32":
        return lookup(cmd_str) if path is None else lookup(cmd_str, path=path)
    if "/" in cmd_str or (os.path.altsep and os.path.altsep in cmd_str):
        if _is_exec_regular_file(cmd_str) and not is_pe_executable(cmd_str) and not is_wsl_drvfs(cmd_str):
            return cmd_str
        return None
    if lookup is not _STDLIB_WHICH:
        found = lookup(cmd_str) if path is None else lookup(cmd_str, path=path)
        if not found or (_is_exec_regular_file(found) and (is_pe_executable(found) or is_wsl_drvfs(found))):
            return None
        return found
    if cmd_str.lower().endswith(_WINDOWS_SUFFIXES):
        return None
    raw_path = os.environ.get("PATH", "") if path is None else path
    for entry in raw_path.split(os.pathsep):
        candidate = os.path.join(entry or ".", cmd_str)
        if _is_exec_regular_file(candidate):
            if is_pe_executable(candidate) or is_wsl_drvfs(candidate):
                continue
            return candidate
    return None
