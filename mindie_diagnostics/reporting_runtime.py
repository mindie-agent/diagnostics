"""User-owned runtime copied from the installed mindie-diagnostics distribution."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_PKG = "mindie_diagnostics"
_DIST = "mindie-diagnostics"
_MAX_FILES = 256
_MAX_BYTES = 8 * 1024 * 1024
_OUT_CAP = 64 * 1024
_DIR = 0o700
_FILE = 0o600
_VERIFY = (
    "import importlib.metadata as m,json,mindie_diagnostics as p;"
    "print(json.dumps({'v':m.version('mindie-diagnostics'),'p':p.__file__}))"
)
_LOCK_WAIT = 12.0  # Bounded seconds to acquire the runtime transaction lock.


def _fail(category):
    raise RuntimeError(category)


def _remaining(deadline):
    """Seconds left of an absolute monotonic budget; fail closed when spent."""
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _fail("runtime_deadline_exceeded")
    return remaining


def _triple(version):
    core = version.split("+", 1)[0].split("-", 1)[0]
    bits = core.split(".")
    if len(bits) != 3:
        return None
    out = []
    for bit in bits:
        if not bit.isdigit() or (len(bit) > 1 and bit.startswith("0")):
            return None
        out.append(int(bit))
    return tuple(out)


def _is_source_hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _revision(dist):
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        meta = json.loads(raw)
    except Exception:
        _fail("source_rejected")
    editable = (meta.get("dir_info") or {}).get("editable")
    if editable is True or (editable is not None and editable is not False):
        _fail("source_rejected")
    commit = (meta.get("vcs_info") or {}).get("commit_id")
    if commit is None:
        return None
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        _fail("source_rejected")
    return commit


def _safe_key(entry):
    rel = Path(str(entry))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        _fail("source_rejected")
    return "/".join(rel.parts)


def _collect(dist, read_bounded):
    version = dist.version
    if not isinstance(version, str) or not version or any(c in version for c in "\r\n\x00"):
        _fail("source_rejected")
    revision = _revision(dist)
    items, infos, total = [], [], 0
    seen = set()
    for entry in dist.files or ():
        parts = Path(str(entry)).parts
        if not parts or (parts[0] != _PKG and not parts[0].endswith('.dist-info')):
            continue
        key = _safe_key(entry)
        if key in seen:
            continue
        seen.add(key)
        parts = key.split("/")
        pkg = parts[0] == _PKG
        info = parts[0].endswith(".dist-info")
        if not pkg and not info:
            continue
        if pkg and ("__pycache__" in parts or parts[-1].endswith(".pyc")):
            continue
        if len(items) + len(infos) >= _MAX_FILES:
            _fail("source_rejected")
        loc = Path(dist.locate_file(key))
        from .fallback import _ancestors_are_real_dirs
        if not _ancestors_are_real_dirs(loc) or loc.is_symlink():
            _fail("source_rejected")
        try:
            data = read_bounded(loc, _MAX_BYTES)
        except Exception:
            _fail("source_rejected")
        if data is None:
            # pip creates an empty dist-info/REQUESTED marker. Validate it with
            # the same no-follow regular-file boundary before preserving it.
            fd = os.open(loc, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size != 0 or (hasattr(os, 'geteuid') and info.st_uid != os.geteuid()):
                    _fail("source_rejected")
                data = b""
            finally:
                os.close(fd)
        if not isinstance(data, bytes):
            _fail("source_rejected")
        total += len(data)
        if total > _MAX_BYTES:
            _fail("source_rejected")
        (items if pkg else infos).append((key, data))
    if not any(key == f"{_PKG}/__init__.py" for key, _ in items):
        _fail("source_rejected")
    digest = hashlib.sha256()
    digest.update(version.encode())
    digest.update(b"\0")
    digest.update((revision or "").encode())
    digest.update(b"\0")
    for key, data in sorted(items):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return version, revision, items + infos, digest.hexdigest()


def _installed(dist, absolute):
    import mindie_diagnostics

    init = Path(absolute(mindie_diagnostics.__file__)).resolve()
    located = Path(absolute(dist.locate_file(f"{_PKG}/__init__.py"))).resolve()
    prefix = Path(absolute(sys.prefix)).resolve()
    if init != located or (init != prefix and prefix not in init.parents):
        _fail("source_rejected")


def _require_dir(path, owned):
    if path.is_symlink() or not path.is_dir():
        _fail("runtime_untrusted")
    st = path.stat()
    if not owned(st):
        _fail("runtime_untrusted")
    if os.name != "nt" and stat.S_IMODE(st.st_mode) != _DIR:
        _fail("runtime_untrusted")


def _ensure_dir(path, ancestors, absolute, owned):
    path = Path(absolute(path))
    if path.is_symlink():
        _fail("runtime_untrusted")
    if path.exists():
        _require_dir(path, owned)
        return path
    if not ancestors(path, create=True):
        _fail("runtime_untrusted")
    if not path.exists():
        os.mkdir(path, _DIR)
    if os.name != "nt":
        os.chmod(path, _DIR)
    _require_dir(path, owned)
    return path


def _write(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, _FILE)
    try:
        if os.write(fd, data) != len(data):
            _fail("runtime_write_failed")
        os.fsync(fd)
    finally:
        os.close(fd)
    if os.name != "nt":
        os.chmod(path, _FILE)


def _mkdirs(base, dest):
    cur = base
    for part in dest.parent.relative_to(base).parts:
        cur = cur / part
        if cur.exists() or cur.is_symlink():
            if cur.is_symlink() or not cur.is_dir():
                _fail("source_rejected")
            continue
        os.mkdir(cur, _DIR)
        if os.name != "nt":
            os.chmod(cur, _DIR)


def _place(site, files, deadline=None):
    base = site.resolve()
    for key, data in files:
        _remaining(deadline)
        dest = site.joinpath(*key.split("/"))
        if base not in dest.resolve().parents:
            _fail("source_rejected")
        _mkdirs(site, dest)
        _write(dest, data)


def _python(gen, owned):
    name = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    py = gen / "venv" / Path(name)
    if not py.is_file() or not owned(py.lstat()):
        _fail("verify_failed")
    if py.is_symlink():
        base = Path(sys._base_executable).resolve()
        if py.resolve() != base or not base.is_relative_to(Path(sys.base_prefix).resolve()):
            _fail("verify_failed")
    # Resolving this path would bypass pyvenv.cfg and the copied distribution.
    return py.absolute()


def _load_json(path, read_bounded, limit):
    if path.is_symlink():
        _fail("runtime_untrusted")
    try:
        raw = read_bounded(path, limit)
        doc = json.loads(raw if isinstance(raw, str) else bytes(raw).decode())
    except RuntimeError:
        raise
    except Exception:
        _fail("runtime_untrusted")
    if not isinstance(doc, dict):
        _fail("runtime_untrusted")
    return doc


def _verify(py, version, site, work, deadline=None):
    out, err = work / f".vout-{uuid.uuid4().hex}", work / f".verr-{uuid.uuid4().hex}"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    try:
        _write(out, b"")
        _write(err, b"")
        remaining = _remaining(deadline)
        with open(out, "wb") as fout, open(err, "wb") as ferr:
            proc = subprocess.run(
                [str(py), "-I", "-c", _VERIFY],
                stdout=fout,
                stderr=ferr,
                timeout=10 if remaining is None else min(10, remaining),
                env=env,
                cwd=str(work),
            )
        if proc.returncode != 0 or out.stat().st_size > _OUT_CAP or err.stat().st_size > _OUT_CAP:
            _fail("verify_failed")
        doc = json.loads(out.read_bytes().decode())
        got = Path(doc["p"]).resolve()
        if doc.get("v") != version or got != (site / _PKG / "__init__.py").resolve():
            _fail("verify_failed")
    except RuntimeError:
        raise
    except Exception:
        _fail("verify_failed")
    finally:
        for path in (out, err):
            try:
                os.unlink(path)
            except OSError:
                pass


def _marker(gen, source_hash, version, revision, read_bounded):
    doc = _load_json(gen / "source.json", read_bounded, _OUT_CAP)
    if (
        doc.get("schema") != 1
        or doc.get("source_hash") != source_hash
        or doc.get("version") != version
        or doc.get("revision") != revision
    ):
        _fail("verify_failed")


def _publish_json(root, name, payload):
    tmp = root / f".{name}-{uuid.uuid4().hex}"
    _write(tmp, json.dumps(payload, separators=(",", ":")).encode() + b"\n")
    os.replace(tmp, root / name)


def _publish(root, payload):
    _publish_json(root, "current.json", payload)


def _root_path(config):
    from mindie_diagnostics.fallback import _as_local_absolute, policy_path

    base = _as_local_absolute(policy_path(config))
    if base is None:
        _fail("runtime_unavailable")
    return Path(base).with_suffix(".runtime")


def _collect_source():
    import importlib.metadata as metadata

    from mindie_diagnostics.fallback import _as_local_absolute, _read_regular_bounded

    dist = metadata.distribution(_DIST)
    _installed(dist, _as_local_absolute)
    return _collect(dist, _read_regular_bounded)


def _read_current(root):
    from mindie_diagnostics.fallback import _read_regular_bounded

    pointer = root / "current.json"
    if not pointer.exists() and not pointer.is_symlink():
        return None
    current = _load_json(pointer, _read_regular_bounded, _OUT_CAP)
    if (
        current.get("schema") != 1
        or not isinstance(current.get("version"), str)
        or _triple(current["version"]) is None
        or not _is_source_hash(current.get("source_hash"))
    ):
        _fail("runtime_untrusted")
    return current


def _source_order(current, version, source_hash):
    """Strict release x.y.z ordering of installed source against the committed runtime."""
    if current["source_hash"] == source_hash:
        return "identical"
    old, new = _triple(current["version"]), _triple(version)
    if old is None or new is None:
        # Unknown or incomparable versions fail closed.
        return "incomparable"
    if new < old:
        return "older"
    if new == old:
        # Same release carrying different source: an explicit ensure chooses.
        return "conflict"
    return "newer"


def _generation_python(root, source_hash):
    from mindie_diagnostics.fallback import _owned

    dest = root / "generations" / source_hash
    _require_dir(dest, _owned)
    return str(_python(dest, _owned))


def _update_record_name(source_hash):
    return f"update-{source_hash}.json"


def _read_update_record(root, source_hash):
    from mindie_diagnostics.fallback import _read_regular_bounded

    path = root / _update_record_name(source_hash)
    if not path.exists() and not path.is_symlink():
        return None
    doc = _load_json(path, _read_regular_bounded, _OUT_CAP)
    target = doc.get("target")
    if (
        doc.get("schema") != 1
        or doc.get("status") not in {"attempting", "failed", "rolled_back", "updated", "aborted"}
        or not isinstance(target, dict)
        or target.get("source_hash") != source_hash
    ):
        _fail("runtime_untrusted")
    return doc


def _write_update_record(root, source_hash, payload):
    """Atomically persist the durable automatic-handoff intent/result."""
    _publish_json(root, _update_record_name(source_hash), payload)


@contextmanager
def _transaction_lock(root, timeout):
    """Bounded exclusive OS advisory lock at the historical .prepare.lock path.

    The file persists between transactions; a stale leftover can no longer
    block or be stolen by a later preparer the way the old O_EXCL marker could.
    """
    path = root / ".prepare.lock"
    if path.is_symlink():
        _fail("runtime_untrusted")
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), _FILE)
    except OSError:
        _fail("runtime_untrusted")
    try:
        if os.name != "nt":
            os.fchmod(fd, _FILE)
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"0")
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    _fail("runtime_busy")
                time.sleep(0.05)
        try:
            yield root
        finally:
            if os.name == "nt":
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
    finally:
        os.close(fd)


@contextmanager
def runtime_transaction(config=None, *, create=True, timeout=None):
    """One bounded exclusive transaction over the user runtime root.

    The lock covers source selection, runtime preparation, service replacement,
    readback and any rollback. Explicit ensure and the automatic handoff share
    this lock. With ``create=False`` a missing root yields None instead of
    performing a first installation.
    """
    from mindie_diagnostics.fallback import _ancestors_are_real_dirs, _as_local_absolute, _owned

    root = _root_path(config)
    if create:
        root = _ensure_dir(root, _ancestors_are_real_dirs, _as_local_absolute, _owned)
    else:
        if root.is_symlink() or not root.is_dir():
            yield None
            return
        _require_dir(root, _owned)
    with _transaction_lock(root, _LOCK_WAIT if timeout is None else timeout):
        yield root


def _create_venv(target, deadline=None):
    """Build the generation venv in a bounded owned stdlib subprocess."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    remaining = _remaining(deadline)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-m", "venv", "--without-pip", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30 if remaining is None else min(30, remaining),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _fail("runtime_deadline_exceeded")
    except OSError:
        _fail("verify_failed")
    if proc.returncode != 0:
        _fail("verify_failed")


def _prepare_locked(config, *, publish=True, source=None, deadline=None):
    """Prepare the committed-source runtime under the caller's transaction lock.

    With ``publish=False`` the immutable generation is materialized and
    verified but the current.json pointer is left untouched; the caller
    publishes it only after the owned service is confirmed on the new runtime.
    ``deadline`` is an absolute monotonic budget shared with the caller; every
    subprocess and copy step fails closed with runtime_deadline_exceeded.
    """
    from mindie_diagnostics.fallback import (
        _ancestors_are_real_dirs,
        _as_local_absolute,
        _owned,
        _read_regular_bounded,
    )

    _remaining(deadline)
    version, revision, files, source_hash = source if source is not None else _collect_source()
    root = _ensure_dir(_root_path(config), _ancestors_are_real_dirs, _as_local_absolute, _owned)
    generations = _ensure_dir(root / "generations", _ancestors_are_real_dirs, _as_local_absolute, _owned)
    current = _read_current(root)
    if current is not None:
        old, new = _triple(current["version"]), _triple(version)
        if old and new and new < old:
            _fail("downgrade_rejected")
    staging = None
    promoted = False
    try:
        dest = generations / source_hash
        if dest.exists() or dest.is_symlink():
            _require_dir(dest, _owned)
            _marker(dest, source_hash, version, revision, _read_regular_bounded)
            py = _python(dest, _owned)
            site = dest / "venv" / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            _verify(py, version, site, dest, deadline)
        else:
            staging = generations / f".staging-{uuid.uuid4().hex}"
            os.mkdir(staging, _DIR)
            if os.name != "nt":
                os.chmod(staging, _DIR)
            _create_venv(staging / "venv", deadline)
            site = staging / "venv" / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            if not site.is_dir():
                _fail("verify_failed")
            _place(site, files, deadline)
            body = {"schema": 1, "source_hash": source_hash, "version": version, "revision": revision}
            _write(staging / "source.json", json.dumps(body, separators=(",", ":")).encode() + b"\n")
            _verify(_python(staging, _owned), version, site, staging, deadline)
            os.replace(staging, dest)
            promoted = True
            staging = None
            py = _python(dest, _owned)
            final_site = dest / "venv" / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            _verify(py, version, final_site, dest, deadline)
        changed = current is None or current.get("source_hash") != source_hash
        previous = None if current is None else current.get("revision")
        if changed and publish:
            _publish(root, {"schema": 1, "source_hash": source_hash, "version": version, "revision": revision})
        return {
            "status": "ready",
            "python": str(py),
            "version": version,
            "revision": revision,
            "source_hash": source_hash,
            "previous_revision": previous,
            "changed": changed,
            "published": bool(changed and publish),
        }
    finally:
        if staging is not None and not promoted:
            shutil.rmtree(staging, ignore_errors=True)


def prepare_runtime(config=None):
    try:
        with runtime_transaction(config):
            return _prepare_locked(config)
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("runtime_unavailable") from None
