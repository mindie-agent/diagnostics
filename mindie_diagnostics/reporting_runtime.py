"""User-owned runtime copied from the installed mindie-diagnostics distribution."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
import venv
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


def _fail(category):
    raise RuntimeError(category)


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


def _place(site, files):
    base = site.resolve()
    for key, data in files:
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


def _verify(py, version, site, work):
    out, err = work / f".vout-{uuid.uuid4().hex}", work / f".verr-{uuid.uuid4().hex}"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    try:
        _write(out, b"")
        _write(err, b"")
        with open(out, "wb") as fout, open(err, "wb") as ferr:
            proc = subprocess.run(
                [str(py), "-I", "-c", _VERIFY],
                stdout=fout,
                stderr=ferr,
                timeout=10,
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


def _publish(root, payload):
    tmp = root / f".current-{uuid.uuid4().hex}"
    _write(tmp, json.dumps(payload, separators=(",", ":")).encode() + b"\n")
    os.replace(tmp, root / "current.json")


def _prepare(config):
    import importlib.metadata as metadata

    from mindie_diagnostics.fallback import (
        _ancestors_are_real_dirs,
        _as_local_absolute,
        _owned,
        _read_regular_bounded,
        policy_path,
    )

    dist = metadata.distribution(_DIST)
    _installed(dist, _as_local_absolute)
    version, revision, files, source_hash = _collect(dist, _read_regular_bounded)
    root = _ensure_dir(
        Path(policy_path(config)).with_suffix(".runtime"),
        _ancestors_are_real_dirs,
        _as_local_absolute,
        _owned,
    )
    generations = _ensure_dir(root / "generations", _ancestors_are_real_dirs, _as_local_absolute, _owned)
    lock = root / ".prepare.lock"
    created = False
    staging = None
    promoted = False
    try:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _FILE)
            created = True
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            _fail("runtime_busy")
        except RuntimeError:
            raise
        except Exception:
            _fail("runtime_unavailable")
        current = None
        pointer = root / "current.json"
        if pointer.exists() or pointer.is_symlink():
            current = _load_json(pointer, _read_regular_bounded, _OUT_CAP)
            if current.get("schema") != 1 or not isinstance(current.get("version"), str):
                _fail("runtime_untrusted")
            old, new = _triple(current["version"]), _triple(version)
            if old and new and new < old:
                _fail("downgrade_rejected")
        dest = generations / source_hash
        if dest.exists() or dest.is_symlink():
            _require_dir(dest, _owned)
            _marker(dest, source_hash, version, revision, _read_regular_bounded)
            py = _python(dest, _owned)
            site = dest / "venv" / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            _verify(py, version, site, dest)
        else:
            staging = generations / f".staging-{uuid.uuid4().hex}"
            os.mkdir(staging, _DIR)
            if os.name != "nt":
                os.chmod(staging, _DIR)
            venv.EnvBuilder(with_pip=False, symlinks=os.name == "posix").create(str(staging / "venv"))
            site = staging / "venv" / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            if not site.is_dir():
                _fail("verify_failed")
            _place(site, files)
            body = {"schema": 1, "source_hash": source_hash, "version": version, "revision": revision}
            _write(staging / "source.json", json.dumps(body, separators=(",", ":")).encode() + b"\n")
            _verify(_python(staging, _owned), version, site, staging)
            os.replace(staging, dest)
            promoted = True
            staging = None
            py = _python(dest, _owned)
            final_site = dest / "venv" / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            _verify(py, version, final_site, dest)
        changed = current is None or current.get("source_hash") != source_hash
        previous = None if current is None else current.get("revision")
        if changed:
            _publish(root, {"schema": 1, "source_hash": source_hash, "version": version, "revision": revision})
        return {
            "status": "ready",
            "python": str(py),
            "version": version,
            "revision": revision,
            "source_hash": source_hash,
            "previous_revision": previous,
            "changed": changed,
        }
    finally:
        if staging is not None and not promoted:
            shutil.rmtree(staging, ignore_errors=True)
        if created:
            try:
                os.unlink(lock)
            except OSError:
                pass


def prepare_runtime(config=None):
    try:
        return _prepare(config)
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("runtime_unavailable") from None
