"""Standalone local diagnostics fallback. Stdlib only.

Safe to copy byte-for-byte into adapter scripts. Import does no file I/O.
Never raises from the public recorder, never calls a service, and never
records exception text, arguments, command lines, or outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_POLICY_SCHEMA = "mindie.diagnostics.reporting.v1"
_POLICY_PURPOSE = "tool_fault_reporting"
_MAX_POLICY_BYTES = 16384
_MAX_EVENT_BYTES = 16384
_MAX_FILE_BYTES = 1024 * 1024
_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,119}\Z")
_INCIDENT_RE = re.compile(r"[0-9a-f]{32}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{7,64}\Z")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,79}\Z")
_REPO_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}/[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z"
)
_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,119}\Z")
_NEVER_REPORTABLE = frozenset({
    "caller",
    "cancelled",
    "business",
    "connectivity",
    "credentials",
    "permission",
    "disabled",
    "prerequisite",
})
_CONSENT_KEYS = frozenset({"config_file", "revision", "repository", "purpose"})
_STATIC_WARN = "mindie diagnostics: local failure record was not written\n"

_lock = threading.Lock()
_warn_lock = threading.Lock()
_warned = False
_process_pid = None
_process_uuid = None


def _after_fork():
    global _lock, _warn_lock, _warned, _process_pid, _process_uuid
    _lock, _warn_lock = threading.Lock(), threading.Lock()
    _warned, _process_pid, _process_uuid = False, None, None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def policy_path(config=None) -> Path:
    """Explicit path, else MINDIE_DIAGNOSTICS_CONFIG, else the platform default."""
    if config is not None:
        return Path(config)
    env = os.environ.get("MINDIE_DIAGNOSTICS_CONFIG")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "mindie-agent" / "diagnostics.json"
    return Path.home() / ".config" / "mindie-agent" / "diagnostics.json"


def default_root() -> Path:
    """MINDIE_DIAGNOSTICS_ROOT, else the platform local state directory."""
    env = os.environ.get("MINDIE_DIAGNOSTICS_ROOT")
    if env:
        return Path(env)
    if os.name == "nt" and not os.environ.get("XDG_STATE_HOME"):
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "mindie" / "diagnostics"
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "mindie" / "diagnostics"
    return Path.home() / ".local" / "state" / "mindie" / "diagnostics"


def state_path(config=None) -> Path:
    """Sibling state directory for the selected policy path."""
    return policy_path(config).with_suffix(".state")


def read_policy(config=None):
    """Validate a local reporting policy. Zero writes. None on any doubt."""
    try:
        path = _as_local_absolute(policy_path(config))
        if path is None or not _ancestors_are_real_dirs(path):
            return None
        raw = _read_regular_bounded(path, _MAX_POLICY_BYTES)
        if raw is None:
            return None
        data = json.loads(raw.decode("utf-8"))
        return _validated_policy(data)
    except Exception:
        return None


def current_consent(config=None):
    """Local consent snapshot, or None when reporting is not enabled."""
    try:
        policy = read_policy(config)
        if not policy or policy.get("decision") != "enabled":
            return None
        path = policy_path(config)
        return {
            "config_file": str(path),
            "revision": policy["revision"],
            "repository": policy["repository"],
            "purpose": policy["purpose"],
        }
    except Exception:
        return None


def consent_allowed(reference) -> bool:
    """True only when a fresh enabled policy still matches this local reference."""
    try:
        if set(reference.keys()) != _CONSENT_KEYS:
            return False
        purpose = reference["purpose"]
        repository = reference["repository"]
        revision = reference["revision"]
        config_file = reference["config_file"]
        if purpose != _POLICY_PURPOSE or not _valid_repo(repository):
            return False
        if not isinstance(revision, str) or _INCIDENT_RE.fullmatch(revision) is None:
            return False
        if not isinstance(config_file, str):
            return False
        policy = read_policy(config_file)
        if not policy or policy.get("decision") != "enabled":
            return False
        return (
            policy["purpose"] == purpose
            and policy["repository"] == repository
            and policy["revision"] == revision
        )
    except Exception:
        return False


def scope_key(public_fingerprint, reference) -> str:
    """sha256 of the canonical consent tuple plus a public fingerprint."""
    try:
        if not isinstance(public_fingerprint, str) or not public_fingerprint:
            return ""
        if len(public_fingerprint) > 256 or not public_fingerprint.isascii():
            return ""
        if any(ord(ch) < 33 or ord(ch) == 127 for ch in public_fingerprint):
            return ""
        if not isinstance(reference, dict) or set(reference.keys()) != _CONSENT_KEYS:
            return ""
        purpose = reference["purpose"]
        repository = reference["repository"]
        revision = reference["revision"]
        if (
            purpose != _POLICY_PURPOSE
            or not _valid_repo(repository)
            or not isinstance(revision, str)
            or _INCIDENT_RE.fullmatch(revision) is None
        ):
            return ""
        canonical = json.dumps(
            [purpose, repository, revision, public_fingerprint],
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def record_failure(
    component,
    operation,
    *,
    stage,
    category,
    exception=None,
    host=None,
    revision=None,
    version=None,
    elapsed_ms=None,
    exit_code=None,
    reportable=True,
    root=None,
    incident_id=None,
) -> dict:
    """Record one local operation failure. Never raises or masks the caller."""
    try:
        if _valid_incident(incident_id):
            return {
                "recorded": True,
                "incident_id": incident_id,
                "logging_failed": False,
            }
        fresh = uuid.uuid4().hex
        event = _build_event(
            fresh,
            component,
            operation,
            stage=stage,
            category=category,
            exception=exception,
            host=host,
            revision=revision,
            version=version,
            elapsed_ms=elapsed_ms,
            exit_code=exit_code,
            reportable=reportable,
        )
        if event is None:
            return {
                "recorded": False,
                "incident_id": None,
                "logging_failed": False,
                "error": "invalid_arguments",
            }
        if _append_event(root, component, event):
            return {
                "recorded": True,
                "incident_id": fresh,
                "logging_failed": False,
            }
        _warn_once()
        return {
            "recorded": False,
            "incident_id": None,
            "logging_failed": True,
            "error": "logging_failed",
        }
    except Exception:
        _warn_once()
        return {
            "recorded": False,
            "incident_id": None,
            "logging_failed": True,
            "error": "logging_failed",
        }


def _build_event(
    incident,
    component,
    operation,
    *,
    stage,
    category,
    exception,
    host,
    revision,
    version,
    elapsed_ms,
    exit_code,
    reportable,
):
    if not _label(component, 80) or not _label(operation, 120):
        return None
    if not _label(stage, 120) or not _label(category, 120):
        return None
    if not isinstance(reportable, bool):
        return None
    attributes = {"stage": stage, "category": category}
    if exception is not None:
        error_type = type(exception).__name__
        if not isinstance(error_type, str) or _ERROR_TYPE_RE.fullmatch(error_type) is None:
            error_type = "Exception"
        attributes["error_type"] = error_type
    if host is not None:
        if not _label(host, 120):
            return None
        attributes["host"] = host
    if exit_code is not None:
        code = _int32(exit_code)
        if code is None:
            return None
        attributes["exit_code"] = code
    duration = None
    if elapsed_ms is not None:
        duration = _duration(elapsed_ms)
        if duration is None:
            return None
    package_revision = None
    if revision is not None:
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            return None
        package_revision = revision
    package_version = None
    if version is not None:
        if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
            return None
        package_version = version
    effective = False if category in _NEVER_REPORTABLE else reportable
    consent = current_consent() if effective else None
    event = {
        "schema": 1,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "component": component,
        "severity": "ERROR",
        "event": "operation.end",
        "operation_id": incident,
        "trace_id": incident,
        "operation": operation,
        "status": "error",
    }
    if duration is not None:
        event["duration_ms"] = duration
    event["attributes"] = attributes
    event["reportable"] = effective
    event["reporting"] = consent
    if package_revision is not None:
        event["package_revision"] = package_revision
    if package_version is not None:
        event["package_version"] = package_version
    return event


def _append_event(root, component, event) -> bool:
    if not _lock.acquire(blocking=False):
        return False
    try:
        line = (json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(line) > _MAX_EVENT_BYTES:
            return False
        base = _as_local_absolute(default_root() if root is None else root)
        if base is None or not _ancestors_are_real_dirs(base, create=True):
            return False
        if _retention_blocked(base):
            return False
        leaf = base / "events" / component
        if not _ensure_owned_tree(base, leaf):
            return False
        # Publish the post-fork id in the name and the payload together.
        proc = _process_instance_id_locked()
        event["process_instance_id"] = proc
        line = (json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(line) > _MAX_EVENT_BYTES:
            return False
        filename = "%s-%s.jsonl" % (os.getpid(), proc)
        return _append_line(leaf, filename, line)
    except Exception:
        return False
    finally:
        _lock.release()


def _retention_blocked(root):
    """Offline-maintained quota gate. Unknown markers fail closed, no hot counter."""
    path = root / "retention-state.json"
    try:
        info = os.lstat(path)
        if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
            return True
    except FileNotFoundError:
        return False
    except OSError:
        return True
    raw = _read_regular_bounded(path, 2048)
    try:
        data = json.loads(raw) if raw is not None else None
        return not (isinstance(data, dict) and data.get("schema") == 1
                    and type(data.get("blocked")) is bool and data["blocked"] is False)
    except (ValueError, TypeError):
        return True


def _append_line(directory, filename, payload) -> bool:
    path = directory / filename
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(os.fspath(path), flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not _owned(info):
            return False
        if info.st_size >= _MAX_FILE_BYTES or info.st_size + len(payload) > _MAX_FILE_BYTES:
            return False
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        return os.write(fd, payload) == len(payload)
    finally:
        os.close(fd)


def _ensure_owned_tree(root, leaf) -> bool:
    """Create only root and descendants. Never chmod ancestors above root."""
    if not _owned_dir(root, create=True):
        return False
    current = root
    for part in leaf.relative_to(root).parts:
        current = current / part
        if not _owned_dir(current, create=True):
            return False
    return True


def _owned_dir(path, *, create) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        if not create:
            return False
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        except OSError:
            return False
        else:
            try:
                os.chmod(path, 0o700)
            except OSError:
                return False
        try:
            info = os.lstat(path)
        except OSError:
            return False
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False
    return _owned(info)


def _ancestors_are_real_dirs(path, *, create=False) -> bool:
    current = Path(path.anchor)
    parent = path.parent
    if parent == path:
        return True
    for part in parent.parts[1:]:
        if part in ("", ".", ".."):
            return False
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if not create:
                return False
            try:
                os.mkdir(current, 0o700)
                info = os.lstat(current)
            except FileExistsError:
                info = os.lstat(current)
            except OSError:
                return False
        except OSError:
            return False
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            return False
    return True


def _read_regular_bounded(path, limit):
    """Read a regular file without following a symlink or blocking on a FIFO."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return None
    if not _owned(info) or info.st_size <= 0 or info.st_size > limit:
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not _owned(info):
            return None
        if info.st_size <= 0 or info.st_size > limit:
            return None
        blob = os.read(fd, limit + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if not blob or len(blob) > limit:
        return None
    return blob


def _validated_policy(data):
    if not isinstance(data, dict):
        return None
    if data.get("schema") != _POLICY_SCHEMA or data.get("purpose") != _POLICY_PURPOSE:
        return None
    decision = data.get("decision")
    if decision not in ("enabled", "disabled"):
        return None
    repository = data.get("repository")
    if not _valid_repo(repository):
        return None
    revision = data.get("revision")
    if not isinstance(revision, str) or _INCIDENT_RE.fullmatch(revision) is None:
        return None
    roots = data.get("roots")
    if not isinstance(roots, list) or len(roots) > 32:
        return None
    clean_roots = []
    for item in roots:
        if not isinstance(item, str) or len(item) > 4096:
            return None
        if _as_local_absolute(item) is None:
            return None
        clean_roots.append(item)
    return {
        "schema": _POLICY_SCHEMA,
        "purpose": _POLICY_PURPOSE,
        "decision": decision,
        "repository": repository,
        "revision": revision,
        "roots": clean_roots,
    }


def _valid_repo(value) -> bool:
    if not isinstance(value, str) or _REPO_RE.fullmatch(value) is None:
        return False
    return True


def _as_local_absolute(value):
    if isinstance(value, Path):
        text = os.fspath(value)
    elif isinstance(value, str):
        text = value
    else:
        return None
    if not text or "\x00" in text or "\n" in text or "\r" in text:
        return None
    path = Path(text)
    if not path.is_absolute():
        return None
    raw = str(path)
    if raw.startswith("\\\\") or raw.startswith("//"):
        return None
    for part in path.parts[1:]:
        if part in ("", ".", ".."):
            return None
    return path


def _label(value, limit) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= limit
        and _LABEL_RE.fullmatch(value) is not None
    )


def _valid_incident(value) -> bool:
    return isinstance(value, str) and _INCIDENT_RE.fullmatch(value) is not None


def _int32(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if -2147483648 <= value <= 2147483647:
        return value
    return None


def _duration(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if not 0 <= value < 10**30:
        return None
    return int(value) if isinstance(value, int) else value


def _owned(info) -> bool:
    if os.name == "nt":
        return True
    try:
        return info.st_uid == os.geteuid()
    except Exception:
        return False


def _process_instance_id_locked() -> str:
    global _process_pid, _process_uuid
    pid = os.getpid()
    if _process_uuid is None or _process_pid != pid:
        _process_uuid = uuid.uuid4().hex
        _process_pid = pid
    return _process_uuid


def _warn_once() -> None:
    global _warned
    if _warned:
        return
    if not _warn_lock.acquire(blocking=False):
        return
    try:
        if _warned:
            return
        _warned = True
        try:
            # The failure result is authoritative. A full stderr pipe must not
            # block business cleanup, and changing its flags would affect peers.
            if os.name == "posix":
                descriptor = sys.stderr.fileno()
                if not os.get_blocking(descriptor):
                    os.write(descriptor, _STATIC_WARN.encode("ascii"))
        except Exception:
            pass
    finally:
        _warn_lock.release()
