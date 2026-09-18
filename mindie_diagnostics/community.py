"""Local contribution consent, independent of task/resource authority.

The workspace owns writing this receipt. Missing, unreadable, malformed or old
receipts never authorize an upload. Private references never enter public data.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re

POLICY_ENV = "MINDIE_COMMUNITY_POLICY"
SCHEMA = "mindie.community.v1"
_ID = re.compile(r"[0-9a-f]{32}\Z")
_UNSET = object()
_POLICY = ContextVar("mindie_community_policy", default=_UNSET)
_GUARD = ContextVar("mindie_community_guard", default=_UNSET)


class ConsentWithdrawn(RuntimeError):
    """No new remote action may begin for this consent revision."""


def _path(value):
    if not isinstance(value, str) or not value or len(value.encode('utf-8')) > 4096 or "\x00" in value:
        raise ValueError("invalid community policy path")
    # A WSL worker can consume a Windows writer's explicit local reference.
    # UNC/network pointers and relative paths are deliberately unsupported.
    windows = PureWindowsPath(value)
    if windows.drive:
        if not re.fullmatch(r"[A-Za-z]:", windows.drive) or not windows.is_absolute():
            raise ValueError("community policy must be a local absolute path")
        if os.name != "nt":
            value = str(Path("/mnt") / windows.drive[0].lower() / Path(*windows.parts[1:]))
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("community policy must be absolute")
    return path


def read_policy(path):
    """Read a bounded current receipt; no discovery, writes or network calls."""
    from .bundle import _safe_open
    try:
        with _safe_open(_path(str(path))) as stream:
            raw = stream.read(16385)
        if len(raw) > 16384:
            return None
        value = json.loads(raw)
        if (not isinstance(value, dict) or value.get("schema") != SCHEMA
                or value.get("decision") not in {"enabled", "disabled"}
                or any(not isinstance(value.get(key), str) or not _ID.fullmatch(value[key])
                       for key in ("workspace_id", "revision"))):
            return None
        return {key: value[key] for key in ("schema", "workspace_id", "decision", "revision")}
    except (OSError, ValueError, TypeError, RecursionError):
        return None


@contextmanager
def bind_community_policy(path):
    """Bind one request's workspace receipt; None explicitly disables sharing."""
    token = _POLICY.set(str(path) if path is not None else None)
    try:
        yield
    finally:
        _POLICY.reset(token)


def current_consent():
    """Snapshot enabled consent for a new local event, never infer from cwd."""
    path = _POLICY.get()
    if path is _UNSET:
        path = os.environ.get(POLICY_ENV)
    value = read_policy(path) if path is not None else None
    if value is None or value["decision"] != "enabled":
        return None
    return {"policy_file": str(path), "workspace_id": value["workspace_id"],
            "revision": value["revision"]}


def consent_allowed(reference):
    if (not isinstance(reference, dict)
            or set(reference) != {"policy_file", "workspace_id", "revision"}):
        return False
    value = read_policy(reference.get("policy_file"))
    return bool(value and value["decision"] == "enabled"
                and all(value[key] == reference[key] for key in ("workspace_id", "revision")))


def scope_key(key, reference):
    """Separate matching failures and operation IDs from independent workspaces."""
    return hashlib.sha256(json.dumps([reference["workspace_id"], reference["revision"], key],
                                    separators=(",", ":")).encode()).hexdigest()


def require_consent(reference):
    if not consent_allowed(reference):
        raise ConsentWithdrawn("community_consent_unavailable_or_withdrawn")


def check_remote_action():
    reference = _GUARD.get()
    if reference is not _UNSET:
        require_consent(reference)


@contextmanager
def guard_consent(reference):
    token = _GUARD.set(reference)
    try:
        require_consent(reference)
        yield
    finally:
        _GUARD.reset(token)
