"""Installed-metadata enrichment for the local failure recorder.

Stdlib only. Import does no file I/O. Never raises and never starts a service.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import sys
import stat
import time
import threading
import uuid
from pathlib import Path

from . import fallback as f

_MAX_FRAMES = 24
_META_CACHE_MAX = 32
_HEX_REV = re.compile(r"[0-9a-f]{7,64}\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,79}\Z")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,119}\Z")
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}(?:\.[A-Za-z_][A-Za-z0-9_]{0,79}){0,12}\Z")

_meta_lock = threading.Lock()
_meta_cache = {}


def _module_name(component):
    if component == "mindie-knowledge":
        return "mindie_knowledge"
    if component == "remote-dev":
        return "remote_dev"
    if not isinstance(component, str):
        return ""
    return component.replace("-", "_")


def _usable_version(value):
    if not isinstance(value, str) or value.lower() == "unknown":
        return None
    if _VERSION.fullmatch(value) is None:
        return None
    return value


def _usable_revision(value):
    if not isinstance(value, str) or value.lower() == "unknown":
        return None
    if _HEX_REV.fullmatch(value) is None:
        return None
    return value


def _lookup_meta(component):
    module_name = _module_name(component)
    if not module_name or not _MODULE.fullmatch(module_name):
        return (None, None)
    loaded = sys.modules.get(module_name)
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if not isinstance(loaded_file, str) or not loaded_file:
        return (None, None)
    dist = importlib.metadata.distribution(component)
    located = os.fspath(dist.locate_file(module_name + "/__init__.py"))
    if os.path.normcase(os.path.abspath(loaded_file)) != os.path.normcase(os.path.abspath(located)):
        return (None, None)
    version = _usable_version(getattr(dist, "version", None))
    revision = None
    raw = dist.read_text("direct_url.json")
    if raw:
        info = json.loads(raw)
        vcs = info.get("vcs_info") if isinstance(info, dict) else None
        if isinstance(vcs, dict):
            revision = _usable_revision(vcs.get("commit_id"))
    return (version, revision)


def _component_meta(component):
    try:
        if _meta_lock.acquire(blocking=False):
            try:
                cached = _meta_cache.get(component)
                if cached is not None:
                    return cached
                found = _lookup_meta(component)
                if len(_meta_cache) >= _META_CACHE_MAX:
                    _meta_cache.pop(next(iter(_meta_cache)))
                _meta_cache[component] = found
                return found
            finally:
                _meta_lock.release()
        return _lookup_meta(component)
    except Exception:
        return (None, None)


def _frame_dict(frame, component_module, line=None):
    name = frame.f_globals.get("__name__")
    if not isinstance(name, str) or _MODULE.fullmatch(name) is None:
        return None
    if name == "mindie_diagnostics.integration":
        return None
    if component_module != "mindie_diagnostics" and (
        name == "mindie_diagnostics" or name.startswith("mindie_diagnostics.")
    ):
        return None
    if name != component_module and not name.startswith(component_module + "."):
        return None
    function = frame.f_code.co_name
    if not isinstance(function, str) or _IDENT.fullmatch(function) is None:
        return None
    line = frame.f_lineno if line is None else line
    if isinstance(line, bool) or not isinstance(line, int) or not 1 <= line <= 10000000:
        return None
    return {"module": name, "function": function, "line": line}


def _stack_frames(component, exception):
    module = _module_name(component)
    if not module:
        return []
    frames = []
    tb = exception.__traceback__ if isinstance(exception, BaseException) else None
    if tb is not None:
        walked = 0
        while tb is not None and walked < _MAX_FRAMES and len(frames) < _MAX_FRAMES:
            item = _frame_dict(tb.tb_frame, module, tb.tb_lineno)
            if item is not None:
                frames.append(item)
            tb = tb.tb_next
            walked += 1
        return frames
    try:
        frame = sys._getframe(1)
    except ValueError:
        return []
    hops = 0
    while frame is not None and hops < 8:
        current = frame.f_globals.get("__name__")
        if current != "mindie_diagnostics.integration":
            break
        frame = frame.f_back
        hops += 1
    walked = 0
    while frame is not None and walked < _MAX_FRAMES:
        item = _frame_dict(frame, module)
        if item is not None:
            frames.append(item)
        frame = frame.f_back
        walked += 1
    return frames


def _attach_frames(event, component, exception):
    frames = _stack_frames(component, exception)
    if not frames:
        return
    canonical = json.dumps(frames, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    event["attributes"]["stack_frames"] = frames
    event["attributes"]["stack_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    """Record one failure. Existing incident ids are not written again."""
    try:
        if f._valid_incident(incident_id):
            return f.record_failure(
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
                root=root,
                incident_id=incident_id,
            )
        if version is None or revision is None:
            meta_version, meta_revision = _component_meta(component)
            if version is None:
                version = meta_version
            if revision is None:
                revision = meta_revision
        fresh = uuid.uuid4().hex
        event = f._build_event(
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
        _attach_frames(event, component, exception)
        from .logging import append_failure_event
        selected_root = f._as_local_absolute(f.default_root() if root is None else root)
        if selected_root is not None and append_failure_event(component, selected_root, event):
            return {"recorded": True, "incident_id": fresh, "logging_failed": False}
        f._warn_once()
        return {
            "recorded": False,
            "incident_id": None,
            "logging_failed": True,
            "error": "logging_failed",
        }
    except Exception:
        f._warn_once()
        return {
            "recorded": False,
            "incident_id": None,
            "logging_failed": True,
            "error": "logging_failed",
        }


def _error(category):
    return {"status": "error", "category": category}


def _merge_roots(existing, supplied):
    if supplied is None:
        supplied_paths = []
    elif isinstance(supplied, (str, Path)) or not isinstance(supplied, (list, tuple)):
        return None
    else:
        supplied_paths = list(supplied)
    merged = []
    seen = set()
    for item in list(existing or []) + supplied_paths:
        path = f._as_local_absolute(item)
        if path is None:
            return None
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)
        if len(merged) > 32:
            return None
    if not merged:
        default = f._as_local_absolute(f.default_root())
        if default is None:
            return None
        merged.append(str(default))
    return merged


def _write_policy(path, payload):
    try:
        old = os.lstat(path)
        if not stat.S_ISREG(old.st_mode) or old.st_nlink != 1 or not f._owned(old):
            return False
    except FileNotFoundError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    tmp = path.parent / (".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    try:
        fd = os.open(tmp, flags, 0o600)
        try:
            if os.write(fd, payload) != len(payload):
                return False
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def configure_reporting(enabled, *, repository="mindie-agent/mindie-agent", config=None, roots=None) -> dict:
    """Write the local reporting policy. Does not install or check a worker."""
    try:
        if type(enabled) is not bool or not f._valid_repo(repository):
            return _error("invalid_arguments")
        path = f._as_local_absolute(f.policy_path(config))
        if path is None or not f._ancestors_are_real_dirs(path, create=True):
            return _error("invalid_configuration")
        existing = f.read_policy(config)
        chosen = _merge_roots(None if existing is None else existing.get("roots"), roots)
        if chosen is None:
            return _error("invalid_configuration")
        decision = "enabled" if enabled else "disabled"
        if (
            existing is not None
            and existing.get("decision") == decision
            and existing.get("repository") == repository
        ):
            revision = existing["revision"]
        else:
            revision = uuid.uuid4().hex
        policy = f._validated_policy({
            "schema": "mindie.diagnostics.reporting.v1",
            "purpose": "tool_fault_reporting",
            "decision": decision,
            "repository": repository,
            "revision": revision,
            "roots": chosen,
        })
        if policy is None:
            return _error("invalid_configuration")
        blob = json.dumps(policy, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(blob) > 16384 or not _write_policy(path, blob):
            return _error("write_failed")
        return {
            "status": "configured",
            "enabled": enabled,
            "repository": repository,
            "configuration_path": str(path),
            "state_path": str(f.state_path(config)),
            "worker": {"status": "not_checked"},
            "recovery_hint": "Run reporting ensure outside the hook to install or verify the worker.",
        }
    except f.PolicyUnavailable:
        return _error("reporting_policy_unavailable")
    except Exception:
        return _error("write_failed")


_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{16,64}\Z")
_STATIC_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_TABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_HEALTH_LIMIT = 65536
_QUERY_DEADLINE = 0.15
_OUTBOX_COLUMNS = (
    "fingerprint",
    "state",
    "attempts",
    "issue_number",
    "issue_url",
    "last_error",
    "payload",
)


def _safe_label(value):
    if isinstance(value, str) and _LABEL.fullmatch(value) is not None:
        return value
    return None


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _worker_view(config):
    state = f._as_local_absolute(f.state_path(config))
    if state is None:
        return {"status": "unavailable", "healthy": False}
    path = state / "worker-health.json"
    try:
        os.lstat(path)
    except FileNotFoundError:
        return {"status": "not_started", "healthy": False}
    except OSError:
        return {"status": "unavailable", "healthy": False}
    raw = f._read_regular_bounded(path, _HEALTH_LIMIT)
    try:
        data = json.loads(raw) if raw is not None else None
        if not isinstance(data, dict) or data.get("schema") != 1:
            raise ValueError
        status = _safe_label(data.get("status"))
        stage = _safe_label(data.get("stage"))
        heartbeat = _finite_number(data.get("heartbeat_at"))
        progress = _finite_number(data.get("progress_at"))
        interval = _finite_number(data.get("interval"))
        pid = data.get("pid")
        if not status or heartbeat is None or progress is None or interval is None or type(pid) is not int or pid <= 0:
            raise ValueError
        age = max(0, time.time() - heartbeat)
        stalled = status == "running" and time.time() - progress > max(1200, interval * 3)
        alive = None
        if os.name == "posix":
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except OSError:
                pass
        if status in {"running", "idle"}:
            if alive is False:
                status = "stopped"
            elif age > 60:
                status = "stale"
            elif alive is not True:
                status = "unverified"
            elif stalled:
                status = "stalled"
        view = {"status": status, "pid": pid, "healthy": status in {"running", "idle"},
                "heartbeat_age_s": age, "progress_stalled": stalled}
        if stage:
            view["stage"] = stage
        view['version'] = _usable_version(data.get('package_version'))
        view['revision'] = _usable_revision(data.get('package_revision'))
        digest = data.get('runtime_source')
        if isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest):
            view['runtime_source'] = digest
        return view
    except (ValueError, TypeError, KeyError, OverflowError):
        return {"status": "unavailable", "healthy": False}


def _queue_absent():
    return {"counts": {}, "recent": []}


def _queue_unavailable():
    return {
        "status": "unavailable",
        "hint": "outbox is not a safe local database",
        "counts": {},
        "recent": [],
    }


def _issue_url(value, repository):
    if not isinstance(value, str) or not f._valid_repo(repository):
        return None
    prefix = "https://github.com/%s/issues/" % repository
    if not value.startswith(prefix):
        return None
    number = value[len(prefix):]
    if not number.isascii() or not number.isdigit() or number.startswith("0") and number != "0":
        return None
    if len(number) > 12:
        return None
    return prefix + number


def _recent_row(row, repository):
    fingerprint, state, attempts, issue_number, issue_url, last_error, payload = row
    item = {}
    if isinstance(fingerprint, str) and _FINGERPRINT.fullmatch(fingerprint) is not None:
        item["fingerprint"] = fingerprint
    label = _safe_label(state)
    if label is not None:
        item["state"] = label
    if type(attempts) is int and 0 <= attempts <= 1000000:
        item["attempts"] = attempts
    if type(issue_number) is int and 0 <= issue_number <= 10**12:
        item["issue_number"] = issue_number
    url = _issue_url(issue_url, repository)
    if url is not None:
        item["issue_url"] = url
    if isinstance(last_error, str) and _STATIC_CODE.fullmatch(last_error) is not None:
        item["last_error"] = last_error
    if isinstance(payload, str) and len(payload.encode('utf-8')) <= 48000:
        try:
            data = json.loads(payload)
            ids = data.get('incident_ids', []) if isinstance(data, dict) else []
            if isinstance(ids, list):
                ids = list(dict.fromkeys(value for value in ids[-80:] if f._valid_incident(value)))
                item['incident_ids'] = ids[-8:]
                item['incident_ids_truncated'] = len(ids) > 8
        except (ValueError, TypeError):
            pass
    return item


def _read_outbox(config, repository):
    import sqlite3
    import time

    state = f._as_local_absolute(f.state_path(config))
    if state is None:
        return _queue_absent()
    path = state / "reporter.sqlite3"
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return _queue_absent()
    except OSError:
        return _queue_unavailable()
    if os.path.islink(path) or not f._ancestors_are_real_dirs(path):
        return _queue_unavailable()
    import stat as statmod
    if not statmod.S_ISREG(info.st_mode) or not f._owned(info) or info.st_nlink != 1 or info.st_size <= 0:
        return _queue_unavailable()
    deadline = time.monotonic() + _QUERY_DEADLINE
    try:
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.1)
    except Exception:
        return _queue_unavailable()
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=100")

        def _progress():
            return 1 if time.monotonic() >= deadline else 0

        conn.set_progress_handler(_progress, 1000)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(incidents)")}
        if not set(_OUTBOX_COLUMNS) <= columns:
            return _queue_unavailable()
        table = "incidents"
        counts = {}
        for state, count in conn.execute("SELECT state, COUNT(*) FROM %s GROUP BY state" % table):
            label = _safe_label(state)
            if label is None or type(count) is not int or count < 0:
                continue
            counts[label] = count
        selected = ", ".join(_OUTBOX_COLUMNS[:-1]) + ", CASE WHEN length(CAST(payload AS BLOB))<=48000 THEN payload ELSE NULL END"
        query = "SELECT %s FROM %s ORDER BY last_seen DESC LIMIT 5" % (selected, table)
        recent = [_recent_row(row, repository) for row in conn.execute(query)]
        return {"counts": counts, "recent": recent}
    except Exception:
        return _queue_unavailable()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _retention_view(roots):
    view = []
    for root in roots[:32]:
        raw = f._read_regular_bounded(Path(root) / "retention-state.json", 2048)
        if raw is None:
            if os.path.lexists(Path(root) / "retention-state.json"):
                view.append({"root": root, "status": "unavailable", "blocked": True})
            continue
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("schema") != 1 or type(data.get("blocked")) is not bool:
                raise ValueError
            item = {"root": root, "blocked": data['blocked'], "limited": data.get('limited') is True}
            for key in ('checked_at', 'remaining_bytes', 'remaining_files', 'unread_files', 'active_or_unknown_files'):
                value = _finite_number(data.get(key))
                if value is not None and value >= 0:
                    item[key] = value
            view.append(item)
        except (ValueError, TypeError):
            view.append({"root": root, "status": "unavailable", "blocked": True})
    return view


def _runtime_view(config):
    path = f.policy_path(config).with_suffix('.runtime') / 'current.json'
    try:
        os.lstat(path)
    except FileNotFoundError:
        return {'status': 'not_prepared'}
    except OSError:
        return {'status': 'unavailable'}
    raw = f._read_regular_bounded(path, 4096)
    if raw is None:
        return {'status': 'unavailable'}
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get('schema') != 1:
            raise ValueError
        return {'status': 'prepared', 'version': _usable_version(value.get('version')),
                'revision': _usable_revision(value.get('revision'))}
    except (ValueError, TypeError):
        return {'status': 'unavailable'}


def reporting_status(*, config=None) -> dict:
    """Read-only bounded local faults, policy, worker, and outbox. Creates nothing."""
    from .local_snapshot import recent_failures
    try:
        policy = f.read_policy(config)
    except f.PolicyUnavailable:
        return {"status": "configuration_unavailable", "category": "reporting_policy_unavailable",
                "local": {"status": "unavailable", "recent": []}}
    roots = policy["roots"] if policy else [str(f.default_root())]
    local = recent_failures(roots)
    local["retention"] = _retention_view(roots)
    try:
        path = f._as_local_absolute(f.policy_path(config))
        if path is None:
            return {"status": "configuration_unavailable", "category": "invalid_configuration", "local": local}
        try:
            os.lstat(path)
        except FileNotFoundError:
            return {"status": "not_configured", "configuration_path": str(path), "local": local}
        if policy is None:
            return {"status": "configuration_unavailable", "category": "invalid_configuration", "local": local}
        return {
            "status": "configured", "enabled": policy["decision"] == "enabled",
            "repository": policy["repository"], "revision": policy["revision"],
            "configuration_path": str(path), "state_path": str(f.state_path(config)),
            "worker": _worker_view(config), "queue": _read_outbox(config, policy["repository"]),
            "local": local, "runtime": _runtime_view(config),
            "recovery_hint": "Local record_ref identifies the bounded log. For a stopped reporter, explicitly run reporting ensure; status never retries publication.",
        }
    except Exception:
        return {"status": "configuration_unavailable", "category": "status_failed", "local": local}
