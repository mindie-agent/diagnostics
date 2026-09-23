"""Standard-library logging with independent process files and bounded records."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import logging as std_logging
import json
import math
import os
from pathlib import Path
import re
import sys
import stat
import threading
import time
import uuid
import weakref

from .context import bind_context, current_context
from .redact import redact_text
from .reader_registry import reader_guard, cursor_snapshot, fully_consumed, pressure_blocked

SCHEMA = 1
MAX_RECORD_BYTES = 16_384
MAX_LOG_BYTES = 1_048_576
BACKUP_COUNT = 3
_COMPONENT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z")
_SENSITIVE = re.compile(r"password|secret|token|credential|authorization|cookie|private.?key|api.?key|command|stdout|stderr|body|transcript|(?:^|_)(?:env|environment|args|argv|payload|text|headers)(?:$|_)", re.I)
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_RECORDERS = {}
_VERSIONS = {}
_LOCK = threading.RLock()
_LIVE_RECORDERS = weakref.WeakSet()
_FAILURE_RECORDERS = OrderedDict()


def _after_fork():
    # Python resets stdlib logging locks, but it cannot reset our own locks.
    # Include recorders replaced by configure while an older operation lives.
    global _LOCK, _FAILURE_RECORDERS
    _LOCK = threading.RLock()
    _FAILURE_RECORDERS = OrderedDict()
    for recorder in list(_LIVE_RECORDERS):
        recorder._mutex = threading.RLock()
        recorder._logger = None
        recorder._path = None
        recorder._pid = None
        recorder._retry_at = 0.0
        recorder.dropped_records = 0
        recorder._process_id = uuid.uuid4().hex
        recorder._birth_pid = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _package(component):
    if component not in _VERSIONS:
        info = {"package_version": "unknown"}
        try:
            distribution_name = 'remote-dev' if component == 'remote-dev' else component
            module_name = 'remote_dev' if distribution_name == 'remote-dev' else distribution_name.replace('-', '_')
            distribution = metadata.distribution(distribution_name)
            module = sys.modules.get(module_name)
            loaded = getattr(module, '__file__', None)
            installed = distribution.locate_file(module_name + '/__init__.py')
            # An installed old wheel can coexist with a PYTHONPATH candidate.
            # Its direct_url commit is not evidence of the code being executed.
            if loaded and Path(loaded).resolve() == Path(installed).resolve():
                info["package_version"] = distribution.version
                direct = json.loads(distribution.read_text("direct_url.json") or "{}")
                revision = direct.get("vcs_info", {}).get("commit_id")
                if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{7,64}", revision):
                    info["package_revision"] = revision
        except Exception:
            pass
        _VERSIONS[component] = info
    return dict(_VERSIONS[component])


def _exception_info(exc):
    """No frame locals, source line reads or absolute paths."""
    try:
        chain, frames, seen = [], [], set()
        current = exc
        while current is not None and id(current) not in seen and len(chain) < 8:
            seen.add(id(current))
            chain.append(type(current).__name__)
            tb = current.__traceback__
            while tb is not None and len(frames) < 24:
                frame = tb.tb_frame
                module = frame.f_globals.get("__name__", "unknown")
                function = frame.f_code.co_name
                frames.append({"module": _text(module, 160), "function": _text(function, 100),
                               "line": tb.tb_lineno})
                tb = tb.tb_next
            current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        canonical = json.dumps({"chain": chain, "frames": frames}, sort_keys=True, separators=(",", ":"))
        return {"error_type": type(exc).__name__, "exception_message": _text(str(exc), 1200),
                "exception_chain": chain, "stack_frames": frames,
                "stack_fingerprint": hashlib.sha256(canonical.encode()).hexdigest()}
    except Exception:
        return {"error_type": type(exc).__name__, "exception_capture_failed": True}


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _text(value, size=512):
    if not isinstance(value, str):
        return "[unsupported]"
    # Sanitize before truncating so a truncated credential cannot evade a rule.
    return redact_text(value[:16_384])[:size]


def _attributes(attributes):
    def value(item, depth=0):
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int):
            return item if abs(item) < 10**30 else "[large integer]"
        if isinstance(item, float):
            return item if math.isfinite(item) else "[nonfinite]"
        if isinstance(item, str):
            return _text(item)
        if depth < 2 and isinstance(item, (tuple, list)):
            return [value(part, depth + 1) for part in item[:12]]
        if depth < 2 and isinstance(item, dict):
            return { _text(key, 80): ("[omitted]" if _SENSITIVE.search(key) else value(part, depth + 1))
                     for key, part in list(item.items())[:16] if isinstance(key, str)}
        return "[unsupported]"
    return {_text(key, 80): ("[omitted]" if _SENSITIVE.search(key) else value(item))
            for key, item in list(attributes.items())[:32] if isinstance(key, str)}


def default_root():
    chosen = os.environ.get("MINDIE_DIAGNOSTICS_ROOT")
    if chosen:
        return Path(chosen).expanduser()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "mindie" / "diagnostics"
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "mindie" / "diagnostics"


class _Handler(std_logging.Handler):
    """Stable bounded segments; only this writer's consumed closed slots recycle."""
    def __init__(self, filename, recorder):
        super().__init__()
        self.recorder = recorder
        self.root = Path(filename).parent.parent.parent
        base = Path(filename)
        self.paths = [base] + [base.with_name(base.name + '.' + str(i))
                              for i in range(1, BACKUP_COUNT + 1)]
        self.stream = None
        self.current = None
        self.owned = {}
        self.credit = None
        self.setFormatter(std_logging.Formatter('%(message)s'))

    def _drop(self):
        self.recorder.dropped_records += 1
        self.recorder._unavailable()

    def _select(self, size, blocked):
        deadline = time.monotonic() + .1
        with reader_guard(self.root, timeout=.05) as state:
            if state['status'] != 'ok':
                return False
            snapshot = None
            start = (self.paths.index(self.current) + 1) % len(self.paths) if self.current else 0
            ordered = self.paths[start:] + self.paths[:start]
            # Fill the bounded window before recycling; then cycle all slots.
            ordered.sort(key=lambda path: path.exists())
            for path in ordered:
                if path == self.current:
                    continue
                if time.monotonic() >= deadline:
                    return False
                try:
                    old = path.lstat()
                except FileNotFoundError:
                    if blocked:
                        continue
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                         | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                    credit = None
                else:
                    if (not stat.S_ISREG(old.st_mode) or old.st_nlink != 1
                            or self.owned.get(path) != (old.st_dev, old.st_ino)):
                        continue
                    if snapshot is None:
                        snapshot = cursor_snapshot(state['readers'],
                                                   timeout=max(0, deadline - time.monotonic()))
                    if (not fully_consumed(snapshot, path, old.st_ino, old.st_size)
                            or (blocked and size > old.st_size)):
                        continue
                    temporary = path.with_name('.segment-' + uuid.uuid4().hex)
                    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                         | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                    try:
                        new = os.fstat(descriptor)
                        current = path.lstat()
                        if ((new.st_dev, new.st_ino) == (old.st_dev, old.st_ino)
                                or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                                != (old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns)):
                            os.close(descriptor)
                            descriptor = None
                            continue
                        if os.name == 'nt':
                            # Windows rejects replacement while this temporary
                            # file is open without delete-sharing. The current
                            # writer remains open until the new slot is ready.
                            os.close(descriptor)
                            descriptor = None
                        os.replace(temporary, path)
                        if descriptor is None:
                            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND
                                                 | getattr(os, 'O_NOFOLLOW', 0)
                                                 | getattr(os, 'O_NONBLOCK', 0))
                            reopened = os.fstat(descriptor)
                            selected = path.lstat()
                            expected = (new.st_dev, new.st_ino)
                            if (not stat.S_ISREG(reopened.st_mode)
                                    or reopened.st_nlink != 1
                                    or not stat.S_ISREG(selected.st_mode)
                                    or (reopened.st_dev, reopened.st_ino) != expected
                                    or (selected.st_dev, selected.st_ino) != expected):
                                raise OSError('diagnostic segment identity changed')
                    except BaseException:
                        if descriptor is not None:
                            os.close(descriptor)
                        raise
                    finally:
                        temporary.unlink(missing_ok=True)
                    credit = old.st_size if blocked else None
                try:
                    info = os.fstat(descriptor)
                    stream = os.fdopen(descriptor, 'wb', buffering=0)
                except BaseException:
                    os.close(descriptor)
                    raise
                previous = self.stream
                self.stream = stream
                self.current = path
                self.owned[path] = (info.st_dev, info.st_ino)
                self.credit = credit
                self.recorder._path = path
                if previous is not None:
                    previous.close()
                return True
        return False

    def emit(self, record):
        try:
            raw = (self.format(record) + '\n').encode('utf-8')
            if len(raw) > MAX_LOG_BYTES:
                self._drop()
                return
            blocked = pressure_blocked(self.root)
            present = os.fstat(self.stream.fileno()).st_size if self.stream is not None else 0
            cap = MAX_LOG_BYTES
            if blocked:
                cap = min(cap, self.credit if self.credit is not None else present)
            if self.stream is None or present + len(raw) > cap:
                if not self._select(len(raw), blocked):
                    self._drop()
                    return
            self.stream.write(raw)
        except Exception:
            self.handleError(record)

    def handleError(self, record):
        self._drop()

    def close(self):
        try:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
        finally:
            super().close()

class Recorder:
    def __init__(self, component, root=None, level=None, version=None):
        self.component = component if isinstance(component, str) and _COMPONENT.fullmatch(component) else "unknown"
        self.root = root
        self.level = _LEVELS.get(str(level or os.environ.get("MINDIE_LOG_LEVEL", "INFO")).upper(), 20)
        self._pid = None
        self._process_id = uuid.uuid4().hex
        self._birth_pid = os.getpid()
        self.package = _package(self.component)
        if version is not None:
            self.package["package_version"] = _text(version, 80)
        self._path = None
        self._logger = None
        self._mutex = threading.RLock()
        self.logging_failed = False
        self.dropped_records = 0
        self._notified = False
        self._retry_at = 0.0
        with _LOCK:
            _LIVE_RECORDERS.add(self)

    def _unavailable(self):
        self.logging_failed = True
        if self._notified:
            return
        self._notified = True
        try:
            descriptor = sys.stderr.fileno()
            # Never change a host-owned descriptor's flags. An optional warning
            # is safe only if the descriptor is already nonblocking.
            if os.name == "posix" and not os.get_blocking(descriptor):
                os.write(descriptor, b"mindie-diagnostics: WARNING diagnostic storage unavailable; business outcome unchanged\n")
        except Exception:
            pass

    def _get_logger(self):
        pid = os.getpid()
        if pid != self._birth_pid:
            self._process_id, self._birth_pid = uuid.uuid4().hex, pid
        if self._pid == pid and self._logger is not None:
            return self._logger
        if time.monotonic() < self._retry_at:
            return None
        root = Path(self.root).expanduser() if self.root is not None else default_root()
        folder = root / "events" / self.component
        if any(parent.is_symlink() for parent in (folder, *folder.parents)):
            raise OSError("diagnostic log directory must not be a symlink")
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            folder.chmod(0o700)  # Only this component's diagnostic leaf, never its parents.
        path = folder / (str(pid) + "-" + self._process_id + ".jsonl")
        logger = std_logging.Logger("mindie." + self.component, level=self.level)
        logger.propagate = False
        logger.addHandler(_Handler(path, self))
        # A fork does not reuse its parent's handler/file. Do not close parent state.
        self._pid, self._path, self._logger = pid, path, logger
        return logger

    @property
    def record_ref(self):
        return str(self._path.absolute()) if self._path is not None else None

    def _emit(self, event):
        severity = _LEVELS.get(event.get("severity", "INFO"), 20)
        if severity < self.level:
            return
        if not self._mutex.acquire(blocking=False):
            self.dropped_records += 1
            self._unavailable()
            return
        try:
            logger = self._get_logger()
            if logger is None:
                self.dropped_records += 1
                return
            event = {**event, **self.package, "process_instance_id": self._process_id}
            encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
                event = {**event, "attributes": {"attributes_omitted": True}}
                encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            logger.log(severity, encoded)
        except Exception:
            self._retry_at = time.monotonic() + 1
            self.dropped_records += 1
            self._unavailable()
        finally:
            self._mutex.release()

    def operation(self, name, *, level="INFO", **attributes):
        return Operation(self, name, attributes, level=level)

    def event(self, level, event, **attributes):
        """Emit a standalone event; operations are preferred for timed work."""
        try:
            self._emit({"schema": SCHEMA, "timestamp": _utc(), "monotonic_ns": time.monotonic_ns(),
                        "pid": os.getpid(), "component": self.component, "severity": _severity(level),
                        "event": _text(event, 120), **current_context(), "attributes": _attributes(attributes)})
        except Exception:
            self._unavailable()

    def close(self):
        with self._mutex:
            if self._logger is not None:
                for handler in self._logger.handlers:
                    try:
                        handler.close()
                    except Exception:
                        self._unavailable()
                self._logger = None


def _severity(level):
    number = _LEVELS.get(str(level).upper(), level if isinstance(level, int) else 20)
    if number >= 50:
        return "CRITICAL"
    if number >= 40:
        return "ERROR"
    if number >= 30:
        return "WARNING"
    return "DEBUG" if number < 20 else "INFO"


class Operation:
    def __init__(self, recorder, name, attributes, parent=None, level="INFO"):
        self.recorder = recorder
        self.name = _text(name, 120)
        self._attributes = attributes
        self._parent = parent
        self._level = level
        self.operation_id = parent.operation_id if parent else uuid.uuid4().hex
        self.phase_id = uuid.uuid4().hex if parent else None
        self.trace_id = None
        self.parent_operation_id = None
        self.started_at = None
        self.finished_at = None
        self._started_ns = None
        self._finished_ns = None
        self.status = "pending"
        self._binding = None
        self._failure = {}
        self._phases = []
        self.parent_phase_id = None

    def __enter__(self):
        inherited = current_context()
        self.parent_phase_id = inherited.get("phase_id") if self._parent else None
        self.trace_id = inherited.get("trace_id") or (self._parent.trace_id if self._parent else None) or uuid.uuid4().hex
        self.parent_operation_id = self._parent.parent_operation_id if self._parent else inherited.get("operation_id")
        self.started_at, self._started_ns = _utc(), time.monotonic_ns()
        self.status = "running"
        context = {"trace_id": self.trace_id, "operation_id": self.operation_id,
                   "parent_operation_id": self.parent_operation_id}
        if self.phase_id:
            context["phase_id"] = self.phase_id
        self._binding = bind_context(context)
        self._binding.__enter__()
        self.event(self._level, "phase.start" if self._parent else "operation.start", **self._attributes)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            clean_exit = isinstance(exc, SystemExit) and exc.code in (None, 0)
            if exc_type is not None and not clean_exit:
                self.status = "cancelled" if exc_type.__name__ in {"CancelledError", "KeyboardInterrupt", "GeneratorExit"} else "error"
                self._failure.setdefault("category", "cancelled" if self.status == "cancelled" else "unhandled_exception")
                self._failure.update(_exception_info(exc))
            elif self.status == "running":
                self.status = "success"
            self.finished_at, self._finished_ns = _utc(), time.monotonic_ns()
            severity = "ERROR" if self.status == "error" else "WARNING" if self.status == "cancelled" else self._level
            self.event(severity, "phase.end" if self._parent else "operation.end", **self._failure)
            if self._parent and len(self._parent._phases) < 32:
                self._parent._phases.append({"phase": self.name, "phase_id": self.phase_id,
                                            "parent_phase_id": self.parent_phase_id,
                                            "status": self.status, "duration_ms": round(self._duration(), 3)})
        finally:
            if self._binding is not None:
                self._binding.__exit__(exc_type, exc, traceback)
        return False

    def event(self, level, event, **attributes):
        try:
            payload = {"schema": SCHEMA, "timestamp": _utc(), "monotonic_ns": time.monotonic_ns(),
                       "pid": os.getpid(), "component": self.recorder.component, "severity": _severity(level),
                       "event": _text(event, 120), "operation_id": self.operation_id, "trace_id": self.trace_id,
                       "parent_operation_id": self.parent_operation_id,
                       "operation": self._parent.name if self._parent else self.name,
                       "status": self.status, "duration_ms": self._duration(),
                       "attributes": _attributes(attributes)}
            if self.phase_id:
                payload.update(phase_id=self.phase_id, phase=self.name, parent_phase_id=self.parent_phase_id)
            # Preserve bounded structured traceback beyond the generic shallow attrs.
            for key in ("exception_chain", "stack_frames", "stack_fingerprint"):
                if key in attributes:
                    payload["attributes"][key] = attributes[key]
            self.recorder._emit(payload)
        except Exception:
            self.recorder._unavailable()

    def phase(self, name, *, level="INFO", **attributes):
        return Operation(self.recorder, name, attributes, parent=self._parent or self, level=level)

    def fail(self, category, *, retryable=False, submission_state=None, **attributes):
        self.status = "error"
        self._failure.update(category=_text(category, 100), retryable=bool(retryable))
        if submission_state is not None:
            self._failure["submission_state"] = _text(submission_state, 80)
        exception = attributes.pop("exception", None)
        self._failure.update(_attributes(attributes))
        if isinstance(exception, BaseException):
            self._failure.update(_exception_info(exception))
        self.event("ERROR", "operation.failure", **self._failure)
        if self._parent:
            self._parent.status = "error"
            self._parent._failure.update(self._failure)

    def _duration(self):
        if self._started_ns is None:
            return 0.0
        return max(0, (self._finished_ns or time.monotonic_ns()) - self._started_ns) / 1_000_000

    def summary(self):
        return {"operation_id": self.operation_id, "trace_id": self.trace_id,
                "parent_operation_id": self.parent_operation_id, "component": self.recorder.component,
                "operation": self._parent.name if self._parent else self.name,
                "status": self.status, "duration_ms": round(self._duration(), 3),
                "started_at": self.started_at, "finished_at": self.finished_at,
                "record_ref": self.recorder.record_ref, "logging_failed": self.recorder.logging_failed,
                "dropped_records": self.recorder.dropped_records,
                "phases": list(self._phases),
                **({"phase_id": self.phase_id, "phase": self.name} if self.phase_id else {})}


def configure(component, *, root=None, level=None, version=None):
    with _LOCK:
        prior = _RECORDERS.get(component)
        resolved_level = _LEVELS.get(str(level or os.environ.get("MINDIE_LOG_LEVEL", "INFO")).upper(), 20)
        if prior is not None and prior.root == root and prior.level == resolved_level and (version is None or prior.package["package_version"] == version):
            return prior
        recorder = Recorder(component, root=root, level=level, version=version)
        _RECORDERS[component] = recorder
        return recorder


def get_recorder(component):
    with _LOCK:
        if component not in _RECORDERS:
            _RECORDERS[component] = Recorder(component)
        return _RECORDERS[component]


def append_failure_event(component, root, event) -> bool:
    """Append a validated failure with exact build/consent and no lock backlog.

    Integration owns the event schema. Contention returns False for its visible
    logging_failed result; ordinary business work never waits behind a writer.
    """
    recorder = None
    acquired = False
    try:
        if not isinstance(component, str) or not _COMPONENT.fullmatch(component):
            return False
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False)
        if len(encoded.encode('utf-8')) > MAX_RECORD_BYTES:
            return False
        selected_root = Path(root).absolute()
        key = (component, str(selected_root))
        if not _LOCK.acquire(blocking=False):
            return False
        try:
            recorder = _FAILURE_RECORDERS.get(key)
            if recorder is None:
                if len(_FAILURE_RECORDERS) >= 32:
                    prior_key, prior = next(iter(_FAILURE_RECORDERS.items()))
                    if not prior._mutex.acquire(blocking=False):
                        return False
                    try:
                        _FAILURE_RECORDERS.pop(prior_key)
                        prior.close()
                    finally:
                        prior._mutex.release()
                recorder = Recorder(component, root=selected_root, level='ERROR')
                _FAILURE_RECORDERS[key] = recorder
            else:
                _FAILURE_RECORDERS.move_to_end(key)
            acquired = recorder._mutex.acquire(blocking=False)
            if not acquired:
                recorder.dropped_records += 1
                recorder.logging_failed = True
                return False
        finally:
            _LOCK.release()
        before = recorder.dropped_records
        logger = recorder._get_logger()
        if logger is None:
            recorder.dropped_records += 1
            return False
        # Process identity belongs to this writer; build and consent stay exact.
        payload = dict(event, process_instance_id=recorder._process_id)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False)
        if len(encoded.encode('utf-8')) > MAX_RECORD_BYTES:
            recorder.dropped_records += 1
            recorder.logging_failed = True
            return False
        logger.error(encoded)
        return recorder.dropped_records == before
    except Exception:
        if recorder is not None:
            recorder.logging_failed = True
            recorder.dropped_records += 1
        return False
    finally:
        if acquired:
            recorder._mutex.release()
