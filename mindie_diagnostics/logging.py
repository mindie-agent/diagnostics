"""Standard-library logging with independent process files and bounded records."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
from logging.handlers import RotatingFileHandler
import logging as std_logging
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
import weakref

from .context import bind_context, current_context
from .community import current_consent
from .redact import redact_text

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


def _after_fork():
    # Python resets stdlib logging locks, but it cannot reset our own locks.
    # Include recorders replaced by configure while an older operation lives.
    global _LOCK
    _LOCK = threading.RLock()
    for recorder in list(_LIVE_RECORDERS):
        recorder._mutex = threading.RLock()
        recorder._logger = None
        recorder._path = None
        recorder._pid = None
        recorder._retry_at = 0.0
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


class _Handler(RotatingFileHandler):
    def __init__(self, filename, recorder):
        super().__init__(filename, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
        self.recorder = recorder
        self.setFormatter(std_logging.Formatter("%(message)s"))

    def _open(self):
        # Called again after every rollover. Do not rely on the caller's umask
        # or create a briefly world-readable file and chmod it afterwards.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.baseFilename, flags, 0o600)
        try:
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            return os.fdopen(descriptor, self.mode, encoding=self.encoding, errors=self.errors)
        except BaseException:
            os.close(descriptor)
            raise

    def handleError(self, record):
        # logging.raiseExceptions must never turn this into payload/traceback output.
        self.recorder._unavailable()


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
            sys.stderr.write("mindie-diagnostics: WARNING diagnostic storage unavailable; business outcome unchanged\n")
            sys.stderr.flush()
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
        try:
            with self._mutex:
                logger = self._get_logger()
                if logger is None:
                    return
                event = {"community": current_consent() if "community" not in event else event["community"],
                         **event, **self.package, "process_instance_id": self._process_id}
                encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
                    event = {**event, "attributes": {"attributes_omitted": True}}
                    encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                logger.log(severity, encoded)
        except Exception:
            self._retry_at = time.monotonic() + 1
            self._unavailable()

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
        self._community = None

    def __enter__(self):
        inherited = current_context()
        self._community = self._parent._community if self._parent else current_consent()
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
            payload["community"] = self._community
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
