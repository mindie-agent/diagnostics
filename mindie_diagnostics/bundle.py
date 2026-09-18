"""Offline, bounded public projection of diagnostics events, never raw state."""
from __future__ import annotations

from datetime import datetime
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import sys

from .redact import REDACTION_PROFILE, redact_text, scan_text

SCHEMA = 1
MAX_BYTES = 1_048_576
MAX_INPUT_BYTES = 8_388_608
MAX_FILES = 128
MAX_EVENTS = 1000
MAX_LINE = 16_384
_ID = re.compile(r"[0-9a-f]{32}\Z")
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,119}\Z")
_LOG = re.compile(r"[0-9]+-[0-9a-f]{32}\.jsonl(?:\.[1-3])?\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_NUMERIC = frozenset({"duration_ms", "elapsed_ms", "input_bytes", "output_bytes", "bytes", "count",
                      "attempt", "exit_code", "file_count", "dropped_count", "timeout_seconds", "omitted_bytes"})
_BOOLEAN = frozenset({"retryable", "quiet", "resources_released", "cache_hit", "ready", "timed_out",
                      "attributes_omitted"})
_LABELS = frozenset({"category", "submission_state", "error_type", "error_code", "tool", "stage", "status", "reason"})
_HASHES = frozenset({"execution_id", "job_id", "source_sha256", "content_sha256"})
_TOP = frozenset({"schema", "timestamp", "monotonic_ns", "pid", "component", "severity", "event",
                  "operation_id", "trace_id", "parent_operation_id", "operation", "status",
                  "duration_ms", "attributes", "phase_id", "phase", "parent_phase_id", "process_ref",
                  "process_instance_id", "package_version", "package_revision", "clock_domain_unknown"})


def _canonical(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _label(value):
    # No free-form strings (message, stdout, path, command, environment, etc.).
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        return None
    result = redact_text(value)
    return result if result == value and not scan_text(result) else None


def _number(value):
    return (type(value) in (int, float) and math.isfinite(value) and abs(value) < 10**30)


def _project(record):
    if not isinstance(record, dict) or type(record.get("schema")) is not int or record["schema"] != SCHEMA:
        return None, 0
    required = ("component", "event")
    if any(_label(record.get(key)) is None for key in required):
        return None, 0
    if record.get("severity") not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        return None, 0
    stamp = record.get("timestamp")
    if not isinstance(stamp, str) or len(stamp) > 32 or not stamp.endswith("Z"):
        return None, 0
    try:
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None, 0
    if type(record.get("monotonic_ns")) is not int or not 0 <= record["monotonic_ns"] < 10**30:
        return None, 0
    if not isinstance(record.get("operation_id"), str) or not _ID.fullmatch(record["operation_id"]):
        return None, 0
    if not isinstance(record.get("trace_id"), str) or not _ID.fullmatch(record["trace_id"]):
        return None, 0
    result = {key: record[key] for key in ("schema", "timestamp", "monotonic_ns", "component",
                                          "severity", "event", "operation_id", "trace_id")}
    omitted = len(set(record) - _TOP)
    for key in ("parent_operation_id", "phase_id", "parent_phase_id"):
        if record.get(key) is None:
            continue
        if isinstance(record[key], str) and _ID.fullmatch(record[key]):
            result[key] = record[key]
        else:
            omitted += 1
    for key in ("operation", "phase"):
        if key in record:
            label = _label(record[key])
            if label is not None:
                result[key] = label
            else:
                omitted += 1
    if record.get("status") in {"pending", "running", "success", "error", "cancelled"}:
        result["status"] = record["status"]
    if _number(record.get("duration_ms")) and record["duration_ms"] >= 0:
        result["duration_ms"] = record["duration_ms"]
    # Do not export machine PID or paths. Process-local clock needs its own key.
    if isinstance(record.get("process_instance_id"), str) and _ID.fullmatch(record["process_instance_id"]):
        result["process_instance_id"] = record["process_instance_id"]
        result["process_ref"] = record["process_instance_id"]
    elif isinstance(record.get("process_ref"), str) and _ID.fullmatch(record["process_ref"]):
        result["process_instance_id"] = record["process_ref"]
        result["process_ref"] = record["process_ref"]
    elif type(record.get("pid")) is int and record["pid"] > 0:
        result["process_ref"] = hashlib.sha256((record["component"] + ":" + str(record["pid"])).encode()).hexdigest()[:16]
        result["clock_domain_unknown"] = True
    elif isinstance(record.get("process_ref"), str) and re.fullmatch(r"[0-9a-f]{16}", record["process_ref"]):
        result["process_ref"] = record["process_ref"]
        result["clock_domain_unknown"] = True
    else:
        return None, 0
    attributes = record.get("attributes", {})
    if not isinstance(attributes, dict):
        return None, 0
    safe = {}
    version = record.get("package_version")
    if isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,79}", version) and not scan_text(version):
        result["package_version"] = version
    revision = record.get("package_revision")
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{7,64}", revision):
        result["package_revision"] = revision
    omitted += max(0, len(attributes) - 64)
    for key, value in islice(attributes.items(), 64):
        if key in _NUMERIC and _number(value):
            safe[key] = value
        elif key in _BOOLEAN and isinstance(value, bool):
            safe[key] = value
        elif key == "error_code" and type(value) is int and -(2**31) <= value < 2**31:
            safe[key] = value
        elif key == "classification" and isinstance(value, str) and value in {"caller", "cancelled", "unknown"}:
            safe[key] = value
        elif key in _LABELS and _label(value) is not None:
            safe[key] = _label(value)
        elif key in _HASHES and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32,64}", value):
            safe[key] = value
        elif key == "stack_fingerprint" and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            safe[key] = value
        elif key == "exception_chain" and isinstance(value, list):
            safe[key] = [name for name in value[:8] if _label(name) is not None]
        elif key == "stack_frames" and isinstance(value, list):
            frames = []
            for frame in value[:24]:
                if not isinstance(frame, dict):
                    continue
                module, function, line = frame.get("module"), frame.get("function"), frame.get("line")
                if not isinstance(module, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,159}", module):
                    continue
                package = module.split(".")[0]
                if not (package.startswith("mindie_") or package == "remote_dev" or package in sys.stdlib_module_names):
                    continue
                if not isinstance(function, str) or not re.fullmatch(r"(?:[A-Za-z_][A-Za-z0-9_]{0,99}|<module>|<lambda>)", function):
                    continue
                if type(line) is not int or not 0 < line < 10**7 or scan_text(module + " " + function):
                    continue
                frames.append({"package": package, "module": module, "function": function, "line": line})
            safe[key] = frames
        else:
            omitted += 1
    result["attributes"] = safe
    # Last mechanical check never upgrades unknown free text into a public field.
    if scan_text(_canonical(result).decode("utf-8")):
        return None, omitted
    return result, omitted


def export_public_event(record):
    """Return a strict, publishable event or None; never copy arbitrary fields."""
    try:
        return _project(record)[0]
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None


def _safe_open(path):
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("symlink diagnostics input refused")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("non-file diagnostics input refused")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    after = os.fstat(fd)
    if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(fd)
        raise ValueError("diagnostic input changed while opening")
    return os.fdopen(fd, "rb")


def _write_output(path, raw):
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("bundle output must not traverse symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, stage = tempfile.mkstemp(prefix=".diagnostics-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
    finally:
        if os.path.exists(stage):
            os.unlink(stage)


def collect_bundle(root, *, operation_id=None, output=None, max_bytes=MAX_BYTES, records=(), include_logs=True):
    """Read existing event segments only. No live probes, install or repair."""
    if type(max_bytes) is not int or not 2048 <= max_bytes <= 32 * MAX_BYTES:
        raise ValueError("max_bytes must be between 2048 and 33554432")
    if operation_id is not None and (not isinstance(operation_id, str) or not _ID.fullmatch(operation_id)):
        raise ValueError("invalid diagnostic operation identifier")
    root = Path(root).expanduser().absolute()
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ValueError("bundle root must not traverse symlinks")
    events_root = root / "events"
    if events_root.is_symlink():
        raise ValueError("event directory must not be a symlink")
    events, omitted, invalid, input_bytes, visited = [], 0, 0, 0, 0
    gaps = set()
    for number, record in enumerate(records):
        if number >= MAX_EVENTS:
            gaps.add("record_limit")
            break
        if operation_id is not None and (not isinstance(record, dict) or record.get("operation_id") != operation_id):
            continue
        try:
            public, removed = _project(record)
        except (ValueError, TypeError, OverflowError, RecursionError):
            public, removed = None, 0
        if public is None:
            invalid += 1
        else:
            events.append(public)
            omitted += removed
    candidates = []
    if include_logs and events_root.is_dir():
        with os.scandir(events_root) as directories:
            for entry in directories:
                visited += 1
                if visited > MAX_FILES * 4:
                    gaps.add("directory_limit")
                    break
                if entry.is_symlink():
                    gaps.add("symlink_omitted")
                    continue
                if not _COMPONENT.fullmatch(entry.name) or not entry.is_dir(follow_symlinks=False):
                    continue
                with os.scandir(entry.path) as files:
                    for file in files:
                        visited += 1
                        if visited > MAX_FILES * 4 or len(candidates) >= MAX_FILES:
                            gaps.add("file_limit")
                            break
                        if file.is_symlink():
                            gaps.add("symlink_omitted")
                        elif _LOG.fullmatch(file.name) and file.is_file(follow_symlinks=False):
                            candidates.append(Path(file.path))
                if len(candidates) >= MAX_FILES:
                    break
    elif include_logs:
        gaps.add("events_missing")
    budget = min(MAX_INPUT_BYTES, max_bytes * 8)
    for path in sorted(candidates):
        if input_bytes >= budget or len(events) >= MAX_EVENTS:
            gaps.add("input_limit")
            break
        try:
            with _safe_open(path) as stream:
                # Prefer the tail of each segment and discard a partial leading line.
                size = os.fstat(stream.fileno()).st_size
                available = min(size, budget - input_bytes)
                if size > available:
                    stream.seek(size - available)
                    gaps.add("input_clipped")
                    stream.readline(MAX_LINE + 1)
                while input_bytes < budget and len(events) < MAX_EVENTS:
                    raw = stream.readline(min(MAX_LINE + 1, budget - input_bytes))
                    if not raw:
                        break
                    input_bytes += len(raw)
                    if not raw.endswith(b"\n"):
                        invalid += 1
                        if len(raw) > MAX_LINE:
                            gaps.add("oversized_event")
                            # Do not parse a tail fragment as a new event.
                            while raw and not raw.endswith(b"\n") and input_bytes < budget:
                                raw = stream.readline(min(MAX_LINE + 1, budget - input_bytes))
                                input_bytes += len(raw)
                        continue
                    try:
                        record = json.loads(raw)
                        if operation_id is not None and (not isinstance(record, dict) or record.get("operation_id") != operation_id):
                            continue
                        public, removed = _project(record)
                        if public is None:
                            invalid += 1
                        else:
                            events.append(public)
                            omitted += removed
                    except (ValueError, TypeError, OverflowError, RecursionError):
                        invalid += 1
        except (OSError, ValueError):
            gaps.add("unreadable_event_file")
    def payload():
        return {"schema": SCHEMA, "events": events,
                "summary": {"components": sorted({item["component"] for item in events}),
                            "operation_ids": sorted({item["operation_id"] for item in events}),
                            "error_count": sum(item["severity"] in {"ERROR", "CRITICAL"} for item in events),
                            "event_count": len(events)},
                "redaction": {"profile": REDACTION_PROFILE, "projection": 1, "omitted_fields": omitted,
                              "invalid_events": invalid, "gaps": sorted(gaps)}}
    result = payload()
    # Reserve room for hashes; clipping is explicit and never emits half JSON.
    while len(_canonical(result)) > max_bytes - 200 and events:
        events.pop(0)
        gaps.add("output_clipped")
        result = payload()
    digest = hashlib.sha256(_canonical(result)).hexdigest()
    result.update(bundle_id=digest, content_sha256=digest)
    encoded = _canonical(result) + b"\n"
    if len(encoded) > max_bytes:
        raise ValueError("bundle metadata exceeds budget")
    if output is not None:
        destination = Path(output).expanduser().absolute()
        if destination == root or destination.is_relative_to(events_root):
            raise ValueError("bundle output must not overwrite event evidence")
        _write_output(destination, encoded)
    return result
