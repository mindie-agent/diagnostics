"""Bounded `gh api --include` subprocess transport and HTTP response parser.

[Role]
Executes isolated `gh api --include` child processes (`_run_gh`), enforces wall-clock
deadlines and combined stdout/stderr byte budgets, and parses HTTP status/headers/JSON
bodies (`_parse_response`) into typed `TransportError` classifications.

[Invariants]
- Never passes tokens on argv or falls back to raw token HTTP clients; relies solely
  on the user's existing `gh` authentication.
- Stderr bytes are counted against `max_bytes` and discarded so private CLI diagnostics
  never enter exception messages or logs.
- `except TimeoutError: raise` precedes `except OSError:` across process-group cleanup.

[Timeouts & Locks]
- Spawns `gh` in an isolated process group (`start_new_session=True` on POSIX,
  `CREATE_NEW_PROCESS_GROUP` on Windows) and reaps the entire group on timeout or
  cancellation (`SIGTERM` -> 2.0s grace -> `SIGKILL`).

[Side-Effect Policy]
- Spawns bounded `gh` subprocesses with temporary stdin files; performs no local state
  or SQLite mutations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time

from .filesystem import which_native


class TransportError(RuntimeError):
    def __init__(
        self, code: str, *, uncertain: bool = False, retry_after: float = 60, permanent: bool = False
    ):
        super().__init__(code)
        self.started = False
        self.code, self.uncertain, self.retry_after, self.permanent = (
            code, uncertain, retry_after, permanent,
        )


def _run_gh(command, data, *, deadline, max_bytes, cancel=None):
    """Run one gh argv. Return (returncode, stdout, total_read) or TransportError.

    Does not know HTTP method: every code uses uncertain=False. Stderr is counted
    and discarded. POSIX process-group kill is owned; Windows taskkill is unverified.
    """
    def _cancelled():
        if cancel is None:
            return False
        is_set = getattr(cancel, "is_set", None)
        return bool(is_set()) if callable(is_set) else bool(cancel)

    if _cancelled():
        raise TransportError("github_cancelled", uncertain=False)
    if time.monotonic() >= deadline:
        raise TransportError("github_timeout", uncertain=False)
    if not command or which_native(command[0], which_fn=shutil.which) is None:
        raise TransportError("github_client_missing", uncertain=False, permanent=True)
    payload = b"" if data is None else bytes(data)
    if len(payload) > 65536:
        raise TransportError("github_output_limit", uncertain=False, permanent=True)

    proc = None
    threads = []
    stop = threading.Event()
    sink = queue.Queue(maxsize=64)
    error = None
    result = None

    def _reader(stream, kind):
        try:
            while not stop.is_set():
                block = stream.read(4096)
                item = (kind, None if not block else block)
                while not stop.is_set():
                    try:
                        sink.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                else:
                    return
                if item[1] is None:
                    return
        except Exception:
            try:
                sink.put((kind, None), timeout=0.05)
            except Exception:
                return

    try:
        with tempfile.TemporaryFile() as stdin_fp:
            if payload:
                stdin_fp.write(payload)
                stdin_fp.seek(0)
            popen_kwargs = {"stdin": stdin_fp, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "bufsize": 0}
            if os.name == "nt":
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
            else:
                popen_kwargs["start_new_session"] = True
            try:
                proc = subprocess.Popen(command, **popen_kwargs)
            except (FileNotFoundError, PermissionError):
                error = TransportError("github_client_missing", uncertain=False, permanent=True)
            else:
                for stream, kind in ((proc.stdout, "o"), (proc.stderr, "e")):
                    thread = threading.Thread(target=_reader, args=(stream, kind), daemon=True)
                    thread.start()
                    threads.append(thread)
                stdout_parts = []
                total = 0
                eof = 0
                while eof < 2 and error is None:
                    if _cancelled():
                        error = TransportError("github_cancelled", uncertain=False)
                        break
                    if time.monotonic() >= deadline:
                        error = TransportError("github_timeout", uncertain=False)
                        break
                    tick_end = time.monotonic() + 0.05
                    while error is None:
                        try:
                            kind, block = sink.get_nowait()
                        except queue.Empty:
                            break
                        if block is None:
                            eof += 1
                            continue
                        if total + len(block) > max_bytes:
                            error = TransportError("github_output_limit", uncertain=False)
                            break
                        total += len(block)
                        if kind == "o":
                            stdout_parts.append(block)
                    if error is None and eof < 2:
                        pause = tick_end - time.monotonic()
                        if pause > 0:
                            time.sleep(pause)
                while error is None:
                    if _cancelled():
                        error = TransportError("github_cancelled", uncertain=False)
                        break
                    now = time.monotonic()
                    if now >= deadline:
                        error = TransportError("github_timeout", uncertain=False)
                        break
                    try:
                        rc = proc.wait(timeout=min(0.05, max(0.0, deadline - now)))
                    except subprocess.TimeoutExpired:
                        continue
                    result = (rc, b"".join(stdout_parts), total)
                    break
    except TransportError as exc:
        error = exc
    except Exception:
        error = TransportError("github_transport_error", uncertain=False)
    except BaseException as exc:
        error = exc

    cleanup_failed = False
    stop.set()
    if proc is not None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    timeout=1, capture_output=True, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except TimeoutError:
                    raise
                except OSError:
                    pass
                until = time.monotonic() + 2.0
                while time.monotonic() < until:
                    try:
                        os.killpg(proc.pid, 0)
                    except TimeoutError:
                        raise
                    except OSError:
                        break
                    time.sleep(0.01)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except TimeoutError:
                    raise
                except OSError:
                    pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                cleanup_failed = True
        except Exception:
            cleanup_failed = True
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except TimeoutError:
                    raise
                except OSError:
                    cleanup_failed = True
    for thread in threads:
        thread.join(2.0)
        if thread.is_alive():
            cleanup_failed = True
    if error is not None:
        error.started = proc is not None
        raise error
    if cleanup_failed:
        error = TransportError("github_cleanup_failed")
        error.started = proc is not None
        raise error
    return result


def _parse_response(raw: bytes, returncode: int, method: str):
    """Classify `gh api --include` stdout. JSON on 2xx and rc 0; else TransportError."""
    def invalid_headers():
        raise TransportError("github_invalid_headers", uncertain=method != "GET")

    def transport():
        raise TransportError("github_transport_error", uncertain=method == "POST")

    if not isinstance(raw, (bytes, bytearray)):
        invalid_headers()
    if len(raw) > 8 * 1024 * 1024:
        raise TransportError("github_oversized", uncertain=method == "POST")

    data = bytes(raw)
    status_re = re.compile(br"HTTP/(?:1\.1|2|2\.0) ([1-5]\d{2})(?:[\t ].*)?\Z")
    header_re = re.compile(br"([!#$%&'*+\-.^_`|~0-9A-Za-z]+):(.*)\Z")
    pos, blocks, budget, final = 0, 0, 0, None

    def read_line(p):
        nl = data.find(b"\n", p)
        if nl < 0:
            return data[p:], len(data)
        return data[p:nl], nl + 1

    while pos < len(data):
        line, nxt = read_line(pos)
        if line.endswith(b"\r"):
            line = line[:-1]
        matched = status_re.match(line)
        if matched is None:
            if line.startswith(b"HTTP/"):
                invalid_headers()
            break
        if blocks >= 16:
            invalid_headers()
        blocks += 1
        budget += nxt - pos
        if budget > 64 * 1024:
            invalid_headers()
        status, headers, pos = int(matched.group(1)), {}, nxt
        while True:
            if pos >= len(data):
                invalid_headers()
            line, nxt = read_line(pos)
            budget += nxt - pos
            if budget > 64 * 1024:
                invalid_headers()
            if line.endswith(b"\r"):
                line = line[:-1]
            pos = nxt
            if line == b"":
                break
            hm = header_re.match(line)
            if hm is None:
                invalid_headers()
            headers[hm.group(1).decode("ascii").casefold()] = hm.group(2).decode("latin-1").strip()
        final = (status, headers, pos)
        if pos >= len(data):
            break
        peek, _ = read_line(pos)
        if peek.endswith(b"\r"):
            peek = peek[:-1]
        if not peek.startswith(b"HTTP/"):
            break

    if final is None:
        if returncode == 4:
            raise TransportError("github_auth_required", permanent=True)
        if returncode != 0:
            transport()
        invalid_headers()

    status, headers, body_at = final
    if status < 200:
        invalid_headers()

    def retry_after() -> float:
        raw_h = headers.get("retry-after")
        if raw_h is None:
            return 60.0
        try:
            seconds = int(raw_h)
        except ValueError:
            try:
                when = parsedate_to_datetime(raw_h)
            except (TypeError, ValueError, IndexError, OverflowError):
                return 60.0
            if when is None:
                return 60.0
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = int((when - datetime.now(timezone.utc)).total_seconds())
        return float(min(3600, max(0, seconds)))

    if status >= 400:
        limited = status == 429 or (
            status == 403 and ("retry-after" in headers or headers.get("x-ratelimit-remaining") == "0")
        )
        if limited:
            raise TransportError(f"github_http_{status}", retry_after=retry_after())
        if status < 500:
            raise TransportError(f"github_http_{status}", permanent=True, uncertain=False)
        raise TransportError(f"github_http_{status}", uncertain=method == "POST")
    if status >= 300:
        raise TransportError(f"github_http_{status}", permanent=True)
    if returncode != 0:
        transport()
    try:
        return json.loads(data[body_at:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise TransportError("github_invalid_reply", uncertain=method == "POST")
