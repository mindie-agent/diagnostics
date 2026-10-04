"""Offline event ingestion and separate, conservative GitHub publication."""
from __future__ import annotations

import hashlib
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
from pathlib import Path
from typing import Any

from .outbox import Outbox
from .reporting import ConsentUnavailable, ConsentWithdrawn, check_remote_action, guard_consent, require_consent

DEFAULT_REPOSITORY = "mindie-agent/mindie-agent"
MARKER = "<!-- mindie-incident:"


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
    if not command or shutil.which(command[0]) is None:
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
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
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
                )
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
                # The leader can exit while descendants still own its pipes.
                # Reap the entire owned group, including that case.
                until = time.monotonic() + 0.2
                while time.monotonic() < until:
                    try:
                        os.killpg(proc.pid, 0)
                    except OSError:
                        break
                    time.sleep(0.01)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                cleanup_failed = True
        except Exception:
            cleanup_failed = True
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    cleanup_failed = True
    for thread in threads:
        thread.join(0.2)
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
    import json
    import re
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

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


def fingerprint(bundle: dict[str, Any]) -> str:
    """Public fault identity excludes consent, incident IDs and host wrappers."""
    from .bundle import export_public_event
    events = bundle.get("events", [])
    failures = [safe for event in events if (safe := export_public_event(event))
                is not None and safe.get("status") == "error"]
    event = failures[-1] if failures else {}
    attrs = event.get("attributes", {})
    component = event.get("component", "unknown")
    module = component.replace("-", "_")
    frames = [frame for frame in attrs.get("stack_frames", [])
              if frame.get("module") == module or frame.get("module", "").startswith(module + ".")]
    identity = {"component": component, "package_version": event.get("package_version"),
                "package_revision": event.get("package_revision"), "operation": event.get("operation"),
                "stage": attrs.get("stage"), "category": attrs.get("category"),
                "error_type": attrs.get("error_type"), "frames": frames}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def issue_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    """Re-project even a caller supplied bundle; never trust a 'redacted' flag."""
    from .bundle import export_public_event
    from .redact import scan_text

    events = [safe for record in bundle.get("events", []) if (safe := export_public_event(record)) is not None]
    if not events or not any(e.get("status") == "error" for e in events):
        raise ValueError("no publishable failure events")
    public = {"schema": "mindie.support.issue.v1", "events": events[-80:],
              "omissions": "Raw output, commands, prompts, environment values, local paths and unknown fields are excluded. Evidence is a bounded window; absent phases may be outside it."}
    # Fit the complete evidence into the issue itself; no expiring attachment URL.
    while len(json.dumps(public, ensure_ascii=True).encode()) > 36000 and len(public["events"]) > 1:
        public["events"].pop(0)
    if not any(event.get('status') == 'error' for event in public['events']):
        raise ValueError('failure outside bounded evidence window')
    public["content_sha256"] = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    if scan_text(json.dumps(public, ensure_ascii=False)):
        raise ValueError("final diagnostic leak scan rejected publication")
    return public


def render_issue(item: dict[str, Any]) -> tuple[str, str]:
    payload = issue_payload(item["payload"])
    failure = next(e for e in reversed(payload["events"]) if e.get("status") == "error")
    component = failure.get("component", "mindie")
    operation = failure.get("operation", "operation")
    title = f"[automatic diagnostic] {component}: {operation} failed"[:200]
    body = (f"{MARKER}{fingerprint(payload)} -->\n\n"
            "MindIE recorded a failed operation. This issue was generated automatically from selected, redacted structured events.\n\n"
            f"Occurrences observed locally before submission: {item['occurrences']}. "
            "The events record observed component failures; they do not establish a root cause.\n\n"
            "<details><summary>Sanitized diagnostic evidence (JSON)</summary>\n\n```json\n"
            + json.dumps(payload, ensure_ascii=True, indent=2) + "\n```\n</details>\n")
    if len(body.encode()) > 60000:
        raise ValueError("issue body exceeds publication limit")
    return title, body


class GitHub:
    """One bounded gh transport using the user's existing authentication."""
    def __init__(self, repository: str = DEFAULT_REPOSITORY, *, executable: str = "gh", timeout: float = 30, cancel=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("invalid GitHub repository")
        if not 0 < timeout <= 300:
            raise ValueError("invalid GitHub cycle timeout")
        self.repository, self.executable, self.timeout = repository, executable, timeout
        self.cancel, self._deadline, self._bytes = cancel, None, 0

    def begin_cycle(self):
        if self._deadline is not None:
            raise RuntimeError("GitHub cycle already active")
        self._deadline, self._bytes = time.monotonic() + self.timeout, 0

    def end_cycle(self):
        self._deadline = None

    def request(self, method: str, path: str, payload: dict | None = None):
        check_remote_action()
        if method not in {"GET", "POST"} or not isinstance(path, str) or not path.startswith(f"repos/{self.repository}/") or any(char in path for char in "\r\n#%") or ".." in path:
            raise TransportError("github_invalid_path", permanent=True)
        standalone = self._deadline is None
        if standalone:
            self.begin_cycle()
        try:
            command = [self.executable, "api", "--include", "--hostname", "github.com", "--method", method,
                       "-H", "Accept: application/vnd.github+json", path]
            data = None
            if payload is not None:
                command += ["--input", "-"]
                data = json.dumps(payload, ensure_ascii=True).encode()
            try:
                code, raw, size = _run_gh(command, data, deadline=self._deadline,
                                          max_bytes=8 * 1024 * 1024 - self._bytes, cancel=self.cancel)
            except TransportError as exc:
                exc.uncertain = method == "POST" and exc.started
                raise
            self._bytes += size
            return _parse_response(raw, code, method)
        finally:
            if standalone:
                self.end_cycle()

    def find_issue(self, item: dict[str, Any]):
        # Listing all states avoids search-index lag; a full window is not absence.
        marker = f"{MARKER}{fingerprint(item['payload'])} -->"
        standalone = self._deadline is None
        if standalone:
            self.begin_cycle()
        try:
            for page in range(1, 21):
                rows = self.request("GET", f"repos/{self.repository}/issues?state=all&sort=updated&direction=desc&per_page=100&page={page}")
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise TransportError("github_invalid_issue_listing")
                for row in rows:
                    body = row.get("body") or ""
                    if not isinstance(body, str):
                        raise TransportError("github_invalid_issue_listing")
                    if "pull_request" not in row and marker in body:
                        return row
                if len(rows) < 100:
                    return None
            raise TransportError("github_reconciliation_window_exhausted", retry_after=3600)
        finally:
            if standalone:
                self.end_cycle()

    def create_issue(self, title: str, body: str):
        return self.request("POST", f"repos/{self.repository}/issues", {"title": title, "body": body})


def _issue_reference(reply, repository):
    if not isinstance(reply, dict) or type(reply.get("number")) is not int or reply["number"] <= 0:
        return False
    return reply.get("html_url") == f"https://github.com/{repository}/issues/{reply['number']}"


def _record_publication_outcome(queue, item, result, **updates):
    """Local receipt failure must not erase an acknowledged external result."""
    try:
        queue.update(item, **updates)
    except Exception as exc:
        return {**result, "publication_status": result["status"], "status": "recording_failed",
                "local_recording": {"status": "failed", "error_type": type(exc).__name__},
                "error": result.get("error") or "publication_receipt_failed"}
    return result


def publish_one(queue: Outbox, github: GitHub) -> dict[str, Any]:
    from .outbox import MAX_AUTOMATIC_CYCLES
    item = queue.claim(lease_seconds=github.timeout + 5)
    if not item:
        return {"status": "idle"}
    exhausted = item["attempts"] >= MAX_AUTOMATIC_CYCLES
    started = False
    try:
        require_consent(item.get("consent"))
        if item["consent"].get("repository") != github.repository:
            raise TransportError("github_repository_consent_mismatch", permanent=True)
        github.begin_cycle()
        started = True
        with guard_consent(item.get("consent")):
            existing = github.find_issue(item)
        require_consent(item.get("consent"))
        if existing:
            if not _issue_reference(existing, github.repository):
                raise TransportError("github_invalid_issue_reference")
            return _record_publication_outcome(queue, item,
                {"status": "reconciled", "issue_url": existing["html_url"], "operation_completed": True},
                state="published", issue_number=existing["number"], issue_url=existing["html_url"], last_error=None)
        if item["state"] == "uncertain":
            state = "exhausted" if exhausted else "uncertain"
            return _record_publication_outcome(queue, item, {"status": state, "submission_state": "uncertain"},
                state=state, next_attempt=queue.clock() + 300,
                last_error="submission_uncertain_budget_exhausted" if exhausted else "submission_uncertain_no_match")
        title, body = render_issue(item)
        require_consent(item.get("consent"))
        queue.begin_post(item)
        item["state"] = "uncertain"
        with guard_consent(item.get("consent")):
            reply = github.create_issue(title, body)
        if not _issue_reference(reply, github.repository):
            raise TransportError("github_invalid_create_reply", uncertain=True)
        return _record_publication_outcome(queue, item,
            {"status": "published", "issue_url": reply["html_url"], "operation_completed": True},
            state="published", issue_number=reply["number"], issue_url=reply["html_url"], last_error=None)
    except ConsentWithdrawn:
        uncertain = item["state"] == "uncertain"
        return _record_publication_outcome(queue, item,
            {"status": "withdrawn", "submission_state": "uncertain" if uncertain else "not_sent"},
            state="withdrawn", last_error="submission_uncertain_reporting_consent_withdrawn" if uncertain else "reporting_consent_unavailable_or_withdrawn")
    except ConsentUnavailable:
        uncertain = item["state"] == "uncertain"
        state = "exhausted" if exhausted else "uncertain" if uncertain else "retry"
        code = ("submission_uncertain_" if uncertain else "") + "reporting_policy_unavailable"
        return _record_publication_outcome(queue, item,
            {"status": state, "error": code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + 300, last_error=code)
    except TransportError as exc:
        uncertain = exc.uncertain or item["state"] == "uncertain"
        state = "permanent-failed" if exc.permanent else "exhausted" if exhausted else "uncertain" if uncertain else "retry"
        code = "submission_uncertain_" + exc.code if uncertain and state in {"permanent-failed", "exhausted"} else exc.code
        delay = max(exc.retry_after, 30 * 2 ** item["attempts"])
        return _record_publication_outcome(queue, item,
            {"status": state, "error": code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + delay, last_error=code)
    except (ValueError, TypeError, KeyError):
        uncertain = item["state"] == "uncertain"
        state = "exhausted" if uncertain and exhausted else "uncertain" if uncertain else "blocked"
        code = "submission_uncertain_invalid_reply" if uncertain else "invalid_or_unsafe_diagnostic_payload"
        return _record_publication_outcome(queue, item,
            {"status": state, "error": code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + 300, last_error=code)
    finally:
        if started:
            github.end_cycle()


def ingest(root: str | Path, queue: Outbox, *, max_files: int = 256, max_bytes: int = 8 * 1024 * 1024, since: float | None = None) -> dict[str, int]:
    from .ingestion import ingest as read_events
    return read_events(root, queue, max_files=max_files, max_bytes=max_bytes, since=since)
