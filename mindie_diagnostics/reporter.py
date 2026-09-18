"""Offline event ingestion and separate, conservative GitHub publication."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .outbox import Outbox
from .community import ConsentWithdrawn, check_remote_action, guard_consent, require_consent

DEFAULT_REPOSITORY = "mindie-agent/mindie-agent"
MARKER = "<!-- mindie-incident:"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        # A token is scoped to api.github.com. Never forward it to a redirect.
        return None


class TransportError(RuntimeError):
    def __init__(self, code: str, *, uncertain: bool = False, retry_after: float = 60):
        super().__init__(code)
        self.code, self.uncertain, self.retry_after = code, uncertain, retry_after


def fingerprint(bundle: dict[str, Any]) -> str:
    events = bundle.get("events", [])
    failures = [event for event in events if event.get("status") == "error"]
    keys = []
    for event in failures[-8:]:
        attrs = event.get("attributes", {})
        keys.append({"component": event.get("component"), "operation": event.get("operation"),
                     "event": event.get("event"), **{key: attrs.get(key) for key in
                     ("category", "error_type", "error_code", "phase", "version", "submission_state", "stack_fingerprint")},
                     "phase": event.get("phase"), "package_version": event.get("package_version"),
                     "package_revision": event.get("package_revision")})
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()


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
    body = (f"{MARKER}{item['fingerprint']} -->\n\n"
            "MindIE recorded a failed operation. This issue was generated automatically from selected, redacted structured events.\n\n"
            f"Occurrences observed locally before submission: {item['occurrences']}. "
            "A diagnostic hypothesis is not a confirmed root cause.\n\n"
            "<details><summary>Sanitized diagnostic evidence (JSON)</summary>\n\n```json\n"
            + json.dumps(payload, ensure_ascii=True, indent=2) + "\n```\n</details>\n")
    if len(body.encode()) > 60000:
        raise ValueError("issue body exceeds publication limit")
    return title, body


class GitHub:
    """Existing gh authentication or an environment token without gh installed."""
    def __init__(self, repository: str = DEFAULT_REPOSITORY, *, executable: str = "gh", timeout: float = 30):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("invalid GitHub repository")
        self.repository, self.executable, self.timeout = repository, executable, timeout

    def request(self, method: str, path: str, payload: dict | None = None):
        check_remote_action()
        token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
        if token and shutil.which(self.executable) is None:
            return self._token_request(method, path, payload, token)
        command = [self.executable, "api", "--hostname", "github.com", "--method", method,
                   "-H", "Accept: application/vnd.github+json", path]
        data = None
        if payload is not None:
            command += ["--input", "-"]
            data = json.dumps(payload, ensure_ascii=True)
        try:
            result = subprocess.run(command, input=data, capture_output=True, text=True,
                                    encoding="utf-8", timeout=self.timeout, check=False)
        except FileNotFoundError as exc:
            raise TransportError("github_client_missing") from exc
        except subprocess.TimeoutExpired as exc:
            raise TransportError("github_timeout", uncertain=method != "GET") from exc
        if result.returncode:
            # Do not save stderr: gh may echo request contents or credential paths.
            status = re.search(r"HTTP (\d{3})", result.stderr)
            code = int(status.group(1)) if status else None
            rejected = code is not None and 400 <= code < 500
            raise TransportError(f"github_http_{code}" if code else "github_transport_error",
                                 uncertain=method != "GET" and not rejected,
                                 retry_after=3600 if code in (403, 429) else 60)
        try:
            return json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise TransportError("github_invalid_reply", uncertain=method != "GET") from exc

    def _token_request(self, method, path, payload, token):
        """Fixed-host HTTPS; credentials never enter command lines or errors."""
        if not isinstance(path, str) or not path.startswith('repos/') or any(char in path for char in '\r\n#'):
            raise TransportError('github_invalid_path')
        data = json.dumps(payload, ensure_ascii=True).encode() if payload is not None else None
        request = Request('https://api.github.com/' + path, data=data, method=method,
                          headers={'Accept': 'application/vnd.github+json', 'Authorization': 'Bearer ' + token,
                                   'Content-Type': 'application/json', 'User-Agent': 'mindie-diagnostics'})
        try:
            check_remote_action()
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise TransportError('github_oversized_reply', uncertain=method != 'GET')
            return json.loads(raw)
        except HTTPError as exc:
            code = exc.code
            raise TransportError(f'github_http_{code}', uncertain=method != 'GET' and not 400 <= code < 500,
                                 retry_after=3600 if code in (403, 429) else 60) from None
        except (URLError, TimeoutError, OSError) as exc:
            raise TransportError('github_transport_error', uncertain=method != 'GET') from None
        except (ValueError, TypeError):
            raise TransportError('github_invalid_reply', uncertain=method != 'GET') from None

    def find_issue(self, item: dict[str, Any]):
        # Direct paginated REST listing avoids search-index eventual consistency.
        marker = f"{MARKER}{item['fingerprint']} -->"
        for page in range(1, 21):
            rows = self.request("GET", f"repos/{self.repository}/issues?state=all&sort=updated&direction=desc&per_page=100&page={page}")
            if not isinstance(rows, list):
                raise TransportError("github_invalid_issue_listing")
            for row in rows:
                if "pull_request" not in row and marker in (row.get("body") or ""):
                    return row
            if len(rows) < 100:
                return None
        raise TransportError("github_reconciliation_window_exhausted", retry_after=3600)

    def create_issue(self, title: str, body: str):
        return self.request("POST", f"repos/{self.repository}/issues", {"title": title, "body": body})


def publish_one(queue: Outbox, github: GitHub) -> dict[str, Any]:
    # Cover twenty bounded reconciliation pages, publication and local work.
    item = queue.claim(lease_seconds=21 * getattr(github, 'timeout', 30) + 60)
    if not item:
        return {"status": "idle"}
    try:
        with guard_consent(item.get('consent')):
            existing = github.find_issue(item)
        require_consent(item.get('consent'))
        if existing:
            queue.update(item, state="published", issue_number=existing["number"], issue_url=existing["html_url"], last_error=None)
            return {"status": "reconciled", "issue_url": existing["html_url"]}
        if item["state"] == "uncertain":
            queue.update(item, state="uncertain", next_attempt=queue.clock() + 300, last_error="submission_uncertain_no_match; automatic repost withheld")
            return {"status": "uncertain"}
        title, body = render_issue(item)
        if not queue.begin_post(item):
            queue.update(item, state="retry", next_attempt=queue.clock() + 3600, last_error="local_hourly_rate_limit")
            return {"status": "rate_limited"}
        with guard_consent(item.get('consent')):
            reply = github.create_issue(title, body)
        if not isinstance(reply, dict) or not isinstance(reply.get("number"), int) or not isinstance(reply.get("html_url"), str):
            raise TransportError("github_invalid_create_reply", uncertain=True)
        queue.update(item, state="published", issue_number=reply["number"], issue_url=reply["html_url"], last_error=None)
        return {"status": "published", "issue_url": reply["html_url"]}
    except ConsentWithdrawn:
        queue.update(item, state='withdrawn', diagnosis_state='withdrawn',
                     last_error='community_consent_unavailable_or_withdrawn')
        return {'status': 'withdrawn'}
    except TransportError as exc:
        uncertain = exc.uncertain or item["state"] == "uncertain"
        delay = max(exc.retry_after, min(3600, 30 * 2 ** min(item["attempts"], 7)))
        queue.update(item, state="uncertain" if uncertain else "retry", next_attempt=queue.clock() + delay, last_error=exc.code)
        return {"status": "uncertain" if uncertain else "retry", "error": exc.code}
    except (ValueError, TypeError, KeyError):
        queue.update(item, state="blocked", last_error="invalid_or_unsafe_diagnostic_payload")
        return {"status": "blocked", "error": "invalid_or_unsafe_diagnostic_payload"}


def ingest(root: str | Path, queue: Outbox, *, max_files: int = 256, max_bytes: int = 8 * 1024 * 1024, since: float | None = None) -> dict[str, int]:
    from .ingestion import ingest as read_events
    return read_events(root, queue, max_files=max_files, max_bytes=max_bytes, since=since)
