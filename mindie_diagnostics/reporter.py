"""Conservative GitHub issue reconciliation and publication for `Outbox` items.

[Role]
Renders sanitized diagnostic issue payloads (`issue_payload`, `render_issue`),
communicates with GitHub via `gh api --include` (`GitHub`), reconciles existing
issue markers (`<!-- mindie-incident:<sha256> -->`), and publishes via `publish_one`.

[Invariants]
- Re-projects and re-scans every bundle via `export_issue_event` and `scan_text`
  before rendering; never trusts a caller-supplied redaction flag.
- Search hits confirm markers; misses require a complete, time-bounded issue
  listing. Incomplete windows never authorize a POST or replay unknown writes.
- Calls `queue.begin_post(item)` BEFORE `POST repos/<repo>/issues` so any process
  death or transport failure after process start records `state="uncertain"`.
- Never overwrites an acknowledged external publication (`published` / `reconciled`)
  as a remote failure if local SQLite receipt update fails (`_record_publication_outcome`).

[Timeouts & Locks]
- Leases `Outbox` items for `github.timeout + 20` seconds (`default 50s`),
  covering transport cleanup and the local receipt.
- `GitHub.begin_cycle` shares a deadline (default 30s) and 8 MiB response budget
  across one search page, up to five listing pages, and POST (reserving up to 15s).
- `_run_gh` spawns `gh` in an isolated process group (`start_new_session=True` /
  `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`) and reaps the entire group on
  timeout/cancel (`SIGTERM` -> 2.0s grace -> `SIGKILL`).

[Side-Effect Policy]
- Mutates external GitHub issues only when `reporting.json` consent is actively
  enabled and matches the item's recorded repository and revision.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from .github_transport import _CREATE_NO_WINDOW, TransportError, _parse_response, _run_gh
from .outbox import Outbox
from .reporting import ConsentUnavailable, ConsentWithdrawn, check_remote_action, guard_consent, require_consent
from .triage_card import (
    FAULT_MARKER,
    MARKER,
    build_fingerprint,
    fault_signature,
    item_public_fingerprint,
    render_occurrence_card,
    render_triage_card,
)

DEFAULT_REPOSITORY = "mindie-agent/mindie-agent"
MAX_GITHUB_RESPONSE_BYTES = 8 * 1024 * 1024
_SEARCH_PAGE_SIZE = 30
_SEARCH_MAX_PAGES = 3
_RECONCILIATION_PAGE_SIZE = 30
_RECONCILIATION_MAX_PAGES = 5
_COMMENT_PAGE_SIZE = 25
_COMMENT_SUB_PAGE_SIZE = 5
_COMMENT_MAX_PAGES = 10
_ISSUE_SCAN_PAGE_SIZE = 30
_ISSUE_SUB_PAGE_SIZE = 10
_ISSUE_SCAN_MAX_PAGES = 5
_MAX_ISSUE_BODY_BYTES = 256 * 1024
_CLOCK_SKEW_MARGIN_SECONDS = 300


class _CommentScanIncomplete(TransportError):
    def __init__(
        self,
        *,
        issue_number: int,
        issue_url: str,
        cursor: dict[str, Any],
        code: str = "github_reconciliation_window_exhausted",
        retry_after: float = 300,
    ):
        super().__init__(code, retry_after=retry_after)
        self.issue_number, self.issue_url, self.cursor = issue_number, issue_url, cursor


class _IssueScanIncomplete(TransportError):
    def __init__(
        self,
        *,
        cursor: dict[str, Any],
        code: str = "github_reconciliation_window_exhausted",
        retry_after: float = 3600,
    ):
        super().__init__(code, retry_after=retry_after)
        self.cursor = cursor


def fingerprint(bundle: dict[str, Any]) -> str:
    """Public fault identity excludes consent, incident IDs and host wrappers."""
    return build_fingerprint(bundle)


def issue_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    """Re-project even a caller supplied bundle; never trust a 'redacted' flag."""
    from .bundle import export_issue_event
    from .redact import scan_text

    events = [safe for record in bundle.get("events", []) if (safe := export_issue_event(record)) is not None]
    if not events or not any(e.get("status") == "error" for e in events):
        raise ValueError("no publishable failure events")
    public = {"schema": "mindie.support.issue.v1", "events": events[-80:],
              "omissions": "Raw output, commands, prompts, environment values, local paths, local correlation IDs, process clocks and unknown fields are excluded. Evidence is a bounded window; absent phases may be outside it."}
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
    return render_triage_card(
        payload,
        incident_fp=item_public_fingerprint({**item, "payload": payload}),
        fault_sig=fault_signature(payload),
        occurrences=int(item.get("occurrences", 1)),
    )


def render_occurrence_comment(item: dict[str, Any], *, issue_state: str | None = None) -> str:
    payload = issue_payload(item["payload"])
    return render_occurrence_card(
        payload,
        incident_fp=item_public_fingerprint({**item, "payload": payload}),
        fault_sig=fault_signature(payload),
        occurrences=int(item.get("occurrences", 1)),
        issue_state=issue_state,
    )


class GitHub:
    """One bounded gh transport using the user's existing authentication."""
    def __init__(self, repository: str = DEFAULT_REPOSITORY, *, executable: str = "gh",
                 timeout: float = 30, cancel=None,
                 max_response_bytes: int = MAX_GITHUB_RESPONSE_BYTES):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("invalid GitHub repository")
        if not 0 < timeout <= 300:
            raise ValueError("invalid GitHub cycle timeout")
        if not 0 < max_response_bytes <= MAX_GITHUB_RESPONSE_BYTES:
            raise ValueError("invalid GitHub response budget")
        self.repository, self.executable, self.timeout = repository, executable, timeout
        self.max_response_bytes = max_response_bytes
        self.cancel, self._deadline, self._bytes = cancel, None, 0
        self._searched_in_cycle = False

    def begin_cycle(self):
        if self._deadline is not None:
            raise RuntimeError("GitHub cycle already active")
        self._deadline, self._bytes = time.monotonic() + self.timeout, 0
        self._searched_in_cycle = False

    def end_cycle(self):
        self._deadline = None
        self._searched_in_cycle = False

    def request(self, method: str, path: str, payload: dict | None = None):
        check_remote_action()
        if method not in {"GET", "POST"} or not isinstance(path, str) or any(char in path for char in "\r\n#") or ".." in path:
            raise TransportError("github_invalid_path", permanent=True)
        endpoint = urlsplit(path)
        query = parse_qs(endpoint.query)
        search = (method == "GET" and endpoint.path == "search/issues"
                  and len(query.get("q", [])) == 1
                  and re.fullmatch(
                      rf'repo:{re.escape(self.repository)} is:issue in:body "[0-9a-f]{{64}}"(?: OR "[0-9a-f]{{64}}")?',
                      query["q"][0]))
        if not search and (not path.startswith(f"repos/{self.repository}/") or "%" in path):
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
                                          max_bytes=self.max_response_bytes - self._bytes, cancel=self.cancel)
            except TransportError as exc:
                exc.uncertain = method == "POST" and exc.started
                raise
            self._bytes += size
            return _parse_response(raw, code, method)
        finally:
            if standalone:
                self.end_cycle()

    def _verify_boundaries(
        self,
        endpoint_prefix: str,
        boundaries: list[tuple[int, int | None]],
        markers: str | tuple[str, ...],
        err_kind: str,
        key_field: str,
    ) -> tuple[bool, dict[str, Any] | None, tuple[TransportError, list[tuple[int, int | None]]] | None]:
        for idx, (b_pos, b_tail) in enumerate(boundaries):
            try:
                probed = self.request("GET", f"{endpoint_prefix}per_page=1&page={b_pos}")
                pmatch = _matching_issue(probed, markers, err_kind)
            except TransportError as exc:
                return False, None, (exc, boundaries[idx:])
            if pmatch is not None:
                return False, pmatch, None
            pval = probed[0].get(key_field) if probed and type(probed[0].get(key_field)) is int else None
            if b_tail is None or pval != b_tail:
                return True, None, None
        return False, None, None

    def _scan_paginated(
        self, endpoint_prefix: str, markers: str | tuple[str, ...], err_kind: str,
        key_field: str, anchor_key: str, full_size: int, sub_size: int, max_pages: int,
        cursor: dict[str, Any] | None, extra: dict[str, Any], default_retry: float, on_incomplete,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        start_page, anchor_val, per_page, boundaries = _unpack_cursor(cursor, anchor_key, full_size)
        page, last_page, last_val, last_full, pages_scanned = start_page, 1, None, False, 0
        last_full_page: int | None = None
        last_full_val: int | None = None
        checked_anchor, used = (start_page == 1 and anchor_val is None), 0
        while used < max_pages:
            used += 1
            try:
                rows = self.request("GET", f"{endpoint_prefix}per_page={per_page}&page={page}")
                match = _matching_issue(rows, markers, err_kind)
            except TransportError as exc:
                if pages_scanned > 0 and last_full:
                    next_cur = _next_overlap_cursor(
                        per_page, full_size, sub_size, start_page, last_page, anchor_key, last_val, extra, boundaries,
                    )
                    err_code = "github_reconciliation_window_exhausted" if exc.code == "github_output_limit" else exc.code
                    raise on_incomplete(next_cur, err_code, max(exc.retry_after, 3600 if exc.code == "github_output_limit" else 300)) from exc
                if isinstance(cursor, dict) and pages_scanned == 0 and exc.code == "github_output_limit" and per_page > 1:
                    eff_sub = sub_size if per_page > sub_size else 1
                    resume = ((start_page - 1) * per_page) // eff_sub + 1
                    next_cur = _pack_cursor(resume, eff_sub, full_size, anchor_key, anchor_val, extra, boundaries)
                    raise on_incomplete(next_cur, "github_reconciliation_window_exhausted", max(exc.retry_after, 3600)) from exc
                raise
            if match is not None:
                return match, {}
            if not checked_anchor:
                checked_anchor = True
                first_val = rows[0].get(key_field) if rows and type(rows[0].get(key_field)) is int else None
                if not rows or (anchor_val is not None and (first_val is None or first_val > anchor_val)):
                    boundaries.clear()
                    start_page, page, anchor_val = 1, 1, None
                    last_full_page, last_full_val = None, None
                    continue
            pages_scanned += 1
            if rows:
                last_page = page
                tval = rows[-1].get(key_field)
                last_val = tval if type(tval) is int and tval > 0 else None
                last_full = len(rows) >= per_page
            if len(rows) < per_page:
                shifted, pmatch, probe_err = self._verify_boundaries(
                    endpoint_prefix, boundaries, markers, err_kind, key_field,
                )
                if probe_err is not None:
                    exc, rem_bounds = probe_err
                    r_page = last_full_page if last_full_page is not None else start_page
                    r_val = last_full_val if last_full_page is not None else anchor_val
                    progressed = len(rem_bounds) < len(boundaries) or r_page > start_page
                    next_cur = (
                        _next_overlap_cursor(per_page, full_size, sub_size, r_page, r_page, anchor_key, r_val, extra, rem_bounds)
                        if exc.code == "github_output_limit" and not progressed and per_page > 1
                        else _pack_cursor(r_page, per_page, full_size, anchor_key, r_val, extra, rem_bounds)
                    )
                    err_code = "github_reconciliation_window_exhausted" if exc.code == "github_output_limit" else exc.code
                    backoff = max(exc.retry_after, 3600 if exc.code == "github_output_limit" else 300)
                    raise on_incomplete(next_cur, err_code, backoff) from exc
                if pmatch is not None:
                    return pmatch, {}
                if shifted:
                    boundaries.clear()
                    start_page, page, anchor_val = 1, 1, None
                    last_page, last_val, last_full, pages_scanned = 1, None, False, 0
                    last_full_page, last_full_val = None, None
                    checked_anchor = True
                    continue
                tail_cur: dict[str, Any] = {**extra, "start_page": last_page, anchor_key: last_val}
                if per_page < full_size:
                    tail_cur["sub"] = per_page
                return None, tail_cur
            last_full_page, last_full_val = page, last_val
            boundaries.append((page * per_page, last_val))
            page += 1
        next_cur = (
            _next_overlap_cursor(
                per_page, full_size, sub_size, start_page, last_page, anchor_key, last_val, extra, boundaries,
            )
            if pages_scanned > 0
            else {**extra, "start_page": 1, **({"sub": per_page} if per_page < full_size else {})}
        )
        raise on_incomplete(next_cur, "github_reconciliation_window_exhausted", default_retry)

    def find_comment(
        self,
        issue_number: int,
        marker: str,
        *,
        issue_url: str | None = None,
        cursor: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        if type(issue_number) is not int or issue_number <= 0:
            raise TransportError("github_invalid_issue_reference")
        resolved_url = issue_url or f"https://github.com/{self.repository}/issues/{issue_number}"
        eff_cur = cursor if isinstance(cursor, dict) and cursor.get("issue_number") == issue_number else None
        match, tail_cur = self._scan_paginated(
            f"repos/{self.repository}/issues/{issue_number}/comments?",
            marker, "github_invalid_comment_listing", "id", "anchor_id",
            _COMMENT_PAGE_SIZE, _COMMENT_SUB_PAGE_SIZE, _COMMENT_MAX_PAGES,
            eff_cur, {"issue_number": issue_number}, 300,
            lambda cur, code, backoff: _CommentScanIncomplete(
                issue_number=issue_number, issue_url=resolved_url, cursor=cur, code=code, retry_after=backoff,
            ),
        )
        if match is not None:
            curl = match.get("html_url") if _comment_reference(match, self.repository, issue_number) else None
            return (curl or resolved_url), {}
        return None, tail_cur

    def _scan_historical_issues(
        self,
        markers: str | tuple[str, ...],
        incident_marker: str,
        fault_marker: str | None,
        *,
        issue_cursor: dict[str, Any] | None = None,
        comment_cursor: dict[str, Any] | None = None,
    ):
        match, _ = self._scan_paginated(
            f"repos/{self.repository}/issues?state=all&sort=created&direction=asc&",
            markers, "github_invalid_issue_listing", "number", "anchor_number",
            _ISSUE_SCAN_PAGE_SIZE, _ISSUE_SUB_PAGE_SIZE, _ISSUE_SCAN_MAX_PAGES,
            issue_cursor if isinstance(issue_cursor, dict) else None, {}, 3600,
            lambda cur, code, backoff: _IssueScanIncomplete(cursor=cur, code=code, retry_after=backoff),
        )
        if match is not None:
            return self._resolve_match(match, incident_marker, fault_marker, cursor=comment_cursor)
        return None

    def _resolve_match(
        self,
        match: dict[str, Any],
        incident_marker: str,
        fault_marker: str | None,
        *,
        cursor: dict[str, Any] | None = None,
    ):
        body = match.get("body") or ""
        if incident_marker in body:
            return match
        if fault_marker and fault_marker in body and _issue_reference(match, self.repository):
            try:
                found_url, tail_cursor = self.find_comment(
                    match["number"], incident_marker, issue_url=match["html_url"], cursor=cursor
                )
            except _CommentScanIncomplete:
                raise
            except TransportError as exc:
                if not exc.permanent or exc.code in {"github_http_403", "github_http_429"}:
                    resume_cur = (
                        cursor
                        if isinstance(cursor, dict) and cursor.get("issue_number") == match["number"]
                        else {"issue_number": match["number"], "start_page": 1}
                    )
                    err_code = "github_reconciliation_window_exhausted" if exc.code == "github_output_limit" else exc.code
                    backoff = max(exc.retry_after, 3600 if exc.code == "github_output_limit" else 300)
                    raise _CommentScanIncomplete(
                        issue_number=match["number"],
                        issue_url=match["html_url"],
                        cursor=resume_cur,
                        code=err_code,
                        retry_after=backoff,
                    ) from exc
                raise
            if found_url:
                extra = {"_reconciled_comment_url": found_url} if "#issuecomment-" in found_url else {}
                return {**match, **extra}
            return {**match, "_needs_occurrence_comment": True, "_comment_cursor": tail_cursor}
        return match

    def find_issue(self, item: dict[str, Any]):
        # Uncertain write reconciliation is strictly separated from historical fault discovery.
        payload = item["payload"]
        has_events = bool(payload.get("events"))
        identity = item_public_fingerprint(item)
        is_uncertain = item.get("state") == "uncertain"
        saved_fault = payload.get("fault_signature") if not has_events else None
        if is_uncertain:
            fault_id = None
        elif isinstance(saved_fault, str) and re.fullmatch(r"[0-9a-f]{64}", saved_fault):
            fault_id = saved_fault
        elif has_events:
            fault_id = fault_signature(payload)
        else:
            fault_id = None
        incident_marker = f"{MARKER}{identity} -->"
        fault_marker = f"{FAULT_MARKER}{fault_id} -->" if fault_id and fault_id != identity else None
        markers = (incident_marker, fault_marker) if fault_marker else incident_marker
        saved_cursor = (
            payload.get("_comment_cursor")
            if isinstance(payload, dict) and isinstance(payload.get("_comment_cursor"), dict)
            else None
        )
        saved_issue_cursor = (
            payload.get("_issue_cursor")
            if isinstance(payload, dict) and isinstance(payload.get("_issue_cursor"), dict)
            else None
        )
        standalone = self._deadline is None
        if standalone:
            self.begin_cycle()
        self._searched_in_cycle = True
        cycle_deadline = self._deadline
        self._deadline -= min(float(self.timeout) / 2, 15.0)
        try:
            pinned_num = item.get("issue_number") if is_uncertain else None
            if type(pinned_num) is int and pinned_num > 0:
                iurl = f"https://github.com/{self.repository}/issues/{pinned_num}"
                found_url, _ = self.find_comment(
                    pinned_num, incident_marker, issue_url=iurl, cursor=saved_cursor
                )
                if not found_url:
                    return None
                extra = {"_reconciled_comment_url": found_url} if "#issuecomment-" in found_url else {}
                return {"number": pinned_num, "html_url": iurl, **extra}
            if not is_uncertain and saved_cursor and type(saved_cursor.get("issue_number")) is int and saved_cursor["issue_number"] > 0:
                cnum = saved_cursor["issue_number"]
                try:
                    pinned = self.request("GET", f"repos/{self.repository}/issues/{cnum}")
                except TransportError as exc:
                    if exc.code != "github_http_404":
                        raise
                    pinned, saved_cursor = None, None
                if isinstance(pinned, dict) and _issue_reference(pinned, self.repository):
                    pbody = pinned.get("body") or ""
                    if incident_marker in pbody or (fault_marker and fault_marker in pbody):
                        return self._resolve_match(pinned, incident_marker, fault_marker, cursor=saved_cursor)
                saved_cursor = None
            if not is_uncertain and saved_issue_cursor:
                return self._scan_historical_issues(
                    markers, incident_marker, fault_marker,
                    issue_cursor=saved_issue_cursor, comment_cursor=saved_cursor,
                )
            q_terms = f'"{identity}" OR "{fault_id}"' if fault_marker else f'"{identity}"'
            fault_candidate = None
            search_incomplete = False
            for page in range(1, _SEARCH_MAX_PAGES + 1):
                search_query = urlencode({
                    "q": f"repo:{self.repository} is:issue in:body {q_terms}",
                    "per_page": _SEARCH_PAGE_SIZE, "page": page,
                })
                reply = self.request("GET", f"search/issues?{search_query}")
                if not isinstance(reply, dict):
                    raise TransportError("github_invalid_issue_search")
                items = reply.get("items")
                exact = _matching_issue(items, incident_marker, "github_invalid_issue_search")
                if exact:
                    return self._resolve_match(exact, incident_marker, fault_marker, cursor=saved_cursor)
                if fault_marker and fault_candidate is None:
                    fault_candidate = _matching_issue(items, fault_marker, "github_invalid_issue_search")
                search_incomplete = search_incomplete or bool(reply.get("incomplete_results"))
                total_count = reply.get("total_count")
                if len(items) < _SEARCH_PAGE_SIZE or (type(total_count) is int and page * _SEARCH_PAGE_SIZE >= total_count):
                    break
            else:
                search_incomplete = True
            if fault_candidate is not None:
                return self._resolve_match(fault_candidate, incident_marker, fault_marker, cursor=saved_cursor)
            if not is_uncertain:
                found = self._scan_historical_issues(
                    markers, incident_marker, fault_marker,
                    issue_cursor=None, comment_cursor=saved_cursor,
                )
                if found is not None:
                    return found
                if search_incomplete:
                    raise TransportError("github_reconciliation_window_exhausted", retry_after=3600)
                return None
            attempt = item.get("attempt_timestamp")
            lower = datetime.fromtimestamp(
                (item["first_seen"] if attempt is None else attempt) - _CLOCK_SKEW_MARGIN_SECONDS,
                timezone.utc,
            ).replace(microsecond=0)
            since = (lower - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for page in range(1, _RECONCILIATION_MAX_PAGES + 1):
                rows = self.request("GET", f"repos/{self.repository}/issues?state=all&sort=created"
                                    f"&direction=desc&since={since}&per_page={_RECONCILIATION_PAGE_SIZE}&page={page}")
                match = _matching_issue(rows, incident_marker, "github_invalid_issue_listing")
                if match:
                    return self._resolve_match(match, incident_marker, None)
                if len(rows) < _RECONCILIATION_PAGE_SIZE:
                    return None
                try:
                    last_created = datetime.fromisoformat(rows[-1]["created_at"].replace("Z", "+00:00"))
                    if last_created < lower:
                        return None
                except (KeyError, ValueError, TypeError, AttributeError) as exc:
                    raise TransportError("github_invalid_issue_listing") from exc
            raise TransportError("github_reconciliation_window_exhausted", retry_after=3600)
        except (_CommentScanIncomplete, _IssueScanIncomplete):
            raise
        except TransportError as exc:
            if exc.code == "github_output_limit":
                raise TransportError("github_reconciliation_window_exhausted", retry_after=3600) from exc
            raise
        finally:
            self._deadline = cycle_deadline
            if standalone:
                self.end_cycle()

    def create_issue(self, title: str, body: str):
        return self.request("POST", f"repos/{self.repository}/issues", {"title": title, "body": body})

    def create_comment(self, issue_number: int, body: str):
        return self.request("POST", f"repos/{self.repository}/issues/{issue_number}/comments", {"body": body})


def _unpack_cursor(
    cursor: dict[str, Any] | None, anchor_key: str, full_size: int
) -> tuple[int, int | None, int, list[tuple[int, int | None]]]:
    start_page, anchor_val, per_page = 1, None, full_size
    boundaries: list[tuple[int, int | None]] = []
    if not isinstance(cursor, dict):
        return start_page, anchor_val, per_page, boundaries
    sub = cursor.get("sub")
    if type(sub) is int and 1 <= sub <= full_size:
        per_page = sub
    sp, aval = cursor.get("start_page"), cursor.get(anchor_key)
    if type(sp) is int and sp >= 1 and (sp == 1 or (type(aval) is int and aval > 0)):
        start_page = sp
        anchor_val = aval if type(aval) is int and aval > 0 else None
    raw_bounds = cursor.get("bounds")
    if start_page > 1 and isinstance(raw_bounds, list):
        max_pos, prev_pos = (start_page - 1) * per_page, 0
        for pair in raw_bounds:
            if (
                isinstance(pair, (list, tuple)) and len(pair) == 2 and type(pair[0]) is int
                and prev_pos < pair[0] <= max_pos and (pair[1] is None or (type(pair[1]) is int and pair[1] > 0))
            ):
                boundaries.append((pair[0], pair[1]))
                prev_pos = pair[0]
            else:
                return 1, None, per_page, []
    elif cursor.get("bounds") is not None:
        return 1, None, per_page, []
    return start_page, anchor_val, per_page, boundaries


def _pack_cursor(
    resume: int,
    eff_sub: int,
    full_size: int,
    anchor_key: str,
    anchor_val: int | None,
    extra: dict[str, Any],
    boundaries: list[tuple[int, int | None]],
) -> dict[str, Any]:
    if resume > 1 and anchor_val is None:
        resume = 1
    out: dict[str, Any] = {**extra, "start_page": resume}
    if eff_sub < full_size:
        out["sub"] = eff_sub
    if anchor_val is not None and resume > 1:
        out[anchor_key] = anchor_val
    kept = [[bp, bt] for bp, bt in boundaries if bp <= (resume - 1) * eff_sub]
    if kept:
        out["bounds"] = kept
    return out


def _next_overlap_cursor(
    per_page: int,
    full_size: int,
    sub_size: int,
    start_page: int,
    last_page: int,
    anchor_key: str,
    last_val: int | None,
    extra: dict[str, Any],
    boundaries: list[tuple[int, int | None]] | None = None,
) -> dict[str, Any]:
    covered = last_page * per_page
    if last_page > start_page:
        eff_sub, resume = per_page, last_page
    elif per_page > sub_size:
        eff_sub = sub_size
        resume = max(1, covered // eff_sub)
    elif per_page > 1:
        eff_sub = 1
        resume = max(1, covered)
    else:
        eff_sub, resume = 1, last_page
    aval = last_val if last_val is not None and resume * eff_sub == covered else None
    return _pack_cursor(resume, eff_sub, full_size, anchor_key, aval, extra, boundaries or [])


def _matching_issue(rows, marker, error_code):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise TransportError(error_code)
    markers = (marker,) if isinstance(marker, str) else tuple(m for m in marker if m)
    validated = []
    for row in rows:
        body = row.get("body")
        if body is None:
            body = ""
        if not isinstance(body, str):
            raise TransportError(error_code)
        if len(body.encode("utf-8")) > _MAX_ISSUE_BODY_BYTES:
            raise TransportError("github_reconciliation_window_exhausted", retry_after=3600)
        if "pull_request" not in row:
            validated.append((row, body))
    for needle in markers:
        for row, body in validated:
            if needle in body:
                return row
    return None


def _issue_reference(reply, repository):
    if not isinstance(reply, dict) or type(reply.get("number")) is not int or reply["number"] <= 0:
        return False
    return reply.get("html_url") == f"https://github.com/{repository}/issues/{reply['number']}"


def _comment_reference(reply, repository, issue_number):
    if not isinstance(reply, dict) or type(reply.get("id")) is not int or reply["id"] <= 0:
        return False
    url = reply.get("html_url")
    return isinstance(url, str) and url.startswith(f"https://github.com/{repository}/issues/{issue_number}#issuecomment-")


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
    # Two transports may each need up to 8s cleanup, plus the SQLite receipt.
    item = queue.claim(lease_seconds=github.timeout + 20)
    if not item:
        return {"status": "idle"}
    prior_uncertain = item["state"] == "uncertain"
    post_invoked = False
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
        if existing and not existing.get("_needs_occurrence_comment"):
            if not _issue_reference(existing, github.repository):
                raise TransportError("github_invalid_issue_reference")
            reconciled_comment = existing.get("_reconciled_comment_url")
            pkind = "comment" if reconciled_comment else "issue"
            res = {"status": "reconciled", "publication_kind": pkind, "issue_url": existing["html_url"], "operation_completed": True}
            if reconciled_comment:
                res["comment_url"] = reconciled_comment
            return _record_publication_outcome(queue, item, res,
                state="published", issue_number=existing["number"], issue_url=existing["html_url"],
                last_error=None, comment_url=reconciled_comment, publication_kind=pkind,
                comment_cursor=None, issue_cursor=None)
        if item["state"] == "uncertain":
            state = "uncertain"
            return _record_publication_outcome(queue, item, {"status": state, "submission_state": "uncertain"},
                state=state, next_attempt=queue.clock() + 300,
                last_error="submission_uncertain_no_match")
        if existing and existing.get("_needs_occurrence_comment"):
            if not _issue_reference(existing, github.repository):
                raise TransportError("github_invalid_issue_reference")
            comment_body = render_occurrence_comment(item, issue_state=existing.get("state"))
            require_consent(item.get("consent"))
            queue.begin_post(
                item,
                issue_number=existing["number"],
                issue_url=existing["html_url"],
                comment_cursor=existing.get("_comment_cursor"),
            )
            item["state"] = "uncertain"
            with guard_consent(item.get("consent")):
                post_invoked = True
                comment_reply = github.create_comment(existing["number"], comment_body)
            if not _comment_reference(comment_reply, github.repository, existing["number"]):
                raise TransportError("github_invalid_create_reply", uncertain=True)
            curl = comment_reply["html_url"]
            return _record_publication_outcome(queue, item,
                {"status": "published", "publication_kind": "comment", "issue_url": existing["html_url"],
                 "comment_url": curl, "operation_completed": True},
                state="published", issue_number=existing["number"], issue_url=existing["html_url"],
                last_error=None, comment_url=curl, publication_kind="comment",
                comment_cursor=None, issue_cursor=None)
        title, body = render_issue(item)
        require_consent(item.get("consent"))
        queue.begin_post(item)
        item["state"] = "uncertain"
        with guard_consent(item.get("consent")):
            post_invoked = True
            reply = github.create_issue(title, body)
        if not _issue_reference(reply, github.repository):
            raise TransportError("github_invalid_create_reply", uncertain=True)
        return _record_publication_outcome(queue, item,
            {"status": "published", "publication_kind": "issue", "issue_url": reply["html_url"], "operation_completed": True},
            state="published", issue_number=reply["number"], issue_url=reply["html_url"],
            last_error=None, publication_kind="issue", comment_cursor=None, issue_cursor=None)
    except ConsentWithdrawn:
        uncertain = prior_uncertain or post_invoked
        return _record_publication_outcome(queue, item,
            {"status": "withdrawn", "submission_state": "uncertain" if uncertain else "not_sent"},
            state="withdrawn", last_error="submission_uncertain_reporting_consent_withdrawn" if uncertain else "reporting_consent_unavailable_or_withdrawn")
    except ConsentUnavailable:
        uncertain = prior_uncertain or post_invoked
        state = "uncertain" if uncertain else "retry"
        code = ("submission_uncertain_" if uncertain else "") + "reporting_policy_unavailable"
        return _record_publication_outcome(queue, item,
            {"status": state, "error": code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + 300, last_error=code)
    except _CommentScanIncomplete as exc:
        uncertain = prior_uncertain or exc.uncertain
        state = "uncertain" if uncertain else "retry"
        delay = max(exc.retry_after, min(3600, 30 * 2 ** min(item["attempts"], 7)))
        return _record_publication_outcome(queue, item,
            {"status": state, "error": exc.code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + delay, last_error=exc.code,
            issue_number=exc.issue_number, issue_url=exc.issue_url, comment_cursor=exc.cursor, issue_cursor=None)
    except _IssueScanIncomplete as exc:
        uncertain = prior_uncertain or exc.uncertain
        state = "uncertain" if uncertain else "retry"
        delay = max(exc.retry_after, min(3600, 30 * 2 ** min(item["attempts"], 7)))
        return _record_publication_outcome(queue, item,
            {"status": state, "error": exc.code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + delay, last_error=exc.code, issue_cursor=exc.cursor)
    except TransportError as exc:
        uncertain = prior_uncertain or exc.uncertain
        recoverable = exc.code in {"github_client_missing", "github_auth_required", "github_http_401", "github_http_403", "github_http_404", "github_http_429"}
        state = "uncertain" if uncertain else "permanent-failed" if exc.permanent and not recoverable else "retry"
        code = "submission_uncertain_" + exc.code if uncertain and state in {"permanent-failed", "exhausted"} else exc.code
        delay = max(exc.retry_after, min(3600, 30 * 2 ** min(item["attempts"], 7)))
        return _record_publication_outcome(queue, item,
            {"status": state, "error": code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + delay, last_error=code)
    except (ValueError, TypeError, KeyError):
        uncertain = prior_uncertain or post_invoked
        state = "uncertain" if uncertain else "blocked"
        code = "submission_uncertain_invalid_reply" if uncertain else "invalid_or_unsafe_diagnostic_payload"
        return _record_publication_outcome(queue, item,
            {"status": state, "error": code, "submission_state": "uncertain" if uncertain else "not_sent"},
            state=state, next_attempt=queue.clock() + 300, last_error=code)
    finally:
        if started:
            github.end_cycle()


def ingest(root: str | Path, queue: Outbox, *, max_files: int = 256, max_bytes: int = 8 * 1024 * 1024, since: float | None = None, config=None, cancel=None, deadline: float | None = None) -> dict[str, int]:
    from .ingestion import ingest as read_events
    return read_events(root, queue, max_files=max_files, max_bytes=max_bytes, since=since,
                       config=config, cancel=cancel, deadline=deadline)
