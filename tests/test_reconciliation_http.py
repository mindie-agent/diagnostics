"""Tier 1: real SQLite and HTTP-to-gh processes exercise reconciliation limits."""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, fingerprint, publish_one
from mindie_diagnostics.reporting import scope_key
from test_reporter import payload


_REQUEST = GitHub.request
_NOW = datetime(2026, 10, 29, tzinfo=timezone.utc).timestamp()


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _issue(repository, number, *, body="", created=_NOW, updated=None):
    return {
        "number": number,
        "html_url": f"https://github.com/{repository}/issues/{number}",
        "body": body,
        "created_at": _iso(created),
        "updated_at": _iso(created if updated is None else updated),
        "state": "open",
    }


@pytest.fixture
def github_http(tmp_path, monkeypatch, reporting_consent):
    state = SimpleNamespace(
        repository=reporting_consent["repository"], indexed=[], issues=[],
        incomplete=False, calls=[], posts=[], response_bytes=[], on_post=None,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _reply(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            state.response_bytes.append(
                len(b"".join(self._headers_buffer)) + 2 + len(data),
            )
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                # Byte-limit cancellation deliberately closes the real pipe/client.
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            state.calls.append(("GET", parsed.path, query))
            if parsed.path == "/search/issues":
                self._reply({
                    "total_count": len(state.indexed),
                    "incomplete_results": state.incomplete,
                    "items": state.indexed[:int(query["per_page"][0])],
                })
                return
            assert parsed.path == f"/repos/{state.repository}/issues"
            assert query["state"] == ["all"]
            assert query["sort"] == ["created"]
            assert query["direction"] == ["desc"]
            since = query["since"][0]
            rows = sorted(
                (row for row in state.issues if row["updated_at"] > since),
                key=lambda row: row["created_at"], reverse=True,
            )
            size, page = int(query["per_page"][0]), int(query["page"][0])
            self._reply(rows[(page - 1) * size:page * size])

        def do_POST(self):
            parsed = urlsplit(self.path)
            state.calls.append(("POST", parsed.path, {}))
            assert parsed.path == f"/repos/{state.repository}/issues"
            posted = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.posts.append(posted)
            if state.on_post is not None:
                state.on_post()
            issue = _issue(state.repository, 900, body=posted["body"])
            state.issues.append(issue)
            self._reply(issue, status=201)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True,
    )
    thread.start()
    helper = Path(__file__).with_name("http_gh_stub.py")
    if os.name == "nt":
        executable = tmp_path / "gh.cmd"
        executable.write_text(f'@"{sys.executable}" "{helper}" %*\n', encoding="utf-8", newline="\n")
    else:
        executable = tmp_path / "gh"
        executable.write_text(f"#!{sys.executable}\n" + helper.read_text(), encoding="utf-8", newline="\n")
        executable.chmod(0o700)
    monkeypatch.setenv("DIAGNOSTICS_TEST_HTTP_PORT", str(server.server_port))
    monkeypatch.setattr(GitHub, "request", _REQUEST)
    state.executable = str(executable)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def _queued(tmp_path, consent, *, uncertain=False):
    now = [_NOW - 86400 * 30]
    queue = Outbox(tmp_path / "queue.db", clock=lambda: now[0])
    bundle = payload()
    queue.enqueue(scope_key(fingerprint(bundle), consent), "a" * 32, bundle, consent=consent)
    now[0] = _NOW
    if uncertain:
        item = queue.claim()
        queue.begin_post(item)
        queue.update(item, state="uncertain", last_error="github_timeout")
    return queue, now


def _client(server, **options):
    return GitHub(server.repository, executable=server.executable, **options)


def _marker():
    return f"<!-- mindie-incident:{fingerprint(payload())} -->"


def _list_calls(server):
    return [call for call in server.calls if call[1].startswith("/repos/") and call[0] == "GET"]


def test_search_positive_confirms_old_issue_without_listing_or_post(
    tmp_path, reporting_consent, github_http,
):
    queue, _ = _queued(tmp_path, reporting_consent)
    existing = _issue(github_http.repository, 11, body=_marker(), created=_NOW - 86400 * 90)
    github_http.indexed = [existing]
    github_http.incomplete = True

    result = publish_one(queue, _client(github_http))

    assert result["status"] == "reconciled"
    assert result["issue_url"] == existing["html_url"]
    assert queue.rows()[0]["state"] == "published"
    assert len(github_http.calls) == 1
    query = github_http.calls[0][2]["q"][0]
    assert f"repo:{github_http.repository}" in query
    assert "is:issue" in query and "in:body" in query
    assert fingerprint(payload()) in query
    assert not github_http.posts


@pytest.mark.parametrize("uncertain", [False, True])
def test_delayed_search_index_uses_since_listing_to_find_issue(
    tmp_path, reporting_consent, github_http, uncertain,
):
    queue, _ = _queued(tmp_path, reporting_consent, uncertain=uncertain)
    github_http.incomplete = True
    existing = _issue(github_http.repository, 21, body=_marker(), created=_NOW + 1)
    github_http.issues = [
        _issue(github_http.repository, number, body="x" * 60000, created=_NOW - 86400 * 90)
        for number in range(100, 350)
    ] + [existing]

    result = publish_one(queue, _client(github_http))

    assert result["status"] == "reconciled"
    assert not github_http.posts
    listing = _list_calls(github_http)
    assert len(listing) == 1
    lower = _NOW if uncertain else _NOW - 86400 * 30
    assert listing[0][2]["since"] == [_iso(lower - 301)]
    assert sum(github_http.response_bytes) < 4096


@pytest.mark.parametrize("uncertain", [False, True])
def test_delayed_index_finds_issue_created_and_updated_exactly_at_window_lower_bound(
    tmp_path, reporting_consent, github_http, uncertain,
):
    queue, _ = _queued(tmp_path, reporting_consent, uncertain=uncertain)
    attempt = _NOW if uncertain else _NOW - 86400 * 30
    lower = attempt - 300
    existing = _issue(github_http.repository, 22, body=_marker(), created=lower, updated=lower)
    github_http.issues = [existing]

    result = publish_one(queue, _client(github_http))

    assert result["status"] == "reconciled"
    assert result["issue_url"] == existing["html_url"]
    assert _list_calls(github_http)[0][2]["since"] == [_iso(lower - 1)]
    assert queue.rows()[0]["state"] == "published"
    assert not github_http.posts


def test_search_miss_requires_complete_listing_before_atomic_post(
    tmp_path, reporting_consent, github_http,
):
    queue, now = _queued(tmp_path, reporting_consent)
    observed = []

    def record_attempt():
        with queue.connect() as db:
            observed.append(tuple(db.execute(
                "SELECT state,attempt_timestamp FROM incidents",
            ).fetchone()))

    github_http.on_post = record_attempt
    result = publish_one(queue, _client(github_http))

    assert result["status"] == "published"
    assert [call[0] for call in github_http.calls] == ["GET", "GET", "POST"]
    assert len(_list_calls(github_http)) == 1
    assert len(github_http.posts) == 1
    assert _marker() in github_http.posts[0]["body"]
    assert observed == [("uncertain", now[0])]


def test_uncertain_no_match_retains_original_since_and_never_reposts(
    tmp_path, reporting_consent, github_http,
):
    queue, now = _queued(tmp_path, reporting_consent, uncertain=True)
    client = _client(github_http)
    for _ in range(2):
        result = publish_one(queue, client)
        assert result["status"] == "uncertain"
        now[0] += 4000

    assert [call[2]["since"] for call in _list_calls(github_http)] == [
        [_iso(_NOW - 301)], [_iso(_NOW - 301)],
    ]
    assert queue.rows()[0]["last_error"] == "submission_uncertain_no_match"
    assert not github_http.posts


def test_full_page_stops_when_created_tail_crosses_attempt_window(
    tmp_path, reporting_consent, github_http,
):
    queue, _ = _queued(tmp_path, reporting_consent, uncertain=True)
    github_http.issues = [
        _issue(github_http.repository, number, created=_NOW + number, updated=_NOW + 100)
        for number in range(1, 30)
    ] + [
        _issue(github_http.repository, number, created=_NOW - 301 - number, updated=_NOW + 100)
        for number in range(100, 160)
    ]

    result = publish_one(queue, _client(github_http))

    assert result["status"] == "uncertain"
    assert len(_list_calls(github_http)) == 1
    assert not github_http.posts


def test_created_tail_equal_to_lower_bound_continues_to_find_same_second_issue(
    tmp_path, reporting_consent, github_http,
):
    queue, _ = _queued(tmp_path, reporting_consent, uncertain=True)
    github_http.issues = [
        _issue(github_http.repository, number, created=_NOW + number, updated=_NOW + 100)
        for number in range(1, 30)
    ] + [
        _issue(github_http.repository, 30, created=_NOW - 300, updated=_NOW),
        _issue(github_http.repository, 31, body=_marker(), created=_NOW - 300, updated=_NOW),
    ]

    result = publish_one(queue, _client(github_http))

    assert result["status"] == "reconciled"
    assert result["issue_url"].endswith("/issues/31")
    assert len(_list_calls(github_http)) == 2
    assert not github_http.posts


@pytest.mark.parametrize("phase", ["search", "listing"])
@pytest.mark.parametrize("uncertain", [False, True])
def test_reconciliation_byte_limit_is_visible_and_never_allows_post(
    tmp_path, reporting_consent, github_http, phase, uncertain,
):
    queue, now = _queued(tmp_path, reporting_consent, uncertain=uncertain)
    row = _issue(github_http.repository, 51, body="x" * 4096)
    if phase == "search":
        github_http.indexed = [row]
    else:
        github_http.issues = [row]
    client = _client(github_http, max_response_bytes=1024)
    for _ in range(2):
        result = publish_one(queue, client)
        assert result["status"] == ("uncertain" if uncertain else "retry")
        assert result["error"] == "github_reconciliation_window_exhausted"
        assert queue.rows()[0]["last_error"] == result["error"]
        assert queue.rows()[0]["next_attempt"] >= now[0] + 3600
        now[0] += 4000
    assert not github_http.posts


def test_search_and_listing_consume_one_real_byte_budget(
    tmp_path, reporting_consent, github_http,
):
    queue, _ = _queued(tmp_path, reporting_consent, uncertain=True)
    github_http.indexed = [_issue(github_http.repository, 61, body="x" * 800)]
    github_http.issues = [_issue(github_http.repository, 62, body="x" * 800 + _marker())]
    limit = 1500
    client = _client(github_http, max_response_bytes=limit)

    result = publish_one(queue, client)

    assert result["status"] == "uncertain"
    assert result["error"] == "github_reconciliation_window_exhausted"
    assert len(github_http.response_bytes) == 2
    assert all(size < limit for size in github_http.response_bytes)
    assert sum(github_http.response_bytes) > limit
    assert client._bytes == github_http.response_bytes[0]
    assert not github_http.posts


def test_page_budget_exhaustion_is_not_remote_absence(
    tmp_path, reporting_consent, github_http,
):
    queue, _ = _queued(tmp_path, reporting_consent)
    github_http.issues = [
        _issue(github_http.repository, number, created=_NOW + number)
        for number in range(1, 181)
    ]

    result = publish_one(queue, _client(github_http))

    assert result["status"] == "retry"
    assert result["error"] == "github_reconciliation_window_exhausted"
    assert len(_list_calls(github_http)) == 5
    assert not github_http.posts
