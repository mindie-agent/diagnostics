import copy
import json
from pathlib import Path

import pytest

from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, TransportError, fingerprint, issue_payload, publish_one, render_issue
from mindie_diagnostics.reporting import scope_key


def payload():
    return {'events': [{'schema': 1, 'timestamp': '2026-09-22T00:00:00Z', 'monotonic_ns': 1,
        'component': 'remote-dev', 'severity': 'ERROR', 'event': 'operation.end',
        'operation_id': 'a' * 32, 'trace_id': 'a' * 32, 'process_instance_id': 'b' * 32,
        'operation': 'remote.read', 'status': 'error', 'package_revision': 'c' * 40,
        'attributes': {'category': 'internal_exception', 'stage': 'tool_call', 'error_type': 'AssertionError'}}]}


class FakeGitHub(GitHub):
    def __init__(self, repository, *, mode='success'):
        super().__init__(repository)
        self.created, self.mode, self.reads, self.writes = None, mode, 0, 0

    def find_issue(self, item):
        self.reads += 1
        if self.mode == 'read_error':
            raise TransportError('github_timeout')
        if self.mode == 'auth':
            raise TransportError('github_http_401', permanent=True)
        return self.created if self.mode == 'lost_found' else None

    def create_issue(self, title, body):
        self.writes += 1
        self.created = {'number': 1, 'html_url': f'https://github.com/{self.repository}/issues/1'}
        if self.mode.startswith('lost'):
            raise TransportError('github_timeout', uncertain=True)
        return self.created


def queue_at(tmp_path, consent):
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    body = payload()
    queue.enqueue(scope_key(fingerprint(body), consent), 'a' * 32, body, consent=consent)
    return queue, now


@pytest.mark.parametrize('mode,final,reads,writes', [
    ('success', 'published', 1, 1), ('lost_found', 'published', 2, 1),
    ('lost_missing', 'uncertain', 4, 1), ('read_error', 'retry', 4, 0),
    ('auth', 'retry', 4, 0),
])
def test_recoverable_reads_retain_work_and_unknown_post_never_replays(tmp_path, reporting_consent, mode, final, reads, writes):
    queue, now = queue_at(tmp_path, reporting_consent)
    github = FakeGitHub(reporting_consent['repository'], mode=mode)
    for _ in range(4):
        publish_one(queue, github)
        now[0] += 4000
    row = queue.rows()[0]
    assert (row['state'], github.reads, github.writes) == (final, reads, writes)
    assert row['incident_ids'] == ['a' * 32]
    if mode == 'lost_missing':
        assert 'uncertain' in row['last_error']


def test_repository_mismatch_stops_before_remote_action(tmp_path, reporting_consent):
    queue, _ = queue_at(tmp_path, reporting_consent)
    github = FakeGitHub('other/repository')
    assert publish_one(queue, github)['status'] == 'permanent-failed'
    assert github.reads == github.writes == 0


@pytest.mark.parametrize("mode", ["not_started", "http_rejection", "read_error"])
def test_proven_not_sent_failure_recovers_to_one_publication(tmp_path, reporting_consent, monkeypatch, mode):
    from mindie_diagnostics import reporter

    queue, now = queue_at(tmp_path, reporting_consent)
    github = GitHub(reporting_consent["repository"])
    monkeypatch.setattr(GitHub, "request", reporter_request)
    posts = []
    broken = [True]

    def transport(command, data, **kwargs):
        method = command[command.index("--method") + 1]
        if method == "GET":
            if broken[0] and mode == "read_error":
                raise TransportError("github_timeout")
            return 0, b"HTTP/2 200 OK\r\n\r\n[]", 22
        if broken[0]:
            if mode == "not_started":
                raise TransportError("github_client_missing", permanent=True)
            return 1, b'HTTP/2 403 Forbidden\r\n\r\n{"message":"Forbidden"}', 48
        posts.append(json.loads(data))
        body = json.dumps({"number": 5, "html_url": f"https://github.com/{github.repository}/issues/5"}).encode()
        return 0, b"HTTP/2 201 Created\r\n\r\n" + body, len(body) + 24

    monkeypatch.setattr(reporter, "_run_gh", transport)
    for _ in range(5):
        assert publish_one(queue, github)["status"] == "retry"
        now[0] += 4000
    broken[0] = False
    assert publish_one(queue, github)["status"] == "published"
    assert len(posts) == 1


# The autouse fixture disables network by replacing this method for every test.
reporter_request = GitHub.request


def test_search_and_post_share_the_original_cycle_deadline(monkeypatch):
    from mindie_diagnostics import reporter

    clock = [100.0]
    monkeypatch.setattr(reporter.time, "monotonic", lambda: clock[0])
    github = GitHub("example/project", timeout=30)
    deadlines = []

    def transport(command, data, **options):
        deadlines.append(options["deadline"])
        clock[0] += 10
        return 0, b"HTTP/2 200 OK\n\n[]", 18

    monkeypatch.setattr(reporter, "_run_gh", transport)
    monkeypatch.setattr(GitHub, "request", reporter_request)
    github.begin_cycle()
    assert github.find_issue({"payload": payload()}) is None
    github.create_issue("synthetic", "synthetic")
    assert deadlines == [115.0, 130.0]
    assert github._deadline == 130.0
    github.end_cycle()


def test_legacy_unknown_receipt_resumes_reconciliation_without_post(tmp_path, reporting_consent, monkeypatch):
    queue, _ = queue_at(tmp_path, reporting_consent)
    item = queue.claim()
    queue.update(item, state="exhausted", last_error="submission_uncertain_budget_exhausted")
    reopened = Outbox(queue.path)
    marker = f"<!-- mindie-incident:{fingerprint(payload())} -->"
    calls = []

    def existing(self, method, path, data=None):
        calls.append(method)
        return [{"number": 4, "html_url": f"https://github.com/{self.repository}/issues/4",
                 "body": marker}]

    monkeypatch.setattr(GitHub, "request", existing)
    assert publish_one(reopened, GitHub(reporting_consent["repository"]))["status"] == "reconciled"
    assert calls == ["GET"]


def test_missing_policy_after_read_blocks_post_without_claiming_revocation(tmp_path, reporting_consent):
    queue, _ = queue_at(tmp_path, reporting_consent)
    github = FakeGitHub(reporting_consent['repository'])
    def revoke(item):
        Path(reporting_consent['config_file']).unlink()
        return None
    github.find_issue = revoke
    assert publish_one(queue, github)['status'] == 'retry'
    assert queue.rows()[0]['last_error'] == 'reporting_policy_unavailable'
    assert github.writes == 0


def test_unsafe_payload_is_blocked_before_post(tmp_path, reporting_consent, monkeypatch):
    queue, _ = queue_at(tmp_path, reporting_consent)
    github = FakeGitHub(reporting_consent['repository'])
    def unsafe(_):
        raise ValueError('private sentinel')
    monkeypatch.setattr('mindie_diagnostics.reporter.render_issue', unsafe)
    assert publish_one(queue, github)['status'] == 'blocked'
    assert github.writes == 0


def test_public_fingerprint_is_independent_of_consent_and_incident():
    original = payload()
    same = copy.deepcopy(original)
    same['events'][0].update(operation_id='d' * 32, timestamp='2026-09-22T01:00:00Z', process_instance_id='e' * 32)
    assert fingerprint(same) == fingerprint(original)
    same['events'][0]['package_revision'] = 'f' * 40
    assert fingerprint(same) != fingerprint(original)
    _, body = render_issue({'fingerprint': 'private_consent_key', 'payload': original, 'occurrences': 1})
    assert fingerprint(original) in body and 'private_consent_key' not in body


def test_issue_projection_excludes_local_identity_and_remains_reprojectable():
    original = payload()
    event = original['events'][0]
    identifiers = ('operation_id', 'trace_id', 'parent_operation_id', 'phase_id',
                   'parent_phase_id', 'process_instance_id', 'process_ref')
    for index, name in enumerate(identifiers, 1):
        event[name] = (f'{index:x}' + 'abcdef0123456789' * 3)[:32]
    event['attributes']['session_id'] = 'private-session-canary'
    before = copy.deepcopy(original)
    projected = issue_payload(original)
    text = json.dumps(projected)
    for name in (*identifiers, 'monotonic_ns', 'clock_domain_unknown', 'pid'):
        assert name not in projected['events'][0]
    assert all(event[name] not in text for name in identifiers)
    assert 'private-session-canary' not in text
    assert original == before
    assert issue_payload(projected) == projected
    assert fingerprint(projected) == fingerprint(original)
    assert projected['events'][0]['package_revision'] == event['package_revision']


def test_sanitized_issue_marker_reconciles_an_uncertain_post(tmp_path, reporting_consent):
    queue, now = queue_at(tmp_path, reporting_consent)

    class LostResponse(GitHub):
        def __init__(self):
            super().__init__(reporting_consent['repository'])
            self.posts = []

        def request(self, method, path, payload=None):
            if method == 'GET':
                return self.posts
            assert method == 'POST'
            issue = dict(payload, number=7,
                         html_url=f'https://github.com/{self.repository}/issues/7')
            self.posts.append(issue)
            raise TransportError('github_timeout', uncertain=True)

    github = LostResponse()
    publish_one(queue, github)
    assert len(github.posts) == 1
    assert queue.rows()[0]['state'] == 'uncertain'
    now[0] += 4000
    assert publish_one(queue, github)['status'] == 'reconciled'
    assert queue.rows()[0]['state'] == 'published'
    assert len(github.posts) == 1
    assert queue.rows()[0]['incident_ids'] == ['a' * 32]
    body = github.posts[0]['body']
    assert 'a' * 32 not in body and 'b' * 32 not in body
    assert fingerprint(payload()) in body


def test_create_issue_guarantees_post_budget_after_slow_find_issue():
    import time

    observed_post_remaining = []

    class SlowSearchGitHub(GitHub):
        def request(self, method, path, payload=None):
            if method == 'GET':
                # Simulate multi-page issue listing consuming nearly all of the initial cycle window.
                self._deadline = time.monotonic() + 0.05
                return []
            assert method == 'POST'
            observed_post_remaining.append(self._deadline - time.monotonic())
            return {'number': 42, 'html_url': f'https://github.com/{self.repository}/issues/42'}

    gh = SlowSearchGitHub('owner/repo', timeout=30)
    gh.begin_cycle()
    try:
        assert gh.find_issue({'payload': payload()}) is None
        created = gh.create_issue('Title', 'Body')
    finally:
        gh.end_cycle()
    assert created['number'] == 42
    assert len(observed_post_remaining) == 1
    assert observed_post_remaining[0] >= 14.0

