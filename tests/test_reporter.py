import copy
from pathlib import Path

import pytest

from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, TransportError, fingerprint, publish_one, render_issue
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
    ('lost_missing', 'exhausted', 3, 1), ('read_error', 'exhausted', 3, 0),
    ('auth', 'permanent-failed', 1, 0),
])
def test_publication_budget_and_unknown_post(tmp_path, reporting_consent, mode, final, reads, writes):
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


def test_revocation_after_read_stops_before_post(tmp_path, reporting_consent):
    queue, _ = queue_at(tmp_path, reporting_consent)
    github = FakeGitHub(reporting_consent['repository'])
    def revoke(item):
        Path(reporting_consent['config_file']).unlink()
        return None
    github.find_issue = revoke
    assert publish_one(queue, github)['status'] == 'withdrawn'
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
