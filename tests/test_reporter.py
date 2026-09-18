import json
import subprocess

import pytest

from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, TransportError, publish_one
from mindie_diagnostics.community import current_consent

pytestmark = pytest.mark.usefixtures('community_consent')


class FakeGitHub:
    def __init__(self, *, lost=False):
        self.created = []
        self.lost = lost

    def find_issue(self, item):
        return self.created[0] if self.created else None

    def create_issue(self, title, body):
        issue = {'number': 1, 'html_url': 'https://github.com/example/project/issues/1'}
        self.created.append(issue)
        if self.lost:
            raise TransportError('lost_reply', uncertain=True)
        return issue


def test_accepted_post_lost_reply_reconciles_without_duplicate(tmp_path, monkeypatch):
    now = [1000.0]
    queue = Outbox(tmp_path / 'queue.db', clock=lambda: now[0])
    queue.enqueue('a', 'o', {}, consent=current_consent())
    monkeypatch.setattr('mindie_diagnostics.reporter.render_issue', lambda _: ('title', 'body'))
    github = FakeGitHub(lost=True)
    assert publish_one(queue, github)['status'] == 'uncertain'
    now[0] += 100
    assert publish_one(queue, github)['status'] == 'reconciled'
    assert len(github.created) == 1


def test_uncertain_missing_issue_never_replays(tmp_path, monkeypatch):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('a', 'o', {}, consent=current_consent())
    item = queue.claim()
    queue.update(item, state='uncertain')
    github = FakeGitHub()
    assert publish_one(queue, github)['status'] == 'uncertain'
    assert not github.created


def test_sanitizer_failure_blocks_post(tmp_path, monkeypatch):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('a', 'o', {}, consent=current_consent())
    monkeypatch.setattr('mindie_diagnostics.reporter.render_issue', lambda _: (_ for _ in ()).throw(ValueError('unsafe')))
    github = FakeGitHub()
    assert publish_one(queue, github)['status'] == 'blocked'
    assert not github.created


def test_rate_limit_is_not_submission_uncertain(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 1, '', 'HTTP 429'))
    with pytest.raises(TransportError) as error:
        GitHub().create_issue('title', 'body')
    assert error.value.retry_after == 3600
    assert not error.value.uncertain


def test_mutating_timeout_is_uncertain(monkeypatch):
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired('gh', 30)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(TransportError) as error:
        GitHub().create_issue('title', 'body')
    assert error.value.uncertain
