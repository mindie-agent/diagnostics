"""Consent transitions use real receipts, log files and queues; no live upload."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import uuid

import pytest

from mindie_diagnostics import bind_community_policy, collect_bundle, configure
from mindie_diagnostics.bot import Grok, diagnose_one, enqueue_issues
from mindie_diagnostics.community import (ConsentWithdrawn, consent_allowed, current_consent,
                                        guard_consent, read_policy)
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, publish_one, render_issue


def policy(path, *, decision='enabled', workspace_id=None):
    receipt = {'schema': 'mindie.community.v1', 'workspace_id': workspace_id or uuid.uuid4().hex,
               'decision': decision, 'revision': uuid.uuid4().hex}
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix('.tmp')
    stage.write_text(json.dumps(receipt), encoding='utf-8')
    os.replace(stage, path)
    return receipt


def revoke(reference):
    policy(Path(reference['policy_file']), decision='disabled', workspace_id=reference['workspace_id'])


def failure(root, path):
    with bind_community_policy(path):
        recorder = configure('test-community', root=root)
        with recorder.operation('prepare') as operation:
            operation.fail('transport', error_type='TimeoutError')
        recorder.close()
    return operation


class NoNetwork:
    repository = 'example/project'

    def __getattr__(self, name):
        if name == 'timeout':
            return 30
        pytest.fail('unexpected external action: ' + name)


class Publisher:
    repository = 'example/project'

    def __init__(self):
        self.calls = []

    def find_issue(self, item):
        self.calls.append('find')

    def create_issue(self, title, body):
        self.calls.append('post')
        self.body = body
        return {'number': 1, 'html_url': 'https://github.com/example/project/issues/1'}


def test_default_and_explicit_disabled_keep_logs_but_never_enqueue(tmp_path, monkeypatch):
    monkeypatch.delenv('MINDIE_COMMUNITY_POLICY', raising=False)
    receipt = tmp_path / 'community.json'
    policy(receipt, decision='disabled')
    for selected in (None, receipt):
        failure(tmp_path, selected)
    queue = Outbox(tmp_path / 'queue.db')
    result = ingest(tmp_path, queue)
    assert result['consent_skipped'] == 2 and result['enqueued'] == 0
    assert collect_bundle(tmp_path)['summary']['error_count'] > 0
    assert publish_one(queue, NoNetwork()) == {'status': 'idle'}
    # Choice later cannot retroactively enroll historical records.
    policy(receipt)
    assert ingest(tmp_path, Outbox(tmp_path / 'new.db'))['enqueued'] == 0


def test_revoke_pending_and_reenable_never_reauthorizes_old_revision(tmp_path, community_consent):
    path = Path(community_consent['policy_file'])
    failure(tmp_path, path)
    queue = Outbox(tmp_path / 'queue.db')
    assert ingest(tmp_path, queue)['enqueued'] == 1
    revoke(community_consent)
    assert publish_one(queue, NoNetwork()) == {'status': 'withdrawn'}
    policy(path, workspace_id=community_consent['workspace_id'])
    queue = Outbox(queue.path)
    assert publish_one(queue, NoNetwork()) == {'status': 'idle'}
    assert not consent_allowed(community_consent)
    assert ingest(tmp_path, Outbox(tmp_path / 'fresh.db'))['enqueued'] == 0
    failure(tmp_path, path)
    assert ingest(tmp_path, queue)['enqueued'] == 1
    assert publish_one(queue, Publisher())['status'] == 'published'


def test_global_worker_scopes_same_failure_and_revokes_only_one_workspace(tmp_path):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    policy(first)
    policy(second)
    failure(tmp_path, first)
    failure(tmp_path, second)
    queue = Outbox(tmp_path / 'queue.db', capacity=2)
    assert ingest(tmp_path, queue)['enqueued'] == 2
    policy(first, decision='disabled')
    assert queue.withdraw_unconsented() == 1
    publisher = Publisher()
    assert publish_one(queue, publisher)['status'] == 'published'
    assert publisher.calls == ['find', 'post']
    assert len(queue.rows()) == 2
    assert {row['state'] for row in queue.rows()} == {'withdrawn', 'published'}
    assert str(second) not in publisher.body and 'policy_file' not in publisher.body
    assert 'workspace_id' not in publisher.body and '"community"' not in publisher.body


def test_revocation_after_reconciliation_stops_issue_post(tmp_path, community_consent):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('a', 'b', {}, consent=community_consent)

    class DuringRead(Publisher):
        def find_issue(self, item):
            revoke(community_consent)

        def create_issue(self, *args):
            pytest.fail('revoked issue must not be posted')

    assert publish_one(queue, DuringRead()) == {'status': 'withdrawn'}
    assert queue.rows()[0]['attempts'] == 0


def test_each_github_reconciliation_page_rechecks_receipt(tmp_path, community_consent, monkeypatch):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('a', 'b', {}, consent=community_consent)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        revoke(community_consent)
        return subprocess.CompletedProcess(command, 0, json.dumps([{'body': ''}] * 100), '')

    monkeypatch.setattr(subprocess, 'run', run)
    assert publish_one(queue, GitHub()) == {'status': 'withdrawn'}
    assert len(calls) == 1


@pytest.mark.parametrize('state', ['pending', 'retry', 'uncertain'])
def test_unscoped_outbox_cannot_use_process_consent(tmp_path, community_consent, state):
    queue = Outbox(tmp_path / 'queue.db')
    queue.enqueue('a', 'b', {})
    item = queue.claim()
    queue.update(item, state=state)
    assert publish_one(queue, NoNetwork()) == {'status': 'withdrawn'}
    assert queue.rows()[0]['attempts'] == 0


def test_forward_only_local_reporter_issues_and_preserve_revocation(tmp_path, community_consent):
    failure(tmp_path, community_consent['policy_file'])
    source, bot = Outbox(tmp_path / 'report.db'), Outbox(tmp_path / 'bot.db')
    assert ingest(tmp_path, source)['enqueued'] == 1
    assert publish_one(source, Publisher())['status'] == 'published'
    assert enqueue_issues(NoNetwork(), bot, source=source) == 1
    assert enqueue_issues(NoNetwork(), bot, source=source) == 0
    revoke(community_consent)
    assert diagnose_one(bot, NoNetwork(), NoNetwork()) == {'status': 'withdrawn'}


def test_unscoped_published_issue_is_not_forwarded_to_model(tmp_path, community_consent):
    source, bot = Outbox(tmp_path / 'report.db'), Outbox(tmp_path / 'bot.db')
    source.enqueue('a', 'b', {})
    item = source.claim()
    source.update(item, state='published', issue_number=1)
    assert enqueue_issues(NoNetwork(), bot, source=source) == 0
    assert source.rows()[0]['diagnosis_state'] == 'withdrawn'


@pytest.mark.parametrize('when', ['before', 'during_read', 'during_model'])
def test_bot_revocation_prevents_model_or_comment(tmp_path, community_consent, when):
    queue = Outbox(tmp_path / 'bot.db')
    queue.enqueue('a', 'b', {'issue_number': 1, 'evidence': {}}, consent=community_consent)
    calls = []

    class Git:
        repository = 'example/project'

        def request(self, method, path, payload=None):
            calls.append(method)
            assert method == 'GET'
            if when == 'during_read':
                revoke(community_consent)
            return []

    class Model:
        def diagnose(self, payload):
            calls.append('model')
            revoke(community_consent)
            return 'Observed: failure. Missing evidence: root cause.'

    if when == 'before':
        revoke(community_consent)
    assert diagnose_one(queue, Git(), Model()) == {'status': 'withdrawn'}
    assert calls == {'before': [], 'during_read': ['GET'], 'during_model': ['GET', 'model']}[when]


def test_grok_rechecks_after_local_inspection_before_paid_request(tmp_path, community_consent, monkeypatch):
    failure(tmp_path, community_consent['policy_file'])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert 'inspect' in command
        revoke(community_consent)
        return subprocess.CompletedProcess(command, 0, '{}', '')

    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(ConsentWithdrawn), guard_consent(community_consent):
        Grok(home=tmp_path / 'home', work=tmp_path / 'work').diagnose(collect_bundle(tmp_path))
    assert len(calls) == 1


@pytest.mark.parametrize('contents', ['{}', '[]', 'null', '{', 'x' * 16385,
                                     '{"schema":"mindie.community.v1","decision":[]}'])
def test_missing_or_malformed_policy_fails_closed(tmp_path, contents):
    path = tmp_path / 'community.json'
    assert read_policy(path) is None
    path.write_text(contents, encoding='utf-8')
    assert read_policy(path) is None
    with bind_community_policy(path):
        assert current_consent() is None


def test_request_bindings_isolate_parallel_workspaces(tmp_path, monkeypatch):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    policy(first)
    policy(second)
    monkeypatch.setenv('MINDIE_COMMUNITY_POLICY', str(first))

    async def request(path):
        with bind_community_policy(path):
            await asyncio.sleep(0)
            return current_consent()

    async def run():
        return await asyncio.gather(request(first), request(second), request(None))

    results = asyncio.run(run())
    assert results[0]['policy_file'] == str(first)
    assert results[1]['policy_file'] == str(second)
    assert results[2] is None
    assert current_consent()['policy_file'] == str(first)


def test_windows_policy_path_is_mapped_for_wsl_reader(monkeypatch):
    from mindie_diagnostics import community
    monkeypatch.setattr(community.os, 'name', 'posix')
    # Avoid instantiating PosixPath on a Windows interpreter by injecting a
    # pure path constructor; the production mapping is exercised in WSL CI.
    from pathlib import PurePosixPath
    monkeypatch.setattr(community, 'Path', PurePosixPath)
    assert str(community._path('D:\\workspace\\.mindie-local\\community.json')) == '/mnt/d/workspace/.mindie-local/community.json'
    with pytest.raises(ValueError):
        community._path('\\\\host\\share\\community.json')
    with pytest.raises(ValueError):
        community._path('relative/community.json')
