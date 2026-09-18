import hashlib
import json
from types import SimpleNamespace

import pytest

from mindie_diagnostics import bind_community_policy, collect_bundle, configure
from mindie_diagnostics.bot import diagnose_one, enqueue_public_issues
from mindie_diagnostics.cli import main, run_cycle
from mindie_diagnostics.health import Health
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import issue_payload


def public_issue(tmp_path):
    with bind_community_policy(None):
        recorder = configure('central-test', root=tmp_path / 'logs')
        with recorder.operation('test.failure') as op:
            op.fail('transport')
        recorder.close()
    payload = issue_payload(collect_bundle(tmp_path / 'logs'))
    return {'number': 1, 'body': '<!-- mindie-incident:' + 'a' * 64 + ' -->\n```json\n' + json.dumps(payload) + '\n```'}, payload


class Model:
    calls = 0

    def diagnose(self, evidence):
        self.calls += 1
        return 'Observed failure. Root cause requires more evidence.'


class Git:
    repository = 'example/project'

    def __init__(self, issue):
        self.issue, self.comments, self.calls = issue, [], []

    def request(self, method, path, payload=None):
        self.calls.append((method, path))
        if method == 'POST':
            row = {'body': payload['body'], 'html_url': 'https://github.com/example/project/issues/1#issuecomment-1'}
            self.comments.append(row)
            return row
        return self.comments if '/comments?' in path else [self.issue]


def test_central_authority_is_explicit_and_cannot_be_inferred_from_issue(tmp_path):
    issue, _ = public_issue(tmp_path)
    github, model = Git(issue), Model()
    queue = Outbox(tmp_path / 'central.db')
    assert enqueue_public_issues(github, queue) == 1
    assert enqueue_public_issues(github, queue) == 0
    count = len(github.calls)
    # A normal worker cannot consume a central queue marker as local consent.
    assert diagnose_one(queue, github, model) == {'status': 'withdrawn'}
    assert len(github.calls) == count and model.calls == 0


def test_explicit_central_mode_diagnoses_public_evidence_without_local_consent(tmp_path):
    issue, _ = public_issue(tmp_path)
    github, model = Git(issue), Model()
    queue = Outbox(tmp_path / 'central.db')
    assert enqueue_public_issues(github, queue) == 1
    assert diagnose_one(queue, github, model, public_repository=github.repository)['status'] == 'published'
    assert model.calls == 1 and len(github.comments) == 1
    assert 'mindie-grok-evidence:' in github.comments[0]['body']


def test_fresh_state_reconciles_published_comment_before_model_call(tmp_path):
    issue, _ = public_issue(tmp_path)
    github, model = Git(issue), Model()
    first_queue = Outbox(tmp_path / 'first-central.db')
    enqueue_public_issues(github, first_queue)
    assert diagnose_one(first_queue, github, model, public_repository=github.repository)['status'] == 'published'
    assert model.calls == 1
    queue = Outbox(tmp_path / 'fresh-central.db')
    enqueue_public_issues(github, queue)
    assert diagnose_one(queue, github, model, public_repository=github.repository)['status'] == 'reconciled'
    assert model.calls == 1 and len(github.comments) == 1


def test_local_and_central_queues_share_evidence_marker(tmp_path):
    issue, payload = public_issue(tmp_path)
    github, model = Git(issue), Model()
    evidence_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    github.comments.append({'body': f'<!-- mindie-grok-diagnosis:{"b" * 64} -->\n<!-- mindie-grok-evidence:{evidence_hash} -->',
                            'html_url': 'https://github.com/example/project/issues/1#issuecomment-1'})
    queue = Outbox(tmp_path / 'fresh-central.db')
    enqueue_public_issues(github, queue)
    assert diagnose_one(queue, github, model, public_repository=github.repository)['status'] == 'reconciled'
    assert model.calls == 0


@pytest.mark.parametrize('body', ['not diagnostic', '<!-- mindie-incident:' + 'a'*64 + ' -->\n```json\n[]\n```',
                                     '<!-- mindie-incident:fake -->\n```json\n{}\n```'])
def test_central_skips_unstructured_or_malicious_issue(tmp_path, body):
    queue = Outbox(tmp_path / 'central.db')
    assert enqueue_public_issues(Git({'number': 1, 'body': body}), queue) == 0


def test_central_cycle_never_touches_local_logs_or_report_queue(tmp_path, monkeypatch):
    from mindie_diagnostics import cli
    issue, _ = public_issue(tmp_path)
    github, model = Git(issue), Model()
    queue = Outbox(tmp_path / 'central.db')
    monkeypatch.setattr(cli, 'ingest', lambda *a, **k: pytest.fail('central mode ingested local logs'))
    monkeypatch.setattr(cli, 'publish_one', lambda *a, **k: pytest.fail('central mode uploaded local issue'))
    recorder = configure('central-worker', root=tmp_path / 'worker-logs')
    with Health(tmp_path / 'state', recorder) as health:
        result = run_cycle(SimpleNamespace(root=[], central_bot=True), None, github, recorder, health, model, queue)
    assert result['reporter'] is None and result['ingestion'] == []
    assert result['bot']['status'] == 'published'


@pytest.mark.parametrize('arguments', [[], ['--central-bot'], ['--central-bot', '--root', 'local']])
def test_worker_rejects_ambiguous_or_missing_mode_inputs(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        main(['worker', '--state', str(tmp_path / 'state'), *arguments])
    assert error.value.code == 2
