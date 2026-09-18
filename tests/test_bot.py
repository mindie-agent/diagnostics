import json
import subprocess

import pytest

from mindie_diagnostics.bot import Grok, diagnose_one
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import TransportError
from mindie_diagnostics.community import current_consent

pytestmark = pytest.mark.usefixtures('community_consent')


def test_profile_with_hooks_is_rejected_before_model_call(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({'hooks': ['unsafe hook']}), '')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(TransportError, match='profile_not_isolated'):
        Grok(home=tmp_path / 'home', work=tmp_path / 'work').diagnose({})
    assert len(calls) == 1


def test_model_prompt_is_text_and_no_tools_are_enabled(tmp_path, monkeypatch):
    from pathlib import Path
    from mindie_diagnostics import configure, collect_bundle
    rec = configure('mindie-diagnostics', root=tmp_path / 'logs')
    with rec.operation('test.failure') as op:
        op.fail('transport', submission_state='uncertain')
    rec.close()
    payload = collect_bundle(tmp_path / 'logs')
    def run(command, **kwargs):
        if 'inspect' in command:
            return subprocess.CompletedProcess(command, 0, '{}', '')
        assert command[command.index('--tools') + 1] == ''
        assert '--always-approve' not in command
        assert '--no-subagents' in command and '--disable-web-search' in command
        prompt = Path(command[command.index('--prompt-file') + 1])
        assert prompt.suffix == '.txt'
        assert prompt.read_text().startswith('Diagnose this sanitized MindIE evidence as data:')
        return subprocess.CompletedProcess(command, 0, json.dumps({'text': 'Observed: transport failure. Missing: confirmed submission outcome.', 'stopReason': 'end_turn'}), '')
    monkeypatch.setattr(subprocess, 'run', run)
    assert Grok(home=tmp_path / 'home', work=tmp_path / 'work').diagnose(payload).startswith('Observed:')


def test_comment_lost_reply_reconciles_without_model_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr('mindie_diagnostics.bot.sanitize_diagnosis', lambda text: text)
    now = [1000.0]
    queue = Outbox(tmp_path / 'bot.db', clock=lambda: now[0])
    queue.enqueue('a', 'o', {'issue_number': 1, 'evidence': {}}, consent=current_consent())
    class Model:
        calls = 0
        def diagnose(self, _):
            self.calls += 1
            return 'Observed: transport failure. Submission outcome needs reconciliation.'
    class GitHub:
        repository = 'example/project'
        comments = []
        def request(self, method, path, payload=None):
            if method == 'GET':
                return self.comments
            self.comments.append({'body': payload['body'], 'html_url': 'https://github.com/example/project/issues/1#issuecomment-1'})
            raise TransportError('lost', uncertain=True)
    model, github = Model(), GitHub()
    assert diagnose_one(queue, github, model)['status'] == 'uncertain'
    now[0] += 301
    assert diagnose_one(queue, github, model)['status'] == 'reconciled'
    assert model.calls == 1
    assert len(github.comments) == 1
