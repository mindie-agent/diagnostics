"""Updater faults travel through the real recorder, consent gate and outbox."""
import json
import subprocess
import sys
import urllib.error

import pytest

from mindie_diagnostics import fallback
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.integration import configure_reporting, record_failure, reporting_status
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, publish_one


def record(root, error):
    return record_failure('mindie-agent-codex', 'release.update',
                          stage='component-inheritance', category='update',
                          version='0.1.5', revision='a' * 40, exception=error,
                          exit_code=getattr(error, 'returncode', None), root=root)


def test_isolated_helper_failure_reaches_one_sanitized_issue(tmp_path, reporting_consent):
    child = subprocess.run([sys.executable, '-I', '-c', 'import missing_mindie_runtime_dependency'],
                           capture_output=True, text=True, timeout=5)
    assert child.returncode != 0 and 'ModuleNotFoundError' in child.stderr
    error = subprocess.CalledProcessError(child.returncode, ['PRIVATE-COMMAND'],
                                         output='PRIVATE-STDOUT', stderr=child.stderr)
    refs = [record(tmp_path, error) for _ in range(2)]
    queue = Outbox(tmp_path / 'queue.db')
    assert ingest(tmp_path, queue, config=reporting_consent['config_file'])['enqueued'] == 2
    assert len(queue.rows()) == 1
    assert set(queue.rows()[0]['incident_ids']) == {ref['incident_id'] for ref in refs}
    writes = []

    class Boundary(GitHub):
        def find_issue(self, item):
            return None

        def create_issue(self, title, body):
            writes.append((title, body))
            return {'number': 1, 'html_url': f'https://github.com/{self.repository}/issues/1'}

    assert publish_one(queue, Boundary(reporting_consent['repository']))['status'] == 'published'
    assert publish_one(queue, Boundary(reporting_consent['repository']))['status'] == 'idle'
    assert len(writes) == 1
    public = json.dumps(writes)
    for private in ('PRIVATE-COMMAND', 'PRIVATE-STDOUT', child.stderr, str(tmp_path), refs[0]['incident_id']):
        assert private not in public
    assert 'component-inheritance' in public and 'exit_code' in public and '0.1.5' in public


@pytest.mark.parametrize('error', [
    urllib.error.URLError('PRIVATE-TOKEN'), TimeoutError('PRIVATE-TOKEN'),
    PermissionError('PRIVATE-TOKEN'), FileNotFoundError('PRIVATE-TOKEN'),
])
def test_environment_failures_stay_visible_without_issue(tmp_path, reporting_consent, error):
    ref = record(tmp_path, error)
    queue = Outbox(tmp_path / 'queue.db')
    assert ingest(tmp_path, queue, config=reporting_consent['config_file'])['enqueued'] == 0
    assert queue.rows() == []
    view = reporting_status(config=reporting_consent['config_file'])['local']['recent'][0]
    assert view['incident_id'] == ref['incident_id']
    assert view['reporting_eligibility'] == 'local_only'
    assert 'PRIVATE-TOKEN' not in json.dumps(view)


def test_no_backfill_and_revocation_remain_visible(tmp_path, monkeypatch):
    config, root = tmp_path / 'policy.json', tmp_path / 'logs'
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_CONFIG', str(config))
    first = record(root, ModuleNotFoundError('PRIVATE-MODULE'))
    configure_reporting(True, config=config, roots=[str(root)])
    view = reporting_status(config=config)['local']['recent'][0]
    assert view['incident_id'] == first['incident_id']
    assert view['reporting_eligibility'] == 'not_authorized_at_capture'
    second = record(root, ModuleNotFoundError('PRIVATE-MODULE'))
    assert reporting_status(config=config)['local']['recent'][0]['reporting_eligibility'] == 'consent_allowed'
    configure_reporting(False, config=config)
    assert reporting_status(config=config)['local']['recent'][0]['reporting_eligibility'] == 'consent_withdrawn'
    queue = Outbox(tmp_path / 'queue.db')
    assert ingest(root, queue, config=config)['enqueued'] == 0
    assert first['incident_id'] != second['incident_id']


def test_standalone_recorder_uses_the_same_update_policy(tmp_path, reporting_consent):
    fallback.record_failure('mindie-agent-cc', 'runtime.prepare', stage='validation',
                            category='update', exception=ModuleNotFoundError('PRIVATE'), root=tmp_path)
    queue = Outbox(tmp_path / 'queue.db')
    assert ingest(tmp_path, queue, config=reporting_consent['config_file'])['enqueued'] == 1
    assert queue.claim()['payload']['events'][-1]['attributes']['error_code'] == 'runtime_dependency_missing'
