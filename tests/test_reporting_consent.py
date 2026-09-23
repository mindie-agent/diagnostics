"""Consent transitions use real receipts, log files and queues; no live upload."""
import json
import os
import uuid

import pytest

from mindie_diagnostics import collect_bundle, configure
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.outbox import Outbox
from mindie_diagnostics.reporter import GitHub, publish_one


class NoNetwork:
    repository = 'example/project'

    def __getattr__(self, name):
        if name == 'timeout':
            return 30
        pytest.fail('unexpected external action: ' + name)


class Publisher:
    repository = 'example/project'
    timeout = 30

    def begin_cycle(self):
        pass

    def end_cycle(self):
        pass

    def __init__(self):
        self.calls = []

    def find_issue(self, item):
        self.calls.append('find')

    def create_issue(self, title, body):
        self.calls.append('post')
        self.body = body
        return {'number': 1, 'html_url': 'https://github.com/example/project/issues/1'}


_REAL_REQUEST = GitHub.request


def reporting_policy(path, enabled=True):
    from mindie_diagnostics.integration import configure_reporting
    result=configure_reporting(enabled, config=path, repository='example/project', roots=[str(path.parent)])
    assert result['status']=='configured'
    from mindie_diagnostics.reporting import current_consent
    return current_consent(path)


def reporting_failure(root, path):
    from unittest.mock import patch
    from mindie_diagnostics.integration import record_failure
    with patch.dict(os.environ, {'MINDIE_DIAGNOSTICS_CONFIG': str(path)}):
        return record_failure('mindie-knowledge','prepare',stage='validate',category='internal',root=root)


def test_default_and_explicit_disabled_keep_logs_but_never_enqueue(tmp_path):
    path=tmp_path/'reporting.json'
    reporting_failure(tmp_path,path)
    reporting_policy(path,False)
    reporting_failure(tmp_path,path)
    queue=Outbox(tmp_path/'queue.db')
    result=ingest(tmp_path,queue)
    assert result['consent_skipped']==2 and result['enqueued']==0
    assert collect_bundle(tmp_path)['summary']['error_count']>0
    assert publish_one(queue,NoNetwork())=={'status':'idle'}
    reporting_policy(path)
    assert ingest(tmp_path,Outbox(tmp_path/'fresh.db'))['enqueued']==0


def test_knowledge_consent_never_enrolls_fault_reporting(tmp_path, monkeypatch):
    receipt = tmp_path / 'community.json'
    receipt.write_text(json.dumps({'schema': 'mindie.community.v1', 'workspace_id': uuid.uuid4().hex,
                                   'revision': uuid.uuid4().hex, 'decision': 'enabled'}), encoding='utf-8')
    monkeypatch.setenv('MINDIE_COMMUNITY_POLICY', str(receipt))
    recorded = reporting_failure(tmp_path, tmp_path / 'missing-reporting.json')
    recorder = configure('test-community', root=tmp_path)
    with recorder.operation('prepare') as operation:
        operation.fail('transport', error_type='TimeoutError')
    recorder.close()
    events = [json.loads(line) for path in tmp_path.glob('events/*/*.jsonl*')
              for line in path.read_text(encoding='utf-8').splitlines()]
    assert recorded['recorded'] is True and operation.summary()['status'] == 'error'
    assert events and all('community' not in event and not event.get('reporting') for event in events)
    assert 'mindie.community.v1' not in json.dumps(events)
    assert ingest(tmp_path, Outbox(tmp_path / 'queue.db'))['enqueued'] == 0


def test_revoke_pending_and_reenable_never_reauthorizes_old_revision(tmp_path):
    from mindie_diagnostics.reporting import consent_allowed as reporting_allowed
    path=tmp_path/'reporting.json';reference=reporting_policy(path)
    reporting_failure(tmp_path,path)
    queue=Outbox(tmp_path/'queue.db');assert ingest(tmp_path,queue)['enqueued']==1
    reporting_policy(path,False)
    assert publish_one(queue,NoNetwork())=={'status':'withdrawn'}
    reporting_policy(path)
    assert publish_one(Outbox(queue.path),NoNetwork())=={'status':'idle'}
    assert not reporting_allowed(reference)
    assert ingest(tmp_path,Outbox(tmp_path/'fresh.db'))['enqueued']==0
    reporting_failure(tmp_path,path)
    assert ingest(tmp_path,queue)['enqueued']==1
    assert publish_one(queue,Publisher())['status']=='published'


def test_global_worker_keeps_authorization_local_and_public_marker_stable(tmp_path):
    first,second=tmp_path/'first.json',tmp_path/'second.json'
    reporting_policy(first);reporting_policy(second)
    reporting_failure(tmp_path,first);reporting_failure(tmp_path,second)
    queue=Outbox(tmp_path/'queue.db',capacity=2)
    assert ingest(tmp_path,queue)['enqueued']==2
    with queue.connect() as db:
        rows=[dict(row) for row in db.execute('SELECT * FROM incidents')]
    from mindie_diagnostics.reporter import fingerprint
    assert len({row['fingerprint'] for row in rows})==2
    assert len({fingerprint(json.loads(row['payload'])) for row in rows})==1
    reporting_policy(first,False)
    assert queue.withdraw_unconsented()==1
    publisher=Publisher();assert publish_one(queue,publisher)['status']=='published'
    assert publisher.calls==['find','post']
    assert {row['state'] for row in queue.rows()}=={'withdrawn','published'}
    assert str(second) not in publisher.body and 'config_file' not in publisher.body
    assert 'workspace_id' not in publisher.body and '"reporting"' not in publisher.body


def test_revocation_after_reconciliation_stops_issue_post(tmp_path):
    path=tmp_path/'reporting.json';reference=reporting_policy(path)
    queue=Outbox(tmp_path/'queue.db');queue.enqueue('a','b',{},consent=reference)
    class DuringRead(Publisher):
        def find_issue(self,item):
            reporting_policy(path,False)
        def create_issue(self,*args):
            pytest.fail('revoked issue must not be posted')
    assert publish_one(queue,DuringRead())=={'status':'withdrawn'}
    assert queue.rows()[0]['attempts']==1  # claim consumes before its GET phase


def test_each_github_reconciliation_page_rechecks_receipt(tmp_path,monkeypatch):
    path=tmp_path/'reporting.json';reference=reporting_policy(path)
    queue=Outbox(tmp_path/'queue.db');queue.enqueue('a','b',{},consent=reference)
    calls=[]
    def run(command,data,**kwargs):
        calls.append(command);reporting_policy(path,False)
        raw=b'HTTP/2 200 OK\r\ncontent-type: application/json\r\n\r\n'+json.dumps([{'body':''}]*100).encode()
        return 0,raw,len(raw)
    from mindie_diagnostics import reporter
    monkeypatch.setattr(GitHub,'request',_REAL_REQUEST)
    monkeypatch.setattr(reporter,'_run_gh',run)
    assert publish_one(queue,GitHub(repository='example/project'))=={'status':'withdrawn'}
    assert len(calls)==1


@pytest.mark.parametrize('state',['pending','retry','uncertain'])
def test_unscoped_outbox_cannot_use_process_consent(tmp_path,state):
    path=tmp_path/'reporting.json';reporting_policy(path)
    queue=Outbox(tmp_path/'queue.db');queue.enqueue('a','b',{})
    item=queue.claim();queue.update(item,state=state)
    assert publish_one(queue,NoNetwork())=={'status':'withdrawn'}
    assert queue.rows()[0]['attempts']==2
