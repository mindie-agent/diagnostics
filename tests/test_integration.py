"""Local diagnostic visibility and consent invariants without a publisher."""
import json
import os
import time
from pathlib import Path

import pytest
from mindie_diagnostics import fallback
from mindie_diagnostics.integration import configure_reporting, record_failure, reporting_status, _worker_view


def test_local_fault_visible_without_upload_and_after_disable(tmp_path, monkeypatch):
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_ROOT', str(tmp_path / 'logs'))
    ref = record_failure('mindie-knowledge', 'observe', stage='end', category='internal',
                         exception=RuntimeError('PRIVATE_BUSINESS_VALUE'))
    assert ref['recorded']
    path=next((tmp_path/'logs/events/mindie-knowledge').glob('*.jsonl'))
    event=json.loads(path.read_text().splitlines()[-1])
    assert event['process_instance_id'] != '0'*32
    assert event['process_instance_id'] in path.name
    status = reporting_status()
    assert status['status'] == 'not_configured'
    assert status['local']['recent'][0]['incident_id'] == ref['incident_id']
    assert not (tmp_path / 'config.json').exists()
    assert configure_reporting(True)['status'] == 'configured'
    receipt = fallback.current_consent()
    assert fallback.consent_allowed(receipt)
    configure_reporting(False)
    assert not fallback.consent_allowed(receipt)
    status = reporting_status()
    assert not status['enabled']
    assert status['local']['recent'][0]['incident_id'] == ref['incident_id']
    assert 'PRIVATE_BUSINESS_VALUE' not in json.dumps(status)
    assert not fallback.state_path().exists()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process probe and FIFO')
def test_unsafe_config_and_dead_health_do_not_block_or_look_running(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_CONFIG', str(config))
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_ROOT', str(tmp_path / 'logs'))
    os.mkfifo(config)
    started = time.monotonic()
    assert configure_reporting(True)['status'] == 'error'
    assert reporting_status()['status'] == 'configuration_unavailable'
    assert time.monotonic() - started < .5
    assert config.is_fifo()
    config.unlink()
    configure_reporting(False)
    state = fallback.state_path();state.mkdir(mode=0o700)
    health = {'schema': 1, 'pid': 99999999, 'status': 'running', 'stage': 'idle',
              'heartbeat_at': time.time(), 'progress_at': time.time(), 'interval': 60}
    (state / 'worker-health.json').write_text(json.dumps(health))
    assert _worker_view(None)['status'] == 'stopped'
    health.update(pid=os.getpid(), heartbeat_at=time.time()-61)
    (state / 'worker-health.json').write_text(json.dumps(health))
    assert _worker_view(None)['status'] == 'stale'


def test_retention_block_does_not_allocate_fake_incident(tmp_path, monkeypatch):
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_ROOT', str(tmp_path))
    marker=tmp_path / 'retention-state.json'
    marker.write_text('{"schema":1,"blocked":true}');marker.chmod(0o600)
    result=record_failure('mindie-knowledge','observe',stage='end',category='internal')
    assert not result['recorded'] and result['logging_failed'] and result['incident_id'] is None
    assert not list((tmp_path / 'events').glob('*/*.jsonl*'))
