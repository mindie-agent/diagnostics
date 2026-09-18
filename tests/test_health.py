import json

from mindie_diagnostics import configure
from mindie_diagnostics.health import Health, read_health


def test_health_distinguishes_liveness_progress_and_degradation(tmp_path):
    now = [1000]
    log = configure('health-test', root=tmp_path / 'logs')
    with Health(tmp_path, log, clock=lambda: now[0]) as health:
        assert read_health(tmp_path, clock=lambda: now[0])['healthy']
        now[0] += 61
        assert read_health(tmp_path, clock=lambda: now[0])['stale']
        health.update(status='degraded', stage='report', error_type='QueueFull')
        assert not read_health(tmp_path, clock=lambda: now[0])['healthy']
        health.update(status='running')
        now[0] += 1201
        health.write()  # Heartbeat alone does not prove useful progress.
        assert read_health(tmp_path, clock=lambda: now[0])['progress_stalled']
    assert read_health(tmp_path, clock=lambda: now[0])['status'] == 'stopped'


def test_health_write_failure_does_not_mask_original_failure(tmp_path):
    path = tmp_path / 'blocked'
    path.write_text('not a directory')
    log = configure('health-failure-test', root=tmp_path / 'logs')
    with Health(path, log) as health:
        health.update(stage='ingest')
    assert read_health(tmp_path)['status'] == 'not_started'
    events = [json.loads(line) for file in (tmp_path / 'logs').rglob('*.jsonl') for line in file.read_text().splitlines()]
    assert any(row['event'] == 'worker.health_unavailable' for row in events)
