from types import SimpleNamespace

from mindie_diagnostics import configure
from mindie_diagnostics import cli
from mindie_diagnostics.health import Health, read_health
from mindie_diagnostics.outbox import Outbox, QueueFull


def test_service_ensure_uses_the_atomic_owner_api(tmp_path, monkeypatch, capsys):
    from mindie_diagnostics import service
    calls = []
    monkeypatch.setattr(service, 'ensure_reporter_service',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {'status': 'installed'})
    assert cli.main(['service', 'ensure', '--root', str(tmp_path / 'logs'),
                     '--state', str(tmp_path / 'state'), '--python', 'installed-python', '--save-token']) == 0
    assert calls[0][0][:2] == ([str(tmp_path / 'logs')], str(tmp_path / 'state'))
    assert calls[0][1]['python'] == 'installed-python' and calls[0][1]['save_token'] is True
    assert 'installed' in capsys.readouterr().out


def test_full_intake_queue_does_not_prevent_publication(tmp_path, monkeypatch):
    recorder = configure('cycle-test', root=tmp_path / 'logs')
    queue = Outbox(tmp_path / 'state' / 'queue.db')
    calls = []
    def ingest(*args, **kwargs):
        raise QueueFull('private details never printed')
    monkeypatch.setattr(cli, 'ingest', ingest)
    monkeypatch.setattr(cli, 'publish_one', lambda *args: calls.append('published') or {'status': 'published'})
    with Health(tmp_path / 'state', recorder) as health:
        result = cli.run_cycle(SimpleNamespace(root=[str(tmp_path / 'logs')]), queue, object(), recorder, health)
        assert calls == ['published'] and result['status'] == 'degraded'
        assert result['ingestion'][0] == {'status': 'degraded', 'error_type': 'QueueFull'}
        assert not read_health(tmp_path / 'state')['healthy']
