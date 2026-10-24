from types import SimpleNamespace
import json

import pytest

from mindie_diagnostics import configure
from mindie_diagnostics import cli
from mindie_diagnostics.health import Health, read_health
from mindie_diagnostics.outbox import Outbox, QueueFull


def test_service_ensure_uses_the_atomic_owner_api(tmp_path, monkeypatch, capsys):
    from mindie_diagnostics import service
    calls = []
    monkeypatch.setattr(service, 'ensure_reporter_service',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {'status': 'installed'})
    config = str(tmp_path / 'reporting.json')
    assert cli.main(['service', 'ensure', '--root', str(tmp_path / 'logs'),
                     '--state', str(tmp_path / 'state'), '--python', 'installed-python',
                     '--reporting-config', config]) == 0
    assert calls[0][0][:2] == ([str(tmp_path / 'logs')], str(tmp_path / 'state'))
    assert calls[0][1]['python'] == 'installed-python'
    assert calls[0][1]['reporting_config'] == config
    for name in ('save_token', 'grok', 'grok_home', 'grok_work', 'central_bot'):
        assert name not in calls[0][1]
    assert 'installed' in capsys.readouterr().out


def test_removed_model_flags_and_command_are_rejected(tmp_path):
    rejected = [
        ['grok-profile', '--home', str(tmp_path / 'home')],
        ['worker', '--state', str(tmp_path / 'state'), '--root', str(tmp_path)],
        ['worker', '--state', str(tmp_path / 'state'), '--reporting-config', str(tmp_path / 'p.json'),
         '--grok', 'grok', '--grok-home', str(tmp_path), '--grok-work', str(tmp_path)],
        ['worker', '--state', str(tmp_path / 'state'), '--reporting-config', str(tmp_path / 'p.json'),
         '--central-bot'],
        ['service', 'ensure', '--root', str(tmp_path), '--state', str(tmp_path / 'state'),
         '--reporting-config', str(tmp_path / 'p.json'), '--save-token'],
        ['service', 'install', '--root', str(tmp_path), '--state', str(tmp_path / 'state'),
         '--reporting-config', str(tmp_path / 'p.json'), '--grok', 'grok'],
        ['service', 'install', '--root', str(tmp_path), '--state', str(tmp_path / 'state'),
         '--reporting-config', str(tmp_path / 'p.json'), '--grok-home', str(tmp_path)],
        ['service', 'ensure', '--root', str(tmp_path), '--state', str(tmp_path / 'state'),
         '--central-bot', '--reporting-config', str(tmp_path / 'p.json')],
        ['service', 'install', '--root', str(tmp_path), '--state', str(tmp_path / 'state')],
    ]
    for argv in rejected:
        with pytest.raises(SystemExit) as error:
            cli.parser().parse_args(argv)
        assert error.value.code == 2
    assert not any(tmp_path.iterdir())


def test_service_status_and_remove_remain_available():
    for action in ('status', 'remove'):
        args = cli.parser().parse_args(['service', action])
        assert args.command == 'service' and args.action == action


def test_service_failure_status_returns_nonzero(monkeypatch, capsys):
    from mindie_diagnostics import service
    for status, expected in (('active', 0), ('failed', 1), ('unknown', 1), ('error', 1)):
        monkeypatch.setattr(service, 'service_status', lambda status=status: {'status': status})
        assert cli.main(['service', 'status']) == expected
        assert json.loads(capsys.readouterr().out) == {'status': status}


def test_reporting_maintain_forwards_optional_flags(monkeypatch, capsys):
    from mindie_diagnostics import reporting_commands
    seen = []

    def maintain(config=None, *, update_running=False, unit_dir=None, budget_seconds=75):
        seen.append((config, update_running, unit_dir, budget_seconds))
        return {'status': 'ok'}

    monkeypatch.setattr(reporting_commands, 'maintain', maintain)
    assert cli.main(['reporting', 'maintain', '--config', '/abs/reporting.json']) == 0
    assert cli.main(['reporting', 'maintain', '--update-running', '--unit-dir', '/abs/units', '--budget-seconds', '4']) == 0
    assert seen == [('/abs/reporting.json', False, None, 75), (None, True, '/abs/units', 4)]
    assert 'ok' in capsys.readouterr().out


def test_status_reports_reporter_queue_only(tmp_path, capsys):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'bot.sqlite3').write_text('not a database', encoding='utf-8')
    (state / 'central-bot.sqlite3').write_text('not a database', encoding='utf-8')
    assert cli.main(['status', '--state', str(state)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {'worker', 'reporter'}
    assert payload['reporter'] == []


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
        assert 'bot' not in result
        assert not read_health(tmp_path / 'state')['healthy']


class CyclePublisher:
    """Local publication boundary: records calls, never touches the network."""

    timeout = 30

    def __init__(self, repository):
        self.repository, self.calls = repository, []

    def begin_cycle(self):
        pass

    def end_cycle(self):
        pass

    def find_issue(self, item):
        self.calls.append('find')

    def create_issue(self, title, body):
        self.calls.append('post')
        return {'number': 1, 'html_url': f'https://github.com/{self.repository}/issues/1'}


def test_worker_stop_during_intake_admits_no_later_stage(tmp_path, monkeypatch, reporting_consent):
    import threading
    from mindie_diagnostics import maintenance
    from mindie_diagnostics.integration import record_failure

    roots = [tmp_path / 'logs-a', tmp_path / 'logs-b']
    for root, operations in zip(roots, (('drain.first', 'drain.second'), ('drain.third',))):
        for operation in operations:
            assert record_failure('mindie-knowledge', operation, stage='connect',
                                  category='internal', root=str(root))['recorded']
    queue = Outbox(tmp_path / 'state' / 'queue.db')

    class StopAfterFirstRecord(threading.Event):
        """The owner's real Event, tripped once intake queued one record."""

        def is_set(self):
            if not super().is_set() and queue.rows():
                self.set()
            return super().is_set()

    started = []
    real_ingest, real_prune, real_publish = cli.ingest, maintenance.prune, cli.publish_one
    monkeypatch.setattr(cli, 'ingest', lambda root, *a, **kw: started.append(('ingest', root)) or real_ingest(root, *a, **kw))
    monkeypatch.setattr(maintenance, 'prune', lambda root, **kw: started.append(('retention', root)) or real_prune(root, **kw))
    monkeypatch.setattr(cli, 'publish_one', lambda *a: started.append(('report', None)) or real_publish(*a))
    publisher = CyclePublisher(reporting_consent['repository'])
    args = SimpleNamespace(root=[str(root) for root in roots],
                           reporting_config=reporting_consent['config_file'])
    recorder = configure('stop-cycle', root=tmp_path / 'state' / 'diagnostics')
    with Health(tmp_path / 'state', recorder) as health:
        stopped = cli.run_cycle(args, queue, publisher, recorder, health,
                                cancel=StopAfterFirstRecord())
        # Intake of the first root is the only stage this cycle ever started.
        assert started == [('ingest', str(roots[0]))]
        assert publisher.calls == []
        assert (stopped['status'], stopped['limited']) == ('degraded', 1)
        assert (stopped['ingestion'][0]['enqueued'], stopped['ingestion'][0]['limited']) == (1, 1)
        assert stopped['retention'] == [] and stopped['reporter'] is None
        # The queued record and its cursor remainder are both retained.
        assert [row['state'] for row in queue.rows()] == ['pending']
        segment = next((roots[0] / 'events' / 'mindie-knowledge').glob('*.jsonl'))
        with queue.connect() as db:
            cursors = {row['path']: row['offset'] for row in db.execute('SELECT path, offset FROM cursors')}
        assert 0 < cursors[str(segment)] < segment.stat().st_size
        assert len(cursors) == 1

        # A cycle entered with the stop already set admits no stage at all and
        # therefore leaves that remainder untouched.
        started.clear()
        requested = threading.Event()
        requested.set()
        idle = cli.run_cycle(args, queue, publisher, recorder, health, cancel=requested)
        assert started == [] and idle['reporter'] is None
        assert (idle['status'], idle['limited']) == ('degraded', 1)

        started.clear()
        resumed = cli.run_cycle(args, queue, publisher, recorder, health, cancel=threading.Event())
    assert [stage for stage, _ in started] == ['ingest', 'retention', 'ingest', 'retention', 'report']
    assert sum(entry['enqueued'] for entry in resumed['ingestion']) == 2
    assert len(queue.rows()) == 3 and resumed['reporter']['status'] == 'published'
    assert publisher.calls == ['find', 'post']


def test_failed_publication_terminal_is_visible_in_health(tmp_path, monkeypatch):
    recorder = configure('terminal-cycle', root=tmp_path/'logs')
    queue = Outbox(tmp_path/'state'/'queue.db')
    for terminal in ('exhausted', 'permanent-failed'):
        monkeypatch.setattr(cli, 'publish_one', lambda *args: {'status': terminal})
        with Health(tmp_path/'state', recorder) as health:
            result = cli.run_cycle(SimpleNamespace(root=[]), queue, object(), recorder, health)
            assert result['status'] == 'degraded'
            assert not read_health(tmp_path/'state')['healthy']
