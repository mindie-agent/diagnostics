"""Support bundle and independent reporter/bot worker commands."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import signal
import threading
from pathlib import Path

from .outbox import Outbox
from .reporter import DEFAULT_REPOSITORY, GitHub, ingest, publish_one


def run_cycle(args, queue, github, recorder, health, grok=None, bot_queue=None, *, since=None):
    """Independent stages: a full intake queue must still be able to drain."""
    result = {'ingestion': [], 'retention': [], 'reporter': None, 'bot': None, 'status': 'ok'}
    with recorder.operation('worker.cycle') as operation:
        def attempt(stage, function):
            health.update(status='running', stage=stage)
            try:
                with operation.phase(stage):
                    value = function()
                if value.get('status') in {'retry', 'uncertain', 'blocked', 'rate_limited'} or value.get('limited'):
                    result['status'] = 'degraded'
                    operation.event('WARNING', 'worker.stage_degraded', stage=stage,
                                    error_code=value.get('error'))
                return value
            except Exception as exc:
                result['status'] = 'degraded'
                operation.fail('diagnostics', exception=exc)
                return {'status': 'degraded', 'error_type': type(exc).__name__}

        if getattr(args, 'central_bot', False):
            from .bot import diagnose_one, enqueue_public_issues
            result['bot_ingestion'] = attempt('bot.public_ingest', lambda: {'enqueued': enqueue_public_issues(github, bot_queue)})
            result['bot'] = attempt('diagnose', lambda: diagnose_one(bot_queue, github, grok, public_repository=github.repository))
        else:
            from .maintenance import prune
            result['consent'] = attempt('consent', lambda: {'withdrawn': queue.withdraw_unconsented()})
            if bot_queue is not None:
                result['bot_consent'] = attempt('bot.consent', lambda: {'withdrawn': bot_queue.withdraw_unconsented()})
            for root in args.root:
                result['ingestion'].append(attempt('ingest', lambda: ingest(root, queue, since=since)))
                result['retention'].append(attempt('retention', lambda: prune(root, queue=queue)))
            result['reporter'] = attempt('report', lambda: publish_one(queue, github))
        if grok and bot_queue and not getattr(args, 'central_bot', False):
            from .bot import diagnose_one, enqueue_issues
            # Intake failure must not prevent already queued diagnoses.
            result['bot_ingestion'] = attempt('bot.ingest', lambda: {'enqueued': enqueue_issues(github, bot_queue, source=queue)})
            result['bot'] = attempt('diagnose', lambda: diagnose_one(bot_queue, github, grok))
    health.update(status='idle' if result['status'] == 'ok' else 'degraded', stage='waiting', last_cycle=result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="mindie-diagnostics", description="Local diagnostics and independently enabled automatic issue reporting")
    sub = result.add_subparsers(dest="command", required=True)
    bundle = sub.add_parser("bundle", help="export a bounded sanitized support bundle, without uploading")
    bundle.add_argument("--root", required=True)
    bundle.add_argument("--operation-id")
    bundle.add_argument("--output")
    status = sub.add_parser("status", help="inspect the local publishing queue without network access")
    status.add_argument("--state", required=True)
    profile = sub.add_parser("grok-profile", help="create a dedicated tool-disabled Grok profile; existing personal config is never replaced")
    profile.add_argument("--home", required=True)
    service = sub.add_parser('service', help='explicitly install, inspect or remove a supervised user worker')
    service_sub = service.add_subparsers(dest='action', required=True)
    install = argparse.ArgumentParser(add_help=False)
    install.add_argument('--root', action='append', default=[])
    install.add_argument('--state', required=True)
    install.add_argument('--repository', default=DEFAULT_REPOSITORY)
    install.add_argument('--python')
    install.add_argument('--gh')
    install.add_argument('--grok')
    install.add_argument('--grok-home')
    install.add_argument('--grok-work')
    install.add_argument('--interval', type=float, default=60)
    install.add_argument('--since')
    install.add_argument('--environment-file', help='optional private 0600 systemd environment file; contents are never logged')
    install.add_argument('--save-token', action='store_true', help='explicitly save the current GitHub token into a private worker credential file')
    install.add_argument('--no-start', action='store_true')
    install.add_argument('--central-bot', action='store_true', help='maintainer mode: diagnose already-public issues; do not ingest or upload local logs')
    service_sub.add_parser('install', parents=[install])
    service_sub.add_parser('ensure', parents=[install], help='add roots to the owned local reporter while retaining state and credentials')
    service_sub.add_parser('status')
    service_sub.add_parser('remove')
    worker = sub.add_parser("worker", help="enable local failure reporting and optional Grok diagnosis")
    worker.add_argument("--root", action="append", default=[], help="explicit diagnostic root; repeat for multiple components/workspaces")
    worker.add_argument("--state", required=True)
    worker.add_argument("--repository", default=DEFAULT_REPOSITORY)
    worker.add_argument("--gh", default="gh")
    worker.add_argument("--grok", help="optional Grok executable; uses a dedicated existing authenticated profile")
    worker.add_argument("--grok-home")
    worker.add_argument("--grok-work")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--interval", type=float, default=60)
    worker.add_argument("--since", help="optional fixed UTC start timestamp; older logs remain local")
    worker.add_argument('--central-bot', action='store_true', help='explicit maintainer authorization to diagnose public issues in a separate state directory')
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == 'service':
        from .service import install_service, ensure_reporter_service, service_status, remove_service, ServiceError
        try:
            if args.action in {'install', 'ensure'}:
                action = ensure_reporter_service if args.action == 'ensure' else install_service
                result = action(args.root, args.state, args.repository, python=args.python, gh=args.gh,
                                         grok=args.grok, grok_home=args.grok_home, grok_work=args.grok_work,
                                         interval=args.interval, since=args.since, environment_file=args.environment_file,
                                         save_token=args.save_token,
                                         central_bot=args.central_bot,
                                         start=not args.no_start)
            else:
                result = service_status() if args.action == 'status' else remove_service()
        except ServiceError as exc:
            print(json.dumps({'status': 'error', 'category': exc.category, 'action': exc.action,
                              'returncode': exc.returncode}), flush=True)
            return 1
        print(json.dumps(result, ensure_ascii=True))
        return 0
    if args.command == "bundle":
        from .bundle import collect_bundle
        print(json.dumps(collect_bundle(args.root, operation_id=args.operation_id, output=args.output), ensure_ascii=True))
        return 0
    if args.command == "grok-profile":
        from .bot import prepare_profile
        home = prepare_profile(args.home)
        print(json.dumps({"home": str(home), "authentication": "Run grok login with GROK_HOME set to this directory."}))
        return 0
    state = Path(args.state).resolve()
    if args.command == "status":
        from .health import read_health
        result = {'worker': read_health(state)}
        for name in ("reporter", "bot", "central-bot"):
            path = state / f"{name}.sqlite3"
            result[name] = Outbox(path).rows() if path.exists() else []
        print(json.dumps(result, ensure_ascii=True))
        return 0
    if args.interval < 5:
        parser().error("worker interval must be at least 5 seconds")
    since = None
    if args.since:
        try:
            parsed = datetime.fromisoformat(args.since.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                raise ValueError('timezone required')
            since = parsed.timestamp()
        except ValueError:
            parser().error('--since must be an ISO timestamp with a timezone')
    if args.grok and not (args.grok_home and args.grok_work):
        parser().error("--grok requires --grok-home and --grok-work")
    if args.central_bot and (args.root or not args.grok):
        parser().error('--central-bot requires --grok and does not accept local --root inputs')
    if not args.central_bot and not args.root:
        parser().error('a local reporting worker requires --root')
    from . import configure, __version__
    recorder = configure("mindie-diagnostics", root=state / "diagnostics", version=__version__)
    queue = None if args.central_bot else Outbox(state / "reporter.sqlite3")
    github = GitHub(args.repository, executable=args.gh)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    grok = bot_queue = None
    if args.grok:
        from .bot import Grok
        grok, bot_queue = Grok(args.grok, home=args.grok_home, work=args.grok_work), Outbox(state / ('central-bot.sqlite3' if args.central_bot else 'bot.sqlite3'))
    from .health import Health
    with Health(state, recorder, interval=args.interval) as health:
        while not stop.is_set():
            result = run_cycle(args, queue, github, recorder, health, grok, bot_queue, since=since)
            print(json.dumps(result, ensure_ascii=True), flush=True)
            if args.once:
                return int(result['status'] != 'ok')
            stop.wait(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
