"""Support bundle and independent reporter worker commands."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import signal
import threading
from pathlib import Path

from .outbox import Outbox
from .reporter import DEFAULT_REPOSITORY, GitHub, ingest, publish_one


def run_cycle(args, queue, github, recorder, health, *, since=None):
    """Independent stages: a full intake queue must still be able to drain."""
    result = {'ingestion': [], 'retention': [], 'reporter': None, 'status': 'ok'}
    with recorder.operation('worker.cycle') as operation:
        def attempt(stage, function):
            health.update(status='running', stage=stage)
            try:
                with operation.phase(stage):
                    value = function()
                if value.get('status') in {'recording_failed', 'retry', 'uncertain', 'blocked', 'rate_limited', 'exhausted', 'permanent-failed'} or value.get('limited'):
                    result['status'] = 'degraded'
                    operation.event('WARNING', 'worker.stage_degraded', stage=stage,
                                    error_code=value.get('error'))
                return value
            except Exception as exc:
                result['status'] = 'degraded'
                operation.fail('diagnostics', exception=exc)
                return {'status': 'degraded', 'error_type': type(exc).__name__}

        from .maintenance import prune
        queue.maintain()
        result['consent'] = attempt('consent', lambda: {'withdrawn': queue.withdraw_unconsented()})
        for root in args.root:
            result['ingestion'].append(attempt('ingest', lambda: ingest(root, queue, since=since)))
            result['retention'].append(attempt('retention', lambda: prune(root, queue=queue)))
        result['reporter'] = attempt('report', lambda: publish_one(queue, github))
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
    service = sub.add_parser('service', help='explicitly install, inspect or remove a supervised user worker')
    service_sub = service.add_subparsers(dest='action', required=True)
    install = argparse.ArgumentParser(add_help=False)
    install.add_argument('--root', action='append', default=[])
    install.add_argument('--state', required=True)
    install.add_argument('--repository', default=DEFAULT_REPOSITORY)
    install.add_argument('--python')
    install.add_argument('--gh')
    install.add_argument('--interval', type=float, default=60)
    install.add_argument('--since')
    install.add_argument('--environment-file', help='optional private 0600 systemd environment file; contents are never logged')
    install.add_argument('--no-start', action='store_true')
    install.add_argument('--reporting-config', required=True, help='independent reporting policy; required')
    service_sub.add_parser('install', parents=[install])
    service_sub.add_parser('ensure', parents=[install], help='add roots to the owned local reporter while retaining state and credentials')
    service_sub.add_parser('status')
    service_sub.add_parser('remove')
    worker = sub.add_parser("worker", help="run the local pure reporter under an independent reporting policy")
    worker.add_argument("--root", action="append", default=[], help="accepted on the command line; each cycle uses the reporting policy roots")
    worker.add_argument("--state", required=True)
    worker.add_argument("--repository", default=DEFAULT_REPOSITORY)
    worker.add_argument("--gh", default="gh")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--interval", type=float, default=60)
    worker.add_argument("--since", help="optional fixed UTC start timestamp; older logs remain local")
    worker.add_argument('--reporting-config', required=True, help='independent reporting policy; required')
    reporting = sub.add_parser('reporting', help='independent local fault logging and optional public reporting')
    actions = reporting.add_subparsers(dest='action', required=True)
    for name in ('status', 'configure', 'ensure', 'maintain'):
        command = actions.add_parser(name)
        command.add_argument('--config')
        if name == 'ensure':
            command.add_argument('--unit-dir', help='explicit owned service directory for isolated installation')
        if name == 'maintain':
            command.add_argument('--update-running', action='store_true',
                                 help='inspect an already enabled healthy worker for a higher runtime version; does not start a stopped worker')
            command.add_argument('--unit-dir', help='owned service directory for that inspection')
            command.add_argument('--budget-seconds', type=float, default=75,
                                 help='absolute seconds shared by offline maintenance and the optional handoff; capped at 75')
        if name == 'configure':
            command.add_argument('--enabled', choices=('true', 'false'), required=True)
            command.add_argument('--repository', default=DEFAULT_REPOSITORY)
            command.add_argument('--root', action='append')
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == 'reporting':
        from .integration import reporting_status, configure_reporting
        from .reporting_commands import ensure, maintain
        try:
            if args.action == 'status':
                result = reporting_status(config=args.config)
            elif args.action == 'configure':
                result = configure_reporting(args.enabled == 'true', repository=args.repository,
                                             config=args.config, roots=args.root)
            elif args.action == 'ensure':
                result = ensure(args.config, unit_dir=args.unit_dir)
            else:
                result = maintain(args.config, update_running=args.update_running, unit_dir=args.unit_dir,
                                  budget_seconds=args.budget_seconds)
        except Exception as exc:
            result = {'status': 'degraded', 'category': 'reporting_operation_failed', 'error_type': type(exc).__name__}
        print(json.dumps(result, ensure_ascii=True))
        return int(result.get('status') in {'error', 'degraded', 'configuration_unavailable'})
    if args.command == 'service':
        from .service import install_service, ensure_reporter_service, service_status, remove_service, ServiceError
        try:
            if args.action in {'install', 'ensure'}:
                action = ensure_reporter_service if args.action == 'ensure' else install_service
                result = action(args.root, args.state, args.repository, python=args.python, gh=args.gh,
                                interval=args.interval, since=args.since, environment_file=args.environment_file,
                                start=not args.no_start, reporting_config=args.reporting_config)
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
    state = Path(args.state).resolve()
    if args.command == "status":
        from .health import read_health
        from .fallback import _lstat_or_missing
        path = state / "reporter.sqlite3"
        result = {'worker': read_health(state), 'reporter': Outbox(path).rows() if _lstat_or_missing(path) is not None else []}
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
    from . import fallback as f
    try:
        policy = f.read_policy(args.reporting_config)
    except f.PolicyUnavailable:
        print(json.dumps({'status': 'configuration_unavailable', 'category': 'reporting_policy_unavailable'}))
        return 1
    if (policy is None or policy['repository'] != args.repository
            or Path(args.state).absolute() != f.state_path(args.reporting_config)):
        print(json.dumps({'status': 'configuration_unavailable', 'category': 'reporting_policy_mismatch'}))
        return 1
    os.environ['MINDIE_DIAGNOSTICS_CONFIG'] = str(f.policy_path(args.reporting_config))
    args.root = list(policy['roots'])
    from . import configure, __version__
    recorder = configure("mindie-diagnostics", root=state / "diagnostics", version=__version__)
    queue = Outbox(state / "reporter.sqlite3")
    stop = threading.Event()
    github = GitHub(args.repository, executable=args.gh, cancel=stop)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    from .health import Health
    with Health(state, recorder, interval=args.interval) as health:
        while not stop.is_set():
            try:
                policy = f.read_policy(args.reporting_config)
            except f.PolicyUnavailable:
                health.update(status='failed', stage='configuration')
                print(json.dumps({'status': 'configuration_unavailable', 'category': 'reporting_policy_unavailable'}), flush=True)
                return 1
            if policy is None:
                health.update(status='failed', stage='configuration')
                print(json.dumps({'status': 'configuration_unavailable', 'category': 'reporting_policy_missing'}), flush=True)
                return 1
            if (policy is None or policy['decision'] != 'enabled' or policy['repository'] != args.repository
                    or Path(args.state).absolute() != f.state_path(args.reporting_config)):
                queue.withdraw_unconsented()
                return 0
            args.root = list(policy['roots'])
            result = run_cycle(args, queue, github, recorder, health, since=since)
            print(json.dumps(result, ensure_ascii=True), flush=True)
            if args.once:
                return int(result['status'] != 'ok')
            stop.wait(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
