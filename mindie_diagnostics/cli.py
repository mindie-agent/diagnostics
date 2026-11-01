"""CLI entry point (`mindie-diagnostics`) for bundles, services, reporting, and worker loop.

[Role]
Provides the `mindie-diagnostics` command-line interface (`bundle`, `status`,
`service`, `reporting`, `worker`, `precision`) and executes the supervised worker loop
(`run_cycle`: `consent` -> `ingest` -> `retention` -> `report`).

[Invariants]
- Each stage in `run_cycle` is isolated so an ingestion or retention failure
  does not prevent the outbox from draining (`publish_one`).
- Once the owner stop event is observed, `run_cycle` keeps every result and
  cursor already collected but starts no further root, retention or remote
  stage, reporting that as `limited` + `degraded`.
- The worker re-reads `reporting.json` at the start of every cycle and
  immediately withdraws unconsented items and exits 0 if reporting is disabled.

[Timeouts & Locks]
- Worker loop sleeps on `stop.wait(args.interval)` (minimum 5s) and handles
  `SIGINT`/`SIGTERM` by setting the shared `stop` event passed to `GitHub` and
  to `ingest`, so a stop ends the running scan between complete records and
  admits no new stage, instead of finishing every root first.

[Side-Effect Policy]
- Emits single-line ASCII JSON results to stdout and exits non-zero on
  `error`, `degraded`, or `configuration_unavailable` states.
"""
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


def run_cycle(args, queue, github, recorder, health, *, since=None, cancel=None):
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

        def admitted(stage):
            """Keep what this cycle already collected; start no further stage."""
            if cancel is None or not cancel.is_set():
                return True
            result['status'], result['limited'] = 'degraded', 1
            operation.event('WARNING', 'worker.stage_not_started', stage=stage,
                            error_code='worker_stop_requested')
            return False

        from .maintenance import prune
        queue.maintain()
        result['consent'] = attempt('consent', lambda: {'withdrawn': queue.withdraw_unconsented()})
        for root in args.root:
            if not admitted('ingest'):
                break
            result['ingestion'].append(attempt('ingest', lambda: ingest(
                root, queue, since=since, config=getattr(args, "reporting_config", None),
                cancel=cancel)))
            if not admitted('retention'):
                break
            result['retention'].append(attempt('retention', lambda: prune(root, queue=queue)))
        if admitted('report'):
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
            command.add_argument('--consent-config', help='shared contribution authority for this profile')
            command.add_argument('--repository', default=DEFAULT_REPOSITORY)
            command.add_argument('--root', action='append')
    precision = sub.add_parser(
        'precision', help='storage-graph precision debugging: inspect capsule, replay, diff, slim MoE')
    pactions = precision.add_subparsers(dest='precision_action', required=True)
    inspect_capsule = pactions.add_parser(
        'inspect-capsule', aliases=['inspect_capsule'],
        help='print a saved call capsule summary')
    inspect_capsule.add_argument('capsule', nargs='?')
    inspect_capsule.add_argument('--capsule', '--capsule-path', '--path', dest='capsule_opt')
    replay = pactions.add_parser(
        'replay-call', aliases=['replay_call', 'call-repro'],
        help='replay one captured call from its pre-call state')
    replay.add_argument('capsule', nargs='?')
    replay.add_argument('--capsule', '--capsule-path', dest='capsule_opt')
    replay.add_argument('--callable', help='explicit module:qualname callable')
    replay.add_argument('--repeat', type=int, default=1)
    replay.add_argument('--det-lock', action='store_true')
    replay.add_argument('--reference', help='explicit reference module:qualname')
    replay.add_argument('--reference-atol', type=float, default=1e-3)
    replay.add_argument('--reference-rtol', type=float, default=1e-3)
    replay.add_argument('--save-output')
    compare = pactions.add_parser(
        'compare-traces', aliases=['compare_traces', 'lockstep-diff'],
        help='compare two lockstep JSON reports')
    compare.add_argument('baseline', nargs='?')
    compare.add_argument('target', nargs='?')
    compare.add_argument('--baseline', '--baseline-file', dest='baseline_opt')
    compare.add_argument('--target', '--target-file', dest='target_opt')
    compare.add_argument('--output')
    slim = pactions.add_parser(
        'slim-moe', aliases=['slim_moe'],
        help='slice routed experts in a safetensors MoE checkpoint')
    slim.add_argument('input_dir', nargs='?')
    slim.add_argument('output_dir', nargs='?')
    slim.add_argument('target', nargs='?', type=int)
    slim.add_argument('--input-dir', '--input', dest='input_dir_opt')
    slim.add_argument('--output-dir', '--output', dest='output_dir_opt')
    slim.add_argument('--target-routed-experts', dest='target_opt', type=int)
    watch = pactions.add_parser(
        'watch', help='inspect an exported npu-watch JSON record file')
    watch.add_argument('records', nargs='?')
    watch.add_argument('--records', '--file', dest='records_opt')
    watch.add_argument('--latch', action='store_true')
    return result


def _jsonable(value, depth=0):
    """Project runtime objects (tensors, exceptions) into JSON-safe values."""
    if depth > 8:
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"type": "tensor", "dtype": str(value.dtype),
                "shape": [int(item) for item in value.shape],
                "device": str(getattr(value, "device", "cpu"))}
    return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _import_callable(spec):
    if not spec:
        return None
    import importlib
    module_name, separator, qualname = spec.partition(":")
    if not separator or not module_name:
        raise ValueError("callable must be module:qualname")
    target = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise ValueError(f"{spec} is not callable")
    return target


def _precision_compare(baseline_path, target_path):
    baseline = _read_json(baseline_path)
    target = _read_json(target_path)

    def index(payload):
        rows = payload.get("metrics") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return None
        indexed = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = (row.get("module_path", ""), row.get("invocation_ordinal", 0),
                   row.get("tree_path", row.get("path", "")), row.get("rank", 0))
            indexed[str(key)] = row
        return indexed

    left, right = index(baseline), index(target)
    if left is None or right is None:
        return {"status": "unsupported", "reason": "metric_rows_required"}
    baseline_status = baseline.get("status") if isinstance(baseline, dict) else None
    target_status = target.get("status") if isinstance(target, dict) else None
    if baseline_status not in (None, "ok") or target_status not in (None, "ok"):
        return {"status": "diverged", "reason": "trace_status_not_ok",
                "baseline_status": baseline_status, "target_status": target_status}
    if set(left) != set(right) or not left:
        return {"status": "diverged", "reason": "metric_keys_mismatch",
                "baseline_metrics": len(left), "target_metrics": len(right),
                "compared": len(set(left) & set(right))}
    numeric = ("max_abs_err", "rel_l2_err", "ulp_max", "sjr", "jitter_max",
               "nonfinite_mismatch_count", "sample_count")
    deltas = {}
    for key in sorted(set(left) & set(right)):
        for field in numeric:
            before, after = left[key].get(field), right[key].get(field)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                deltas.setdefault(field, []).append(float(after) - float(before))
    summary = {}
    for field, values in deltas.items():
        values = sorted(values)
        summary[field] = {
            "count": len(values),
            "max_delta": max(abs(item) for item in values),
            "median_delta": values[len(values) // 2],
        }
    return {
        "status": "ok",
        "compared": len(set(left) & set(right)),
        "baseline_metrics": len(left),
        "target_metrics": len(right),
        "deltas": summary,
    }


def _run_precision(args):
    action = args.precision_action.replace("_", "-")
    if action == "call-repro":
        action = "replay-call"
    if action == "lockstep-diff":
        action = "compare-traces"
    if action == "inspect-capsule":
        from .precision import capsule_summary
        capsule = getattr(args, "capsule", None) or args.capsule_opt
        if not capsule:
            return {"status": "error", "reason": "capsule_required"}, 1
        return capsule_summary(capsule), 0
    if action == "replay-call":
        from .precision import replay_call
        capsule = getattr(args, "capsule", None) or args.capsule_opt
        if not capsule:
            return {"status": "error", "reason": "capsule_required"}, 1
        import torch
        from .precision.call_repro import _apply_det_lock
        _apply_det_lock(torch, args.det_lock)
        fn = _import_callable(args.callable)
        reference = _import_callable(args.reference)
        result = replay_call(
            capsule, fn, repeat=args.repeat, det_lock=args.det_lock,
            reference_fn=reference, reference_atol=args.reference_atol,
            reference_rtol=args.reference_rtol)
        if args.save_output and result.get("outputs") is not None:
            import torch
            torch.save(result["outputs"], args.save_output)
        return _jsonable(result), (0 if result.get("status") == "ok" else 1)
    if action == "compare-traces":
        baseline = getattr(args, "baseline", None) or args.baseline_opt
        target = getattr(args, "target", None) or args.target_opt
        if not baseline or not target:
            return {"status": "error", "reason": "baseline_and_target_required"}, 1
        result = _precision_compare(baseline, target)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=True)
                handle.write("\n")
        return result, 0 if result.get("status") == "ok" else 1
    if action == "slim-moe":
        from .precision.stimulus import slim_moe_checkpoint
        input_dir = getattr(args, "input_dir", None) or args.input_dir_opt
        output_dir = getattr(args, "output_dir", None) or args.output_dir_opt
        target = getattr(args, "target", None) or args.target_opt
        if not input_dir or not output_dir or not target:
            return {"status": "error", "reason": "input_output_target_required"}, 1
        return _jsonable(slim_moe_checkpoint(input_dir, output_dir, target)), 0
    if action == "watch":
        records = getattr(args, "records", None) or args.records_opt
        if not records:
            return {"status": "unsupported", "reason": "records_file_required"}, 1
        payload = _read_json(records)
        if args.latch:
            return _jsonable(payload.get("latch", payload)), 0
        return _jsonable(payload), 0
    return {"status": "error", "reason": "unknown_precision_action"}, 2


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == 'precision':
        try:
            result, code = _run_precision(args)
        except Exception as exc:
            result = {'status': 'error', 'category': 'precision_operation_failed',
                      'error_type': type(exc).__name__, 'error': str(exc)[:240]}
            code = 1
        print(json.dumps(_jsonable(result), ensure_ascii=True, default=str))
        return int(code)
    if args.command == 'reporting':
        from .integration import reporting_status, configure_reporting
        from .reporting_commands import ensure, maintain
        try:
            if args.action == 'status':
                result = reporting_status(config=args.config)
            elif args.action == 'configure':
                result = configure_reporting(args.enabled == 'true', repository=args.repository,
                                             config=args.config, roots=args.root,
                                             consent_config=args.consent_config)
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
        return 0 if result.get("status") not in {"unknown", "error", "failed"} else 1
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
                health.update(status='degraded', stage='configuration')
                print(json.dumps({'status': 'configuration_unavailable', 'category': 'reporting_policy_unavailable'}), flush=True)
                if args.once:
                    return 1
                stop.wait(args.interval)
                continue
            if policy is None:
                health.update(status='degraded', stage='configuration')
                print(json.dumps({'status': 'configuration_unavailable', 'category': 'reporting_policy_missing'}), flush=True)
                if args.once:
                    return 1
                stop.wait(args.interval)
                continue
            if (policy is None or policy['decision'] != 'enabled' or policy['repository'] != args.repository
                    or Path(args.state).absolute() != f.state_path(args.reporting_config)):
                queue.withdraw_unconsented()
                from .reader_registry import unregister_reader
                for root in args.root:
                    if (Path(root) / '.readers' / 'registry.json').exists():
                        try:
                            unregister_reader(root, queue.path, timeout=0.5)
                        except TimeoutError:
                            raise
                        except (OSError, ValueError, RuntimeError):
                            pass
                return 0
            args.root = list(policy['roots'])
            result = run_cycle(args, queue, github, recorder, health, since=since, cancel=stop)
            print(json.dumps(result, ensure_ascii=True), flush=True)
            if args.once:
                return int(result['status'] != 'ok')
            stop.wait(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
