"""Explicit local reporting operations; never invoked by recording a fault."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import fallback as f
from .integration import reporting_status, _worker_view


def _state(config):
    path = f._as_local_absolute(f.state_path(config))
    if path is None or not f._ancestors_are_real_dirs(path, create=True) or not f._owned_dir(path, create=True):
        raise ValueError('unsafe_state_directory')
    return path


def maintain(config=None):
    """Offline queue expiry/intake and log retention, including reporting OFF."""
    from .maintenance import prune
    from .outbox import Outbox
    from .ingestion import ingest
    if f._as_local_absolute(f.policy_path(config)) is None:
        raise ValueError('invalid_configuration')
    policy = f.read_policy(config)
    roots = policy['roots'] if policy else [str(f.default_root())]
    path = f.state_path(config) / 'reporter.sqlite3'
    queue = Outbox(path) if path.exists() else None
    result = {'status': 'ok', 'network': False, 'ingestion': [], 'retention': []}
    if queue is not None:
        queue.maintain()
        queue.withdraw_unconsented()
    for root in roots:
        try:
            if queue is not None:
                result['ingestion'].append(ingest(root, queue, max_files=32, max_bytes=1024*1024))
            result['retention'].append(prune(root, queue=queue))
        except (OSError, ValueError, RuntimeError) as exc:
            result['status'] = 'degraded'
            result['retention'].append({'status': 'unavailable', 'category': type(exc).__name__})
    if any(row.get('limited') for row in result['ingestion'] + result['retention']):
        result['status'] = 'degraded'
    return result


def ensure(config=None, *, unit_dir=None):
    """Explicitly choose installed source and start the one owned pure reporter."""
    policy = f.read_policy(config)
    if policy is None:
        return {'status': 'configuration_unavailable', 'category': 'reporting_not_configured'}
    if policy['decision'] != 'enabled':
        return {'status': 'disabled', 'worker': _worker_view(config)}
    gh = shutil.which('gh')
    if not gh:
        return {'status': 'configuration_unavailable', 'category': 'gh_unavailable'}
    from .reporting_runtime import prepare_runtime
    from .outbox import Outbox
    from .reader_registry import register_reader
    from .service import ensure_reporter_service, service_status, ServiceError
    try:
        runtime = prepare_runtime(config)
        state = _state(config)
        queue = Outbox(state / 'reporter.sqlite3')
        for root in policy['roots']:
            register_reader(root, queue.path)
        native = ensure_reporter_service(policy['roots'], state, policy['repository'],
            python=runtime['python'], gh=gh, reporting_config=str(f.policy_path(config)), unit_dir=unit_dir)
        deadline = time.monotonic() + 8
        worker = _worker_view(config)
        while time.monotonic() < deadline and not (worker.get('healthy') and worker.get('runtime_source') == runtime['source_hash']):
            time.sleep(.1)
            worker = _worker_view(config)
        native_status = service_status(unit_dir=unit_dir)
        ready = worker.get('healthy') and worker.get('runtime_source') == runtime['source_hash'] and native_status.get('status') == 'active'
        return {'status': 'running' if ready else 'degraded',
                'runtime': runtime, 'service': native, 'service_status': native_status, 'worker': worker,
                'recovery_hint': 'Inspect reporting status; an explicit reporting ensure may restart a stopped worker.'}
    except ServiceError as exc:
        return {'status': 'degraded', 'category': exc.category}
    except (OSError, ValueError, RuntimeError) as exc:
        # Runtime exceptions contain only static product categories.
        category = str(exc) if isinstance(exc, RuntimeError) and str(exc).replace('_','').isalnum() and len(str(exc))<80 else 'reporting_ensure_failed'
        return {'status': 'degraded', 'category': category}
