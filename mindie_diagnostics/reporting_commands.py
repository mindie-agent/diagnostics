"""Explicit local reporting lifecycle commands (`maintain`, `ensure`, `upgrade_running`).

[Role]
Implements offline maintenance (`maintain`), explicit runtime + native service
provisioning (`ensure`), and bounded automatic runtime handoff (`upgrade_running`).

[Invariants]
- Never invoked automatically when a tool records a local failure.
- `upgrade_running` only upgrades an already-active healthy worker when the
  installed distribution is strictly newer (`_source_order == "newer"`) and at
  least 60s of budget remain (`61s` from `maintain`).
- `current.json` is published only AFTER the replaced service passes health and
  `runtime_source` readback verification; otherwise rolls back to `old_python`.

[Timeouts & Locks]
- `maintain` shares a single monotonic deadline capped at `75.0s` (or
  `_UPDATE_BUDGET_S + 15.0`).
- `upgrade_running` reserves `_ROLLBACK_RESERVE_S = 20.0s` for rollback before
  `deadline`, caps lock acquisition at `_LOCK_WAIT_S = 12.0s`, and polls worker
  readback for up to `_READBACK_S = 8.0s`.

[Side-Effect Policy]
- Coordinates runtime generation preparation and native service installation
  under `runtime_transaction`.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import fallback as f
from .integration import reporting_status, _worker_view


def _state(config):
    path = f._as_local_absolute(f.state_path(config))
    if path is None or path.is_symlink():
        raise ValueError('unsafe_state_directory')
    try:
        f._private_mkdir(path, parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError('unsafe_state_directory') from exc
    if not path.is_dir():
        raise ValueError('unsafe_state_directory')
    return path


def maintain(config=None, *, update_running=False, unit_dir=None, budget_seconds=75):
    """Offline queue expiry/intake and log retention, including reporting OFF.

    One absolute budget (finite, nonnegative, capped at 75s) is captured at
    entry and shared by offline cleanup and the optional handoff; the clock is
    never reset. With ``update_running`` an additional bounded automatic
    handoff of an already running owned reporter to the installed source is
    attempted via :func:`upgrade_running`, but only when at least 61s remain
    (complete 60s handoff plus 1s exit margin). OFF, unconfigured,
    not-running, current, older-source and insufficient-budget skips are
    normal; a failed, suppressed or conflicting handoff degrades the
    aggregate status.
    """
    from .maintenance import prune
    from .outbox import Outbox
    from .ingestion import ingest
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)):
        raise ValueError('invalid_budget')
    if not math.isfinite(budget_seconds) or budget_seconds < 0:
        raise ValueError('invalid_budget')
    deadline = time.monotonic() + min(float(budget_seconds), max(75.0, _UPDATE_BUDGET_S + 15.0))
    if f._as_local_absolute(f.policy_path(config)) is None:
        raise ValueError('invalid_configuration')
    try:
        policy = f.read_policy(config)
    except f.PolicyUnavailable:
        return {'status': 'configuration_unavailable', 'category': 'reporting_policy_unavailable', 'network': False}
    roots = policy['roots'] if policy else [str(f.default_root())]
    path = f.state_path(config) / 'reporter.sqlite3'
    queue = Outbox(path) if f._lstat_or_missing(path) is not None else None
    result = {'status': 'ok', 'network': False, 'ingestion': [], 'retention': []}
    if queue is not None:
        queue.maintain()
        queue.withdraw_unconsented()
    enabled = bool(policy and policy.get('decision') == 'enabled')
    active_queue = queue if enabled else None
    for root in roots:
        if time.monotonic() >= deadline:
            result['retention'].append({'status': 'unavailable', 'category': 'budget_exhausted', 'limited': True})
            break
        try:
            if not enabled and (Path(root) / '.readers' / 'registry.json').exists():
                from .reader_registry import unregister_reader
                try:
                    unregister_reader(root, path, timeout=min(.2, max(0, deadline - time.monotonic())))
                except TimeoutError:
                    raise
                except (OSError, ValueError, RuntimeError):
                    pass
            if active_queue is not None:
                result['ingestion'].append(ingest(root, active_queue, max_files=32, max_bytes=1024*1024,
                                                 config=config))
            result['retention'].append(prune(root, queue=active_queue))
        except TimeoutError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            result['status'] = 'degraded'
            result['retention'].append({'status': 'unavailable', 'category': type(exc).__name__})
    if any(row.get('limited') for row in result['ingestion'] + result['retention']):
        result['status'] = 'degraded'
    if update_running:
        if deadline - time.monotonic() < 61:
            update = {'status': 'skipped', 'reason': 'insufficient_budget'}
        else:
            update = upgrade_running(config, unit_dir=unit_dir, deadline=deadline - 1)
        result['reporter_update'] = update
        if update.get('status') not in {'current', 'skipped', 'updated'}:
            result['status'] = 'degraded'
    return result


def ensure(config=None, *, unit_dir=None):
    """Explicitly choose installed source and start the one owned pure reporter."""
    deadline = time.monotonic() + 75
    runner = _bounded_runner(deadline - 5)
    try:
        policy = f.read_policy(config)
    except f.PolicyUnavailable:
        return {'status': 'configuration_unavailable', 'category': 'reporting_policy_unavailable'}
    if policy is None:
        return {'status': 'configuration_unavailable', 'category': 'reporting_not_configured'}
    if policy['decision'] != 'enabled':
        return {'status': 'disabled', 'worker': _worker_view(config)}
    if unit_dir is None:
        unit_dir = f.service_directory(config)
    consent_revision = policy.get('revision')
    gh = shutil.which('gh')
    if not gh:
        return {'status': 'configuration_unavailable', 'category': 'gh_unavailable'}
    from .reporting_runtime import _prepare_locked, runtime_transaction
    from .outbox import Outbox
    from .reader_registry import register_reader
    from .service import ensure_reporter_service, service_status, remove_service, ServiceError
    try:
        # The explicit path cooperates with the same runtime transaction lock
        # as an automatic handoff: preparation, replacement and readback are
        # never interleaved with another runtime transaction.
        with runtime_transaction(config, timeout=min(12, max(0, deadline - time.monotonic()))) as root:
            runtime = _prepare_locked(config, deadline=deadline - 13)
            if _enabled_policy(config, consent_revision) is None:
                # Withdrawn or regranted during preparation: never start up.
                return {'status': 'degraded', 'category': 'reporting_authorization_changed'}
            state = _state(config)
            queue = Outbox(state / 'reporter.sqlite3')
            for root_dir in policy['roots']:
                register_reader(root_dir, queue.path)
            migration = None
            if policy.get("consent_config"):
                # Only the legacy owned instance bound to this exact policy
                # may retire. Another profile's service remains untouched.
                migration = remove_service(expected_reporting_config=config, runner=runner)
            native = ensure_reporter_service(policy['roots'], state, policy['repository'],
                python=runtime['python'], gh=gh, reporting_config=str(f.policy_path(config)), unit_dir=unit_dir,
                expected_consent_revision=consent_revision, runner=runner,
                since="1970-01-01T00:00:00Z")
            readback_deadline = min(deadline - 5, time.monotonic() + 8)
            worker = _worker_view(config)
            while time.monotonic() < readback_deadline and not (worker.get('healthy') and worker.get('runtime_source') == runtime['source_hash']):
                time.sleep(.1)
                worker = _worker_view(config)
            native_status = service_status(unit_dir=unit_dir, runner=runner)
            ready = worker.get('healthy') and worker.get('runtime_source') == runtime['source_hash'] and native_status.get('status') == 'active'
            return {'status': 'running' if ready else 'degraded',
                    'runtime': runtime, 'service': native, 'service_status': native_status, 'worker': worker,
                    'legacy_service': migration,
                    'recovery_hint': 'Inspect reporting status; an explicit reporting ensure may restart a stopped worker.'}
    except ServiceError as exc:
        return {'status': 'degraded', 'category': exc.category}
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        # Runtime exceptions contain only static product categories.
        category = str(exc) if isinstance(exc, RuntimeError) and str(exc).replace('_','').isalnum() and len(str(exc))<80 else 'reporting_ensure_failed'
        return {'status': 'degraded', 'category': category}


_UPDATE_BUDGET_S = 180.0
_READBACK_S = 8.0
_ROLLBACK_RESERVE_S = 20.0
_LOCK_WAIT_S = 12.0


def _utcnow():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _bounded_runner(deadline):
    """Cap existing subprocess.run timeouts at the remaining absolute budget."""
    def run(command, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, 0)
        timeout = kwargs.pop('timeout', None)
        bounded_timeout = remaining if timeout is None else min(timeout, remaining)
        return subprocess.run(command, timeout=bounded_timeout, **kwargs)

    run.deadline = deadline
    return run


def _safe_category(exc, fallback_category):
    # Runtime failures carry only static product categories, never payloads.
    text = str(exc)
    if isinstance(exc, RuntimeError) and text.replace('_', '').isalnum() and len(text) < 80:
        return text
    return fallback_category


def _enabled_policy(config, required_revision=None):
    policy = f.read_policy(config)
    if policy is None or policy['decision'] != 'enabled':
        return None
    if required_revision is not None and policy.get('revision') != required_revision:
        # A consent revoke/regrant yields a new revision; do not cross it.
        return None
    return policy


def upgrade_running(config=None, *, unit_dir=None, deadline=None):
    """Bounded automatic handoff of the running owned reporter to installed source."""
    from .reporting_runtime import (
        _collect_source,
        _generation_python,
        _prepare_locked,
        _publish,
        _read_current,
        _source_order,
        runtime_transaction,
    )
    from .service import ServiceError, ensure_reporter_service, service_status

    now = time.monotonic()
    if deadline is None:
        deadline = now + _UPDATE_BUDGET_S
    else:
        if deadline - now < 60.0:
            return {'status': 'skipped', 'reason': 'insufficient_budget'}
        deadline = min(now + _UPDATE_BUDGET_S, deadline)
    forward = deadline - _ROLLBACK_RESERVE_S
    runner = _bounded_runner(forward)
    try:
        policy = f.read_policy(config)
        if policy is None:
            return {'status': 'skipped', 'reason': 'reporting_not_configured'}
        if policy['decision'] != 'enabled':
            return {'status': 'skipped', 'reason': 'reporting_off'}
        if unit_dir is None:
            unit_dir = f.service_directory(config)
        consent_revision = policy['revision']
        gh = shutil.which('gh')
        if not gh:
            return {'status': 'skipped', 'reason': 'gh_unavailable'}
        native = service_status(unit_dir=unit_dir, runner=runner)
        if native.get('status') != 'active':
            return {'status': 'skipped', 'reason': 'reporter_service_not_running',
                    'service_status': native.get('status')}
        worker = _worker_view(config)
        if worker.get('status') == 'unavailable':
            return {'status': 'degraded', 'reason': 'reporter_worker_unavailable'}
        if not worker.get('healthy'):
            return {'status': 'skipped', 'reason': 'reporter_worker_not_running',
                    'worker_status': worker.get('status')}
        wait = max(0.0, min(_LOCK_WAIT_S, forward - time.monotonic()))
        with runtime_transaction(config, create=False, timeout=wait) as root:
            if root is None:
                return {'status': 'skipped', 'reason': 'no_committed_runtime'}
            current = _read_current(root)
            if current is None:
                return {'status': 'skipped', 'reason': 'no_committed_runtime'}
            previous = {'version': current['version'], 'revision': current.get('revision'),
                        'source_hash': current['source_hash']}
            if worker.get('runtime_source') != current['source_hash']:
                return {'status': 'skipped', 'reason': 'worker_source_mismatch', 'previous': previous}
            version, revision, files, source_hash = _collect_source()
            target = {'version': version, 'revision': revision, 'source_hash': source_hash}
            order = _source_order(current, version, source_hash)
            if order == 'identical':
                return {'status': 'current', 'reason': 'identical_source',
                        'previous': previous, 'new': target}
            if order == 'older':
                return {'status': 'skipped', 'reason': 'older_source',
                        'previous': previous, 'new': target}
            if order == 'conflict':
                return {'status': 'source_conflict', 'reason': 'same_version_different_source',
                        'previous': previous, 'new': target}
            if order == 'incomparable':
                return {'status': 'degraded', 'reason': 'source_version_incomparable',
                        'previous': previous, 'new': target}
            policy = _enabled_policy(config, consent_revision)
            if policy is None:
                return {'status': 'skipped', 'reason': 'policy_withdrawn',
                        'previous': previous, 'new': target}
            old_python = _generation_python(root, current['source_hash'])
            mutated = False
            failure_action = None
            failure_returncode = None
            try:
                runtime = _prepare_locked(config, publish=False, source=(version, revision, files, source_hash),
                                          deadline=forward)
                policy = _enabled_policy(config, consent_revision)
                if policy is None:
                    return {'status': 'skipped', 'reason': 'policy_withdrawn',
                            'previous': previous, 'new': target}
                if not _worker_view(config).get('healthy'):
                    return {'status': 'skipped', 'reason': 'reporter_worker_not_running',
                            'previous': previous, 'new': target}
                state = _state(config)
                mutated = True
                replacement = ensure_reporter_service(
                    policy['roots'], state, policy['repository'], python=runtime['python'], gh=gh,
                    reporting_config=str(f.policy_path(config)), unit_dir=unit_dir, runner=runner,
                    expected_consent_revision=consent_revision)
                readback = min(time.monotonic() + _READBACK_S, forward)
                worker = _worker_view(config)
                while time.monotonic() < readback and not (
                        worker.get('healthy') and worker.get('runtime_source') == source_hash):
                    time.sleep(.1)
                    worker = _worker_view(config)
                native_status = service_status(unit_dir=unit_dir, runner=runner)
                if (worker.get('healthy') and worker.get('runtime_source') == source_hash
                        and native_status.get('status') == 'active'):
                    _publish(root, {'schema': 1, 'source_hash': source_hash,
                                    'version': version, 'revision': revision})
                    return {'status': 'updated', 'reason': 'reporter_updated',
                            'previous': previous, 'new': target,
                            'runtime': {'python': runtime['python']},
                            'service': {'status': native_status.get('status'),
                                        'changed': replacement.get('changed')},
                            'rollback': 'not_needed'}
                failure = 'readback_unconfirmed'
            except subprocess.TimeoutExpired:
                failure = 'update_deadline_exceeded'
            except ServiceError as exc:
                failure = exc.category
                failure_action = exc.action
                failure_returncode = exc.returncode
            except TimeoutError:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                failure = _safe_category(exc, 'reporter_update_failed')
            extra = {}
            if failure_action is not None:
                extra['failure_action'] = failure_action
            if failure_returncode is not None:
                extra['failure_returncode'] = failure_returncode
            if not mutated:
                return {'status': 'degraded', 'reason': failure, 'previous': previous, 'new': target,
                        'rollback': 'not_needed', **extra}
            rollback_action = None
            rollback_returncode = None
            policy = _enabled_policy(config, consent_revision) if time.monotonic() < deadline - 5 else None
            if time.monotonic() >= deadline - 5:
                rollback = 'deadline_exceeded'
            elif policy is None:
                rollback = 'skipped_policy_withdrawn'
            else:
                rollback_runner = _bounded_runner(deadline)
                try:
                    ensure_reporter_service(policy['roots'], _state(config), policy['repository'],
                        python=old_python, gh=gh, reporting_config=str(f.policy_path(config)),
                        unit_dir=unit_dir, runner=rollback_runner,
                        expected_consent_revision=consent_revision)
                    readback = min(time.monotonic() + _READBACK_S, deadline - 2)
                    worker = _worker_view(config)
                    while time.monotonic() < readback and not (
                            worker.get('healthy') and worker.get('runtime_source') == current['source_hash']):
                        time.sleep(.1)
                        worker = _worker_view(config)
                    native_status = service_status(unit_dir=unit_dir, runner=rollback_runner)
                    rollback = ('restored'
                                if worker.get('healthy')
                                and worker.get('runtime_source') == current['source_hash']
                                and native_status.get('status') == 'active'
                                else 'unproven')
                except subprocess.TimeoutExpired:
                    rollback = 'deadline_exceeded'
                except ServiceError as exc:
                    rollback = 'failed'
                    rollback_action = exc.action
                    rollback_returncode = exc.returncode
                except TimeoutError:
                    raise
                except (OSError, ValueError, RuntimeError):
                    rollback = 'failed'
            if rollback_action is not None:
                extra['rollback_action'] = rollback_action
            if rollback_returncode is not None:
                extra['rollback_returncode'] = rollback_returncode
            return {'status': 'degraded', 'reason': failure, 'previous': previous, 'new': target,
                    'rollback': rollback, **extra}
    except subprocess.TimeoutExpired:
        return {'status': 'degraded', 'reason': 'update_deadline_exceeded'}
    except ServiceError as exc:
        result = {'status': 'degraded', 'reason': exc.category}
        if exc.action is not None:
            result['failure_action'] = exc.action
        if exc.returncode is not None:
            result['failure_returncode'] = exc.returncode
        return result
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        return {'status': 'degraded', 'reason': _safe_category(exc, 'reporter_update_failed')}
