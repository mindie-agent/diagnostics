"""Explicit local reporting lifecycle commands (`maintain`, `ensure`, `upgrade_running`).

[Role]
Implements offline maintenance (`maintain`), explicit runtime + native service
provisioning (`ensure`), and bounded automatic runtime handoff (`upgrade_running`).

[Invariants]
- Never invoked automatically when a tool records a local failure.
- `maintain` phases (consent, intake, retention) are independent: each degrades
  the status without cancelling the others; intake needs an enabled policy.
- `upgrade_running` upgrades only a verified live worker, including degraded
  workers, when the installed distribution is strictly newer (`_source_order == "newer"`) and at
  least 60s of budget remain (`61s` from `maintain`).
- `current.json` is published only AFTER the replaced service passes health and
  `runtime_source` readback verification; otherwise rolls back to `old_python`.
- Generation retirement is admitted only when the live worker's source hash is
  confirmed, or a native service definition independently resolves it; an
  unavailable/unconfirmable worker defers retirement
  (`worker_generation_unconfirmed`) with a non-fatal warning.
- Opportunistic Windows orphan-profile cleanup runs from `maintain` inside the
  remaining monotonic budget and never changes the aggregate status.

[Timeouts & Locks]
- `maintain`'s capped monotonic budget admits stages instead of
  capping wall clock: roots, intake and retention start only with
  `_SETTLEMENT_RESERVE_S = 3.0s` left, intake scans until `deadline - 3.0s`, and
  work already inside a lawful 3-5s SQLite wait may outlive the budget.
- Retention receives the remaining budget minus that settlement reserve;
  reader snapshots admit their SQLite wait only when it fits.
- The retirement guard's native status probe uses a runner capped at the
  remaining budget; profile cleanup is capped at 3s and starts only with
  `>= 0.5s` left.
- `upgrade_running` reserves `_ROLLBACK_RESERVE_S = 20.0s` for rollback before
  `deadline`, caps lock acquisition at `_LOCK_WAIT_S = 12.0s`, and polls worker
  readback for up to `_READBACK_S = 8.0s`.

[Side-Effect Policy]
- Coordinates runtime generation preparation and native service installation
  under `runtime_transaction`. Opportunistic retention/generation retirement is
  bounded and never changes a healthy runtime or service status.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import fallback as f
from .filesystem import which_native
from .integration import reporting_status, _worker_view


def _state(config):
    path = f._as_local_absolute(f.state_path(config))
    if path is None or path.is_symlink():
        raise ValueError('unsafe_state_directory')
    try:
        f._private_mkdir(path, parents=True, exist_ok=True)
    except TimeoutError:
        raise
    except OSError as exc:
        raise ValueError('unsafe_state_directory') from exc
    if not path.is_dir():
        raise ValueError('unsafe_state_directory')
    return path


def maintain(config=None, *, update_running=False, unit_dir=None, budget_seconds=75):
    """Offline queue expiry/intake and log retention, including reporting OFF.

    One absolute budget (finite, nonnegative) is captured at
    entry (capped at ``max(75, _UPDATE_BUDGET_S + 15)``) and shared by offline
    cleanup and the optional handoff; the clock is
    never reset. It admits stages rather than capping total runtime: a root,
    its intake or its retention starts only while ``_SETTLEMENT_RESERVE_S``
    remains, intake scans until ``deadline - _SETTLEMENT_RESERVE_S``, and work
    already inside a lawful SQLite wait may finish after the budget. A budget
    smaller than that reserve therefore completes no root and says so
    (``budget_exhausted``) instead of starting a scan it must cut off.
    With ``update_running`` an additional bounded automatic
    handoff of an already running owned reporter to the installed source is
    attempted via :func:`upgrade_running`, but only when at least 61s remain
    (complete 60s handoff plus 1s exit margin). OFF, unconfigured,
    not-running, current, older-source and insufficient-budget skips are
    normal; a failed, suppressed or conflicting handoff degrades the
    aggregate status.

    Phases are independent: consent withdrawal, per-root intake and per-root
    local retention each report their own outcome and degrade the aggregate
    status without cancelling the others. Intake stays fail-closed (it runs
    only under an enabled policy); retention is purely local and runs for the
    roots this policy already declares. A policy file that exists but cannot
    be read still returns ``configuration_unavailable`` before any root is
    known, so its retention remains blocked by design, not by accident.
    """
    from .maintenance import prune
    from .outbox import Outbox
    from .ingestion import ingest
    from .reporting import ConsentUnavailable
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
        try:
            result['consent'] = {'withdrawn': queue.withdraw_unconsented()}
        except ConsentUnavailable:
            # An unreadable authority is not a revocation: queued evidence and
            # its processing budget stay. Intake below is already fail-closed,
            # and purely local retention of these known roots does not depend
            # on this read, so it must not be cancelled by it.
            result['status'] = 'degraded'
            result['consent'] = {'status': 'unavailable',
                                 'category': 'reporting_policy_unavailable'}
    enabled = bool(policy and policy.get('decision') == 'enabled')
    active_queue = queue if enabled else None
    for root in roots:
        if deadline - time.monotonic() < _SETTLEMENT_RESERVE_S:
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
        except TimeoutError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            result['status'] = 'degraded'
            result['retention'].append({'status': 'unavailable', 'category': type(exc).__name__})
        if active_queue is not None:
            try:
                # The scan stops one settlement reserve early so the enqueue and
                # cursor write it already started keep their lawful SQLite wait.
                result['ingestion'].append(ingest(root, active_queue, max_files=32, max_bytes=1024*1024,
                                                 config=config,
                                                 deadline=deadline - _SETTLEMENT_RESERVE_S))
            except TimeoutError:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                # Intake stops for this root and stays visible; the segment it
                # failed on keeps its cursor, and nothing authorized a write.
                result['status'] = 'degraded'
                result['ingestion'].append({'status': 'unavailable', 'category': type(exc).__name__})
        if deadline - time.monotonic() < _SETTLEMENT_RESERVE_S:
            # Intake already spent this budget: do not start a scan that would
            # only be cut off inside its own reader-cursor read.
            result['status'] = 'degraded'
            result['retention'].append({'status': 'unavailable', 'category': 'budget_exhausted', 'limited': True})
            continue
        try:
            result['retention'].append(prune(
                root, queue=active_queue,
                timeout=max(0, deadline - time.monotonic() - _SETTLEMENT_RESERVE_S),
            ))
        except TimeoutError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            result['status'] = 'degraded'
            result['retention'].append({'status': 'unavailable', 'category': type(exc).__name__})
    # The worker writes `worker.cycle` events under its own state directory.
    # That root is not a policy root (and must never be ingested back into the
    # GitHub outbox), so it needs an explicit local-only retention pass.
    state_diag = Path(f.state_path(config)).absolute() / 'diagnostics'
    known_roots = {os.path.normcase(os.path.abspath(os.path.expanduser(root)))
                   for root in roots}
    if (os.path.normcase(str(state_diag)) not in known_roots
            and (state_diag / 'events').exists()
            and deadline - time.monotonic() >= _SETTLEMENT_RESERVE_S):
        try:
            result['retention'].append(prune(
                state_diag, queue=None,
                timeout=max(0, deadline - time.monotonic() - _SETTLEMENT_RESERVE_S),
            ))
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
    # Retroactive convergence: installed users carry legacy generations that
    # predate this release. Opportunistically use the same runtime transaction
    # after ordinary maintenance, but never let cleanup change the outcome.
    try:
        retirement = _maintain_retirement(config, deadline, unit_dir=unit_dir)
    except TimeoutError:
        retirement = {'retirement_warning': 'TimeoutError'}
    if retirement:
        result.update(retirement)
    try:
        profiles = _maintain_orphan_profiles(config, deadline)
    except TimeoutError:
        profiles = {'orphan_profiles_warning': 'TimeoutError'}
    if profiles:
        result.update(profiles)
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
    gh = which_native('gh', which_fn=shutil.which)
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
            response = {'status': 'running' if ready else 'degraded',
                        'runtime': runtime, 'service': native, 'service_status': native_status, 'worker': worker,
                        'legacy_service': migration,
                        'recovery_hint': 'Inspect reporting status; an explicit reporting ensure may restart a stopped worker.'}
            if ready:
                # The new current is healthy: current + previous + the live
                # worker generation are the only rollback-closure artifacts.
                decision = _retirement_decision(
                    config, unit_dir=unit_dir, worker=worker, deadline=deadline)
                if not decision['allowed']:
                    response.update(_deferred_retirement())
                else:
                    try:
                        current = _read_current_or_none(root)
                        previous = (current.get('previous_source_hash') if current
                                    else runtime.get('previous_source_hash'))
                        extra = [previous, worker.get('runtime_source'), *decision['keep']]
                        response.update(_retirement_summary(root, runtime['source_hash'], extra))
                    except TimeoutError:
                        response['retirement_warning'] = 'TimeoutError'
            return response
    except ServiceError as exc:
        return {'status': 'degraded', 'category': exc.category}
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        # Runtime exceptions contain only static product categories.
        category = str(exc) if isinstance(exc, RuntimeError) and str(exc).replace('_','').isalnum() and len(str(exc))<80 else 'reporting_ensure_failed'
        return {'status': 'degraded', 'category': category}


# One lawful `Outbox.connect` wait (`timeout=3.0`/`busy_timeout=3000`): the
# smallest remainder in which a fresh scan can still settle its own SQLite work.
_SETTLEMENT_RESERVE_S = 3.0
_UPDATE_BUDGET_S = 180.0
_READBACK_S = 8.0
_ROLLBACK_RESERVE_S = 20.0
_LOCK_WAIT_S = 12.0
_SOURCE_HASH = re.compile(r'[0-9a-f]{64}\Z')
_SERVICE_HASH = re.compile(rb'([0-9a-f]{64})[\\/]+venv[\\/]+(?:Scripts|bin)[\\/]')


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
        if os.name == 'nt':
            kwargs.setdefault('creationflags', getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return subprocess.run(command, timeout=bounded_timeout, **kwargs)

    run.deadline = deadline
    run._is_real_runner = True
    return run


def _safe_category(exc, fallback_category):
    # Runtime failures carry only static product categories, never payloads.
    text = str(exc)
    if isinstance(exc, RuntimeError) and text.replace('_', '').isalnum() and len(text) < 80:
        return text
    return fallback_category


def _read_current_or_none(root):
    from .reporting_runtime import _read_current
    try:
        return _read_current(root)
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError):
        return None


def _deferred_retirement():
    """Non-fatal retirement summary when the live worker generation is unknown."""
    summary = {'status': 'deferred', 'reason': 'worker_generation_unconfirmed'}
    return {'retirement': summary, 'retirement_status': summary['status'],
            'retirement_reason': summary['reason'],
            'retirement_warning': summary['reason']}


def _verified_worker_source(view):
    if not isinstance(view, dict):
        return None
    source = view.get('runtime_source')
    return source if isinstance(source, str) and _SOURCE_HASH.fullmatch(source) else None


def _service_definition_files(unit_dir, config):
    """Bounded candidate definitions for the installed native worker."""
    files = []
    if unit_dir is not None:
        try:
            with os.scandir(Path(unit_dir)) as scan:
                for entry in scan:
                    if len(files) >= 16:
                        break
                    try:
                        if entry.is_file(follow_symlinks=False):
                            files.append(Path(entry.path))
                    except TimeoutError:
                        raise
                    except OSError:
                        continue
        except TimeoutError:
            raise
        except OSError:
            pass
    try:
        from .service import _unit_path
        files.append(_unit_path(unit_dir))
    except TimeoutError:
        raise
    except (AttributeError, OSError, ValueError):
        pass
    try:
        from .platform_service import _paths as service_paths
        manifest, _label, unit = service_paths(unit_dir)
        files.extend((manifest, unit))
    except TimeoutError:
        raise
    except (AttributeError, OSError, ValueError):
        pass
    return files


def _service_definition_source(unit_dir, config):
    """Resolve the generation hash from an installed service definition."""
    seen = set()
    for path in _service_definition_files(unit_dir, config):
        try:
            key = os.fspath(path)
        except TypeError:
            continue
        if key in seen:
            continue
        seen.add(key)
        try:
            raw = f._read_regular_bounded(path, 32768)
        except TimeoutError:
            raise
        except (OSError, ValueError):
            continue
        if not raw:
            continue
        match = _SERVICE_HASH.search(raw)
        if match is not None:
            return match.group(1).decode('ascii')
    return None


def _retirement_decision(config, *, unit_dir=None, worker=None, deadline=None):
    """Never retire generations while a live worker's source is unconfirmed."""
    try:
        view = worker if isinstance(worker, dict) else _worker_view(config)
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError):
        return {'allowed': False, 'keep': ()}
    source = _verified_worker_source(view)
    if source is not None:
        return {'allowed': True, 'keep': (source,)}
    status = view.get('status') if isinstance(view, dict) else None
    maybe_alive = bool(
        (isinstance(view, dict)
         and (view.get('healthy') is True or view.get('running') is True
              or (type(view.get('pid')) is int and view.get('pid') > 0
                  and status not in ('stopped', 'not_started'))))
        or status not in (None, 'not_started', 'stopped', 'absent'))
    try:
        from .service import service_status
        native = service_status(
            unit_dir=unit_dir,
            runner=_bounded_runner(deadline) if deadline is not None else None,
        )
    except subprocess.TimeoutExpired:
        native = None
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError):
        native = None
    if not isinstance(native, dict):
        return {'allowed': False, 'keep': ()}
    # An unavailable/alive-but-unidentified worker may be a process from an
    # older generation than the unit now names, so a service definition is not
    # proof of its runtime while the worker itself could still be alive.
    if maybe_alive:
        return {'allowed': False, 'keep': ()}
    if native.get('status') not in (None, 'absent'):
        resolved = _service_definition_source(unit_dir, config)
        if resolved is not None:
            return {'allowed': True, 'keep': (resolved,)}
        return {'allowed': False, 'keep': ()}
    return {'allowed': True, 'keep': ()}


def _retirement_summary(root, keep_source_hash, extra_keep, *, budget_seconds=3.0):
    """Best-effort cleanup summary; it never invalidates a healthy handoff."""
    from .reporting_runtime import retire_other_generations
    keep = [value for value in extra_keep if value]
    try:
        retired = retire_other_generations(
            root, keep_source_hash, extra_keep=keep, budget_seconds=budget_seconds,
        )
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        return {'retirement_warning': _safe_category(exc, type(exc).__name__)}
    summary = {'retired_generations': list(retired)}
    deferred = getattr(retired, 'deferred', ())
    if deferred:
        summary['retirement_deferred'] = list(deferred)
    return summary


def _maintain_retirement(config, deadline, *, unit_dir=None):
    """Opportunistic retroactive cleanup under the existing runtime lock."""
    from .reporting_runtime import runtime_transaction
    pointer = Path(f.policy_path(config)).with_suffix('.runtime') / 'current.json'
    if not pointer.exists() and not pointer.is_symlink():
        return None
    if deadline - time.monotonic() < 1.5:
        return None
    decision = _retirement_decision(config, unit_dir=unit_dir, deadline=deadline)
    if not decision['allowed']:
        return _deferred_retirement()
    try:
        with runtime_transaction(config, create=False, timeout=0.2) as root:
            if root is None:
                return None
            current = _read_current_or_none(root)
            if current is None:
                return None
            extra = [current.get('previous_source_hash'), *decision['keep']]
            budget = max(0.0, min(3.0, deadline - time.monotonic() - 1.0))
            return _retirement_summary(
                root, current['source_hash'], extra, budget_seconds=budget,
            )
    except TimeoutError:
        raise
    except RuntimeError as exc:
        if str(exc) == 'runtime_busy':
            # Another ensure/upgrade owns the transaction: normal and quiet.
            return None
        return {'retirement_warning': _safe_category(exc, 'runtime_retirement_failed')}
    except (OSError, ValueError) as exc:
        return {'retirement_warning': type(exc).__name__}


def _maintain_orphan_profiles(config, deadline):
    """Opportunistically resume marked Windows profile cleanup within budget."""
    remaining = deadline - time.monotonic()
    if remaining < 0.5:
        return None
    from .runtime_retirement import prune_orphan_windows_profiles
    runtime_root = Path(f.policy_path(config)).with_suffix('.runtime')
    try:
        retired = prune_orphan_windows_profiles(
            runtime_root, budget_seconds=max(0.0, min(3.0, remaining - 0.5)))
    except TimeoutError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        return {'orphan_profiles_warning': _safe_category(
            exc, 'orphan_profile_cleanup_failed')}
    summary = {'retired_profiles': list(retired)}
    deferred = getattr(retired, 'deferred', ())
    if deferred:
        summary['deferred_profiles'] = list(deferred)
    volume = getattr(retired, 'volume_unavailable', ())
    if volume:
        summary['deferred_volume_unavailable'] = list(volume)
    swept = getattr(retired, 'swept', ())
    if swept:
        summary['swept_trash'] = list(swept)
    if summary['retired_profiles'] or deferred or volume or swept:
        return {'orphan_profiles': summary}
    return None


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
        gh = which_native('gh', which_fn=shutil.which)
        if not gh:
            return {'status': 'skipped', 'reason': 'gh_unavailable'}
        native = service_status(unit_dir=unit_dir, runner=runner)
        if native.get('status') != 'active':
            return {'status': 'skipped', 'reason': 'reporter_service_not_running',
                    'service_status': native.get('status')}
        worker = _worker_view(config)
        if worker.get('status') == 'unavailable':
            return {'status': 'degraded', 'reason': 'reporter_worker_unavailable'}
        if not (worker.get('healthy') or worker.get('running')):
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
                worker = _worker_view(config)
                if not (worker.get('healthy') or worker.get('running')):
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
                    updated = {'status': 'updated', 'reason': 'reporter_updated',
                               'previous': previous, 'new': target,
                               'runtime': {'python': runtime['python']},
                               'service': {'status': native_status.get('status'),
                                           'changed': replacement.get('changed')},
                               'rollback': 'not_needed'}
                    decision = _retirement_decision(
                        config, unit_dir=unit_dir, worker=worker, deadline=deadline)
                    if not decision['allowed']:
                        updated.update(_deferred_retirement())
                    else:
                        try:
                            committed = _read_current_or_none(root)
                            previous_hash = (committed.get('previous_source_hash') if committed
                                             else current['source_hash'])
                            extra = [previous_hash, worker.get('runtime_source'),
                                     *decision['keep']]
                            updated.update(_retirement_summary(root, source_hash, extra))
                        except TimeoutError:
                            updated['retirement_warning'] = 'TimeoutError'
                    return updated
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
