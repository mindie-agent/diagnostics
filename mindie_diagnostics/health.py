"""Atomic local worker liveness and progress heartbeat (`worker-health.json`).

[Role]
Maintains `<state>/worker-health.json` with a background 15s daemon heartbeat
thread and stage progress timestamps, and provides `read_health` for local
status inspection.

[Invariants]
- A worker is considered `stale` if `now - heartbeat_at > 60s` and
  `progress_stalled` if `status == "running"` for `> max(1200s, interval * 3)`.
- Records `runtime_source` (generation SHA-256) and `process_created` (on Windows)
  to prevent PID-reuse false positives and verify runtime handoffs.

[Timeouts & Locks]
- `self.lock` (`threading.RLock`) serializes concurrent heartbeat and stage
  updates; `__exit__` joins the heartbeat thread with a 2.0s timeout.

[Side-Effect Policy]
- Writes `worker-health.json` atomically via `_write_output`; write failures log
  a warning event without crashing the worker.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import sys
import time

from .bundle import _write_output
from .fallback import _lstat_or_missing


def read_health(state, *, clock=time.time):
    path = Path(state) / 'worker-health.json'
    try:
        info = _lstat_or_missing(path)
        if info is None:
            return {'status': 'not_started', 'healthy': False}
        if path.is_symlink() or info.st_size > 65536:
            raise ValueError('invalid health file')
        data = json.loads(path.read_text(encoding='utf-8'))
        now = clock()
        data['stale'] = now - data['heartbeat_at'] > 60
        data['progress_stalled'] = (data['status'] == 'running' and
                                    now - data['progress_at'] > max(1200, data['interval'] * 3))
        data['healthy'] = not data['stale'] and not data['progress_stalled'] and data['status'] in {'running', 'idle'}
        return data
    except TimeoutError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        return {'status': 'unreadable', 'healthy': False}


class Health:
    def __init__(self, state, recorder, *, interval=60, clock=time.time):
        self.path, self.recorder, self.clock = Path(state) / 'worker-health.json', recorder, clock
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.data = {'schema': 1, 'pid': os.getpid(), 'started_at': clock(), 'interval': interval,
                     'heartbeat_at': clock(), 'progress_at': clock(), 'status': 'running', 'stage': 'starting'}
        if os.name == 'nt':
            from .worker_process import windows_process
            alive, created = windows_process(os.getpid())
            if alive is True and type(created) is int and created > 0:
                self.data['process_created'] = created
        from .integration import _component_meta
        from . import fallback as f
        version, revision = _component_meta('mindie-diagnostics')
        self.data.update(package_version=version, package_revision=revision)
        raw = f._read_regular_bounded(Path(sys.prefix).parent / 'source.json', 4096)
        try:
            source = json.loads(raw) if raw is not None else {}
            digest = source.get('source_hash')
            if isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest):
                self.data['runtime_source'] = digest
        except (ValueError, TypeError, AttributeError):
            pass
        self.thread = None

    def update(self, **fields):
        with self.lock:
            self.data.update(fields, progress_at=self.clock())
            self.write()

    def write(self):
        with self.lock:
            self.data['heartbeat_at'] = self.clock()
            try:
                _write_output(self.path, json.dumps(self.data, ensure_ascii=True).encode())
            except Exception as exc:
                self.recorder.event('WARNING', 'worker.health_unavailable', error_type=type(exc).__name__)

    def _heartbeat(self):
        while not self.stop.wait(15):
            self.write()

    def __enter__(self):
        self.write()
        self.thread = threading.Thread(target=self._heartbeat, name='mindie-worker-heartbeat', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=2)
        self.update(status='failed' if exc_type or self.data['status'] == 'failed' else 'stopped', stage='stopped')
