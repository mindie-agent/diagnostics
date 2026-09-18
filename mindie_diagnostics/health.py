"""Local, atomic worker liveness and progress; no network or workload authority."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

from .bundle import _write_output


def read_health(state, *, clock=time.time):
    path = Path(state) / 'worker-health.json'
    try:
        if path.is_symlink() or path.stat().st_size > 65536:
            raise ValueError('invalid health file')
        data = json.loads(path.read_text(encoding='utf-8'))
        now = clock()
        data['stale'] = now - data['heartbeat_at'] > 60
        data['progress_stalled'] = (data['status'] == 'running' and
                                    now - data['progress_at'] > max(1200, data['interval'] * 3))
        data['healthy'] = not data['stale'] and not data['progress_stalled'] and data['status'] in {'running', 'idle'}
        return data
    except FileNotFoundError:
        return {'status': 'not_started', 'healthy': False}
    except (OSError, ValueError, TypeError, KeyError):
        return {'status': 'unreadable', 'healthy': False}


class Health:
    def __init__(self, state, recorder, *, interval=60, clock=time.time):
        self.path, self.recorder, self.clock = Path(state) / 'worker-health.json', recorder, clock
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.data = {'schema': 1, 'pid': os.getpid(), 'started_at': clock(), 'interval': interval,
                     'heartbeat_at': clock(), 'progress_at': clock(), 'status': 'running', 'stage': 'starting'}
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
        self.update(status='failed' if exc_type else 'stopped', stage='stopped')

