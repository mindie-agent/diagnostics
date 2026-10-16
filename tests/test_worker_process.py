"""Worker PID reuse and Windows read-only process identity."""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mindie_diagnostics import fallback, integration, worker_process


@pytest.mark.parametrize("observed,expected,status", [
    ((True, 123456), 123456, "idle"),
    ((True, 654321), 123456, "stopped"),
    ((False, None), 123456, "stopped"),
    ((None, None), 123456, "unverified"),
    ((True, 123456), None, "unverified"),
])
def test_windows_worker_health_requires_matching_process_identity(
        tmp_path, monkeypatch, observed, expected, status):
    config = tmp_path / "reporting.json"
    state = fallback.state_path(config)
    state.mkdir()
    health = dict(schema=1, pid=123, status="idle", interval=60,
                  heartbeat_at=time.time(), progress_at=time.time())
    if expected is not None:
        health["process_created"] = expected
    (state / "worker-health.json").write_text(json.dumps(health))
    monkeypatch.setattr(integration, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(worker_process, "windows_process", lambda pid: observed)
    view = integration._worker_view(config)
    assert view["status"] == status
    assert view["healthy"] is (status == "idle")


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process APIs")
def test_native_windows_identity_is_read_only_and_detects_exit():
    alive, created = worker_process.windows_process(os.getpid())
    assert alive is True and type(created) is int and created > 0
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        child_alive, child_created = worker_process.windows_process(process.pid)
        assert child_alive is True and type(child_created) is int and child_created > 0
        assert process.poll() is None
        process.terminate()
        process.wait(timeout=5)
        assert worker_process.windows_process(process.pid)[0] is False
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
