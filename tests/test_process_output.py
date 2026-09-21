"""Actual FD capture, rotation and launcher-independent lifetime; no remote I/O."""
import json
import os
from pathlib import Path
import subprocess
import sys


def records(root):
    return [json.loads(line) for path in root.glob("events/*/*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]


def test_capture_setup_failure_preserves_result_and_original_error(tmp_path, monkeypatch):
    import pytest
    from mindie_diagnostics import configure, capture_output
    from mindie_diagnostics import process_output
    rec = configure("mindie-output-fixture", root=tmp_path)
    monkeypatch.setattr(process_output.os, "pipe", lambda: (_ for _ in ()).throw(OSError("pipe unavailable")))
    with rec.operation("output.fixture") as operation:
        with capture_output(operation):
            result = 17
    assert result == 17
    error = RuntimeError("original failure")
    with pytest.raises(RuntimeError) as caught:
        with rec.operation("output.fixture") as operation:
            with capture_output(operation):
                raise error
    assert caught.value is error
    events = records(tmp_path)
    assert any(row["event"] == "process.output.gap" for row in events)
    assert events[-1]["status"] == "error"
    rec.close()


def test_capture_setup_cancellation_releases_scope(tmp_path, monkeypatch):
    import pytest
    from mindie_diagnostics import configure, capture_output, process_output
    rec = configure("mindie-output-cancel", root=tmp_path)
    original_pipe = process_output.os.pipe
    def interrupted():
        raise KeyboardInterrupt()
    monkeypatch.setattr(process_output.os, "pipe", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with rec.operation("output.cancel") as op:
            with capture_output(op):
                pytest.fail("cancellation was swallowed")
    monkeypatch.setattr(process_output.os, "pipe", original_pipe)
    with rec.operation("output.recovered") as op:
        with capture_output(op):
            pass
    assert not any(row.get("attributes", {}).get("reason") == "capture_already_active" for row in records(tmp_path))
    rec.close()


def test_actual_unwritable_storage_does_not_recurse_or_hide_fallback(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    program = '''
import os, sys, time
from mindie_diagnostics import configure, capture_output
rec = configure("output-failure", root=sys.argv[1], level="INFO")
with rec.operation("output.fixture", level="DEBUG") as op:
    with capture_output(op):
        os.write(2, b"child stderr\\n")
        deadline = time.monotonic() + 3
        while not rec.logging_failed and time.monotonic() < deadline:
            time.sleep(0.001)
        assert rec.logging_failed
        os.write(1, b"child still running\\n")
assert op.summary()["status"] == "success"
print("business complete")
'''
    result = subprocess.run([sys.executable, "-c", program, str(blocked)],
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == b"business complete"
    assert result.stderr == b""  # Blocking stderr is optional; logging_failed above is authoritative.
    assert len(result.stderr) < 200


def test_second_fd_failure_retains_bounded_first_stream_and_restores(tmp_path):
    program = '''
import os, sys
from mindie_diagnostics import configure, capture_output, process_output
rec = configure("output-partial", root=sys.argv[1])
original_pipe = process_output.os.pipe
calls = 0
def pipe():
    global calls
    calls += 1
    if calls == 2:
        raise OSError("second pipe unavailable")
    return original_pipe()
process_output.os.pipe = pipe
with rec.operation("output.fixture") as op:
    with capture_output(op):
        os.write(1, b"captured first stream\\n")
        os.write(2, b"unchanged second stream\\n")
print("after capture")
assert op.summary()["status"] == "success"
'''
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == b"after capture"
    assert result.stderr.strip() == b"unchanged second stream"
    events = records(tmp_path)
    assert any(row.get("attributes", {}).get("preview") == "captured first stream" for row in events)
    assert any(row.get("attributes", {}).get("reason") == "capture_unavailable" for row in events)


def test_actual_service_fd_output_is_bounded_and_redacted(tmp_path):
    program = '''
import os, sys
import mindie_diagnostics.logging as log
from mindie_diagnostics import capture_output
log.MAX_LOG_BYTES = 4096
rec = log.configure("mindie-knowledge", root=sys.argv[1], level="DEBUG")
print("before capture")
with rec.operation("knowledge.daemon.fixture") as op:
    with capture_output(op):
        for i in range(150):
            os.write(1, ("normal child line %s " % i + "x" * 160 + "\\n").encode())
    with capture_output(op):
        os.write(2, b"password=")
        os.write(2, ("split" + "credential\\n").encode())
    with capture_output(op):
        os.write(1, b"y" * 40000 + b"\\n")
    with capture_output(op):
        os.write(2, b"partial secret without newline")
rec.close()
print("after capture")
'''
    root = tmp_path / "logs"
    result = subprocess.run([sys.executable, "-c", program, str(root)], capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.replace(b"\r\n", b"\n") == b"before capture\nafter capture\n"
    assert result.stderr == b""
    files = list(root.glob("events/*/*.jsonl*"))
    assert 1 < len(files) <= 4
    assert sum(path.stat().st_size for path in files) <= 4 * 4096
    text = "".join(path.read_text(encoding="utf-8") for path in files)
    assert "splitcredential" not in text and "partial secret" not in text
    assert "before capture" not in text and "after capture" not in text
    assert "oversized_line" in text and "incomplete_line" in text
    assert "redacted:credential" in text


def test_service_output_survives_launcher_exit(tmp_path):
    child = tmp_path / "child.py"
    done = tmp_path / "done"
    root = tmp_path / "logs"
    child.write_text('''
from pathlib import Path
import os, sys, time
from mindie_diagnostics import configure
from mindie_diagnostics import capture_output
rec = configure("mindie-knowledge", root=sys.argv[1])
with rec.operation("knowledge.daemon.fixture") as op:
    with capture_output(op):
        time.sleep(0.3)
        os.write(1, b"after launcher exit\\n")
rec.close()
Path(sys.argv[2]).write_text("complete")
''', encoding="utf-8")
    launcher = '''
import subprocess, sys
import os
def _popen_kwargs():
    return {"stdin": subprocess.DEVNULL, **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True})}
subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **_popen_kwargs())
'''
    result = subprocess.run([sys.executable, "-c", launcher, str(child), str(root), str(done)],
                            capture_output=True, timeout=10,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])
                                 + os.pathsep + os.environ.get("PYTHONPATH", "")})
    assert result.returncode == 0, result.stderr
    import time
    deadline = time.monotonic() + 8
    while not done.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert done.exists()
    events = records(root)
    assert any(row.get("attributes", {}).get("preview") == "after launcher exit" for row in events)
    assert events[-1]["status"] == "success"

