import json
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_diagnostics import bind_context, configure, current_context
from mindie_diagnostics import logging as module


def rows(root):
    return [json.loads(line) for path in root.glob("events/*/*.jsonl*") for line in path.read_text(encoding="utf-8").splitlines()]


def test_real_records_phases_context_and_no_stdout(tmp_path, capsys):
    recorder = configure("tests", root=tmp_path, level="DEBUG")
    with bind_context({"trace_id": "a" * 32, "operation_id": "b" * 32}):
        with recorder.operation("request") as op:
            with op.phase("read") as phase:
                assert current_context()["phase_id"] == phase.phase_id
                op.event("DEBUG", "read.detail", count=2)
            assert "phase_id" not in current_context()
    result = op.summary()
    recorder.close()
    assert result["status"] == "success" and result["duration_ms"] >= 0
    assert result["started_at"].endswith("Z") and result["finished_at"].endswith("Z")
    assert result["trace_id"] == "a" * 32 and result["parent_operation_id"] == "b" * 32
    assert result["phases"][0]["phase"] == "read"
    events = rows(tmp_path)
    assert [event["event"] for event in events] == ["operation.start", "phase.start", "read.detail", "phase.end", "operation.end"]
    assert all(event["monotonic_ns"] > 0 for event in events)
    assert len({event["process_instance_id"] for event in events}) == 1
    assert capsys.readouterr().out == ""


def test_original_exception_and_failure_outcome(tmp_path):
    recorder = configure("tests", root=tmp_path)
    original = RuntimeError("business failure")
    with pytest.raises(RuntimeError) as caught:
        with recorder.operation("run") as op:
            raise original
    assert caught.value is original
    assert op.summary()["status"] == "error"
    with recorder.operation("returned_failure") as returned:
        returned.fail("transport", retryable=False, submission_state="unknown")
    assert returned.summary()["status"] == "error"
    recorder.close()
    assert rows(tmp_path)[-1]["attributes"]["submission_state"] == "unknown"


def test_disk_failure_never_masks_business_or_writes_blocking_stderr(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("private error detail")
    monkeypatch.setattr(module._Handler, "emit", fail)
    recorder = configure("tests", root=tmp_path)
    with recorder.operation("healthy") as op:
        pass
    with pytest.raises(ValueError, match="original"):
        with recorder.operation("bad"):
            raise ValueError("original")
    assert op.summary()["status"] == "success" and op.summary()["logging_failed"]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert "private error" not in captured.err
    recorder.close()


def test_unwritable_root_is_nonblocking(tmp_path, capsys):
    path = tmp_path / "not-directory"
    path.write_text("original")
    recorder = configure("tests", root=path)
    with recorder.operation("run") as op:
        pass
    assert op.summary()["status"] == "success" and op.summary()["logging_failed"]
    assert path.read_text() == "original"
    assert capsys.readouterr().err == ""


def test_debug_scopes_do_not_write_but_error_does(tmp_path):
    recorder = configure("tests", root=tmp_path)
    with recorder.operation("heartbeat", level="DEBUG"):
        pass
    assert not (tmp_path / "events").exists()
    with recorder.operation("heartbeat", level="DEBUG") as op:
        op.fail("lost")
    recorder.close()
    assert all(event["severity"] == "ERROR" for event in rows(tmp_path))


def test_attributes_are_bounded_and_sensitive_values_omitted(tmp_path):
    class Poison:
        def __str__(self):
            raise AssertionError("must not stringify business objects")
    recorder = configure("tests", root=tmp_path)
    with recorder.operation("run") as op:
        op.event("INFO", "detail", argv=["private"], api_key="private", unknown=Poison(),
                 large="文" * 30000, env={"HOME": "private"})
    recorder.close()
    data = rows(tmp_path)[1]["attributes"]
    assert data["unknown"] == "[unsupported]"
    assert data["argv"] == data["api_key"] == data["env"] == "[omitted]"
    assert len(data["large"]) <= 512
    assert all(len(line.encode()) <= module.MAX_RECORD_BYTES for p in tmp_path.glob("events/*/*") for line in p.read_text(encoding="utf-8").splitlines())


def test_two_real_processes_rotate_independently(tmp_path):
    program = """import sys
from mindie_diagnostics import configure
from mindie_diagnostics import logging as m
m.MAX_LOG_BYTES=2048
r=configure('tests',root=sys.argv[1])
for i in range(24):
    with r.operation('work') as op:
        op.event('INFO','item',count=i)
r.close()
"""
    processes = [subprocess.Popen([sys.executable, "-c", program, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    for process in processes:
        out, err = process.communicate(timeout=15)
        assert process.returncode == 0 and not out and not err
    paths = list(tmp_path.glob("events/tests/*"))
    assert len({p.name.split("-")[0] for p in paths}) == 2
    assert len(paths) <= 2 * (module.BACKUP_COUNT + 1)
    assert all(p.stat().st_size <= 2048 for p in paths)
    assert rows(tmp_path)


def test_system_exit_zero_success_and_nonzero_failure(tmp_path):
    recorder = configure("tests", root=tmp_path)
    for code in (0, None, 2):
        with pytest.raises(SystemExit):
            with recorder.operation("cli") as op:
                raise SystemExit(code)
        assert op.status == ("success" if code in (None, 0) else "error")
    recorder.close()


def test_configure_is_idempotent_and_exception_stack_has_no_source_or_locals(tmp_path):
    recorder = configure("tests", root=tmp_path, version="0.1.0")
    assert configure("tests", root=tmp_path, version="0.1.0") is recorder
    private = "ghp_" + "A" * 36
    with pytest.raises(RuntimeError):
        with recorder.operation("call"):
            try:
                raise ValueError("token=" + private)
            except ValueError as cause:
                raise RuntimeError("wrapper") from cause
    recorder.close()
    event = rows(tmp_path)[-1]
    assert event["package_version"] == "0.1.0"
    attributes = event["attributes"]
    assert attributes["exception_chain"] == ["RuntimeError", "ValueError"]
    assert attributes["stack_frames"]
    assert all(set(frame) == {"module", "function", "line"} for frame in attributes["stack_frames"])
    assert private not in json.dumps(event)


def test_nested_phase_parent_and_bounded_summary(tmp_path):
    recorder = configure("tests", root=tmp_path)
    with recorder.operation("run") as op:
        with op.phase("outer") as outer:
            with outer.phase("inner") as inner:
                pass
        assert inner.summary()["phase_id"] != outer.summary()["phase_id"]
        for _ in range(40):
            with op.phase("repeat"):
                pass
    assert len(op.summary()["phases"]) == 32
    assert op.summary()["phases"][0]["parent_phase_id"] == outer.phase_id
    recorder.close()

def test_posix_private_leaf_and_rotated_files(tmp_path, monkeypatch):
    import os
    import stat
    import pytest
    import mindie_diagnostics.logging as log
    if os.name == "nt":
        pytest.skip("POSIX mode bits")
    before = stat.S_IMODE(tmp_path.stat().st_mode)
    monkeypatch.setattr(log, "MAX_LOG_BYTES", 1024)
    rec = log.configure("private-mode", root=tmp_path)
    previous = os.umask(0)
    try:
        for _ in range(30):
            rec.event("INFO", "sample", preview="x" * 400)
    finally:
        os.umask(previous)
        rec.close()
    leaf = tmp_path / "events" / "private-mode"
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o700
    assert stat.S_IMODE(tmp_path.stat().st_mode) == before
    files = list(leaf.iterdir())
    assert len(files) > 1
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)

def test_fork_resets_locks_held_by_another_thread(tmp_path):
    import os
    import signal
    import threading
    import time
    import pytest
    import mindie_diagnostics.logging as log
    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    old = log.configure("fork-locks", root=tmp_path / "old")
    old.event("INFO", "parent.sample")
    current = log.configure("fork-locks", root=tmp_path / "new")
    ready, release = threading.Event(), threading.Event()
    def hold():
        with log._LOCK, old._mutex, current._mutex:
            ready.set()
            release.wait(10)
    thread = threading.Thread(target=hold)
    thread.start()
    assert ready.wait(2)
    pid = os.fork()
    if pid == 0:
        try:
            old.event("INFO", "child.old")
            current.event("INFO", "child.current")
            log.get_recorder("fork-new").event("DEBUG", "child.registry")
            os._exit(0 if old.record_ref and current.record_ref else 2)
        except BaseException:
            os._exit(3)
    status = None
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            completed, child_status = os.waitpid(pid, os.WNOHANG)
            if completed:
                status = child_status
                break
            time.sleep(0.01)
        if status is None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        assert status is not None, "child logging waited for an inherited non-owning thread lock"
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        thread.join(3)
        old.close()
        current.close()
def test_import_override_does_not_claim_installed_revision(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from mindie_diagnostics import logging as implementation
    class Distribution:
        version = '1.0.0'
        def locate_file(self, path):
            return tmp_path / 'installed' / path
        def read_text(self, name):
            return '{"vcs_info":{"commit_id":"' + 'a' * 40 + '"}}'
    monkeypatch.setattr(implementation.metadata, 'distribution', lambda _: Distribution())
    monkeypatch.setitem(sys.modules, 'mindie_test', SimpleNamespace(__file__=str(tmp_path / 'candidate' / 'mindie_test' / '__init__.py')))
    monkeypatch.setattr(implementation, '_VERSIONS', {})
    assert implementation._package('mindie-test') == {'package_version': 'unknown'}
    monkeypatch.setitem(sys.modules, 'mindie_test', SimpleNamespace(__file__=str(tmp_path / 'installed' / 'mindie_test' / '__init__.py')))
    monkeypatch.setattr(implementation, '_VERSIONS', {})
    assert implementation._package('mindie-test')['package_revision'] == 'a' * 40


def test_failure_events_roll_with_exact_metadata_and_bounded_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'MAX_LOG_BYTES', 1024)
    event = {'schema': 1, 'event': 'operation.end', 'status': 'error',
             'package_revision': 'a' * 40, 'package_version': 'selected-native',
             'reporting': None, 'reportable': False, 'attributes': {'stage': 'protocol'}}
    for _ in range(100):
        assert module.append_failure_event('failure-stream', tmp_path, event)
    paths = list(tmp_path.glob('events/failure-stream/*.jsonl*'))
    assert len(paths) == 4
    assert all({k: v for k, v in row.items() if k != "process_instance_id"} == event
               and len(row["process_instance_id"]) == 32
               and row["process_instance_id"] != "0" * 32 for row in rows(tmp_path))
    for i in range(40):
        assert module.append_failure_event('cache-' + str(i), tmp_path, event)
    assert len(module._FAILURE_RECORDERS) <= 32


def test_failure_writer_lock_contention_does_not_queue(tmp_path):
    import threading
    import time
    event = {'event': 'operation.end', 'status': 'error'}
    assert module.append_failure_event('contention', tmp_path, event)
    recorder = module._FAILURE_RECORDERS[('contention', str(tmp_path.absolute()))]
    for lock in (module._LOCK, recorder._mutex):
        ready, release = threading.Event(), threading.Event()
        def hold():
            with lock:
                ready.set()
                release.wait(3)
        thread = threading.Thread(target=hold)
        thread.start()
        try:
            assert ready.wait(1)
            start = time.monotonic()
            assert not module.append_failure_event('contention', tmp_path, event)
            assert time.monotonic() - start < .1
        finally:
            release.set()
            thread.join(timeout=1)
        assert not thread.is_alive()


def test_warning_only_uses_already_nonblocking_descriptor(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    if os.name != 'posix':
        pytest.skip('POSIX descriptor semantics')
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        os.set_blocking(read_fd, False)
        monkeypatch.setattr(module.sys, 'stderr', SimpleNamespace(fileno=lambda: write_fd))
        recorder = module.Recorder('warning-pipe', root=tmp_path)
        recorder._unavailable()
        recorder._unavailable()
        assert os.read(read_fd, 4096).count(b'storage unavailable') == 1
        while True:
            try:
                os.write(write_fd, b'x' * 4096)
            except BlockingIOError:
                break
        second = module.Recorder('full-warning-pipe', root=tmp_path)
        second._unavailable()
        assert second.logging_failed
        os.set_blocking(write_fd, True)
        third = module.Recorder('blocking-warning-pipe', root=tmp_path)
        third._unavailable()  # A full blocking pipe must not be written at all.
        assert third.logging_failed and os.get_blocking(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_windows_slot_replacement_closes_temp_before_rename(tmp_path, monkeypatch):
    import os

    class WindowsCalls:
        name = 'nt'

        def __init__(self):
            self.temporary = {}
            self.replacements = 0

        def __getattr__(self, name):
            return getattr(os, name)

        def open(self, path, flags, *args):
            descriptor = os.open(path, flags, *args)
            if Path(path).name.startswith('.segment-'):
                self.temporary[descriptor] = Path(path)
            return descriptor

        def close(self, descriptor):
            os.close(descriptor)
            self.temporary.pop(descriptor, None)

        def replace(self, source, destination):
            assert Path(source) not in self.temporary.values()
            self.replacements += 1
            return os.replace(source, destination)

    calls = WindowsCalls()
    monkeypatch.setattr(module, 'os', calls)
    monkeypatch.setattr(module, 'MAX_LOG_BYTES', 1024)
    recorder = module.Recorder('windows-slot', root=tmp_path)
    try:
        for _ in range(24):
            recorder.event('INFO', 'sample', detail='x' * 400)
        assert calls.replacements > 0
        assert not calls.temporary
        assert not recorder.logging_failed and recorder.dropped_records == 0
        assert len(list(tmp_path.glob('events/windows-slot/*.jsonl*'))) == 4
        assert rows(tmp_path)
    finally:
        recorder.close()


def test_cold_first_write_is_not_vetoed_by_reader_setup_clock(tmp_path, monkeypatch):
    """The old select() set a 100ms deadline before entering reader_guard."""
    import time
    from mindie_diagnostics import reader_registry

    clock = [1_000.0]

    def monotonic():
        return clock[0]

    monkeypatch.setattr(time, "monotonic", monotonic)
    real_guard = reader_registry.reader_guard

    @contextmanager
    def guard_that_costs_101ms(*args, **kwargs):
        clock[0] += 0.101
        with real_guard(*args, **kwargs) as state:
            yield state

    monkeypatch.setattr(module, "reader_guard", guard_that_costs_101ms)
    recorder = module.Recorder("cold-first", root=tmp_path)
    try:
        recorder.event("INFO", "cold", detail="x" * 40)
    finally:
        recorder.close()

    paths = list(tmp_path.glob("events/cold-first/*.jsonl*"))
    assert paths, (
        f"cold first record was dropped before segment allocation: "
        f"dropped_records={recorder.dropped_records}, "
        f"logging_failed={recorder.logging_failed}"
    )
    assert recorder.dropped_records == 0 and not recorder.logging_failed
    assert [row["event"] for row in rows(tmp_path)] == ["cold"]
    assert not (tmp_path / ".readers").exists()


@contextmanager
def _locked_registry(tmp_path):
    """Hold the real reader registry lock in a spawned process with bounded waits."""
    import multiprocessing

    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from lock_worker import hold_registry_lock

    context = multiprocessing.get_context("spawn")
    ready_receive, ready_send = context.Pipe(duplex=False)
    release_receive, release_send = context.Pipe(duplex=False)
    process = context.Process(
        target=hold_registry_lock,
        args=(str(tmp_path), ready_send, release_receive),
        name="diagnostics-registry-lock",
        daemon=True,
    )
    process.start()
    ready_send.close()
    release_receive.close()
    try:
        if not ready_receive.poll(5.0):
            raise AssertionError(
                f"registry lock worker did not report within 5s: "
                f"pid={process.pid}, exitcode={process.exitcode}"
            )
        try:
            status, detail = ready_receive.recv()
        except EOFError as exc:
            raise AssertionError(
                f"registry lock worker exited before reporting: "
                f"pid={process.pid}, exitcode={process.exitcode}, error={exc}"
            ) from None
        assert status == "locked", (
            f"registry lock worker failed: pid={process.pid}, "
            f"exitcode={process.exitcode}, detail={detail}"
        )
        yield
    finally:
        try:
            release_send.send("release")
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            release_send.close()
        process.join(5.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        ready_receive.close()


def test_free_rotation_slot_allocates_while_reader_registry_is_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_LOG_BYTES", 512)
    recorder = module.Recorder("free-slot", root=tmp_path)
    try:
        recorder.event("INFO", "first", detail="x" * 200)
        with _locked_registry(tmp_path):
            recorder.event("INFO", "second", detail="x" * 200)
        paths = list(tmp_path.glob("events/free-slot/*.jsonl*"))
        assert len(paths) == 2, (
            f"a free rotation slot was dropped while reader metadata was busy: "
            f"paths={paths}, dropped_records={recorder.dropped_records}, "
            f"logging_failed={recorder.logging_failed}"
        )
        assert not recorder.logging_failed and recorder.dropped_records == 0
        assert sorted(row["event"] for row in rows(tmp_path)) == ["first", "second"]
    finally:
        recorder.close()


def test_full_window_waits_for_registered_reader_cursors(tmp_path, monkeypatch):
    from mindie_diagnostics.ingestion import ingest
    from mindie_diagnostics.outbox import Outbox

    monkeypatch.setattr(module, "MAX_LOG_BYTES", 512)
    queue = Outbox(tmp_path / "queue.db")
    # Real ingestion creates the WAL-backed cursors schema and registers the reader.
    assert ingest(tmp_path, queue)["scanned_bytes"] == 0
    recorder = module.Recorder("reader-window", root=tmp_path)
    try:
        for index in range(4):
            recorder.event("INFO", "sample", index=index, detail="x" * 200)
        assert not recorder.logging_failed
        paths = sorted(tmp_path.glob("events/reader-window/*.jsonl*"))
        assert len(paths) == 4
        before = {path: path.stat().st_size for path in paths}

        recorder.event("INFO", "unread", detail="x" * 200)
        assert recorder.dropped_records == 1 and recorder.logging_failed
        assert sorted(tmp_path.glob("events/reader-window/*.jsonl*")) == paths
        assert {path: path.stat().st_size for path in paths} == before

        consumed = ingest(tmp_path, queue)
        assert consumed["scanned_bytes"] > 0 and consumed["files"] == 4
        recorder.event("INFO", "consumed", detail="x" * 200)
        assert recorder.dropped_records == 1
        assert sorted(tmp_path.glob("events/reader-window/*.jsonl*")) == paths
        assert "consumed" in {row["event"] for row in rows(tmp_path)}
    finally:
        recorder.close()
