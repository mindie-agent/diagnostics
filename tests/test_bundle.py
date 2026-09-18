import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_diagnostics import configure, collect_bundle, export_public_event
from mindie_diagnostics.bundle import _canonical


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO race')
def test_regular_input_replaced_by_fifo_cannot_block_open(tmp_path):
    script = '''import os,sys
from pathlib import Path
from mindie_diagnostics.bundle import _safe_open
path=Path(sys.argv[1]);path.write_bytes(b'fixture')
original=os.open
def swap(value, flags, *args):
    path.unlink();os.mkfifo(path)
    return original(value, flags, *args)
os.open=swap
try:
    with _safe_open(path):
        raise AssertionError('FIFO accepted')
except ValueError:
    pass
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'raced-input')],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def example(**changes):
    return {"schema": 1, "timestamp": "2026-09-13T12:00:00.000000Z",
            "monotonic_ns": 100, "pid": 42, "component": "tests", "severity": "ERROR",
            "event": "operation.end", "operation_id": "a" * 32, "trace_id": "b" * 32,
            "operation": "run", "status": "error", "duration_ms": 12.5, "attributes": {},
            **changes}


def test_projection_never_exports_unknown_text_or_paths_and_is_idempotent():
    secret = "ghp_" + "A" * 36
    public = export_public_event(example(message=secret, credentials=secret,
        attributes={"count": 2, "category": "transport", "message": secret, "path": "/home/" + "private-user", "nested": {"secret": secret}}))
    assert public["attributes"] == {"count": 2, "category": "transport"}
    assert secret not in json.dumps(public)
    assert "pid" not in public and "message" not in public
    assert export_public_event(public) == public


@pytest.mark.parametrize("changes", [
    {"schema": 2}, {"schema": True}, {"trace_id": "invalid"}, {"operation_id": "../"},
    {"timestamp": "not-utc"}, {"attributes": []}, {"pid": -1}, {"severity": "BAD"},
    {"component": "arbitrary body"}, {"monotonic_ns": -1},
])
def test_unknown_event_shape_rejected(changes):
    assert export_public_event(example(**changes)) is None


def test_real_bundle_operation_filter_hash_and_immutable_source(tmp_path):
    recorder = configure("tests", root=tmp_path)
    with recorder.operation("failure") as op:
        op.fail("transport", submission_state="unknown")
    with recorder.operation("success") as other:
        pass
    recorder.close()
    original = Path(op.summary()["record_ref"]).read_bytes()
    output = tmp_path / "bundle.json"
    result = collect_bundle(tmp_path, operation_id=op.operation_id, output=output)
    assert result["summary"]["operation_ids"] == [op.operation_id]
    assert result["summary"]["error_count"] > 0
    assert Path(op.summary()["record_ref"]).read_bytes() == original
    assert json.loads(output.read_bytes()) == result
    unsigned = {key: value for key, value in result.items() if key not in {"bundle_id", "content_sha256"}}
    assert hashlib.sha256(_canonical(unsigned)).hexdigest() == result["bundle_id"] == result["content_sha256"]


def test_missing_backend_and_unknown_raw_files_are_not_read(tmp_path):
    (tmp_path / "credentials.json").write_text("sensitive")
    result = collect_bundle(tmp_path, records=[example()])
    assert result["summary"]["event_count"] == 1
    assert "events_missing" in result["redaction"]["gaps"]
    assert "sensitive" not in json.dumps(result)


def test_bounded_output_and_invalid_record_count(tmp_path):
    events = [example(operation_id=f"{i:032x}") for i in range(100)]
    result = collect_bundle(tmp_path, records=events, max_bytes=2048)
    assert len(_canonical(result)) + 1 <= 2048
    assert "output_clipped" in result["redaction"]["gaps"]
    invalid = collect_bundle(tmp_path, records=({"schema": 99} for _ in range(2000)))
    assert invalid["redaction"]["invalid_events"] == 1000
    assert "record_limit" in invalid["redaction"]["gaps"]


def test_bad_and_partial_lines_preserved_but_not_published(tmp_path):
    path = tmp_path / "events" / "tests" / ("42-" + "c" * 32 + ".jsonl")
    path.parent.mkdir(parents=True)
    raw = b"bad\n" + json.dumps(example()).encode() + b"\n{partial"
    path.write_bytes(raw)
    result = collect_bundle(tmp_path)
    assert result["summary"]["event_count"] == 1
    assert result["redaction"]["invalid_events"] == 2
    assert path.read_bytes() == raw


def test_symlink_and_overwrite_guards(tmp_path):
    path = tmp_path / "events" / "tests" / ("42-" + "c" * 32 + ".jsonl")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(example()) + "\n")
    with pytest.raises(ValueError, match="overwrite"):
        collect_bundle(tmp_path, output=path)
    link = tmp_path / "link"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink"):
        collect_bundle(link)


def test_output_write_failure_leaves_existing_artifact(tmp_path, monkeypatch):
    output = tmp_path / "bundle.json"
    output.write_text("previous")
    def fail(*args):
        raise OSError("full")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        collect_bundle(tmp_path, records=[example()], output=output)
    assert output.read_text() == "previous"
    assert not list(tmp_path.glob(".diagnostics-*"))


def test_selected_records_do_not_scan_logs(tmp_path, monkeypatch):
    def prohibited(*args):
        raise AssertionError("must not scan unrelated tasks")
    monkeypatch.setattr(os, "scandir", prohibited)
    result = collect_bundle(tmp_path, records=[example()], include_logs=False)
    assert result["summary"]["event_count"] == 1


def test_process_clock_domains_and_structured_frames():
    frame = {"module": "mindie_diagnostics.bundle", "function": "collect_bundle", "line": 100}
    attributes = {"exception_chain": ["ValueError"], "stack_frames": [frame, {**frame, "module": "private_customer"}],
                  "stack_fingerprint": "a" * 64, "exception_message": "never publish free text"}
    first = export_public_event(example(process_instance_id="c" * 32, attributes=attributes))
    second = export_public_event(example(process_instance_id="d" * 32, attributes=attributes))
    assert first["process_ref"] != second["process_ref"]
    assert "clock_domain_unknown" not in first
    assert len(first["attributes"]["stack_frames"]) == 1
    assert first["attributes"]["stack_frames"][0]["package"] == "mindie_diagnostics"
    assert "exception_message" not in first["attributes"]
    assert export_public_event(first) == first
