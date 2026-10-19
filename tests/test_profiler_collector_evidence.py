"""Caller-labeled steps file and kernel-DB clock alignment tests."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from mindie_profiler.collector_evidence import (
    CollectorEvidenceError,
    align_capture_to_db,
    build_steps_document,
    find_capture_db,
    inspect_kernel_db_clock_range,
    write_steps_file,
)
from mindie_profiler.framework_capture import build_capture_plan, scan_capture_artifacts
from mindie_profiler.segment import discover_rank_sources, load_steps_file


def _capture() -> dict:
    plan = build_capture_plan(2, 5)
    now = time.time_ns()
    perf = time.perf_counter_ns()
    records = []
    for step in range(5):
        start = now + step * 1_000_000_000
        end = start + 30_000_000
        records.append({
            "step": step,
            "segment": "skipped" if step < 1 else ("warmup" if step == 1 else "active"),
            "wall_start_ns": start,
            "wall_end_ns": end,
            "wall_start_ts": start / 1e9,
            "wall_end_ts": end / 1e9,
            "perf_start_ns": perf + step * 1_000_000_000,
            "perf_end_ns": perf + step * 1_000_000_000 + 30_000_000,
            "duration_ms": 30.0,
        })
    active = [item for item in records if item["segment"] == "active"]
    active_ms = [item["duration_ms"] for item in active]
    timing = {
        "count": len(active_ms),
        "total_ms": sum(active_ms),
        "min_ms": min(active_ms),
        "max_ms": max(active_ms),
        "mean_ms": sum(active_ms) / len(active_ms),
    }
    return {
        "status": "ok",
        "device": 0,
        "plan": plan,
        "step_records": records,
        "steps": {"active": timing},
        "phases": {"total": 100.0},
        "capture_wall_start_ns": records[0]["wall_start_ns"] - 1_000_000,
        "capture_wall_end_ns": records[-1]["wall_end_ns"] + 1_000_000,
        "capture_wall_start_ts": (records[0]["wall_start_ns"] - 1_000_000) / 1e9,
        "capture_wall_end_ts": (records[-1]["wall_end_ns"] + 1_000_000) / 1e9,
        "active_wall_start_ns": active[0]["wall_start_ns"],
        "active_wall_end_ns": active[-1]["wall_end_ns"],
        "active_wall_start_ts": active[0]["wall_start_ts"],
        "active_wall_end_ts": active[-1]["wall_end_ts"],
        "artifacts": {"artifacts": []},
    }


def test_capture_shape_matches_ds_eng_08_capture_contract():
    from mindie_profiler.compare_acceptance import _capture_matches

    capture = _capture()
    profiled = {"timing": dict(capture["steps"]["active"]), "duration_ms": 100.0}
    ok, reasons = _capture_matches(
        {"capture": capture, "physical_card": 0},
        {"_step_ids": [2, 3, 4]},
        profiled,
    )
    assert ok is True, reasons
    assert reasons == []


def test_steps_file_uses_callers_active_ids_and_actual_wall_bounds(tmp_path: Path):
    capture = _capture()
    document = build_steps_document(capture)
    assert document["time_unit"] == "ns"
    assert document["clock_domain"] == "unix_wall"
    assert [item["step"] for item in document["steps"]] == [2, 3, 4]
    for item in document["steps"]:
        assert item["start"] == item["start_ns"]
        assert item["end"] == item["end_ns"]
        assert isinstance(item["start_ns"], int)
        assert isinstance(item["end_ns"], int)
        assert item["start_ts"] == item["start_ns"] / 1e9
        assert item["perf_start_ns"] < item["perf_end_ns"]

    path = tmp_path / "steps.json"
    write_steps_file(path, capture)
    windows = load_steps_file(path, unit="ns")
    assert [window.step for window in windows] == [2, 3, 4]
    assert windows[0].start_ns == document["steps"][0]["start_ns"]
    assert windows[-1].end_ns == document["steps"][-1]["end_ns"]
    assert json.loads(path.read_text())["source"] == "caller_declared_capture_active_records"


def test_steps_file_rejects_failed_capture_or_mismatched_ids():
    capture = _capture()
    capture["status"] = "missing_db_export"
    with pytest.raises(CollectorEvidenceError, match="capture status"):
        build_steps_document(capture)

    capture = _capture()
    capture["step_records"][2]["step"] = 99
    with pytest.raises(CollectorEvidenceError, match="do not equal"):
        build_steps_document(capture)


def _db(tmp_path: Path, *, minimum: int, maximum: int) -> Path:
    path = tmp_path / "kernel.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE TASK (startNs INTEGER, endNs INTEGER)")
    connection.execute("INSERT INTO TASK VALUES (?, ?)", (minimum, maximum))
    connection.commit()
    connection.close()
    return path


def test_real_a2_unranked_db_and_host_step_windows():
    root = Path(__file__).parent / "fixtures" / "profiler_capture_a2"
    capture = json.loads((root / "capture.json").read_text())
    active = [row for row in capture["step_records"] if row["segment"] == "active"]
    scan = scan_capture_artifacts(root)
    assert scan["by_kind"] == {"db": 1}
    ranks = discover_rank_sources(root, source="db")
    assert len(ranks) == 1
    clock = inspect_kernel_db_clock_range(ranks[0].db_path, active_steps=active)
    assert clock["clock_domain"] == "unix_epoch_ns"
    assert clock["status"] == "ok"
    alignment = align_capture_to_db(capture, clock)
    assert alignment["status"] == "aligned"
    # Host work begins before its first kernel. Requiring the kernel extent
    # to contain the caller's interval rejected this actual capture.
    assert alignment["db_contains_active"] is False
    assert alignment["capture_contains_db"] is True
    assert alignment["every_active_step_has_device_tasks"] is True
    assert alignment["offset_applied_ns"] == 0
    assert alignment["offset_guessed"] is False
    unrelated = [{**row, "wall_start_ns": row["wall_start_ns"] + 3_600_000_000_000,
                  "wall_end_ns": row["wall_end_ns"] + 3_600_000_000_000} for row in active]
    bad_clock = inspect_kernel_db_clock_range(ranks[0].db_path, active_steps=unrelated)
    assert align_capture_to_db(capture, bad_clock)["status"] == "partial_overlap"


def test_db_clock_unknown_domain_is_not_forced_to_match(tmp_path: Path):
    capture = _capture()
    path = _db(tmp_path, minimum=10, maximum=20)
    clock = inspect_kernel_db_clock_range(path)
    assert clock["clock_domain"] == "unknown_non_epoch_ns"
    assert clock["status"] == "unknown_clock_domain"
    alignment = align_capture_to_db(capture, clock)
    assert alignment["status"] == "unavailable"
    assert alignment["offset_applied_ns"] == 0
    assert alignment["offset_guessed"] is False


def test_find_capture_db_uses_nonempty_artifact(tmp_path: Path):
    capture = _capture()
    rank = tmp_path / "rank0" / "ASCEND_PROFILER_OUTPUT"
    rank.mkdir(parents=True)
    db = rank / "ascend_pytorch_profiler_0.db"
    db.write_bytes(b"db")
    capture["artifacts"] = {"artifacts": [{
        "kind": "db", "path": str(db), "non_empty": True, "size_bytes": 2, "modified_ns": 1,
    }]}
    assert find_capture_db(capture) == db
