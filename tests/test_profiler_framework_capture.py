"""Framework capture plan, artifact, and live-driver boundary tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_profiler.framework_capture import (
    EXPERIMENTAL_CONFIG,
    ProfileWindowError,
    build_capture_plan,
    evaluate_capture_success,
    export_budget_exceeded,
    run_framework_capture,
    scan_capture_artifacts,
)


def test_zero_based_window_maps_to_exact_schedule():
    plan = build_capture_plan(0, 3)
    assert plan["schedule"] == {
        "skip_first": 0, "wait": 0, "warmup": 0, "active": 3, "repeat": 1,
    }
    assert plan["capture"]["captured_step_ids"] == [0, 1, 2]
    assert plan["execution"]["model_step_ids"] == [0, 1, 2]
    assert plan["execution"]["terminal_flush_step_id"] == 3
    assert plan["execution"]["terminal_flush_is_model_step"] is False
    assert plan["inequality"]["required_prof_step_calls"] == 4
    assert plan["inequality"]["capture_start_step"] == 0
    assert plan["inequality"]["capture_end_step"] == 3
    assert plan["inequality"]["satisfied"] is True
    assert plan["experimental_config"] == EXPERIMENTAL_CONFIG
    assert plan["experimental_config"] == {
        "profiler_level": "Level1", "aic_metrics": "PipeUtilization", "export_type": "Db",
    }


def test_nonzero_window_uses_skip_first_and_warmup_without_crossing():
    plan = build_capture_plan(2, 5)
    assert plan["schedule"] == {
        "skip_first": 1, "wait": 0, "warmup": 1, "active": 3, "repeat": 1,
    }
    assert plan["capture"]["captured_step_ids"] == [2, 3, 4]
    assert plan["capture"]["step_identity_source"] == "caller_declared_range"
    # 1 skipped + 1 warmup + 3 active + 1 terminal flush = 6 = end_step + 1
    assert plan["inequality"]["required_prof_step_calls"] == 6
    assert plan["inequality"]["capture_within_declared"] is True
    assert plan["inequality"]["satisfied"] is True


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (2, 2), (3, 2)])
def test_invalid_windows_are_rejected_before_capture(start, end):
    with pytest.raises(ProfileWindowError):
        build_capture_plan(start, end)


def test_repeat_is_rejected_for_explicit_single_window():
    with pytest.raises(ProfileWindowError, match="repeat=1"):
        build_capture_plan(0, 3, repeat=2)


def test_artifact_scan_uses_real_files_and_excludes_empty_db(tmp_path: Path):
    rank = tmp_path / "rank0" / "ASCEND_PROFILER_OUTPUT"
    rank.mkdir(parents=True)
    (rank / "ascend_pytorch_profiler_0.db").write_bytes(b"not-a-real-db-but-nonempty")
    (rank / "ascend_pytorch_profiler_1.db").write_bytes(b"")
    (rank / "kernel_details.csv").write_text("kernel,time\n", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("ignored", encoding="utf-8")

    scan = scan_capture_artifacts(tmp_path)
    assert scan["ok"] is True
    assert scan["by_kind"]["db"] == 2
    assert scan["by_kind"]["kernel_details"] == 1
    non_empty_db = [item for item in scan["artifacts"] if item["kind"] == "db" and item["non_empty"]]
    assert len(non_empty_db) == 1
    assert non_empty_db[0]["size_bytes"] == len(b"not-a-real-db-but-nonempty")


def test_live_driver_plan_json_does_not_require_torch():
    driver = Path(__file__).with_name("live_profiler_collection_acceptance.py")
    completed = subprocess.run(
        [sys.executable, str(driver), "--mode", "plan", "--start-step", "0", "--end-step", "3"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["schema_version"] == "mindie-framework-capture-plan/1"
    assert plan["inequality"]["satisfied"] is True


def _load_live_driver():
    path = Path(__file__).with_name("live_profiler_collection_acceptance.py")
    spec = importlib.util.spec_from_file_location("live_profiler_collection_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_online_db_export_verification_requires_nonempty_rank_db(tmp_path: Path):
    driver = _load_live_driver()
    rank = tmp_path / "rank0" / "ASCEND_PROFILER_OUTPUT"
    rank.mkdir(parents=True)
    assert driver.verify_db_export(tmp_path)["status"] == "failed"
    (rank / "ascend_pytorch_profiler_0.db").write_bytes(b"db")
    ok = driver.verify_db_export(tmp_path, expected_ranks=1)
    assert ok["status"] == "ok"
    assert ok["rank_count"] == 1
    assert ok["artifacts"][0]["size_bytes"] == 2
    assert driver.verify_db_export(tmp_path, expected_ranks=2)["status"] == "failed"


def test_success_contract_requires_real_nonempty_db_and_contained_active(tmp_path: Path):
    rank = tmp_path / "rank0" / "ASCEND_PROFILER_OUTPUT"
    rank.mkdir(parents=True)
    db = rank / "ascend_pytorch_profiler_0.db"
    db.write_bytes(b"real-db-placeholder")
    artifacts = scan_capture_artifacts(tmp_path)
    decision = evaluate_capture_success(
        artifacts, stop_acknowledged=True, capture_contains_active=True,
        export_within_budget=True,
    )
    assert decision["status"] == "ok"
    assert decision["requirements"] == {
        "stop_acknowledged": True,
        "artifact_scan_ok": True,
        "non_empty_db_present": True,
        "capture_contains_active": True,
        "export_within_budget": True,
    }

    empty = tmp_path / "empty"
    empty.mkdir()
    assert evaluate_capture_success(
        scan_capture_artifacts(empty), stop_acknowledged=True,
        capture_contains_active=True, export_within_budget=True,
    )["status"] == "missing_db_export"
    assert evaluate_capture_success(
        artifacts, stop_acknowledged=False, capture_contains_active=True,
        export_within_budget=True,
    )["status"] == "failed"
    assert evaluate_capture_success(
        artifacts, stop_acknowledged=True, capture_contains_active=False,
        export_within_budget=True,
    )["status"] == "failed"
    assert evaluate_capture_success(
        artifacts, stop_acknowledged=True, capture_contains_active=True,
        export_within_budget=False,
    )["status"] == "export_over_budget"


def test_export_budget_uses_milliseconds_against_seconds():
    assert export_budget_exceeded(600.0, 1.0) is False
    assert export_budget_exceeded(1000.0, 1.0) is False
    assert export_budget_exceeded(1000.1, 1.0) is True
    assert export_budget_exceeded(1.5, 0.001) is True


def test_run_capture_without_torch_npu_is_explicit_failure(tmp_path: Path):
    if importlib.util.find_spec("torch_npu") is not None:
        pytest.skip("torch_npu is installed; this test only covers the absent-runtime boundary")
    result = run_framework_capture(
        lambda step: None, start_step=0, end_step=1, output_dir=tmp_path / "capture",
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "torch_npu_missing"
    assert result["step_records"] == []
    assert result["success_requirements"] is None


def test_live_workload_help_does_not_require_torch():
    script = Path(__file__).with_name("live_framework_capture_workload.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0
    assert "--start-step" in completed.stdout


def test_live_framework_mode_requires_explicit_run_binding(tmp_path: Path):
    driver = Path(__file__).with_name("live_profiler_collection_acceptance.py")
    completed = subprocess.run(
        [sys.executable, str(driver), "--mode", "framework", "--output-dir", str(tmp_path / "run")],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert "--run-id" in payload["error"]
    assert not (tmp_path / "run").exists()


def test_live_online_cycle_refuses_without_workload_command():
    driver = Path(__file__).with_name("live_profiler_collection_acceptance.py")
    completed = subprocess.run(
        [sys.executable, str(driver), "--mode", "online-cycle",
         "--endpoint", "http://127.0.0.1:9", "--execution-id", "exec", "--profile-dir", "/tmp/profile"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert "--workload-command" in payload["error"]
