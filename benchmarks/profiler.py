"""[Role] Benchmark bounded C7 analysis, comparison, memory and roofline logic.
[Invariants] Synthetic inputs assert known semantics; no hardware or product gain claim.
[Timeouts & Locks] Fixed finite iterations; SQLite opens are read-only during measurement.
[Side-Effect Policy] One temporary schema fixture and the caller's result JSON.
"""
from __future__ import annotations

# ruff: noqa: E402 -- the source checkout is the benchmark import root.

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mindie_profiler import analyze_profile
from mindie_profiler.compare import compare_profiles
from mindie_profiler.memory_profile import attribute_memory
from mindie_profiler.modeling_inputs import (
    HardwareSpec,
    KernelInputs,
    estimate_ideal_hardware_lower_bound,
)
from tests.test_profiler_core import STEP4, STEP5, build_run
from tests.test_profiler_memory import _manifest_full

SCENE = {
    "framework": "synthetic", "phase": "decode", "tp": 1, "dp": 1,
    "graph_mode": "eager", "clock_domain": "synthetic",
    "runtime": {"fixture": "release"}, "device": {"fixture": "two-rank"},
    "model": "synthetic", "shape": [4, 8], "dtype": "f32",
    "input_signature": "fixed-synthetic-events",
}


def measure(name, iterations, operation, check):
    check(operation())
    started = time.perf_counter()
    for _ in range(iterations):
        result = operation()
    elapsed = time.perf_counter() - started
    check(result)
    return {
        "name": name, "unit": "us/op",
        "value": elapsed * 1_000_000 / iterations, "iterations": iterations,
    }


def check_report(report):
    assert report["step_accounting"]["steady_steps"] == [4, 5]
    assert len(report["window_accounting"]) == 4
    first = report["window_accounting"][0]
    assert first["busy_union_ms"] == 0.04
    assert first["kernel_sum_ms"] == 0.061
    assert first["host_gap_ms"] == 0.005
    assert report["step_accounting"]["slowest_rank"]["rank_index"] == 1


def check_comparison(result):
    assert result["comparable"]
    assert result["matching"]["steps"]["status"] == "same_steps"
    assert result["step_accounting"]["metrics"]["busy_union_ms_p50"]["delta"] == 0
    assert result["operator_sequence"]["status"] == "not_available"


def check_memory(result):
    device = result["devices"][0]
    components = {row["key"]: row for row in device["components"]}
    assert device["known_sum_bytes"] == 4_100_000_000
    assert device["residual"]["bytes"] == 150_000_000
    assert not components["runtime"]["included_in_sum"]
    assert not components["activation"]["included_in_sum"]


def check_ideal(result):
    assert result.status == "modeled"
    assert math.isclose(result.compute_ms, 128 / 1e12 * 1000)
    assert math.isclose(result.memory_ms, 112 / 1e9 * 1000)
    assert result.lower_bound_ms == result.memory_ms


def main():
    with tempfile.TemporaryDirectory(prefix="mindie-profiler-benchmark-") as directory:
        root = build_run(Path(directory))
        files = sorted(root.rglob("*.db"))
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

        def analyze():
            return analyze_profile(root, windows=[STEP4, STEP5], scene=SCENE)

        metrics = [measure("c7_two_rank_two_step_analysis", 5, analyze, check_report)]
        baseline, candidate = analyze(), analyze()
        metrics.append(measure(
            "c7_two_rank_comparison", 50,
            lambda: compare_profiles(baseline, candidate), check_comparison,
        ))
        # This synthetic measured-peak claim is solely a fixed arithmetic input.
        manifest = copy.deepcopy(_manifest_full())
        metrics.append(measure(
            "c7_memory_non_additive_attribution", 200,
            lambda: attribute_memory(manifest), check_memory,
        ))
        inputs = KernelInputs(
            shape=(2, 4, 8), dtype="bf16", layout="row_major",
            hardware=HardwareSpec("synthetic-roofline", 1e12, 1e9), op_kind="matmul",
        )
        metrics.append(measure(
            "c7_local_ideal_roofline", 1000,
            lambda: estimate_ideal_hardware_lower_bound(inputs), check_ideal,
        ))
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        assert before == after
    Path(os.environ["MINDIE_BENCH_RESULT_PATH"]).write_text(
        json.dumps({"status": "pass", "metrics": metrics}) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
