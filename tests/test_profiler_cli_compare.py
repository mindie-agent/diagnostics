"""Exercise comparison through the real analyzer and public CLI."""

import json
from pathlib import Path

from mindie_profiler.__main__ import main
from mindie_profiler.analyze import analyze_profile
from mindie_profiler.events import Window


SOURCE = Path(__file__).parent / "fixtures" / "profiler_core" / "rank_0"
SCENE = {
    "framework": "vllm-ascend", "phase": "decode", "tp": 1, "dp": 1,
    "graph_mode": "eager", "clock_domain": "aligned_syscnt",
    "input_signature": "test-workload-same-inputs",
    "runtime": {"soc": "Ascend910B4", "framework": {"torch_npu": "2.10.0.post4"}},
    "device": {"card": 0, "host": "a3-165"},
    "model": "test-mlp", "shape": [8, 1024], "dtype": "float32",
}


def _structure():
    return {
        "event_sequence_shape": ["Linear", "ReLU", "Linear"],
        "operator_sequence": ["0:Linear", "1:ReLU", "2:Linear"],
        "modules": [
            {"module_id": "0", "module_name": "0", "module_type": "Linear", "ordinal": 0,
             "invocation_count": 2, "wall_ms_p50": 0.10, "busy_union_ms_p50": 0.08,
             "host_gap_ms_p50": 0.02, "mapping_coverage": 1.0},
            {"module_id": "1", "module_name": "1", "module_type": "ReLU", "ordinal": 1,
             "invocation_count": 2, "wall_ms_p50": 0.05, "busy_union_ms_p50": 0.04,
             "host_gap_ms_p50": 0.01, "mapping_coverage": 1.0},
        ],
        "layers": [],
    }


def test_compare_roots_uses_each_runs_explicit_steps(tmp_path, capsys):
    baseline_steps = tmp_path / "baseline-steps.json"
    candidate_steps = tmp_path / "candidate-steps.json"
    scene_file = tmp_path / "scene.json"
    baseline_steps.write_text(json.dumps({"steps": [
        {"step": 4, "start_us": 0, "end_us": 60},
        {"step": 5, "start_us": 100, "end_us": 160},
    ]}))
    candidate_steps.write_text(json.dumps({"steps": [
        {"step": 5, "start_us": 100, "end_us": 160},
    ]}))
    scene_file.write_text(json.dumps(SCENE))
    output = tmp_path / "comparison.json"
    result = main([
        "compare", "--baseline-root", str(SOURCE), "--candidate-root", str(SOURCE),
        "--baseline-steps-file", str(baseline_steps),
        "--candidate-steps-file", str(candidate_steps),
        "--baseline-scene-file", str(scene_file), "--candidate-scene-file", str(scene_file),
        "--rank", "rank_0", "--output", str(output),
    ])
    capsys.readouterr()
    assert result == 0
    comparison = json.loads(output.read_text())
    assert comparison["applicability"]["compared_steps"] == [5]
    assert comparison["applicability"]["step_basis"] == "recomputed_from_window_accounting"
    assert comparison["step_accounting"]["metrics"]["wall_ms_p50"]["delta"] == 0


def test_compare_summaries_cannot_promote_unknown_scene(tmp_path, capsys):
    report = analyze_profile(SOURCE, windows=[Window(0, 60_000, step=4)])
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(report))
    result = main([
        "compare", "--baseline-summary", str(summary), "--candidate-summary", str(summary),
    ])
    comparison = json.loads(capsys.readouterr().out)
    assert result == 3
    assert not comparison["comparable"]


def test_compare_roots_accepts_explicit_structure_files(tmp_path, capsys):
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps({"steps": [{"step": 4, "start_us": 0, "end_us": 60}]}))
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(SCENE))
    structure = tmp_path / "module-trace.json"
    structure.write_text(json.dumps({"structure": _structure()}))
    output = tmp_path / "comparison.json"
    result = main([
        "compare", "--baseline-root", str(SOURCE), "--candidate-root", str(SOURCE),
        "--baseline-steps-file", str(steps), "--candidate-steps-file", str(steps),
        "--baseline-scene-file", str(scene), "--candidate-scene-file", str(scene),
        "--baseline-structure-file", str(structure),
        "--candidate-structure-file", str(structure),
        "--rank", "rank_0", "--output", str(output),
    ])
    capsys.readouterr()
    assert result == 0
    comparison = json.loads(output.read_text())
    assert comparison["matching"]["run_binding"]["status"] == "match"
    assert comparison["tables"]["modules"]["status"] == "comparable"
    assert comparison["module_scope"] == "tiny_mlp_leaf_modules"


def test_compare_roots_rejects_mismatched_actual_binding(tmp_path, capsys):
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps({"steps": [{"step": 4, "start_us": 0, "end_us": 60}]}))
    baseline_scene = tmp_path / "baseline-scene.json"
    candidate_scene = tmp_path / "candidate-scene.json"
    baseline_scene.write_text(json.dumps(SCENE))
    mismatch = dict(SCENE)
    mismatch["device"] = {"card": 1, "host": "a3-165"}
    candidate_scene.write_text(json.dumps(mismatch))
    output = tmp_path / "comparison.json"
    result = main([
        "compare", "--baseline-root", str(SOURCE), "--candidate-root", str(SOURCE),
        "--baseline-steps-file", str(steps), "--candidate-steps-file", str(steps),
        "--baseline-scene-file", str(baseline_scene),
        "--candidate-scene-file", str(candidate_scene),
        "--rank", "rank_0", "--output", str(output),
    ])
    capsys.readouterr()
    assert result == 3
    comparison = json.loads(output.read_text())
    assert comparison["status"] == "not_comparable"
    assert any(item["code"] == "run_binding_field_mismatch" for item in comparison["reasons"])
